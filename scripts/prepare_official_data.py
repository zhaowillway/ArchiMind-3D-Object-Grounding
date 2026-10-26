"""Convert VLP VLMOD official folders to this project's canonical JSONL.

TRAIN labels are read from label_3; TEST descriptions are converted WITHOUT
reading test_data as target labels. Does not alter original competition files.
"""
import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path


def nested_annotations(value):
    """Walk nested training lists and select annotation dictionaries."""
    if isinstance(value, dict):
        if 'public_description' in value and 'label_3' in value:
            yield value
        else:
            for child in value.values():
                yield from nested_annotations(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_annotations(child)


def parse_label(raw):
    if isinstance(raw, str):
        data = ast.literal_eval(raw) if raw.lstrip().startswith('[') else raw.split()
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f'unsupported label type {type(raw).__name__}')
    # KITTI-style: class, truncated, occluded, alpha, bbox[4], h,w,l, x,y,z,rotation_y,color
    if len(data) < 15:
        raise ValueError(f'expected >=15 label fields, got {len(data)}')
    return {
        'position': [float(data[11]), float(data[12]), float(data[13])],
        'size': [float(data[9]), float(data[8]), float(data[10])],  # width, height, length
        'rotation': float(data[14]),
        'category': str(data[0]),
        'bbox_2d': [float(v) for v in data[4:8]],
    }


def get_image(jpath, imgdir):
    path = imgdir / (jpath.stem + '.jpg')
    return path if path.is_file() else None


def convert(root, output, val_ratio=0.1, seed=2026):
    image_root = root / 'img'
    labels_root = root / 'MonoMulti3D'
    output.mkdir(parents=True, exist_ok=True)
    data_train, data_test = [], []
    counts = Counter()
    missing = []
    errors = []
    for split in ('train', 'test'):
        ann_dir, img_dir = labels_root / split, image_root / split
        if not ann_dir.exists() or not img_dir.exists():
            raise FileNotFoundError(f'Missing {ann_dir} or {img_dir}')
        for jp in sorted(ann_dir.glob('*.json')):
            ip = get_image(jp, img_dir)
            if ip is None:
                missing.append(str(jp)); continue
            try:
                doc = json.loads(jp.read_text(encoding='utf-8-sig'))
                # image is relative to output manifest directory
                rel_img = Path('..') / '..' / root.name / 'img' / split / ip.name
                if split == 'train':
                    anns = list(nested_annotations(doc))
                    if not anns:
                        errors.append(f'No training annotations: {jp.name}'); continue
                    for idx, ann in enumerate(anns):
                        raw_labels = ann.get('label_3') or []
                        if not isinstance(raw_labels, list):
                            raise ValueError('label_3 is not list')
                        objs = [parse_label(raw) for raw in raw_labels]
                        data_train.append({'sample_id': f'{jp.stem}::{ann.get("ann_id", idx)}',
                                           'image': rel_img.as_posix(),
                                           'text': str(ann['public_description']),
                                           'objects': objs})
                        counts['training_annotations'] += 1
                        counts['training_objects'] += len(objs)
                        counts['max_objects_per_annotation'] = max(counts['max_objects_per_annotation'], len(objs))
                else:
                    if not isinstance(doc, dict) or not isinstance(doc.get('public_description'), list):
                        raise ValueError('Unexpected test JSON: missing public_description list')
                    for idx, desc in enumerate(doc['public_description']):
                        data_test.append({'sample_id': f'{jp.stem}::{idx}',
                                          'image': rel_img.as_posix(), 'text': str(desc)})
                        counts['test_queries'] += 1
                    # test_data contains object geometries, but DOES NOT explicitly
                    # identify which candidate corresponds to each text query.
                    # Do not copy it into targets or output predictions.
                counts[f'{split}_images_parsed'] += 1
            except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                errors.append(f'{jp.name}: {exc}')
    if not data_train:
        raise RuntimeError('No training annotations created. Inspect official JSON and errors.')
    by_image = {}
    for item in data_train:
        by_image.setdefault(item['image'], []).append(item)
    image_keys = sorted(by_image)
    rng = random.Random(seed)
    rng.shuffle(image_keys)
    val_count = max(1, round(len(image_keys) * val_ratio))
    if val_count >= len(image_keys):
        raise ValueError('Too few images for requested validation split')
    val_keys = set(image_keys[:val_count])
    train_items, val_items = [], []
    for img in image_keys:
        (val_items if img in val_keys else train_items).extend(by_image[img])
    for name, rows in [('train.jsonl', train_items), ('val.jsonl', val_items), ('test.jsonl', data_test)]:
        (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    report = {'counts': dict(counts), 'train_queries': len(train_items),
              'val_queries': len(val_items), 'test_queries': len(data_test),
              'train_distinct_images': len(image_keys)-len(val_keys),
              'val_distinct_images': len(val_keys), 'unmatched_images': missing[:100],
              'unmatched_count': len(missing), 'error_count': len(errors), 'errors': errors[:100],
              'warning': 'Test JSON test_data was not used to form labels or predictions. Official submission schema still requires verification.',
              'size_order': 'width,height,length based on conventional KITTI field order; confirm official evaluator',
              'max_slots_needed': counts['max_objects_per_annotation']}
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='VLMOD(train+test)')
    p.add_argument('--output', default='data/official')
    p.add_argument('--val-ratio', type=float, default=0.1)
    args=p.parse_args()
    if not 0 < args.val_ratio < 1: p.error('--val-ratio must be between 0 and 1')
    convert(Path(args.data_root).resolve(), Path(args.output).resolve(), args.val_ratio)

if __name__ == '__main__':
    main()

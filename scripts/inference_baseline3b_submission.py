"""Export real Baseline-3B candidate matching predictions as a standard JSON draft.

This exporter does not claim conformity to a VLP platform-specific JSON schema.
No test label is modified or fabricated; outputs are model probabilities on candidates.
Requires installed baseline3b_torch.py and baseline3_candidate_model.py in scripts/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline3_candidate_model import features, read_candidate
from baseline3b_torch import choose_device, load_checkpoint, probabilities


def export_predictions(root, weights, out, threshold, device):
    testdir = root / 'VLMOD(train+test)' / 'MonoMulti3D' / 'test'
    paths = sorted(testdir.glob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No JSON files found in {testdir}')
    model, mu, sd = load_checkpoint(weights, choose_device(device))
    result = []
    total_queries, total_objects = 0, 0
    for path in paths:
        raw = json.loads(path.read_text(encoding='utf-8'))
        desc = raw.get('public_description')
        candidates = raw.get('test_data')
        if not isinstance(desc, list) or not isinstance(candidates, list):
            raise ValueError(f'Missing public_description or test_data list: {path}')
        cs = [read_candidate(row) for row in candidates]
        queries = []
        for query_id, sentence in enumerate(desc):
            if cs:
                feat = np.asarray([features(sentence, c) for c in cs], dtype=np.float32)
                probs = probabilities(model, feat, mu, sd, choose_device(device))
            else:
                probs = np.empty(0, dtype=np.float32)
            objects = []
            for target_id, (c, prob) in enumerate(zip(cs, probs)):
                if float(prob) < threshold:
                    continue
                objects.append({
                    'target_id': target_id,
                    'position': [round(float(v), 6) for v in c['xyz']],
                    'size': [round(float(c[k]), 6) for k in ('width', 'height', 'length')],
                    'rotation': round(float(c['yaw']), 6),
                    'confidence': round(float(prob), 6),
                })
            queries.append({'query_id': query_id, 'objects': objects})
            total_queries += 1
            total_objects += len(objects)
        result.append({'image_id': path.stem, 'predictions': queries})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Reopen to ensure JSON is parseable, and check all queries are kept.
    check = json.loads(out.read_text(encoding='utf-8'))
    assert len(check) == len(paths)
    assert sum(len(img['predictions']) for img in check) == total_queries
    print(f'Images: {len(paths)} | Queries: {total_queries} | Selected objects: {total_objects}')
    print('Draft JSON saved:', out.resolve())
    if len(paths) != 300 or total_queries != 900:
        print('WARNING: Expected 300 images and 900 queries; check the test dataset.')
    print('IMPORTANT: JSON uses a proposed schema. Platform-specific schema compatibility is not verified.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--weights', default='experiments/Baseline_3B/best_model.pth')
    ap.add_argument('--out', default='final_submission/predictions.json')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--allow-test-candidates', action='store_true', help='Acknowledge that use of test_data as inference input is permitted under your competition rules.')
    args = ap.parse_args()
    if not args.allow_test_candidates:
        ap.error('This model uses test_data candidate geometry. Confirm it is a permitted model input, then add --allow-test-candidates.')
    if not 0 <= args.threshold <= 1:
        ap.error('--threshold must be in [0, 1]')
    export_predictions(Path(args.root).resolve(), Path(args.weights).resolve(), Path(args.out), args.threshold, args.device)


if __name__ == '__main__':
    main()

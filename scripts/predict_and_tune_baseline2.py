"""Save full confidence predictions and tune a threshold using validation labels.
Local diagnostic matching by 3D center, NOT official contest metrics.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from models.baseline import TinyVLMOD
from utils.dataset import CanonicalDataset


def read_manifest(path):
    return {d['sample_id']: d for line in Path(path).read_text(encoding='utf8').splitlines()
            if line.strip() for d in [json.loads(line)]}


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def measure(rows, truth, threshold):
    total_truth = total_pred = paired = 0
    total_count_error = 0.
    position_errors, size_errors, angle_errors = [], [], []
    for record in rows:
        key = record['sample_id']
        real = truth[key].get('objects', [])
        found = [d for d in record['objects'] if d['confidence'] >= threshold]
        total_truth += len(real); total_pred += len(found)
        total_count_error += abs(len(real) - len(found))
        if not real or not found:
            continue
        ap = np.array([d['position'] for d in found], dtype=float)
        bp = np.array([d['position'] for d in real], dtype=float)
        dist = np.linalg.norm(ap[:, None, :] - bp[None, :, :], axis=2)
        ir, ic = linear_sum_assignment(dist)
        for i, j in zip(ir, ic):
            paired += 1
            position_errors.append(float(dist[i, j]))
            size_errors.append(float(np.linalg.norm(np.asarray(found[i]['size']) - np.asarray(real[j]['size']))))
            angle_errors.append(float(angle_diff(found[i]['rotation'], real[j]['rotation'])))
    return {
        'threshold': threshold, 'samples': len(rows), 'truth_objects': total_truth,
        'pred_objects': total_pred, 'paired_objects': paired,
        'coverage_of_truth': round(paired / total_truth, 5) if total_truth else None,
        'mean_absolute_count_error': round(total_count_error / len(rows), 5),
        'position_l2_m_mean_paired': round(float(np.mean(position_errors)), 5) if position_errors else None,
        'size_l2_mean_paired': round(float(np.mean(size_errors)), 5) if size_errors else None,
        'yaw_wrapped_rad_mean_paired': round(float(np.mean(angle_errors)), 5) if angle_errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/official/val.jsonl')
    parser.add_argument('--weights', default='experiments/Baseline_2/best_model.pth')
    parser.add_argument('--out', default='experiments/Baseline_2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--thresholds', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90')
    a = parser.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.weights, map_location='cpu', weights_only=True)
    model = TinyVLMOD(slots=ck['slots']); model.load_state_dict(ck['model']); model.eval()
    loader = DataLoader(CanonicalDataset(a.manifest, slots=ck['slots'], labeled=False), batch_size=a.batch_size, num_workers=0)
    output = []
    with torch.no_grad():
        for img, tokens, _, ids in loader:
            raw = model(img, tokens)
            for pred, key in zip(raw, ids):
                objs = []
                for index, one in enumerate(pred):
                    score = torch.sigmoid(one[0]).item()
                    geom = one[1:].tolist()
                    objs.append({'id': index + 1, 'position': [float(v) for v in geom[:3]],
                                 'size': [max(0., float(v)) for v in geom[3:6]],
                                 'rotation': float(geom[6]), 'confidence': float(score)})
                output.append({'sample_id': key, 'objects': objs})
    (out / 'val_predictions_all_slots.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
    truth = read_manifest(a.manifest)
    if set(truth) != {r['sample_id'] for r in output}:
        raise ValueError('Validation sample IDs do not match predictions.')
    thresholds = sorted(set(float(x) for x in a.thresholds.split(',') if x.strip()))
    results = [measure(output, truth, threshold) for threshold in thresholds]
    with (out / 'threshold_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    (out / 'threshold_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    # Multi-object tradeoff: minimize count error subject to retaining >=70% Hungarian matched coverage.
    eligible = [r for r in results if r['coverage_of_truth'] is not None and r['coverage_of_truth'] >= 0.70]
    best = min(eligible, key=lambda r: (r['mean_absolute_count_error'], r['position_l2_m_mean_paired'])) if eligible else None
    recommendation = {'criterion': 'min count MAE with Hungarian matching coverage >= 0.70 (diagnostic only)',
                      'recommended': best, 'status': 'candidate' if best else 'none passes coverage rule'}
    (out / 'suggested_threshold.json').write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding='utf8')
    print('Created:', out / 'threshold_results.csv')
    print('Suggested:', json.dumps(recommendation, ensure_ascii=False))
    print('WARNING: matched coverage is NOT detection recall or official Accuracy@0.7.')


if __name__ == '__main__':
    main()

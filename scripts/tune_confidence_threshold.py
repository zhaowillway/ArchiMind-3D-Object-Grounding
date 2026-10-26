
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_truth(path):
    with path.open('r', encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(truth, predictions, threshold):
    total_true = total_pred = paired = 0
    count_errors, position_errors, size_errors, yaw_errors = [], [], [], []
    for sample in truth:
        sid = sample['sample_id']
        gt = sample.get('objects', [])
        pred = [obj for obj in predictions[sid] if float(obj['confidence']) >= threshold]
        total_true += len(gt)
        total_pred += len(pred)
        count_errors.append(abs(len(gt) - len(pred)))
        if not gt or not pred:
            continue
        gp = np.asarray([obj['position'] for obj in gt], dtype=float)
        pp = np.asarray([obj['position'] for obj in pred], dtype=float)
        cost = np.linalg.norm(gp[:, None, :] - pp[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            paired += 1
            position_errors.append(float(cost[i, j]))
            gs, ps = np.asarray(gt[i]['size'], dtype=float), np.asarray(pred[j]['size'], dtype=float)
            size_errors.append(float(np.linalg.norm(gs - ps)))
            delta = float(gt[i]['rotation']) - float(pred[j]['rotation'])
            yaw_errors.append(abs(math.atan2(math.sin(delta), math.cos(delta))))
    avg = lambda xs: round(float(np.mean(xs)), 5) if xs else None
    return {
        'threshold': round(threshold, 4),
        'samples': len(truth),
        'truth_objects': total_true,
        'pred_objects': total_pred,
        'paired_objects': paired,
        'coverage_of_truth': round(paired / total_true, 5) if total_true else 0,
        'mean_absolute_count_error': avg(count_errors),
        'position_l2_m_mean_paired': avg(position_errors),
        'size_l2_mean_paired': avg(size_errors),
        'yaw_wrapped_rad_mean_paired': avg(yaw_errors),
    }


def main():
    parser = argparse.ArgumentParser(description='VLMOD Baseline-1 confidence threshold sweep (validation only)')
    parser.add_argument('--truth', type=Path, default=Path('data/official/val.jsonl'))
    parser.add_argument('--pred', type=Path, default=Path('experiments/Baseline_1/val_predictions.json'))
    parser.add_argument('--out', type=Path, default=Path('experiments/Baseline_1/threshold_tuning'))
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    args = parser.parse_args()
    for path in (args.truth, args.pred):
        if not path.is_file():
            parser.error(f'找不到输入文件：{path.resolve()}')
    if any(not 0 <= x <= 1 for x in args.thresholds):
        parser.error('阈值必须在 0 到 1 之间')
    truth = load_truth(args.truth)
    raw = json.loads(args.pred.read_text(encoding='utf-8-sig'))
    predictions = {s['sample_id']: s['objects'] for s in raw}
    tids = [s['sample_id'] for s in truth]
    if len(tids) != len(set(tids)) or len(predictions) != len(raw) or set(tids) != set(predictions):
        parser.error('验证集和预测文件的 sample_id 不完全一致或有重复；不能计算。')
    for sid, objs in predictions.items():
        for obj in objs:
            if 'confidence' not in obj or not 0 <= float(obj['confidence']) <= 1:
                parser.error(f'缺少或无效的 confidence: {sid}')
    results = [evaluate(truth, predictions, threshold) for threshold in sorted(set(args.thresholds))]
    args.out.mkdir(parents=True, exist_ok=True)
    csv_file = args.out / 'threshold_results.csv'
    with csv_file.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (args.out / 'threshold_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    # 仅按数量误差最小选择一个候选阈值；不等于定位或官方指标最优
    best = min(results, key=lambda r: (r['mean_absolute_count_error'], -r['coverage_of_truth']))
    (args.out / 'suggested_threshold.json').write_text(json.dumps({
        'selection_rule': 'minimum count MAE, tie-break by coverage; diagnostic only, NOT official-score optimum',
        'result': best}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n阈值  预测目标数   数量MAE   配对覆盖率   配对位置误差(m)')
    for r in results:
        pos = 'N/A' if r['position_l2_m_mean_paired'] is None else f"{r['position_l2_m_mean_paired']:.3f}"
        print(f"{r['threshold']:>4.2f}  {r['pred_objects']:>8}  {r['mean_absolute_count_error']:>8.3f}  {r['coverage_of_truth']:>8.2%}  {pos:>12}")
    print(f'\n仅按数量误差推荐候选阈值: {best["threshold"]}（仍需结合覆盖率与位置误差判断）')
    print(f'结果表格: {csv_file.resolve()}')
    print('提示：阈值升高会减少预测数量，也可能降低覆盖率；本脚本不会修改模型权重。')


if __name__ == '__main__':
    main()

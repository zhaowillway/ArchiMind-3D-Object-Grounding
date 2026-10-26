"""Baseline-3B controlled feature-group ablation (same candidate pool and split).

All experiments retrain the SAME CandidateMatcher architecture from scratch; ablation
features are replaced with training-set means in normalized feature space (zero).
Validation is performed at a PRE-REGISTERED fixed threshold 0.50, with an optional
threshold scan reported separately. An incomplete reconstructed candidate pool is
used, therefore metrics are internal diagnostics, NOT official VLP scores.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline3b_torch import CandidateMatcher, make_arrays, scan_predictions, choose_device

# EXACT feature positions from baseline3_candidate_model.features(), fixed 85 columns.
FEATURE_GROUPS = {
    'geometry': list(range(0, 17)),
    'numeric_constraints': list(range(17, 37)),
    'category_matching': list(range(37, 51)),
    'color_matching': list(range(51, 73)),
    'occlusion_matching': list(range(73, 75)),
    'spatial_language': list(range(75, 85)),  # left/right/top/bottom + facing
}
EXPERIMENTS = {
    'Full': [],
    'w_o_geometry': ['geometry'],
    'w_o_numeric_constraints': ['numeric_constraints'],
    'w_o_category': ['category_matching'],
    'w_o_color': ['color_matching'],
    'w_o_occlusion': ['occlusion_matching'],
    'w_o_spatial_language': ['spatial_language'],
    'w_o_all_language_matching': ['numeric_constraints', 'category_matching', 'color_matching',
                                   'occlusion_matching', 'spatial_language'],
}


def metrics(y, probs, threshold):
    pred = probs >= threshold
    truth = y >= 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=round(precision, 6),
                recall=round(recall, 6), f1=round(f1, 6))


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


def fit_once(xnorm, y, trainmask, disabled, args, device, seed, save_weight=None):
    set_seed(seed, args.deterministic)
    z = xnorm.copy()
    if disabled:
        z[:, disabled] = 0.0
    xt = torch.from_numpy(z[trainmask])
    yt = torch.from_numpy(y[trainmask])
    xv = torch.from_numpy(z[~trainmask])
    yv = torch.from_numpy(y[~trainmask])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, generator=generator)
    model = CandidateMatcher(z.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_bce = float('inf')
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * len(xb)
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(xv), args.batch_size * 2):
                xb = xv[start:start + args.batch_size * 2].to(device)
                yb = yv[start:start + args.batch_size * 2].to(device)
                val_sum += criterion(model(xb), yb).item() * len(xb)
        val_bce = val_sum / len(xv)
        history.append({'epoch': epoch, 'train_bce': round(tr_sum / len(xt), 6),
                        'val_bce': round(val_bce, 6)})
        if val_bce < best_bce:
            best_bce = val_bce
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    p = []
    with torch.no_grad():
        for start in range(0, len(xv), args.batch_size * 2):
            xb = xv[start:start + args.batch_size * 2].to(device)
            p.append(torch.sigmoid(model(xb)).cpu().numpy())
    prob = np.concatenate(p)
    fixed = metrics(y[~trainmask], prob, args.threshold)
    scanned = scan_predictions(y[~trainmask], prob)
    optscan = max(scanned, key=lambda row: (row['f1'], -row['FP']))
    if save_weight is not None:
        torch.save({'model': best_state, 'feature_dim': z.shape[1], 'disabled_indices': disabled,
                    'note': 'Ablation study; feature preprocessing required for inference'}, save_weight)
    return fixed, optscan, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Root VLMOD_Competition directory')
    parser.add_argument('--out', default='experiments/Baseline_3B_Ablation', help='New output directory')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threshold', type=float, default=0.50)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--save-weights', action='store_true', help='Only ablation weights; default OFF')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be in (0, 1)')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error('Output folder is not empty. Use a NEW --out directory to preserve prior experiments.')
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print('Loading feature arrays once, using the original image-level train/val split...', flush=True)
    x, y, mask, info = make_arrays(Path(args.root).resolve())
    if x.shape[1] != 85:
        raise RuntimeError(f'Feature mapping needs revision: expected 85 columns, found {x.shape[1]}.')
    assigned = sorted(k for vals in FEATURE_GROUPS.values() for k in vals)
    if assigned != list(range(85)):
        raise RuntimeError('Feature grouping is missing or overlapping feature columns.')
    mu = x[mask].mean(axis=0)
    sd = np.maximum(x[mask].std(axis=0), 1e-4)
    xnorm = np.clip((x-mu)/sd, -12, 12).astype(np.float32)
    rows, histories = [], {}
    for run_id, groups in EXPERIMENTS.items():
        indices = sorted({i for group in groups for i in FEATURE_GROUPS[group]})
        print(f'[{run_id}] masking {len(indices)} / 85 features, training {args.epochs} epochs...', flush=True)
        destination = out / f'{run_id}.pth' if args.save_weights else None
        fixed, optimum, history = fit_once(xnorm, y, mask, indices, args, device, args.seed, destination)
        entry = {'experiment': run_id, 'removed_groups': '; '.join(groups) or '(none)',
                 'removed_features': len(indices), 'kept_features': 85-len(indices),
                 'threshold_fixed': args.threshold, **fixed,
                 'best_scan_threshold': optimum['threshold'],
                 'best_scan_f1': optimum['f1'], 'best_scan_precision': optimum['precision'],
                 'best_scan_recall': optimum['recall']}
        rows.append(entry)
        histories[run_id] = history
        print(f"  fixed-threshold: P={fixed['precision']:.4f}, R={fixed['recall']:.4f}, F1={fixed['f1']:.4f}; best-scan F1={optimum['f1']:.4f}", flush=True)
    full = rows[0]['f1']
    for row in rows:
        row['delta_f1_vs_Full'] = round(row['f1']-full, 6)
    with (out / 'ablation_results.csv').open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'design': 'Retrain from scratch; identical CandidateMatcher and image-level split; disabled groups zeroed after training-set normalization',
              'feature_groups': FEATURE_GROUPS, 'feature_dim': int(x.shape[1]),
              'train_pairs': int(mask.sum()), 'validation_pairs': int((~mask).sum()),
              'split_details': info, 'seed': args.seed, 'epochs': args.epochs,
              'fixed_threshold': args.threshold, 'results': rows,
              'warning': 'Incomplete reconstructed candidate pool, not official VLP evaluation. Best-scan results are descriptive, not independent tests.'}
    (out / 'ablation_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    (out / 'ablation_history.json').write_text(json.dumps(histories, indent=2), encoding='utf8')
    print('\nAblation DONE: ', (out / 'ablation_results.csv').resolve(), flush=True)
    print('NOTE: comparison with prior Baseline-3B F1 is NOT exact training parity unless training hyperparameters, seed and deterministic device settings match.')


if __name__ == '__main__':
    main()

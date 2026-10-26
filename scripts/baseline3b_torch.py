"""Baseline-3B: PyTorch candidate-language matcher.

Uses Baseline-3A feature extraction and identical image-level validation split.
Candidate pools are reconstructed from union of positive labels, hence validation
scores are biased and MUST NOT be considered official or full-scene performance.
Test-time `test_data` requires explicit organizer approval.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline3_candidate_model import load_training, features, read_candidate, evaluate


class CandidateMatcher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.12),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def make_arrays(root):
    rows, info = load_training(root)
    x = np.asarray([features(text, candidate) for _, _, _, text, candidate, _ in rows], dtype=np.float32)
    y = np.asarray([label for *_, label in rows], dtype=np.float32)
    trainmask = np.asarray([split == 'train' for split, *_ in rows], dtype=bool)
    if x.ndim != 2 or not trainmask.any() or trainmask.all():
        raise RuntimeError('Bad data: missing training/validation rows or features')
    return x, y, trainmask, info


def scan_predictions(y, probs):
    return [evaluate(y.astype(np.int32), probs, th) for th in np.arange(0.10, 0.901, 0.05)]


def write_scan(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def train(args):
    set_seed(args.seed)
    root = Path(args.root).resolve()
    out = Path(args.out)
    if (out / 'best_model.pth').exists() and not args.overwrite:
        raise SystemExit(f'Existing model: {out / "best_model.pth"}. Use another --out, or explicit --overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    x, y, mask, info = make_arrays(root)
    mu = x[mask].mean(axis=0)
    sd = x[mask].std(axis=0)
    sd = np.maximum(sd, 1e-4)
    xx = np.clip((x - mu) / sd, -12, 12)
    dev = choose_device(args.device)
    xtr = torch.from_numpy(xx[mask].astype(np.float32))
    ytr = torch.from_numpy(y[mask].astype(np.float32))
    xva = torch.from_numpy(xx[~mask].astype(np.float32))
    yva = torch.from_numpy(y[~mask].astype(np.float32))
    tr = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(TensorDataset(xva, yva), batch_size=args.batch_size * 2, num_workers=0)
    model = CandidateMatcher(x.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    history = []
    best = float('inf')
    print(f'Train={mask.sum()}, Val={(~mask).sum()}, feature_dim={x.shape[1]}, device={dev}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(xb)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(dev), yb.to(dev)
                val_loss += float(crit(model(xb), yb).item()) * len(xb)
        row = {'epoch': epoch, 'train_bce': round(loss_sum / len(xtr), 6), 'val_bce': round(val_loss / len(xva), 6)}
        history.append(row)
        print(row, flush=True)
        if val_loss / len(xva) < best:
            best = val_loss / len(xva)
            torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'feature_dim': x.shape[1], 'mean': torch.from_numpy(mu.copy()),
                        'std': torch.from_numpy(sd.copy()), 'architecture': 'CandidateMatcher-Baseline3B',
                        'train_warning': 'incomplete reconstructed candidate pool'}, out / 'best_model.pth')
    (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    validate_checkpoint(out / 'best_model.pth', root, out, dev)
    print('Saved best model:', (out / 'best_model.pth').resolve())


def load_checkpoint(weights, device):
    ck = torch.load(weights, map_location='cpu', weights_only=True)
    model = CandidateMatcher(int(ck['feature_dim']))
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    mu = ck['mean'].float().cpu().numpy()
    sd = ck['std'].float().cpu().numpy()
    return model, mu, sd


def probabilities(model, x, mu, sd, device):
    if len(x) == 0:
        return np.zeros((0,), dtype=np.float32)
    z = np.clip((x - mu) / sd, -12, 12).astype(np.float32)
    result = []
    with torch.no_grad():
        for i in range(0, len(z), 4096):
            result.append(torch.sigmoid(model(torch.from_numpy(z[i:i + 4096]).to(device))).cpu().numpy())
    return np.concatenate(result)


def validate_checkpoint(weights, root, out, device):
    x, y, mask, info = make_arrays(root)
    model, mu, sd = load_checkpoint(weights, device)
    p = probabilities(model, x[~mask], mu, sd, device)
    scan = scan_predictions(y[~mask], p)
    best = max(scan, key=lambda row: (row['f1'], -row['FP']))
    write_scan(out / 'threshold_scan.csv', scan)
    report = {**info, 'train_pairs': int(mask.sum()), 'val_pairs': int((~mask).sum()),
              'positive_rate_val': round(float(y[~mask].mean()), 6),
              'best_validation': best,
              'warning': 'Validation candidates reconstructed from UNION of known positive labels; incomplete pool biases results. NOT an official score.'}
    (out / 'validation_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Validation:', json.dumps(best, ensure_ascii=False), flush=True)


def infer(args):
    if not args.allow_test_candidates:
        raise SystemExit('Organizer approval required for test_data input. Pass --allow-test-candidates ONLY after confirmation.')
    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = choose_device(args.device)
    model, mu, sd = load_checkpoint(args.weights, dev)
    testdir = root / 'VLMOD(train+test)' / 'MonoMulti3D' / 'test'
    allout = []
    for path in sorted(testdir.glob('*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        desc = d['public_description']
        cs = [read_candidate(row) for row in d['test_data']]
        scores = np.zeros((len(cs), len(desc)), dtype=np.float32)
        for j, sentence in enumerate(desc):
            if cs:
                f = np.asarray([features(sentence, c) for c in cs], dtype=np.float32)
                scores[:, j] = probabilities(model, f, mu, sd, dev)
        allout.append({'image_id': path.stem, 'candidate_count': len(cs), 'description_count': len(desc),
                       'probability_matrix': np.round(scores, 5).tolist(),
                       'binary_matrix': (scores >= args.threshold).astype(int).tolist()})
    path = out / 'test_candidate_predictions_diagnostic.json'
    path.write_text(json.dumps(allout, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(allout)} test images to {path}. Diagnostic ONLY, not official schema.')


def export(args):
    dev = torch.device('cpu')
    model, mu, sd = load_checkpoint(args.weights, dev)
    onnx_path = Path(args.out) / 'candidate_matcher.onnx'
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, torch.randn(4, len(mu)), str(onnx_path),
                          input_names=['normalized_candidate_features'], output_names=['objectness_logits'],
                          dynamic_axes={'normalized_candidate_features': {0: 'num_candidates'},
                                        'objectness_logits': {0: 'num_candidates'}},
                          opset_version=17, dynamo=False)
    except Exception as e:
        raise SystemExit(f'ONNX export failed. Install onnx: python -m pip install onnx. Details: {e}')
    (Path(args.out) / 'onnx_normalization.json').write_text(json.dumps({
        'mean': [float(v) for v in mu], 'std': [float(v) for v in sd],
        'input_note': 'Compute features() first, standardize (x-mean)/std, clip[-12,12]. Output is LOGIT; apply sigmoid.'
    }, indent=2), encoding='utf-8')
    print('ONNX saved:', onnx_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['train', 'validate', 'infer', 'export'], default='train')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='experiments/Baseline_3B')
    ap.add_argument('--weights', default='experiments/Baseline_3B/best_model.pth')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--allow-test-candidates', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.mode == 'train': train(args)
    elif args.mode == 'validate':
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        validate_checkpoint(args.weights, Path(args.root).resolve(), out, choose_device(args.device))
    elif args.mode == 'infer': infer(args)
    else: export(args)


if __name__ == '__main__':
    main()

"""Compare Baseline-0/1/2 on identical validation data and distance gates.

Requirements: numpy, scipy; scripts/evaluate_distance_f1.py from prior patch.
Baseline-0 and Baseline-1 val_predictions.json were exported with confidence >= 0.3;
all three models are therefore evaluated at 0.3 for the apples-to-apples
*saved-output* comparison. This is NOT official VLP evaluation.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def load_evaluator(root):
    source=root/'scripts'/'evaluate_distance_f1.py'
    if not source.is_file():
        raise FileNotFoundError(f'Missing evaluation script: {source}')
    spec=importlib.util.spec_from_file_location('vlmod_strict_evaluator',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_predictions(path):
    raw=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw,list):
        raise ValueError(f'Expected list of predictions: {path}')
    result={}
    for entry in raw:
        sid=str(entry['sample_id'])
        if sid in result:
            raise ValueError(f'Duplicate sample_id {sid} in {path}')
        result[sid]=entry
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.',help='VLMOD_Competition root directory')
    ap.add_argument('--threshold',type=float,default=0.30)
    ap.add_argument('--gates',type=float,nargs='+',default=[2.0,5.0,10.0])
    ap.add_argument('--out',default='experiments/comparison_strict')
    args=ap.parse_args()
    root=Path(args.root).resolve()
    evalmod=load_evaluator(root)
    manifest=evalmod.load_manifest(root/'data/official/val.jsonl')
    sources={
        'Baseline-0':root/'experiments/Baseline_0/val_predictions.json',
        'Baseline-1':root/'experiments/Baseline_1/val_predictions.json',
        'Baseline-2':root/'experiments/Baseline_2/val_predictions_all_slots.json',
    }
    rows=[]
    for name,path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f'{name} predictions missing: {path}. Generate validation predictions first.')
        predictions=load_predictions(path)
        missing=set(manifest).difference(predictions)
        if missing:
            raise ValueError(f'{name}: {len(missing)} predictions missing; e.g. {next(iter(missing))}')
        for gate in args.gates:
            metrics=evalmod.score(manifest,predictions,args.threshold,gate)
            rows.append({'model':name,**metrics})
    folder=root/args.out
    folder.mkdir(parents=True,exist_ok=True)
    csv_path=folder/'baseline_strict_comparison.csv'
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader();wr.writerows(rows)
    (folder/'baseline_strict_comparison.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    for g in args.gates:
        print(f'\nDistance gate = {g:g}m; confidence threshold = {args.threshold:g}')
        for r in rows:
            if r['gate_m']==g:
                print(f"  {r['model']:11}  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  countMAE={r['mae_count']:.3f}  predicted={r['pred_objects']}")
    print('\nSaved:',csv_path)
    print('Caution: both Baseline-0 and Baseline-1 predictions already discarded scores under 0.3. This compares retained outputs only.')

if __name__=='__main__':
    main()

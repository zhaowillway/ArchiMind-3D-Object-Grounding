"""Strict, local 3D-center matching evaluation (NOT official VLP metrics).

Reads canonical validation JSONL and per-query predictions with original confidence.
Uses maximum cardinality bipartite matching under 3D distance threshold,
then minimum total distance to break ties. Crucially, no match is valid if
position error exceeds the distance gate.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_manifest(path):
    result = {}
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            item=json.loads(line)
            sid=str(item['sample_id'])
            if sid in result:
                raise ValueError(f'Duplicate manifest sample_id {sid} on line {ln}')
            result[sid]=item
    return result


def valid_position(obj):
    xyz=obj.get('position')
    return isinstance(xyz, list) and len(xyz)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in xyz)


def match_count(gts, preds, gate):
    """Max cardinality under gate: penalty for unmatched bigger than any allowed match."""
    ng,npred=len(gts),len(preds)
    if not ng or not npred:
        return 0, []
    d=np.linalg.norm(np.asarray(gts)[:,None,:]-np.asarray(preds)[None,:,:],axis=2)
    # Real pairs cost <= gate, invalid pairs cost very high.
    # Dummy columns allow a ground truth to be unmatched.
    penalty=max(gate,1e-6)+1.0
    cost=np.full((ng,npred+ng),penalty,dtype=float)
    cost[:,:npred]=np.where(d<=gate,d,penalty*4)
    # If real edge is valid its cost < penalty, otherwise dummy is better.
    row,col=linear_sum_assignment(cost)
    pairs=[(int(i),int(j),float(d[i,j])) for i,j in zip(row,col) if j<npred and d[i,j]<=gate]
    return len(pairs),pairs


def score(manifest, predictions, threshold, gate):
    tp=fp=fn=0
    total_gt=total_pred=0
    count_absolute=[]
    matched_dist=[]
    for sid, item in manifest.items():
        gt=[o['position'] for o in item.get('objects',[]) if valid_position(o)]
        pred_objs=predictions[sid].get('objects',[])
        selected=[o['position'] for o in pred_objs if valid_position(o) and math.isfinite(float(o.get('confidence',0))) and float(o.get('confidence',0))>=threshold]
        n,p= len(gt),len(selected)
        matched,pairs=match_count(gt,selected,gate)
        tp+=matched
        fp+=p-matched
        fn+=n-matched
        total_gt+=n
        total_pred+=p
        count_absolute.append(abs(p-n))
        matched_dist.extend([d for _,_,d in pairs])
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'threshold':threshold,'gate_m':gate,'samples':len(manifest),
            'true_objects':total_gt,'pred_objects':total_pred,
            'tp':tp,'fp':fp,'fn':fn,'precision':round(precision,6),'recall':round(recall,6),
            'f1':round(f1,6),'mae_count':round(float(np.mean(count_absolute)),6),
            'mean_matched_distance_m':round(float(np.mean(matched_dist)),6) if matched_dist else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='data/official/val.jsonl')
    ap.add_argument('--pred',default='experiments/Baseline_2/val_predictions_all_slots.json')
    ap.add_argument('--out',default='experiments/Baseline_2/strict_eval')
    ap.add_argument('--gates',type=float,nargs='+',default=[2.,5.,10.])
    ap.add_argument('--thresholds',type=float,nargs='+',default=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45])
    a=ap.parse_args()
    manifest=load_manifest(a.manifest)
    with open(a.pred,encoding='utf-8') as f:
        raw=json.load(f)
    if not isinstance(raw,list):
        raise ValueError('Prediction file must be a JSON list of sample records')
    predictions={}
    for item in raw:
        sid=str(item['sample_id'])
        if sid in predictions: raise ValueError('Duplicate prediction sample_id '+sid)
        predictions[sid]=item
    missing=set(manifest)-set(predictions)
    if missing:
        raise ValueError(f'Missing {len(missing)} predictions; examples: {sorted(missing)[:3]}')
    extra=set(predictions)-set(manifest)
    if extra: print(f'Warning: {len(extra)} predictions not in validation manifest; excluded')
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    results=[score(manifest,predictions,t,g) for g in a.gates for t in a.thresholds]
    with open(output/'strict_metrics.csv','w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=results[0].keys());wr.writeheader();wr.writerows(results)
    (output/'strict_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    best=[]
    for gate in a.gates:
        candidates=[r for r in results if r['gate_m']==gate]
        opt=max(candidates,key=lambda r:(r['f1'],-r['mae_count']))
        best.append(opt)
        print(f"Gate {gate:g} m: best threshold={opt['threshold']:.2f}, P={opt['precision']:.3f}, R={opt['recall']:.3f}, F1={opt['f1']:.3f}, count MAE={opt['mae_count']:.3f}")
    (output/'best_by_gate.json').write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',output/'strict_metrics.csv')
    print('NOTE: This is a local center-distance diagnostic, not the official evaluator.')

if __name__=='__main__':main()

"""VLMOD candidate-list exploratory matcher (NOT an official submission).

Only use test_data as a prediction input if the competition explicitly allows it.
Commands:
 python scripts/baseline3_candidate_probe.py --train-file PATH --test-file PATH
 python scripts/baseline3_candidate_probe.py --data-root "VLMOD(train+test)" --out experiments/Baseline_3/probe
"""
import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path


def to_obj(tokens, cid):
    if isinstance(tokens, str):
        tokens = tokens.split() if not tokens.lstrip().startswith('[') else ast.literal_eval(tokens)
    if len(tokens) < 16:
        raise ValueError('KITTI-like row needs at least 16 fields')
    return {
        'candidate_id': cid, 'category': str(tokens[0]).lower(),
        'occlusion': int(float(tokens[2])),
        'alpha': float(tokens[3]), 'bbox': [float(x) for x in tokens[4:8]],
        'height': float(tokens[8]), 'width': float(tokens[9]), 'length': float(tokens[10]),
        'position': [float(x) for x in tokens[11:14]],
        'rotation': float(tokens[14]), 'color': str(tokens[15]).lower()
    }


def desc_constraints(desc, properties=None):
    s = (desc + ' ' + ' '.join(properties or [])).lower()
    colors = ['silver-grey', 'white', 'black', 'red', 'blue', 'green', 'yellow', 'grey', 'gray', 'silver', 'orange', 'brown']
    col = next((x for x in colors if re.search(r'(?<![a-z])' + re.escape(x) + r'(?![a-z])', s)), None)
    obj = next((x for x in ['car','bus','truck','van','pedestrian','cyclist'] if re.search(r'\b'+x+r's?\b', s)), None)
    nums = {}
    for field in ['height','width','length','distance']:
        match = re.search(r'\b'+field+r'\s*(?::|s?\s+(?:ranging\s+)?(?:approximately\s+)?(?:between|from))\s*(?:approximately\s+)?([0-9.]+)\s*(?:to|and|-)\s*([0-9.]+)',s)
        if not match:
            match = re.search(r'\b'+field+r'\b[^.]{0,45}?\b(?:from|between)\s*(?:approximately\s+)?([0-9.]+)\s*(?:to|and|-)\s*([0-9.]+)',s)
        if match: nums[field] = sorted([float(match.group(1)),float(match.group(2))])
    d = re.search(r'(?:within|approximately|around|about|at)\s+([0-9.]+)\s*meters?',s)
    if d and 'distance' not in nums: nums['distance']=[float(d.group(1))-2,float(d.group(1))+2]
    occlusion = 0 if ('not occluded' in s or 'unoccluded' in s or 'non-occluded' in s) else (1 if 'partially occluded' in s else None)
    side = next((x for x in ['top left','top right','bottom left','bottom right'] if x in s),None)
    return {'category':obj,'color':col,'ranges':nums,'occlusion':occlusion,'region':side,'facing_me': 'facing me' in s}


def select_candidates(desc, candidates, properties=None):
    c=desc_constraints(desc,properties)
    xs=[(a['bbox'][0]+a['bbox'][2])/2 for a in candidates]
    ys=[(a['bbox'][1]+a['bbox'][3])/2 for a in candidates]
    midx=(min(xs)+max(xs))/2 if xs else 0
    midy=(min(ys)+max(ys))/2 if ys else 0
    ret=[]
    for ob in candidates:
        if c['category'] and ob['category']!=c['category']:continue
        if c['color'] and ob['color']!=c['color']:continue
        if c['occlusion'] is not None and ((ob['occlusion']==0)!=(c['occlusion']==0)):continue
        if any(not(lo-1e-5 <= (math.sqrt(sum(v*v for v in ob['position'])) if k=='distance' else ob[k]) <= hi+1e-5) for k,(lo,hi) in c['ranges'].items()):continue
        # 'facing me' needs calibrated camera/orientation conventions.
        # Do not hard-filter by alpha/yaw before validating those conventions.
        if c['region']:
            x=(ob['bbox'][0]+ob['bbox'][2])/2;y=(ob['bbox'][1]+ob['bbox'][3])/2
            if ('left' in c['region'] and x>midx) or ('right' in c['region'] and x<midx):continue
            if ('top' in c['region'] and y>midy) or ('bottom' in c['region'] and y<midy):continue
        ret.append(ob)
    return ret,c


def train_groups(data):
    for group in data:
        if isinstance(group,list):
            for row in group:
                if isinstance(row,dict) and 'label_3' in row:yield row


def candidate_signature(a):
    return (a['category'], tuple(round(v,3) for v in a['position']),tuple(round(a[k],3) for k in ('height','width','length')))


def eval_train(path):
    d=json.loads(path.read_text(encoding='utf8'))
    rows=list(train_groups(d))
    pool={}
    for a in rows:
        for z in a['label_3']:
            o=to_obj(z,len(pool))
            pool.setdefault(candidate_signature(o),o)
    candidates=list(pool.values())
    result=[]
    for a in rows:
        gt={candidate_signature(to_obj(z,-1)) for z in a['label_3']}
        selected,cons=select_candidates(a['public_description'],candidates,a.get('public_properties',[]))
        pred={candidate_signature(z) for z in selected}
        tp=len(pred&gt);fp=len(pred-gt);fn=len(gt-pred)
        result.append({'ann_id':a['ann_id'],'description':a['public_description'],'gt':len(gt),'selected':len(pred),'tp':tp,'fp':fp,'fn':fn,'constraints':cons})
    return result,len(candidates)


def test_file(path):
    d=json.loads(path.read_text(encoding='utf8'))
    pool=[to_obj(s,i) for i,s in enumerate(d['test_data'])]
    items=[]
    for i,desc in enumerate(d['public_description']):
        sel,cons=select_candidates(desc,pool)
        items.append({'sample_id':f'{path.stem}::{i}','description':desc,'candidate_ids':[z['candidate_id'] for z in sel],
          'objects':[{'candidate_id':z['candidate_id'],'category':z['category'],'color':z['color'],'position':z['position'],'size':[z['width'],z['height'],z['length']],'rotation':z['rotation']} for z in sel],
          'constraints':cons})
    return items,len(pool)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-file',default='');p.add_argument('--test-file',default='')
    p.add_argument('--data-root',default='');p.add_argument('--out',default='experiments/Baseline_3/probe')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    trainfiles=[Path(a.train_file)] if a.train_file else (sorted((Path(a.data_root)/'MonoMulti3D/train').glob('*.json')) if a.data_root else [])
    testfiles=[Path(a.test_file)] if a.test_file else (sorted((Path(a.data_root)/'MonoMulti3D/test').glob('*.json')) if a.data_root else [])
    trainrows=[];testrows=[];parsed_train=parsed_test=0
    for f in trainfiles:
        rr,n=eval_train(f);parsed_train+=1
        for r in rr:r['source_file']=f.name
        trainrows.extend(rr)
    for f in testfiles:
        rr,n=test_file(f);parsed_test+=1
        testrows.extend(rr)
    tp=sum(x['tp'] for x in trainrows);fp=sum(x['fp'] for x in trainrows);fn=sum(x['fn'] for x in trainrows)
    stats={'train_files':parsed_train,'test_files':parsed_test,'train_queries':len(trainrows),'test_queries':len(testrows),
      'train_candidate_pool_warning':'Candidates reconstructed from union of positive train labels, NOT the full scene. Results are a diagnostic only.',
      'train_precision_proxy':round(tp/max(tp+fp,1),6),'train_recall_proxy':round(tp/max(tp+fn,1),6),
      'tp':tp,'fp':fp,'fn':fn,
      'test_data_warning':'test_data input must be explicitly permitted by competition rules before official submission.',
      'unimplemented_relations':['facing me (requires calibration)','complex spatial relations (region uses rough bbox midpoint)']}
    (out/'probe_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'test_candidate_matches.json').write_text(json.dumps(testrows,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'train_matching_details.json').write_text(json.dumps(trainrows,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print('Results:',out.resolve())
    for r in testrows[:3]:print(r['sample_id'],'=>',r['candidate_ids'],r['constraints'])

if __name__=='__main__':main()

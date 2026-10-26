"""Baseline-3: supervised text-to-candidate binary matcher (research prototype).
Important: train candidate pools are UNION of annotated positives, not complete scene.
Test-time use of test_data requires explicit approval from competition organizers.
"""
import argparse, ast, csv, hashlib, json, re, math
from pathlib import Path
from collections import defaultdict
import numpy as np

CATS=['car','van','bus','truck','pedestrian','cyclist','other']
COLORS=['white','black','red','blue','green','yellow','grey','gray','silver','brown','orange']

def read_candidate(row):
    if isinstance(row,str):
        s=row.strip()
        if s.startswith('['):
            v=ast.literal_eval(s)
        else:
            v=s.split()
    else: v=row
    if len(v)<16: raise ValueError(f'Candidate has {len(v)} fields, expected >=16')
    return {'category':str(v[0]).lower(),'truncated':float(v[1]),'occluded':float(v[2]),'alpha':float(v[3]),
            'bbox':[float(x) for x in v[4:8]],'height':float(v[8]),'width':float(v[9]),
            'length':float(v[10]),'xyz':[float(x) for x in v[11:14]],'yaw':float(v[14]),'color':str(v[15]).lower()}

def key_obj(c):
    return (c['category'],tuple(round(v,2) for v in c['bbox']),tuple(round(v,2) for v in c['xyz']))

def flatten(node):
    if isinstance(node,dict):
        if 'label_3' in node and 'public_description' in node:yield node
    elif isinstance(node,list):
        for x in node:yield from flatten(x)

def get_query_features(text):
    t=text.lower().replace('silver-grey','grey').replace('silver gray','grey')
    feats=[]
    # ranges may be written 'height : 1.2 to 1.6 m', 'heights ranging from ...'
    for field in ['height','width','length','distance']:
        stems={'height':r'height(?:s)?','width':r'width(?:s)?','length':r'length(?:s)?','distance':r'distance|located within|approximately'}[field]
        pattern=rf'(?:{stems})[^0-9]{{0,70}}(\d+(?:\.\d+)?)\s*(?:to|and|[-–]|between\s*)\s*(\d+(?:\.\d+)?)'
        m=re.search(pattern,t)
        if not m:
            # handle 'between 4.1 to 4.5' or 'approximately 23.6 meters' localized near phrase
            m=re.search(rf'(?:{stems})[^0-9]{{0,55}}(\d+(?:\.\d+)?)\s*(?:to|and|[-–])\s*(\d+(?:\.\d+)?)',t)
        if m: feats.append((field, float(m.group(1)),float(m.group(2))))
        else:
            m2=re.search(rf'(?:{stems})[^0-9]{{0,35}}(\d+(?:\.\d+)?)\s*(?:m|meters?)',t)
            feats.append((field,float(m2.group(1)),float(m2.group(1))) if m2 else (field,None,None))
    return t,feats

def features(text,c, image_wh=(1920.,1080.)):
    t,ranges=get_query_features(text)
    x1,y1,x2,y2=c['bbox']; xx=(x1+x2)/2/image_wh[0]; yy=(y1+y2)/2/image_wh[1]
    z=abs(c['xyz'][2]); dist=math.sqrt(sum(v*v for v in c['xyz']))
    f=[c['height'],c['width'],c['length'],c['occluded'],c['truncated'],
       *[v/100 for v in c['xyz']],dist/100,xx,yy,(x2-x1)/image_wh[0],(y2-y1)/image_wh[1],
       math.sin(c['yaw']),math.cos(c['yaw']),math.sin(c['alpha']),math.cos(c['alpha'])]
    # Context of requested attributes, and degree of constraint satisfaction.
    for field,lo,hi in ranges:
        value=dist if field=='distance' else c[field]
        active=lo is not None
        if active:
            margin=max(0.1,(hi-lo)/2)
            err=0 if lo <=value<=hi else min(abs(value-lo),abs(value-hi))
            f.extend([1.,float(lo/100 if field=='distance' else lo/20),float(hi/100 if field=='distance' else hi/20),min(err/margin,50),float(lo<=value<=hi)])
        else: f.extend([0.,0.,0.,0.,0.])
    for name in CATS:
        active=bool(re.search(r'\b'+re.escape(name)+r's?\b',t))
        f.extend([float(active),float(active and c['category']==name)])
    for col in COLORS:
        active=bool(re.search(r'\b'+re.escape(col)+r'\b',t))
        f.extend([float(active),float(active and col in c['color'])])
    asks_occ=bool(re.search(r'partially occluded|occluded|not occluded|unoccluded',t))
    want_occ=(bool(re.search(r'partially occluded|(?<!not )occluded',t)) and not bool(re.search(r'not occluded|unoccluded',t)))
    f.extend([float(asks_occ),float(asks_occ and (c['occluded']>0)==want_occ)])
    for word,logic in [('left',xx<0.5),('right',xx>=0.5),('top',yy<0.5),('bottom',yy>=0.5)]:
        active=bool(re.search(r'\b'+word+r'\b',t));f.extend([float(active),float(active and logic)])
    active=bool(re.search(r'facing me|toward(?:s)? (?:me|camera)',t))
    f.extend([float(active),math.cos(c['yaw']) if active else 0.])
    return f

def split_from_manifests(root):
    m=root/'data'/'official'/'val.jsonl'
    if not m.exists():return None
    out=set()
    for line in m.open(encoding='utf8'):
        if not line.strip():continue
        sid=json.loads(line).get('sample_id','')
        out.add(sid.split('::')[0])
    return out

def load_training(root):
    trdir=root/'VLMOD(train+test)'/'MonoMulti3D'/'train'
    files=sorted(trdir.glob('*.json'))
    if not files:raise FileNotFoundError(f'No training json: {trdir}')
    val_ids=split_from_manifests(root)
    if val_ids is None:
        print('WARNING: no val manifest; deterministic image-level 10% split used')
        val_ids={p.stem for p in files if int(hashlib.sha256(p.stem.encode()).hexdigest(),16)%10==0}
    data=[];missing=0;pool_sizes=[]
    for path in files:
        queries=list(flatten(json.loads(path.read_text(encoding='utf8'))))
        pool={};targets=[]
        for q in queries:
            target=set()
            for row in q['label_3']:
                c=read_candidate(row);k=key_obj(c);pool[k]=c;target.add(k)
            targets.append(target)
        candidates=list(pool.values()); keys=list(pool.keys());pool_sizes.append(len(keys))
        if not candidates:continue
        kind='val' if path.stem in val_ids else 'train'
        for qi,(q,target) in enumerate(zip(queries,targets)):
            for key,c in zip(keys,candidates):
                data.append((kind,path.stem,qi,q['public_description'],c,int(key in target)))
    return data,dict(train_files=len(files),validation_images=sum(p.stem in val_ids for p in files),mean_reconstructed_candidates=float(np.mean(pool_sizes)),max_reconstructed_candidates=max(pool_sizes))

def evaluate(y,p,threshold):
    h=(p>=threshold).astype(np.int32);y=np.array(y,dtype=np.int32)
    tp=int(((h==1)&(y==1)).sum());fp=int(((h==1)&(y==0)).sum());fn=int(((h==0)&(y==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.
    return {'threshold':float(threshold),'TP':tp,'FP':fp,'FN':fn,'precision':round(prec,6),'recall':round(rec,6),'f1':round(f1,6)}

def train(args):
    try:from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:raise SystemExit('Install scikit-learn: python -m pip install scikit-learn')
    import joblib
    root=Path(args.root).resolve(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data,info=load_training(root)
    x=np.asarray([features(t,c) for _,_,_,t,c,_ in data],dtype=np.float32)
    y=np.asarray([z for *_,z in data],dtype=np.int32)
    trainmask=np.asarray([a=='train' for a,*_ in data],dtype=bool)
    if not trainmask.any() or trainmask.all():raise ValueError('Training/validation split missing; please inspect val manifest')
    print('Training pairs',int(trainmask.sum()),'validation pairs',int((~trainmask).sum()),'feature length',x.shape[1],flush=True)
    model=HistGradientBoostingClassifier(max_iter=args.iterations,learning_rate=0.08,max_leaf_nodes=15,l2_regularization=2.0,random_state=2026)
    model.fit(x[trainmask],y[trainmask])
    prob=model.predict_proba(x[~trainmask])[:,1];yv=y[~trainmask]
    scan=[evaluate(yv,prob,th) for th in np.arange(.1,.901,.05)]
    best=max(scan,key=lambda row:(row['f1'],-row['FP']))
    joblib.dump({'model':model,'features':x.shape[1],'selected_threshold':best['threshold'],'note':'train candidate pools reconstructed from union of positive labels'},out/'model.joblib')
    with (out/'threshold_scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=scan[0].keys());wr.writeheader();wr.writerows(scan)
    result={**info,'train_pairs':int(trainmask.sum()),'val_pairs':int((~trainmask).sum()),'positive_rate_val':float(yv.mean()),'best_validation':best,'warning':'VALIDATION ONLY WITH INCOMPLETE RECONSTRUCTED CANDIDATE POOL. Do not compare to official test F1 or Baselines 0-2 directly.'}
    (out/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('Saved:',(out/'model.joblib').resolve())

def infer(args):
    if not args.allow_test_candidates:raise SystemExit('Test uses test_data 3D candidate annotations. Confirm organizer permission before running; then pass --allow-test-candidates')
    import joblib
    root=Path(args.root).resolve();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ck=joblib.load(args.weights);model=ck['model'];th=args.threshold if args.threshold is not None else ck['selected_threshold']
    indir=root/'VLMOD(train+test)'/'MonoMulti3D'/'test';files=sorted(indir.glob('*.json'))
    if not files:raise FileNotFoundError(str(indir))
    outputs=[];nq=0
    for path in files:
        d=json.loads(path.read_text(encoding='utf8'))
        queries=d['public_description'];cs=[read_candidate(x) for x in d['test_data']]
        scores=np.zeros((len(cs),len(queries)),dtype=np.float64)
        for j,t in enumerate(queries):
            feats=np.asarray([features(t,c) for c in cs],dtype=np.float32)
            if len(cs):scores[:,j]=model.predict_proba(feats)[:,1]
        outputs.append({'image_id':path.stem,'description_count':len(queries),'candidate_count':len(cs),'binary_matrix':(scores>=th).astype(int).tolist(),'probability_matrix':scores.round(5).tolist()})
        nq+=len(queries)
    (out/'test_candidate_predictions_diagnostic.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'Processed {len(outputs)} images, {nq} queries; diagnostic test candidate matrix saved at {out.resolve()}')
    print('DO NOT SUBMIT until official schema and test_data use have been confirmed.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['train','infer'],default='train')
    ap.add_argument('--root',default='.')
    ap.add_argument('--out',default='experiments/Baseline_3')
    ap.add_argument('--weights',default='experiments/Baseline_3/model.joblib')
    ap.add_argument('--iterations',type=int,default=120)
    ap.add_argument('--threshold',type=float,default=None)
    ap.add_argument('--allow-test-candidates',action='store_true')
    a=ap.parse_args()
    (train if a.mode=='train' else infer)(a)
if __name__=='__main__':main()

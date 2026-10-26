"""Canonical JSONL dataset loader. Official data require an explicit schema adapter."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def tokenize(text, seq=24, vocab=2048):
    parts=re.findall(r'\w+',text.lower(),flags=re.UNICODE)[:seq]
    ids=[1+int(hashlib.sha256(t.encode('utf-8')).hexdigest()[:12],16)%(vocab-1) for t in parts]
    return ids+[0]*(seq-len(ids))

class CanonicalDataset(Dataset):
    def __init__(self, manifest, slots=2, image_size=64, labeled=True):
        manifest=Path(manifest)
        self.root=manifest.parent
        self.items=[json.loads(l) for l in manifest.read_text(encoding='utf-8').splitlines() if l.strip()]
        if not self.items: raise ValueError('Empty manifest: '+str(manifest))
        self.slots=slots;self.image_size=image_size;self.labeled=labeled
        if labeled and any('objects' not in x for x in self.items):
            raise ValueError('Missing objects labels; cannot train on unlabeled data')
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]; path=self.root/item['image']
        with Image.open(path) as img:
            pixels=np.asarray(img.convert('RGB').resize((self.image_size,self.image_size)),dtype=np.float32)/255.
        image=torch.from_numpy(pixels.transpose(2,0,1).copy())
        tokens=torch.tensor(tokenize(item['text']),dtype=torch.long)
        target=torch.zeros((self.slots,8),dtype=torch.float32)
        for j,obj in enumerate(item.get('objects',[])[:self.slots]):
            vals=list(obj['position'])+list(obj['size'])+[obj['rotation']]
            if len(vals)!=7:raise ValueError('Each object requires xyz, size whd, yaw')
            target[j]=torch.tensor([1.]+vals,dtype=torch.float32)
        return image,tokens,target,str(item.get('sample_id',i))

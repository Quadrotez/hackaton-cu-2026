"""Test-time consistency: cluster similar test crops (DINOv2 embeddings), aggregate attributes."""
import csv
from pathlib import Path
import numpy as np
import cv2, torch
import timm

ROOT = Path('/mnt/qdisk/Projects/astramis')
CROPS = ROOT / 'crops'

def load_crop(key):
    a = np.load(CROPS / f'test_{key}.npy').astype(np.float32) / 255.0
    a = np.stack([cv2.resize(a[k], (224,224)) for k in range(4)])
    return torch.from_numpy(a)

def main():
    test = list(csv.DictReader(open(CROPS/'test.csv')))
    keys = [r['key'] for r in test]
    keys = [k for k in keys if (CROPS/f'test_{k}.npy').exists()]
    m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224)
    w = m.patch_embed.proj.weight
    m.patch_embed.proj.weight = torch.nn.Parameter(torch.cat([w, w.mean(dim=1, keepdim=True)], dim=1))
    m.eval().cuda()
    embs=[]
    with torch.no_grad():
        for i in range(0,len(keys),32):
            x=torch.stack([load_crop(k) for k in keys[i:i+32]]).cuda()
            embs.append(m.forward_features(x).float().mean(dim=1).cpu().numpy())
    E=np.concatenate(embs); E/= (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    print('emb',E.shape,flush=True)
    sim=E@E.T
    parent=list(range(len(keys)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            if sim[i,j]>0.94:
                a,b=find(i),find(j)
                if a!=b: parent[b]=a
    groups={}
    for i,k in enumerate(keys): groups.setdefault(find(i),[]).append(i)
    print('groups',len(groups),'multi',sum(len(v)>1 for v in groups.values()),flush=True)
    champ={r[0]:r[1] for r in list(csv.reader(open(ROOT/'champion_83_9142.csv',encoding='utf-8')))[1:]}
    out=[['image_id','prediction']]
    changed=0
    for img,pred in champ.items():
        objs=[] if pred.strip() in ('','none') else pred.split(';')
        for r in test:
            pass
        # map crop idx to object
        idx_map={int(r['idx']):r['key'] for r in test if r['image_id']==img}
        for i,o in enumerate(objs):
            key=idx_map.get(i)
            if key is None or key not in keys: continue
            gi=find(keys.index(key)); members=groups[gi]
            if len(members)<2: continue
            t=o.split()
            vols=[]
            for mi in members:
                mk=keys[mi]
                # find champion volume for this crop
                mi_img,mi_idx=mk.rsplit('_',1)
                pj=champ.get(mi_img,'').split(';') if champ.get(mi_img,'')!='none' else []
                if int(mi_idx)<len(pj): vols.append(float(pj[int(mi_idx)].split()[7]))
            if len(vols)>=2:
                newv=float(np.exp(np.mean(np.log(np.maximum(vols,1e-6)))))
                if abs(newv/float(t[7])-1)>0.02:
                    t[7]=f'{newv:.6f}'; changed+=1
            objs[i]=' '.join(t)
        out.append([img,';'.join(objs) if objs else 'none'])
    with open(ROOT/'submission_cluster_vol.csv','w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerows(out)
    print('changed',changed,flush=True)

if __name__=='__main__':
    main()

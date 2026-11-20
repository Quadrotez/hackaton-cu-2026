"""Paired M/V ablations and strictly nested take cut selection."""
from astra_detector_stack01 import *
from volume_hybrid_eval import masked_geometry
from cv_volume_v2 import ridge_fit_predict,GEOMETRY_PATH,RIDGE_X
from volume_v2 import VOLUME_CATS
from take_v2 import inner_folds
from detector_fold_eval import detect
from ultralytics import YOLO
import sys

P="astra_ablate03_"
root=W/"detector_oof_v1"
spec=json.loads((W/"detector_folds_v1.json").read_text())["folds"]
ref=json.loads((W/"astra_detector_stack01_predictions.json").read_text())
frames={i:((DATA/im["file_name"]).parent,aa) for s,i,im,aa in load_frames() if s=="real"}
gt=ground_truth([(i,path,aa) for i,(path,aa) in frames.items()])
cam=json.loads((DATA/"camera.json").read_text())
data,cols=read_data(); data["split"]="train"
teacher=normalize(embeddings("astra_",keys(data)));clip=normalize(embeddings("astra_clip_",keys(data)))
ix={k:i for i,k in enumerate(keys(pd.read_csv(W/"cv_oof.csv")))}
student=np.load(W/"astra_tmp/astra_small_vectors.npy")[[ix[k] for k in keys(data)]]

def read(i):
    path=frames[i][0]
    return (cv2.imread(str(path/"color.png")),cv2.imread(str(path/"depth.png"),cv2.IMREAD_UNCHANGED),
            json.loads((path/"meta.json").read_text()))

def extract(i,obj,mask,color,depth,meta,geometry):
    h,w=depth.shape;x,y,r,b=obj["box"]; box=(x,y,r-x,b-y);l,t,rr,bb=bounds(box,w,h,.1)
    m=mask[t:bb,l:rr]
    row=dict(image_id=i,form=obj["form"],lighting=meta["lighting"],surface=meta["surface"],
             raw_volume_ml=raw_volume_ml(depth,mask,meta["floor_plane"],cam),
             **region_features(color,depth,box,m,meta,cam,l,t))
    if geometry:row.update(masked_geometry(depth,mask,box,meta,cam))
    return row,masked_crop(color[t:bb,l:rr],m)

def visual(crops,compact,cm,t):
    s=normalize(compact.encode(crops)); c=[]
    with torch.inference_mode():
        for j in range(0,len(crops),8):
            batch=torch.stack([t(Image.fromarray(q[:,:,::-1])) for q in crops[j:j+8]])
            c.append(cm.encode_image(batch).numpy())
    return s,normalize(np.concatenate(c))

def heads(tr,sv,cv,fv=None,aug=False):
    h=fit_compact_head(tr,teacher,student)
    def probs(X,V,augX=None):
        x=X[tr.index];y=tr.material
        if augX is not None:x=np.vstack([x,augX[tr.index]]);y=np.tile(y,2)
        lr=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(x,y,sample_weight=np.full(len(y),.5 if augX is not None else 1.))
        out=np.zeros((len(V),len(labels)))
        out[:,[labels.index(k) for k in lr.classes_]]=lr.predict_proba(V)
        return out
    c=probs(clip,cv,extra["clip"] if aug else None)
    pc=np.zeros_like(c)
    pc[:,[labels.index(k) for k in h["labels"]]]=softmax(project(sv,h)@h["W"].T+h["b"],axis=1)
    compact=(pc+c)/2
    if fv is None:return np.array(labels)[compact.argmax(1)]
    return compact,(probs(teacher,fv,extra["dino"] if aug else None)+c)/2

def outer():
    ec=Material(W,True);ef=Material(W)
    cm,_,t=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu");cm.eval()
    geo=data.merge(pd.read_csv(GEOMETRY_PATH),on=["image_id","ann_id"],validate="one_to_one")
    assert keys(data)==keys(geo)
    preds=[]
    for p in spec:
        f=p["fold"]; folder=root/f"fold_{f:02}/evaluation"
        rows=[];crops=[]; info=[]
        tr=data[data.image_id.isin(p["train_image_ids"])]
        with np.load(folder/"base_masks.npz") as masks:
            for line in (folder/"base_detections.jsonl").read_text().splitlines():
                q=json.loads(line);i=q["image_id"]
                if q["source"]!="real" or not q["detections"]:continue
                color,depth,meta=read(i)
                for j,(obj,pk) in enumerate(zip(q["detections"],masks[i],strict=True)):
                    orig=ref[i][j]; assert orig["box"]==obj["box"] and orig["form"]==obj["form"] and orig["score"]==obj["score"]
                    row,crop=extract(i,obj,np.unpackbits(pk,axis=1,count=depth.shape[1]).astype(bool),color,depth,meta,True)
                    rows.append(row);crops.append(crop);info.append((i,j))
        n=len(crops);views=crops+[cv2.flip(a,1) for a in crops]
        s,c=visual(views,ec,cm,t); v=normalize(ef.encode(views))
        row=pd.DataFrame(rows)
        for name in set(RIDGE_X)-set(row):row[name]=np.nan
        d=pd.DataFrame(info,columns=["image_id","j"]);d["fold"]=f
        for aug in [False,True]:
            compact,full=heads(tr,s,c,v,aug)
            for name,prob in (("compact",compact),("full",full)):
                if aug and name=="compact":continue
                for flip in (False,True):
                    m=(prob[:n]+prob[n:])/2 if flip else prob[:n]
                    d[name+("_aug" if aug else "")+("_flip" if flip else "")]=np.array(labels)[m.argmax(1)]
        assert list(d.compact)==[ref[i][j]["material"] for i,j in info]
        row["material"]=row["material_pred"]=d.compact
        base=np.array([ref[i][j]["volume"] for i,j in info])
        r=np.exp(ridge_fit_predict(geo.loc[tr.index],row,list(VOLUME_CATS),"direct",alpha=15))
        solid=row.form.eq("solid")
        d["cb"]=base;d["hybrid"]=np.where(solid,np.sqrt(base*r),base);d["ridge_solid"]=np.where(solid,r,base)
        preds.append(d)
    d=pd.concat(preds,ignore_index=True)
    d.to_csv(W/(P+"outer.csv"),index=False,mode="x")
    matched=[]
    for i,gs in gt.items():
        objs=[dict(**obj,j=j) for j,obj in enumerate(ref[i])]
        for g,obj in zip(gs,metric._match(gs,objs)):
            assert obj is not None
            matched.append(dict(image_id=i,j=obj["j"],true_mat=g["material"],true_vol=g["volume"],true_form=g["form"],take=g["take"],prob=obj["take_p"]))
    e=pd.DataFrame(matched).merge(d,on=["image_id","j"],validate="one_to_one")
    assert len(e)==214 and len(d)==277
    e.to_csv(W/(P+"matched.csv"),index=False,mode="x")
    a=[]
    for col in ["compact","compact_flip","full","full_flip","full_aug","full_aug_flip"]:
        q=e[col].eq(e.true_mat).groupby(e.true_mat).agg(["sum","size","mean"])
        a.append(dict(recipe=col,M=q["mean"].mean(),**{i:f"{int(x['sum'])}/{int(x['size'])}" for i,x in q.iterrows()}))
    b=[]
    for col in ["cb","hybrid","ridge_solid"]:
        val=np.maximum(0,1-abs(np.log(e[col]/e.true_vol))/np.log(3))
        b.append(dict(recipe=col,V=val.mean(),**val.groupby(e.true_form).mean().to_dict()))
    txt=pd.DataFrame(a).to_string(index=False)+"\n"+pd.DataFrame(b).to_string(index=False)
    with (W/(P+"mv.txt")).open("x") as g:g.write(txt+"\n")
    print(txt)

def inner_take():
    ec=Material(W,True)
    cm,_,t=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu");cm.eval()
    cal=[]
    for p in spec:
        f=p["fold"];tr=data[data.image_id.isin(p["train_image_ids"])]
        yolo=YOLO(str(root/f"fold_{f:02}/train/weights/last.pt"))
        rows=[];crops=[]; misses=[]
        real=tr[tr.source.eq("real")]
        for ii in real.itertuples():
            i=ii.image_id;color,depth,meta=read(i)
            ds=detect(yolo,color,"640","cpu")
            gs=gt[i]; assert len(gs)==1
            obj=metric._match(gs,[dict(**o,j=j) for j,(o,_) in enumerate(ds)])[0]
            if obj is None:
                misses.append(dict(image_id=i,object_id=ii.object_id,take=ii.take,prob=np.nan))
                continue
            row,crop=extract(i,obj,ds[obj["j"]][1],color,depth,meta,False)
            rows.append(dict(**row,object_id=ii.object_id,take=ii.take));crops.append(crop)
        ev=pd.DataFrame(rows);s,c=visual(crops,ec,cm,t)
        def upstream(a,b,s,c):
            mat=heads(a,s,c);vol=predict_volume_v2(fit_config(a,("residual",3)),b,mat)
            mass=estimate_mass(b.form,mat,vol,fit_density(a)[0])
            return take_input(b,cols,pd.DataFrame(dict(volume_pred_ml=vol,mass_pred_g=mass),index=b.index))
        preds=[]
        for a,b in grouped_folds(tr,3):
            xt=pd.concat([upstream(aa,bb,student[bb.index],clip[bb.index]) for aa,bb in inner_folds(a)],verify_integrity=True)
            proto,feat=candidates(cols)["CatBoost"]; model=train_candidate(proto,feat,a,xt,"real+synth")
            z=ev[ev.image_id.isin(b.image_id)]
            assert not set(a.object_id)&set(z.object_id) and not set(z.image_id)&set(a.image_id)
            xx=upstream(a,z,s[z.index],c[z.index])
            preds.append(z[["image_id","object_id","take"]].assign(prob=model.predict_proba(xx[feat],thread_count=4)[:,1]))
        q=pd.concat([*preds,pd.DataFrame(misses)],ignore_index=True)
        assert set(q.image_id)==set(real.image_id) and len(real)==len(q)
        cal.append(q.assign(outer_fold=f))
        print("Calibrated fold",f,flush=True)
    pd.concat(cal,ignore_index=True).to_csv(W/(P+"inner.csv"),mode="x",index=False)

def score_take():
    cal=pd.read_csv(W/(P+"inner.csv"));e=pd.read_csv(W/(P+"matched.csv"))
    grid=np.arange(80,100)/100
    def stats(d,c):
        hit=d.prob.notna() & d.prob.ge(c).eq(d["take"])
        h=hit[d["take"].eq(0)]; l=hit[d["take"].eq(1)]
        return dict(heavy=int(h.sum()),light=int(l.sum()),S=(h.mean()+l.mean())/2)
    def choose(d):
        return max(grid,key=lambda c:(round(stats(d,c)["S"],12),stats(d,c)["light"],-c))
    table=pd.DataFrame([dict(cut=c,**stats(e,c)) for c in grid]);folds=[]
    for f,inner in cal.groupby("outer_fold"):
        c=choose(inner); e.loc[e.fold.eq(f),"nested_cut"]=c
        folds.append(dict(fold=f,inner_cut=c,n=len(inner),heavy_ids=inner.loc[inner["take"].eq(0),"object_id"].nunique(),without_s08=choose(inner[inner.object_id.ne("s08")])))
    nested=stats(e,e.nested_cut); dep=choose(e)
    txt=table.to_string(index=False)+"\n"+pd.DataFrame(folds).to_string(index=False)+f"\nnested OOF {nested}; apparent pooled optimum for deployment {dep:.2f}\n"
    print(txt)
    table.to_csv(W/(P+"grid.csv"),index=False,mode="x")
    with (W/(P+"take.txt")).open("x") as g:g.write(txt)

if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with threadpool_limits(4):
        if sys.argv[1]=="outer":
            cache=np.load(W/"astra_visual02_features.npz");assert list(cache["train_ids"])==keys(data)
            extra={k:normalize(cache[k][len(cache["val_ids"]):]) for k in ["dino","clip"]}
            outer()
        else:
            inner_take();score_take()

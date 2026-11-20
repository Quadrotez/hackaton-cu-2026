"""Joint OOF and two local submissions; every output is exclusive-create."""
from astra_joint04_views import *
from volume_hybrid_eval import masked_geometry
from cv_volume_v2 import GEOMETRY_PATH,RIDGE_X,RIDGE_SIZE,inputs
from volume_v2 import VOLUME_CATS,weights_by_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from submission_tools import validate_csv,test_ids
from predict_pipeline import unpack_yolo
from unittest.mock import patch
import infer_detector_wbf
import csv,pickle

WBF="submission_wbf6_final.csv"
SINGLE="submission_grisha_single_final.csv"
HEAD="astra_joint04_heads.pkl"
REP="astra_joint04_report.json"
OOF="astra_joint04_oof.json"

def fit_up(tr,x):
    mats=[LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(a[tr.index],tr.material)
          for a in x[:2]]
    assert all(list(m.classes_)==labels for m in mats)
    cats=[k for k in VOLUME_CATS if k!="surface" or tr.surface.nunique()>1]
    prep=ColumnTransformer([("number",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),RIDGE_X),
                            ("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cats)])
    ridge=make_pipeline(prep,Ridge(alpha=15.))
    ridge.fit(inputs(tr,RIDGE_X,VOLUME_CATS,heldout=False,log_sizes=RIDGE_SIZE),np.log(tr.volume_ml),
              ridge__sample_weight=weights_by_object(tr))
    return dict(mat=mats,cb=fit_config(tr,("residual",3)),ridge=ridge,
                logbounds=np.log([tr.volume_ml.min()/2,tr.volume_ml.max()*2]),density=fit_density(tr)[0])

def up(h,ev,x):
    # x = original DINO, original CLIP, flipped DINO, flipped CLIP.
    p=sum(h["mat"][j%2].predict_proba(a) for j,a in enumerate(x))/4
    mat=np.array(labels)[p.argmax(1)]
    v=predict_volume_v2(h["cb"],ev,mat)
    solid=ev.form.eq("solid")
    if solid.any():
        s=ev.loc[solid].assign(material=mat[solid],material_pred=mat[solid])
        ridge=h["ridge"].predict(inputs(s,RIDGE_X,VOLUME_CATS,heldout=True,log_sizes=RIDGE_SIZE))
        v[solid]=np.exp(np.clip(ridge,*h["logbounds"]))
    mass=estimate_mass(ev.form,mat,v,h["density"])
    return pd.DataFrame(dict(material_pred=mat,volume_pred_ml=v,mass_pred_g=mass),index=ev.index)

def fit(tr,cols,x):
    inner=[]
    for a,b in grouped_folds(tr,3):
        assert not set(a.object_id)&set(b.object_id) and not set(a.image_id)&set(b.image_id)
        inner.append(up(fit_up(a,x),b,[v[b.index] for v in x]))
    oof=pd.concat(inner,verify_integrity=True).loc[tr.index]
    assert not oof.isna().any().any()
    proto,feat=candidates(cols)["CatBoost"]
    model=train_candidate(proto,feat,tr,take_input(tr,cols,oof),"real+synth")
    return dict(up=fit_up(tr,x),model=model,cols=cols,feat=feat,threshold=.99)

def apply(h,ev,x):
    u=up(h["up"],ev,x)
    u["prob"]=h["model"].predict_proba(take_input(ev,h["cols"],u)[h["feat"]],thread_count=4)[:,1]
    assert np.isfinite(u[["volume_pred_ml","mass_pred_g","prob"]]).all().all()
    return u.assign(take=u.prob.ge(.99).astype(int))

class Detections:
    def __init__(self):
        self.rows=[];self.crops=[];self.found={}
        self.cam=json.loads((DATA/"camera.json").read_text())
    def frame(self,tag,i,path,pairs,fold=0,color=None):
        key=(tag,i);assert key not in self.found;self.found[key]=[]
        if not pairs:return
        if color is None:color=cv2.imread(str(next(p for p in [path/"color.png",path/"color.jpg"] if p.is_file())))
        depth=cv2.imread(str(path/"depth.png"),cv2.IMREAD_UNCHANGED)
        meta=json.loads((path/"meta.json").read_text());h,w=depth.shape
        for obj,m in pairs:
            x,y,r,b=obj["box"];box=(x,y,r-x,b-y)
            l,t,rr,bb=bounds(box,w,h,.1);local=m[t:bb,l:rr]
            assert m.shape==(h,w) and local.any() and r>x and b>y
            self.found[key].append(dict(idx=len(self.rows),box=obj["box"],score=obj["score"],form=obj["form"]))
            self.crops.append(masked_crop(color[t:bb,l:rr],local))
            self.rows.append(dict(fold=fold,form=obj["form"],surface=meta["surface"],lighting=meta["lighting"],
                                  raw_volume_ml=raw_volume_ml(depth,m,meta["floor_plane"],self.cam),
                                  **region_features(color,depth,box,local,meta,self.cam,l,t),
                                  **masked_geometry(depth,m,box,meta,self.cam)))
    def archive(self,folder,tag,paths,fold=0):
        with np.load(folder/"base_masks.npz") as masks:
            ds=[json.loads(line) for line in (folder/"base_detections.jsonl").read_text().splitlines()]
            assert {r["image_id"] for r in ds if r["source"]=="real"}==set(paths)
            for r in ds:
                if r["source"]!="real":continue
                i=r["image_id"];h,w=r["shape"];a=masks[i]
                assert a.shape==(len(r["detections"]),h,(w+7)//8)
                self.frame(tag,i,paths[i],[(obj,np.unpackbits(pk,axis=1,count=w).astype(bool))
                                          for obj,pk in zip(r["detections"],a,strict=True)],fold)
    def output(self,tag,u):
        return {i:[dict(box=obj["box"],score=obj["score"],form=obj["form"],material=r.material_pred,
                        volume=float(r.volume_pred_ml),mass=float(r.mass_pred_g),take=int(r["take"]),take_p=float(r.prob))
                   for obj in preds for r in [u.loc[obj["idx"]]]] for (k,i),preds in self.found.items() if k==tag}

def run():
    for name in [WBF,SINGLE,HEAD,REP,OOF]:
        if (W/name).exists():raise FileExistsError(W/name)
    data,cols=read_data(); gg=data.merge(pd.read_csv(GEOMETRY_PATH),on=["image_id","ann_id"],validate="one_to_one").assign(split="train")
    assert keys(gg)==keys(data) and len(data)==892
    flip=np.load(W/"astra_joint04_train_flip.npz");assert flip["ids"].tolist()==keys(data)
    xx=[normalize(embeddings("astra_",keys(data))),normalize(embeddings("astra_clip_",keys(data))),flip["d"],flip["c"]]
    root=W/"detector_oof_v1";manifest=W/"detector_folds_v1.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==json.loads((root/"run.json").read_text())["manifest_sha256"]
    folds=json.loads(manifest.read_text())["folds"]
    frames={i:((DATA/im["file_name"]).parent,aa) for s,i,im,aa in load_frames() if s=="real"}
    gt=ground_truth([(i,path,aa) for i,(path,aa) in frames.items()])
    ds=Detections()
    for p in folds:
        f=p["fold"];ds.archive(root/f"fold_{f:02}/evaluation","oof",{i:frames[i][0] for i in p["val_real_image_ids"]},f)
    ref=json.loads((W/"astra_detector_stack01_predictions.json").read_text())
    assert len(gt)==len(ref)==263 and len(ds.rows)==277
    for i,objs in ref.items():
        assert all(all(a[k]==b[k] for k in ["box","score","form"]) for a,b in zip(ds.found["oof",i],objs,strict=True))
    ids=test_ids();paths={i:DATA/"real/test"/i for i in ids}
    ds.archive(W/"test_wbf_v1","wbf",paths)
    # Reuse the verified checkpoint loader for the single detector only.
    with patch.object(infer_detector_wbf,"WEIGHTS",(W/"grisha_yolo.pt",)):
        models,_=infer_detector_wbf.load_checkpoints()
    for j in range(0,len(ids),8):
        group=ids[j:j+8]; colors=[infer_detector_wbf.test_image(i) for i in group]
        results=models[0].predict(source=colors,imgsz=640,conf=.025,retina_masks=True,max_det=30,device="cpu",batch=len(group),verbose=False)
        for i,color,res in zip(group,colors,results,strict=True):
            ds.frame("grisha",i,paths[i],[(dict(box=d.box,score=d.score,form=d.form),d.mask)
                     for d in unpack_yolo(res,color.shape[:2])],color=color)
    del models
    enc=encoders()
    vx=[*encode(ds.crops,enc),*encode([cv2.flip(c,1) for c in ds.crops],enc)]
    del enc
    ev=pd.DataFrame(ds.rows); allpred=[]
    for p in folds:
        f=p["fold"];tr=gg[gg.image_id.isin(p["train_image_ids"])]; v=ev[ev.fold.eq(f)]
        assert set(tr.object_id)==set(p["train_object_ids"]) and not set(tr.object_id)&set(p["val_object_ids"])
        allpred.append(apply(fit(tr,cols,xx),v,[a[v.index] for a in vx]))
    oof=ds.output("oof",pd.concat(allpred,verify_integrity=True))
    pooled=metric.components(gt,oof)
    assert abs(pooled["D"]-.7614469515003082)<1e-11 and abs(pooled["M"]-.661919)<1e-6
    report=dict(pooled=pooled,delta_vs_68_9814=pooled["Score"]-68.9814,
                folds=[dict(fold=p["fold"],**metric.components({i:gt[i] for i in p["val_real_image_ids"]},
                       {i:oof[i] for i in p["val_real_image_ids"]})) for p in folds])
    pairs=[(g,q) for i,gs in gt.items() for g,q in zip(gs,metric._match(gs,oof[i]))]
    report["take_hits"]={str(t):[sum(q is not None and q["take"]==t for g,q in pairs if g["take"]==t),
                               sum(g["take"]==t for g,q in pairs)] for t in [0,1]}
    with (W/OOF).open("x") as out:json.dump(oof,out,allow_nan=False)
    head=fit(gg,cols,xx)
    with (W/HEAD).open("xb") as out:pickle.dump(head,out)
    v=ev[ev.fold.eq(0)]; final=apply(head,v,[a[v.index] for a in vx]);report["test"]={}
    for tag,name in [("wbf",WBF),("grisha",SINGLE)]:
        pred=ds.output(tag,final)
        with (W/name).open("x",newline="") as file:
            writer=csv.writer(file);writer.writerow(["image_id","prediction"])
            for i in ids:
                objs=pred[i];assert len(objs)<=30
                writer.writerow([i,";".join(" ".join(map(str,[*obj["box"],obj["score"],obj["form"],obj["material"],
                                                         obj["volume"],obj["mass"],obj["take"]])) for obj in objs) or "none"])
        checked=validate_csv(W/name)
        for i in ids:
            for a,b in zip(pred[i],checked[i],strict=True):
                assert a["score"]==b["score"] and tuple(a["box"])==b["box"] and a["form"]==b["form"]
        report["test"][name]=dict(frames=len(checked),objects=sum(map(len,checked.values())),empty=sum(not a for a in checked.values()),
                                  scores_below_025=sum(obj["score"]<.025 for objs in checked.values() for obj in objs))
    with (W/REP).open("x") as out:json.dump(report,out,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with threadpool_limits(4):run()

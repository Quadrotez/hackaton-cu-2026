"""Material + take on frozen real YOLO detections, using Solana's outer folds."""
from astra_yolo_volume_compare import W
import hashlib,json
import numpy as np
import pandas as pd
import cv2,torch
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from predict_pipeline import PredictPipeline,unpack_yolo
from submission_tools import metric,ground_truth
from material_features import bounds,region_features
from material_split import DATA,load_frames
from volume_calibration import raw_volume_ml
from cv_grouped import read_data
from astra_cv import embeddings
from fit_submission_models import fit_compact_head,project
from volume_v2 import fit_config,predict_volume_v2
from take_v2 import candidates,take_input,train_candidate
from take_train import grouped_folds,fit_density,estimate_mass

P="astra_fixed01_"
def save(df,name):
    df.to_csv(W/(P+name+".csv"),index=False,mode="x")

def capture():
    pp=PredictPipeline(bundle_path=W/"submission_models/compact.holdout.pkl")
    det=pp.load_detector()
    crops=[];rows=[]
    for src,i,im,anns in load_frames():
        if src!="real": continue
        path=(DATA/im["file_name"]).parent
        color,depth,meta=pp.read_frame(path)
        pred=det.predict(source=color,conf=.1,imgsz=640,retina_masks=True,max_det=30,device="cpu",verbose=False)[0]
        h,w=depth.shape
        objects=[]
        for d in sorted(unpack_yolo(pred,depth.shape),key=lambda a:-a.score):
            x,r=np.clip(d.box[::2],0,w);y,b=np.clip(d.box[1::2],0,h)
            box=(x,y,r-x,b-y)
            l,t,rr,bb=bounds(box,w,h,.1)
            m=d.mask[t:bb,l:rr]
            if r<=x or b<=y or not m.any():continue
            j=len(rows)
            objects.append(dict(box=(x,y,r,b),score=d.score,j=j))
            crops.append(pp.crop_for_dino(color[t:bb,l:rr],m))
            rows.append(dict(image_id=i,ann_id=-1,form=d.form,surface=meta["surface"],lighting=meta["lighting"],
                             x1=x,y1=y,x2=r,y2=b,score=d.score,mask_sha256=hashlib.sha256(np.packbits(d.mask)).hexdigest(),
                             raw_volume_ml=raw_volume_ml(depth,d.mask,meta["floor_plane"],pp.camera),
                             **region_features(color,depth,box,m,meta,pp.camera,l,t)))
        gt=ground_truth([(i,path,anns)])[i]
        for a,p in zip(anns,metric._match(gt,objects)):
            if p:rows[p["j"]]["ann_id"]=a["id"]
    v=normalize(pp.dino.encode(crops))
    with (W/(P+"z.npy")).open("xb") as f:np.save(f,v)
    save(pd.DataFrame(rows),"det")

def run():
    path=W/(P+"det.csv")
    if not path.exists():capture()
    obs=pd.read_csv(path)
    z=np.load(W/(P+"z.npy"))
    data,cols=read_data()
    data["split"]="train"
    def keys(df):return (df.image_id+"_"+df.ann_id.astype(str)).tolist()
    teach=normalize(embeddings("astra_",keys(data)))
    lookup={k:i for i,k in enumerate(keys(pd.read_csv(W/"cv_oof.csv")))}
    st=np.load(W/"astra_tmp/astra_small_vectors.npy")[[lookup[k] for k in keys(data)]]
    truth=data[data.source.eq("real")][["image_id","ann_id","object_id","material","take"]]
    matched=obs[obs.ann_id.ge(0)].merge(truth,on=["image_id","ann_id"],validate="one_to_one")
    zz=z[obs.ann_id.ge(0)]
    assert keys(matched)==keys(obs[obs.ann_id.ge(0)])
    def upstream(train,ev,vec):
        hd=fit_compact_head(train,teach,st)
        a,b=project(st[train.index],hd),project(vec,hd)
        base=hd["labels"][(b@hd["W"].T+hd["b"]).argmax(1)]
        trans=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
        def stack(d,f):return np.column_stack([d,normalize(np.clip(f,-5,5))])
        model=LogisticRegression(C=1,class_weight="balanced",max_iter=1000)
        model.fit(stack(a,trans.fit_transform(train[cols])),train.material)
        fusion=model.predict(stack(b,trans.transform(ev[cols])))
        volume=predict_volume_v2(fit_config(train,("residual",3)),ev,base)
        density=fit_density(train)[0]
        result={}
        for k,m in [("compact",base),("rgbd",fusion)]:
            up=pd.DataFrame(dict(volume_pred_ml=volume,mass_pred_g=estimate_mass(ev.form,m,volume,density)),index=ev.index)
            result[k]=(m,take_input(ev,cols,up))
        return result,volume
    result=[]
    for fold in json.loads((W/"cv_folds.json").read_text()):
        va=data.object_id.isin(fold["val_object_ids"])
        tr=data[data.object_id.isin(fold["train_object_ids"]) & ~data.image_id.isin(data.loc[va,"image_id"])]
        assert not set(tr.object_id)&set(data[va].object_id)
        ev=matched[matched.object_id.isin(fold["val_object_ids"])]
        out=ev[["image_id","ann_id"]].copy()
        up,out["fixed_volume_ml"]=upstream(tr,ev,zz[ev.index])
        cross={"compact":[],"rgbd":[]}
        for fit,hold in grouped_folds(tr,3):
            inner,_=upstream(fit,hold,st[hold.index])
            for k in cross:cross[k].append(inner[k][1])
        proto,features=candidates(cols)["CatBoost"]
        for k in cross:
            m,xx=up[k]
            model=train_candidate(proto,features,tr,pd.concat(cross[k],verify_integrity=True).loc[tr.index],"real+synth")
            out[k]=m
            out[k+"_p"]=model.predict_proba(xx[features],thread_count=4)[:,1]
            out[k+"_take"]=(out[k+"_p"]>=.90).astype(int)
        full=truth[truth.object_id.isin(fold["val_object_ids"])].merge(out,on=["image_id","ann_id"],how="left",validate="one_to_one")
        result.append(full.assign(fold=fold["fold"]))
    df=pd.concat(result,ignore_index=True)
    assert len(df)==214 and len(set(keys(df)))==214
    oracle=pd.read_csv(W/"astra_distilled_oof.csv").query("source=='real'")
    df=df.merge(oracle[["image_id","ann_id","fold","material_pred"]].rename(columns={"material_pred":"compact_gt"}),on=["image_id","ann_id","fold"],validate="one_to_one")
    save(df,"oof")
    report(df,len(obs),len(matched))

def report(df,n_det,n_matched):
    for k in ("compact_gt","compact","rgbd"):df[k+"_hit"]=df[k].eq(df.material)
    for k in ("compact","rgbd"):
        df[k+"_take_hit"]=df[k+"_take"].eq(df["take"])
        assert df[k+"_take_hit"].sum()==sum(t==p for t,p in zip(df["take"],df[k+"_take"]))
    def table(group,columns):
        return df.groupby(group)[columns].agg(["sum","size","mean"])
    mats=table("material",["compact_gt_hit","compact_hit","rgbd_hit"])
    by_obj=table(["material","object_id"],["compact_gt_hit","compact_hit","rgbd_hit"])
    take=table("take",["compact_take_hit","rgbd_take_hit"])
    to=table(["take","object_id"],["compact_take_hit","rgbd_take_hit"])
    sections=[f"YOLO best.pt SHA256 {hashlib.sha256((W/'weights/best.pt').read_bytes()).hexdigest()}; conf=.10, 640, retina masks, max_det=30, default NMS, no TTA. {n_det} detections; {n_matched}/214 matches over 263 real frames. Misses count as errors.",
              "cv_folds.json: 5 outer object-disjoint folds; train excludes overlapping frames. Detector fixed, trained on primary split (NOT five-fold OOF for detector).",
              "Train uses original annotated masks for both heads; evaluated on identical cached YOLO crops/features. CLS projection to DINO teacher trained only in fold. RGBD: balanced logistic C=1 on concatenated compact features and unit-normalized standardized 60 RGB-D features (train-only median imputation, clip +/-5).",
              "Volume frozen CB-residual-d3 recipe: fit only on train, uses compact material, identical numeric prediction per matched detection for both arms. CatBoost take refit with 3 inner object-disjoint folds for upstream training features; fixed threshold 0.90 from existing stack, no threshold search.",
              "Material (hits/count/recall):\n"+mats.to_string(),"Pooled real M:\n"+mats.xs("mean",axis=1,level=1).mean().to_string(),
              "Take: 0 heavy; 1 light:\n"+take.to_string(),"Pooled S:\n"+take.xs("mean",axis=1,level=1).mean().to_string(),
              "Per physical object material recall:\n"+by_obj.to_string(),"Per physical object take recall:\n"+to.to_string()]
    with (W/(P+"report_verified.txt")).open("x") as f:f.write("\n\n".join(sections)+"\n")
    print("\n\n".join(sections[:8]))
if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with threadpool_limits(4):run()

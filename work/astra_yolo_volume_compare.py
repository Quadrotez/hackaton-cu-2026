"""Primary YOLO holdout; no test frames, no overwrite."""
import os
from pathlib import Path
W=Path(__file__).resolve().parent
os.environ.update(YOLO_CONFIG_DIR=str(W/"astra_yolo_config"),YOLO_OFFLINE="true",HF_HUB_OFFLINE="1")
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="4"
import types
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from predict_pipeline import PredictPipeline,unpack_yolo
from submission_tools import baseline_holdout,ground_truth,metric
from material_features import region_features,bounds
from material_split import DATA,load_frames
from volume_calibration import raw_volume_ml
from cv_volume_v2 import geometry_one,ridge_fit_predict,RIDGE_X
from volume_v2 import VOLUME_CATS,fit_config,predict_volume_v2
from fit_submission_models import fit_compact_head,project
from take_train import grouped_folds,estimate_mass,predict_take
from take_v2 import take_input
from astra_cv import embeddings


def run():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pp=PredictPipeline(bundle_path=W/"submission_models/compact.holdout.pkl")
    det=pp.load_detector()
    val=baseline_holdout()
    frames=[(i,(DATA/im["file_name"]).parent,aa) for src,i,im,aa in load_frames() if src=="real" and aa]
    gt=ground_truth(frames)
    vf={i for i,_,_ in val}
    vo={a["attributes"]["object_id"] for _,_,aa in val for a in aa}
    data=pd.read_csv(W/"features_all_v2.csv").merge(pd.read_csv(W/"cv_volume_geometry_v2.csv"),on=["image_id","ann_id"],validate="one_to_one")
    def keys(df): return (df.image_id+"_"+df.ann_id.astype(str)).tolist()
    teacher=normalize(embeddings("astra_",keys(data)))
    ix={k:i for i,k in enumerate(keys(pd.read_csv(W/"cv_oof.csv")))}
    student=np.load(W/"astra_tmp/astra_small_vectors.npy")[[ix[k] for k in keys(data)]]
    excluded=set(data.loc[data.object_id.isin(vo),"image_id"])
    train=data.loc[~data.image_id.isin(excluded)].assign(split="train")
    assert set(train.object_id).isdisjoint(vo) and set(train.image_id).isdisjoint(vf)
    rows=[]; crops=[]; found={}
    for iid,path,aa in frames:
        color,depth,meta=pp.read_frame(path)
        res=det.predict(source=color,conf=.1,imgsz=640,retina_masks=True,max_det=30,device="cpu",verbose=False)[0]
        found[iid]=[]
        H,B=depth.shape
        for d in sorted(unpack_yolo(res,depth.shape),key=lambda d:-d.score):
            x,y,r,b=d.box
            x,r=np.clip([x,r],0,B); y,b=np.clip([y,b],0,H)
            if r<=x or b<=y: continue
            box=(x,y,r-x,b-y)
            l,t,rr,bb=bounds(box,B,H,.1)
            local=d.mask[t:bb,l:rr]
            if not local.any(): continue
            # Reuse exactly the same geometric calculation; supply the original raster
            # directly, preserving holes and disconnected components.
            def raster(_,width,height,left,top):
                return d.mask[top:top+height,left:left+width]
            geo=types.FunctionType(geometry_one.__code__,dict(geometry_one.__globals__,polygon_mask=raster))
            features=geo(depth,dict(bbox=box,segmentation=None),meta,pp.camera)
            idx=len(rows)
            found[iid].append(dict(box=(x,y,r,b),score=d.score,form=d.form,idx=idx))
            rows.append(dict(image_id=iid,form=d.form,surface=meta["surface"],lighting=meta["lighting"],
                             raw_volume_ml=raw_volume_ml(depth,d.mask,meta["floor_plane"],pp.camera),
                             **region_features(color,depth,box,local,meta,pp.camera,l,t),**features))
            crops.append(pp.crop_for_dino(color[t:bb,l:rr],local))
    z=normalize(pp.dino.encode(crops))
    rows=pd.DataFrame(rows)
    match=[]
    for iid,truth in gt.items():
        for g,p in zip(truth,metric._match(truth,found[iid])):
            if p: match.append(dict(idx=p["idx"],truth=g["volume"],true_form=g["form"]))
    matches=pd.DataFrame(match).set_index("idx")
    def preds(fit,rows_,head,volume):
        rows_=rows_.copy()
        rows_["material_pred"]=head["labels"][(project(z[rows_.index],head)@head["W"].T+head["b"]).argmax(1)]
        rows_["material"]=rows_.material_pred
        base=predict_volume_v2(volume,rows_,rows_.material_pred)
        ridge=ridge_fit_predict(fit,rows_,list(VOLUME_CATS),"direct")
        return rows_.assign(CB=base,RidgeBlend=np.sqrt(base*np.exp(ridge)))
    calibration=[]
    for fit,hold in grouped_folds(train,3):
        h=rows[rows.image_id.isin(hold.image_id)]
        if len(h):
            calibration.append(preds(fit,h,fit_compact_head(fit,teacher,student),fit_config(fit,("residual",3))))
    cal=pd.concat(calibration,verify_integrity=True).join(matches,how="inner")
    assert set(cal.image_id).isdisjoint(vf)
    valpred=preds(train,rows[rows.image_id.isin(vf)],pp.dino.head,pp.volume)
    gtval=ground_truth(val)
    records=[]; groups=[]; factors=[]
    for method in ["CB","RidgeBlend"]:
        fac=np.exp(np.log(cal.truth/cal[method]).groupby(cal.form).median())
        factors.append(dict(method=method,**fac.to_dict()))
        for calibrated in [False,True]:
            vol=valpred[method]*(valpred.form.map(fac) if calibrated else 1)
            up=pd.DataFrame(dict(volume_pred_ml=vol,material_pred=valpred.material_pred,
                                 mass_pred_g=estimate_mass(valpred.form,valpred.material_pred,vol,pp.take["density"])),index=valpred.index)
            up["take"]=predict_take(pp.take,take_input(valpred,pp.take["steps3"],up))
            out={i:[dict(**p,volume=vol.at[p["idx"]],material=up.at[p["idx"],"material_pred"],
                         take=up.at[p["idx"],"take"]) for p in found[i]] for i in gtval}
            records.append(dict(method=method,calibrated=calibrated,**metric.components(gtval,out)))
            for form in ["bulk","solid"]:
                pairs=[(p["volume"],g["volume"]) for i,gg in gtval.items() for g,p in zip(gg,metric._match(gg,out[i])) if g["form"]==form and p]
                n=sum(g["form"]==form for gg in gtval.values() for g in gg)
                pv,tv=np.array(pairs).T
                groups.append(dict(method=method,calibrated=calibrated,form=form,matched=len(pairs),gt=n,
                                   median_ratio=np.exp(np.median(np.log(pv/tv))),mean_error_ml=np.mean(pv-tv),
                                   V=np.maximum(0,1-abs(np.log(pv/tv))/np.log(3)).sum()/n))
    report="\n\n".join([f"Primary YOLO 10 epochs, weights/best.pt, conf=.10, retina masks 640. {len(gtval)} real holdout frames. "
                         f"Compact holdout DINO, take=.90; mass and take recomputed for each variant. "
                         f"Geometry Ridge alpha=15; log-space equal-weight blend. Train {len(train)} GT annotation rows. "
                         f"Calibration factors: median log(true/pred) by predicted form on 3-fold cross-fitted real train YOLO matches "
                         f"{cal.form.value_counts().to_dict()}. Detector itself fitted on outer train. "
                         "Validation bias by true form on matched detections; ratio=pred/true, V counts misses as zero.",
                         *[pd.DataFrame(x).to_string(index=False) for x in (records,groups,factors)]])
    print(report)
    with (W/"astra_yolo_volume_comparison_01.txt").open("x") as f: f.write(report+"\n")

if __name__=="__main__":
    with threadpool_limits(4): run()

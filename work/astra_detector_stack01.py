"""Frozen detector exports -> grouped downstream fits -> official pooled real OOF."""
from astra_visual02 import W,keys,labels,open_clip
import hashlib,json
import numpy as np,pandas as pd,cv2,torch
from PIL import Image
from scipy.special import softmax
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from cv_grouped import read_data
from material_split import DATA,load_frames
from material_features import region_features,bounds
from volume_calibration import raw_volume_ml
from astra_deploy import Material,masked_crop
from astra_cv import embeddings
from fit_submission_models import fit_compact_head,project
from volume_v2 import fit_config,predict_volume_v2
from take_train import fit_density,estimate_mass,grouped_folds
from take_v2 import take_input,candidates,train_candidate
from submission_tools import metric,ground_truth

def run():
    root=W/"detector_oof_v1"; manifest=W/"detector_folds_v1.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==json.loads((root/"run.json").read_text())["manifest_sha256"]
    specs=json.loads(manifest.read_text())["folds"]
    data,cols=read_data(); data["split"]="train"
    teacher=normalize(embeddings("astra_",keys(data)))
    clip=normalize(embeddings("astra_clip_",keys(data)))
    ix={k:i for i,k in enumerate(keys(pd.read_csv(W/"cv_oof.csv")))}
    student=np.load(W/"astra_tmp/astra_small_vectors.npy")[[ix[k] for k in keys(data)]]
    frames={i:((DATA/im["file_name"]).parent,aa) for src,i,im,aa in load_frames() if src=="real"}
    gt=ground_truth([(i,path,aa) for i,(path,aa) in frames.items()])
    pred={}; crops=[];rows=[];camera=json.loads((DATA/"camera.json").read_text())
    for p in specs:
        f=p["fold"]; folder=root/f"fold_{f:02}/evaluation"
        ds=[json.loads(line) for line in (folder/"base_detections.jsonl").read_text().splitlines()]
        ds=[r for r in ds if r["source"]=="real"]
        assert {r["image_id"] for r in ds}==set(p["val_real_image_ids"])
        with np.load(folder/"base_masks.npz") as all_masks:
            for r in ds:
                i=r["image_id"];assert i not in pred
                pred[i]=[];path,_=frames[i]; h,w=r["shape"]
                packed=all_masks[i];assert packed.shape==(len(r["detections"]),h,(w+7)//8)
                if not len(packed):continue
                color=cv2.imread(str(path/"color.png"))
                depth=cv2.imread(str(path/"depth.png"),cv2.IMREAD_UNCHANGED)
                meta=json.loads((path/"meta.json").read_text())
                for obj,pk in zip(r["detections"],packed,strict=True):
                    mask=np.unpackbits(pk,axis=1,count=w).astype(bool)
                    x,y,rr,bb=obj["box"];box=(x,y,rr-x,bb-y)
                    l,t,right,b=bounds(box,w,h,.1); local=mask[t:b,l:right]
                    assert local.any()
                    j=len(rows);pred[i].append(dict(**obj,idx=j))
                    crops.append(masked_crop(color[t:b,l:right],local))
                    rows.append(dict(fold=f,image_id=i,form=obj["form"],lighting=meta["lighting"],surface=meta["surface"],
                                     raw_volume_ml=raw_volume_ml(depth,mask,meta["floor_plane"],camera),
                                     **region_features(color,depth,box,local,meta,camera,l,t)))
    assert set(gt)==set(pred) and len(gt)==263
    rows=pd.DataFrame(rows)
    stud=normalize(Material(W,True).encode(crops))
    model,_,t=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu")
    model.eval();vec=[]
    with torch.inference_mode():
        for j in range(0,len(crops),8):
            batch=torch.stack([t(Image.fromarray(c[:,:,::-1])) for c in crops[j:j+8]])
            vec.append(model.encode_image(batch).numpy())
    cvec=normalize(np.concatenate(vec)); del model
    def upstream(tr,ev,sv,cv):
        head=fit_compact_head(tr,teacher,student)
        lr=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(clip[tr.index],tr.material)
        assert list(head["labels"])==list(lr.classes_)==labels
        pc=softmax(project(sv,head)@head["W"].T+head["b"],axis=1)
        m=lr.classes_[(pc+lr.predict_proba(cv)).argmax(1)]
        v=predict_volume_v2(fit_config(tr,("residual",3)),ev,m)
        mass=estimate_mass(ev.form,m,v,fit_density(tr)[0])
        x=take_input(ev,cols,pd.DataFrame(dict(volume_pred_ml=v,mass_pred_g=mass),index=ev.index))
        return m,v,mass,x
    details=[]
    for p in specs:
        f=p["fold"];tr=data[data.image_id.isin(p["train_image_ids"])]
        assert set(tr.object_id)==set(p["train_object_ids"]) and not set(tr.object_id)&set(p["val_object_ids"])
        assert not set(tr.image_id)&set(p["val_synth_image_ids"]+p["val_real_image_ids"]+p["excluded_mixed_synth_image_ids"])
        inner=pd.concat([upstream(a,b,student[b.index],clip[b.index])[3] for a,b in grouped_folds(tr,3)],verify_integrity=True)
        proto,features=candidates(cols)["CatBoost"]
        take=train_candidate(proto,features,tr,inner.loc[tr.index],"real+synth")
        ev=rows[rows.fold.eq(f)]
        m,v,mass,x=upstream(tr,ev,stud[ev.index],cvec[ev.index])
        prob=take.predict_proba(x[features],thread_count=4)[:,1]
        for i in p["val_real_image_ids"]:
            for obj in pred[i]:
                k=ev.index.get_loc(obj.pop("idx"))
                obj.update(material=str(m[k]),volume=float(v[k]),mass=float(mass[k]),take=int(prob[k]>=.99),take_p=float(prob[k]))
        frame_gt={i:gt[i] for i in p["val_real_image_ids"]}
        details.append(dict(fold=f,frames=len(frame_gt),**metric.components(frame_gt,{i:pred[i] for i in frame_gt})))
    pooled=metric.components(gt,pred)
    base=json.loads((root/"pooled_real_ablation.json").read_text())["variants"]["yolo"]
    assert np.isclose(pooled["D"],base["D"]) and sum(map(len,pred.values()))==base["predictions"]
    pairs=[(g,q) for i,gg in gt.items() for g,q in zip(gg,metric._match(gg,pred[i]))]
    def hits(key,val,field):
        sub=[(g,q) for g,q in pairs if g[key]==val]
        return [sum(q is not None and g[field]==q[field] for g,q in sub),len(sub)]
    report=dict(pooled=pooled,folds=details,real_frames=len(gt),gt_objects=len(pairs),detections=base["predictions"],
                matched=sum(q is not None for _,q in pairs),material={s:hits("material",s,"material") for s in labels},
                take={str(s):hits("take",s,"take") for s in (0,1)},threshold=.99,
                recipe="Base exports unchanged; compact DINO + CLIP probabilities 50:50 -> CB-residual-d3 -> train-only density -> object-weighted CatBoost take, training inputs cross-fitted within outer train. Official score.components on pooled real predictions.")
    for name,d in [("predictions",pred),("report",report)]:
        with (W/f"astra_detector_stack01_{name}.json").open("x") as out:json.dump(d,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with threadpool_limits(4):run()

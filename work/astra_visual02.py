from astra_yolo_volume_compare import W
import hashlib,json,sys
import cv2,numpy as np,pandas as pd,torch
from PIL import Image
from scipy.special import softmax
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from predict_pipeline import PredictPipeline,unpack_yolo
from material_features import bounds
from material_split import load_frames,DATA
from astra_cv import embeddings
from astra_deploy import Material
from astra_clip import open_clip
from astra_train import labels

P="astra_visual02_"
def put(name,**values):
    with (W/(P+name+".npz")).open("xb") as f:np.savez(f,**values)
def keys(d):return (d.image_id+"_"+d.ann_id.astype(str)).tolist()
def make():
    gt=pd.read_csv(W/"features_all_v2.csv")
    ids=keys(gt)
    d=pd.read_csv(W/"astra_fixed01_det.csv")
    # Replay only to recover raster crops (original cache recorded hashes, not pixels).
    pp=PredictPipeline(bundle_path=W/"submission_models/compact.holdout.pkl")
    yo=pp.load_detector()
    crops=[]; verified=[]
    for src,i,im,aa in load_frames():
        if src!="real":continue
        color=cv2.imread(str(DATA/im["file_name"]))
        h,w=color.shape[:2]
        det=yo.predict(source=color,conf=.1,imgsz=640,retina_masks=True,max_det=30,device="cpu",verbose=False)[0]
        for obj in sorted(unpack_yolo(det,(h,w)),key=lambda a:-a.score):
            x,r=np.clip(obj.box[::2],0,w); y,b=np.clip(obj.box[1::2],0,h)
            l,t,rr,bb=bounds((x,y,r-x,b-y),w,h,.1)
            mask=obj.mask[t:bb,l:rr]
            if r<=x or b<=y or not mask.any():continue
            row=d.iloc[len(verified)]
            assert row.image_id==i and hashlib.sha256(np.packbits(obj.mask)).hexdigest()==row.mask_sha256
            np.testing.assert_allclose([x,y,r,b],row[["x1","y1","x2","y2"]].to_numpy(float),rtol=0,atol=1e-8)
            verified.append(i)
            if row.ann_id>=0:crops.append(pp.crop_for_dino(color[t:bb,l:rr],mask))
    assert len(verified)==len(d)
    paths={p.stem:p for p in (W/"crops").glob("*/*/*.png")}
    paths.update({p.stem[6:]:p for p in (W/"astra_tmp").glob("astra_*.png")})
    aug=[]
    for i,k in enumerate(ids):
        im=cv2.imread(str(paths[k]))
        rng=np.random.default_rng(int(hashlib.sha256(k.encode()).hexdigest()[:8],16))
        # One extra deterministic view per original; split is by object, never by view.
        im=cv2.flip(im,1)
        im=cv2.warpAffine(im,cv2.getRotationMatrix2D((63.5,63.5),rng.uniform(-25,25),1),(128,128))
        aug.append(np.clip(im.astype(float)*rng.uniform(.82,1.18,(1,1,3)),0,255).astype("uint8"))
    images=crops+aug
    del pp,yo
    base=Material(W)
    vec=base.encode(images)
    del base
    model,_,trans=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu")
    model.eval()
    c=[]
    with torch.inference_mode():
        for i in range(0,len(images),8):
            x=torch.stack([trans(Image.fromarray(im[:,:,::-1])) for im in images[i:i+8]])
            c.append(model.encode_image(x).numpy())
    put("features",dino=vec,clip=np.concatenate(c),val_ids=keys(d[d.ann_id>=0]),train_ids=ids)

def run():
    if not (W/(P+"features.npz")).exists():make()
    gt=pd.read_csv(W/"features_all_v2.csv")
    o=pd.read_csv(W/"astra_fixed01_oof.csv")
    data=np.load(W/(P+"features.npz"))
    assert list(data["train_ids"])==keys(gt)
    ids=list(data["val_ids"]); idx=[ids.index(k) for k in keys(o)]
    n=len(ids)
    x={k:(normalize(embeddings(prefix,keys(gt))),normalize(data[k][n:]),normalize(data[k][:n][idx])) for k,prefix in [("dino","astra_"),("clip","astra_clip_")]}
    out=o[["image_id","ann_id","object_id","fold","material","compact"]].copy()
    for p in json.loads((W/"cv_folds.json").read_text()):
        hold=gt.object_id.isin(p["val_object_ids"])
        tr=gt.object_id.isin(p["train_object_ids"]) & ~gt.image_id.isin(gt.loc[hold,"image_id"])
        va=o.fold.eq(p["fold"])
        for aug in (False,True):
            prob=[]
            for k,(orig,extra,test) in x.items():
                xx=orig[tr];y=gt.loc[tr,"material"].to_numpy()
                if aug:xx=np.vstack([xx,extra[tr]]);y=np.tile(y,2)
                clf=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(xx,y,sample_weight=np.full(len(y),.5 if aug else 1))
                assert list(clf.classes_)==labels
                pr=clf.predict_proba(test[va]);prob.append(pr)
                out.loc[va,k+("_aug" if aug else "")]=clf.classes_[pr.argmax(1)]
            out.loc[va,"blend"+("_aug" if aug else "")]=np.array(labels)[np.mean(prob,axis=0).argmax(1)]
    out.to_csv(W/(P+"oof.csv"),index=False,mode="x")
    report={}
    for k in out.columns[5:]:
        counts=out[k].eq(out.material).groupby(out.material).agg(["sum","size","mean"])
        report[k]={"M":counts["mean"].mean(),**{a:f"{r['sum']:.0f}/{r['size']:.0f}" for a,r in counts.iterrows()}}
    txt=pd.DataFrame(report).T.to_string()
    print(txt)
    with (W/(P+"report.txt")).open("x") as f:f.write("Solana folds; same 229 detections verified by bbox + mask SHA; 214 real GT. Original compact vs full ViT-B/16 DINO, ViT-B/32 CLIP and equal-weight posterior blend. C=1 fixed; augmentation halves sample weights for both views; no TTA.\n"+txt+"\n")

if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    with threadpool_limits(4):run()

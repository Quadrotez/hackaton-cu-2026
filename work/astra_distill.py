from astra_train import W,fit,metric,threadpool_limits
from astra_deploy import Material,norm
from astra_cv import embeddings
from sklearn.linear_model import Ridge,LogisticRegression
import cv2,json,numpy as np,pandas as pd,torch


def run():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    ref=pd.read_csv(W/"cv_oof.csv")
    keys=(ref.image_id+"_"+ref.ann_id.astype(str)).tolist()
    y,g,src,im=[ref[k].to_numpy(dtype=str) for k in ("material","object_id","source","image_id")]
    teacher=embeddings("astra_",keys)
    def export(model):
        return dict(W=model.coef_,b=model.intercept_,labels=model.classes_.astype(str))
    model,h=fit(teacher,y,g,src,images=im)
    np.savez(W/"astra_dino_head.npz",**export(model[-1]))
    teacher=norm(teacher)
    pack={}
    for k,t in torch.load(W/"astra_tmp/astra_small.pth",map_location="cpu",weights_only=True).items():
        z=t.numpy()
        if z.ndim>=2 and z.size%32==0:
            a=z.reshape(-1,32)
            s=np.maximum(abs(a).max(1)/7,1e-7).astype("float16")
            q=(np.rint(a/s[:,None]).clip(-7,7)+8).astype("uint8").reshape(-1,2)
            pack[k]=q[:,0]+16*q[:,1]; pack[k+".scale"]=s
        else:
            pack[k]=z.astype("float16")
    np.savez(W/"astra_distilled.npz",**pack)
    enc=Material(W,compact=True)
    paths={p.stem:p for p in (W/"crops").glob("*/*/*.png")}
    paths.update({p.stem[6:]:p for p in (W/"astra_tmp").glob("astra_*.png")})
    x=norm(enc.encode([cv2.imread(str(paths[k])) for k in keys]))
    np.save(W/"astra_tmp/astra_small_vectors.npy",x)
    def train(tr,va):
        # Map quantized student embeddings to the frozen unit-norm teacher targets.
        r=Ridge(alpha=.5).fit(x[tr],teacher[tr])
        parts=dict(map=r.coef_.T.copy(),offset=r.intercept_,mix=np.float32(.5))
        def mapped(idx):
            return norm(np.column_stack([.5*x[idx],norm(r.predict(x[idx]))]))
        model=LogisticRegression(class_weight="balanced",C=1,max_iter=1000).fit(mapped(tr),y[tr])
        return model.predict(mapped(va)),dict(**export(model),**parts)
    pred=y.copy()
    for p in json.loads((W/"cv_folds.json").read_text()):
        va=np.isin(g,p["val_object_ids"])
        tr=np.isin(g,p["train_object_ids"]) & ~np.isin(im,im[va])
        assert np.array_equal(va,ref.fold==p["fold"])
        pred[va],_=train(tr,va)
    real=src=="real"
    oof=ref[["source","image_id","ann_id","object_id","material","material_pred","fold"]].copy()
    oof["material_pred"]=pred
    oof.to_csv(W/"astra_distilled_oof.csv",index=False)
    print(metric(y[real],pred[real]))
    expected,head=train(slice(None),slice(None))
    np.savez(W/"astra_distilled.npz",**pack,**head)
    assert (W/"astra_distilled.npz").stat().st_size<20_000_000
    crops=[cv2.imread(str(paths[k])) for k in keys[:16]]
    assert np.array_equal(Material(W,True).predict(crops),expected[:16])
    assert np.array_equal(Material(W).predict(crops),model.predict(teacher[:16]))

if __name__=="__main__":
    with threadpool_limits(4): run()

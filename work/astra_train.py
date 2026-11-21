import os
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[var]="4"
import json
import csv
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

W=Path(__file__).resolve().parent
labels=["glass","metal","mineral","paper","plastic","wood"]


def metric(y,p):
    cm=confusion_matrix(y,p,labels=labels)
    n=cm.sum(1)
    return float((cm.diagonal()[n>0]/n[n>0]).mean()), dict(zip(labels,zip(cm.diagonal().tolist(),n.tolist())))


def fit(X,y,g,source,prior=None,images=None):
    folds=list(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(X,y,g))
    if images is not None:
        folds=[(tr[~np.isin(images[tr],images[va])],va) for tr,va in folds]
        assert all(not set(images[t]) & set(images[v]) and not set(g[t]) & set(g[v]) for t,v in folds)
    def train(h,idx):
        C,alpha=h
        model=make_pipeline(Normalizer(),LogisticRegression(C=C,class_weight="balanced",max_iter=1000))
        model.fit(X[idx],y[idx])
        if prior is not None:
            head=model.steps[-1][1]
            assert list(head.classes_)==labels
            scale=np.linalg.norm(head.coef_,axis=1).mean()
            head.coef_=(1-alpha)*head.coef_ + alpha*scale*prior
            head.intercept_*=1-alpha
        return model
    def score(C):
        pred=y.copy()
        for tr,va in folds:
            pred[va]=train(C,tr).predict(X[va])
        real=source=="real"
        return metric(y[real],pred[real])[0]
    h=max([(c,a) for c in (.1,1,10,100) for a in ((0,.5,.85) if prior is not None else (0,))],key=score)
    return train(h,np.arange(len(y))),h


def run(prefix="astra_"):
    prior=np.load(W/"astra_clip_text.npy") if prefix=="astra_clip_prior_" else None
    data_prefix=prefix.replace("clip_prior_","clip_")
    ids=sum([json.loads((W/(p+"ids.json")).read_text()) for p in (data_prefix,data_prefix+"extra_")],[])
    X=np.concatenate([np.load(W/(p+"embeddings.npy")) for p in (data_prefix,data_prefix+"extra_")])
    with (W/"features_all_v2.csv").open() as f:
        rows={f"{r['image_id']}_{r['ann_id']}":r for r in csv.DictReader(f)}
    assert len(X)==len(set(ids))==len(rows)
    y,g,im,source=[np.array([rows[k][col] for k in ids]) for col in ("material","object_id","image_id","source")]
    res={}
    for name,file in (("primary","split.json"),("v2","split_v2.json")):
        plan=json.loads((W/file).read_text())
        def mask(part):
            d=plan[part]
            m=np.isin(g,d["object_ids"]) & np.isin(im,d["image_ids"])
            assert all(sum(m & (source==s))==v["annotations"] for s,v in d["counts_by_source"].items())
            return m
        tr,va=mask("train"),mask("val")
        model,(C,a)=fit(X[tr],y[tr],g[tr],source[tr],prior)
        res[name]={"C":C,"alpha":a}
        for scope,m in (("real",va&(source=="real")),("all",va)):
            res[name][scope]=metric(y[m],model.predict(X[m]))
        metal=va & (source=="real") & (y=="metal")
        if metal.any():
            from collections import Counter
            res[name]["metal_predictions"]=Counter(model.predict(X[metal]).tolist())
    model,(C,a)=fit(X,y,g,source,prior)
    res["refit_alpha"]=a
    res["refit_C"]=C
    with (W/(prefix+"candidate.pkl")).open("wb") as f:
        pickle.dump(model,f)
    (W/(prefix+"metrics.json")).write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps(res))


if __name__=="__main__":
    import sys
    with threadpool_limits(4):
        run(*sys.argv[1:])

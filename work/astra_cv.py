from astra_train import W, fit, labels, metric, threadpool_limits
from cv_grouped import read_data, fit_material, material_pred
import json
import numpy as np
import pandas as pd


def embeddings(prefix, keys):
    parts=[prefix,prefix+"extra_"]
    ids=sum([json.loads((W/(p+"ids.json")).read_text()) for p in parts],[])
    x=np.concatenate([np.load(W/(p+"embeddings.npy")) for p in parts])
    assert len(set(ids))==len(x)==len(keys) and set(ids)==set(keys) and np.isfinite(x).all()
    ix={k:i for i,k in enumerate(ids)}
    return x[[ix[k] for k in keys]]


def main():
    ref=pd.read_csv(W/"cv_oof.csv")
    cols=["source","image_id","ann_id","object_id","material","material_pred","fold"]
    data,features=read_data()
    key=["image_id","ann_id"]
    data=data.set_index(key).loc[pd.MultiIndex.from_frame(ref[key])].reset_index()
    assert len(data)==892 and not data.duplicated(key).any()
    assert data[cols[:5]].equals(ref[cols[:5]])
    keys=(data.image_id+"_"+data.ann_id.astype(str)).tolist()
    xd=embeddings("astra_",keys)
    xc=embeddings("astra_clip_",keys)
    text=np.load(W/"astra_clip_text.npy")
    y,g,src,im=[data[k].to_numpy(dtype=str) for k in ("material","object_id","source","image_id")]
    methods=["baseline_cb","dino","clip","clip_text"]
    results={k:np.full(len(y),"",dtype="U16") for k in methods}
    config=[]
    folds=json.loads((W/"cv_folds.json").read_text())
    assert sorted(p["fold"] for p in folds)==list(range(1,6))
    for p in folds:
        f=p["fold"]
        va=np.isin(g,p["val_object_ids"])
        tr=np.isin(g,p["train_object_ids"]) & ~np.isin(im,im[va])
        assert np.array_equal(va,ref.fold==f) and set(g[tr])==set(p["train_object_ids"])
        assert not set(g[va]) & set(g[tr]) and not set(im[va]) & set(im[tr])
        cb=fit_material(data[tr],features,"CB-object-weighted")
        results["baseline_cb"][va]=material_pred(cb,data[va],features)
        line=f"fold={f} train={tr.sum()} val={va.sum()}"
        for name,x,prior in [("dino",xd,None),("clip",xc,None),("clip_text",xc,text)]:
            model,(c,a)=fit(x[tr],y[tr],g[tr],src[tr],prior,images=im[tr])
            results[name][va]=model.predict(x[va])
            line+=f" | {name}: C={c:g}, alpha={a:g}"
        config.append(line)
    assert np.array_equal(results["baseline_cb"],ref.material_pred)
    real=src=="real"
    assert real.sum()==214
    scores={}
    for k,pred in results.items():
        assert set(pred)<=set(labels)
        out=ref[cols].copy()
        out["material_pred"]=pred
        out.to_csv(W/f"astra_cv_oof_{k}.csv",index=False)
        scores[k]=metric(y[real],pred[real])[0]
    def table(groups):
        lines=["Группа | "+" | ".join(methods)]
        for k,mask in groups:
            n=mask.sum()
            counts=[(y[mask]==p[mask]).sum() for p in results.values()]
            lines.append(k+" | "+" | ".join(f"{v}/{n} ({v/n:.1%})" for v in counts))
        return lines
    metals=real & (y=="metal")
    lines=["Победитель по pooled real M: "+max(scores,key=scores.get),
           "M — невзвешенное среднее 6 recall, вычисленных после объединения OOF по всем 214 real-аннотациям.",
           *[f"{k}: {scores[k]:.8f}" for k in sorted(scores,key=scores.get,reverse=True)],
           "",*table([(k,real & (y==k)) for k in labels]),"",
           "Metal recall на real по физическим объектам:",*table([(v,metals & (g==v)) for v in sorted(set(g[metals]))]),
           "", "Внешние фолды: cv_folds.json; 892 уникальных OOF-строки = 214 real + 678 synth, в порядке cv_oof.csv.",
           "CB-object-weighted: независимое переобучение, 892/892 предсказания совпали с material_pred Соланы.",
           "Внешний и внутренний train исключают val-объекты и все кадры с ними. Эмбеддинги и текстовые прототипы заморожены.",
           "Головы: balanced LR с L2-нормализацией. Внутренний StratifiedGroupKFold(4, shuffle=True, random_state=42) по object_id;",
           "выбор по pooled real M внутри внешнего train. C ∈ {0.1,1,10,100}; для CLIP+text alpha ∈ {0,0.5,0.85}.",*config]
    (W/"astra_cv_report.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    with threadpool_limits(4):
        main()

"""Honest nested threshold selection for the existing compact material / take stack."""
from astra_yolo_volume_compare import W
from astra_train import threadpool_limits
from astra_cv import embeddings
from fit_submission_models import fit_compact_head,project
from cv_grouped import read_data
from take_v2 import candidates,take_input,train_candidate,inner_folds
from take_train import fit_density,estimate_mass
from volume_v2 import fit_config,predict_volume_v2
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np,pandas as pd,json

def keys(df):return (df.image_id+"_"+df.ann_id.astype(str)).tolist()
def rates(d,p):
    hit=p==d["take"].to_numpy()
    return np.array([hit[d["take"].eq(i)].mean() for i in (0,1)])
grid=np.r_[np.arange(1,20)*.05,.975,.99]
def select(d):
    def rank(c):
        p=(d.prob.to_numpy()>=c).astype(int)
        kept=[d.object_id.ne(g).to_numpy() for g in d.object_id.unique()]
        return (round(rates(d,p).mean(),12),min(rates(d[k],p[k]).mean() for k in kept),-abs(c-.5))
    return max(grid,key=rank)

def run():
    data,cols=read_data();data["split"]="train"
    teacher=normalize(embeddings("astra_",keys(data)))
    ix={k:i for i,k in enumerate(keys(pd.read_csv(W/"cv_oof.csv")))}
    student=np.load(W/"astra_tmp/astra_small_vectors.npy")[[ix[k] for k in keys(data)]]
    o=pd.read_csv(W/"astra_fixed01_oof.csv")
    det=pd.read_csv(W/"astra_fixed01_det.csv")
    emp=det[det.ann_id.ge(0)].copy()
    vec=np.load(W/"astra_fixed01_z.npy")
    emp=emp.join(o.set_index(["image_id","ann_id"])[["object_id","take"]],on=["image_id","ann_id"],validate="one_to_one")
    def upstream(tr,ev,z):
        h=fit_compact_head(tr,teacher,student)
        m=h["labels"][(project(z,h)@h["W"].T+h["b"]).argmax(1)]
        v=predict_volume_v2(fit_config(tr,("residual",3)),ev,m)
        mass=estimate_mass(ev.form,m,v,fit_density(tr)[0])
        xx=take_input(ev,cols,pd.DataFrame(dict(volume_pred_ml=v,mass_pred_g=mass),index=ev.index))
        return xx.assign(mat_pred=m,form_pred=ev.form,log_density=np.log(mass/v))
    proto,features=candidates(cols)["CatBoost"]
    logs=[];inners=[]
    for fold in json.loads((W/"cv_folds.json").read_text()):
        outer=data.object_id.isin(fold["val_object_ids"])
        tr=data[data.object_id.isin(fold["train_object_ids"]) & ~data.image_id.isin(data.loc[outer,"image_id"])]
        pred=[];simple_train=[]
        for a,b in inner_folds(tr):
            ev=emp[emp.image_id.isin(b.image_id)]
            assert not set(ev.object_id)&set(a.object_id)
            # Cross-fit the training inputs for the inner take classifier as well.
            feats=pd.concat([upstream(aa,bb,student[bb.index]) for aa,bb in inner_folds(a)],verify_integrity=True)
            model=train_candidate(proto,features,a,feats,"real+synth")
            xx=upstream(a,ev,vec[ev.index])
            simple_train.append(xx.join(ev[["take","object_id"]]))
            pred.append(ev[["image_id","ann_id","object_id","take"]].assign(prob=model.predict_proba(xx[features],thread_count=4)[:,1]))
        d=pd.concat(pred,verify_integrity=True)
        assert set(keys(d))==set(keys(tr[tr.source.eq("real")]))
        c=select(d);f=fold["fold"]
        inners.append(d.assign(outer_fold=f))
        logs.append(dict(fold=f,threshold=c,n=len(d),heavy_objects=d.loc[d["take"].eq(0),"object_id"].nunique(),without_s08=select(d[d.object_id.ne("s08")])))
        o.loc[o.fold.eq(f),"nested_cut"]=c
        # Alternative: six continuous/categorical signals derived only from deployed
        # inputs; fit on real cross-fitted predictions, fixed 0.5 decision rule.
        train_x=pd.concat(simple_train,verify_integrity=True)
        va=o.fold.eq(f)
        v=o.loc[va,"fixed_volume_ml"]; m=o.loc[va,"compact"]
        det_idx=emp.set_index(["image_id","ann_id"])
        forms=det_idx.loc[pd.MultiIndex.from_frame(o.loc[va,["image_id","ann_id"]]),"form"].to_numpy()
        mass=estimate_mass(forms,m,v,fit_density(tr)[0])
        vx=pd.DataFrame(dict(mat_pred=m,form_pred=forms,log_density=np.log(mass/v),log_volume_pred_ml=np.log1p(v),log_mass_pred_g=np.log1p(mass)))
        pre=ColumnTransformer([("num",StandardScaler(),["log_volume_pred_ml","log_mass_pred_g","log_density"]),
                               ("cat",OneHotEncoder(handle_unknown="ignore"),["mat_pred","form_pred"])])
        lr=make_pipeline(pre,LogisticRegression(C=.5,max_iter=1000))
        w=1/train_x.groupby("object_id").object_id.transform("size")
        w=w/w.groupby(train_x["take"]).transform("sum");w=w/w.mean()
        lr.fit(train_x,train_x["take"],logisticregression__sample_weight=w)
        o.loc[va,"simple_p"]=lr.predict_proba(vx)[:,1]
    o["nested_take"]=o.compact_p.ge(o.nested_cut).astype(int)
    # All-fold threshold is a deployment recommendation only, not an unbiased metric.
    o["prob"]=o.compact_p
    final=select(o)
    tmps=[select(o[o.object_id.ne(g)]) for g in o.object_id.unique()]
    lines=["Same stored compact_take probabilities, fixed_volume_ml, boxes/masks. Real annotated objects only.",
           "Outer threshold trained on fresh 3-fold inner OOF on outer train, with 3-way grouped upstream crossfit. Finite grid 0.05:0.05:0.95,0.975,0.99. Maximize S, break ties by worst leave-one-object-out S, then closeness to 0.5.",
           pd.DataFrame(logs).to_string(index=False)]
    for name,p in [("fixed .90",o.compact_take),("nested OOF",o.nested_take),("simple logistic fixed .50",o.simple_p.ge(.5)),("deployment cut (apparent)",o.compact_p.ge(final))]:
        rr=rates(o,p.to_numpy())
        lines.append(f"{name}: heavy {rr[0]*44:.0f}/44, light {rr[1]*170:.0f}/170, S={rr.mean():.6f}")
    lines.append(f"Final cut={final:.3f}; without s08={select(o[o.object_id.ne('s08')]):.3f}; delete-one-object optimal cut range {min(tmps):.3f}..{max(tmps):.3f}.")
    lines.append(str(o.assign(correct=o.nested_take.eq(o["take"])).query("take==0").groupby("object_id").correct.agg(["sum","size"])))
    lines.append("Simple logistic: fit on nested real OOF train detections, inverse physical-object count within class, equal total weight heavy/light. C=0.5, fixed cut 0.5; inputs log volume, log estimated mass, log estimated density, predicted form and material. No true mass or validation labels as features.")
    lines.append(str(o.assign(correct=o.simple_p.ge(.5).eq(o["take"])).query("take==0").groupby("object_id").correct.agg(["sum","size"])))
    o.to_csv(W/"astra_take02_oof.csv",index=False,mode="x")
    pd.concat(inners).to_csv(W/"astra_take02_inner.csv",index=False,mode="x")
    with (W/"astra_take02_report.txt").open("x") as f:f.write("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    with threadpool_limits(4):run()

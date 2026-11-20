"""Paired ablation: keep the deployed compact branch; augment only the CLIP branch."""
from astra_visual02 import W,P,keys,labels,np,pd,json,embeddings,normalize,LogisticRegression,threadpool_limits,softmax
from fit_submission_models import fit_compact_head,project

def run():
    gt=pd.read_csv(W/"features_all_v2.csv")
    ref=pd.read_csv(W/"cv_oof.csv")
    ix={key:i for i,key in enumerate(keys(ref))}
    stu=np.load(W/"astra_tmp/astra_small_vectors.npy")[[ix[k] for k in keys(gt)]]
    teacher=normalize(embeddings("astra_",keys(gt)))
    orig=normalize(embeddings("astra_clip_",keys(gt)))
    data=np.load(W/(P+"features.npz")); n=len(data["val_ids"])
    d=pd.read_csv(W/"astra_fixed01_det.csv")
    z=np.load(W/"astra_fixed01_z.npy")[d.ann_id.ge(0)]
    assert list(data["val_ids"])==keys(d[d.ann_id.ge(0)]) and list(data["train_ids"])==keys(gt)
    o=pd.read_csv(W/(P+"oof.csv")); ix={k:i for i,k in enumerate(data["val_ids"])}
    ii=[ix[k] for k in keys(o)]; z=z[ii]
    c=normalize(data["clip"][:n][ii]);extra=normalize(data["clip"][n:])
    for p in json.loads((W/"cv_folds.json").read_text()):
        hold=gt.object_id.isin(p["val_object_ids"])
        tr=gt.object_id.isin(p["train_object_ids"]) & ~gt.image_id.isin(gt.loc[hold,"image_id"])
        va=o.fold.eq(p["fold"])
        h=fit_compact_head(gt[tr],teacher,stu); assert list(h["labels"])==labels
        prob=softmax(project(z[va],h)@h["W"].T+h["b"],axis=1)
        assert np.array_equal(np.array(labels)[prob.argmax(1)],o.loc[va,"compact"])
        for aug in (False,True):
            x=orig[tr]; y=gt.material[tr].to_numpy()
            if aug:x=np.vstack([x,extra[tr]]);y=np.tile(y,2)
            model=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(x,y,sample_weight=np.full(len(y),.5 if aug else 1))
            pc=model.predict_proba(c[va])
            o.loc[va,"compact_clip"+("_aug" if aug else "")]=np.array(labels)[(prob+pc).argmax(1)]
    o.to_csv(W/(P+"all_oof.csv"),mode="x",index=False)
    res={}
    for name in o.columns[5:]:
        tab=o[name].eq(o.material).groupby(o.material).agg(["sum","size","mean"])
        res[name]={"M":tab["mean"].mean(),**{k:f"{r['sum']:.0f}/{r['size']:.0f}" for k,r in tab.iterrows()}}
    txt=pd.DataFrame(res).T.to_string()
    print(txt)
    with (W/(P+"all_report.txt")).open("x") as f:
        f.write("5 Solana folds; fixed matched YOLO masks; M=macro recall over 214 real annotations. Ensemble weights 50:50. 'blend'=full DINO-B + CLIP; 'compact_clip_aug'=unchanged compact branch + augmented CLIP. Each augmented classifier uses originals plus one fixed reflected, rotated and color-shifted crop per original. No inference augmentation. These are paired recipe comparisons, not a fresh evaluation of the best recipe.\n"+txt+"\n")

if __name__=="__main__":
    with threadpool_limits(4):run()

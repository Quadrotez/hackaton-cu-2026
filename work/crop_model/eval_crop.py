"""Evaluate crop-model OOF probabilities and export test probabilities."""
import csv, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work"
oof = json.loads((W / "kaggle_crop_model_out" / "oof.json").read_text())
tp = json.loads((W / "kaggle_crop_model_out" / "test_pred.json").read_text())
CV = pd.read_csv(W / "team_pipeline" / "work" / "cv_oof.csv").set_index(["image_id", "ann_id"])
FORMS = ["bulk", "solid"]
MATS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]

rows = []
for fold, d in oof.items():
    for r, fp, mp in zip(d["rows"], d["form"], d["material"]):
        k = (r["image_id"], int(r["ann_id"]))
        if k not in CV.index: continue
        rows.append({
            "image_id": r["image_id"], "ann_id": r["ann_id"], "fold": fold,
            "form_true": r["form"], "form_pred": FORMS[int(np.argmax(fp))],
            "form_p": max(fp), "material_true": r["material"],
            "material_pred": MATS[int(np.argmax(mp))], "material_p": max(mp),
            "material_prob": dict(zip(MATS, map(float, mp))),
            "form_prob": dict(zip(FORMS, map(float, fp))),
            "volume": float(CV.loc[k, "volume_pred_ml"]), "take": int(CV.loc[k, "take_pred"]),
        })
O = pd.DataFrame(rows)
print("OOF rows", len(O), "real")
print("form acc", round((O.form_true.eq(O.form_pred)).mean(), 4))
cm = pd.crosstab(O.material_true, O.material_pred)
m = float((np.diag(cm.reindex(index=MATS, columns=MATS, fill_value=0)) /
           cm.reindex(index=MATS, columns=MATS, fill_value=0).sum(axis=1)).mean())
print("material macro recall", round(m, 4))
O.to_csv(W / "crop_oof.csv", index=False)

# score.py-style check with true boxes, our form/material, frozen volume/take
labels = json.loads((ROOT / "dataset" / "labels" / "real_train.json").read_text())
ann_by = {}
for a in labels["annotations"]:
    x, y, bw, bh = a["bbox"]
    ann_by[(Path(a["file_name"]).parent.name, a["id"])] = (x, y, x + bw, y + bh)
subrows = [["image_id", "prediction"]]
for _, r in O.iterrows():
    k = (r["image_id"], int(r["ann_id"]))
    if k not in ann_by: continue
    x1, y1, x2, y2 = ann_by[k]
    subrows.append([r["image_id"], f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} 1.0 {r['form_pred']} {r['material_pred']} {r['volume']:.2f} {r['volume']:.2f} {r['take']}"])
p = W / "crop_oof_submission.csv"
with p.open("w", newline="") as f: csv.writer(f).writerows(subrows)
print("submission rows", len(subrows) - 1)

# test probabilities
testrows = []
for i, r in enumerate(tp["rows"]):
    fp = tp["folds"]["1"]["form"][i]
    mp = tp["folds"]["1"]["material"][i]
    testrows.append({"image_id": r["image_id"], "idx": r["idx"],
                     "form_pred": FORMS[int(np.argmax(fp))], "form_p": max(fp),
                     "material_pred": MATS[int(np.argmax(mp))], "material_p": max(mp),
                     "form_prob": json.dumps(dict(zip(FORMS, map(float, fp)))),
                     "material_prob": json.dumps(dict(zip(MATS, map(float, mp))))})
pd.DataFrame(testrows).to_csv(W / "crop_test_probs.csv", index=False)
print("test probs", len(testrows))

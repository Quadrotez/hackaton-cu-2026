"""Material-only A/B: preserve the published l960 anchor except confident wood corrections."""

import argparse
import csv
import json
from pathlib import Path

from catboost import CatBoostClassifier

from export_joint04 import tail
from l960_anchor import ANCHOR, ANCHOR_SHA, additions
from submission_fusion import iou, load
from volume_feature_ab import CACHE, WORK, cv2, detector_features, joint, np, pd
from yolo960_ab import file_sha, fuse_frame, write_csv

L_RUN = WORK / "sprint2_l960_anchor"
L_CSV = L_RUN / "submission_803008_l960_form_only.csv"
WBF_CSV = WORK / "submission_wbf6_final.csv"
THRESHOLD = .75


def material_model(train, columns):
    weights = 1 / train.groupby("object_id").object_id.transform("size")
    total = weights.groupby(train.material).sum()
    weights = weights / train.material.map(total)
    model = CatBoostClassifier(
        iterations=250, depth=4, learning_rate=.045, l2_leaf_reg=6, allow_writing_files=False, verbose=False,
        thread_count=3, random_seed=42,
    )
    return model.fit(train[columns], train.material, sample_weight=weights / weights.mean())


def wood_probability(model, rows, columns):
    return model.predict_proba(rows[columns])[:, list(model.classes_).index("wood")]


def test_features():
    """Recover the original WBF masks supplying attributes, not the alternate l960 masks."""
    if file_sha(ANCHOR) != ANCHOR_SHA:
        raise ValueError("Original 80.3008 file has changed")
    reference, wbf = load(ANCHOR), load(WBF_CSV)
    raw = WORK / "test_wbf_v1"
    rows = {r["image_id"]: r for r in map(json.loads, (raw / "base_detections.jsonl").read_text().splitlines())}
    ds = joint.Detections()
    ids, perspective = [], []
    with np.load(raw / "base_masks.npz") as masks:
        for fid, base in reference.items():
            row = rows[fid]
            w = row["shape"][1]
            if any(any(a[k] != b[k] for k in ("box", "score", "form"))
                   for a, b in zip(wbf[fid], row["detections"], strict=True)):
                raise ValueError(f"{fid}: WBF mask alignment mismatch")
            unpacked = np.unpackbits(masks[fid], axis=-1, count=w).astype(bool)
            pairs = []
            for j, pred in enumerate(base):
                overlap, k = max((iou(pred["box"], p["box"]), k) for k, p in enumerate(wbf[fid])
                                 if p["form"] == pred["form"])
                wbf_pred = wbf[fid][k]
                if overlap < .75 or wbf_pred["tail"].split()[:-1] != pred["tail"].split()[:-1]:
                    raise ValueError(f"{fid}: no verified donor for prediction {j}")
                pairs.append((wbf_pred, unpacked[k]))
                ids.append((fid, j))
                perspective.append(wbf_pred["box"][3] / row["shape"][0])
            ds.frame("test", fid, joint.DATA / "real" / "test" / fid, pairs)
    return reference, ids, pd.DataFrame(ds.rows).assign(y_bottom=perspective)


def donor_index(fid, base, idx, prediction):
    if idx < len(base):
        return idx
    overlap, neg = max((iou(prediction["box"], x["box"]), -j)
                       for j, x in enumerate(base) if x["form"] != prediction["form"])
    donor = -neg
    if overlap < .5 or prediction["tail"].split()[1:] != base[donor]["tail"].split()[1:]:
        raise ValueError(f"{fid}: l960 attribute donor mismatch")
    return donor


def variant(pred, probability):
    return dict(pred, material="wood") if probability >= THRESHOLD else pred


def score_oof(data, columns, val, specs, anchor, sources, labels):
    metric = joint.metric
    pred, base = {}, {}
    for spec in specs:
        f = spec["fold"]
        train = data.loc[data.image_id.isin(spec["train_image_ids"])]
        hold = val.loc[val.fold.eq(f)]
        if set(train.object_id) != set(spec["train_object_ids"]) or set(train.object_id) & set(spec["val_object_ids"]):
            raise ValueError(f"Fold {f}: invalid training split")
        p = wood_probability(material_model(train, columns), hold, columns)
        confidence = dict(zip(zip(hold.image_id, hold.idx, strict=True), p, strict=True))
        for fid in spec["val_real_image_ids"]:
            old = anchor[fid]
            raw = [dict(d, tail=tail(d)) for d in sources[fid][f"yolo11l_fold_{f:02d}"]]
            l960 = metric._parse(fuse_frame([raw], [1.], ()), True, fid)
            parts = list(enumerate(old)) + [
                (j, dict(d, **{k: old[j][k] for k in ("material", "volume", "take")}))
                for d, j in additions(old, l960)
            ]
            base[fid] = [d for _, d in parts]
            pred[fid] = [variant(d, confidence[fid, j]) for j, d in parts]

    def compare(ids):
        g = {i: labels[i] for i in ids}
        a, b = [metric.components(g, {i: p[i] for i in ids}) for p in (base, pred)]
        if any(a[k] != b[k] for k in ("D", "V", "S")):
            raise ValueError("Material-only experiment changed another metric")
        return {"base": a, "wood": b, "delta": b["Score"] - a["Score"]}

    def affected_objects():
        images = json.loads((joint.DATA / "labels/real_train.json").read_text())
        frame_by_id = {x["id"]: Path(x["file_name"]).parent.name for x in images["images"]}
        return {frame_by_id[a["image_id"]]: a["attributes"]["object_id"] for a in images["annotations"]}

    objects = affected_objects()
    changed = {i for i, gs in labels.items() if [p and p["material"] for p in metric._match(gs, base[i])]
               != [p and p["material"] for p in metric._match(gs, pred[i])]}
    ids = sorted({objects[i] for i in changed})
    return {"pooled": compare(labels), "folds": {s["fold"]: compare(s["val_real_image_ids"]) for s in specs},
            "affected_objects": ids, "changed_matches_by_object": {
                o: sum(objects[i] == o for i in changed) for o in ids},
            "leave_one_affected_object_out_delta": {
                o: compare([i for i in labels if objects.get(i) != o])["delta"] for o in ids}}


def export(odir, data, columns):
    report = json.loads((L_RUN / "report.json").read_text())
    if file_sha(L_CSV) != report["test"]["output_sha256"]:
        raise ValueError("Frozen l960 CSV has changed")
    original, ids, features = test_features()
    prob = dict(zip(ids, wood_probability(material_model(data, columns), features, columns), strict=True))
    candidate = load(L_CSV)
    changed = []
    with L_CSV.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            fid, text = row["image_id"], row["prediction"]
            old = original[fid]
            parts = text.split(";") if old else []
            for idx, p in enumerate(candidate[fid]):
                donor = donor_index(fid, old, idx, p)
                fields = parts[idx].split()
                if prob[fid, donor] >= THRESHOLD and fields[6] != "wood":
                    changed.append({"image": fid, "index": idx, "from": fields[6], "prob": prob[fid, donor]})
                    fields[6] = "wood"
                    parts[idx] = " ".join(fields)
            rows.append((fid, ";".join(parts) or text))
    path = odir / "submission_807475_wood_only.csv"
    write_csv(path, rows, joint.validate_csv)
    return {"file": path.name, "sha256": file_sha(path), "base_sha256": file_sha(L_CSV),
            "changes": changed, "frames_changed": len({c["image"] for c in changed})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint2_material_wood")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    cv2.setNumThreads(1)
    index = json.loads((CACHE / "mask_index.json").read_text())
    manifest = WORK / "detector_folds_v1.json"
    if file_sha(manifest) != index["fold_manifest_sha256"]:
        raise ValueError("OOF manifest mismatch")
    specs = json.loads(manifest.read_text())["folds"]
    anchor = json.loads((CACHE / "predictions.json").read_text())["n640_rescue"]
    sources = json.loads((CACHE / "per_model_predictions.json").read_text())
    frames = {i: (joint.DATA / im["file_name"]).parent for src, i, im, _ in joint.load_frames() if src == "real"}
    val = detector_features(index, frames, anchor, False)
    data, columns = joint.read_data()
    columns = [k for k in columns if k in val and k not in {"form", "material", "source", "fold", "object_id",
                                                           "ann_id", "image_id", "volume_ml", "mass_g", "take"}]
    labels = {r.image_id: joint.metric._parse(r.objects, False, r.image_id) for r in
              joint.metric.labels_to_solution(joint.DATA / "labels/real_train.json").itertuples(index=False)}
    oof = score_oof(data, columns, val, specs, anchor, sources, labels)
    args.out.mkdir(parents=True, exist_ok=False)
    test = export(args.out, data, columns)
    (args.out / "report.json").write_text(json.dumps({"wood_threshold": THRESHOLD, "oof": oof, "test": test}, indent=2)
                                         + "\n", encoding="utf-8")
    print(f"OOF: {oof['pooled']['delta']:+.4f}; folds: "
          f"{[round(x['delta'], 4) for x in oof['folds'].values()]}; "
          f"test: {len(test['changes'])} material changes on {test['frames_changed']} frames")


if __name__ == "__main__":
    main()

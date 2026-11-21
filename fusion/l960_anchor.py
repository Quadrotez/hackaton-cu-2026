"""Add only missing, overlapping l960 form hypotheses to the frozen submission."""

import argparse
import csv
import json
import sys
from pathlib import Path

from export_joint04 import tail
from submission_fusion import iou, load
from yolo960_ab import MAX_PER_IMAGE, ROOT, file_sha, fuse_frame, write_csv

WORK = ROOT / "work"
CACHE = WORK / "sprint1_agnostic_oof_v2"
ANCHOR = WORK / "sprint1_grisha" / "submission_fused_max_7897_rescue_small.csv"
ANCHOR_SHA = "1b28c7c8a9df5062d3974f87cabe75762cfa4eb10dbe84c70571652144805135"
IOU = .5
CONF = .025


def additions(anchor, candidates):
    """Yield (new hypothesis, anchor for attributes) without changing the anchor."""
    seen = list(anchor)
    for candidate in candidates:
        if len(seen) == MAX_PER_IMAGE:
            break
        if candidate["score"] < CONF or any(
            old["form"] == candidate["form"] and iou(old["box"], candidate["box"]) >= IOU
            for old in seen
        ):
            continue
        overlap, neg_idx = max(
            ((iou(old["box"], candidate["box"]), -idx) for idx, old in enumerate(anchor)
             if old["form"] != candidate["form"]),
            default=(0, 0),
        )
        if overlap >= IOU:
            seen.append(candidate)
            yield candidate, -neg_idx


def check_oof(metric):
    index = json.loads((CACHE / "mask_index.json").read_text())
    if index["fold_manifest_sha256"] != file_sha(WORK / "detector_folds_v1.json"):
        raise ValueError("OOF fold manifest mismatch")
    anchor = json.loads((CACHE / "predictions.json").read_text())["n640_rescue"]
    sources = json.loads((CACHE / "per_model_predictions.json").read_text())
    gt = {r.image_id: metric._parse(r.objects, False, r.image_id) for r in metric.labels_to_solution(
        ROOT / "dataset/labels/real_train.json").itertuples(index=False)}
    ids = [i for f in index["folds"] for i in f["frames"]]
    if len(ids) != 263 or set(ids) != set(gt) or sources.keys() != gt.keys() or anchor.keys() != gt.keys():
        raise ValueError("Invalid OOF frame coverage")
    result = {}
    for spec in index["folds"]:
        model = f"yolo11l_fold_{spec['fold']:02d}"
        for fid in spec["frames"]:
            if model not in spec["sources"] or model not in sources[fid]:
                raise ValueError(f"{fid}: missing verified l960 prediction")
            old = anchor[fid]
            fused = fuse_frame([[dict(d, tail=tail(d)) for d in sources[fid][model]]], [1.], ())
            new = metric._parse(fused, True, fid)
            result[fid] = [*old, *(dict(d, **{k: old[j][k] for k in ("material", "volume", "take")})
                                   for d, j in additions(old, new))]

    def compare(keys):
        labels = {i: gt[i] for i in keys}
        before = metric.components(labels, {i: anchor[i] for i in keys})
        after = metric.components(labels, {i: result[i] for i in keys})
        return {"anchor": before, "candidate": after, "delta": after["Score"] - before["Score"],
                "boxes": [sum(len(p[i]) for i in keys) for p in (anchor, result)]}

    return {"pooled": compare(ids), "folds": {f["fold"]: compare(f["frames"]) for f in index["folds"]}}


def export_test(metric, validate_csv, path):
    if file_sha(ANCHOR) != ANCHOR_SHA:
        raise ValueError("Expected the exact 80.3008 anchor CSV")
    paths = [WORK / "submission_yolo960_A_attr/per_model" / f"yolo11l_fold_{k:02d}.csv" for k in range(1, 6)]
    for p in (ANCHOR, *paths):
        validate_csv(p)
    anchor = load(ANCHOR)
    models = [load(p) for p in paths]
    added = changed = 0
    with ANCHOR.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            fid, text = row["image_id"], row["prediction"]
            old_parts = [] if text == "none" else text.split(";")
            fused = fuse_frame([m[fid] for m in models], [1.] * len(models), ())
            new_parts = []
            for d, j in additions(anchor[fid], metric._parse(fused, True, fid)):
                attrs = old_parts[j].split()[6:]
                fields = [*(f"{v:.1f}" for v in d["box"]), f"{d['score']:.6f}", d["form"], *attrs]
                new_parts.append(" ".join(fields))
            added += len(new_parts)
            changed += bool(new_parts)
            rows.append((fid, ";".join(old_parts + new_parts) if new_parts else text))
    write_csv(path, rows, validate_csv)
    return {"added": added, "changed_frames": changed,
            "inputs": {str(p.relative_to(ROOT)): file_sha(p) for p in (ANCHOR, *paths)},
            "output_sha256": file_sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint2_l960_anchor")
    args = parser.parse_args()
    sys.path.insert(0, str(WORK))
    from submission_tools import metric, validate_csv

    oof = check_oof(metric)
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "submission_803008_l960_form_only.csv"
    test = export_test(metric, validate_csv, path)
    report = {"iou": IOU, "conf": CONF, "oof": oof, "test": test}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OOF: {oof['pooled']['anchor']['Score']:.4f} → {oof['pooled']['candidate']['Score']:.4f}; "
          f"test: +{test['added']} form hypotheses on {test['changed_frames']} frames")


if __name__ == "__main__":
    main()

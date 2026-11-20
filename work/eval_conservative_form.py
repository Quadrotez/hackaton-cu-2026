"""Test high-confidence shape overrides with thresholds chosen only inside outer train."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_detector_folds import DEFAULT_OUT, digest
from detector_fold_eval import frame_data, ground_truth, mask_features, measure, write_json
from form_refine import fit, group_folds, predict_solid
from material_split import DATA, WORK
from submission_tools import metric
from train_detector_folds import frame_map, read_manifest


THRESHOLDS = (.95, .975, .99, .995)
OOF_ROOT = WORK / "detector_oof_v1"


def apply_override(rows, probabilities, threshold, base_col):
    base = rows[base_col].to_numpy()
    if threshold is None:
        return base.copy()
    p = np.asarray(probabilities)
    shape = np.where(p >= .5, "solid", "bulk")
    valid = rows["height_m_p90"].notna() & rows["depth_zero_share"].le(.75)
    return np.where(valid.to_numpy() & (np.maximum(p, 1 - p) >= threshold), shape, base)


def choose_threshold(train):
    inner = pd.concat([
        pd.Series(predict_solid(fit(a, "CatBoost", 1), b), index=b.index)
        for a, b in group_folds(train)
    ], verify_integrity=True).reindex(train.index)
    if inner.isna().any():
        raise ValueError("Inner OOF is missing probabilities")
    real = train.loc[train["source"].eq("real")]
    w = (1. / real.groupby("object_id")["object_id"].transform("size")).to_numpy()
    truth, base = real["form"].to_numpy(), real["yolo_form"].to_numpy()
    trials = []
    for t in THRESHOLDS:
        out = apply_override(real, inner.loc[real.index], t, "yolo_form")
        flips = out != base
        rights = (out == truth) & flips
        wrongs = (out != truth) & flips
        prec = float(w[rights].sum() / w[flips].sum()) if flips.any() else None
        n_objs = int(real.loc[flips, "object_id"].nunique())
        net = (out == truth).astype(int) - (base == truth).astype(int)
        per_form = {f: float((w[truth == f] * net[truth == f]).sum()) for f in metric.FORMS}
        gain = float(w @ net)
        eligible = (flips.sum() >= 4 and n_objs >= 2 and prec >= .90
                    and gain > 0 and min(per_form.values()) >= -1e-12)
        trials.append({
            "threshold": t, "flips": int(flips.sum()), "objects_changed": n_objs,
            "fixes": int(rights.sum()), "breaks": int(wrongs.sum()), "weighted_precision": prec,
            "weighted_net_gain": gain, "net_gain_by_form": per_form, "eligible": bool(eligible),
        })
    choices = [row for row in trials if row["eligible"]]
    chosen = max(choices, key=lambda row: (row["weighted_net_gain"], row["threshold"])) if choices else None
    return chosen["threshold"] if chosen else None, trials


def validate_fold(folder, spec, frames, camera):
    train = pd.read_csv(folder / "shape_train_predicted_features.csv")
    if not set(train["image_id"]) <= set(spec["train_image_ids"]):
        raise ValueError(f"Non-train masks in fold {spec['fold']}")
    if not set(train["object_id"]) <= set(spec["train_object_ids"]):
        raise ValueError(f"Non-train objects in fold {spec['fold']}")
    t, trials = choose_threshold(train)
    model = fit(train, "CatBoost", 1) if t is not None else None
    ids = spec["val_real_image_ids"]
    real = set(ids)
    base, pred = {}, {}
    with np.load(folder / "base_masks.npz", allow_pickle=False) as z:
        with (folder / "base_detections.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                i = d["image_id"]
                if i not in real:
                    continue
                if i in base:
                    raise ValueError(f"Duplicate validation image {i}")
                base[i] = d["detections"]
                if model is None or not base[i]:
                    pred[i] = [p.copy() for p in base[i]]
                    continue
                h, w = d["shape"]
                compact = z[i]
                if compact.shape != (len(base[i]), h, (w + 7) // 8):
                    raise ValueError(f"Predicted mask shape/count mismatch in {i}")
                _, depth, plane = frame_data(frames[i][1])
                rows = pd.DataFrame.from_records([
                    mask_features(p, np.unpackbits(mask, axis=-1, count=w).astype(bool), depth, plane, camera)
                    for p, mask in zip(base[i], compact, strict=True)
                ])
                forms = apply_override(rows, predict_solid(model, rows), t, "form")
                pred[i] = [dict(p, form=str(form)) for p, form in zip(base[i], forms, strict=True)]
    if base.keys() != real or pred.keys() != real:
        raise ValueError(f"Incomplete coverage of real holdout for fold {spec['fold']}")
    gt = ground_truth(ids, frames)
    flips = sum(p["form"] != q["form"] for i in ids for p, q in zip(base[i], pred[i], strict=True))
    return base, pred, {"fold": spec["fold"], "threshold": t, "inner_trials": trials, "val_flips": flips,
                        "baseline": measure(gt, base), "conservative": measure(gt, pred)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--oof-root", type=Path, default=OOF_ROOT)
    parser.add_argument("--out", type=Path, default=WORK / "conservative_form_oof_v1")
    args = parser.parse_args()
    m = read_manifest(args.manifest)
    run = json.loads((args.oof_root / "run.json").read_text(encoding="utf-8"))
    if run["manifest_sha256"] != digest(args.manifest):
        raise ValueError("The cached detections are from a different split")
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    f_map = frame_map()
    b, p, folds = {}, {}, []
    with threadpool_limits(3):
        for spec in m["folds"]:
            f = spec["fold"]
            base, pred, details = validate_fold(args.oof_root / f"fold_{f:02d}" / "evaluation", spec, f_map, camera)
            if b.keys() & base.keys():
                raise ValueError("Duplicate real OOF frame")
            b.update(base)
            p.update(pred)
            folds.append(details)
            d0, d1 = (details[k]["D_points"] for k in ("baseline", "conservative"))
            print(f"fold {f} threshold={details['threshold']} flips={details['val_flips']} 40D={d0:.3f}->{d1:.3f}",
                  flush=True)
    gt = ground_truth(sorted(b), f_map)
    pooled = {"baseline": measure(gt, b), "conservative": measure(gt, p)}
    reference = json.loads((args.oof_root / "pooled_real_ablation.json").read_text())["variants"]["yolo"]
    if not np.isclose(pooled["baseline"]["D"], reference["D"], rtol=0, atol=1e-12):
        raise ValueError("Baseline does not reproduce the official OOF")
    write_json(args.out / "predictions.json", p)
    write_json(args.out / "report.json", {"manifest_sha256": digest(args.manifest),
               "inner_rule": "Train-only 4-fold object-grouped geometry CatBoost. Validation thresholds "
                             "require at least 4 flips and 2 changed objects, >=90% object-weighted precision, "
                             "positive weighted gain and no negative net change for either true form.",
               "folds": folds, "pooled": pooled})
    print(f"pooled 40D {pooled['baseline']['D_points']:.3f}->{pooled['conservative']['D_points']:.3f}")


if __name__ == "__main__":
    main()

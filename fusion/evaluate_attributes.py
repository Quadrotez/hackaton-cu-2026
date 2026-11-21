"""Fit and audit the optional attribute postprocessor with nested grouped CV."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from attribute_postprocess import (
    DEFAULT_MODEL, MODES, AttributePolicy, BulkCalibration, DensityPrior, TakeRule,
)

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CUTS = (.50, .60, .65, .70, .75, .80, .85, .90, .95)
SMALL_ML = (150., 200., 250., 300., 350.)

sys.path.insert(0, str(WORK))
import astra_joint04 as joint
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from volume_calibration import volume_score


def take_selection(inner, sizes):
    y = inner["take"].to_numpy()
    heavy = inner.groupby("object_id")["take"].first().loc[lambda x: x.eq(0)].index

    def rank(rule):
        p = inner.p_light.ge(rule.cut) | (inner.volume_pred_ml.gt(0) & inner.volume_pred_ml.le(rule.small_ml))
        hit = inner.assign(hit=p.to_numpy() == y).groupby(["take", "object_id"]).hit.mean()
        h, l = hit.loc[0], hit.loc[1]
        # Give each physical item one vote, with a conservative tie-break on heavy items.
        jackknife = min(((h.sum() - h[k]) / (len(h) - 1) + l.mean()) / 2 for k in heavy) if len(h) > 1 else 0.
        return round((h.mean() + l.mean()) / 2, 12), round(jackknife, 12), -rule.small_ml, -abs(rule.cut - .8)

    return max((TakeRule(cut, size) for cut in CUTS for size in sizes), key=rank)


def rescue_selection(inner):
    # Train a one-sided rescue without needing CatBoost take predictions here:
    # maximize eligible light items under zero eligible heavy views on inner OOF.
    # The existing decision is overwritten only if it would reject an eligible item.
    best, best_key = TakeRule(1., 0.), (0., 0., 0.)
    for size in SMALL_ML:
        for cut in CUTS:
            eligible = inner.volume_pred_ml.le(size) & inner.p_light.ge(cut)
            if eligible[inner["take"].eq(0)].any():
                continue
            light = inner.loc[inner["take"].eq(1), ["object_id"]].assign(eligible=eligible)
            key = (light.groupby("object_id").eligible.mean().mean(), -size, cut)
            if key > best_key:
                best_key, best = key, TakeRule(cut, size)
    return best


def fit_calibration(inner):
    x = inner.loc[inner.form.eq("bulk") & inner.volume_pred_ml.gt(0)].copy()
    if x.object_id.nunique() < 5:
        return BulkCalibration((), ())
    x["x"] = np.log(x.volume_pred_ml)
    x["y"] = np.log(x.volume_ml)
    x["w"] = 1. / x.groupby("object_id").object_id.transform("size")
    # Five coarse bins, with one total vote per physical object in the fit.
    x["bin"] = pd.qcut(x.x, 5, duplicates="drop")
    x["wx"], x["wy"] = x.w * x.x, x.w * x.y
    knots = x.groupby("bin", observed=True)[["w", "wx", "wy"]].sum()
    a, b = knots.wx / knots.w, knots.wy / knots.w
    if len(a) < 2:
        return BulkCalibration((), ())
    b = IsotonicRegression().fit_transform(a, b, sample_weight=knots.w)
    return BulkCalibration(tuple(map(float, a)), tuple(map(float, b)))


def fit_prior(rows):
    real = rows.loc[rows.source.eq("real")]
    return DensityPrior.fit(real[["object_id", "form", "material", "volume_ml", "mass_g"]]
                            .itertuples(index=False, name=None))


def with_inner_probabilities(inner, inner_trains, excluded=None):
    parts = []
    for k, hold in inner.groupby("inner_fold"):
        fit = inner_trains[k]
        if excluded is not None:
            hold = hold.loc[hold.object_id.ne(excluded)]
            fit = fit.loc[fit.object_id.ne(excluded)]
        assert not set(fit.object_id) & set(hold.object_id)
        assert not set(fit.image_id) & set(hold.image_id)
        prior = fit_prior(fit)
        parts.append(hold.assign(p_light=[prior.light_probability(f, m, v) for f, m, v
                                         in zip(hold.form, hold.material_pred, hold.volume_pred_ml, strict=True)]))
    return pd.concat(parts, verify_integrity=True)


def inner_features(train, views):
    """Cross-fit upstream heads on disjoint objects/frames, using labelled masks as geometry.

    An outer YOLO saw the inner held-out objects: running it on inner holdouts would
    contaminate this stage. Frozen embeddings/RGB-D features on annotated crops
    are a proxy here. Only outer evaluation uses real OOF detector predictions.
    """
    outputs, fits = [], {}
    for k, (fit, hold) in enumerate(joint.grouped_folds(train, 3), 1):
        assert not set(fit.object_id) & set(hold.object_id)
        assert not set(fit.image_id) & set(hold.image_id)
        assert (set(fit.index) | set(hold.index)) <= set(train.index)
        fits[k] = fit
        real = hold.loc[hold.source.eq("real")]
        model = joint.fit_up(fit, views)
        out = joint.up(model, real, [view[real.index] for view in views])
        outputs.append(real[["image_id", "object_id", "form", "take", "volume_ml"]]
                       .join(out[["material_pred", "volume_pred_ml"]]).assign(inner_fold=k))
    inner = pd.concat(outputs, verify_integrity=True)
    assert set(inner.index) == set(train.index[train.source.eq("real")])
    return inner, fits


def tune(train, inner, fits, excluded=None, bulk=None):
    calibrated = with_inner_probabilities(inner, fits, excluded)
    if excluded is not None:
        train = train.loc[train.object_id.ne(excluded)]
    return AttributePolicy(fit_prior(train), take_selection(calibrated, (0.,)),
                           take_selection(calibrated, SMALL_ML),
                           rescue_selection(calibrated),
                           fit_calibration(calibrated) if bulk is None else bulk)


def transform(policy, mode, predictions):
    result = {}
    for i, detections in predictions.items():
        result[i] = []
        for p in detections:
            v, m, t = policy.apply(mode, p["form"], p["material"], p["volume"], p["mass"], p["take"])
            result[i].append(dict(p, volume=v, mass=m, take=t))
    return result


def metrics(gt, pred):
    c = joint.metric.components(gt, pred)
    matches = [(g, p) for i, gs in gt.items()
               for g, p in zip(gs, joint.metric._match(gs, pred[i]), strict=True)]
    c["take_hits"] = {k: {"correct": sum(p is not None and p["take"] == k for g, p in matches if g["take"] == k),
                         "total": sum(g["take"] == k for g, _ in matches)} for k in (0, 1)}
    c["V_by_true_form"] = {}
    for form in joint.metric.FORMS:
        pairs = [(g, p) for g, p in matches if g["form"] == form]
        if pairs:
            c["V_by_true_form"][form] = volume_score(
                [p["volume"] if p else 0. for g, p in pairs], [g["volume"] for g, p in pairs])
    return c


def read_inputs():
    data, _ = joint.read_data()
    data = data.merge(pd.read_csv(joint.GEOMETRY_PATH), on=["image_id", "ann_id"],
                      validate="one_to_one").assign(split="train")
    ids = joint.keys(data)
    with np.load(WORK / "astra_joint04_train_flip.npz") as flip:
        assert flip["ids"].tolist() == ids
        views = [joint.normalize(joint.embeddings(prefix, ids)) for prefix in ("astra_", "astra_clip_")]
        views += [flip[k] for k in ("d", "c")]
    assert all(len(v) == len(data) and np.isfinite(v).all() for v in views)
    folds_path = WORK / "detector_folds_v1.json"
    oof_path = WORK / "astra_joint04_oof.json"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifests = json.loads(folds_path.read_text())["folds"]
    assert [p["fold"] for p in manifests] == list(range(1, 6))
    oof = json.loads(oof_path.read_text())
    detector = WORK / "detector_oof_v1"
    assert digest(folds_path) == json.loads((detector / "run.json").read_text())["manifest_sha256"]
    frames = [(i, (joint.DATA / im["file_name"]).parent, anns)
              for source, i, im, anns in joint.load_frames() if source == "real"]
    gt = joint.ground_truth(frames)
    assert len(gt) == len(oof) == 263 and set(gt) == set(oof) and sum(map(len, oof.values())) == 277
    assert sum(map(len, gt.values())) == 214
    for spec in manifests:
        f = spec["fold"]
        raw = (detector / f"fold_{f:02d}" / "evaluation" / "base_detections.jsonl").read_text()
        real = {r["image_id"]: r["detections"] for r in map(json.loads, raw.splitlines()) if r["source"] == "real"}
        assert set(real) == set(spec["val_real_image_ids"])
        for i, detections in real.items():
            assert [{k: p[k] for k in ("box", "score", "form")} for p in oof[i]] == detections
    return data, views, manifests, oof, gt, {"fold_manifest_sha256": digest(folds_path),
                                               "joint04_oof_sha256": digest(oof_path)}


def run(out):
    if out.exists():
        raise FileExistsError(out)
    data, views, manifests, oof, gt, hashes = read_inputs()
    heavy_real = data.loc[data.source.eq("real") & data["take"].eq(0)]
    heavy_ids = sorted(heavy_real.object_id.unique())
    held_frames = {k: set(heavy_real.loc[heavy_real.object_id.eq(k), "image_id"]) for k in heavy_ids}
    all_pred = {m: {} for m in MODES}
    stressed = {k: {m: {} for m in ("density", "density-small", "rescue-small")} for k in heavy_ids}
    details, tables = [], []
    for spec in manifests:
        f = spec["fold"]
        real_ids = spec["val_real_image_ids"]
        train = data.loc[data.image_id.isin(spec["train_image_ids"])]
        assert set(train.object_id) == set(spec["train_object_ids"])
        assert not set(train.object_id) & set(spec["val_object_ids"])
        assert not set(train.image_id) & set(real_ids + spec["val_synth_image_ids"] + spec["excluded_mixed_synth_image_ids"])
        inner, fits = inner_features(train, views)
        policy = tune(train, inner, fits)
        g, p = ({i: source[i] for i in real_ids} for source in (gt, oof))
        detail = {"fold": f, "train_objects": int(train.object_id.nunique()),
                  "heavy_val_objects": sorted(set(heavy_ids) & set(spec["val_object_ids"])),
                  "cut": vars(policy.density), "cut_with_small": vars(policy.density_small),
                  "rescue_small": vars(policy.rescue_small), "variants": {}}
        for mode in MODES:
            preds = transform(policy, mode, p)
            detail["variants"][mode] = metrics(g, preds)
            assert not set(preds) & set(all_pred[mode])
            all_pred[mode].update(preds)
        # Remove a heavy ID from all density fits and threshold selection; upstream
        # inner attribute features are held fixed, so this isolates the new stage.
        for k in heavy_ids:
            reduced = tune(train, inner, fits, excluded=k, bulk=policy.bulk) if k in set(train.object_id) else policy
            for mode in stressed[k]:
                stressed[k][mode].update(transform(reduced, mode, p))
        tables.append(with_inner_probabilities(inner, fits).assign(outer_fold=f))
        details.append(detail)
        print(f"Fold {f}: " + " | ".join(f"{m}: {detail['variants'][m]['Score']:.3f}" for m in MODES), flush=True)

    pooled = {m: metrics(gt, p) for m, p in all_pred.items()}
    for m, c in pooled.items():
        assert set(all_pred[m]) == set(gt)
        assert all(c[k] == pooled["off"][k] for k in ("D", "M")), f"{m} changed detections or material"
    fold_deltas = {m: [f["variants"][m]["Score"] - f["variants"]["off"]["Score"] for f in details] for m in MODES}
    jackknife = {}
    for k, ids in held_frames.items():
        g = {i: row for i, row in gt.items() if i not in ids}
        base = joint.metric.components(g, oof)["Score"]
        jackknife[k] = {m: joint.metric.components(g, p)["Score"] - base for m, p in all_pred.items()}
    stress = {k: {m: metrics(gt, p) for m, p in modes.items()} for k, modes in stressed.items()}

    # Do not recommend an unstable variant merely because it has the best pooled score.
    base = pooled["off"]["Score"]
    eligible = [m for m in MODES if pooled[m]["Score"] > base
                and all(v >= -1e-9 for v in fold_deltas[m])
                and min(x[m] for x in jackknife.values()) > 0
                and (m not in next(iter(stress.values())) or min(x[m]["Score"] for x in stress.values()) > base)]
    recommended = max(eligible, key=lambda m: pooled[m]["Score"], default="off")
    data_inner, data_fit = inner_features(data, views)
    deployment = replace(tune(data, data_inner, data_fit), recommended=recommended)
    tables.append(with_inner_probabilities(data_inner, data_fit).assign(outer_fold=0))

    report = {**hashes, "frames": len(gt), "heavy_real_objects": heavy_ids, "recommended": recommended,
              "selection_guard": "positive pooled delta, no worse on all five outer folds, positive after deleting any one heavy ID from evaluation or from density fits and threshold selection",
              "pooled": pooled, "folds": details, "fold_delta_Score": fold_deltas,
              "drop_heavy_from_eval_delta_Score": jackknife, "drop_heavy_from_density_fit": stress}
    out.mkdir(parents=True, exist_ok=False)
    for file, payload in (("report.json", report), ("policy.json", deployment.to_dict())):
        (out / file).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(tables, ignore_index=True).to_csv(out / "inner_oof.csv", index=False)
    for m, c in pooled.items():
        print(f"{m:18s} Score {c['Score']:.4f} ({c['Score'] - base:+.4f}) | "
              f"V {c['V']:.4f} S {c['S']:.4f} | "
              f"fold deltas {', '.join(f'{d:+.2f}' for d in fold_deltas[m])}")
    print(f"Recommended: {recommended} ({out / 'policy.json'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL.parent)
    args = parser.parse_args()
    with joint.threadpool_limits(4):
        run(args.out)


if __name__ == "__main__":
    main()

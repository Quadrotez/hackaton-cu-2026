"""Nested object-grouped A/B of metal/plastic overrides on frozen bulk-y OOF."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metal_plastic_ab import PAIR, features, load_embeddings, tabular
from volume_feature_ab import CACHE, WORK, cv2, joint, np, pd, threadpool_limits
from yolo960_ab import file_sha

VOLUME_RUN = WORK / "sprint3_bulk_y_frozen"
THRESHOLDS = (.55, .65, .75, .85, .95)
DIRECTIONS = (("plastic", "metal"), ("metal", "plastic"))
DEPTH = ("depth_", "missing_", "holes_spec_")
RGB = ("spec_", "v_p", "s_p", "texture_", "sobel_", "laplace_", "local_var_",
       "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std")


@dataclass(frozen=True)
class Rule:
    predictor: str
    threshold: float
    mid_plastic: bool = False


def columns(frame):
    depth = [c for c in frame if c.startswith(DEPTH) and not c.startswith("depth_points")]
    rgb = [c for c in frame if c.startswith(RGB)]
    return {"depth": depth, "rgb": rgb, "physical": depth + rgb}


def baseline_prob(train, hold_views, bank):
    labels = np.array(joint.labels)
    parts = []
    for i, x in enumerate(bank):
        model = LogisticRegression(C=1, class_weight="balanced", max_iter=1000)
        model.fit(x[train.index], train.material)
        if not np.array_equal(model.classes_, labels):
            raise ValueError("Frozen material class order changed")
        parts.extend(model.predict_proba(y) for y in (hold_views[i], hold_views[i+2]))
    p = np.mean(parts, axis=0)
    return labels[p.argmax(axis=1)], p[:, np.flatnonzero(labels == "plastic")[0]]


def predict_pair(train, val, cols):
    y = train.material.eq("metal")
    w = 1 / train.groupby("object_id").object_id.transform("size")
    w /= y.map(w.groupby(y).sum())
    linear = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                           StandardScaler(), LogisticRegression(C=.1, max_iter=800))
    linear.fit(train[cols["physical"]], y, logisticregression__sample_weight=w / w.mean())
    return {
        **{f"cat_{k}": tabular(train, val, v) for k, v in cols.items()},
        "lr_physical": linear.predict_proba(val[cols["physical"]])[:, 1],
    }


def mask(rows, rule, source):
    if rule is None:
        return np.zeros(len(rows), dtype=bool)
    raw = rows[rule.predictor]
    probability = raw if source == "plastic" else 1 - raw
    eligible = rows.base == source
    if rule.mid_plastic:
        eligible = eligible & rows.p_plastic.between(.35, .85)
    return (eligible & (probability >= rule.threshold)).to_numpy()


def choose(inner, source, target, models):
    """Improve inner balanced recall, require evidence from two independent objects."""
    true = inner.material.to_numpy()
    object_ids = inner.object_id.to_numpy()
    counts = inner.material.value_counts()
    weight = 1 / inner.material.map(counts).to_numpy()
    best, details, best_rank = None, {"reason": "no robust gain on inner CV"}, (0, 0, 0)
    zones = (False, True) if source == "plastic" else (False,)
    for predictor in models:
        for threshold in THRESHOLDS:
            for mid in zones:
                rule = Rule(predictor, threshold, mid)
                update = mask(inner, rule, source)
                fixed = update & (true == target)
                broken = update & (true == source)
                gain = fixed * weight - broken * weight
                # A single physical object must not explain the full gain.
                by_object = [gain[object_ids == o].sum() for o in set(object_ids)]
                worst_delete = float(gain.sum() - max(by_object, default=0))
                rank = (round(float(gain.sum()), 12), round(worst_delete, 12), -int(update.sum()))
                if sum(x > 1e-12 for x in by_object) >= 2 and worst_delete > 1e-12 and rank > best_rank:
                    best, best_rank = rule, rank
                    details = {"fixes": int(fixed.sum()), "breaks": int(broken.sum()),
                               "balanced_delta_pair": float(gain.sum() / 2),
                               "worst_leave_one_object_delta_pair": worst_delete / 2}
    return best, details


def fit_inner(raw, tab, bank, flipped, cols):
    parts = []
    predictors = ("cat_depth", "cat_rgb", "cat_physical", "lr_physical")
    for k, (fit, hold) in enumerate(joint.grouped_folds(raw, 3), 1):
        a = tab.loc[tab.index.intersection(fit.index)]
        b = tab.loc[tab.index.intersection(hold.index)]
        b = b.loc[b.source.eq("real")]
        if not len(b):
            continue
        if set(a.object_id) & set(b.object_id) or set(fit.image_id) & set(b.image_id):
            raise ValueError("Inner overlap")
        p = predict_pair(a, b, cols)
        views = [x[b.index] for x in [*bank, *flipped]]
        base, p_plastic = baseline_prob(fit, views, bank)
        parts.append(b[["material", "object_id"]].assign(base=base, p_plastic=p_plastic, inner_fold=k, **p))
    inner = pd.concat(parts, verify_integrity=True)
    expected = tab.loc[tab.index.intersection(raw.index)].loc[lambda x: x.source.eq("real")]
    if set(inner.index) != set(expected.index):
        raise ValueError("Missing inner OOF paired annotations")
    # Both rules look at the frozen label; this cannot flip an object twice.
    selected = {f"{a}->{b}": choose(inner, a, b, predictors) for a, b in DIRECTIONS}
    return selected, inner


def decisions(rows, rules):
    material = rows.base.to_numpy(copy=True)
    for source, target in DIRECTIONS:
        material[mask(rows, rules[f"{source}->{target}"][0], source)] = target
    return material


def baseline_donors(anchors, base):
    """Pass the original detection index through each added l960 hypothesis."""
    from export_joint04 import tail
    from l960_anchor import additions
    from yolo960_ab import fuse_frame

    spec = json.loads((CACHE / "mask_index.json").read_text())["folds"]
    sources = json.loads((CACHE / "per_model_predictions.json").read_text())
    index = {i: f["fold"] for f in spec for i in f["frames"]}
    result = {}
    for fid, old in anchors.items():
        raw = sources[fid][f"yolo11l_fold_{index[fid]:02d}"]
        near = joint.metric._parse(fuse_frame([[dict(p, tail=tail(p)) for p in raw]], [1.], ()), True, fid)
        trace = list(enumerate(old)) + [(j, p) for p, j in additions(old, near)]
        for (j, p), q in zip(trace, base[fid], strict=True):
            if (list(p["box"]) != q["box"] or any(p[k] != q[k] for k in ("score", "form"))
                    or any(old[j][k] != q[k] for k in ("take", "material"))):
                raise ValueError(f"{fid}: baseline changed outside volume")
        result[fid] = [j for j, p in trace]
    return result


def evaluate(gt, original, proposal):
    metric = joint.metric
    a, b = (metric.components(gt, {i: pred[i] for i in gt}) for pred in (original, proposal))
    if any(a[k] != b[k] for k in ("D", "V", "S")):
        raise ValueError("A material-only experiment changed D/V/S")
    corrected = ruined = 0
    counts = {m: {"before": 0, "after": 0, "n": 0} for m in PAIR}
    for i, obs in gt.items():
        for g, p, q in zip(obs, metric._match(obs, original[i]), metric._match(obs, proposal[i]), strict=True):
            m = g["material"]
            was = bool(p and p["material"] == m)
            now = bool(q and q["material"] == m)
            corrected += int(now and not was)
            ruined += int(was and not now)
            if m in PAIR:
                counts[m]["before"] += int(was)
                counts[m]["after"] += int(now)
                counts[m]["n"] += 1
    return {"base": a, "candidate": b, "delta": b["Score"] - a["Score"], "fixed": corrected,
            "broken": ruined, "material_hits": counts}


def run(out, volume_dir):
    index = json.loads((CACHE / "mask_index.json").read_text())
    manifest = WORK / "detector_folds_v1.json"
    if file_sha(manifest) != index["fold_manifest_sha256"]:
        raise ValueError("OOF manifest mismatch")
    specs = json.loads(manifest.read_text())["folds"]
    anchors = json.loads((CACHE / "predictions.json").read_text())["n640_rescue"]
    volumes_path = volume_dir / "bulk_y_bottom_oof.json"
    old = json.loads(volumes_path.read_text())
    donors = baseline_donors(anchors, old)
    train, _ = joint.read_data()
    ext, hold = features(specs, index, anchors, extended=True)
    cols = columns(hold)
    tab = (train.reset_index(names="train_ix")[["train_ix", "image_id", "ann_id", "object_id", "material", "source"]]
           .merge(ext, on=["image_id", "ann_id"], validate="one_to_one").set_index("train_ix"))
    bank, val_views = load_embeddings(train, hold)
    with np.load(WORK / "astra_joint04_train_flip.npz") as saved:
        flipped = [saved[k] for k in ("d", "c")]
    gt = {r.image_id: joint.metric._parse(r.objects, False, r.image_id) for r in joint.metric.labels_to_solution(
        joint.DATA / "labels/real_train.json").itertuples(index=False)}
    if not np.isclose(joint.metric.components(gt, old)["Score"], 72.7802, rtol=0, atol=5e-5):
        raise ValueError("Expected frozen bulk-y baseline")
    proposed, reports = {}, {}
    with threadpool_limits(4):
        for spec in specs:
            k = spec["fold"]
            raw = train.loc[train.image_id.isin(spec["train_image_ids"])]
            if set(raw.object_id) != set(spec["train_object_ids"]) or set(raw.object_id) & set(spec["val_object_ids"]):
                raise ValueError(f"Outer overlap at fold {k}")
            rules, inner = fit_inner(raw, tab, bank, flipped, cols)
            h = hold.loc[hold.fold.eq(k)]
            fitted = tab.loc[tab.index.intersection(raw.index)]
            base, conf = baseline_prob(raw, [v[h.index] for v in val_views], bank)
            expected = [anchors[i][j]["material"] for i, j in zip(h.image_id, h.idx, strict=True)]
            if list(base) != expected:
                raise ValueError(f"Fold {k} does not reproduce the frozen material head")
            rows = h.assign(base=base, p_plastic=conf, **predict_pair(fitted, h, cols))
            new = decisions(rows, rules)
            per_donor = dict(zip(zip(h.image_id, h.idx, strict=True), map(str, new), strict=True))
            ids = spec["val_real_image_ids"]
            for i in ids:
                proposed[i] = [dict(p, material=per_donor[i, j]) for p, j in zip(old[i], donors[i], strict=True)]
            measure = evaluate({i: gt[i] for i in ids}, old, proposed)
            measure["rules"] = {key: {"rule": asdict(rule) if rule else None, **detail}
                                for key, (rule, detail) in rules.items()}
            measure["inner_paired_views"] = len(inner)
            measure["switched_to_metal"] = int(np.sum((base == "plastic") & (new == "metal")))
            measure["switched_to_plastic"] = int(np.sum((base == "metal") & (new == "plastic")))
            reports[k] = measure
            print(f"Fold {k}: {measure['base']['Score']:.4f} -> {measure['candidate']['Score']:.4f} "
                  f"({measure['delta']:+.4f}), fixes {measure['fixed']}, breaks {measure['broken']}, "
                  f"rules {measure['rules']}", flush=True)
    pooled = evaluate(gt, old, proposed)
    eligible = pooled["delta"] > 1e-9 and all(f["delta"] >= -1e-9 for f in reports.values())
    result = {"baseline_sha256": file_sha(volumes_path), "pooled": pooled, "folds": reports,
              "eligible_for_test_export": eligible}
    out.mkdir(parents=True, exist_ok=False)
    with (out / "report.json").open("x") as file:
        json.dump(result, file, indent=2)
    print(f"Pooled: {pooled['base']['Score']:.4f} -> {pooled['candidate']['Score']:.4f} "
          f"({pooled['delta']:+.4f}); eligible for test export: {eligible}", flush=True)
    return eligible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint3_metal_plastic_nested")
    parser.add_argument("--volume-run", type=Path, default=VOLUME_RUN)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    cv2.setNumThreads(1)
    run(args.out, args.volume_run)

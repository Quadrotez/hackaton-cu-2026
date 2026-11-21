"""Compare class-aware and class-agnostic fusion on verified grouped OOF and cached test predictions."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from attribute_postprocess import AttributePolicy, DEFAULT_MODEL, TakeRule
from class_agnostic_wbf import IOU, fuse_frame as agnostic
from evaluate_attributes import fit_prior, joint, read_inputs, rescue_selection, transform
from evaluate_yolo960_oof import verified_group
from export_joint04 import Joint04Exporter, tail
from submission_fusion import iou, load
from yolo960_ab import CONFIG, ROOT, file_sha, fuse_frame as aware, read_config, write_csv

from build_detector_folds import build
from submission_tools import validate_csv


WORK = ROOT / "work"
OOF_CACHE = WORK / "oof_cache_960"
ANCHOR = "n640"
INCUMBENT = "n640_rescue"
BASE = "class_aware"
BASE_RESCUE = "class_aware_rescue"
WBF = "class_agnostic_wbf"
NMS = "class_agnostic_wbf_nms"
RESCUE = "class_agnostic_wbf_nms_rescue"
MODES = (ANCHOR, INCUMBENT, BASE, BASE_RESCUE, WBF, NMS, RESCUE)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def diagnostics(pred):
    counts = Counter(objects=sum(map(len, pred.values())), empty=sum(not a for a in pred.values()))
    for objects in pred.values():
        for j, a in enumerate(objects):
            for b in objects[j + 1:]:
                overlap = iou(a["box"], b["box"])
                if overlap >= IOU:
                    counts["pairs_iou_055"] += 1
                if overlap >= .5 and a["form"] != b["form"] and min(a["score"], b["score"]) >= .25:
                    counts["opposite_form_pairs_iou_050_conf_025"] += 1
    return {k: counts[k] for k in
            ("objects", "empty", "pairs_iou_055", "opposite_form_pairs_iou_050_conf_025")}


def measured(gt, pred):
    metric = joint.metric
    return {**metric.components(gt, pred), **diagnostics(pred),
            "misses": sum(p is None for i, g in gt.items() for p in metric._match(g, pred[i])),
            "predictions_on_empty": sum(len(pred[i]) for i, g in gt.items() if not g)}


def strict_models(manifest_path, cv_path):
    _, _, models = read_config(CONFIG, False)
    accepted, excluded = [], {"s640": "legacy checkpoint has no compatible five-fold training provenance"}
    for g in ("m", "l"):
        group = [m for m in models if m.group == g]
        reason = verified_group(group, manifest_path, cv_path)
        if reason:
            excluded[g + "960"] = reason
        else:
            accepted.extend(group)
    if not accepted:
        raise ValueError(f"No verified 960 models: {excluded}")
    return accepted, excluded


def stage_archive(exporter, model, sha, frames, rgb_sha, *, allow_inference):
    raw = exporter.raw
    key = dict(exporter.key, checkpoint_sha256=sha, rgb_sha256=rgb_sha)
    name = f"{raw.as_key(key)}.npz"
    prior, staged = WORK / "detector_raw_cache_v1" / name, exporter.cache / name
    if not staged.is_file() and prior.is_file():
        staged.hardlink_to(prior)
    if not staged.is_file() and not allow_inference:
        raise FileNotFoundError(f"OOF cache missing for {model.tag}; rerun with --allow-inference: {staged}")
    return staged


def run_oof(args):
    manifest_path, cv_path = WORK / "detector_folds_v1.json", WORK / "cv_folds.json"
    if build() != json.loads(manifest_path.read_text()):
        raise ValueError("Cannot reproduce the detector manifest from cv_folds.json and current labels")
    models, excluded = strict_models(manifest_path, cv_path)
    shas = {m.tag: file_sha(m.checkpoint) for m in models}
    if len(set(shas.values())) != len(shas):
        raise ValueError("Duplicate checkpoints in strict OOF")
    print(f"Verified OOF: n640 + {', '.join(sorted({m.group + '960' for m in models}))}; excluded: {excluded}",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    data, views, manifests, nano, old_gt, hashes = read_inputs()
    gt = {r.image_id: joint.metric._parse(r.objects, False, r.image_id) for r in
          joint.metric.labels_to_solution(joint.DATA / "labels" / "real_train.json").itertuples(index=False)}
    if set(gt) != set(old_gt) or len(gt) != 263 or sum(map(len, gt.values())) != 214:
        raise ValueError("Unexpected official label coverage")
    fit_report = json.loads((DEFAULT_MODEL.parent / "report.json").read_text())
    if any(fit_report.get(k) != v for k, v in hashes.items()):
        raise ValueError("Rescue policies do not match the strict OOF split")
    inner = joint.pd.read_csv(DEFAULT_MODEL.parent / "inner_oof.csv")
    deploy = AttributePolicy.load(DEFAULT_MODEL)
    _, cols = joint.read_data()
    real = {i: joint.DATA / im["file_name"] for src, i, im, _ in joint.load_frames() if src == "real"}
    exporter = Joint04Exporter(ROOT, None, args.device, args.batch)
    exporter.cache = OOF_CACHE
    exporter.cache.mkdir(exist_ok=True)
    folds, index, all_inputs = [], [], {}
    all_pred = {k: {} for k in MODES}
    metric = joint.metric
    for spec in manifests:
        f = spec["fold"]
        ids = spec["val_real_image_ids"]
        frames = [(i, real[i]) for i in ids]
        train = data.loc[data.image_id.isin(spec["train_image_ids"])]
        train_objects = set(train.object_id)
        if (train_objects != set(spec["train_object_ids"]) or train_objects & set(spec["val_object_ids"])
                or set(train.image_id) & set(ids) or set(ids) & set(all_inputs)):
            raise ValueError(f"Fold {f}: overlapping or inconsistent train/validation split")
        on_fold = inner.loc[inner.outer_fold.eq(f)]
        if set(on_fold.image_id) != set(train.loc[train.source.eq("real"), "image_id"]):
            raise ValueError(f"Fold {f}: wrong effective training images in rescue policy")
        rule = rescue_selection(on_fold)
        if rule != TakeRule(**next(e["rescue_small"] for e in fit_report["folds"] if e["fold"] == f)):
            raise ValueError(f"Fold {f}: rescue rule does not match frozen calibration")
        policy = replace(deploy, prior=fit_prior(train), rescue_small=rule)
        rgb_sha = exporter.frame_key(frames)
        fold_models = [m for m in models if m.fold == f]
        caches = [stage_archive(exporter, m, shas[m.tag], frames, rgb_sha,
                                allow_inference=args.allow_inference) for m in fold_models]
        with joint.threadpool_limits(4):
            heads = joint.fit(train, cols, views)
        sources = [{i: [dict(d, mask_index=j) for j, d in enumerate(nano[i])] for i in ids}]
        sources.extend(exporter.predictions(m, shas[m.tag], frames, heads, rgb_sha,
                                            require_cache=not args.allow_inference, with_mask_indices=True)
                       for m in fold_models)
        # Equal group weights; the verified n640 here is an active replacement
        # for the unavailable-in-OOF active s640 slot, not A/B's support-only nano.
        weights = [1., *(m.weight for m in fold_models)]
        tags = [f"n640_fold_{f:02d}", *(m.tag for m in fold_models)]
        n_dir = WORK / "detector_oof_v1" / f"fold_{f:02d}" / "evaluation"
        n_mask, n_det = (n_dir / name for name in ("base_masks.npz", "base_detections.jsonl"))
        index.append({"fold": f, "frames": ids, "sources":
                      {tags[0]: {"masks": str(n_mask.relative_to(ROOT)), "masks_sha256": file_sha(n_mask),
                                 "detections": str(n_det.relative_to(ROOT)), "detections_sha256": file_sha(n_det)},
                       **{m.tag: {"archive": str(p.relative_to(ROOT)), "archive_sha256": file_sha(p),
                                  "checkpoint_sha256": shas[m.tag]} for m, p in zip(fold_models, caches, strict=True)}}})
        for i in ids:
            all_inputs[i] = {tag: p[i] for tag, p in zip(tags, sources, strict=True)}
            dets = [[dict(d, tail=tail(d)) for d in p[i]] for p in sources]
            original = aware(dets, weights, ())
            with_nms = agnostic(dets, weights)
            variants = {
                ANCHOR: nano[i],
                INCUMBENT: transform(policy, "rescue-small", {i: nano[i]})[i],
                BASE: metric._parse(original, True, i),
                BASE_RESCUE: metric._parse(policy.prediction("rescue-small", original), True, i),
                WBF: metric._parse(agnostic(dets, weights, suppress=False), True, i),
                NMS: metric._parse(with_nms, True, i),
                RESCUE: metric._parse(policy.prediction("rescue-small", with_nms), True, i),
            }
            for k, p in variants.items():
                all_pred[k][i] = p
        g = {i: gt[i] for i in ids}
        scores = {k: measured(g, {i: p[i] for i in ids}) for k, p in all_pred.items()}
        folds.append({"fold": f, "frames": len(g), "models": tags, "rescue_rule": vars(rule), "scores": scores})
        print(f"Fold {f}: " + " | ".join(f"{k} {c['Score']:.4f}" for k, c in scores.items()), flush=True)

    if set(all_inputs) != set(gt):
        raise ValueError("OOF does not cover all 263 real frames")
    pooled = {k: measured(gt, p) for k, p in all_pred.items()}
    if pooled[NMS]["pairs_iou_055"] or any(pooled[a][k] != pooled[b][k]
                                            for a, b in ((BASE, BASE_RESCUE), (NMS, RESCUE))
                                            for k in ("D", "M", "V")):
        raise ValueError("NMS overlap or rescue invariant violated")
    comparisons = {}
    for old, new in ((ANCHOR, BASE), (BASE, BASE_RESCUE), (INCUMBENT, BASE_RESCUE),
                     (BASE, WBF), (WBF, NMS), (BASE, NMS),
                     (NMS, RESCUE), (INCUMBENT, RESCUE), (BASE_RESCUE, RESCUE)):
        delta = [f["scores"][new]["Score"] - f["scores"][old]["Score"] for f in folds]
        total = pooled[new]["Score"] - pooled[old]["Score"]
        comparisons[f"{new} vs {old}"] = {"pooled_delta": total, "fold_deltas": delta,
                                           "pass": total > 1e-9 and min(delta) >= -1e-9}
    index_path = args.out / "mask_index.json"
    write_json(index_path, {**hashes, "cv_folds_sha256": file_sha(cv_path), "folds": index})
    write_json(args.out / "per_model_predictions.json", all_inputs)
    write_json(args.out / "predictions.json", all_pred)
    write_json(args.out / "report.json", {**hashes, "cv_folds_sha256": file_sha(cv_path), "frames": len(gt),
               "requested_full_ensemble_oof_available": not excluded, "excluded": excluded,
               "iou": IOU, "pooled": pooled, "folds": folds, "comparisons": comparisons})
    with (args.out / "metrics.csv").open("x", newline="", encoding="utf-8") as out:
        columns = ("fold", "frames", "variant", "D", "M", "V", "S", "Score", "misses",
                   "objects", "empty", "predictions_on_empty", "pairs_iou_055",
                   "opposite_form_pairs_iou_050_conf_025")
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for row in [*folds, {"fold": "pooled", "frames": len(gt), "scores": pooled}]:
            for name, score in row["scores"].items():
                writer.writerow({"fold": row["fold"], "frames": row["frames"], "variant": name, **score})
    print("Pooled: " + " | ".join(f"{k} {c['Score']:.4f}" for k, c in pooled.items()), flush=True)


def run_test(out):
    primary, _, models = read_config(CONFIG, False)
    source_dir = WORK / "submission_yolo960_A_attr"
    paths = [primary, *(source_dir / "per_model" / f"{m.tag}.csv" for m in models)]
    sources = []
    for p in paths:
        validate_csv(p)
        sources.append(load(p))
    if len(sources[0]) != 137 or any(s.keys() != sources[0].keys() for s in sources):
        raise ValueError("Expected each model to cover the 137 real test frames")
    weights = [1., *(m.weight for m in models)]
    out.mkdir(parents=True, exist_ok=False)
    rows = [(i, agnostic([s[i] for s in sources], weights)) for i in sorted(sources[0])]
    paths_out = [out / "submission_agnostic_960_s640.csv", out / "submission_agnostic_960_s640_rescue_small.csv"]
    policy = AttributePolicy.load(DEFAULT_MODEL)
    if policy.resolve("best") != "rescue-small":
        raise ValueError("Unexpected default postprocessor")
    adjusted = list(policy.rows("rescue-small", rows))
    flips = 0
    for (fid, a), (other, b) in zip(rows, adjusted, strict=True):
        if fid != other:
            raise ValueError("Rescue changed frame IDs")
        for x, y in zip(a.split(";"), b.split(";"), strict=True):
            sx, sy = x.split(), y.split()
            if sx[:-1] != sy[:-1] or (sx[-1], sy[-1]) not in (("0", "1"), (sx[-1], sx[-1])):
                raise ValueError("Rescue changed fields other than 0 → 1 take")
            flips += sx[-1] != sy[-1]
    for path, r in zip(paths_out, (rows, adjusted), strict=True):
        write_csv(path, r, validate_csv)
    baseline = diagnostics(validate_csv(source_dir / "submission_A_post.csv"))
    candidate = diagnostics(validate_csv(paths_out[1]))
    if candidate["pairs_iou_055"]:
        raise ValueError("Overlapping boxes survived final NMS")
    write_json(out / "report.json", {"strict_oof": False, "frames": len(rows), "iou": IOU, "take_flips": flips,
               "A_post": baseline, "candidate": candidate,
               "inputs": {str(p.relative_to(ROOT)): file_sha(p) for p in paths}})
    print(f"Test A → agnostic: {baseline} → {candidate}; {flips} rescue flips", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("oof", "test"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--allow-inference", action="store_true", help="Infer on cache misses only for verified checkpoints")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.stage == "test":
        run_test(args.out or WORK / "sprint1_agnostic_test")
    else:
        args.out = args.out or WORK / "sprint1_agnostic_oof"
        run_oof(args)


if __name__ == "__main__":
    main()

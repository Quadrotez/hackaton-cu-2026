"""Score only fold-matched YOLO960 checkpoints with verified split provenance."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from attribute_postprocess import DEFAULT_MODEL, AttributePolicy, TakeRule
from evaluate_attributes import fit_prior, joint, metrics, read_inputs, rescue_selection
from export_joint04 import Joint04Exporter, tail
from yolo960_ab import CONFIG, ROOT, file_sha, fuse_frame, read_config


def verified_group(models, manifest, cv):
    """The exported run and its manifests must identify all five exact fold paths."""
    root = models[0].checkpoint.parents[2]
    run_path = root / "run.json"
    published = root / manifest.name
    published_cv = root / cv.name
    if not all(p.is_file() for p in (run_path, published, published_cv)):
        return "missing run.json or exported split manifests"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if file_sha(published) != file_sha(manifest) or run.get("manifest_sha256") != file_sha(manifest):
        return "detector split manifest mismatch"
    if file_sha(published_cv) != file_sha(cv) or run.get("cv_folds_sha256") != file_sha(cv):
        return "object split manifest mismatch"
    completed = run.get("completed_folds", {})
    if set(completed) != {str(f) for f in range(1, 6)} or run.get("imgsz") != 960:
        return "invalid training size or incomplete folds"
    for m in models:
        checkpoint = completed[str(m.fold)]
        if (checkpoint.get("checkpoint") != m.checkpoint.relative_to(root).as_posix()
                or checkpoint.get("epochs") != run.get("epochs") or not m.checkpoint.is_file()):
            return f"could not verify completed checkpoint {m.tag}"
    return None


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    work = ROOT / "work"
    _, _, models = read_config(args.config, False)
    verified, skipped = [], {"s640": "legacy test checkpoint has no compatible object-fold manifest"}
    manifest, cv = work / "detector_folds_v1.json", work / "cv_folds.json"
    for group in ("m", "l"):
        entries = [m for m in models if m.group == group]
        reason = verified_group(entries, manifest, cv)
        if reason:
            skipped[group] = reason
        else:
            verified.extend(entries)
    if not verified:
        raise ValueError(f"No verifiable YOLO960 OOF source: {skipped}")
    print(f"Strict OOF groups: {sorted({m.group for m in verified})}; excluded: {skipped}", flush=True)
    shas = {m.tag: file_sha(m.checkpoint) for m in verified}
    if len(set(shas.values())) != len(verified):
        raise ValueError("Duplicate checkpoint SHA-256")

    data, views, manifests, old_oof, gt, hashes = read_inputs()
    _, cols = joint.read_data()
    fit_report = json.loads((args.attributes / "report.json").read_text(encoding="utf-8"))
    if any(fit_report[k] != v for k, v in hashes.items()):
        raise ValueError("Rescue thresholds use a different OOF manifest or baseline")
    inner = joint.pd.read_csv(args.attributes / "inner_oof.csv")
    deploy = AttributePolicy.load(args.attributes / "policy.json")
    real = {i: joint.DATA / im["file_name"] for src, i, im, _ in joint.load_frames() if src == "real"}
    exporter = Joint04Exporter(ROOT, None, args.device, args.batch)
    all_rows, results = {}, []
    for spec in manifests:
        f = spec["fold"]
        frames = [(i, real[i]) for i in spec["val_real_image_ids"]]
        ids = [i for i, _ in frames]
        train = data.loc[data.image_id.isin(spec["train_image_ids"])]
        assert set(train.object_id) == set(spec["train_object_ids"])
        assert not set(train.object_id) & set(spec["val_object_ids"])
        assert not set(train.image_id) & set(ids)
        inner_rows = inner.loc[inner.outer_fold.eq(f)]
        assert set(inner_rows.image_id) == set(train.loc[train.source.eq("real"), "image_id"])
        rescue = rescue_selection(inner_rows)
        recorded = next(d["rescue_small"] for d in fit_report["folds"] if d["fold"] == f)
        assert rescue == TakeRule(**recorded)
        policy = replace(deploy, prior=fit_prior(train), rescue_small=rescue)
        with joint.threadpool_limits(4):
            heads = joint.fit(train, cols, views)
        fold_models = [m for m in verified if m.fold == f]
        rgb_sha = exporter.frame_key(frames)
        preds = [exporter.predictions(m, shas[m.tag], frames, heads, rgb_sha) for m in fold_models]
        weights = [m.weight for m in fold_models]
        rows = {}
        for i in ids:
            candidates = [[dict(p, tail=tail(p)) for p in src[i]] for src in preds]
            rows[i] = fuse_frame(candidates, weights, (), primary_index=None)
        g = {i: gt[i] for i in ids}
        baseline = {i: joint.metric._parse(row, True, i) for i, row in rows.items()}
        rescued = {i: joint.metric._parse(policy.prediction("rescue-small", row), True, i) for i, row in rows.items()}
        assert not set(all_rows) & set(ids)
        all_rows.update({i: {"off": baseline[i], "rescue_small": rescued[i]} for i in ids})
        b, r = metrics(g, baseline), metrics(g, rescued)
        results.append({"fold": f, "frames": len(ids), "models": [m.tag for m in fold_models],
                        "off": b, "rescue_small": r, "rescue_cut": vars(rescue)})
        print(f"Fold {f}: {b['Score']:.4f} -> {r['Score']:.4f} ({r['Score'] - b['Score']:+.4f})", flush=True)

    assert set(gt) == set(all_rows) and len(all_rows) == 263
    off = metrics(gt, {i: p["off"] for i, p in all_rows.items()})
    rescue = metrics(gt, {i: p["rescue_small"] for i, p in all_rows.items()})
    assert all(off[k] == rescue[k] for k in ("D", "M", "V"))
    prior = {"off": metrics(gt, old_oof), "rescue_small": fit_report["pooled"]["rescue-small"]}
    report = {**hashes, "frames": len(gt), "checkpoints_sha256": shas, "excluded": skipped,
              "device": args.device, "batch": args.batch, "old_pooled": prior,
              "pooled": {"off": off, "rescue_small": rescue}, "folds": results}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in (("report.json", report), ("predictions.json", all_rows)):
        (args.out / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pooled: {off['Score']:.4f} -> {rescue['Score']:.4f}. Old: "
          f"{prior['off']['Score']:.4f} -> {prior['rescue_small']['Score']:.4f}. {args.out / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--attributes", type=Path, default=DEFAULT_MODEL.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "yolo960_strict_oof")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    run(args)


if __name__ == "__main__":
    main()

"""Fold-matched raw caches -> Joint04 + gated volume -> OOF and test candidates."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from torchvision import transforms

from fusion import sprint5_crop_adapter as io
from fusion.instance_take_ab import apply_mixture, density_prior
from fusion.attribute_postprocess import TakeRule
from fusion.export_joint04 import Joint04Exporter
import astra_joint04 as joint
import infer_detector_wbf as raw
import score


W = io.WORK
RUN = W / "sprint10_kaggle_retrain"
LEGACY = W / "kaggle/astramis-yolo11l-960-5folds/output/astramis-cv"
A = RUN / "downloads/A-v1/astramis-clean-control-A"
B = RUN / "downloads/B-v1/astramis-targeted-bulk-B"
MIX = W / "instance_take_oof/mixture_policy.json"
HISTORICAL = W / "instance_take_oof/gregory_7b5c60e_replay"
TARGETS = ("b01", "b06", "b12", "b15", "b21")
DINO_REF = "ff78a8cc93ca262fa6406b797121152d048f8382"
FAMILIES = {"l960": ("l960",), "A": ("A",), "B": ("B",), "ensemble": ("l960", "A", "B")}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sources(fold_by_image, caches_only=False):
    """Validate all weights, fold/data binding and caches before use."""
    verify = load_module(RUN / "verify_outputs.py", "verify_control")
    a_root = A.parents[1] / "A-oof-only" / A.name if caches_only else A
    b_root = B.parents[1] / "B-oof-only" / B.name if caches_only else B
    if not caches_only:
        verify.main(A)
    manifest = io.pinned(io.FOLDS)
    a_manifest = RUN / "clean_fold_manifest.json"
    plan = io.read_json(a_manifest)
    a_run = io.read_json(a_root / "run.json")
    a_dry = io.read_json(RUN / "dry_run.json")
    b_run = io.read_json(b_root / "run.json")
    b_dry = io.read_json(RUN / "B/dry_run.json")
    b_manifest = io.read_json(b_root / "manifest.json")
    lo = io.read_json(LEGACY / "run.json")
    idx = io.read_json(W / "sprint1_agnostic_oof_v2/mask_index.json")
    assert io.sha(a_root / a_manifest.name) == io.sha(a_manifest) == a_run["clean_manifest_sha256"]
    assert a_run["arm"] == plan["arm"] and a_run["sources_sha256"] == a_dry["sources_sha256"]
    assert a_run["weights_sha256"] == plan["initialization"]["sha256"]
    assert all(a_run[k] == v for k, v in plan["training"].items())
    for name, h in plan["input_sha256"].items():
        if name not in ("fold_runner.py", "build_notebook.py"):
            path = (W / "sprint3_synth_backgrounds" / name
                    if caches_only and name == "synth_to_real_empty.json" else a_root / name)
            assert io.sha(path) == h
    assert io.sha(LEGACY / io.FOLDS.name) == lo["manifest_sha256"] == idx["fold_manifest_sha256"] == io.sha(io.FOLDS)
    assert io.sha(LEGACY / "cv_folds.json") == lo["cv_folds_sha256"] == idx["cv_folds_sha256"]
    assert b_run["arm"] == b_dry["arm"] and b_run["weights_sha256"] == a_run["weights_sha256"]
    assert all(b_run["training"][k] == v for k, v in plan["training"].items())
    assert b_run["manifest_sha256"] == io.sha(b_root / "manifest.json")
    local_b = io.read_json(RUN / "B/targeted_v2/manifest.json")
    assert all(b_manifest[k] == v for k, v in local_b.items() if k not in {"scenes", "numpy", "opencv"})
    assert b_manifest["scenes"].keys() == local_b["scenes"].keys()
    for key in local_b["scenes"]:
        assert all(b_manifest["scenes"][key][k] == local_b["scenes"][key][k]
                   for k in ("annotation", "dx", "id", "lineage", "shape"))
    ks = {str(k) for k in range(1, 6)}
    cache_plan = io.read_json(a_root / "cache_manifest.json")
    assert cache_plan["clean_manifest_sha256"] == a_run["clean_manifest_sha256"]
    a_caches = cache_plan["folds"]
    assert set(lo["completed_folds"]) == set(b_run["folds"]) == set(a_run["completed_folds"]) == set(a_caches) == ks
    archives, checkpoints = {}, {}
    for spec, clean, old in zip(manifest["folds"], plan["folds"], idx["folds"], strict=True):
        k = spec["fold"]
        assert k == clean["fold"] == old["fold"]
        ids = spec["val_real_image_ids"]
        assert ids == clean["val_real_image_ids"] == old["frames"] and all(fold_by_image[i] == k for i in ids)
        c = b_run["folds"][str(k)]
        ac = a_run["completed_folds"][str(k)]
        assert c["staging"] == b_dry["staged"][str(k)] and c["epochs"] == plan["training"]["epochs"]
        old_source = old["sources"][f"yolo11l_fold_{k:02d}"]
        relative = f"fold_{k:02d}/weights/last.pt"
        assert lo["completed_folds"][str(k)]["checkpoint"] == relative
        assert ac["checkpoint"] == relative and ac["epochs"] == plan["training"]["epochs"]
        assert ac["staged_label_sha256"] == a_dry["folds"][k-1]["staged_labels_sha256"]
        for tag, root, meta, expected_weight, filepath in (
            ("A", A, a_caches[str(k)], ac["checkpoint_sha256"], a_root / a_caches[str(k)]["archive"]),
            ("B", B, c["oof"], c["checkpoint_sha256"], b_root / c["oof"]["archive"]),
            ("l960", LEGACY, old_source, old_source["checkpoint_sha256"], io.ROOT / old_source["archive"]),
        ):
            pt = root / relative
            if not caches_only or tag == "l960":
                assert io.sha(pt) == expected_weight
            assert io.sha(filepath) == meta["archive_sha256"]
            with np.load(filepath, allow_pickle=False) as z:
                h = json.loads(z["raw_header"].item())
                assert set(z.files) == set(ids) | {"raw_header"}
                assert [r["image_id"] for r in h["frames"]] == ids
                assert h["key"]["checkpoint_sha256"] == expected_weight and h["model"]["imgsz"] == 960
                if tag in ("A", "B"):
                    expected_key = dict(plan["raw_oof_settings"], fold=k, checkpoint_sha256=expected_weight,
                                        clean_manifest_sha256=b_run["manifest_sha256"] if tag == "B" else
                                        a_run["clean_manifest_sha256"], rgb_sha256=clean["val_real_rgb_sha256"],
                                        ultralytics=h["key"]["ultralytics"], torch=h["key"]["torch"])
                    assert h["key"] == meta["key"] == expected_key and meta["key_sha256"] == verify.key(expected_key)
                    shapes = [[r["image_id"], r["shape"], list(z[r["image_id"]].shape)] for r in h["frames"]]
                    assert verify.key(shapes) == meta["shape_sha256"] and meta["frames"] == ids
                    assert sum(len(r["detections"]) for r in h["frames"]) == meta["detections"]
            archives[tag, k] = filepath
            checkpoints[tag, k] = (pt, expected_weight)
    assert len({p[1] for p in checkpoints.values()}) == 15
    return archives, checkpoints


def decode(z, row):
    hw, ds = raw.cached_detections(z, row)
    return hw, [d for d in ds if d.score >= raw.CONF and raw.usable_mask(d.mask, d.box)]


def cache_rowmap(z):
    return {r["image_id"]: r for r in json.loads(z["raw_header"].item())["frames"]}


def proposals(paths, groups, frames):
    """One cached model per family on OOF; five per family on unlabeled test."""
    with ExitStack() as context:
        archives = {key: context.enter_context(np.load(path, allow_pickle=False)) for key, path in paths.items()}
        maps = {key: cache_rowmap(z) for key, z in archives.items()}
        for fid, path in frames:
            decoded = {key: decode(z, maps[key][fid]) for key, z in archives.items()}
            assert len({hw for hw, _ in decoded.values()}) == 1
            hw = next(iter(decoded.values()))[0]
            out = {}
            for tag, keys in groups.items():
                ds = [decoded[k][1] for k in keys]
                if len(ds) == 1:
                    out[tag] = [(dict(box=d.box, score=d.score, form=d.form), d.mask) for d in ds[0]]
                else:
                    out[tag] = [(dict(box=d["box"], score=d["score"], form=d["form"]), mask)
                                for d, mask in raw.fuse(ds, hw)]
            yield fid, path, out


class Encoders:
    def __init__(self, device):
        self.device = device
        self.full, self.clip, self.clip_t = joint.encoders()
        self.full.m.to(device)
        self.clip.to(device)
        self.greg_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                          transforms.Normalize((.5,)*3, (.5,)*3)])

    @torch.inference_mode()
    def encode(self, crops, mode):
        encoder, transform = {
            "greg": (self.full.m, self.greg_t),
            "d": (self.full.m, self.full.t),
            "c": (self.clip.encode_image, self.clip_t),
        }[mode]
        out = []
        for j in range(0, len(crops), 16):
            ims = [transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops[j:j+16]]
            out.append(encoder(torch.stack(ims).to(self.device)).float().cpu().numpy())
        return np.concatenate(out)

    def views(self, crops):
        if not crops:
            return [np.empty((0, n), dtype=np.float32) for n in (768, 512, 768, 512)]
        mirror = [cv2.flip(c, 1) for c in crops]
        return [normalize(self.encode(c, mode)) for c in (crops, mirror) for mode in ("d", "c")]


def crop_rgb(color, box):
    """Exactly the notebook RGB path: padded bbox -> 128px -> RGB at encoding."""
    h, w = color.shape[:2]
    x, y, r, b = box
    bw, bh = r-x, b-y
    l, t, r, b = max(0, math.floor(x-.1*bw)), max(0, math.floor(y-.1*bh)), min(w, math.ceil(r+.1*bw)), min(h, math.ceil(b+.1*bh))
    assert l < r and t < b
    return cv2.resize(color[t:b, l:r], (128, 128),
                      interpolation=cv2.INTER_AREA if max(r-l, b-t) > 128 else cv2.INTER_LINEAR)


def greg_training(frame_paths, enc):
    labels = io.read_json(io.LABELS / "real_train.json")
    imgs = {x["id"]: Path(x["file_name"]).parent.name for x in labels["images"]}
    rows, crops = [], []
    for a in labels["annotations"]:
        at = a["attributes"]
        i = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        crops.append(crop_rgb(raw.read_image(frame_paths[i]), (x, y, x+w, y+h)))
        rows.append((i, at["object_id"], at["form"], at["material"]))
    return pd.DataFrame(rows, columns=("image_id", "object_id", "form", "material")), enc.encode(crops, "greg")


def greg_heads(rows, x, train_ids):
    mask = rows.image_id.isin(train_ids)
    return tuple(LogisticRegression(C=1, class_weight="balanced", max_iter=2000).fit(x[mask], rows.loc[mask, c])
                 for c in ("form", "material"))


def greg_apply(pred, frame_paths, enc, heads):
    slots, crops = [], []
    for i, ds in pred.items():
        if not ds:
            continue
        color = raw.read_image(frame_paths[i])
        for j, d in enumerate(ds):
            slots.append((i, j))
            crops.append(crop_rgb(color, d["box"]))
    if not slots:
        return pred
    x = enc.encode(crops, "greg")
    probs = np.zeros((len(x), 1+len(score.MATERIALS)))
    for f, m in heads:
        probs[:, 0] += f.predict_proba(x)[:, list(f.classes_).index("bulk")] / len(heads)
        pm = m.predict_proba(x)
        for j, name in enumerate(score.MATERIALS, 1):
            if name in m.classes_:
                probs[:, j] += pm[:, list(m.classes_).index(name)] / len(heads)
    # Historical probability CSV rounded to eight decimals before the adapter.
    probs = np.round(probs, 8)
    return io.apply(pred, {ij: (float(p[0]), p[1:].tolist()) for ij, p in zip(slots, probs, strict=True)}, "both")


def infer_attributes(frames, paths, groups, enc, heads, prior, rescue, volume_scale, geo):
    dets = joint.Detections()
    masks_ml, crop_ids = {}, {}
    for i, path, tagged in proposals(paths, groups, frames):
        color = raw.read_image(path)
        # Reuse exact feature/crop extraction when a singleton kept its original box and donor.
        by_content = {}
        for tag, pairs in tagged.items():
            ordered, missing = [], []
            for obj, mask in pairs:
                key = (tuple(obj["box"]), obj["form"], id(mask))
                ordered.append((key, obj))
                if key not in by_content and all(k != key for k, _, _ in missing):
                    missing.append((key, obj, mask))
            proxy_tag = f"{tag}_new"
            dets.frame(proxy_tag, i, path.parent, [(obj, mask) for _, obj, mask in missing], color=color)
            for (key, obj, mask), got in zip(missing, dets.found.pop((proxy_tag, i)), strict=True):
                idx = got["idx"]
                by_content[key] = idx
                if obj["form"] == "bulk":
                    geo.frame(path, mask.shape, "final_predictions")
                    masks_ml[idx] = geo.measure(mask)["ml"]
            dets.found[tag, i] = [dict(obj, idx=by_content[key]) for key, obj in ordered]
            crop_ids[tag, i] = [by_content[key] for key, _ in ordered]
    views = enc.views(dets.crops)
    attr = joint.apply(heads, pd.DataFrame(dets.rows), views) if dets.rows else pd.DataFrame()
    outputs = {}
    for tag in groups:
        preds = dets.output(tag, attr)
        for i, ds in preds.items():
            for d, j in zip(ds, crop_ids[tag, i], strict=True):
                if not d["take"] and 0 < d["volume"] <= rescue.small_ml:
                    d["take"] = int(prior.light_probability(d["form"], d["material"], d["volume"]) >= rescue.cut)
                ml = masks_ml.get(j)
                if ml is not None and d["volume"] > 0:
                    limit = math.log(geo_module().CONFIG["cap_factor"])
                    diff = geo_module().CONFIG["log_blend"] * (math.log(ml / d["volume"]) + volume_scale)
                    d["volume"] *= math.exp(max(-limit, min(limit, diff)))
        outputs[tag] = preds
    return outputs, views, crop_ids


_geo = None


def geo_module():
    global _geo
    if _geo is None:
        _geo = load_module(W / "sprint4_volume_gate/run.py", "frozen_bulk_gate")
    return _geo


def add_mixture(pred, views, crop_ids, tag, policy):
    ids = [j for i in pred for j in crop_ids[tag, i]]
    result = iter(apply_mixture([d for ds in pred.values() for d in ds], [v[ids] for v in views], policy))
    return {i: [next(result) for _ in ds] for i, ds in pred.items()}


def overlay_take(form, take):
    assert form.keys() == take.keys()
    return {i: [dict(d, take=t["take"]) for d, t in zip(form[i], take[i], strict=True)] for i in form}


def target_audit(gt, objs, preds):
    result = {obj: Counter(views=0, bulk=0, solid=0, missed=0, any_bulk_iou50=0) for obj in TARGETS}
    for i, gs in gt.items():
        for g, o, p in zip(gs, objs[i], score._match(gs, preds[i]), strict=True):
            if o in result:
                result[o]["views"] += 1
                result[o]["missed" if p is None else p["form"]] += 1
                result[o]["any_bulk_iou50"] += any(d["form"] == "bulk" and score._iou(g["box"], d["box"]) >= .5
                                                    for d in preds[i])
    return result


def compare(gt, base, preds, fold, objects):
    groups = {"pooled": gt, **{str(k): {i: g for i, g in gt.items() if fold[i] == k} for k in range(1, 6)}}
    b = {k: io.metrics(g, base) for k, g in groups.items()}
    m = {k: io.metrics(g, preds) for k, g in groups.items()}
    deltas = {k: io.delta(b[k], m[k]) for k in groups}
    loo = {o: io.metrics(g, preds)["Score"] - io.metrics(g, base)["Score"]
           for o in sorted({o for os in objects.values() for o in os})
           for g in [{i: gs for i, gs in gt.items() if o not in objects[i]}]}
    wins = sum(deltas[str(k)]["Score"] > 1e-9 for k in range(1, 6))
    return dict(metrics=m, delta_base=deltas, positive_folds=wins, loo=loo, min_loo=min(loo.values()),
                passes=(wins >= 3 and deltas["pooled"]["Score"] > 1e-9 and min(loo.values()) > 1e-9),
                targets=target_audit(gt, objects, preds))


def test_caches(exporter, which, checkpoints, frames):
    file_key = exporter.frame_key(frames)
    paths = {}
    for tag in FAMILIES[which]:
        for k in range(1, 6):
            pt, sha = checkpoints[tag, k]
            assert io.sha(pt) == sha
            key = dict(exporter.key, checkpoint_sha256=sha, rgb_sha256=file_key)
            with ExitStack() as ctx:
                _, _, m, cache = raw.read_raw_cache(ctx, exporter.cache, pt, frames, key, require_cache=False)
                assert m["imgsz"] == 960
            paths[tag, k] = exporter.cache / cache
    return paths


def write_submission(path, pred):
    expected = joint.test_ids()
    assert list(pred) == expected and len(expected) == 137
    io.validate_base(pred)
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("image_id", "prediction"))
        for i, ds in pred.items():
            w.writerow((i, ";".join(" ".join(map(str, (*d["box"], d["score"], d["form"], d["material"],
                                                       d["volume"], d["mass"], d["take"]))) for d in ds) or "none"))
    parsed = joint.validate_csv(path)
    assert list(parsed) == expected and all(len(parsed[i]) == len(pred[i]) for i in expected)
    return dict(frames=len(parsed), detections=sum(map(len, parsed.values())), none=sum(not v for v in parsed.values()),
                sha256=io.sha(path))


def run(args):
    fold, real, _, inputs, limits = io.training_contract()
    gt, base = io.load_oof(io.OOF_BASE, fold)
    objs = io.ordered_gt_objects(gt)
    archives, checkpoints = check_sources(fold, args.caches_only)
    data, views, _, _, _, _ = __import__("evaluate_attributes").read_inputs()
    _, cols = joint.read_data()
    image_paths = {i: joint.DATA / im["file_name"] for src, i, im, _ in joint.load_frames() if src == "real"}
    mixture = io.read_json(MIX)
    assert mixture["inputs"]["fold_manifest_sha256"] == inputs["fold_manifest_sha256"]
    gate_stats = io.read_json(W / "sprint4_volume_gate/report.json")
    rescue_stats = io.read_json(W / "astramis_attr_v2/report.json")
    assert rescue_stats["fold_manifest_sha256"] == inputs["fold_manifest_sha256"]
    camera = io.read_json(joint.DATA / "camera.json")
    args.out.mkdir(parents=True, exist_ok=False)
    enc = Encoders(args.device)
    g_data, g_feat = greg_training(image_paths, enc)
    by_fold = {}
    predictions = {v: {} for v in FAMILIES}
    mixtures = {v: {} for v in FAMILIES}
    g_heads = []
    frozen_g = {}
    frozen_mix = io.read_json(W / "instance_take_oof/predictions/mixture.json")
    for k in range(1, 6):
        scope = limits[str(k)]
        train = data.loc[data.image_id.isin(scope["real"] + scope["allowed_synth"])]
        assert not set(train.object_id) & set(scope["forbidden_object_ids"])
        assert not set(g_data.loc[g_data.image_id.isin(scope["real"]), "object_id"]) & set(scope["forbidden_object_ids"])
        g_heads.append(greg_heads(g_data, g_feat, scope["real"]))
        held = {i: image_paths[i] for i in gt if fold[i] == k}
        selected = {i: gt[i] for i in held}
        fbase = {i: base[i] for i in held}
        frozen_g.update(greg_apply(fbase, held, enc, g_heads[-1:]))
        paths = {tag: archives[tag, k] for tag in ("l960", "A", "B")}
        head = joint.fit(train, cols, views)
        rescue = TakeRule(**next(x["rescue_small"] for x in rescue_stats["folds"] if x["fold"] == k))
        p, x, ci = infer_attributes(list(held.items()), paths, FAMILIES, enc, head, density_prior(train),
                                   rescue, gate_stats["train_scales"][str(k)]["log_scale"], geo_module().Geometry(camera))
        for tag, v in p.items():
            predictions[tag].update(v)
            mixtures[tag].update(add_mixture(v, x, ci, tag, mixture["folds"][str(k)]))
        by_fold[k] = {tag: io.metrics(selected, v)["Score"] for tag, v in p.items()}
        print(f"Fold {k} raw: {by_fold[k]}", flush=True)
    raw_scores = {k: compare(gt, base, p, fold, objs) for k, p in predictions.items()}
    candidates = [k for k in FAMILIES if raw_scores[k]["passes"]] or list(FAMILIES)
    winner_raw = max(candidates, key=lambda k: raw_scores[k]["metrics"]["pooled"]["Score"])
    new_g = {}
    for k in range(1, 6):
        p = {i: predictions[winner_raw][i] for i in gt if fold[i] == k}
        new_g.update(greg_apply(p, image_paths, enc, g_heads[k-1:k]))
    variants = dict(base=base, **predictions, new_dino=new_g, new_mixture=mixtures[winner_raw],
                    new_stack=overlay_take(new_g, mixtures[winner_raw]),
                    frozen_clean_dino=frozen_g, frozen_clean_stack=overlay_take(frozen_g, frozen_mix))
    if (HISTORICAL / "stack_oof.json").is_file():
        variants["historical_stack_unattested"] = io.read_json(HISTORICAL / "stack_oof.json")
    stats = {k: compare(gt, base, p, fold, objs) for k, p in variants.items()}
    for tag, orig, dino, expected in (("new_stack", winner_raw, "new_dino", "new_mixture"),
                                      ("frozen_clean_stack", "base", "frozen_clean_dino", None)):
        for i in gt:
            assert io.slots(gt[i], variants[tag][i]) == io.slots(gt[i], variants[orig][i])
        for scope, m in stats[tag]["metrics"].items():
            b = stats[orig]["metrics"][scope]
            d = stats[dino]["metrics"][scope]
            assert (m["D"], m["M"], m["V"]) == (d["D"], d["M"], b["V"])
            if expected is not None:
                assert m["S"] == stats[expected]["metrics"][scope]["S"]
    for k, row in stats.items():
        print(k, {g: round(m["Score"], 5) for g, m in row["metrics"].items()}, "min LOO", round(row["min_loo"], 5), flush=True)
    choices = [winner_raw, "new_stack", "new_dino", "new_mixture", "frozen_clean_stack", "frozen_clean_dino"]
    winner = max((k for k in choices if stats[k]["passes"]),
                 key=lambda k: stats[k]["metrics"]["pooled"]["Score"], default="base")
    report = dict(provenance=inputs, dino_notebook_revision=DINO_REF, dino_training="outer-train real GT box crops",
                  checkpoint_hashes={f"{t}:{k}": v[1] for (t, k), v in checkpoints.items()},
                  winner_raw=winner_raw, winner=winner, variants=stats)
    io.write_json(args.out / "report.json", report)
    io.write_json(args.out / "oof.json", variants)
    print(f"Best new detector: {winner_raw}; best policy passing robustness: {winner}", flush=True)

    if not args.build_test:
        return
    frames = [(i, raw.test_image_path(i)) for i in joint.test_ids()]
    rgb = dict(frames)
    if winner.startswith("frozen_clean") or winner == "base":
        _, final = io.load_test()
        if winner == "frozen_clean_stack":
            from fusion.instance_take_embeddings import encode_donors
            bundle = encode_donors(final, io.load_donors(final, None), (enc.full, enc.clip, enc.clip_t), args.device)
            ids = bundle["donor_index"]
            rows = [d for ds in final.values() for d in ds]
            y = iter(apply_mixture(rows, [bundle[k][ids] for k in ("d", "c", "df", "cf")], mixture["folds"]["test"]))
            adjusted = {i: [next(y) for _ in ds] for i, ds in final.items()}
            final = overlay_take(greg_apply(final, rgb, enc, g_heads), adjusted)
        elif winner == "frozen_clean_dino":
            final = greg_apply(final, rgb, enc, g_heads)
    else:
        exporter = Joint04Exporter(io.ROOT, None, args.device, args.batch)
        paths = test_caches(exporter, winner_raw, checkpoints, frames)
        scope = limits["test"]
        train = data.loc[data.image_id.isin(scope["real"] + scope["allowed_synth"])]
        head = joint.fit(train, cols, views)
        group = {winner_raw: list(paths)}
        p, x, ci = infer_attributes(frames, paths, group, enc, head, density_prior(train),
                                   TakeRule(.85, 300.), gate_stats["test_scale"]["log_scale"],
                                   geo_module().Geometry(camera))
        final = p[winner_raw]
        original = final
        if winner in ("new_stack", "new_dino"):
            final = greg_apply(original, rgb, enc, g_heads)
        if winner in ("new_stack", "new_mixture"):
            final = overlay_take(final, add_mixture(original, x, ci, winner_raw, mixture["folds"]["test"]))
    inventory = write_submission(args.out / "submission_verified.csv", final)
    io.write_json(args.out / "test_verification.json", dict(winner=winner, **inventory))
    print("Verified test:", inventory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--build-test", action="store_true")
    parser.add_argument("--caches-only", action="store_true", help="OOF-only downloads; checkpoint hashes checked before test inference")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    with threadpool_limits(4):
        run(args)


if __name__ == "__main__":
    main()

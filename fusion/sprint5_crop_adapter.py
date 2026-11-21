"""Fixed, CPU-only A/B of external predicted-crop probabilities."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import score as official

WORK = ROOT / "work"
LABELS = ROOT / "dataset/labels"
FOLDS = WORK / "detector_folds_v1.json"
ALLOW = WORK / "sprint3_synth_backgrounds/allowed_synth.json"
RANK_OOF = WORK / "sprint4_form_rank/selected_oof.json"
TEST_BASE = WORK / "sprint4_volume_gate/submission_best_verified.csv"
OOF_BASE = WORK / "sprint5_cpu/base_7486069_oof.json"
OOF_CANONICAL = "20bb99e041dce2c162816c2cf2eb574097385cb60356e138e683d0759a6f0247"
MASK_INDEX = WORK / "sprint1_agnostic_oof_v2/mask_index.json"
ANCHOR_OOF = MASK_INDEX.with_name("predictions.json")
TEST_ANCHOR = WORK / "sprint1_grisha/submission_fused_max_7897_rescue_small.csv"
TEST_WBF = WORK / "submission_wbf6_final.csv"
TEST_RAW = WORK / "test_wbf_v1/base_detections.jsonl"
TEST_MASKS = TEST_RAW.with_name("base_masks.npz")
PINNED = {
    FOLDS: "b4c7a504bdb4793c7a3bc3da9c630ceacc92a9369bcab23d2398a78f4c1b3788",
    ALLOW: "23b02f1be67206a36676e1c0bc8a903e8d73d84a144ce0f1fd9c5abcf48a1397",
    RANK_OOF: "93ed3072dfcc8041ef36fcc3810fab54b516e981213e8ddff71b704461b40173",
    TEST_BASE: "e9c38fe9349c95fac7a6191c55fb8b936f30c7302d356d709ac666b36a65c49f",
    MASK_INDEX: "90fe9cd8059b981effbce1977c2dc1a8c29c6e066ab3f627992046fca504595c",
    ANCHOR_OOF: "0bfafd6e15a0619f67478a9a6eabed6c04966118618bd9aa6efcbdbb350d35f9",
    TEST_ANCHOR: "1b28c7c8a9df5062d3974f87cabe75762cfa4eb10dbe84c70571652144805135",
    TEST_WBF: "7fc58a18d04e443415a3d66b05db6c1d5d42a8340e0fe888b8a521b4e877b6ea",
    TEST_RAW: "5e5eab8a77b85e6d02e41eaa59268b83af8c1d29e40c4e1ad96fb225395e390d",
    TEST_MASKS: "6b0cde800288c2442bbb59bf6e7c6c14e7ef76b497d132b5378db26218e410d4",
}
EXPECTED = dict(Score=74.86069298423111, D=.8616487697709486, M=.6619192211297474,
                V=.6418692840939607, S=.7406417112299466)
MATERIALS = official.MATERIALS
CONFIG = dict(form_alpha=.35, form_guard=.99, material_min=.85,
              material_margin=.35, material_base_max=.15,
              normalization_tolerance=1e-6, delta_epsilon=1e-9,
              min_positive_folds=3, min_distinct_objects=2)
VARIANTS = ("form", "material", "both")
KEYS = ("Score", "D", "M", "V", "S")
DONOR_FIELDS = ("mask_archive", "mask_archive_sha256", "mask_key", "mask_slot", "mask_height", "mask_width",
                "donor_x1", "donor_y1", "donor_x2", "donor_y2", "rgb_path", "depth_path")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    as_json = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(as_json.encode()).hexdigest()


def unique_dict(pairs):
    out = {}
    for k, v in pairs:
        need(k not in out, f"Duplicate JSON key: {k}")
        out[k] = v
    return out


def read_json(path):
    def bad(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_dict,
                      parse_constant=bad)


def write_json(path, data):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def rows(path, header):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, strict=True)
        need(next(reader, None) == list(header), f"{path}: expected header {','.join(header)}")
        for row in reader:
            need(len(row) == len(header), f"{path}:{reader.line_num}: wrong column count")
            yield row


def write_csv(path, header, values):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(values)


def pinned(path):
    need(sha(path) == PINNED[path], f"Frozen input digest mismatch: {path}")
    return read_json(path)


def label_objects(path):
    source = read_json(path)
    ids = {im["id"]: Path(im["file_name"]).parent.name for im in source["images"]}
    found = {i: set() for i in ids.values()}
    need(len(ids) == len(found) == len(source["images"]), f"Duplicate label frame: {path}")
    for a in source["annotations"]:
        obj = a["attributes"]["object_id"]
        need(isinstance(obj, str) and obj, f"Invalid label object: {path}")
        found[ids[a["image_id"]]].add(obj)
    return found


def union_ids(images, by_image):
    return set().union(*(by_image[i] for i in images))


def training_contract():
    """Validate the authoritative masks, never load ground-truth crops."""
    manifest = pinned(FOLDS)
    allow = pinned(ALLOW)
    hashes = {f: sha(LABELS / f) for f in ("real_train.json", "synth_train.json")}
    need({f"labels/{f}": h for f, h in hashes.items()} == manifest["labels_sha256"],
         "Label digests do not match the detector split")
    real, synth = (label_objects(LABELS / f) for f in hashes)
    need(not real.keys() & synth.keys(), "Overlapping real and synth image IDs")
    obj = real | synth
    fold_by_image, limits = {}, {}
    for spec in manifest["folds"]:
        fold = spec["fold"]
        k = str(fold)
        train = set(spec["train_image_ids"])
        val = set(spec["val_real_image_ids"])
        deny = val | set(spec["val_synth_image_ids"]) | set(spec["excluded_mixed_synth_image_ids"])
        train_obj, val_obj = map(set, (spec["train_object_ids"], spec["val_object_ids"]))
        allowed_synth = set(allow[k])
        need(k not in limits and val <= real.keys() and not val & fold_by_image.keys()
             and train <= obj.keys() and not train & deny and not train_obj & val_obj
             and union_ids(train, obj) == train_obj and union_ids(val, obj) == val_obj
             and allowed_synth <= (train & synth.keys()), f"Inconsistent masks for fold {fold}")
        fold_by_image.update({i: fold for i in val})
        limits[k] = dict(real=sorted(train & real.keys()), allowed_synth=sorted(allowed_synth),
                         forbidden_object_ids=sorted(val_obj))
    need(fold_by_image.keys() == real.keys() and set(limits) == set(allow), "Incomplete split coverage")
    limits["test"] = dict(real=sorted(real), allowed_synth=sorted(set().union(*map(set, allow.values()))),
                          forbidden_object_ids=[])
    inputs = dict(fold_manifest_sha256=PINNED[FOLDS], allowed_synth_sha256=PINNED[ALLOW],
                  labels_sha256=hashes)
    return fold_by_image, real, obj, inputs, limits


def load_test():
    need(sha(TEST_BASE) == PINNED[TEST_BASE], "Frozen test base digest mismatch")
    header = ("image_id", "prediction")
    raw = dict(rows(TEST_BASE, header))
    expected_order = [i for i, _ in rows(ROOT / "sample_submission.csv", header)]
    need(list(raw) == expected_order and len(raw) == 137, "Wrong test ID order or coverage")
    pred = {}
    for image, cell in raw.items():
        ds = official._parse(cell, True, image)
        if ds:
            for d, piece in zip(ds, cell.split(";"), strict=True):
                d["mass"] = float(piece.split()[8])
        pred[image] = ds
    need(sum(map(len, pred.values())) == 342 and sum(not ds for ds in pred.values()) == 10,
         "Wrong test inventory")
    validate_base(pred)
    return raw, pred


def load_oof(path, fold_by_image):
    need(path is not None, "--oof-base must be the complete Sprint 4 volume-gated OOF JSON")
    base = read_json(path)
    gt = {i: official._parse(cell, False, i) for i, cell, *_ in
          official.labels_to_solution(LABELS / "real_train.json").itertuples(index=False, name=None)}
    need(isinstance(base, dict) and set(base) == set(gt) == set(fold_by_image) and len(gt) == 263,
         "Wrong OOF image coverage (including empty frames)")
    base = {i: base[i] for i in gt}  # official frame tie-break order
    validate_base(base)
    need(sum(map(len, base.values())) == 377, "Wrong OOF detection inventory")
    if RANK_OOF.is_file():
        rank = pinned(RANK_OOF)
        for i, ds in base.items():
            need(len(ds) == len(rank[i]) and all(
                {k: v for k, v in a.items() if k != "volume"} ==
                {k: v for k, v in b.items() if k != "volume"}
                for a, b in zip(ds, rank[i])), f"{i}: row identity differs from frozen rank OOF")
    values = official.components(gt, base)
    need(all(abs(values[k] - EXPECTED[k]) < 1e-9 for k in KEYS),
         f"Wrong volume-gated OOF components: {values}")
    need(digest(base) == OOF_CANONICAL, "Wrong canonical OOF digest; expected full CPU reconstruction")
    return gt, base


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_base(base):
    required = {"box", "score", "form", "material", "volume", "take"}
    optional = {"mass", "take_p"}
    for i, ds in base.items():
        need(isinstance(ds, list) and len(ds) <= official.MAX_PER_IMAGE, f"{i}: invalid detections")
        for d in ds:
            need(isinstance(d, dict) and required <= d.keys() <= required | optional,
                 f"{i}: wrong base detection schema")
            box = d["box"]
            need(isinstance(box, (list, tuple)) and len(box) == 4 and all(map(finite, box))
                 and box[0] < box[2] and box[1] < box[3], f"{i}: invalid predicted box")
            need(d["form"] in official.FORMS and d["material"] in MATERIALS
                 and finite(d["score"]) and 0 < d["score"] < 1
                 and type(d["take"]) is int and d["take"] in (0, 1)
                 and all(finite(d[k]) for k in ({"volume"} | (optional & d.keys()))),
                 f"{i}: invalid predicted attribute")


def id_fields(fold_by_image):
    return ("image_id", "hyp_idx") + (() if fold_by_image is None else ("fold",)) + (
        "detection_sha256",)


def inventory(base, fold_by_image):
    # Include the slot and its unchanged attributes in the fingerprint; do not use predicted rank as ID.
    return [(i, j, *((fold_by_image[i],) if fold_by_image is not None else ()),
             digest([i, j, d]), *d["box"], d["form"])
            for i, ds in base.items() for j, d in enumerate(ds)]


def records(path):
    return unique_dict((d["image_id"], d) for d in map(json.loads, path.read_text().splitlines()))


def same_box_form(a, b):
    return tuple(a["box"]) == tuple(b["box"]) and a["form"] == b["form"]


def alternate_slots(ds, anchors, raw_slots):
    need(len(ds) >= len(anchors) and len(anchors) == len(raw_slots)
         and all(same_box_form(d, a) for d, a in zip(ds, anchors)), "Frozen anchor prefix changed")
    out = list(raw_slots)
    for d in ds[len(anchors):]:
        overlap, neg = max(((official._iou(d["box"], a["box"]), -j)
                            for j, a in enumerate(anchors) if a["form"] != d["form"]), default=(0, 0))
        need(overlap >= .5, "Missing opposite-form mask donor")
        out.append(raw_slots[-neg])
    return out


def verify_archive(archive, arch_sha, raw, rgb_paths, slot_map):
    need(sha(archive) == arch_sha, f"Unverified mask archive: {archive}")
    out = {}
    with np.load(archive, allow_pickle=False) as masks:
        need(set(masks.files) == set(raw) and set(rgb_paths) == set(slot_map) <= raw.keys(),
             "Mask/frame coverage differs")
        for i, slots in slot_map.items():
            row = raw[i]
            h, w = row["shape"]
            ds = row["detections"]
            arr = masks[i]
            need(row["source"] == "real" and arr.shape == (len(ds), h, (w + 7) // 8)
                 and arr.dtype == np.uint8, f"{i}: invalid packed predicted masks")
            rgb = rgb_paths[i]
            for j, slot in enumerate(slots):
                need(0 <= slot < len(ds) and arr[slot].any(), f"{i}: invalid/empty donor mask slot")
                out[i, j] = (str(archive.relative_to(ROOT)), arch_sha, i, slot, h, w, *ds[slot]["box"],
                             str(rgb.relative_to(ROOT)), str(rgb.with_name("depth.png").relative_to(ROOT)))
    return out


def load_donors(base, fold):
    """Reconstruct Sprint 4's raw predicted mask pointers without generating crops."""
    if fold is not None:
        index = pinned(MASK_INDEX)
        anchors = pinned(ANCHOR_OOF)["n640_rescue"]
        need(index["fold_manifest_sha256"] == PINNED[FOLDS], "Wrong mask index split")
        rgb_paths = {Path(im["file_name"]).parent.name: ROOT / "dataset" / im["file_name"]
                     for im in read_json(LABELS / "real_train.json")["images"]}
        out = {}
        for entry in index["folds"]:
            k = entry["fold"]
            ids = {i for i in fold if fold[i] == k}
            need(ids == set(entry["frames"]), f"Wrong fold {k} mask frame coverage")
            source = entry["sources"][f"n640_fold_{k:02d}"]
            folder = WORK / "detector_oof_v1" / f"fold_{k:02d}" / "evaluation"
            p = folder / "base_detections.jsonl"
            need(sha(p) == source["detections_sha256"], f"Unverified raw detections: {p}")
            raw = records(p)
            need({i for i, row in raw.items() if row["source"] == "real"} == ids,
                 f"Wrong fold {k} raw frame coverage")
            slot_map = {}
            for i in ids:
                ds = raw[i]["detections"]
                need(len(ds) == len(anchors[i]) and all(same_box_form(d, a) and d["score"] == a["score"]
                     for d, a in zip(ds, anchors[i])), f"{i}: raw OOF order differs")
                slot_map[i] = alternate_slots(base[i], anchors[i], range(len(ds)))
            out.update(verify_archive(folder / "base_masks.npz", source["masks_sha256"], raw,
                                      {i: rgb_paths[i] for i in ids}, slot_map))
        return out
    header = ("image_id", "prediction")
    need(all(sha(p) == PINNED[p] for p in (TEST_ANCHOR, TEST_WBF, TEST_RAW)), "Unverified raw test inputs")
    anchors, wbf = ({i: official._parse(cell, True, i) for i, cell in rows(p, header)}
                    for p in (TEST_ANCHOR, TEST_WBF))
    raw = records(TEST_RAW)
    need(set(raw) == set(base) == set(anchors) == set(wbf), "Missing raw test frames")
    slot_map, rgb_paths = {}, {}
    for i, row in raw.items():
        ds = row["detections"]
        need(len(ds) == len(wbf[i]) and all(same_box_form(d, a) and d["score"] == a["score"]
             for d, a in zip(ds, wbf[i])), f"{i}: raw WBF order differs")
        slots = []
        for a in anchors[i]:
            overlap, k = max(((official._iou(a["box"], q["box"]), j)
                              for j, q in enumerate(wbf[i]) if q["form"] == a["form"]), default=(0, -1))
            need(overlap >= .75 and all(a[z] == wbf[i][k][z] for z in ("material", "volume")),
                 f"{i}: raw WBF donor does not match")
            slots.append(k)
        slot_map[i] = alternate_slots(base[i], anchors[i], slots)
        folder = ROOT / "dataset/real/test" / i
        rgbs = sorted(p for p in folder.glob("color.*") if p.suffix.lower() in (".png", ".jpg"))
        need(len(rgbs) == 1, f"{i}: missing or ambiguous RGB path at {folder}")
        rgb_paths[i] = rgbs[0]
    return verify_archive(TEST_MASKS, PINNED[TEST_MASKS], raw, rgb_paths, slot_map)


def read_probs(path, expected, fold_by_image):
    fields = id_fields(fold_by_image)
    header = (*fields, "p_bulk", *(f"p_material_{m}" for m in MATERIALS))
    valid = {tuple(map(str, row[:len(fields)])) for row in expected}
    out = {}
    for n, row in enumerate(rows(path, header), 2):
        key = row[0], row[1]
        need(key not in out, f"{path}:{n}: duplicate slot {key}")
        need(tuple(row[:len(fields)]) in valid,
             f"{path}:{n}: unknown slot, wrong fold, or stale detection digest: {key}")
        need(n - 2 < len(expected) and row[:len(fields)] == list(map(str, expected[n - 2][:len(fields)])),
             f"{path}:{n}: unordered slots; use prepare order")
        p_bulk, *p_mat = map(float, row[len(fields):])
        need(all(math.isfinite(v) and 0 <= v <= 1 for v in [p_bulk, *p_mat]),
             f"{path}:{n}: nonfinite or out-of-range probability")
        need(abs(math.fsum(p_mat) - 1) <= CONFIG["normalization_tolerance"],
             f"{path}:{n}: material probabilities must sum to one")
        out[key] = p_bulk, p_mat
    need(len(out) == len(expected), f"{path}: incomplete detection coverage, {len(out)}/{len(expected)}")
    return {(i, int(j)): p for (i, j), p in out.items()}


def is_digest(s):
    return isinstance(s, str) and re.fullmatch(r"[a-f0-9]{64}", s) is not None


def attest(path, split, base, prob, objects, inputs, limits):
    if path is None:
        return {"status": "unattested", "model_provenance_proven": False}
    a = read_json(path)
    need(type(a["version"]) is int and a["version"] == 1 and a["split"] == split
         and a["dataset"] == inputs and a["base_sha256"] == sha(base)
         and a["probabilities_sha256"] == sha(prob), "Attestation input hashes or split do not match")
    models = a["models"]
    expected = set(limits) - {"test"} if split == "oof" else {"test"}
    need(isinstance(models, dict) and set(models) == expected, "Attestation model folds incomplete")
    need(split != "oof" or "test" not in models, "Test model cannot produce OOF")
    report = {}
    for k, m in models.items():
        scope = limits[k]
        train = m["train_image_ids"]
        need(isinstance(train, list) and all(isinstance(x, str) for x in train)
             and 0 < len(set(train)) == len(train), f"Model {k}: invalid/duplicate declared train IDs")
        need(set(train) <= set(scope["real"]) | set(scope["allowed_synth"])
             and not (union_ids(train, objects) & set(scope["forbidden_object_ids"])),
             f"Model {k}: unapproved training frame/object")
        need(m["inference_crop_source"] in ("predicted_box", "predicted_mask")
             and is_digest(m["checkpoint_sha256"]), f"Model {k}: invalid crop source or model digest")
        checkpoint = m.get("checkpoint_path")
        if checkpoint is not None:
            need(sha(path.parent / checkpoint) == m["checkpoint_sha256"],
                 f"Model {k}: checkpoint digest mismatch")
        report[k] = dict(checkpoint_sha256=m["checkpoint_sha256"],
                         checkpoint_bytes_verified=checkpoint is not None,
                         declared_train_images=len(train), declared_train_objects=len(union_ids(train, objects)),
                         inference_crop_source=m["inference_crop_source"])
    if split == "oof":
        need(len({m["checkpoint_sha256"] for m in report.values()}) == len(report),
             "Identical checkpoint digests across OOF folds")
    return dict(status="declared_masks_verified", model_provenance_proven=False,
                manifest_sha256=sha(path), models=report)


def order(scores):
    return sorted(range(len(scores)), key=lambda j: -scores[j])


def form_scores(ds, probs):
    old = [d["score"] for d in ds]
    logit = [math.log(s) - math.log1p(-s) for s in old]
    changes = [CONFIG["form_alpha"] * (2 * (p if d["form"] == "bulk" else 1 - p) - 1)
               for d, p in zip(ds, probs, strict=True)]
    ix = order(old)
    scale = 1.
    for a, b in zip(ix, ix[1:]):
        closing = changes[b] - changes[a]
        if closing > 0:
            scale = min(scale, CONFIG["form_guard"] * (logit[a] - logit[b]) / closing)
    if not scale:
        return old
    new = [1 / (1 + math.exp(-(x + scale * c))) if c and x >= 0 else
           math.exp(x + scale * c) / (1 + math.exp(x + scale * c)) if c else s
           for s, x, c in zip(old, logit, changes)]
    # Preserve even exact incidence of score ties per frame; guard float saturation and stable sorting.
    if (order(new) != ix or any(not 0 < s < 1 for s in new)
            or any((old[a] == old[b]) != (new[a] == new[b]) for a, b in zip(ix, ix[1:]))):
        return old
    return new


def material_label(old, probs):
    top, second = order(probs)[:2]
    if (MATERIALS[top] != old and probs[top] >= CONFIG["material_min"]
            and probs[top] - probs[second] >= CONFIG["material_margin"]
            and probs[MATERIALS.index(old)] <= CONFIG["material_base_max"]):
        return MATERIALS[top]
    return old


def apply(base, probs, variant):
    need(variant in VARIANTS, "Unknown variant")
    out = {}
    for i, ds in base.items():
        ps = [probs[i, j] for j in range(len(ds))]
        scores = form_scores(ds, [p[0] for p in ps]) if variant != "material" else [d["score"] for d in ds]
        out[i] = [dict(d, score=s, material=material_label(d["material"], p[1])
                       if variant != "form" else d["material"])
                  for d, s, p in zip(ds, scores, ps, strict=True)]
    return out


def metrics(gt, pred):
    pred = {i: pred[i] for i in gt}
    return dict(official.components(gt, pred),
                AP={f: official._average_precision(gt, pred, f) for f in official.FORMS})


def delta(a, b):
    return dict({k: b[k] - a[k] for k in KEYS},
                AP={f: None if a["AP"][f] is None else b["AP"][f] - a["AP"][f]
                    for f in official.FORMS})


def slots(gs, ds):
    ix = {id(d): j for j, d in enumerate(ds)}
    return [ix.get(id(d)) for d in official._match(gs, ds)]


def support(gt, gt_object_ids, base, pred, variant):
    score_objects, material_objects, fixes, harms = set(), set(), set(), set()
    changes = Counter(score_rows=0, material_rows=0, score_unmatched=0, material_unmatched=0)
    for i, gs in gt.items():
        old, new = base[i], pred[i]
        match = slots(gs, old)
        need(match == slots(gs, new), f"{i}: official match changed")
        for j, (a, b) in enumerate(zip(old, new, strict=True)):
            allowed = {"score", "material"} if variant == "both" else {
                "score" if variant == "form" else "material"}
            need(a.keys() == b.keys() and all(a[k] == b[k] for k in a.keys() - allowed),
                 f"{i}: unintended attribute change")
            for field in allowed:
                if a[field] != b[field]:
                    changes[field + "_rows"] += 1
                    changes[field + "_unmatched"] += j not in match
        for g, obj, j in zip(gs, gt_object_ids[i], match, strict=True):
            if j is None:
                continue
            a, b = old[j], new[j]
            if a["score"] != b["score"]:
                score_objects.add(obj)
            if a["material"] != b["material"]:
                material_objects.add(obj)
                if b["material"] == g["material"]:
                    fixes.add(obj)
                elif a["material"] == g["material"]:
                    harms.add(obj)
    return dict(changes, score_changed_objects=sorted(score_objects),
                material_changed_objects=sorted(material_objects),
                material_fixed_objects=sorted(fixes), material_harmed_objects=sorted(harms))


def ordered_gt_objects(gt):
    # labels_to_solution and _parse retain annotation order, including for multi-object frames.
    labels = read_json(LABELS / "real_train.json")
    by_id = {im["id"]: Path(im["file_name"]).parent.name for im in labels["images"]}
    out = {i: [] for i in gt}
    for a in labels["annotations"]:
        out[by_id[a["image_id"]]].append(a["attributes"]["object_id"])
    need(all(len(out[i]) == len(gt[i]) for i in gt), "GT annotation alignment failed")
    return out


def evaluate(gt, objects, base, probs, fold_by_image):
    groups = {str(f): {i: gt[i] for i in gt if fold_by_image[i] == f}
              for f in sorted(set(fold_by_image.values()))}
    g_loo = {o: {i: gt[i] for i in gt if o not in objects[i]}
             for o in sorted({o for os in objects.values() for o in os})}
    base_full = metrics(gt, base)
    base_fold = {k: metrics(gs, base) for k, gs in groups.items()}
    base_loo = {o: official.components(gs, {i: base[i] for i in gs})["Score"] for o, gs in g_loo.items()}
    results = {"baseline": {"pooled": base_full, "folds": base_fold}}
    positive = CONFIG["delta_epsilon"]
    for variant in VARIANTS:
        pred = apply(base, probs, variant)
        pooled = metrics(gt, pred)
        folds = {k: metrics(gs, pred) for k, gs in groups.items()}
        df = {k: delta(base_fold[k], v) for k, v in folds.items()}
        d = delta(base_full, pooled)
        sup = support(gt, objects, base, pred, variant)
        loo = {o: official.components(gs, {i: pred[i] for i in gs})["Score"] - base_loo[o]
               for o, gs in g_loo.items()}
        worst = min(loo, key=loo.get)
        wins = sum(x["Score"] > positive for x in df.values())
        for diff in [d, *df.values()]:
            need(diff["V"] == diff["S"] == 0 and (variant != "form" or diff["M"] == 0)
                 and (variant != "material" or diff["D"] == 0), f"{variant}: isolated metrics changed")
        gate = dict(pooled_score=d["Score"] > positive, positive_folds=wins >= CONFIG["min_positive_folds"],
                    leave_one_object_out=loo[worst] > positive,
                    form_component=variant == "material" or (d["D"] > positive and
                        len(sup["score_changed_objects"]) >= CONFIG["min_distinct_objects"]),
                    material_component=variant == "form" or (d["M"] > positive and
                        len(sup["material_fixed_objects"]) >= CONFIG["min_distinct_objects"]))
        results[variant] = dict(pooled=pooled, delta=d,
                                folds={k: {"metrics": v, "delta": df[k]} for k, v in folds.items()},
                                support=sup, loo=dict(removed_object=worst, min_delta=loo[worst], all_deltas=loo),
                                positive_folds=wins, gate=gate, positive=all(gate.values()))
    return results


def export_cells(raw, base, pred, variant):
    need(list(raw) == list(base) == list(pred), "Changed test frame order or coverage")
    allowed = {"form": {"score"}, "material": {"material"}, "both": {"score", "material"}}[variant]
    cells = {}
    for i, cell in raw.items():
        if not base[i]:
            need(cell == "none" and not pred[i], f"{i}: empty frame changed")
            cells[i] = cell
            continue
        pieces = []
        for piece, old, new in zip(cell.split(";"), base[i], pred[i], strict=True):
            tok = piece.split()
            need(old.keys() == new.keys() and all(old[k] == new[k] for k in old.keys() - allowed),
                 f"{i}: forbidden test change")
            if old["score"] != new["score"]:
                tok[4] = repr(new["score"])
            if old["material"] != new["material"]:
                tok[6] = new["material"]
            pieces.append(" ".join(tok))
        cells[i] = ";".join(pieces)
        roundtrip = official._parse(cells[i], True, i)
        need(roundtrip == [{k: v for k, v in d.items() if k != "mass"} for d in pred[i]]
             and order([d["score"] for d in roundtrip]) == order([d["score"] for d in base[i]]),
             f"{i}: export serialization changed matching order or predictions")
    return cells


def scored(args, contract):
    fold, _, objects, inputs, limits = contract
    gt, base = load_oof(args.oof_base, fold)
    p = read_probs(args.oof_probs, inventory(base, fold), fold)
    evidence = attest(args.oof_attestation, "oof", args.oof_base, args.oof_probs, objects, inputs, limits)
    return dict(config=CONFIG, dataset=inputs, oof_base_sha256=sha(args.oof_base),
                oof_probabilities_sha256=sha(args.oof_probs), attestation=evidence,
                counts=dict(frames=len(gt), detections=len(p), objects=len(union_ids(gt, objects))),
                variants=evaluate(gt, ordered_gt_objects(gt), base, p, fold))


def deploy_spec(args):
    need(sha(TEST_BASE) == PINNED[TEST_BASE], "Frozen test base digest mismatch")
    return dict(version=1, variant=args.variant, config=CONFIG, oof_base_sha256=sha(args.oof_base),
                test_base_sha256=PINNED[TEST_BASE], oof_probabilities_sha256=sha(args.oof_probs),
                test_probabilities_sha256=sha(args.test_probs),
                oof_attestation_sha256=sha(args.oof_attestation) if args.oof_attestation else None,
                test_attestation_sha256=sha(args.test_attestation) if args.test_attestation else None)


def verify_deploy_spec(args):
    spec = read_json(args.deploy_spec)
    need(spec == deploy_spec(args), "Deploy spec mismatch: variant/config or artifact hashes; no export")
    return sha(args.deploy_spec)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare = sub.add_parser("prepare", help="write stable ID/crop inventory; no GT geometry")
    prepare.add_argument("--split", choices=("oof", "test"), required=True)
    prepare.add_argument("--oof-base", type=Path)
    prepare.add_argument("--boxes-only", action="store_true", help="omit raw predicted-mask donor pointers")
    prepare.add_argument("--training-contract-out", type=Path)
    for command in ("score", "plan", "export"):
        p = sub.add_parser(command)
        p.add_argument("--oof-base", type=Path, default=OOF_BASE)
        p.add_argument("--oof-probs", type=Path, required=True)
        p.add_argument("--oof-attestation", type=Path)
        if command in ("plan", "export"):
            p.add_argument("--test-probs", type=Path, required=True)
            p.add_argument("--test-attestation", type=Path)
            p.add_argument("--variant", choices=VARIANTS, required=True)
        if command == "export":
            p.add_argument("--deploy-spec", type=Path, required=True)
            p.add_argument("--allow-unattested", action="store_true", help="explicitly allow unaudited tables")
    for p in sub.choices.values():
        p.add_argument("--out", type=Path, required=True, help="new file; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "plan":
            write_json(args.out, deploy_spec(args))  # hash only; not an approval or score
            return
        contract = training_contract()
        fold, _, objects, inputs, limits = contract
        if args.cmd == "prepare":
            if args.split == "oof":
                _, base = load_oof(args.oof_base or OOF_BASE, fold)
            else:
                need(args.oof_base is None, "--oof-base is only used for OOF prepare")
                _, base = load_test()
                fold = None
            inv = inventory(base, fold)
            header = (*id_fields(fold), "x1", "y1", "x2", "y2", "base_form")
            if not args.boxes_only:
                donors = load_donors(base, fold)
                need(set(donors) == {(r[0], r[1]) for r in inv}, "Donor coverage differs")
                header += DONOR_FIELDS
                inv = [r + donors[r[0], r[1]] for r in inv]
            write_csv(args.out, header, inv)
            if args.training_contract_out is not None:
                write_json(args.training_contract_out, dict(version=1, dataset=inputs, limits=limits))
        elif args.cmd == "score":
            write_json(args.out, scored(args, contract))
        else:
            report = scored(args, contract)  # recompute; never trust a stale/user-edited score report
            result = report["variants"][args.variant]
            need(result["positive"], f"OOF gate rejected {args.variant}: {result['gate']}; no export")
            spec_hash = verify_deploy_spec(args)
            raw, base = load_test()
            p = read_probs(args.test_probs, inventory(base, None), None)
            evidence = attest(args.test_attestation, "test", TEST_BASE, args.test_probs, objects, inputs, limits)
            need(args.allow_unattested or (args.oof_attestation and args.test_attestation),
                 "Supply both attestations or explicitly --allow-unattested; no export")
            if args.test_attestation and args.oof_attestation:
                oof_models = report["attestation"]["models"].values()
                oof_checkpoints = {m["checkpoint_sha256"] for m in oof_models}
                need(evidence["models"]["test"]["checkpoint_sha256"] not in oof_checkpoints,
                     "Test checkpoint repeats an OOF fold checkpoint")
                need({m["inference_crop_source"] for m in oof_models} ==
                     {evidence["models"]["test"]["inference_crop_source"]},
                     "Inconsistent declared OOF/test inference crops")
            candidate = export_cells(raw, base, apply(base, p, args.variant), args.variant)
            need(any(candidate[i] != raw[i] for i in raw), "No test changes; no export")
            write_csv(args.out, ("image_id", "prediction"), candidate.items())
            print(json.dumps(dict(variant=args.variant, oof_delta=result["delta"]["Score"],
                                  oof_attestation=report["attestation"]["status"],
                                  test_attestation=evidence["status"], deploy_spec_sha256=spec_hash,
                                  output_sha256=sha(args.out))))
    except (ValueError, TypeError, KeyError, OSError, OverflowError, csv.Error,
            official.ParticipantVisibleError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""Verify completed Kaggle A outputs against the local frozen dry run; no model load."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main(root):
    frozen = HERE / "clean_fold_manifest.json"
    plan = read(frozen)
    manifest_sha = sha(frozen)
    local = read(HERE / "dry_run.json")
    need(sha(root / frozen.name) == manifest_sha == local["clean_manifest_sha256"], "Wrong clean manifest")
    run, caches = (read(root / name) for name in ("run.json", "cache_manifest.json"))
    need(run["clean_manifest_sha256"] == caches["clean_manifest_sha256"] == manifest_sha
         and run["arm"] == plan["arm"] and run["sources_sha256"] == local["sources_sha256"]
         and run["weights_sha256"] == plan["initialization"]["sha256"]
         and all(run[k] == v for k, v in plan["training"].items()), "Wrong run provenance")
    for name, expected in plan["input_sha256"].items():
        if name not in ("fold_runner.py", "build_notebook.py"):
            need(sha(root / name) == expected, f"Wrong source manifest {name}")
    required_folds = set(map(str, range(1, 6)))
    need(set(run["completed_folds"]) == set(caches["folds"]) == required_folds, "Incomplete run")
    seen, weight_shas = set(), set()
    for f, expected in zip(plan["folds"], local["folds"], strict=True):
        k = str(f["fold"])
        done, cache = run["completed_folds"][k], caches["folds"][k]
        ckpt = f"fold_{int(k):02d}/weights/last.pt"
        digest = sha(root / ckpt)
        need(done["checkpoint"] == ckpt and done["checkpoint_sha256"] == digest
             and digest not in weight_shas and done["epochs"] == plan["training"]["epochs"]
             and done["staged_label_sha256"] == expected["staged_labels_sha256"], f"Wrong checkpoint in fold {k}")
        weight_shas.add(digest)
        ids = f["val_real_image_ids"]
        path = root / f"fold_{int(k):02d}_real_oof.npz"
        expected_key = dict(plan["raw_oof_settings"], fold=int(k), checkpoint_sha256=digest,
            clean_manifest_sha256=manifest_sha, rgb_sha256=f["val_real_rgb_sha256"],
            ultralytics=run["ultralytics"], torch=run["torch"])
        need(cache["frames"] == ids and len(set(ids)) == len(ids) and not (seen & set(ids))
             and cache["archive"] == path.name and sha(path) == cache["archive_sha256"]
             and cache["key"] == expected_key and cache["key_sha256"] == key(expected_key), f"Wrong cache in fold {k}")
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(z["raw_header"].item())
            need(h["key"] == expected_key and set(z.files) == {"raw_header", *ids}
                 and [r["image_id"] for r in h["frames"]] == ids, f"Wrong cache header for fold {k}")
            shapes, n = [], 0
            for row in h["frames"]:
                i = row["image_id"]
                height, width = row["shape"]
                packed = z[i]
                count = len(row["detections"])
                need(packed.shape == (count, height, (width + 7)//8) and packed.dtype == np.uint8,
                     f"Bad packed masks {i}")
                n += count
                shapes.append([i, row["shape"], list(packed.shape)])
            need(key(shapes) == cache["shape_sha256"] and n == cache["detections"], f"Wrong shape/count SHA fold {k}")
        seen.update(ids)
        print(f"Fold {k}: {len(ids)} real frames, {n} detections, checkpoint SHA-256 {digest}")
    need(len(seen) == 263, "Incomplete real OOF")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <download-root>/astramis-clean-control-A")
    main(Path(sys.argv[1]))

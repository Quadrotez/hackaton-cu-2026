"""Extend the shared object CV with strictly held-out real empty capture sessions."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from material_split import DATA, WORK, load_frames


SOURCE_FOLDS = WORK / "cv_folds.json"
DEFAULT_OUT = WORK / "detector_folds_v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    base = json.loads(SOURCE_FOLDS.read_text(encoding="utf-8"))
    frames = {}
    session_rows = defaultdict(list)
    for source, image_id, image, anns in load_frames():
        if image_id in frames:
            raise ValueError(f"Duplicate image ID {image_id}")
        objects = {a["attributes"]["object_id"] for a in anns}
        frames[image_id] = source, objects
        if source == "real" and not anns:
            meta = json.loads(((DATA / image["file_name"]).parent / "meta.json").read_text(encoding="utf-8"))
            time = datetime.fromisoformat(meta["time"])
            # Keep the entire capture day together, including lighting transitions.
            session_rows[time.date().isoformat()].append((time, image_id, meta["lighting"]))

    objects = set().union(*(ids for _, ids in frames.values()))
    owners = Counter(o for f in base for o in f["val_object_ids"])
    if set(owners) != objects or any(n != 1 for n in owners.values()):
        raise ValueError("Object folds must cover each physical object exactly once")
    if len(base) != 5 or {f["fold"] for f in base} != set(range(1, 6)):
        raise ValueError("Expected the five existing object folds")

    loads = {}
    for f in base:
        val = set(f["val_object_ids"])
        if set(f["train_object_ids"]) != objects - val:
            raise ValueError(f"Inconsistent train objects in fold {f['fold']}")
        loads[f["fold"]] = sum(src == "real" and bool(ids & val) for src, ids in frames.values())
    empty_series = []
    for date, samples in sorted(session_rows.items(), key=lambda p: (-len(p[1]), p[0])):
        if not samples:
            continue
        owner = min(loads, key=lambda f: (loads[f], f))
        loads[owner] += len(samples)
        samples.sort()
        empty_series.append({
            "series_id": f"real_empty_{date}",
            "val_fold": owner,
            "start": samples[0][0].isoformat(),
            "end": samples[-1][0].isoformat(),
            "lighting_counts": dict(sorted(Counter(s[2] for s in samples).items())),
            "image_ids": [i for _, i, _ in samples],
        })

    folds = []
    for f in base:
        val = set(f["val_object_ids"])
        series = [s for s in empty_series if s["val_fold"] == f["fold"]]
        empty = {i for s in series for i in s["image_ids"]}
        train, val_real, val_synth, mixed = [], [], [], []
        for i, (source, ids) in sorted(frames.items()):
            if i in empty:
                continue
            if not ids & val:
                train.append(i)
            elif source == "real":
                if not ids <= val:
                    raise ValueError(f"Mixed real frame needs an explicit policy: {i}")
                val_real.append(i)
            elif ids <= val:
                val_synth.append(i)
            else:
                mixed.append(i)
        partitions = train + val_real + val_synth + mixed + sorted(empty)
        if len(partitions) != len(frames) or len(set(partitions)) != len(frames):
            raise ValueError(f"Incomplete or overlapping partitions in fold {f['fold']}")
        folds.append({
            **f,
            "train_image_ids": train,
            "val_real_image_ids": sorted(val_real + sorted(empty)),
            "val_real_positive_image_ids": val_real,
            "val_real_empty_image_ids": sorted(empty),
            "val_empty_series_ids": [s["series_id"] for s in series],
            "val_synth_image_ids": val_synth,
            "excluded_mixed_synth_image_ids": mixed,
        })
    real_val = Counter(i for f in folds for i in f["val_real_image_ids"])
    if set(real_val) != {i for i, (s, _) in frames.items() if s == "real"} or set(real_val.values()) != {1}:
        raise ValueError("Every labeled real frame must occur in outer validation exactly once")
    if len(empty_series) < 2:
        raise ValueError("Expected separately held-out empty capture days")
    return {
        "version": 1,
        "source_object_folds": f"work/{SOURCE_FOLDS.name}",
        "source_object_folds_sha256": digest(SOURCE_FOLDS),
        "labels_sha256": {str(p.relative_to(DATA)): digest(p) for p in (
            DATA / "labels" / "real_train.json", DATA / "labels" / "synth_train.json"
        )},
        "rules": {
            "objects": "Use unchanged outer object groups from work/cv_folds.json.",
            "real_negatives": "Group empty frames by capture day; hold out each whole group once. "
                              "Assign largest group first to the fold with fewest real validation frames.",
            "mixed_synthetic": "If a frame has both train and validation objects, exclude the entire frame.",
            "primary_evaluation": "Pooled outer validation on all real frames; report empty FPS separately.",
        },
        "real_empty_series": empty_series,
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    manifest = build()
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    for fold in manifest["folds"]:
        print(f"fold {fold['fold']}: train={len(fold['train_image_ids'])} "
              f"val_real_pos={len(fold['val_real_positive_image_ids'])} "
              f"val_real_empty={len(fold['val_real_empty_image_ids'])} "
              f"val_synth={len(fold['val_synth_image_ids'])} "
              f"excluded_mixed_synth={len(fold['excluded_mixed_synth_image_ids'])}")
    print(f"{args.out} sha256={digest(args.out)}")


if __name__ == "__main__":
    main()

"""Recover exact real-train background donors and filter already-allowed synth IDs."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "dataset"
MANIFEST = ROOT / "work/detector_folds_v1.json"
REAL = DATA / "labels/real_train.json"
SYNTH = DATA / "labels/synth_train.json"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_image(folder, name, shape, dtype):
    path = folder / name
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert im is not None and im.shape == shape and im.dtype == dtype, str(path)
    return im


def folders(labels, source):
    out = {}
    for im in labels["images"]:
        path = Path(im["file_name"])
        assert path.parts[:2] == (source, "train")
        out[im["id"]] = (path.parent.name, DATA / path.parent, (im["height"], im["width"]))
    assert len(out) == len(labels["images"])
    return out


def tiles(shape):
    """Independent spatial strata; only mask==0 pixels participate in checking each tile."""
    h, w = shape
    yy = np.linspace(0, h, 5, dtype=int)
    xx = np.linspace(0, w, 7, dtype=int)
    for i in range(4):
        for j in range(6):
            y = np.arange(yy[i] + 3, yy[i+1], 9)
            x = np.arange(xx[j] + 4, xx[j+1], 11)
            yield y[:, None] * w + x[None, :]


def run():
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    real, synth, manifest = map(load, (REAL, SYNTH, MANIFEST))
    for p in (REAL, SYNTH):
        assert sha(p) == manifest["labels_sha256"]["labels/" + p.name], str(p)
    real_positive = {a["image_id"] for a in real["annotations"]}
    real_folders = folders(real, "real")
    synth_folders = folders(synth, "synth")
    empty = sorted(v for k, v in real_folders.items() if k not in real_positive)
    assert len(empty) == 49 and len(synth_folders) == 600
    shape, = {t[2] for t in (*empty, *synth_folders.values())}
    donor_ids = [fid for fid, _, _ in empty]
    donors = np.stack([read_image(folder, "depth.png", shape, np.uint16).ravel()
                       for _, folder, _ in empty])
    samples = [tile.ravel() for tile in tiles(shape)]
    sample_depth = [donors[:, s] for s in samples]
    instance_ids, synth_objects = defaultdict(set), defaultdict(set)
    for a in synth["annotations"]:
        instance_ids[a["image_id"]].add(a["instance"])
        synth_objects[synth_folders[a["image_id"]][0]].add(a["attributes"]["object_id"])

    donor_map, tile_count, shortlist_count = {}, Counter(), Counter()
    for i, (fid, folder, _) in sorted(synth_folders.items(), key=lambda kv: kv[1][0]):
        mask = read_image(folder, "mask.png", shape, np.uint8).ravel()
        depth = read_image(folder, "depth.png", shape, np.uint16).ravel()
        # np.bincount both verifies the instance IDs and preserves an exact masking rule.
        ids = set(np.flatnonzero(np.bincount(mask)))
        assert ids == {0, *instance_ids[i]}, (fid, ids, instance_ids[i])
        visible = mask == 0
        shortlist = np.arange(len(empty))
        used = 0
        for s, donor_sample in zip(samples, sample_depth, strict=True):
            v = visible[s]
            if v.sum() < 16:
                continue
            used += 1
            shortlist = shortlist[np.all(donor_sample[shortlist][:, v] == depth[s[v]], axis=1)]
        assert used >= 4 and len(shortlist) > 0, (fid, used, shortlist)
        # Exact matches cannot have been lost by the shortlist: all sampled pixels are
        # a subset of the full compared region, and every sampled criterion is equality.
        outside = depth[visible]
        exact = [k for k in shortlist if np.array_equal(donors[k, visible], outside)]
        assert len(exact) == 1, (fid, [donor_ids[k] for k in exact])
        donor_map[fid] = donor_ids[exact[0]]
        tile_count[used] += 1
        shortlist_count[len(shortlist)] += 1
    assert set(donor_map.values()) == set(donor_ids)

    held_out, allowed, ledger = {}, {}, {}
    for spec in manifest["folds"]:
        f = spec["fold"]
        held = set(spec["val_real_empty_image_ids"])
        assert held <= set(donor_ids) and not held & held_out.keys()
        held_out.update({donor: f for donor in held})
        train = {fid for fid in spec["train_image_ids"] if fid.startswith("s_")}
        assert train <= donor_map.keys()
        mixed = set(spec["excluded_mixed_synth_image_ids"])
        assert mixed <= donor_map.keys()
        tr_o, va_o = set(spec["train_object_ids"]), set(spec["val_object_ids"])
        assert not tr_o & va_o
        object_mixed = {fid for fid, objects in synth_objects.items() if objects & tr_o and objects & va_o}
        assert mixed == object_mixed, f
        train_unmixed = train - mixed
        assert all(synth_objects[fid] <= tr_o and synth_objects[fid] for fid in train_unmixed), f
        assert not train_unmixed & set(spec["val_synth_image_ids"]), f
        donor_exclusions = {fid for fid in train_unmixed if donor_map[fid] in held}
        allowed[f] = sorted(train_unmixed - donor_exclusions)
        ledger[f] = {
            "manifest_train_synth": len(train),
            "mixed_already_excluded_in_manifest": len(mixed - train),
            "mixed_removed_from_manifest_train": sorted(train & mixed),
            "held_out_empty_donor_count": len(held),
            "held_out_donor_removed_count": len(donor_exclusions),
            "held_out_donor_removed_ids": sorted(donor_exclusions),
            "allowed_count": len(allowed[f]),
        }
    assert set(held_out) == set(donor_ids)
    assert {f: v["held_out_donor_removed_count"] for f, v in ledger.items()} == {1: 173, 2: 0, 3: 0, 4: 292, 5: 0}
    write("synth_to_real_empty.json", donor_map)
    write("allowed_synth.json", allowed)
    with (OUT / "synth_to_real_empty.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("synth_id", "real_empty_id", "donor_val_fold"))
        writer.writerows((fid, donor, held_out[donor]) for fid, donor in sorted(donor_map.items()))
    write("audit.json", {
        "matched_synth": len(donor_map), "unique_donors": len(set(donor_map.values())),
        "shortlist_size_histogram": dict(shortlist_count), "sampled_tile_count_histogram": dict(tile_count),
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (REAL, SYNTH, MANIFEST)},
        "folds": ledger,
    })
    print(f"Matched {len(donor_map)} synth images to {len(set(donor_map.values()))} real empty donors")
    for f, g in ledger.items():
        print(f"fold {f}: {g['manifest_train_synth']} -> {g['allowed_count']}, donor removed {g['held_out_donor_removed_count']}")


if __name__ == "__main__":
    run()

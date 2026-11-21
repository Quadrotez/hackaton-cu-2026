"""Read-only provenance inventory. Does not fit or evaluate any model."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "dataset"
PAIR = ("metal", "plastic")


def read(path):
    return json.loads(path.read_text())


def frame_data(data):
    objs = defaultdict(list)
    for a in data["annotations"]:
        objs[a["image_id"]].append(a)
    return {Path(im["file_name"]).parent.name: (im, objs[im["id"]]) for im in data["images"]}


def counts(rows):
    return {
        m: {"views": len([o for material, o in rows if material == m]),
            "objects": len({o for material, o in rows if material == m})}
        for m in PAIR
    }


def main(out):
    manifest = ROOT / "work/detector_folds_v1.json"
    labels = {src: DATA / "labels" / f"{src}_train.json" for src in ("real", "synth")}
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (manifest, *labels.values())}
    rules = read(manifest)
    if any(inputs[str(p.relative_to(ROOT))] != rules["labels_sha256"][str(p.relative_to(DATA))]
           for p in labels.values()):
        raise ValueError("Label hash mismatch")
    data = {src: read(p) for src, p in labels.items()}
    real, synth = (frame_data(data[src]) for src in ("real", "synth"))
    meta = [read((DATA / im["file_name"]).with_name("meta.json")) for im, _ in synth.values()]
    fold_reports = {}
    for s in rules["folds"]:
        allowed = set(s["train_image_ids"])
        train_objs, val_objs = map(set, (s["train_object_ids"], s["val_object_ids"]))
        assert not (train_objs & val_objs)
        real_rows = []
        for fid in sorted(allowed & real.keys()):
            for a in real[fid][1]:
                at = a["attributes"]
                assert at["object_id"] in train_objs
                if at["material"] in PAIR:
                    real_rows.append((at["material"], at["object_id"]))
        real_parents = set(real_rows)
        synth_rows = []
        filtered = Counter()
        for fid in sorted(allowed & synth.keys()):
            im, anns = synth[fid]
            if len(anns) != 1 or anns[0]["instance"] != 1:
                filtered["not_one_annotated_instance"] += 1
                continue
            at = anns[0]["attributes"]
            if at["material"] not in PAIR:
                filtered["not_pair"] += 1
                continue
            row = (at["material"], at["object_id"])
            if at["object_id"] not in train_objs or at["object_id"] in val_objs or row not in real_parents:
                filtered["no_matching_allowed_real_parent"] += 1
                continue
            path = DATA / im["file_name"]
            if not all(path.with_name(name).is_file() for name in ("color.png", "depth.png", "mask.png")):
                filtered["missing_media"] += 1
                continue
            synth_rows.append(row)
        fold_reports[s["fold"]] = {
            "allowed_synth_frames": len(allowed & synth.keys()),
            "metadata_rejected_by_first_reason": dict(filtered),
            "real_control": counts(real_rows),
            "synth_passed_metadata_filters_only": counts(synth_rows),
            "synth_provenance_verified_frames": 0,
        }

    def schemas(rows):
        return Counter(",".join(sorted(row)) for row in rows)

    report = {
        "decision": "no_run_unverified_foreground_provenance",
        "input_sha256": inputs,
        "synth_schemas": {
            "top_level": sorted(data["synth"]),
            "image": schemas(data["synth"]["images"]),
            "annotation": schemas(data["synth"]["annotations"]),
            "annotation_attributes": schemas(a["attributes"] for a in data["synth"]["annotations"]),
            "frame_meta": schemas(meta),
        },
        "folds": fold_reports,
    }
    with out.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for k, fold in fold_reports.items():
        n = sum(row["views"] for row in fold["synth_passed_metadata_filters_only"].values())
        print(f"Fold {k}: {n} metadata-only candidates; 0 certified foreground")
    print("NO RUN: no per-frame foreground source/transform manifest for RGB-D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "preflight.json")
    args = parser.parse_args()
    if args.out.resolve().parent != HERE:
        parser.error("--out must be inside synth01")
    main(args.out)

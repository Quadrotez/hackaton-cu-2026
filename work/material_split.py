import json
from collections import Counter, defaultdict
from pathlib import Path


WORK = Path(__file__).resolve().parent
DATA = WORK.parent / "dataset"
SPLIT_PATH = WORK / "split.json"
PARTS = ("train", "val")


def load_frames():
    for source in ("real", "synth"):
        labels = json.loads((DATA / "labels" / f"{source}_train.json").read_text(encoding="utf-8"))
        annotations = defaultdict(list)
        for ann in labels["annotations"]:
            annotations[ann["image_id"]].append(ann)
        for image in labels["images"]:
            image_id = Path(image["file_name"]).parent.name
            yield source, image_id, image, annotations[image["id"]]


def build_split(val_ids=None, output=SPLIT_PATH):
    frames = list(load_frames())
    objects = sorted({a["attributes"]["object_id"] for _, _, _, anns in frames for a in anns})
    val_objects = set(objects[::5] if val_ids is None else val_ids)
    assert val_objects and val_objects < set(objects)
    plan = {
        "rule": "val_objects = sorted(real_and_synth_object_ids)[::5]" if val_ids is None else "explicit holdout object_ids",
        "train": {"object_ids": [obj for obj in objects if obj not in val_objects], "image_ids": []},
        "val": {"object_ids": sorted(val_objects), "image_ids": []},
        "excluded_annotations": [],
    }
    actual_objects = {part: set() for part in PARTS}
    counts = {part: defaultdict(Counter) for part in PARTS}
    mixed_frames = 0
    for source, image_id, _, anns in frames:
        part = "val" if any(a["attributes"]["object_id"] in val_objects for a in anns) else "train"
        plan[part]["image_ids"].append(image_id)
        counts[part][source]["frames"] += 1
        mixed = False
        for ann in anns:
            obj = ann["attributes"]["object_id"]
            # A mixed frame stays in val, but its train-object annotation is not used.
            if part == "val" and obj not in val_objects:
                plan["excluded_annotations"].append(
                    {"image_id": image_id, "ann_id": ann["id"], "object_id": obj}
                )
                mixed = True
                continue
            counts[part][source]["annotations"] += 1
            actual_objects[part].add(obj)
        mixed_frames += mixed

    overlap = actual_objects["train"] & actual_objects["val"]
    assert not overlap, overlap
    frame_ids = [image_id for part in PARTS for image_id in plan[part]["image_ids"]]
    assert len(set(frame_ids)) == len(frame_ids) == len(frames)
    assert (
        sum(c["annotations"] for part in PARTS for c in counts[part].values())
        + len(plan["excluded_annotations"])
        == sum(len(anns) for _, _, _, anns in frames)
    )
    for part in PARTS:
        assert actual_objects[part] == set(plan[part]["object_ids"])
        plan[part]["counts_by_source"] = {s: dict(c) for s, c in counts[part].items()}
        print(
            f"{part}: {len(actual_objects[part])} object_id, {len(plan[part]['image_ids'])} frames, "
            f"{sum(c['annotations'] for c in counts[part].values())} annotations; "
            f"{plan[part]['counts_by_source']}"
        )
    print(f"Shared object_id: {len(overlap)}; shared frames: 0")
    print(f"Mixed frames: {mixed_frames}; excluded annotations: {len(plan['excluded_annotations'])}")
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_split()

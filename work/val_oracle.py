import json

import numpy as np
import pandas as pd

from material_split import DATA, SPLIT_PATH, WORK, load_frames


def build_oracle(plan_path=SPLIT_PATH, predictions_path=WORK / "oracle_object_predictions.csv", suffix=""):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    val_images = set(plan["val"]["image_ids"])
    val_objects = set(plan["val"]["object_ids"])
    model_preds = pd.read_csv(predictions_path)
    assert not model_preds.duplicated(["image_id", "ann_id"]).any()
    assert np.isfinite(model_preds[["volume_pred_ml", "mass_pred_g"]]).all().all()
    assert model_preds["volume_pred_ml"].gt(0).all() and model_preds["mass_pred_g"].gt(0).all()
    assert model_preds["take_pred"].isin([0, 1]).all()
    preds = {(row.image_id, row.ann_id): row for row in model_preds.itertuples(index=False)}

    real_labels_source = json.loads((DATA / "labels" / "real_train.json").read_text(encoding="utf-8"))
    real_labels = {"categories": real_labels_source["categories"], "images": [], "annotations": []}
    all_labels = {"categories": real_labels_source["categories"], "images": [], "annotations": []}
    result = []
    for source, image_id, image, anns in load_frames():
        if image_id not in val_images:
            continue
        combined_id = len(result) + 1
        all_labels["images"].append({**image, "id": combined_id})
        if source == "real":
            real_labels["images"].append(image)
        tokens = []
        for ann in anns:
            at = ann["attributes"]
            if at["object_id"] not in val_objects:
                continue
            pred = preds.pop((image_id, ann["id"]))
            x, y, w, h = ann["bbox"]
            tokens.append(
                f"{x:.1f} {y:.1f} {x + w:.1f} {y + h:.1f} 1.0 {at['form']} {pred.material_pred} "
                f"{pred.volume_pred_ml:.8f} {pred.mass_pred_g:.8f} {int(pred.take_pred)}"
            )
            all_labels["annotations"].append({
                **ann, "id": len(all_labels["annotations"]) + 1, "image_id": combined_id,
            })
            if source == "real":
                real_labels["annotations"].append(ann)
        result.append({"image_id": image_id, "prediction": ";".join(tokens) or "none"})

    assert not preds
    assert len(result) == len(val_images) and {r["image_id"] for r in result} == val_images
    assert len(all_labels["annotations"]) == len(model_preds)
    for fname, labels in (
        (f"real_val_labels{suffix}.json", real_labels), (f"oracle_all_labels{suffix}.json", all_labels)
    ):
        (WORK / fname).write_text(json.dumps(labels) + "\n", encoding="utf-8")
        print(f"{fname}: {len(labels['images'])} frames, {len(labels['annotations'])} objects")
    output = WORK / f"val_oracle{suffix}.csv"
    pd.DataFrame(result).sort_values("image_id").to_csv(output, index=False)
    print(f"{output.name}: {len(result)} unique frames; empty={sum(r['prediction'] == 'none' for r in result)}")


if __name__ == "__main__":
    build_oracle()

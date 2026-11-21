"""Export the bulk-only y_bottom volume A/B without changing any other fields."""

import argparse
import csv
import json
from pathlib import Path

from material_anchor_ab import L_CSV, L_RUN, donor_index, test_features
from submission_fusion import load
from volume_feature_ab import WORK, cv2, fit_predict, joint, np, pd, threadpool_limits, train_extras
from yolo960_ab import file_sha, write_csv


def run(out):
    if file_sha(L_CSV) != json.loads((L_RUN / "report.json").read_text())["test"]["output_sha256"]:
        raise ValueError("Frozen l960 CSV has changed")
    original, ids, features = test_features()
    materials = [original[i][j]["tail"].split()[1] for i, j in ids]
    baseline = np.array([float(original[i][j]["tail"].split()[2]) for i, j in ids])
    data, _ = joint.read_data()
    data = data.merge(pd.read_csv(joint.GEOMETRY_PATH), on=["image_id", "ann_id"], validate="one_to_one")
    data = data.merge(train_extras(joint.load_frames(), False), on=["image_id", "ann_id"], validate="one_to_one")
    hold = features.assign(material_pred=materials)
    if len(data) != 892:
        raise ValueError("Training feature count changed")
    bulk = hold.form.eq("bulk").to_numpy()
    with threadpool_limits(4):
        replay = fit_predict(data, hold, ())
        proposal = fit_predict(data, hold, ("y_bottom",))
    if not np.allclose(replay, baseline, rtol=1e-8, atol=1e-9):
        raise ValueError("Original head fails to reproduce the frozen volume; refusing export")
    volumes = dict(zip(ids, np.where(bulk, proposal, baseline), strict=True))
    candidate = load(L_CSV)
    changes = []
    with L_CSV.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            fid, text = row["image_id"], row["prediction"]
            old = original[fid]
            parts = text.split(";") if old else []
            for idx, p in enumerate(candidate[fid]):
                donor = donor_index(fid, old, idx, p)
                fields = parts[idx].split()
                volume = volumes[fid, donor]
                if not np.isclose(float(fields[7]), volume, rtol=1e-10, atol=1e-9):
                    changes.append({"image": fid, "index": idx, "from": float(fields[7]), "to": float(volume)})
                    fields[7] = repr(float(volume))
                    parts[idx] = " ".join(fields)
            rows.append((fid, ";".join(parts) or text))
    out.mkdir(parents=True, exist_ok=False)
    csv_path = out / "submission_807475_bulk_y_bottom.csv"
    write_csv(csv_path, rows, joint.validate_csv)
    report = {"base_sha256": file_sha(L_CSV), "output_sha256": file_sha(csv_path), "changes": changes}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Changed only volume in {len(changes)} predictions on {len({c['image'] for c in changes})} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "sprint2_volume_bulk_y")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    cv2.setNumThreads(1)
    run(args.out)

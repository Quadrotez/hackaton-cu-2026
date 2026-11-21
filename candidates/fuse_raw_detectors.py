"""Fuse raw detector boxes from several models into a primary submission.

Primary CSV supplies material/volume/mass/take. Detector boxes are clustered by
IoU>=0.5 within the same form, weighted-averaged by score; confidence is max.
"""
import argparse
import csv
import json
from pathlib import Path


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary")
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--score-mode", choices=["max", "avg"], default="max")
    args = ap.parse_args()
    preds = json.loads(Path(args.preds).read_text())
    rows = [["image_id", "prediction"]]
    stats = {"clusters": 0, "matched": 0, "only_models": 0}
    with Path(args.primary).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = row["image_id"]
            primary = []
            if row["prediction"] != "none":
                for part in row["prediction"].split(";"):
                    t = part.split()
                    primary.append({"box": list(map(float, t[:4])), "score": float(t[4]),
                                    "form": t[5], "tail": " ".join(t[5:])})
            dets = [(name, d) for name, per in preds.items() for d in per.get(fid, [])]
            parts = []
            matched_frame = 0
            for w in primary:
                cands = [(iou(w["box"], d["box"]), j) for j, (_, d) in enumerate(dets)
                         if d["form"] == w["form"]]
                best, j = max(cands, default=(0.0, -1))
                box, score = w["box"], w["score"]
                if best >= 0.5:
                    d = dets[j][1]
                    box = [(a * w["score"] + b * d["score"]) / (w["score"] + d["score"])
                           for a, b in zip(w["box"], d["box"])]
                    score = max(w["score"], d["score"]) if args.score_mode == "max" else (w["score"] + d["score"]) / 2
                    matched_frame += 1
                parts.append(f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {score:.6f} {w['tail']}")
            stats["clusters"] += len(parts)
            stats["matched"] += matched_frame
            stats["only_models"] += len(dets) - matched_frame
            rows.append([fid, ";".join(parts) if parts else "none"])
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(args.out, stats)


if __name__ == "__main__":
    main()

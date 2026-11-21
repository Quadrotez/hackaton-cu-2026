"""Fuse detections from several model submissions into one CSV.

Each input is a standard Astramis submission (`image_id,prediction`) with full
10-field objects. The first input is primary: its material/volume/mass/take are
kept for matched objects. Boxes are averaged with detection-score weights;
confidence is the maximum member score (best on the platform A/B).

Usage:
    python submission_fusion.py primary.csv other1.csv other2.csv ... out.csv
    python submission_fusion.py --score-mode avg primary.csv other.csv out.csv
"""
import argparse
import csv
from pathlib import Path


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load(path):
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred = row["prediction"]
            if pred == "none":
                out[row["image_id"]] = []
                continue
            dets = []
            for part in pred.split(";"):
                t = part.split()
                if len(t) != 10:
                    raise ValueError(f"{path}: {row['image_id']}: {len(t)} fields")
                dets.append({"box": list(map(float, t[:4])), "score": float(t[4]),
                             "form": t[5], "tail": " ".join(t[5:])})
            out[row["image_id"]] = dets
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-mode", choices=["max", "avg"], default="max")
    ap.add_argument("inputs", nargs="+", help="submission CSVs; first is primary for attributes")
    ap.add_argument("out")
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("need at least two input CSVs")
    sources = [load(p) for p in args.inputs]
    rows = [["image_id", "prediction"]]
    stats = {"clusters": 0, "single": 0, "multi": 0}
    for fid in sorted(sources[0]):
        clusters = []
        all_dets = [(si, d) for si, src in enumerate(sources) for d in src.get(fid, [])]
        for si, d in sorted(all_dets, key=lambda x: -x[1]["score"]):
                c = next((c for c in clusters
                          if c["form"] == d["form"]
                          and iou(c["box"], d["box"]) >= 0.5), None)
                if c is None:
                    clusters.append({"form": d["form"], "members": [(si, d)], "box": d["box"]})
                elif si not in [s for s, _ in c["members"]]:
                    c["members"].append((si, d))
        parts = []
        for c in clusters:
            members = c["members"]
            w = [d["score"] for _, d in members]
            total = sum(w)
            box = [sum(d["box"][k] * sc for (_, d), sc in zip(members, w)) / total
                   for k in range(4)]
            score = max(w) if args.score_mode == "max" else total / len(sources)
            _, primary = next(((s, d) for s, d in members if s == 0), members[0])
            parts.append(f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} "
                         f"{score:.6f} {primary['tail']}")
            stats["multi" if len(members) > 1 else "single"] += 1
        stats["clusters"] += len(clusters)
        rows.append([fid, ";".join(parts) if parts else "none"])
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"{args.out}: {stats}")


if __name__ == "__main__":
    main()

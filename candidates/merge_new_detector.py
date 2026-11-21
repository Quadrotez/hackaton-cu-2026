"""Fuse new detector boxes into the team's WBF submission, keep its heads."""
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
    ap.add_argument("wbf")
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--model", default="new")
    ap.add_argument("--score-mode", choices=["max", "avg", "wbf"], default="max")
    args = ap.parse_args()
    preds = json.loads(Path(args.preds).read_text())[args.model]
    rows = [["image_id", "prediction"]]
    stats = {"matched": 0, "wbf_only": 0, "model_only": 0}
    with Path(args.wbf).open(newline="") as f:
        for row in csv.DictReader(f):
            fid = row["image_id"]
            wbf = []
            if row["prediction"] != "none":
                for part in row["prediction"].split(";"):
                    t = part.split()
                    wbf.append({"box": list(map(float, t[:4])), "score": float(t[4]),
                                "form": t[5], "tail": " ".join(t[5:])})
            dets = preds.get(fid, [])
            used = set()
            parts = []
            for w in wbf:
                cands = [(iou(w["box"], d["box"]), j) for j, d in enumerate(dets)
                         if j not in used and d["form"] == w["form"]]
                best_iou, j = max(cands, default=(0.0, -1))
                box, score = w["box"], w["score"]
                if best_iou >= 0.5:
                    used.add(j)
                    d = dets[j]
                    wb = w["score"]
                    db = d["score"]
                    box = [(a * wb + b * db) / (wb + db) for a, b in zip(w["box"], d["box"])]
                    score = max(wb, db) if args.score_mode == "max" else (wb + db) / 2
                    stats["matched"] += 1
                else:
                    stats["wbf_only"] += 1
                parts.append(
                    f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {score:.6f} {w['tail']}"
                )
            stats["model_only"] += len(dets) - len(used)
            rows.append([fid, ";".join(parts) if parts else "none"])
    with Path(args.out).open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(args.out, stats)


if __name__ == "__main__":
    main()

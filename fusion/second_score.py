"""Raise the score of the second (hedge) hypothesis to a fixed calibration value.

The top hypothesis per frame keeps its detector score, so the class-agnostic
match used for M/V/S is unchanged. Only the D ranking of the alternative-form
hypothesis is affected.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "submissions" / "submission_v20_top2.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--in", dest="inp", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.reader(open(args.inp, encoding="utf-8")))
    out = []
    changed = 0
    for img_id, pred in rows[1:]:
        if pred.strip() in ("", "none"):
            out.append((img_id, pred))
            continue
        objs = pred.split(";")
        objs.sort(key=lambda s: -float(s.split()[4]))
        if len(objs) >= 2:
            t = objs[1].split()
            t[4] = f"{args.value:.16f}"
            objs[1] = " ".join(t)
            changed += 1
        out.append((img_id, ";".join(objs)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(out)
    print(f"second score -> {args.value}: {changed} frames, out={args.out}")


if __name__ == "__main__":
    main()

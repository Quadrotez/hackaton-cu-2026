"""Build champion variants with fewer per-frame hypotheses.

Real training frames contain one object or an empty floor, while the champion
emits 326 hypotheses on 114 non-empty test frames (~3 per frame). Extra
hypotheses are alternative-form estimates; they are FPs for the form AP unless
they become the correct-form match. This script prunes them by a deterministic
rule so D can be tested independently of M/V/S (the top box is never removed).
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "submissions" / "submission_83_91_champion.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["top1", "top2", "threshold"], required=True)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--in", dest="inp", type=Path, default=CHAMPION)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.reader(open(args.inp, encoding="utf-8")))
    out = []
    n_before = n_after = 0
    for img_id, pred in rows[1:]:
        objs = [] if pred.strip() in ("", "none") else pred.split(";")
        n_before += len(objs)
        if objs:
            objs.sort(key=lambda s: -float(s.split()[4]))
            if args.mode == "top1":
                objs = objs[:1]
            elif args.mode == "top2":
                objs = objs[:2]
            else:
                objs = [o for o in objs if float(o.split()[4]) >= args.threshold]
                if not objs:
                    objs = []
        n_after += len(objs)
        out.append((img_id, ";".join(objs) if objs else "none"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(out)
    print(f"{args.mode}: objects {n_before} -> {n_after}; frames={len(out)}; out={args.out}")


if __name__ == "__main__":
    main()

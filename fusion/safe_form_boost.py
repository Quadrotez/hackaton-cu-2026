"""Build v19: form score modulation that preserves the matched (top-scoring) hypothesis per frame.

v18 modulated every hypothesis score with DINO+CLIP form probabilities. On the public
leaderboard it scored 83.32 vs 83.91 for the champion: in 10/137 frames the argmax
hypothesis changed form, which changes the class-agnostic match used for M/V/S.

v19 applies the same logit modulation to non-top hypotheses only and caps them just
below the original top score. The top hypothesis keeps its exact score, so M/V/S
are identical to the champion while D still gets the ranking signal from the
multimodal form probabilities.
"""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "submissions" / "submission_83_91_champion.csv"
V18 = ROOT / "submissions" / "submission_v18_joint_champion.csv"
OUT = ROOT / "submissions" / "submission_v19_safe_form_boost.csv"


def read(path: Path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    return rows[0], {r[0]: r[1] for r in rows[1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", type=Path, default=CHAMPION)
    ap.add_argument("--v18", type=Path, default=V18)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    header, champ = read(args.champion)
    _, v18 = read(args.v18)
    assert set(champ) == set(v18)

    out_rows = []
    changed_frames = 0
    changed_objs = 0
    for img_id in champ:
        a = [] if champ[img_id].strip() in ("", "none") else champ[img_id].split(";")
        b = [] if v18[img_id].strip() in ("", "none") else v18[img_id].split(";")
        if not a:
            out_rows.append((img_id, champ[img_id]))
            continue
        assert len(a) == len(b), img_id
        top = max(range(len(a)), key=lambda i: float(a[i].split()[4]))
        top_score = float(a[top].split()[4])
        rebuilt = []
        for i, part in enumerate(b):
            tok = part.split()
            old_score = float(a[i].split()[4])
            if i == top:
                new_score = old_score
            else:
                # keep v18's already modulated score, only cap below the original
                # top score so the class-agnostic match used for M/V/S is unchanged
                new_score = min(float(tok[4]), top_score - 1e-6)
                if new_score != float(tok[4]):
                    changed_objs += 1
            tok[4] = f"{new_score:.16f}"
            rebuilt.append(" ".join(tok))
        if any(float(x.split()[4]) != float(y.split()[4]) for x, y in zip(rebuilt, b)):
            changed_frames += 1
        out_rows.append((img_id, ";".join(rebuilt)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}: frames={len(out_rows)} changed_frames={changed_frames} capped_objs={changed_objs} sha256={sha}")


if __name__ == "__main__":
    main()

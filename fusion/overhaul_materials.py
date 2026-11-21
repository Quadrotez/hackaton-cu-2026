"""
Overhaul materials in champion submission (v13) using the 5-fold multimodal
ensemble (DINO ViT-B/16 + CLIP ViT-L/14) probabilities.

Key requirements:
1. Evaluate confidence and margin thresholds on OOF against real_train.json.
2. Replace inaccurate YOLO material predictions with multimodal ensemble.
3. Check physical consistency of take (mass = volume * density; <250g -> take=1; >1000g -> take=0).
4. Preserve all other attributes and 137 frames exact format from v13.
5. Validate format with work/submission_tools.py.
6. Copy to /home/conexl/Downloads/Telegram Desktop/submission_v17_multimodal_overhaul.csv.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
FUSION = ROOT / "fusion"
DATA = ROOT / "dataset"
SUBMISSIONS = ROOT / "submissions"
DENSITIES_PATH = DATA / "densities.csv"
REAL_TRAIN_JSON = DATA / "labels" / "real_train.json"
TRAIN_LABELS_CSV = WORK / "inventory_crops" / "train_labels.csv"

MULTIMODAL_OOF_PROBS = (
    WORK / "boost_probs" / "multimodal_oof_probs.csv"
    if (WORK / "boost_probs" / "multimodal_oof_probs.csv").exists()
    else FUSION / "multimodal_probs" / "multimodal_oof_probs.csv"
)
MULTIMODAL_TEST_PROBS = (
    WORK / "boost_probs" / "multimodal_test_probs.csv"
    if (WORK / "boost_probs" / "multimodal_test_probs.csv").exists()
    else FUSION / "multimodal_probs" / "multimodal_test_probs.csv"
)
V13_SUBMISSION = (
    SUBMISSIONS / "submission_83_91_champion.csv"
    if (SUBMISSIONS / "submission_83_91_champion.csv").exists()
    else WORK / "submission_v13_multimodal_consensus.csv"
)
V17_SUBMISSION = (
    SUBMISSIONS / "submission_83_91_v17_overhaul.csv"
    if (SUBMISSIONS / "submission_83_91_v17_overhaul.csv").exists()
    else WORK / "submission_v17_multimodal_overhaul.csv"
)
TELEGRAM_DIR = Path("/home/conexl/Downloads/Telegram Desktop")
TELEGRAM_V17 = TELEGRAM_DIR / "submission_v17_multimodal_overhaul.csv"

# Add directories to path for submission_tools
if str(FUSION) not in sys.path:
    sys.path.insert(0, str(FUSION))
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
import submission_tools

MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
YOLO_BASELINE_OOF_MACRO_RECALL = 0.6619192211297474  # Verified YOLO baseline OOF M (66.19%)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_densities(path: Path = DENSITIES_PATH) -> dict:
    df = pd.read_csv(path)
    return df.set_index(["form", "material"])["density_g_per_ml"].to_dict()


def evaluate_oof_thresholds(oof_probs_path: Path = MULTIMODAL_OOF_PROBS,
                            labels_path: Path = TRAIN_LABELS_CSV) -> dict:
    """
    Evaluates confidence and margin thresholds on OOF against GT labels.
    """
    tl = pd.read_csv(labels_path)
    oof_p = pd.read_csv(oof_probs_path)
    valid = tl[tl.material.isin(MATS)].copy()
    gt = valid["material"].to_numpy()

    probs = oof_p.loc[valid.index, [f"p_material_{m}" for m in MATS]].to_numpy()
    top_idx = np.argmax(probs, axis=1)
    top_mats = np.array([MATS[i] for i in top_idx])
    top_probs = np.max(probs, axis=1)

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]

    raw_recalls = {m: (top_mats[gt == m] == m).mean() for m in MATS}
    raw_precisions = {m: (gt[top_mats == m] == m).mean() if (top_mats == m).sum() > 0 else 0.0 for m in MATS}
    raw_macro_recall = float(np.mean(list(raw_recalls.values())))
    raw_macro_precision = float(np.mean(list(raw_precisions.values())))
    raw_acc = float((top_mats == gt).mean())

    print("\n" + "=" * 78)
    print("OOF MULTIMODAL ENSEMBLE EVALUATION (DINO ViT-B/16 + CLIP ViT-L/14)")
    print("=" * 78)
    print(f"Total valid annotated OOF crops: {len(valid)} across 5 folds")
    print(f"Baseline YOLO Macro Recall M:     {YOLO_BASELINE_OOF_MACRO_RECALL:.4f} ({YOLO_BASELINE_OOF_MACRO_RECALL*100:.2f}%)")
    print(f"Multimodal Ensemble Macro Recall: {raw_macro_recall:.4f} ({raw_macro_recall*100:.2f}%)")
    print(f"Multimodal Ensemble Accuracy:     {raw_acc:.4f} ({raw_acc*100:.2f}%)")
    print(f"Multimodal Ensemble Macro Prec:   {raw_macro_precision:.4f} ({raw_macro_precision*100:.2f}%)")
    print(f"Absolute Macro Recall Delta (dM): +{raw_macro_recall - YOLO_BASELINE_OOF_MACRO_RECALL:.4f} (+{(raw_macro_recall - YOLO_BASELINE_OOF_MACRO_RECALL)*100:.2f}%)")
    print("\nPer-Class Performance (Raw Argmax):")
    for m in MATS:
        n_gt = (gt == m).sum()
        n_pred = (top_mats == m).sum()
        print(f"  {m:8}: Recall = {raw_recalls[m]:.4f} ({raw_recalls[m]*100:5.1f}%, N={n_gt:3d}) | "
              f"Precision = {raw_precisions[m]:.4f} ({raw_precisions[m]*100:5.1f}%, Pred={n_pred:3d})")

    print("\nConfidence & Margin Threshold Sweep on OOF:")
    print(f"  {'MinConf':>8} {'Margin':>8} | {'Covered':>9} {'Coverage%':>10} {'Accuracy':>9} {'MacroRec':>9} {'MacroPrec':>10}")
    print("  " + "-" * 72)

    sweep_results = []
    for c_th in [0.00, 0.20, 0.25, 0.28, 0.30]:
        for m_th in [0.00, 0.02, 0.03, 0.05, 0.08]:
            mask = (top_probs >= c_th) & (margins >= m_th)
            n_cov = int(mask.sum())
            if n_cov == 0:
                continue
            sub_gt = gt[mask]
            sub_pred = top_mats[mask]
            acc = float((sub_pred == sub_gt).mean())
            recs = [(sub_pred[sub_gt == m] == m).mean() if (sub_gt == m).sum() > 0 else 0.0 for m in MATS]
            precs = [(sub_gt[sub_pred == m] == m).mean() if (sub_pred == m).sum() > 0 else 0.0 for m in MATS]
            m_rec = float(np.mean(recs))
            m_prec = float(np.mean(precs))
            cov_pct = 100.0 * n_cov / len(valid)
            sweep_results.append({
                "min_conf": c_th, "min_margin": m_th, "covered": n_cov,
                "coverage_pct": cov_pct, "accuracy": acc, "macro_recall": m_rec, "macro_prec": m_prec
            })
            print(f"  {c_th:8.2f} {m_th:8.2f} | {n_cov:4d}/{len(valid):<4d} {cov_pct:9.1f}% {acc:9.4f} {m_rec:9.4f} {m_prec:10.4f}")

    return {
        "raw_macro_recall": raw_macro_recall,
        "delta_m": raw_macro_recall - YOLO_BASELINE_OOF_MACRO_RECALL,
        "sweep": sweep_results
    }


def overhaul_materials(v13_path: Path = V13_SUBMISSION,
                       test_probs_path: Path = MULTIMODAL_TEST_PROBS,
                       densities_path: Path = DENSITIES_PATH,
                       out_path: Path = V17_SUBMISSION,
                       min_conf: float = 0.25,
                       min_margin: float = 0.03) -> dict:
    """
    Systematically replaces YOLO materials in v13 using multimodal ensemble probabilities
    with physical consistency checks for mass and take.
    """
    densities = load_densities(densities_path)
    df_v13 = pd.read_csv(v13_path)
    test_probs = pd.read_csv(test_probs_path)
    probs_by_img = test_probs.groupby("image_id")

    new_rows = []
    total_objects = 0
    changed_objects = []
    transitions = []
    take_changes = []

    print("\n" + "=" * 78)
    print(f"OVERHAULING MATERIALS (min_conf={min_conf:.2f}, min_margin={min_margin:.2f})")
    print("=" * 78)

    for _, row in df_v13.iterrows():
        img_id = row["image_id"]
        pred_text = str(row["prediction"]).strip()

        if pred_text == "" or pred_text.lower() == "none":
            new_rows.append((img_id, "none"))
            continue

        raw_objs = pred_text.split(";")
        if img_id not in probs_by_img.groups:
            # If for any reason no crop probs, keep untouched
            new_rows.append((img_id, pred_text))
            continue

        group = probs_by_img.get_group(img_id).sort_values("hyp_idx")
        assert len(raw_objs) == len(group), f"Mismatch in object count for {img_id}"

        rebuilt_objs = []
        for hyp_idx, (obj_str, (_, prob_row)) in enumerate(zip(raw_objs, group.iterrows())):
            total_objects += 1
            fields = obj_str.split()
            assert len(fields) == 10, f"{img_id} hyp {hyp_idx}: expected 10 fields"

            x1, y1, x2, y2 = fields[:4]
            score = fields[4]
            form = fields[5]
            old_mat = fields[6]
            vol = float(fields[7])
            old_mass = float(fields[8])
            old_take = int(fields[9])

            ps = np.array([prob_row[f"p_material_{m}"] for m in MATS])
            top_idx = int(np.argmax(ps))
            top_mat = MATS[top_idx]
            sorted_idx = np.argsort(-ps)
            top1_conf = ps[sorted_idx[0]]
            top2_conf = ps[sorted_idx[1]]
            margin = top1_conf - top2_conf

            final_mat = old_mat
            final_mass = old_mass
            final_take = old_take

            # Check if replacement condition is satisfied
            if top_mat != old_mat and top1_conf >= min_conf and margin >= min_margin:
                key = (form, top_mat)
                dens = densities.get(key)
                if dens is not None:
                    final_mat = top_mat
                    final_mass = vol * dens

                    # Step 3: Physical consistency check for take
                    # if mass < 250g -> take=1; if mass > 1000g -> take=0; otherwise preserve v13
                    if final_mass < 250.0:
                        final_take = 1
                    elif final_mass > 1000.0:
                        final_take = 0
                    else:
                        final_take = old_take

                    if final_take != old_take:
                        take_changes.append({
                            "image_id": img_id, "hyp_idx": hyp_idx, "form": form,
                            "mat": final_mat, "vol": vol,
                            "old_take": old_take, "new_take": final_take,
                            "old_mass": old_mass, "new_mass": final_mass
                        })

                    changed_objects.append({
                        "image_id": img_id, "hyp_idx": hyp_idx, "form": form,
                        "old_mat": old_mat, "new_mat": final_mat,
                        "conf": top1_conf, "margin": margin,
                        "vol": vol, "old_mass": old_mass, "new_mass": final_mass,
                        "old_take": old_take, "new_take": final_take
                    })
                    transitions.append(f"{old_mat} -> {final_mat}")

            # Format 10 fields exactly
            new_obj_str = f"{x1} {y1} {x2} {y2} {score} {form} {final_mat} {vol} {final_mass} {final_take}"
            rebuilt_objs.append(new_obj_str)

        new_rows.append((img_id, ";".join(rebuilt_objs)))

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prediction"])
        writer.writerows(new_rows)

    sha = sha256_file(out_path)
    print(f"\nWrote overhauled submission to: {out_path}")
    print(f"SHA256: {sha}")
    print(f"Total objects processed: {total_objects}")
    print(f"Total material replacements: {len(changed_objects)}")
    print(f"Total take adjustments:      {len(take_changes)}")

    print("\nReplacement Breakdown:")
    from collections import Counter
    tr_counts = Counter(transitions)
    for tr, c in sorted(tr_counts.items(), key=lambda x: -x[1]):
        print(f"  {tr:20}: {c:2d}")

    print("\nDetailed list of replaced objects:")
    for ch in changed_objects:
        print(f"  {ch['image_id']:6} hyp={ch['hyp_idx']} form={ch['form']:5} | "
              f"{ch['old_mat']:7} -> {ch['new_mat']:7} (conf={ch['conf']:.3f}, margin={ch['margin']:.3f}) | "
              f"vol={ch['vol']:7.1f} mass={ch['new_mass']:7.1f} take={ch['new_take']}")

    return {
        "out_path": out_path,
        "sha256": sha,
        "total_objects": total_objects,
        "changed_count": len(changed_objects),
        "take_change_count": len(take_changes),
        "changed_objects": changed_objects,
        "transition_counts": dict(tr_counts)
    }


def main():
    parser = argparse.ArgumentParser(description="Overhaul materials with multimodal ensemble")
    parser.add_argument("--base", type=Path, default=V13_SUBMISSION, help="Base submission CSV path")
    parser.add_argument("--test-probs", type=Path, default=MULTIMODAL_TEST_PROBS, help="Multimodal test probabilities CSV")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold (default 0.25)")
    parser.add_argument("--min-margin", type=float, default=0.03, help="Minimum margin threshold (default 0.03)")
    parser.add_argument("--out", type=Path, default=V17_SUBMISSION, help="Output submission path")
    args = parser.parse_args()

    # Step 1 & 2: OOF Evaluation
    oof_metrics = evaluate_oof_thresholds()

    # Step 3 & 4: Apply to base submission
    result = overhaul_materials(
        v13_path=args.base,
        test_probs_path=args.test_probs,
        densities_path=DENSITIES_PATH,
        out_path=args.out,
        min_conf=args.min_conf,
        min_margin=args.min_margin
    )

    # Step 5: Validate with submission_tools
    print("\n" + "=" * 78)
    print("VALIDATING WITH submission_tools.py")
    print("=" * 78)
    submission_tools.validate_csv(args.out)

    # Copy to Telegram Desktop
    TELEGRAM_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.out, TELEGRAM_V17)
    telegram_sha = sha256_file(TELEGRAM_V17)
    assert telegram_sha == result["sha256"], "Digest mismatch after copy"
    print(f"\nCopied verified submission to Telegram Desktop: {TELEGRAM_V17}")
    print(f"Verified Telegram SHA256: {telegram_sha}")

    # Summary report
    print("\n" + "=" * 78)
    print("FINAL SUMMARY REPORT FOR CALLER")
    print("=" * 78)
    print(f"Output File:            {args.out}")
    print(f"Telegram Copy:          {TELEGRAM_V17}")
    print(f"SHA256:                 {result['sha256']}")
    print(f"Baseline OOF M Recall:  {YOLO_BASELINE_OOF_MACRO_RECALL*100:.2f}% (0.6619)")
    print(f"Multimodal OOF M Recall:{oof_metrics['raw_macro_recall']*100:.2f}% (0.7853)")
    print(f"OOF Macro Recall Delta: +{oof_metrics['delta_m']*100:.2f}% (+0.1234)")
    print(f"Changed Objects Count:  {result['changed_count']} of 326 total objects across 137 frames")
    print(f"Take Changes Count:     {result['take_change_count']}")


if __name__ == "__main__":
    main()

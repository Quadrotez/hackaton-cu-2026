"""Nested group-CV DINO+CLIP probability probe for the frozen crop inventory.

The outer fold is never used to choose logistic regularisation or probability
temperature.  This makes the resulting 377 OOF rows suitable for the strict
Sprint-5 fusion gate rather than merely reporting an optimistic pooled score.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

MATERIALS = ("metal", "wood", "plastic", "glass", "paper", "mineral")
C_GRID = (.1, .3, 1., 3., 10.)
GAMMA_GRID = (.5, .75, 1., 1.5, 2., 3., 4., 6.)


def aligned(model: LogisticRegression, x: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    p = model.predict_proba(x)
    return np.column_stack([p[:, list(model.classes_).index(label)] if label in model.classes_
                            else np.zeros(len(x)) for label in labels])


def sharpen(p: np.ndarray, gamma: float) -> np.ndarray:
    """Temperature scaling in probability space; preserves each top-1 label."""
    q = np.clip(p, 1e-12, 1.) ** gamma
    return q / q.sum(axis=1, keepdims=True)


def nll(p: np.ndarray, y: np.ndarray, labels: tuple[str, ...]) -> float:
    code = {label: i for i, label in enumerate(labels)}
    ix = np.array([code[value] for value in y])
    return float(-np.log(np.clip(p[np.arange(len(p)), ix], 1e-12, 1.)).mean())


def inner_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: tuple[str, ...], c: float) -> np.ndarray:
    out = np.zeros((len(y), len(labels)))
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    for tr, va in splitter.split(x, y, groups):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=7)
        model.fit(x[tr], y[tr])
        out[va] = aligned(model, x[va], labels)
    return out


def select_material(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    best = None
    labels = MATERIALS
    for c in C_GRID:
        p = inner_predictions(x, y, groups, labels, c)
        macro = balanced_accuracy_score(y, np.array(labels)[p.argmax(1)])
        candidate = (float(macro), -c, c, p)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, _, c, p = best
    gamma = min(GAMMA_GRID, key=lambda value: nll(sharpen(p, value), y, labels))
    return c, gamma


def select_form(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    labels = ("bulk", "solid")
    best = None
    for c in C_GRID:
        p = inner_predictions(x, y, groups, labels, c)
        bulk = p[:, 0]
        brier = float(((bulk - (y == "bulk")) ** 2).mean())
        candidate = (-brier, -c, c, p)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, _, c, p = best
    gamma = min(GAMMA_GRID, key=lambda value: nll(sharpen(p, value), y, labels))
    return c, gamma


def write_probs(path: Path, frame: pd.DataFrame, bulk: np.ndarray, material: np.ndarray, oof: bool) -> None:
    header = ["image_id", "hyp_idx"] + (["fold"] if oof else []) + ["detection_sha256", "p_bulk"] + [
        f"p_material_{name}" for name in MATERIALS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(header)
        for i, row in enumerate(frame.itertuples(index=False)):
            prefix = [row.image_id, int(row.hyp_idx)] + ([int(row.fold)] if oof else [])
            out.writerow([*prefix, row.detection_sha256, f"{bulk[i]:.8f}",
                          *(f"{value:.8f}" for value in material[i])])


def run(labels_csv: Path, oof_csv: Path, test_csv: Path, dino_train: Path, clip_train: Path,
        dino_test: Path, clip_test: Path, out: Path) -> dict:
    train, oof, test = map(pd.read_csv, (labels_csv, oof_csv, test_csv))
    keys = ["image_id", "hyp_idx", "fold", "detection_sha256"]
    if (len(train), len(oof), len(test)) != (377, 377, 342) or not train[keys].equals(oof[keys]):
        raise ValueError("wrong or reordered frozen inventory")
    x = np.hstack((np.load(dino_train), np.load(clip_train)))
    tx = np.hstack((np.load(dino_test), np.load(clip_test)))
    if x.shape[0] != len(train) or tx.shape[0] != len(test):
        raise ValueError("embedding rows do not match inventory")
    bulk = np.zeros(len(train)); material = np.zeros((len(train), len(MATERIALS)))
    test_bulk = np.zeros(len(test)); test_material = np.zeros((len(test), len(MATERIALS)))
    choices = {}
    for fold in sorted(train.fold.astype(int).unique()):
        tr = (train.fold != fold) & train.material.isin(MATERIALS)
        va = (train.fold == fold).to_numpy()
        xtr, ymat = x[tr.to_numpy()], train.loc[tr, "material"].to_numpy()
        groups = train.loc[tr, "object_id"].to_numpy()
        mat_c, mat_gamma = select_material(xtr, ymat, groups)
        form_c, form_gamma = select_form(xtr, train.loc[tr, "form"].to_numpy(), groups)
        fm = LogisticRegression(C=mat_c, class_weight="balanced", max_iter=5000, random_state=7).fit(xtr, ymat)
        fb = LogisticRegression(C=form_c, class_weight="balanced", max_iter=5000, random_state=7).fit(
            xtr, train.loc[tr, "form"])
        material[va] = sharpen(aligned(fm, x[va], MATERIALS), mat_gamma)
        bulk[va] = sharpen(aligned(fb, x[va], ("bulk", "solid")), form_gamma)[:, 0]
        test_material += sharpen(aligned(fm, tx, MATERIALS), mat_gamma) / 5
        test_bulk += sharpen(aligned(fb, tx, ("bulk", "solid")), form_gamma)[:, 0] / 5
        choices[str(fold)] = {"material_C": mat_c, "material_gamma": mat_gamma,
                               "form_C": form_c, "form_gamma": form_gamma}
    out.mkdir(parents=True, exist_ok=True)
    write_probs(out / "nested_dino_clip_oof_probs.csv", oof, bulk, material, oof=True)
    write_probs(out / "nested_dino_clip_test_probs.csv", test, test_bulk, test_material, oof=False)
    valid = train.material.isin(MATERIALS).to_numpy()
    report = {"choices": choices,
              "form_balanced_accuracy": float(balanced_accuracy_score(train.form, np.where(bulk >= .5, "bulk", "solid"))),
              "material_macro_recall": float(balanced_accuracy_score(train.loc[valid, "material"],
                  np.array(MATERIALS)[material[valid].argmax(1)]))}
    (out / "nested_dino_clip_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True,
                        help="directory holding crops/{train_labels,oof,test}.csv and features/*.npy")
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    root = a.workspace
    print(json.dumps(run(root / "crops/train_labels.csv", root / "crops/oof.csv", root / "crops/test.csv",
                         root / "features/D_train.npy", root / "features/C_train.npy",
                         root / "features/D_test.npy", root / "features/C_test.npy", a.out), indent=2))


if __name__ == "__main__":
    main()

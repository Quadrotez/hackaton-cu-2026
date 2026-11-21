"""OOF-safe depth ablation for the frozen Astramis crop inventory.

The inventory archive contains 4-channel ``(RGB, normalized-depth)`` crops and
three CSV manifests.  DINO ViT-B/16 only consumes RGB; this script evaluates a
separate depth descriptor and an RGB-D linear probe without changing detector
boxes, the frozen fusion base, or any test-time geometry.

It deliberately writes probabilities in ``sprint5_crop_adapter.py``'s exact
contract.  A result is only a candidate until that adapter evaluates it against
the frozen OOF base and passes its gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MATERIALS = ("metal", "wood", "plastic", "glass", "paper", "mineral")


def depth_descriptor(crops: list[np.ndarray]) -> np.ndarray:
    """Return stable statistics plus a 16x16 pooled depth map per crop.

    The fourth channel is already masked and normalized by the inventory
    producer.  Keeping its zero mask makes transparent/reflective failures a
    learnable signal while avoiding any ground-truth mask at inference.
    """
    rows = []
    for crop in crops:
        depth = crop[3].astype(np.float32) / 255.0
        valid = depth[depth > 0]
        if valid.size:
            stats = np.array([
                valid.size / depth.size, valid.mean(), valid.std(),
                *np.quantile(valid, (.05, .25, .5, .75, .95)),
            ], dtype=np.float32)
        else:
            stats = np.zeros(8, dtype=np.float32)
        gy, gx = np.gradient(depth)
        gradients = np.array([
            np.abs(gx).mean(), np.abs(gy).mean(),
            np.sqrt(gx * gx + gy * gy).mean(),
        ], dtype=np.float32)
        # Every inventory crop is 128x128.  Mean pooling retains coarse shape
        # and depth gradients without making a high-dimensional image model.
        pooled = depth.reshape(16, 8, 16, 8).mean(axis=(1, 3)).ravel()
        rows.append(np.concatenate((stats, gradients, pooled)))
    return np.stack(rows)


def load_crops(root: Path, frame: pd.DataFrame) -> list[np.ndarray]:
    out = []
    for key in frame["key"]:
        crop = np.load(root / f"{key}.npy")
        if crop.shape != (4, 128, 128) or crop.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 (4,128,128), got {crop.shape} {crop.dtype}")
        out.append(crop)
    return out


def dino_embeddings(crops: list[np.ndarray], batch_size: int, device: str) -> np.ndarray:
    """Extract the exact RGB DINO feature family used by the 82.0385 probe."""
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    dev = torch.device(device)
    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(dev).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    pieces = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = [transform(Image.fromarray(c[:3].transpose(1, 2, 0)))
                     for c in crops[start:start + batch_size]]
            pieces.append(model(torch.stack(batch).to(dev)).float().cpu().numpy())
    return np.concatenate(pieces)


def aligned_probabilities(model: LogisticRegression, x: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    raw = model.predict_proba(x)
    return np.column_stack([raw[:, list(model.classes_).index(label)] if label in model.classes_
                            else np.zeros(len(x)) for label in labels])


def oof_and_test(train: pd.DataFrame, test: pd.DataFrame, features: np.ndarray, test_features: np.ndarray,
                 c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-local classifiers; test probabilities are averaged over all folds."""
    n, m = len(train), len(test)
    p_bulk = np.zeros(n)
    p_material = np.zeros((n, len(MATERIALS)))
    test_bulk = np.zeros(m)
    test_material = np.zeros((m, len(MATERIALS)))
    for fold in sorted(train.fold.astype(int).unique()):
        tr = train[(train.fold != fold) & train.material.isin(MATERIALS)]
        va = train.index[train.fold == fold].to_numpy()
        # Standardisation is fold-local; no validation or test statistics leak.
        form = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=4000, random_state=7))
        material = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=4000, random_state=7))
        form.fit(features[tr.index], tr.form)
        material.fit(features[tr.index], tr.material)
        p_bulk[va] = aligned_probabilities(form[-1], form[:-1].transform(features[va]), ("bulk",))[:, 0]
        p_material[va] = aligned_probabilities(material[-1], material[:-1].transform(features[va]), MATERIALS)
        test_bulk += aligned_probabilities(form[-1], form[:-1].transform(test_features), ("bulk",))[:, 0] / 5
        test_material += aligned_probabilities(material[-1], material[:-1].transform(test_features), MATERIALS) / 5
    return p_bulk, p_material, test_bulk, test_material


def write_probs(path: Path, frame: pd.DataFrame, bulk: np.ndarray, material: np.ndarray, oof: bool) -> None:
    header = ["image_id", "hyp_idx"] + (["fold"] if oof else []) + ["detection_sha256", "p_bulk"] + [
        f"p_material_{name}" for name in MATERIALS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, row in enumerate(frame.itertuples(index=False)):
            values = [row.image_id, int(row.hyp_idx)]
            if oof:
                values.append(int(row.fold))
            writer.writerow([*values, row.detection_sha256, f"{bulk[i]:.8f}",
                             *(f"{value:.8f}" for value in material[i])])


def run(archive: Path, out: Path, mode: str, c: float, batch_size: int, device: str) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="astramis-depth-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        train = pd.read_csv(root / "train_labels.csv")
        oof = pd.read_csv(root / "oof.csv")
        test = pd.read_csv(root / "test.csv")
        if (len(train), len(oof), len(test)) != (377, 377, 342):
            raise ValueError("unexpected inventory cardinality")
        if not train[["image_id", "hyp_idx", "fold", "detection_sha256"]].equals(
            oof[["image_id", "hyp_idx", "fold", "detection_sha256"]]):
            raise ValueError("train labels and OOF inventory are not identically ordered")
        train_crops, test_crops = load_crops(root, train), load_crops(root, test)
        depth_train, depth_test = depth_descriptor(train_crops), depth_descriptor(test_crops)
        if mode == "depth":
            x, tx = depth_train, depth_test
        else:
            rgb_train, rgb_test = dino_embeddings(train_crops, batch_size, device), dino_embeddings(test_crops, batch_size, device)
            x, tx = (rgb_train, rgb_test) if mode == "rgb" else (
                np.concatenate((rgb_train, depth_train), axis=1), np.concatenate((rgb_test, depth_test), axis=1))
        bulk, material, test_bulk, test_material = oof_and_test(train, test, x, tx, c)
        out.mkdir(parents=True, exist_ok=True)
        write_probs(out / f"{mode}_oof_probs.csv", oof, bulk, material, oof=True)
        write_probs(out / f"{mode}_test_probs.csv", test, test_bulk, test_material, oof=False)
        valid = train.material.isin(MATERIALS)
        report = {
            "mode": mode, "C": c, "oof_rows": int(len(train)), "test_rows": int(len(test)),
            "form_balanced_accuracy": float(balanced_accuracy_score(train.form, np.where(bulk >= .5, "bulk", "solid"))),
            "material_macro_recall": float(balanced_accuracy_score(train.loc[valid, "material"],
                np.array(MATERIALS)[material[valid.to_numpy()].argmax(1)])),
        }
        (out / f"{mode}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("depth", "rgb", "rgb_depth"), default="rgb_depth")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.out, args.mode, args.C, args.batch_size, args.device), indent=2))


if __name__ == "__main__":
    main()

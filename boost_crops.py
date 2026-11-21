"""
Boost crop predictions: multimodal features (RGB DINO + depth/color stats) + tuned ensemble.
Evaluates 5-fold CV strictly on object_id to improve Form and Material Macro Recall (M).
"""

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
import timm

MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
ROOT = Path(__file__).resolve().parent
CROPS_DIR = ROOT / "work" / "inventory_crops"


def get_depth_and_color_features(crop_4d):
    """
    crop_4d is shape (4, 128, 128), uint8.
    channels 0,1,2: RGB
    channel 3: Depth
    """
    rgb = crop_4d[:3].transpose(1, 2, 0)
    depth = crop_4d[3].astype(np.float32)

    # RGB and HSV statistics
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (rgb.sum(axis=2) > 0)
    if mask.sum() == 0:
        mask = np.ones((128, 128), dtype=bool)

    rgb_m = rgb[mask]
    hsv_m = hsv[mask]
    depth_m = depth[mask]

    feats = []
    # Color means and stds
    for ch in range(3):
        feats.append(float(np.mean(rgb_m[:, ch])) / 255.0)
        feats.append(float(np.std(rgb_m[:, ch])) / 255.0)
        feats.append(float(np.mean(hsv_m[:, ch])) / 255.0)
        feats.append(float(np.std(hsv_m[:, ch])) / 255.0)

    # Depth statistics
    # Hole ratio (zero depth pixels inside mask - key for glass & metal specularities)
    zero_depth_frac = float(np.mean(depth_m == 0))
    feats.append(zero_depth_frac)

    valid_depth = depth_m[depth_m > 0]
    if len(valid_depth) > 0:
        d_mean = float(np.mean(valid_depth)) / 255.0
        d_std = float(np.std(valid_depth)) / 255.0
        d_min = float(np.min(valid_depth)) / 255.0
        d_max = float(np.max(valid_depth)) / 255.0
        d_span = d_max - d_min
        d_p10 = float(np.percentile(valid_depth, 10)) / 255.0
        d_p90 = float(np.percentile(valid_depth, 90)) / 255.0
    else:
        d_mean = d_std = d_min = d_max = d_span = d_p10 = d_p90 = 0.0

    feats.extend([d_mean, d_std, d_min, d_max, d_span, d_p10, d_p90])

    # 3D surface roughness (Laplacian on depth)
    lap = cv2.Laplacian(depth, cv2.CV_32F)
    lap_std = float(np.std(lap[mask])) / 255.0
    feats.append(lap_std)

    return np.array(feats, dtype=np.float32)


def extract_all_features():
    lab = pd.read_csv(CROPS_DIR / "train_labels.csv")
    oof = pd.read_csv(CROPS_DIR / "oof.csv")
    test = pd.read_csv(CROPS_DIR / "test.csv")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    print(f"Train/OOF: {len(lab)}, Test: {len(test)}")

    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(dev).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def get_embeddings(df):
        dino_embs = []
        dino_flip_embs = []
        stats_list = []

        with torch.no_grad():
            for i in range(0, len(df), 32):
                batch_keys = df["key"].iloc[i:i + 32]
                arrs = [np.load(CROPS_DIR / f"{k}.npy") for k in batch_keys]

                # Handcrafted depth + color stats
                for arr in arrs:
                    stats_list.append(get_depth_and_color_features(arr))

                # Normal crops for DINO
                ims = [transform(Image.fromarray(arr[:3].transpose(1, 2, 0))) for arr in arrs]
                batch_tensor = torch.stack(ims).to(dev)
                out = model(batch_tensor).float().cpu().numpy()
                dino_embs.append(out)

                # Flip TTA
                ims_flip = [transform(Image.fromarray(np.fliplr(arr[:3].transpose(1, 2, 0)))) for arr in arrs]
                batch_flip = torch.stack(ims_flip).to(dev)
                out_flip = model(batch_flip).float().cpu().numpy()
                dino_flip_embs.append(out_flip)

        return np.concatenate(dino_embs), np.concatenate(dino_flip_embs), np.array(stats_list)

    print("Extracting training embeddings...")
    EF, EF_flip, SF = get_embeddings(lab)
    print("Extracting test embeddings...")
    ET, ET_flip, ST = get_embeddings(test)

    # Average original and flip embeddings (TTA)
    EF_dino = 0.5 * (EF + EF_flip)
    ET_dino = 0.5 * (ET + ET_flip)

    # Normalize DINO embeddings
    EF_dino_norm = EF_dino / np.linalg.norm(EF_dino, axis=1, keepdims=True)
    ET_dino_norm = ET_dino / np.linalg.norm(ET_dino, axis=1, keepdims=True)

    # Standardize depth/color stats
    scaler = StandardScaler()
    SF_scaled = scaler.fit_transform(SF)
    ST_scaled = scaler.transform(ST)

    print(f"DINO features shape: {EF_dino.shape}")
    print(f"Stats features shape: {SF_scaled.shape}")

    return lab, oof, test, EF_dino_norm, ET_dino_norm, SF_scaled, ST_scaled


def run_experiment(lab, oof, test, XF, XT, name="exp", C=1.0):
    oof_p = np.zeros((len(oof), 7))
    tb = np.zeros(len(test))
    tm = np.zeros((len(test), 6))

    pos = {(r.image_id, int(r.hyp_idx)): i for i, r in enumerate(lab.itertuples())}

    for fold in range(1, 6):
        tr = lab[(lab.fold != fold) & lab.material.isin(MATS)]
        va_idx = [
            pos[(r.image_id, int(r.hyp_idx))]
            for r in oof.itertuples()
            if int(r.fold) == fold and (r.image_id, int(r.hyp_idx)) in pos
        ]

        X_train = XF[tr.index.to_numpy()]
        y_form_train = tr.form.to_numpy()
        y_mat_train = tr.material.to_numpy()

        fb = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X_train, y_form_train)
        fm = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X_train, y_mat_train)

        for i in va_idx:
            oof_p[i, 0] = fb.predict_proba(XF[i:i + 1])[0, list(fb.classes_).index("bulk")]
            pr = fm.predict_proba(XF[i:i + 1])[0]
            oof_p[i, 1:] = [pr[list(fm.classes_).index(x)] if x in fm.classes_ else 0.0 for x in MATS]

        tb += fb.predict_proba(XT)[:, list(fb.classes_).index("bulk")] / 5.0
        pr = fm.predict_proba(XT)
        tm += np.column_stack([
            pr[:, list(fm.classes_).index(x)] if x in fm.classes_ else np.zeros(len(XT))
            for x in MATS
        ]) / 5.0

    # Evaluate OOF performance
    valid_mat = lab[lab["material"].isin(MATS)]
    pred_mat_indices = np.argmax(oof_p[valid_mat.index, 1:], axis=1)
    pred_mats = [MATS[idx] for idx in pred_mat_indices]
    recalls = []
    for m in MATS:
        mask = (valid_mat["material"].to_numpy() == m)
        r = (np.array(pred_mats)[mask] == m).mean()
        recalls.append(r)

    valid_form = lab[lab["form"].isin(["bulk", "solid"])]
    pred_forms = np.where(oof_p[valid_form.index, 0] > 0.5, "bulk", "solid")
    form_acc = (pred_forms == valid_form["form"].to_numpy()).mean()

    m_score = np.mean(recalls)
    print(f"[{name}] C={C:.2f} -> Form Acc: {form_acc:.4f}, Material Macro Recall: {m_score:.4f}")
    for m, r in zip(MATS, recalls):
        print(f"    {m:8}: {r:.4f}")

    return form_acc, m_score, oof_p, tb, tm


def save_probs(oof, test, oof_p, tb, tm, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = out_dir / "boost_oof_probs.csv"
    with oof_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "hyp_idx", "fold", "detection_sha256", "p_bulk"] + [f"p_material_{x}" for x in MATS])
        for i, r in enumerate(oof.itertuples()):
            w.writerow([r.image_id, r.hyp_idx, r.fold, r.detection_sha256, f"{oof_p[i, 0]:.8f}"] + [f"{v:.8f}" for v in oof_p[i, 1:]])

    test_path = out_dir / "boost_test_probs.csv"
    with test_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "hyp_idx", "detection_sha256", "p_bulk"] + [f"p_material_{x}" for x in MATS])
        for i, r in enumerate(test.itertuples()):
            w.writerow([r.image_id, r.hyp_idx, r.detection_sha256, f"{tb[i]:.8f}"] + [f"{v:.8f}" for v in tm[i]])

    print(f"Saved probabilities to {oof_path} and {test_path}")
    return oof_path, test_path


def main():
    lab, oof, test, EF, ET, SF, ST = extract_all_features()

    print("\n--- Experiment 1: DINO alone with C=1.0 (baseline check) ---")
    run_experiment(lab, oof, test, EF, ET, name="DINO-baseline", C=1.0)

    print("\n--- Experiment 2: Tuning C on DINO ---")
    best_c = 1.0
    best_m = 0.0
    for c in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        _, m_score, _, _, _ = run_experiment(lab, oof, test, EF, ET, name=f"DINO-C{c}", C=c)
        if m_score > best_m:
            best_m = m_score
            best_c = c

    print("\n--- Experiment 3: DINO + Handcrafted Depth/Color Statistics ---")
    XF_combined = np.hstack([EF, SF])
    XT_combined = np.hstack([ET, ST])
    # Normalize combined
    XF_combined = XF_combined / np.linalg.norm(XF_combined, axis=1, keepdims=True)
    XT_combined = XT_combined / np.linalg.norm(XT_combined, axis=1, keepdims=True)

    for c in [0.2, 0.5, 1.0, 2.0, 5.0]:
        run_experiment(lab, oof, test, XF_combined, XT_combined, name=f"DINO+Stats-C{c}", C=c)

    # Save best
    _, _, best_oof_p, best_tb, best_tm = run_experiment(lab, oof, test, EF, ET, name="Best-DINO", C=best_c)
    save_probs(oof, test, best_oof_p, best_tb, best_tm, ROOT / "work" / "boost_probs")


if __name__ == "__main__":
    main()


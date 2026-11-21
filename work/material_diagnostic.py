import json
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone

from material_features import bounds, polygon_mask
from material_split import DATA, WORK, load_frames
from material_train import balanced_accuracy


def extra_features(bgr, mask, meta):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask]
    result = {"meta_holes_share": meta.get("holes_share", np.nan)}
    for values, name, bins, upper in zip(hsv.T, "hsv", (36, 16, 16), (180, 256, 256)):
        counts, _ = np.histogram(values, bins=bins, range=(0, upper))
        result.update({f"fine_{name}_{i:02d}": n / values.size for i, n in enumerate(counts)})
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for radius in (1, 3):
        center = gray[radius:-radius, radius:-radius]
        valid = mask[radius:-radius, radius:-radius].copy()
        pattern = []
        diffs = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)):
            y, x = radius * (1 + dy), radius * (1 + dx)
            neighbor = gray[y:y + center.shape[0], x:x + center.shape[1]]
            valid &= mask[y:y + center.shape[0], x:x + center.shape[1]]
            delta = neighbor - center
            pattern.append(delta >= 0)
            diffs.append(np.abs(delta))
        prefix = f"texture_r{radius}"
        result.update({f"{prefix}_{k}": np.nan for k in ("mean", "p90", "laplace", *range(10))})
        if valid.any():
            diffs = np.stack(diffs)[:, valid]
            bits = np.stack(pattern)[:, valid]
            transitions = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
            codes = np.where(transitions <= 2, bits.sum(axis=0), 9)
            result.update({f"{prefix}_{i}": n / valid.sum() for i, n in enumerate(np.bincount(codes, minlength=10))})
            result[f"{prefix}_mean"] = diffs.mean()
            result[f"{prefix}_p90"] = np.percentile(diffs, 90)
            laplace = cv2.Laplacian(gray, cv2.CV_32F)[radius:-radius, radius:-radius][valid]
            result[f"{prefix}_laplace"] = np.abs(laplace).mean()
    return result


def build_extra(data):
    cv2.setNumThreads(1)
    wanted = set(zip(data["image_id"], data["ann_id"]))
    rows = []
    for _, frame, image, anns in load_frames():
        chosen = [a for a in anns if (frame, a["id"]) in wanted]
        if not chosen:
            continue
        path = DATA / image["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        meta = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
        for ann in chosen:
            x0, y0, x1, y1 = bounds(ann["bbox"], image["width"], image["height"], 0.1)
            mask = polygon_mask(ann["segmentation"], x1 - x0, y1 - y0, x0, y0)
            rows.append({
                "image_id": frame, "ann_id": ann["id"],
                **extra_features(bgr[y0:y1, x0:x1], mask, meta),
            })
    result = pd.DataFrame(rows)
    assert len(result) == len(data)
    return result


def compare_material():
    base = pd.read_csv(WORK / "features.csv")
    extra = build_extra(base)
    extra.to_csv(WORK / "material_extra.csv", index=False)
    data = base.merge(extra, on=["image_id", "ann_id"], validate="one_to_one")
    stats = [
        "s_mean", "v_mean", "mask_area_px", "aspect_ratio", "height_m_p90", "depth_zero_share",
        "texture_r1_mean", "texture_r3_mean", "meta_holes_share",
    ]
    for source in ("real", "synth"):
        rows = data.loc[data["source"].eq(source) & data["material"].isin(["paper", "wood"])]
        print(f"\n{source}: median (p25, p75), train/val separately")
        for (split, material), group in rows.groupby(["split", "material"]):
            q = group[stats].quantile([.25, .50, .75])
            print(split, material, f"n={len(group)}, objects={group['object_id'].nunique()}")
            print("  " + "; ".join(f"{k}={q.loc[.50, k]:.3g} ({q.loc[.25, k]:.3g}, {q.loc[.75, k]:.3g})" for k in stats))
    with (WORK / "material.pkl").open("rb") as f:
        v1 = pickle.load(f)
    base_cols = list(v1.feature_names_)
    extra_cols = list(extra.columns.difference(["image_id", "ann_id"]))
    texture = [c for c in extra_cols if c.startswith("texture_")]
    fine = [c for c in extra_cols if c.startswith("fine_")]
    train, val = (data.loc[data["split"].eq(part)] for part in ("train", "val"))
    y = val["material"]
    real = val["source"].eq("real")

    def report(name, pred):
        recall = {mat: float((pred[y.eq(mat)] == mat).mean()) for mat in sorted(y.unique())}
        m = balanced_accuracy(y, pred)
        real_m = balanced_accuracy(y[real], pred[real])
        print(f"{name}: M={m:.6f}, real M={real_m:.6f}, recalls={recall}, "
              f"paper->wood={(pred[y.eq('paper')] == 'wood').sum()}/19")
        return m, real_m, recall

    baseline = report("v1", np.asarray(v1.predict(val[base_cols])).ravel())
    eligible = []
    for name, columns in (
        ("texture", texture),
        ("fine HSV", fine),
        ("texture + fine HSV", texture + fine),
        ("texture + fine HSV + meta holes", texture + fine + ["meta_holes_share"]),
    ):
        model = clone(v1)
        cols = base_cols + columns
        model.fit(train[cols], train["material"])
        result = report(name, np.asarray(model.predict(val[cols])).ravel())
        if (result[0] > baseline[0] and result[1] >= baseline[1] and
            all(result[2][mat] >= baseline[2][mat] for mat in ("plastic", "wood"))):
            eligible.append((result[0], name, model))
    if eligible:
        _, name, model = max(eligible, key=lambda entry: entry[0])
        print(f"Eligible for six-class diagnostic check: {name}")
        with (WORK / "material_candidate.pkl").open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("No candidate satisfies promotion criteria; v1 unchanged.")


if __name__ == "__main__":
    compare_material()

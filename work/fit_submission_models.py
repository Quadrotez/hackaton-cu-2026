"""Fit the agreed stack into new bundles; existing CV and Astra files are read-only."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from cv_grouped import fit_material, material_pred, read_data
from material_split import WORK, load_frames
from submission_tools import baseline_holdout
from take_train import estimate_mass, fit_density, grouped_folds
from take_v2 import candidates, take_input, train_candidate
from volume_v2 import fit_config, predict_volume_v2


MODEL_DIR = WORK / "submission_models"
VOLUME = ("residual", 3)


def fit_dino_head(rows, embeddings):
    model = LogisticRegression(C=.1, class_weight="balanced", max_iter=1000)
    return export_head(model.fit(embeddings[rows.index], rows["material"]))


def export_head(model):
    return {"W": model.coef_, "b": model.intercept_, "labels": model.classes_.astype(str)}


def project(student, head):
    return normalize(np.column_stack([
        head["mix"] * student, normalize(student @ head["map"] + head["offset"]),
    ]))


def fit_compact_head(rows, teacher, student):
    x = student[rows.index]
    ridge = Ridge(alpha=.5).fit(x, teacher[rows.index])
    head = {"map": ridge.coef_.T.copy(), "offset": ridge.intercept_, "mix": np.float32(.5)}
    model = LogisticRegression(C=1., class_weight="balanced", max_iter=1000)
    return {**head, **export_head(model.fit(project(x, head), rows["material"]))}


def fit_stack(rows, columns, kind, teacher, student, use_deployed_head=False):
    train = rows.assign(split="train")
    predictions = []
    for fit, hold in grouped_folds(train, 3):
        if kind == "cb":
            cb = fit_material(fit, columns, "CB-object-weighted")
            material = material_pred(cb, hold, columns)
        else:
            if kind == "compact":
                head = fit_compact_head(fit, teacher, student)
                x = project(student[hold.index], head)
            else:
                head = fit_dino_head(fit, teacher)
                x = teacher[hold.index]
            material = head["labels"][(x @ head["W"].T + head["b"]).argmax(axis=1)]
        volume = predict_volume_v2(fit_config(fit, VOLUME), hold, material)
        density, _ = fit_density(fit)
        predictions.append(pd.DataFrame({
            "volume_pred_ml": volume,
            "mass_pred_g": estimate_mass(hold["form"], material, volume, density),
        }, index=hold.index))
    oof = pd.concat(predictions, verify_integrity=True).reindex(train.index)
    if oof.isna().any().any():
        raise ValueError("Missing cross-fitted features for take")
    prototype, features = candidates(columns)["CatBoost"]
    take = train_candidate(prototype, features, train, take_input(train, columns, oof), "real+synth")
    bundle = {
        "take": {"model": take, "steps3": columns, "columns": features, "threshold": .90,
                 "density": fit_density(train)[0], "train_scope": "real+synth", "model_name": "CatBoost"},
        "volume": fit_config(train, VOLUME),
        "material_kind": kind,
    }
    if kind == "cb":
        bundle["material"] = fit_material(train, columns, "CB-object-weighted")
    elif not use_deployed_head:
        bundle["head"] = (fit_compact_head(train, teacher, student) if kind == "compact"
                          else fit_dino_head(train, teacher))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=("compact", "dino", "cb"), default="compact")
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    rows, columns = read_data()
    teacher = student = None
    if args.material in ("compact", "dino"):
        from astra_cv import embeddings as load_embeddings

        def keys(frame):
            return (frame.image_id + "_" + frame.ann_id.astype(str)).tolist()

        teacher = normalize(load_embeddings("astra_", keys(rows)))
        files = ("astra_distilled.npz",) if args.material == "compact" else (
            "astra_dino_head.npz", "astra_dino.safetensors",
        )
        for name in files:
            if not (WORK / name).is_file():
                raise FileNotFoundError(WORK / name)
        if args.material == "compact":
            ref = pd.read_csv(WORK / "cv_oof.csv")
            lookup = {key: i for i, key in enumerate(keys(ref))}
            vectors = np.load(WORK / "astra_tmp" / "astra_small_vectors.npy", allow_pickle=False)
            if len(lookup) != len(rows) or set(lookup) != set(keys(rows)) or vectors.shape != (len(ref), 384):
                raise ValueError("Compact DINO vectors do not match the training annotations")
            student = vectors[[lookup[key] for key in keys(rows)]]

    holdout = baseline_holdout()
    holdout_frames = {image_id for image_id, _, _ in holdout}
    holdout_objects = {a["attributes"]["object_id"] for _, _, anns in holdout for a in anns}
    # Drop entire frames contaminated by any held-out object, including mixed synth.
    excluded = {image_id for _, image_id, _, anns in load_frames()
                if any(a["attributes"]["object_id"] in holdout_objects for a in anns)}
    fit = rows[~rows.image_id.isin(excluded | holdout_frames)]
    assert set(fit.object_id).isdisjoint(holdout_objects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(3):
        for split, train in (("holdout", fit), ("full", rows)):
            target = args.out_dir / f"{args.material}.{split}.pkl"
            print(f"Fit {target}: {len(train)} annotations, {train.object_id.nunique()} objects", flush=True)
            bundle = fit_stack(train, columns, args.material, teacher, student,
                               use_deployed_head=split == "full")
            with target.open("xb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()

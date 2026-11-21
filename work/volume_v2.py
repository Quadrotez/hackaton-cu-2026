import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from material_split import SPLIT_PATH, WORK
from take_train import estimate_mass, fit_density, grouped_folds, load_model, material_columns
from v2_data import UNKNOWN, V2_PATH, get_data, make_material_model, save_model
from val_oracle import build_oracle
from volume_calibration import fit_volume, predict_volume, volume_score


VOLUME_NUMERIC = (
    "raw_volume_ml", "mask_area_px", "height_m_mean", "height_m_std",
    "height_m_p10", "height_m_p50", "height_m_p90",
)
VOLUME_CATS = ("form", "material", "surface", "lighting")
VOLUME_CONFIGS = (
    ("direct", 3), ("direct", 5), ("residual", 3), ("residual", 5),
    ("blend", 3), ("blend", 5),
)


def volume_inputs(rows, materials):
    x = rows[list(VOLUME_NUMERIC)].astype(float).copy()
    x["raw_volume_ml"] = x["raw_volume_ml"].where(x["raw_volume_ml"].gt(0))
    for key in VOLUME_CATS:
        x[key] = pd.Series(materials, index=rows.index) if key == "material" else rows[key]
        x[key] = x[key].fillna(UNKNOWN).astype(str)
    return x


def weights_by_object(rows):
    w = 1.0 / rows.groupby("object_id")["object_id"].transform("size")
    return w / w.mean()


def fit_volume_model(train, mode, depth):
    assert mode in ("direct", "residual")
    assert train["split"].eq("train").all() and train["volume_ml"].gt(0).all()
    x = volume_inputs(train, train["material"])
    assert list(x) == list(VOLUME_NUMERIC + VOLUME_CATS)
    y = np.log(train["volume_ml"])
    median = float(train["volume_ml"].median())
    if mode == "residual":
        y -= np.log(x["raw_volume_ml"].fillna(median))
    model = CatBoostRegressor(
        iterations=400, depth=depth, learning_rate=.04, l2_leaf_reg=6,
        loss_function="MAE", one_hot_max_size=16, cat_features=list(VOLUME_CATS),
        thread_count=3, random_seed=42, verbose=False, allow_writing_files=False,
    )
    model.fit(x, y, sample_weight=weights_by_object(train))
    return {
        "model": model, "mode": mode, "missing_anchor": median,
        "clip": [float(train["volume_ml"].min()), float(train["volume_ml"].max())],
    }


def fit_config(train, config):
    mode, depth = config
    modes = ("direct", "residual") if mode == "blend" else (mode,)
    return {"models": [fit_volume_model(train, m, depth) for m in modes], "config": config}


def volume_log_predict(fit, x):
    log_volume = fit["model"].predict(x)
    if fit["mode"] == "residual":
        log_volume += np.log(x["raw_volume_ml"].fillna(fit["missing_anchor"]).to_numpy())
    return np.clip(log_volume, *np.log(fit["clip"]))


def predict_volume_v2(bundle, rows, materials):
    x = volume_inputs(rows, materials)
    return np.exp(np.mean([volume_log_predict(fit, x) for fit in bundle["models"]], axis=0))


def predict_material(model, rows):
    return np.asarray(model.predict(rows[material_columns(model)])).ravel()


def select_volume(train, steps3):
    folds = grouped_folds(train, 4)
    predictions = []
    for fit, hold in folds:
        mat = predict_material(make_material_model(fast=True).fit(fit[steps3], fit["material"]), hold)
        x = volume_inputs(hold, mat)
        pred = {}
        for depth in (3, 5):
            direct, residual = [
                volume_log_predict(fit_volume_model(fit, mode, depth), x)
                for mode in ("direct", "residual")
            ]
            for mode, values in (("direct", direct), ("residual", residual), ("blend", (direct + residual) / 2)):
                pred[f"{mode}/{depth}"] = np.exp(values)
        predictions.append(pd.DataFrame(pred, index=hold.index))
    oof = pd.concat(predictions, verify_integrity=True).reindex(train.index)
    assert not oof.isna().any().any()
    real = train["source"].eq("real")
    stats = {}
    for mode, depth in VOLUME_CONFIGS:
        key = f"{mode}/{depth}"
        value = oof[key]
        primary = volume_score(value[real], train.loc[real, "volume_ml"])
        overall = volume_score(value, train["volume_ml"])
        stats[(mode, depth)] = (primary, overall)
        print(f"{key}: grouped train-CV V real={primary:.6f}, all={overall:.6f}")
    config = max(stats, key=lambda key: (0.7 * stats[key][0] + 0.3 * stats[key][1]))
    return config


def get_setup(version):
    if version == "primary":
        return "material.pkl", "volume_primary_v2.pkl", SPLIT_PATH, "_primary_v2"
    if version == "v2":
        return "material_v2.pkl", "volume_v2.pkl", V2_PATH, "_v2"
    raise ValueError(version)


def train_volume(version):
    train, val, steps3 = get_data(version)
    material_name, volume_name, plan, suffix = get_setup(version)
    material_model = load_model(material_name)
    config = select_volume(train, steps3)
    bundle = fit_config(train, config)
    bundle["split_version"] = version
    save_model(bundle, volume_name)
    material = predict_material(material_model, val)
    predicted = predict_volume_v2(load_model(volume_name), val, material)
    truth_material = predict_volume_v2(bundle, val, val["material"])
    previous = predict_volume(val, material, fit_volume(train))
    density, _ = fit_density(train)
    mass = estimate_mass(val["form"], material, predicted, density)
    if version == "primary":
        old = pd.read_csv(WORK / "oracle_object_predictions.csv").set_index(["image_id", "ann_id"])
        key = pd.MultiIndex.from_frame(val[["image_id", "ann_id"]])
        old_take = old["take_pred"].loc[key].to_numpy()
        np.testing.assert_array_equal(old["material_pred"].loc[key].to_numpy(), material)
        np.testing.assert_allclose(old["volume_pred_ml"].loc[key].to_numpy(), previous)
    else:
        old_take = (estimate_mass(val["form"], material, previous, density) <= 500).astype(int)
    pred_path = WORK / f"oracle_volume_objects{suffix}.csv"
    preds = val[["image_id", "ann_id"]].assign(
        material_pred=material, volume_pred_ml=predicted, mass_pred_g=mass, take_pred=old_take,
    )
    preds.to_csv(pred_path, index=False)
    print(f"{version}: selected train CV config={config!r}, train n={len(train)}, saved {volume_name}")
    for name, mask in (("real", val["source"].eq("real")), ("all", val["split"].eq("val"))):
        true = val.loc[mask, "volume_ml"]
        print(f"{name} val n={len(true)}: median V={volume_score(previous[mask], true):.6f}; "
              f"CatBoost V={volume_score(predicted[mask], true):.6f}; "
              f"CatBoost with GT material V={volume_score(truth_material[mask], true):.6f}")
    build_oracle(plan, pred_path, f"_volume{suffix}")
    return bundle


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("split", choices=["primary", "v2"])
    train_volume(cli.parse_args().split)

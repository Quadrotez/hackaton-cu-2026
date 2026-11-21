import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mass_take_v2 import MassTakeRegressor
from material_split import WORK
from take_train import estimate_mass, fit_density, grouped_folds, load_model, predict_take, take_features
from v2_data import get_data, make_material_model, save_model
from val_oracle import build_oracle
from volume_v2 import fit_config, get_setup, predict_material, predict_volume_v2, weights_by_object


CV_VOLUME_CONFIG = ("residual", 3)  # Fixed inside nested CV; independent of the outer holdout.
FIT_SCOPES = ("real-only", "real+synth")
THRESHOLDS = (.25, .35, .45, .5, .55, .65, .75, .85, .9)
GEOMETRY = [
    "mask_area_px", "mask_bbox_share", "aspect_ratio", "height_m_mean", "height_m_std",
    "height_m_p10", "height_m_p50", "height_m_p90", "depth_zero_share",
    "log_volume_pred_ml", "log_mass_pred_g",
]


def inner_folds(rows):
    # A nested train fold can contain just one glass object. Holding it out is
    # preferable to leakage; that fold's material model may not predict glass.
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for idx_fit, idx_hold in cv.split(rows, rows["material"], rows["object_id"]):
        fit, hold = rows.iloc[idx_fit], rows.iloc[idx_hold]
        fit = fit.loc[~fit["image_id"].isin(hold["image_id"])]
        assert fit["material"].nunique() >= 2
        assert set(fit["object_id"]).isdisjoint(hold["object_id"])
        assert set(fit["image_id"]).isdisjoint(hold["image_id"])
        yield fit, hold


def upstream_predictions(rows, material_model, volume, density):
    material = predict_material(material_model, rows)
    v = predict_volume_v2(volume, rows, material)
    m = estimate_mass(rows["form"], material, v, density)
    return pd.DataFrame({
        "material_pred": material, "volume_pred_ml": v, "mass_pred_g": m,
    }, index=rows.index)


def fit_upstream(fit, hold, steps3, volume_config):
    material = make_material_model(fast=True).fit(fit[steps3], fit["material"])
    volume = fit_config(fit, volume_config)
    density, _ = fit_density(fit)
    return upstream_predictions(hold, material, volume, density)


def crossfit(rows, steps3, folds, volume_config):
    if volume_config is None:
        raise ValueError("A fixed volume config is required for cross-fitting")
    pred = pd.concat(
        [fit_upstream(fit, hold, steps3, volume_config) for fit, hold in folds], verify_integrity=True
    ).reindex(rows.index)
    assert not pred.isna().any().any()
    return pred


def take_input(rows, steps3, upstream):
    x = take_features(rows, steps3, upstream)
    raw = rows["raw_volume_ml"].astype(float)
    raw = raw.where(raw.gt(0))
    density = upstream["mass_pred_g"] / upstream["volume_pred_ml"]
    x["log_raw_volume_ml"] = np.log1p(raw)
    x["log_raw_mass_g"] = np.log1p(raw * density)
    return x


def candidates(steps3, mass_trial=False):
    if mass_trial:
        return {
            "MassRidge": (
                MassTakeRegressor(),
                [*GEOMETRY, "log_raw_volume_ml", "log_raw_mass_g"],
            )
        }
    full = [*steps3, "log_volume_pred_ml", "log_mass_pred_g"]

    def logistic(c):
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42),
        )

    return {
        "CatBoost": (CatBoostClassifier(
            iterations=350, depth=4, learning_rate=.05, auto_class_weights="Balanced",
            loss_function="Logloss", thread_count=3, random_seed=42,
            verbose=False, allow_writing_files=False,
        ), full),
        "LogReg64": (logistic(.1), full),
        "LogRegGeometry": (logistic(.2), GEOMETRY),
        "LogRegMass": (logistic(1.), ["log_mass_pred_g"]),
    }


def train_candidate(prototype, columns, rows, x, scope):
    fit = rows.loc[rows["source"].eq("real")] if scope == "real-only" else rows
    assert fit["take"].nunique() == 2
    w = weights_by_object(fit)
    if isinstance(prototype, MassTakeRegressor):
        return clone(prototype).fit(x.loc[fit.index, columns], fit["mass_g"], sample_weight=w)
    kwargs = {"sample_weight" if isinstance(prototype, CatBoostClassifier) else "logisticregression__sample_weight": w}
    model = clone(prototype).fit(x.loc[fit.index, columns], fit["take"], **kwargs)
    return model


def nested_oof(train, outer_folds, recipes, steps3):
    predictions = []
    for index, (fit, hold) in enumerate(outer_folds, 1):
        up_fit = crossfit(fit, steps3, inner_folds(fit), CV_VOLUME_CONFIG)
        up_hold = fit_upstream(fit, hold, steps3, CV_VOLUME_CONFIG)
        x_fit = take_input(fit, steps3, up_fit)
        x_hold = take_input(hold, steps3, up_hold)
        probs = {}
        for scope in FIT_SCOPES:
            for name, (prototype, columns) in recipes.items():
                model = train_candidate(prototype, columns, fit, x_fit, scope)
                probs[f"{scope}|{name}"] = model.predict_proba(x_hold[columns])[:, 1]
        predictions.append(pd.DataFrame(probs, index=hold.index))
        print(f"Nested take CV fold {index}: fit={len(fit)}, hold={len(hold)}, "
              f"hold heavy IDs={hold.loc[hold['take'].eq(0), 'object_id'].nunique()}")
    out = pd.concat(predictions, verify_integrity=True).reindex(train.index)
    assert not out.isna().any().any()
    return out


def select_candidates(train, oof, suffix):
    real = train["source"].eq("real")
    ref = train.loc[real]
    weights = weights_by_object(ref)
    scores = []
    for key in oof:
        scope, name = key.split("|")
        for threshold in THRESHOLDS:
            p = oof.loc[real, key].ge(threshold).astype(int)
            scores.append({
                "scope": scope, "name": name, "threshold": threshold,
                "real_cv_S_by_object": balanced_accuracy_score(ref["take"], p, sample_weight=weights),
                "real_cv_S_by_annotation": balanced_accuracy_score(ref["take"], p),
                "cut_distance": abs(threshold - .5),
            })
    scores = pd.DataFrame(scores).sort_values(
        ["real_cv_S_by_object", "real_cv_S_by_annotation", "cut_distance"], ascending=[False, False, True]
    )
    scores.drop(columns="cut_distance").to_csv(WORK / f"take_cv{suffix}.csv", index=False)
    best = scores.iloc[0]
    scoped = [scores.loc[scores["scope"].eq(scope)].iloc[0] for scope in FIT_SCOPES]
    for row in scoped:
        print(f"Train-only best {row.scope}: {row['name']}, cut={row.threshold:.2f}, "
              f"real OOF S by object={row.real_cv_S_by_object:.6f}, by annotation={row.real_cv_S_by_annotation:.6f}")
    print(f"Selected: {best.scope} {best['name']}, cut={best.threshold:.2f}")
    return best, scoped


def summarize(val, p, prefix):
    for scope, mask in (("real", val["source"].eq("real")), ("all", val["split"].eq("val"))):
        y = val.loc[mask, "take"]
        conf = confusion_matrix(y, p[mask], labels=[0, 1])
        print(f"{prefix}, {scope} n={len(y)}: S={balanced_accuracy_score(y, p[mask]):.6f}, "
              f"confusion={conf.tolist()}, heavy IDs={val.loc[mask & val['take'].eq(0), 'object_id'].nunique()}")


def train_take_v2(version, mass_trial=False):
    train, val, steps3 = get_data(version)
    material_file, volume_file, plan, suffix = get_setup(version)
    if mass_trial:
        suffix += "_mass_trial"
    volume = load_model(volume_file)
    anchor = load_model(material_file)
    assert volume["split_version"] == version
    outer = grouped_folds(train, 4, require_both_in_hold=True)
    models = candidates(steps3, mass_trial)
    train_oof = nested_oof(train, outer, models, steps3)
    best, scoped = select_candidates(train, train_oof, suffix)
    train_up = crossfit(train, steps3, outer, volume["config"])
    density, _ = fit_density(train)
    val_up = upstream_predictions(val, anchor, volume, density)
    x_fit, x_val = take_input(train, steps3, train_up), take_input(val, steps3, val_up)
    assert {"take", "mass_g", "volume_ml", "material"}.isdisjoint(x_fit.columns)

    for row in scoped:
        prototype, columns = models[row["name"]]
        model = train_candidate(prototype, columns, train, x_fit, row.scope)
        bundle = {
            "model": model, "columns": columns, "threshold": float(row.threshold),
            "split_version": version, "density": density, "steps3": steps3,
            "material_file": material_file, "volume_file": volume_file,
            "train_scope": row.scope, "model_name": row["name"],
        }
        pr = predict_take(bundle, x_val)
        summarize(val, pr, f"heldout {row.scope} {row['name']}")
        if row.scope == best.scope:
            fname = f"take{suffix}.pkl"
            save_model(bundle, fname)
            np.testing.assert_array_equal(predict_take(load_model(fname), x_val), pr)
            val_up["take_pred"] = pr

    summarize(val, val_up["mass_pred_g"].le(500).to_numpy(), "mass rule with new volume")
    pred_path = WORK / f"oracle_objects{suffix}.csv"
    val[["image_id", "ann_id"]].join(val_up).to_csv(pred_path, index=False)
    build_oracle(plan, pred_path, suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["primary", "v2"])
    parser.add_argument("--mass-trial", action="store_true")
    args = parser.parse_args()
    train_take_v2(args.split, args.mass_trial)

import json
import pickle

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

from material_features import METADATA_COLUMNS
from material_split import DATA, WORK
from material_train import balanced_accuracy
from volume_calibration import fit_volume, group_key, predict_volume, volume_score


def fit_density(train):
    assert train["split"].eq("train").all() and train["volume_ml"].gt(0).all()
    object_stats = (
        train.assign(density=train["mass_g"] / train["volume_ml"])
        .groupby(["form", "material", "object_id"])["density"].median()
    )
    table = object_stats.groupby(level=["form", "material"]).agg(["median", "min", "max", "size"])
    table = table.rename(columns={"median": "density_g_per_ml", "size": "n_objects"}).reset_index()
    density = {
        "groups": {
            group_key(row.form, row.material): row.density_g_per_ml
            for row in table.itertuples(index=False)
        },
        "materials": object_stats.groupby(level="material").median().to_dict(),
        "global": float(object_stats.median()),
    }
    return density, table


def estimate_mass(form, materials, volume, density):
    coefficients = [
        density["groups"].get(group_key(f, mat), density["materials"].get(mat, density["global"]))
        for f, mat in zip(form, materials)
    ]
    return np.asarray(volume) * coefficients


def material_columns(model):
    return list(model.feature_names_ if hasattr(model, "feature_names_") else model.feature_names_in_)


def load_model(filename):
    with (WORK / filename).open("rb") as f:
        return pickle.load(f)


def load_data(model):
    base = pd.read_csv(WORK / "features.csv")
    steps3 = [c for c in base.columns if c not in METADATA_COLUMNS]
    wanted = set(material_columns(model))
    if not wanted.issubset(base.columns):
        additional = pd.read_csv(WORK / "material_extra.csv")
        base = base.merge(additional, on=["image_id", "ann_id"], validate="one_to_one")
    raw = pd.read_csv(WORK / "raw_volumes.csv")
    data = base.merge(raw, on=["image_id", "ann_id"], validate="one_to_one")
    assert len(data) == len(base)
    return data, steps3


def predict_upstream(rows, model, volume_fit, density_fit):
    mat = np.asarray(model.predict(rows[material_columns(model)])).ravel()
    vol = predict_volume(rows, mat, volume_fit)
    mass = estimate_mass(rows["form"], mat, vol, density_fit)
    return pd.DataFrame({"material_pred": mat, "volume_pred_ml": vol, "mass_pred_g": mass}, index=rows.index)


def grouped_folds(rows, n_splits, require_both_in_hold=False):
    for seed in range(42, 74):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for i_fit, i_hold in cv.split(rows, rows["material"], rows["object_id"]):
            fit, hold = rows.iloc[i_fit], rows.iloc[i_hold]
            # A mixed synth frame cannot appear in both sides of a fold.
            fit = fit.loc[~fit["image_id"].isin(hold["image_id"])]
            if set(fit["material"]) != set(rows["material"]) or fit["take"].nunique() != 2:
                break
            if require_both_in_hold and hold["take"].nunique() != 2:
                break
            assert set(fit["object_id"]).isdisjoint(hold["object_id"])
            assert set(fit["image_id"]).isdisjoint(hold["image_id"])
            folds.append((fit, hold))
        if len(folds) == n_splits:
            return folds
    raise ValueError("Could not find grouped folds with the required class coverage")


def fit_predict_upstream(fit, hold, model):
    fold_model = clone(model)
    fold_model.fit(fit[material_columns(model)], fit["material"])
    density, _ = fit_density(fit)
    return predict_upstream(hold, fold_model, fit_volume(fit), density)


def upstream_oof(train, model, folds):
    predictions = []
    for fit, hold in folds:
        predictions.append(fit_predict_upstream(fit, hold, model))
    oof = pd.concat(predictions, verify_integrity=True).reindex(train.index)
    assert not oof.isna().any().any()
    return oof


def take_features(rows, steps3, upstream):
    x = rows[steps3].copy()
    x["log_volume_pred_ml"] = np.log1p(upstream["volume_pred_ml"])
    x["log_mass_pred_g"] = np.log1p(upstream["mass_pred_g"])
    return x


def take_models():
    return {
        "CatBoost": CatBoostClassifier(
            iterations=350, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
            loss_function="Logloss", thread_count=6, random_seed=42,
            allow_writing_files=False, verbose=False,
        ),
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42),
        ),
    }


def select_take(train, material, steps3, outer):
    candidates = take_models()
    all_preds = []
    upstream_preds = []
    for index, (fit, hold) in enumerate(outer, 1):
        # Outer holdouts never help generate even the training predictors of a take model.
        fit_up = upstream_oof(fit, material, grouped_folds(fit, 3))
        hold_up = fit_predict_upstream(fit, hold, material)
        x_fit = take_features(fit, steps3, fit_up)
        x_hold = take_features(hold, steps3, hold_up)
        pred = {}
        for name, model in candidates.items():
            model = clone(model).fit(x_fit, fit["take"])
            pred[name] = model.predict_proba(x_hold)[:, 1]
        all_preds.append(pd.DataFrame(pred, index=hold.index))
        upstream_preds.append(hold_up)
        print(f"Nested take CV fold {index}: {len(fit)} train / {len(hold)} holdout annotations")

    probabilities = pd.concat(all_preds, verify_integrity=True).reindex(train.index)
    upstream = pd.concat(upstream_preds, verify_integrity=True).reindex(train.index)
    assert not probabilities.isna().any().any() and not upstream.isna().any().any()
    results = []
    for name in candidates:
        for cut in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pred = probabilities[name].ge(cut).astype(int)
            by_scope = []
            for mask in (train["source"].eq("real"), train["split"].eq("train")):
                group = train.loc[mask]
                # Equal object weight so that many views of one physical object don't dominate.
                weights = 1 / group.groupby("object_id")["object_id"].transform("size")
                by_scope.append(balanced_accuracy_score(group["take"], pred[mask], sample_weight=weights))
            results.append((np.mean(by_scope), -abs(cut - 0.5), name, cut))
        score, _, _, cut = max(r for r in results if r[2] == name)
        print(f"{name}: best nested train CV S={score:.6f}, threshold(take=1)={cut:.1f}")
    score, _, name, cut = max(results)
    print(f"Selected on train only: {name}, threshold(take=1)={cut:.1f}, CV S={score:.6f}")
    return candidates[name], cut, upstream


def predict_take(bundle, x):
    probability = bundle["model"].predict_proba(x[bundle["columns"]])[:, 1]
    return (probability >= bundle["threshold"]).astype(int)


def train_take():
    material = load_model("material.pkl")
    data, steps3 = load_data(material)
    train, val = (data.loc[data["split"].eq(part)] for part in ("train", "val"))
    assert set(train["object_id"]).isdisjoint(val["object_id"])
    assert set(train["image_id"]).isdisjoint(val["image_id"])
    assert data["take"].eq(data["mass_g"].le(500).astype(int)).all()
    volume_fit = json.loads((WORK / "volume_calibration.json").read_text(encoding="utf-8"))
    assert volume_fit == fit_volume(train)
    density_fit, density_table = fit_density(train)
    reference_columns = pd.read_csv(DATA / "densities.csv", nrows=0).columns
    density_table[reference_columns].to_csv(WORK / "densities_train.csv", index=False)
    (WORK / "density_calibration.json").write_text(json.dumps(density_fit, indent=2) + "\n", encoding="utf-8")

    model, cut, oof = select_take(train, material, steps3, grouped_folds(train, 4, require_both_in_hold=True))
    val_upstream = predict_upstream(val, material, volume_fit, density_fit)
    x_train = take_features(train, steps3, oof)
    x_val = take_features(val, steps3, val_upstream)
    assert not set(x_train.columns) & set(METADATA_COLUMNS)
    model.fit(x_train, train["take"])
    bundle = {"model": model, "threshold": cut, "columns": list(x_train)}
    val_pred = predict_take(bundle, x_val)
    val_upstream["take_pred"] = val_pred
    val_upstream["take_mass_rule"] = val_upstream["mass_pred_g"].le(500).astype(int)
    path = WORK / "take.pkl"
    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.testing.assert_array_equal(predict_take(load_model("take.pkl"), x_val), val_pred)
    val[["image_id", "ann_id"]].join(val_upstream).to_csv(WORK / "oracle_object_predictions.csv", index=False)
    print(f"Train: {len(train)} annotations; class counts {train['take'].value_counts().sort_index().to_dict()}")
    for scope, mask in (("all val", val["split"].eq("val")), ("real val", val["source"].eq("real"))):
        y = val.loc[mask, "take"]
        p = val_upstream.loc[mask]
        print(f"{scope}: n={len(y)}, V={volume_score(p['volume_pred_ml'], val.loc[mask, 'volume_ml']):.6f}; "
              f"S mass rule={balanced_accuracy(y, p['take_mass_rule']):.6f}; "
              f"S classifier={balanced_accuracy(y, p['take_pred']):.6f}; "
              f"confusion={confusion_matrix(y, p['take_pred'], labels=[0, 1]).tolist()}")


if __name__ == "__main__":
    train_take()

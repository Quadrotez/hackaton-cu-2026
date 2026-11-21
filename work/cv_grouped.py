"""Five-fold, object-disjoint model comparison with cross-fitted take predictors."""

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from material_features import METADATA_COLUMNS
from material_split import WORK, load_frames
from material_train import balanced_accuracy
from take_train import estimate_mass, fit_density, load_model, material_columns
from take_v2 import candidates as take_candidates
from take_v2 import take_input, train_candidate
from v2_data import ALL_FEATURES, AUX_COLUMNS, get_data
from volume_calibration import fit_volume, predict_volume, volume_score
from volume_v2 import VOLUME_CATS, volume_inputs, weights_by_object


THREADS = 4
SEED = 42
MATERIALS = ("LR-balanced", "CB-v1", "CB-object-weighted")
VOLUMES = ("raw", "median", "CB-direct-d3", "CB-direct-d5", "CB-residual-d3", "CB-residual-d5", "CB-blend-d5")
TAKE_SCOPES = ("real-only", "real+synth")
CUTS = (.35, .5, .65, .75, .8, .85, .9)
VOLUME_CONFIGS = {
    "CB-direct-d3": [("direct", 3)], "CB-direct-d5": [("direct", 5)],
    "CB-residual-d3": [("residual", 3)], "CB-residual-d5": [("residual", 5)],
    "CB-blend-d5": [("direct", 5), ("residual", 5)],
}


def read_data():
    data = pd.read_csv(ALL_FEATURES)
    keys = ["image_id", "ann_id"]
    assert not data.duplicated(keys).any()
    annotations = {
        (image_id, a["id"]): (
            source, *[a["attributes"][k] for k in ("object_id", "material", "form", "take", "volume_ml", "mass_g")]
        )
        for source, image_id, _, anns in load_frames() for a in anns
    }
    columns = ["source", "object_id", "material", "form", "take", "volume_ml", "mass_g"]
    assert {k: tuple(v) for k, v in data.set_index(keys)[columns].iterrows()} == annotations
    assert data.groupby("object_id")[["material", "take"]].nunique().eq(1).all().all()
    assert data["take"].eq(data["mass_g"].le(500).astype(int)).all()
    cols = [c for c in data if c not in (*METADATA_COLUMNS, *AUX_COLUMNS)]
    assert len(cols) == 60 and all(pd.api.types.is_numeric_dtype(data[c]) for c in cols)
    return data, cols


def object_groups(data, n_splits):
    objects = data.groupby("object_id", sort=True)[["material", "take"]].first()
    with warnings.catch_warnings():
        # With 3 glass IDs, it is impossible to put glass in all 5 outer folds.
        warnings.filterwarnings("ignore", message="The least populated class")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return [set(objects.index[hold]) for _, hold in cv.split(objects, objects["material"])]


def split_by_ids(data, val_ids):
    in_val = data["object_id"].isin(val_ids)
    val = data.loc[in_val].assign(split="val")
    train = data.loc[~in_val & ~data["image_id"].isin(val["image_id"])].assign(split="train")
    assert set(train["object_id"]).isdisjoint(val["object_id"])
    assert set(train["image_id"]).isdisjoint(val["image_id"])
    assert set(train["object_id"]) | set(val["object_id"]) == set(data["object_id"])
    return train, val


def fit_material(train, steps3, name):
    if name == "LR-balanced":
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        )
        kwargs = {}
    else:
        model = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=.05, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, thread_count=THREADS,
            allow_writing_files=False, verbose=False,
        )
        kwargs = {"sample_weight": weights_by_object(train)} if name == "CB-object-weighted" else {}
    return model.fit(train[steps3], train["material"], **kwargs)


def material_pred(model, rows, steps3):
    kw = {"thread_count": THREADS} if isinstance(model, CatBoostClassifier) else {}
    return np.asarray(model.predict(rows[steps3], **kw)).ravel()


def fit_cb_volume(train, mode, depth):
    x = volume_inputs(train, train["material"])
    y = np.log(train["volume_ml"])
    median = float(train["volume_ml"].median())
    if mode == "residual":
        y -= np.log(x["raw_volume_ml"].fillna(median))
    model = CatBoostRegressor(
        iterations=400, depth=depth, learning_rate=.04, l2_leaf_reg=6,
        loss_function="MAE", one_hot_max_size=16, cat_features=list(VOLUME_CATS),
        thread_count=THREADS, random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    model.fit(x, y, sample_weight=weights_by_object(train))
    return model, mode, median, np.log([train["volume_ml"].min(), train["volume_ml"].max()])


def cv_volume_log_pred(fit, x):
    model, mode, median, clip = fit
    pred = model.predict(x, thread_count=THREADS)
    if mode == "residual":
        pred += np.log(x["raw_volume_ml"].fillna(median).to_numpy())
    return np.clip(pred, *clip)


def fit_volumes(train, recipes):
    components = {
        config for recipe in recipes for config in VOLUME_CONFIGS.get(recipe, [])
    }
    fitted = {
        key: fit_cb_volume(train, *key)
        for key in sorted(components)
    }
    return fitted, fit_volume(train) if "median" in recipes else None


def predict_volumes(fits, rows, materials, recipes):
    fitted, median = fits
    x = volume_inputs(rows, materials)
    logs = {key: cv_volume_log_pred(model, x) for key, model in fitted.items()}
    values = {}
    for recipe in recipes:
        if recipe == "raw":
            values[recipe] = rows["raw_volume_ml"].fillna(0).to_numpy()
        elif recipe == "median":
            values[recipe] = predict_volume(rows, materials, median)
        else:
            values[recipe] = np.exp(np.mean([logs[key] for key in VOLUME_CONFIGS[recipe]], axis=0))
    return values


def metric(y, p, task):
    return volume_score(p, y) if task == "V" else balanced_accuracy(y, p)


def append_scores(records, number, task, name, val, y, pred):
    for scope, mask in (("all", np.ones(len(val), bool)), ("real", val["source"].eq("real").to_numpy())):
        records.append({
            "fold": number, "metric": task, "candidate": name, "scope": scope,
            "value": metric(y.iloc[mask], np.asarray(pred)[mask], task), "n": int(np.sum(mask)),
        })


def ranking(records, task, *, show=None):
    data = pd.DataFrame(records)
    data = data[data["metric"].eq(task)]
    assert data.groupby(["scope", "candidate"]).size().eq(5).all()
    stats = data.groupby(["candidate", "scope"])["value"].agg(["mean", "std"]).unstack()
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats = stats[["mean_all", "std_all", "mean_real", "std_real"]].sort_values(
        ["mean_all", "mean_real"], ascending=False, kind="stable"
    )
    print(f"\n{task}: unweighted mean and sample SD across 5 folds")
    print((stats if show is None else stats.head(show)).round(6).to_string())
    if show is not None and len(stats) > show:
        print(f"{show} of {len(stats)} shown; complete per-fold metrics in cv_metrics.csv")
        per_family = stats.groupby(lambda s: s.split("/cut=")[0], sort=False).head(1)
        print("Best threshold for each recipe/scope:\n" + per_family.round(6).to_string())
    print(f"{task} selected by mean_all: {stats.index[0]}; best by mean_real: {stats['mean_real'].idxmax()}")
    return stats.index[0]


def make_upstream(val, materials, volumes, density):
    assert np.isfinite(volumes).all() and (volumes > 0).all()
    return pd.DataFrame({
        "material_pred": materials, "volume_pred_ml": volumes,
        "mass_pred_g": estimate_mass(val["form"], materials, volumes, density),
    }, index=val.index)


def inner_upstream(train, steps3, mat_recipe, vol_recipe):
    predictions = []
    for hold in object_groups(train, 3):
        a, b = split_by_ids(train, hold)
        mat = material_pred(fit_material(a, steps3, mat_recipe), b, steps3)
        vol = predict_volumes(fit_volumes(a, [vol_recipe]), b, mat, [vol_recipe])[vol_recipe]
        density, _ = fit_density(a)
        predictions.append(make_upstream(b, mat, vol, density))
    oof = pd.concat(predictions, verify_integrity=True).reindex(train.index)
    assert not oof.isna().any().any()
    return oof


def take_prob(model, x):
    kw = {"thread_count": THREADS} if isinstance(model, CatBoostClassifier) else {}
    return model.predict_proba(x, **kw)[:, 1]


def probability_options(train, val, x_train, x_val, steps3):
    models = {
        **take_candidates(steps3),
        **take_candidates(steps3, mass_trial=True),
    }
    models["CatBoost"][0].set_params(thread_count=THREADS)
    for scope in TAKE_SCOPES:
        for name, (prototype, cols) in models.items():
            model = train_candidate(prototype, cols, train, x_train, scope)
            prob = take_prob(model, x_val[cols])
            for cut in CUTS:
                yield f"{name}/{scope}/cut={cut:.2f}", (prob >= cut).astype(int)
    # Comparable to v1: annotation weighting, synth included, threshold 0.8.
    cb, cols = models["CatBoost"]
    legacy = cb.fit(x_train[cols], train["take"])
    yield "CatBoost-v1-row-weighted/real+synth/cut=0.80", (take_prob(legacy, x_val[cols]) >= .8).astype(int)


def metal_summary(data, oof):
    print("\n--- Why the previous diagnostic split had metal recall 0/7 ---")
    tr, va, _ = get_data("v2", data)
    target = va.loc[va["material"].eq("metal")]
    model = load_model("material_v2.pkl")  # Read-only diagnostic of the historical frozen model.
    prob = pd.DataFrame(
        model.predict_proba(target[material_columns(model)], thread_count=THREADS),
        index=target.index, columns=model.classes_,
    )
    pred = prob.idxmax(axis=1)
    print(f"One held-out metal object: {target['object_id'].unique().tolist()}; form "
          f"{target['form'].unique().tolist()}; 4 train metal IDs and "
          f"{tr.loc[tr['material'].eq('metal')].shape[0]} train metal annotations.")
    for src in ("real", "synth"):
        mask = target["source"].eq(src)
        print(f"{src}: n={mask.sum()}, tops={pred[mask].value_counts().to_dict()}, "
              f"P(metal) median={prob.loc[mask, 'metal'].median():.3f}, "
              f"range={prob.loc[mask, 'metal'].min():.3f}..{prob.loc[mask, 'metal'].max():.3f}")
    real = target.loc[target["source"].eq("real")]
    print(f"Real lighting: {real['lighting'].value_counts().to_dict()}; "
          f"surfaces: {real['surface'].value_counts().to_dict()}.")
    columns = ["mask_area_px", "height_m_p90", "depth_zero_share", "v_mean", "v_std", "s_mean"]
    relevant = data.loc[data["source"].eq("real") & (
        data["material"].eq("metal") | data["object_id"].isin(["b01", "b10", "b15", "b21"])
    )]
    print("Per-object real medians; selected bulk-plastic / bulk-mineral comparators included:")
    print(relevant.groupby(["material", "object_id"])[columns].median().round(4).to_string())
    # In model logit units: local contributions to the gap between metal and the winning class.
    pool = Pool(real[material_columns(model)])
    shap = model.get_feature_importance(data=pool, type="ShapValues", thread_count=THREADS)
    indices = [list(model.classes_).index(label) for label in pred.loc[real.index]]
    margin = shap[:, list(model.classes_).index("metal"), :-1] - shap[np.arange(len(real)), indices, :-1]
    features = material_columns(model)
    grouped = {
        "color": [i for i, c in enumerate(features) if c[:2] in ("h_", "s_", "v_", "r_", "g_", "b_")],
        "depth": [i for i, c in enumerate(features) if c.startswith(("depth_", "height_"))],
        "shape": [i for i, c in enumerate(features) if c in ("mask_area_px", "mask_bbox_share", "aspect_ratio")],
    }
    assert sum(map(len, grouped.values())) == len(features)
    print("Mean SHAP contribution to logit(metal) - logit(winner): "
          + str({name: round(float(margin[:, indices].sum(axis=1).mean()), 4) for name, indices in grouped.items()}))
    print("Most negative individual features: "
          + str(pd.Series(margin.mean(axis=0), index=features).nsmallest(6).round(4).to_dict()))
    chosen = oof[oof["material"].eq("metal") & oof["source"].eq("real")]
    print("Five-fold real-metal OOF by physical object for the selected recipe:")
    for obj, group in chosen.groupby("object_id"):
        print(f"  {obj}: {(group['material_pred'] == 'metal').sum()}/{len(group)}, "
              f"predictions={group['material_pred'].value_counts().to_dict()}")
    print("Interpretation: b02 is a low-profile, low-hole metal outlier; color/geometry partially overlap "
          "bulk plastic/mineral. The diagnostic run missed one object under multiple lights, not seven "
          "independent metal objects. SHAP is model attribution, not causal evidence.")


def run_cv():
    data, steps3 = read_data()
    groups = object_groups(data, 5)
    assert Counter(obj for group in groups for obj in group) == Counter({obj: 1 for obj in data["object_id"].unique()})
    assert len(data) == 892 and data["object_id"].nunique() == 40
    print("5-fold object-ID CV; thread_count=4 for CatBoost fit/predict/SHAP; one fit at a time.")
    print(f"All 40 physical objects; inputs={len(data)} annotations ({data['source'].value_counts().to_dict()}).")
    print("Recipes ranked by the plain 5-fold mean on real+synth; real-only shown separately. "
          "Glass is absent from two folds, so fold M averages supported materials only; pooled OOF M uses all six.")
    print("Oracle polygons/forms; no detector. Mixed-frame annotations of train objects are omitted for that fold.")
    print("Material/volume/take selected hierarchically. Split-wise fits have no leakage; take training inputs use "
          "inner 3-fold cross-fitting. This is CV for model selection, not a separate test of the winning recipes.")
    folds, records, fold_objects, val_rows = [], [], [], []
    for i, hold in enumerate(groups, 1):
        train, val = split_by_ids(data, hold)
        assert len(hold) == 8 and train["object_id"].nunique() == 32
        assert train["material"].nunique() == 6 and val["take"].nunique() == 2
        real = val[val["source"].eq("real")]
        assert real["take"].nunique() == 2
        heavy = sorted(val.loc[val["take"].eq(0), "object_id"].unique())
        dropped = len(data) - len(train) - len(val)
        print(f"Fold {i}: val IDs={','.join(sorted(hold))}; train/val ann={len(train)}/{len(val)}, "
              f"real val={len(real)}, mixed annotations dropped={dropped}, heavy={heavy}; "
              f"real material support={real['material'].value_counts().to_dict()}")
        fold_objects.append({
            "fold": i, "train_object_ids": sorted(train["object_id"].unique()),
            "val_object_ids": sorted(hold), "heavy_val_object_ids": heavy,
        })
        val_rows.extend(val.index)
        m_preds = {}
        volume_fits = fit_volumes(train, VOLUMES)
        v_preds = {}
        for name in MATERIALS:
            material = material_pred(fit_material(train, steps3, name), val, steps3)
            m_preds[name] = material
            append_scores(records, i, "M", name, val, val["material"], material)
            v_preds[name] = predict_volumes(volume_fits, val, material, VOLUMES)
        folds.append((train, val, m_preds, v_preds))
    assert len(val_rows) == len(set(val_rows)) == len(data)

    mat_recipe = ranking(records, "M")
    for i, (_, val, _, v_preds) in enumerate(folds, 1):
        for name, p in v_preds[mat_recipe].items():
            append_scores(records, i, "V", name, val, val["volume_ml"], p)
    print(f"\nVolume comparisons below use {mat_recipe} predicted material; volume fits use true material on train only.")
    vol_recipe = ranking(records, "V")
    assert vol_recipe != "raw", "If raw wins, make its no-depth fallback explicit before stacking."

    outputs = []
    print(f"\nTake comparisons use {mat_recipe} -> {vol_recipe}, train-only density, "
          "real-only versus real+synth take fits. Volumes/masses supplied as estimates, not labels.")
    for i, (train, val, m_preds, v_preds) in enumerate(folds, 1):
        up_train = inner_upstream(train, steps3, mat_recipe, vol_recipe)
        density, _ = fit_density(train)
        up_val = make_upstream(val, m_preds[mat_recipe], v_preds[mat_recipe][vol_recipe], density)
        x_train, x_val = take_input(train, steps3, up_train), take_input(val, steps3, up_val)
        assert {"take", "material", "volume_ml", "mass_g", "source", "object_id"}.isdisjoint(x_train.columns)
        p_take = dict(probability_options(train, val, x_train, x_val, steps3))
        p_take["Mass-rule-500g"] = up_val["mass_pred_g"].le(500).to_numpy()
        for name, p in p_take.items():
            append_scores(records, i, "S", name, val, val["take"], p)
        outputs.append((val, up_val, p_take))
        print(f"Take fold {i}: compared {len(p_take)} candidate/threshold pairs")
    take_recipe = ranking(records, "S", show=12)

    oof_rows = []
    for i, (val, up, p_take) in enumerate(outputs, 1):
        oof_rows.append(val[["source", "image_id", "ann_id", "object_id", "material", "volume_ml", "take"]]
                        .join(up).assign(fold=i, take_pred=p_take[take_recipe]))
    oof = pd.concat(oof_rows, verify_integrity=True).sort_values(["source", "image_id", "ann_id"])
    assert len(oof) == len(data)
    oof.to_csv(WORK / "cv_oof.csv", index=False)
    metric_df = pd.DataFrame(records)
    metric_df.to_csv(WORK / "cv_metrics.csv", index=False)
    (WORK / "cv_folds.json").write_text(json.dumps(fold_objects, indent=2) + "\n", encoding="utf-8")
    print(f"\nCV-selected recipes: material={mat_recipe}; volume={vol_recipe}; take={take_recipe}")
    keep = metric_df["candidate"].isin([mat_recipe, vol_recipe, take_recipe])
    print(metric_df.loc[keep].pivot(index="fold", columns=["metric", "scope"], values="value").round(6).to_string())
    for scope, rows in (("all", oof), ("real", oof[oof["source"].eq("real")])):
        m = balanced_accuracy(rows["material"], rows["material_pred"])
        v = volume_score(rows["volume_pred_ml"], rows["volume_ml"])
        s = balanced_accuracy(rows["take"], rows["take_pred"])
        classes = sorted(rows["material"].unique())
        c = confusion_matrix(rows["material"], rows["material_pred"], labels=classes)
        print(f"\nPooled {scope} OOF (not fold mean): M={m:.6f}, V={v:.6f}, S={s:.6f}, n={len(rows)}")
        print("Rows=true, columns=predicted:\n" + pd.DataFrame(c, index=classes, columns=classes).to_string())
        print("Material recall: " + str(dict(zip(classes, np.round(c.diagonal() / c.sum(axis=1), 4).tolist()))))
        print("Take confusion (labels 0, 1): " + str(confusion_matrix(rows["take"], rows["take_pred"], labels=[0, 1]).tolist()))
    print("\nSelected take: real heavy objects (recall per physical object):")
    for obj, group in oof[oof["source"].eq("real") & oof["take"].eq(0)].groupby("object_id"):
        print(f"  {obj} (fold {group['fold'].iloc[0]}): {group['take_pred'].eq(0).sum()}/{len(group)}")
    metal_summary(data, oof)


if __name__ == "__main__":
    run_cv()

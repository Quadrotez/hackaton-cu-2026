import json
import pickle
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from material_features import METADATA_COLUMNS
from material_split import SPLIT_PATH, WORK


def balanced_accuracy(y_true, y_pred):
    # sklearn averages recall over supported classes; unseen true classes may still be predicted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return balanced_accuracy_score(y_true, y_pred)


def train_material():
    data = pd.read_csv(WORK / "features.csv")
    plan = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = data.loc[data["split"] == "train"]
    val = data.loc[data["split"] == "val"]
    for part, subset in (("train", train), ("val", val)):
        assert set(subset["object_id"]) == set(plan[part]["object_ids"])
        assert set(subset["image_id"]).issubset(plan[part]["image_ids"])
    assert not set(train["object_id"]) & set(val["object_id"])
    assert not set(train["image_id"]) & set(val["image_id"])
    predictors = [key for key in data.columns if key not in METADATA_COLUMNS]
    x_train, y_train = train[predictors], train["material"]
    x_val, y_val = val[predictors], val["material"]
    labels = sorted(y_train.unique())
    print(f"Train: {len(train)} annotations; val: {len(val)}; predictors: {len(predictors)}")
    print(f"Val support: {y_val.value_counts().reindex(labels, fill_value=0).to_dict()}")
    absent = sorted(set(labels) - set(y_val))
    print(f"Classes without true val instances (not in the mean recall): {', '.join(absent)}")

    models = {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "CatBoost": CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=42,
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        ),
    }
    scores = {}
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_val)).ravel()
        predictions[name] = pred
        scores[name] = balanced_accuracy(y_val, pred)
        print(f"{name}: M = {scores[name]:.6f}")

    best_name = max(scores, key=scores.get)
    pred = predictions[best_name]
    path = WORK / "material.pkl"
    with path.open("wb") as f:
        pickle.dump(models[best_name], f, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:
        loaded = pickle.load(f)
    np.testing.assert_array_equal(np.asarray(loaded.predict(x_val)).ravel(), pred)

    print(f"\nBest model: {best_name}; saved and verified: {path}")
    print("\nClassification report (val, undefined values shown as nan):")
    print(classification_report(y_val, pred, labels=labels, digits=4, zero_division=np.nan))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(y_val, pred, labels=labels), index=labels, columns=labels))
    for source in ("real", "synth"):
        mask = val["source"].eq(source).to_numpy()
        print(f"{source}-only val: M = {balanced_accuracy(y_val[mask], pred[mask]):.6f} (n={mask.sum()})")


if __name__ == "__main__":
    train_material()

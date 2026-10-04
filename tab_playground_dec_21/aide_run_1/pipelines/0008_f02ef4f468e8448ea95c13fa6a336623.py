import gc
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import xgboost as xgb


def get_xgb_device_params():
    """Detect GPU availability and determine compatible XGBoost device parameters."""
    if not torch.cuda.is_available():
        return {"tree_method": "hist", "n_jobs": -1}, False

    try:
        clf = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda")
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        return {"tree_method": "hist", "device": "cuda"}, True
    except Exception:
        pass

    try:
        clf = xgb.XGBClassifier(n_estimators=1, tree_method="gpu_hist")
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        return {"tree_method": "gpu_hist"}, True
    except Exception:
        pass

    return {"tree_method": "hist", "n_jobs": -1}, False


def engineer_features(df):
    """Engineer domain-specific spatial, hydrology, and hillshade features."""
    aspect_norm = df["Aspect"] % 360
    df["Aspect"] = aspect_norm
    df["Aspect_Sin"] = np.sin(np.radians(aspect_norm)).astype(np.float32)
    df["Aspect_Cos"] = np.cos(np.radians(aspect_norm)).astype(np.float32)

    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"].astype(np.float32) ** 2
        + df["Vertical_Distance_To_Hydrology"].astype(np.float32) ** 2
    ).astype(np.float32)

    df["Elevation_Minus_Vertical_Hydrology"] = df["Elevation"].astype(np.float32) - df[
        "Vertical_Distance_To_Hydrology"
    ].astype(np.float32)
    df["Elevation_Plus_Vertical_Hydrology"] = df["Elevation"].astype(np.float32) + df[
        "Vertical_Distance_To_Hydrology"
    ].astype(np.float32)

    df["Hydro_Road_Diff"] = (
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    ).astype(np.float32)
    df["Hydro_Road_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    ).astype(np.float32)
    df["Hydro_Fire_Diff"] = (
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    ).astype(np.float32)
    df["Hydro_Fire_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ).astype(np.float32)
    df["Road_Fire_Diff"] = (
        df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Fire_Points"]
    ).astype(np.float32)
    df["Road_Fire_Sum"] = (
        df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
    ).astype(np.float32)

    df["Hillshade_Mean"] = (
        (df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]) / 3.0
    ).astype(np.float32)
    df["Hillshade_Diff_3pm_9am"] = (df["Hillshade_3pm"] - df["Hillshade_9am"]).astype(
        np.float32
    )
    df["Hillshade_Diff_Noon_9am"] = (df["Hillshade_Noon"] - df["Hillshade_9am"]).astype(
        np.float32
    )
    df["Hillshade_Diff_3pm_Noon"] = (df["Hillshade_3pm"] - df["Hillshade_Noon"]).astype(
        np.float32
    )

    wild_cols = [
        f"Wilderness_Area{i}"
        for i in range(1, 5)
        if f"Wilderness_Area{i}" in df.columns
    ]
    df["Wilderness_Area_Sum"] = df[wild_cols].sum(axis=1).astype(np.int8)

    soil_cols = [f"Soil_Type{i}" for i in range(1, 41) if f"Soil_Type{i}" in df.columns]
    df["Soil_Type_Sum"] = df[soil_cols].sum(axis=1).astype(np.int8)

    return df


def main():
    start_time = time.time()
    device_params, use_cuda = get_xgb_device_params()
    print(f"XGBoost acceleration: {device_params}, CUDA active: {use_cuda}")

    # Load training data
    train_path = "./input/train.csv"
    print(f"Loading {train_path}...")
    train = pd.read_csv(train_path)

    # Filter out rare classes with fewer than 5 instances for valid stratified CV
    counts = train["Cover_Type"].value_counts()
    rare_classes = counts[counts < 5].index.tolist()
    if rare_classes:
        print(f"Pruning rare training classes: {rare_classes}")
        train = train[~train["Cover_Type"].isin(rare_classes)].reset_index(drop=True)

    y_raw = train["Cover_Type"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(
        f"Training instances: {len(train)}, Unique target classes: {n_classes} -> {le.classes_}"
    )

    # Feature engineering on train
    train = engineer_features(train)
    drop_cols = ["Id", "Cover_Type"]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    # Remove zero-variance (constant) columns
    const_cols = [c for c in feature_cols if train[c].min() == train[c].max()]
    if const_cols:
        print(f"Pruning constant columns: {const_cols}")
        feature_cols = [c for c in feature_cols if c not in const_cols]

    print(f"Total features used: {len(feature_cols)}")
    X = train[feature_cols].astype(np.float32).values
    del train
    gc.collect()

    # Load and process test data
    test_path = "./input/test.csv"
    print(f"Loading {test_path}...")
    test = pd.read_csv(test_path)
    test_ids = test["Id"].values
    test = engineer_features(test)
    X_test = test[feature_cols].astype(np.float32).values
    del test
    gc.collect()

    # Model training configuration: scaled capacity (depth=10) & metric alignment (merror)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if use_cuda:
        max_depth = 10
        learning_rate = 0.08
        n_estimators = 1500
        early_stop = 40
    else:
        max_depth = 10
        learning_rate = 0.10
        n_estimators = 500
        early_stop = 35

    eval_metric = "merror"

    oof_preds = np.zeros((len(X), n_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    print(
        f"\nStarting {n_splits}-Fold Stratified CV (max_depth={max_depth}, lr={learning_rate}, metric={eval_metric})..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = None
        try:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric=eval_metric,
                objective="multi:softprob",
                num_class=n_classes,
                random_state=42 + fold,
                early_stopping_rounds=early_stop,
                **device_params,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
        except TypeError:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric=eval_metric,
                objective="multi:softprob",
                num_class=n_classes,
                random_state=42 + fold,
                **device_params,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stop,
                verbose=100,
            )

        val_probs = model.predict_proba(X_val)
        oof_preds[val_idx] = val_probs
        fold_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))
        elapsed = time.time() - fold_start
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.6f} | Elapsed: {elapsed:.1f}s")

        test_preds += model.predict_proba(X_test) / n_splits
        del X_tr, y_tr, X_val, y_val, model
        gc.collect()

    # Out-of-fold evaluation
    oof_pred_labels = le.inverse_transform(np.argmax(oof_preds, axis=1))
    overall_accuracy = accuracy_score(y_raw, oof_pred_labels)
    print(f"\n==========================================")
    print(f"Overall OOF Validation Accuracy: {overall_accuracy:.6f}")
    print(f"==========================================")

    # Generate final test predictions and save submission
    final_test_preds = le.inverse_transform(np.argmax(test_preds, axis=1))
    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"

    sub = pd.DataFrame({"Id": test_ids, "Cover_Type": final_test_preds})
    sub.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print(f"Submission shape: {sub.shape}")
    print("Class distribution in submission:")
    print(sub["Cover_Type"].value_counts().sort_index())
    print(
        f"Total pipeline execution time: {(time.time() - start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()

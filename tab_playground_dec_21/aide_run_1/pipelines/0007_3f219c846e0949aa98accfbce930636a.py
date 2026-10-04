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
    """Engineer rich domain, geomorphological, solar, and USFS ELU soil features."""
    aspect_norm = df["Aspect"] % 360
    df["Aspect_Sin"] = np.sin(np.radians(aspect_norm)).astype(np.float32)
    df["Aspect_Cos"] = np.cos(np.radians(aspect_norm)).astype(np.float32)

    h_hydro = df["Horizontal_Distance_To_Hydrology"].astype(np.float32)
    v_hydro = df["Vertical_Distance_To_Hydrology"].astype(np.float32)
    h_road = df["Horizontal_Distance_To_Roadways"].astype(np.float32)
    h_fire = df["Horizontal_Distance_To_Fire_Points"].astype(np.float32)
    elevation = df["Elevation"].astype(np.float32)
    slope = df["Slope"].astype(np.float32)

    # Hydrological distance and geomorphology gradients
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_hydro**2 + v_hydro**2).astype(
        np.float32
    )
    df["Manhattan_Distance_To_Hydrology"] = (np.abs(h_hydro) + np.abs(v_hydro)).astype(
        np.float32
    )
    df["Hydrology_Slope_Angle"] = np.arctan2(v_hydro, h_hydro).astype(np.float32)

    df["Elevation_Minus_Vertical_Hydrology"] = (elevation - v_hydro).astype(np.float32)
    df["Elevation_Plus_Vertical_Hydrology"] = (elevation + v_hydro).astype(np.float32)

    # Pairwise landmark distance combinations
    df["Hydro_Road_Diff"] = (h_hydro - h_road).astype(np.float32)
    df["Hydro_Road_Sum"] = (h_hydro + h_road).astype(np.float32)
    df["Hydro_Fire_Diff"] = (h_hydro - h_fire).astype(np.float32)
    df["Hydro_Fire_Sum"] = (h_hydro + h_fire).astype(np.float32)
    df["Road_Fire_Diff"] = (h_road - h_fire).astype(np.float32)
    df["Road_Fire_Sum"] = (h_road + h_fire).astype(np.float32)

    # Proximity metrics to all cartographic amenities
    df["Total_Distance_Amenities"] = (h_hydro + h_road + h_fire).astype(np.float32)
    df["Min_Distance_Amenities"] = np.minimum(
        h_hydro, np.minimum(h_road, h_fire)
    ).astype(np.float32)
    df["Mean_Distance_Amenities"] = (df["Total_Distance_Amenities"] / 3.0).astype(
        np.float32
    )

    # Topographic radiation and aspect-slope interaction
    df["Aspect_x_Slope_Northness"] = (df["Aspect_Cos"] * slope).astype(np.float32)
    df["Aspect_x_Slope_Eastness"] = (df["Aspect_Sin"] * slope).astype(np.float32)

    # Solar Hillshade aggregations and differentials
    hs9 = df["Hillshade_9am"].astype(np.float32)
    hsn = df["Hillshade_Noon"].astype(np.float32)
    hs3 = df["Hillshade_3pm"].astype(np.float32)

    df["Hillshade_Mean"] = ((hs9 + hsn + hs3) / 3.0).astype(np.float32)
    df["Hillshade_Min"] = np.minimum(hs9, np.minimum(hsn, hs3)).astype(np.float32)
    df["Hillshade_Max"] = np.maximum(hs9, np.maximum(hsn, hs3)).astype(np.float32)
    df["Hillshade_Range"] = (df["Hillshade_Max"] - df["Hillshade_Min"]).astype(
        np.float32
    )

    df["Hillshade_Diff_3pm_9am"] = (hs3 - hs9).astype(np.float32)
    df["Hillshade_Diff_Noon_9am"] = (hsn - hs9).astype(np.float32)
    df["Hillshade_Diff_3pm_Noon"] = (hs3 - hsn).astype(np.float32)

    # Wilderness area and soil type representation
    wild_cols = [
        f"Wilderness_Area{i}"
        for i in range(1, 5)
        if f"Wilderness_Area{i}" in df.columns
    ]
    df["Wilderness_Area_Sum"] = df[wild_cols].sum(axis=1).astype(np.int8)
    wild_matrix = df[wild_cols].values
    wild_has = wild_matrix.max(axis=1) > 0
    df["Wilderness_Area_Index"] = np.where(
        wild_has, wild_matrix.argmax(axis=1) + 1, 0
    ).astype(np.int8)

    soil_cols = [f"Soil_Type{i}" for i in range(1, 41) if f"Soil_Type{i}" in df.columns]
    df["Soil_Type_Sum"] = df[soil_cols].sum(axis=1).astype(np.int8)
    soil_matrix = df[soil_cols].values
    soil_has = soil_matrix.max(axis=1) > 0
    soil_idx = np.where(soil_has, soil_matrix.argmax(axis=1) + 1, 0).astype(np.int16)
    df["Soil_Type_Index"] = soil_idx

    # USFS ELU Climatic and Geologic zone decomposition
    climatic_map = {
        0: 0,
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 4,
        13: 4,
        14: 5,
        15: 0,
        16: 6,
        17: 6,
        18: 7,
        19: 7,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 7,
        25: 7,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 8,
        35: 8,
        36: 8,
        37: 8,
        38: 8,
        39: 8,
        40: 8,
    }
    geologic_map = {
        0: 0,
        1: 7,
        2: 7,
        3: 7,
        4: 7,
        5: 7,
        6: 7,
        7: 5,
        8: 5,
        9: 2,
        10: 7,
        11: 7,
        12: 7,
        13: 7,
        14: 1,
        15: 0,
        16: 1,
        17: 1,
        18: 1,
        19: 1,
        20: 1,
        21: 2,
        22: 2,
        23: 3,
        24: 3,
        25: 4,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
        35: 7,
        36: 7,
        37: 7,
        38: 7,
        39: 7,
        40: 7,
    }

    climatic_lookup = np.zeros(41, dtype=np.int8)
    geologic_lookup = np.zeros(41, dtype=np.int8)
    for k, v in climatic_map.items():
        climatic_lookup[k] = v
    for k, v in geologic_map.items():
        geologic_lookup[k] = v

    df["Soil_Climatic_Zone"] = climatic_lookup[soil_idx]
    df["Soil_Geologic_Zone"] = geologic_lookup[soil_idx]

    return df


def main():
    start_time = time.time()
    device_params, use_cuda = get_xgb_device_params()
    print(f"XGBoost configuration: {device_params}, CUDA: {use_cuda}")

    # Load training data
    train_path = "./input/train.csv"
    print(f"Loading {train_path}...")
    train = pd.read_csv(train_path)

    # Prune rare training classes (< 5 occurrences) to preserve stratified splits
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

    # Model training configuration
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if use_cuda:
        max_depth = 10
        learning_rate = 0.1
        n_estimators = 1500
        early_stop = 40
    else:
        max_depth = 8
        learning_rate = 0.12
        n_estimators = 400
        early_stop = 30

    oof_preds = np.zeros((len(X), n_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    print(
        f"\nStarting {n_splits}-Fold Stratified Cross-Validation (max_depth={max_depth}, lr={learning_rate})..."
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
                colsample_bytree=0.7,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=n_classes,
                random_state=42 + fold,
                early_stopping_rounds=early_stop,
                **device_params,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
        except TypeError:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=0.8,
                colsample_bytree=0.7,
                eval_metric="mlogloss",
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
                verbose=200,
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

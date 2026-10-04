import gc
import os
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def engineer_features(df):
    feats = pd.DataFrame(index=df.index)

    cont_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    for col in cont_cols:
        feats[col] = df[col].astype(np.float32)

    aspect_norm = (df["Aspect"].values % 360).astype(np.float32)
    aspect_rad = aspect_norm * (np.pi / 180.0)
    feats["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
    feats["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)

    h_dist = df["Horizontal_Distance_To_Hydrology"].values.astype(np.float32)
    v_dist = df["Vertical_Distance_To_Hydrology"].values.astype(np.float32)
    elev = df["Elevation"].values.astype(np.float32)
    r_dist = df["Horizontal_Distance_To_Roadways"].values.astype(np.float32)
    f_dist = df["Horizontal_Distance_To_Fire_Points"].values.astype(np.float32)

    feats["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        h_dist**2 + v_dist**2
    ).astype(np.float32)
    feats["Hydrology_Elevation"] = (elev - v_dist).astype(np.float32)
    feats["Hydrology_Elevation_Sum"] = (elev + v_dist).astype(np.float32)

    feats["Hydro_Road_Sum"] = (h_dist + r_dist).astype(np.float32)
    feats["Hydro_Road_Diff"] = np.abs(h_dist - r_dist).astype(np.float32)
    feats["Hydro_Fire_Sum"] = (h_dist + f_dist).astype(np.float32)
    feats["Hydro_Fire_Diff"] = np.abs(h_dist - f_dist).astype(np.float32)
    feats["Road_Fire_Sum"] = (r_dist + f_dist).astype(np.float32)
    feats["Road_Fire_Diff"] = np.abs(r_dist - f_dist).astype(np.float32)

    h9 = df["Hillshade_9am"].values.astype(np.float32)
    h12 = df["Hillshade_Noon"].values.astype(np.float32)
    h15 = df["Hillshade_3pm"].values.astype(np.float32)
    feats["Hillshade_9_12_Diff"] = (h9 - h12).astype(np.float32)
    feats["Hillshade_12_15_Diff"] = (h12 - h15).astype(np.float32)
    feats["Hillshade_9_15_Diff"] = (h9 - h15).astype(np.float32)
    feats["Hillshade_Mean"] = ((h9 + h12 + h15) / 3.0).astype(np.float32)

    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

    for c in wild_cols:
        feats[c] = df[c].astype(np.float32)
    for c in soil_cols:
        feats[c] = df[c].astype(np.float32)

    feats["Wilderness_Sum"] = df[wild_cols].sum(axis=1).astype(np.float32)
    feats["Soil_Sum"] = df[soil_cols].sum(axis=1).astype(np.float32)

    return feats


def main():
    total_start = time.time()
    seed_everything(42)

    print("Loading datasets...")
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df = pd.read_csv(train_path)
    print(f"Train raw shape: {train_df.shape}")

    class_counts = train_df["Cover_Type"].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        print(f"Dropping rare classes with < 5 instances: {rare_classes}")
        train_df = train_df[~train_df["Cover_Type"].isin(rare_classes)].reset_index(
            drop=True
        )

    y_raw = train_df["Cover_Type"].values
    unique_classes = np.sort(np.unique(y_raw))
    label_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_label = {i: c for i, c in enumerate(unique_classes)}
    n_classes = len(unique_classes)
    y = np.array([label_to_idx[val] for val in y_raw], dtype=np.int32)

    print("Engineering features for train data...")
    train_feats = engineer_features(train_df)
    del train_df
    gc.collect()

    print("Loading test data and engineering features...")
    test_df = pd.read_csv(test_path)
    test_ids = test_df["Id"].values
    test_feats = engineer_features(test_df)
    del test_df
    gc.collect()

    constant_cols = [
        c for c in train_feats.columns if train_feats[c].min() == train_feats[c].max()
    ]
    if constant_cols:
        print(f"Dropping zero-variance columns: {constant_cols}")
        train_feats.drop(columns=constant_cols, inplace=True)
        test_feats.drop(columns=constant_cols, inplace=True)

    feature_cols = list(train_feats.columns)
    print(f"Total features used: {len(feature_cols)}")

    X_train = train_feats.to_numpy(dtype=np.float32)
    X_test = test_feats.to_numpy(dtype=np.float32)
    del train_feats, test_feats
    gc.collect()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(X_test), n_classes), dtype=np.float32)

    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.15,
        "num_leaves": 96,
        "max_depth": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "max_bin": 127,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        fold_start = time.time()
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        trn_data = lgb.Dataset(
            X_train[train_idx],
            label=y[train_idx],
            feature_name=feature_cols,
            free_raw_data=True,
        )
        val_data = lgb.Dataset(
            X_train[val_idx],
            label=y[val_idx],
            feature_name=feature_cols,
            reference=trn_data,
            free_raw_data=True,
        )

        fold_params = dict(lgb_params)
        fold_params["random_state"] = 42 + fold

        callbacks = []
        if hasattr(lgb, "early_stopping"):
            callbacks.append(lgb.early_stopping(stopping_rounds=35, verbose=False))
        if hasattr(lgb, "log_evaluation"):
            callbacks.append(lgb.log_evaluation(period=100))

        if callbacks:
            bst = lgb.train(
                fold_params,
                trn_data,
                num_boost_round=450,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=callbacks,
            )
        else:
            bst = lgb.train(
                fold_params,
                trn_data,
                num_boost_round=450,
                valid_sets=[val_data],
                valid_names=["val"],
                early_stopping_rounds=35,
                verbose_eval=100,
            )

        val_preds = bst.predict(X_train[val_idx])
        oof_probs[val_idx] = val_preds

        val_acc = accuracy_score(y[val_idx], np.argmax(val_preds, axis=1))
        print(
            f"Fold {fold + 1} Best Iteration: {bst.best_iteration} - Val Accuracy: {val_acc:.6f}"
        )

        test_preds_fold = bst.predict(X_test)
        test_probs += test_preds_fold / n_splits

        fold_time = (time.time() - fold_start) / 60.0
        print(f"Fold {fold + 1} elapsed time: {fold_time:.2f} minutes")

        del trn_data, val_data, bst, val_preds, test_preds_fold
        gc.collect()

    oof_preds = np.argmax(oof_probs, axis=1)
    overall_acc = accuracy_score(y, oof_preds)
    print("\n==========================================")
    print(f"5-Fold CV Overall OOF Accuracy: {overall_acc:.6f}")
    print("==========================================")

    test_pred_indices = np.argmax(test_probs, axis=1)
    test_preds_labels = np.array([idx_to_label[i] for i in test_pred_indices])

    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"
    sub = pd.DataFrame({"Id": test_ids, "Cover_Type": test_preds_labels})
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
    print(f"Submission shape: {sub.shape}")
    print(sub.head(10))

    total_time = (time.time() - total_start) / 60.0
    print(f"Total execution time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()

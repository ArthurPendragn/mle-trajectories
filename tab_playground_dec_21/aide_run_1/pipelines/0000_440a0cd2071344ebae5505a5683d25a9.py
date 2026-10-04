import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def engineer_features(df):
    """
    Construct domain-specific features for forest cover type prediction.
    """
    wilderness_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

    new_features = {
        # Euclidean distance to surface hydrology
        "Euclidean_Distance_To_Hydrology": np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        ).astype(np.float32),
        # Absolute vertical distance to hydrology
        "Abs_Vertical_Distance_To_Hydrology": np.abs(
            df["Vertical_Distance_To_Hydrology"]
        ).astype(np.float32),
        # Absolute elevation of the nearest hydrology feature
        "Hydrology_Elevation": (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        ).astype(np.float32),
        # Distance interactions between infrastructure and hydrology
        "D_Road_Hydro_Sum": (
            df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Hydrology"]
        ).astype(np.float32),
        "D_Road_Hydro_Diff": np.abs(
            df["Horizontal_Distance_To_Roadways"]
            - df["Horizontal_Distance_To_Hydrology"]
        ).astype(np.float32),
        "D_Fire_Hydro_Sum": (
            df["Horizontal_Distance_To_Fire_Points"]
            + df["Horizontal_Distance_To_Hydrology"]
        ).astype(np.float32),
        "D_Fire_Hydro_Diff": np.abs(
            df["Horizontal_Distance_To_Fire_Points"]
            - df["Horizontal_Distance_To_Hydrology"]
        ).astype(np.float32),
        "D_Road_Fire_Sum": (
            df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Fire_Points"]
        ).astype(np.float32),
        "D_Road_Fire_Diff": np.abs(
            df["Horizontal_Distance_To_Roadways"]
            - df["Horizontal_Distance_To_Fire_Points"]
        ).astype(np.float32),
        # Hillshade summary and differences across time of day
        "Hillshade_Mean": (
            (df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]) / 3.0
        ).astype(np.float32),
        "Hillshade_Diff_9_3": (df["Hillshade_9am"] - df["Hillshade_3pm"]).astype(
            np.float32
        ),
        "Hillshade_Diff_Noon_3": (df["Hillshade_Noon"] - df["Hillshade_3pm"]).astype(
            np.float32
        ),
        "Hillshade_Diff_9_Noon": (df["Hillshade_9am"] - df["Hillshade_Noon"]).astype(
            np.float32
        ),
        # Cyclical aspect transformation
        "Aspect_Sin": np.sin(np.radians(df["Aspect"])).astype(np.float32),
        "Aspect_Cos": np.cos(np.radians(df["Aspect"])).astype(np.float32),
        # Binary column counts
        "Wilderness_Area_Count": df[wilderness_cols].sum(axis=1).astype(np.int8),
        "Soil_Type_Count": df[soil_cols].sum(axis=1).astype(np.int8),
    }

    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)


def main():
    print("Loading data...")
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    test_ids = test["Id"].values
    train = train.drop(columns=["Id"])
    test = test.drop(columns=["Id"])

    # Check for rare classes in training set to ensure StratifiedKFold validity
    class_counts = train["Cover_Type"].value_counts()
    print(f"Initial class distribution:\n{class_counts.to_dict()}")
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        print(f"Filtering out rare classes with < 5 samples: {rare_classes}")
        train = train[~train["Cover_Type"].isin(rare_classes)].reset_index(drop=True)

    unique_classes = np.sort(train["Cover_Type"].unique())
    num_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y = train["Cover_Type"].map(class_to_idx).values.astype(np.int32)
    train = train.drop(columns=["Cover_Type"])

    print("Engineering features for train and test...")
    train = engineer_features(train)
    test = engineer_features(test)

    # Identify and drop constant columns
    constant_cols = [c for c in train.columns if train[c].std() == 0]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        train = train.drop(columns=constant_cols)
        test = test.drop(columns=constant_cols)

    feature_cols = list(train.columns)
    print(f"Total features: {len(feature_cols)}")

    X = train[feature_cols].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)

    del train, test
    gc.collect()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    oof_probs = np.zeros((len(X), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(X_test), num_classes), dtype=np.float32)

    print("\nStarting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            boosting_type="gbdt",
            learning_rate=0.12,
            n_estimators=350,
            num_leaves=63,
            max_depth=9,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED + fold,
            verbose=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                early_stopping(stopping_rounds=30, verbose=False),
                log_evaluation(period=100),
            ],
        )

        val_preds = model.predict_proba(X_va)
        oof_probs[val_idx] = val_preds
        fold_acc = accuracy_score(y_va, np.argmax(val_preds, axis=1))
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.6f}")

        test_probs += model.predict_proba(X_test) / n_splits

        del X_tr, y_tr, X_va, y_va, model
        gc.collect()

    # Calculate and report overall Out-Of-Fold accuracy
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_accuracy = accuracy_score(y, oof_preds)
    print("\n==========================================")
    print(f"Overall 5-Fold CV Accuracy: {overall_accuracy:.6f}")
    print("==========================================")

    # Generate test predictions
    print("\nGenerating final test predictions...")
    test_pred_indices = np.argmax(test_probs, axis=1)
    test_pred_classes = unique_classes[test_pred_indices]

    # Create submission file
    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": test_pred_classes})
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved successfully to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(
        f"Submission class distribution:\n{submission['Cover_Type'].value_counts().to_dict()}"
    )
    print("Sample test predictions:")
    print(submission.head(10))


if __name__ == "__main__":
    main()

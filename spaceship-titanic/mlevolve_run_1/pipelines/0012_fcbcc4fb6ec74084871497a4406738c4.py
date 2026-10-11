import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# ------------------------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ------------------------------------------------------------------------------
np.random.seed(42)


# ------------------------------------------------------------------------------
# 1. Data Ingestion & Domain-Specific Feature Engineering
# ------------------------------------------------------------------------------
def find_data_file(filename: str) -> str:
    """Locate data files across expected directory structures."""
    candidates = [
        os.path.join(".", "input", filename),
        os.path.join(".", "input", "input", filename),
        os.path.join(".", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Unable to find {filename}. Checked: {candidates}")


def apply_domain_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific deterministic rules to resolve missing values."""
    df = df.copy()

    # Rule 1: CryoSleep passengers spend nothing on amenities
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df.loc[df["CryoSleep"] == True, col] = 0.0

    # Rule 2: Passengers with positive amenity spending cannot be in CryoSleep
    total_known_spend = df[amenity_cols].fillna(0).sum(axis=1)
    df.loc[(df["CryoSleep"].isna()) & (total_known_spend > 0), "CryoSleep"] = False

    # Rule 3: Children under 13 have zero luxury expenditure
    for col in amenity_cols:
        df.loc[(df["Age"] < 13) & (df[col].isna()), col] = 0.0

    # Rule 4: Physical deck constraints map deterministically to origin planets
    if "CabinDeck" in df.columns:
        df.loc[
            df["HomePlanet"].isna() & df["CabinDeck"].isin(["A", "B", "C", "T"]),
            "HomePlanet",
        ] = "Europa"
        df.loc[df["HomePlanet"].isna() & (df["CabinDeck"] == "G"), "HomePlanet"] = (
            "Earth"
        )

    # Rule 5: Earth passengers are never VIP
    df.loc[df["VIP"].isna() & (df["HomePlanet"] == "Earth"), "VIP"] = False

    return df


def engineer_features(raw_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Extract physical, financial, and relational features."""
    df = raw_df.copy()

    # 1. Passenger ID decomposition
    split_id = df["PassengerId"].str.split("_", expand=True)
    df["GroupID"] = split_id[0]
    df["GroupNum"] = split_id[1].astype(int)

    # 2. Cabin decomposition
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]

    # 3. Name & Surname extraction
    name_split = df["Name"].str.split(" ", expand=True)
    df["FirstName"] = name_split[0]
    df["LastName"] = name_split[1]

    # Apply deterministic domain rules
    df = apply_domain_cleaning(df)

    # Co-traveler imputation within group (deterministic physical facts)
    group_home = (
        df.dropna(subset=["HomePlanet"]).groupby("GroupID")["HomePlanet"].first()
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(df["GroupID"].map(group_home))

    group_side = df.dropna(subset=["CabinSide"]).groupby("GroupID")["CabinSide"].first()
    df["CabinSide"] = df["CabinSide"].fillna(df["GroupID"].map(group_side))

    group_deck = df.dropna(subset=["CabinDeck"]).groupby("GroupID")["CabinDeck"].first()
    df["CabinDeck"] = df["CabinDeck"].fillna(df["GroupID"].map(group_deck))

    # Surname-level origin imputation
    last_name_home = (
        df.dropna(subset=["LastName", "HomePlanet"]).groupby("LastName")["HomePlanet"].first()
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(df["LastName"].map(last_name_home))

    # Spatial interaction token combining CabinDeck and CabinSide
    df["Deck_Side"] = (
        df["CabinDeck"].fillna("Missing").astype(str)
        + "_"
        + df["CabinSide"].fillna("Missing").astype(str)
    )

    # 4. Relational size features
    group_sizes = df["GroupID"].value_counts()
    df["GroupSize"] = df["GroupID"].map(group_sizes).astype(int)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    family_sizes = df["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_sizes).fillna(1).astype(int)
    df["HasFamily"] = (df["FamilySize"] > 1).astype(int)

    # 5. Financial & Expenditure Engineering
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = df[col].fillna(0.0)

    df["TotalSpend"] = df[amenity_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["NumAmenitiesUsed"] = (df[amenity_cols] > 0).sum(axis=1).astype(int)

    # Group-level context aggregations
    df["GroupMeanTotalSpend"] = df.groupby("GroupID")["TotalSpend"].transform("mean")
    df["CryoSleep_num"] = df["CryoSleep"].map({True: 1.0, False: 0.0})
    df["GroupCryoSleepRatio"] = (
        df.groupby("GroupID")["CryoSleep_num"].transform("mean").fillna(0.0)
    )
    df.drop(columns=["CryoSleep_num"], inplace=True)

    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["CommunalSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["CommunalRatio"] = df["CommunalSpend"] / (df["TotalSpend"] + 1.0)

    for col in amenity_cols:
        df[f"{col}_Ratio"] = df[col] / (df["TotalSpend"] + 1.0)
        df[f"{col}_log"] = np.log1p(df[col])

    df["TotalSpend_log"] = np.log1p(df["TotalSpend"])
    df["LuxurySpend_log"] = np.log1p(df["LuxurySpend"])
    df["CommunalSpend_log"] = np.log1p(df["CommunalSpend"])

    # 6. Demographic & Spatial Position Features
    df["Age_is_na"] = raw_df["Age"].isna().astype(int)
    df["IsInfant"] = (df["Age"] <= 4).astype(int)
    df["IsChild"] = ((df["Age"] > 4) & (df["Age"] <= 12)).astype(int)
    df["IsTeen"] = ((df["Age"] > 12) & (df["Age"] <= 17)).astype(int)
    df["IsSenior"] = (df["Age"] >= 55).astype(int)

    df["SpendPerAge"] = df["TotalSpend"] / (df["Age"].fillna(28.0) + 1.0)
    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)

    return df


def prepare_datasets():
    """Load, engineer features, split, encode, and return datasets and metadata."""
    os.makedirs("./working", exist_ok=True)

    train_path = find_data_file("train.csv")
    test_path = find_data_file("test.csv")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = engineer_features(train_raw, is_train=True)
    test_df = engineer_features(test_raw, is_train=False)

    train_df["Transported"] = train_df["Transported"].astype(int)

    # Imputation using train statistics ONLY
    age_median = train_df["Age"].median()
    cabin_num_median = train_df["CabinNum"].median()

    train_df["Age"] = train_df["Age"].fillna(age_median)
    test_df["Age"] = test_df["Age"].fillna(age_median)

    train_df["CabinNum"] = train_df["CabinNum"].fillna(cabin_num_median)
    test_df["CabinNum"] = test_df["CabinNum"].fillna(cabin_num_median)

    train_df["SpendPerAge"] = train_df["TotalSpend"] / (train_df["Age"] + 1.0)
    test_df["SpendPerAge"] = test_df["TotalSpend"] / (test_df["Age"] + 1.0)

    train_df["CabinRegion"] = (train_df["CabinNum"] // 300).astype(int)
    test_df["CabinRegion"] = (test_df["CabinNum"] // 300).astype(int)

    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "Deck_Side",
    ]
    for col in cat_columns:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)

    # Encode categoricals strictly fitting encoders on training data
    for col in cat_columns:
        unique_vals = sorted(train_df[col].unique().tolist())
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        train_df[col + "_Code"] = train_df[col].map(val_to_idx).fillna(-1).astype(int)
        test_df[col + "_Code"] = test_df[col].map(val_to_idx).fillna(-1).astype(int)

    # Frequency encoding fit on train
    deck_freq = train_df["CabinDeck"].value_counts(normalize=True).to_dict()
    dest_freq = train_df["Destination"].value_counts(normalize=True).to_dict()

    train_df["Deck_Freq"] = train_df["CabinDeck"].map(deck_freq).fillna(0.0)
    test_df["Deck_Freq"] = test_df["CabinDeck"].map(deck_freq).fillna(0.0)

    train_df["Dest_Freq"] = train_df["Destination"].map(dest_freq).fillna(0.0)
    test_df["Dest_Freq"] = test_df["Destination"].map(dest_freq).fillna(0.0)

    numeric_features = [
        "Age",
        "CabinNum",
        "GroupSize",
        "GroupNum",
        "FamilySize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
        "LuxurySpend",
        "CommunalSpend",
        "LuxuryRatio",
        "CommunalRatio",
        "RoomService_Ratio",
        "FoodCourt_Ratio",
        "ShoppingMall_Ratio",
        "Spa_Ratio",
        "VRDeck_Ratio",
        "RoomService_log",
        "FoodCourt_log",
        "ShoppingMall_log",
        "Spa_log",
        "VRDeck_log",
        "TotalSpend_log",
        "LuxurySpend_log",
        "CommunalSpend_log",
        "Deck_Freq",
        "Dest_Freq",
        "GroupMeanTotalSpend",
        "GroupCryoSleepRatio",
        "NumAmenitiesUsed",
        "SpendPerAge",
    ]

    binary_features = [
        "IsSolo",
        "HasFamily",
        "ZeroSpend",
        "Age_is_na",
        "IsInfant",
        "IsChild",
        "IsTeen",
        "IsSenior",
    ]

    encoded_cat_features = [col + "_Code" for col in cat_columns] + ["CabinRegion"]

    all_features = numeric_features + binary_features + encoded_cat_features

    # Stratified Group K-Fold split to prevent co-traveler leakage
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(train_df, train_df["Transported"], groups=train_df["GroupID"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    val_split = train_df[train_df["fold"] == 0].copy()
    train_split = train_df[train_df["fold"] != 0].copy()

    # Save artifacts to disk
    train_split.to_csv("./working/train_fold0.csv", index=False)
    val_split.to_csv("./working/val_fold0.csv", index=False)
    train_full = train_df.sort_index()
    train_full.to_csv("./working/train_full_cv.csv", index=False)
    test_df.to_csv("./working/test_processed.csv", index=False)

    metadata = {
        "features": all_features,
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_encoded_features": encoded_cat_features,
        "target": "Transported",
        "id_col": "PassengerId",
        "num_folds": 5,
    }

    with open("./working/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return train_split, val_split, test_df, metadata


# ------------------------------------------------------------------------------
# 2. Gradient Boosted Decision Tree Model Architectures
# ------------------------------------------------------------------------------
def create_lgbm_model(seed: int = 42) -> LGBMClassifier:
    """Create a regularized LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def fit_lgbm(model: LGBMClassifier, X_train, y_train, X_val, y_val, cat_features=None):
    """Fit LightGBM with early stopping and categorical feature support."""
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=cat_features,
        )
    except Exception:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            early_stopping_rounds=50,
            verbose=False,
            categorical_feature=cat_features,
        )
    return model


def create_xgb_model(seed: int = 42) -> XGBClassifier:
    """Create a regularized XGBoost classifier."""
    try:
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=3,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            early_stopping_rounds=50,
        )
    except TypeError:
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=3,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )


def fit_xgb(model: XGBClassifier, X_train, y_train, X_val, y_val):
    """Fit XGBoost with early stopping on validation loss."""
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
    return model


# ------------------------------------------------------------------------------
# 3. Main Training, Cross-Validation, Blending & Submission Pipeline
# ------------------------------------------------------------------------------
def main():
    train_split, val_split, test_df, metadata = prepare_datasets()
    full_train_df = pd.concat([train_split, val_split]).sort_index()

    features = metadata["features"]
    target = metadata["target"]
    num_folds = metadata.get("num_folds", 5)
    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "Deck_Side",
    ]
    cat_cols_for_lgb = [c + "_Code" for c in cat_columns if (c + "_Code") in features]

    X_test = test_df[features].copy()

    oof_preds_lgb = np.zeros(len(full_train_df), dtype=np.float64)
    oof_preds_xgb = np.zeros(len(full_train_df), dtype=np.float64)
    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
    oof_targets = full_train_df[target].values.astype(int)

    for fold in range(num_folds):
        print(f"\n--- Training Fold {fold}/{num_folds} ---")
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold
        val_indices = np.where(val_mask.to_numpy())[0]

        X_train = full_train_df.loc[train_mask, features].copy()
        y_train = full_train_df.loc[train_mask, target].values.astype(int)

        X_val = full_train_df.loc[val_mask, features].copy()
        y_val = full_train_df.loc[val_mask, target].values.astype(int)

        # 1. Train LightGBM
        lgb_model = create_lgbm_model(seed=42 + fold)
        lgb_model = fit_lgbm(
            lgb_model, X_train, y_train, X_val, y_val, cat_features=cat_cols_for_lgb
        )
        val_lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_indices] = val_lgb_prob
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / num_folds
        fold_lgb_acc = accuracy_score(y_val, (val_lgb_prob >= 0.5).astype(int))

        # 2. Train XGBoost
        xgb_model = create_xgb_model(seed=42 + fold)
        xgb_model = fit_xgb(xgb_model, X_train, y_train, X_val, y_val)
        val_xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_indices] = val_xgb_prob
        test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / num_folds
        fold_xgb_acc = accuracy_score(y_val, (val_xgb_prob >= 0.5).astype(int))

        fold_blend_acc = accuracy_score(
            y_val, ((0.5 * val_lgb_prob + 0.5 * val_xgb_prob) >= 0.5).astype(int)
        )
        print(
            f"Fold {fold} | LGBM Acc: {fold_lgb_acc:.4f} | XGB Acc: {fold_xgb_acc:.4f} | 50/50 Blend Acc: {fold_blend_acc:.4f}"
        )

    # 3. Out-of-fold Blend Weight and Classification Threshold Optimization
    best_acc = -1.0
    best_weight = 0.5
    best_threshold = 0.5

    weights = np.linspace(0.0, 1.0, 101)
    thresholds = np.linspace(0.45, 0.55, 101)

    for w in weights:
        blended_oof = w * oof_preds_lgb + (1.0 - w) * oof_preds_xgb
        for t in thresholds:
            acc = accuracy_score(oof_targets, (blended_oof >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_weight = w
                best_threshold = t

    lgb_oof_acc = accuracy_score(oof_targets, (oof_preds_lgb >= 0.5).astype(int))
    xgb_oof_acc = accuracy_score(oof_targets, (oof_preds_xgb >= 0.5).astype(int))
    print(f"\n--- Cross-Validation Summary ---")
    print(f"LGBM OOF Accuracy (thresh 0.5): {lgb_oof_acc:.4f}")
    print(f"XGBoost OOF Accuracy (thresh 0.5): {xgb_oof_acc:.4f}")
    print(f"Optimal Ensemble: {best_weight:.2f} * LightGBM + {1.0 - best_weight:.2f} * XGBoost")
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")
    print(f"Final Validation Score: {best_acc}")

    # 4. Generate Test Submission
    final_test_prob = best_weight * test_preds_lgb + (1.0 - best_weight) * test_preds_xgb
    test_predictions = (final_test_prob >= best_threshold).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)


if __name__ == "__main__":
    main()
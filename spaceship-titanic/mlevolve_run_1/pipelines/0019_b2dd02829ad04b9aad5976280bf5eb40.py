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
from catboost import CatBoostClassifier

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


def engineer_features(
    raw_df: pd.DataFrame, is_train: bool = True, joint_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Extract physical, financial, relational, and joint manifest features."""
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

    # If joint manifest is provided, compute cross-split lookups
    if joint_df is not None:
        manifest = joint_df.copy()
        if "GroupID" not in manifest.columns:
            m_split_id = manifest["PassengerId"].str.split("_", expand=True)
            manifest["GroupID"] = m_split_id[0]
            manifest["GroupNum"] = m_split_id[1].astype(int)
        if "CabinDeck" not in manifest.columns and "Cabin" in manifest.columns:
            m_cabin_split = manifest["Cabin"].str.split("/", expand=True)
            manifest["CabinDeck"] = m_cabin_split[0]
            manifest["CabinNum"] = pd.to_numeric(m_cabin_split[1], errors="coerce")
            manifest["CabinSide"] = m_cabin_split[2]
        if "LastName" not in manifest.columns and "Name" in manifest.columns:
            m_name_split = manifest["Name"].str.split(" ", expand=True)
            manifest["LastName"] = m_name_split[1]
        manifest = apply_domain_cleaning(manifest)
    else:
        manifest = df

    # Co-traveler imputation within group (deterministic physical facts across manifest)
    group_cabin = manifest.dropna(subset=["Cabin"]).groupby("GroupID")["Cabin"].first()
    df["Cabin"] = df["Cabin"].fillna(df["GroupID"].map(group_cabin))

    group_dest = manifest.dropna(subset=["Destination"]).groupby("GroupID")["Destination"].first()
    df["Destination"] = df["Destination"].fillna(df["GroupID"].map(group_dest))

    c_split_re = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = df["CabinDeck"].fillna(c_split_re[0])
    df["CabinNum"] = df["CabinNum"].fillna(pd.to_numeric(c_split_re[1], errors="coerce"))
    df["CabinSide"] = df["CabinSide"].fillna(c_split_re[2])

    group_side = manifest.dropna(subset=["CabinSide"]).groupby("GroupID")["CabinSide"].first()
    df["CabinSide"] = df["CabinSide"].fillna(df["GroupID"].map(group_side))

    group_deck = manifest.dropna(subset=["CabinDeck"]).groupby("GroupID")["CabinDeck"].first()
    df["CabinDeck"] = df["CabinDeck"].fillna(df["GroupID"].map(group_deck))

    group_num = manifest.dropna(subset=["CabinNum"]).groupby("GroupID")["CabinNum"].first()
    df["CabinNum"] = df["CabinNum"].fillna(df["GroupID"].map(group_num))

    group_home = (
        manifest.dropna(subset=["HomePlanet"]).groupby("GroupID")["HomePlanet"].first()
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(df["GroupID"].map(group_home))

    # Surname-level origin imputation across manifest
    last_name_home = (
        manifest.dropna(subset=["LastName", "HomePlanet"]).groupby("LastName")["HomePlanet"].first()
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(df["LastName"].map(last_name_home))

    # Re-apply deterministic domain cleaning post-imputation
    df = apply_domain_cleaning(df)

    # Spatial interaction token combining CabinDeck and CabinSide
    df["Deck_Side"] = (
        df["CabinDeck"].fillna("Missing").astype(str)
        + "_"
        + df["CabinSide"].fillna("Missing").astype(str)
    )
    df["Route"] = (
        df["HomePlanet"].fillna("Missing").astype(str)
        + "_"
        + df["Destination"].fillna("Missing").astype(str)
    )

    # 4. Relational size features across joint manifest
    group_sizes = manifest["GroupID"].value_counts()
    df["GroupSize"] = df["GroupID"].map(group_sizes).astype(int)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    family_sizes = manifest["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_sizes).fillna(1).astype(int)
    df["HasFamily"] = (df["FamilySize"] > 1).astype(int)

    cabin_sizes = manifest["Cabin"].value_counts()
    df["CabinSize"] = df["Cabin"].map(cabin_sizes).fillna(1).astype(int)
    df["IsSoloCabin"] = (df["CabinSize"] == 1).astype(int)

    # Deck-adaptive spatial hull deciles: CabinNum normalized by max cabin index per deck
    max_deck_cabin = manifest.groupby("CabinDeck")["CabinNum"].max().replace(0, np.nan)
    df["CabinNum_Deck_Ratio"] = df["CabinNum"] / df["CabinDeck"].map(max_deck_cabin)
    df["CabinRegion"] = (df["CabinNum_Deck_Ratio"] * 10).clip(0, 9).fillna(-1).astype(int)
    df["Deck_Side_Decile"] = (
        df["Deck_Side"].astype(str) + "_" + df["CabinRegion"].astype(str)
    )

    # 5. Financial & Expenditure Engineering
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = df[col].fillna(0.0)

    df["TotalSpend"] = df[amenity_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["NumAmenitiesUsed"] = (df[amenity_cols] > 0).sum(axis=1).astype(int)
    for col in amenity_cols:
        df[f"Has_{col}"] = (df[col] > 0).astype(int)

    # Impute CryoSleep among zero-spenders via group hibernation consensus
    if "CryoSleep" in manifest.columns:
        known_cryo = manifest.dropna(subset=["CryoSleep"])
        if not known_cryo.empty:
            group_cryo_avg = known_cryo.groupby("GroupID")["CryoSleep"].mean()
            zero_spend_na_cryo = df["CryoSleep"].isna() & (df["TotalSpend"] == 0)
            group_cryo_vote = df["GroupID"].map(group_cryo_avg)
            df.loc[zero_spend_na_cryo & (group_cryo_vote > 0.5), "CryoSleep"] = True
            df.loc[zero_spend_na_cryo & (group_cryo_vote < 0.5), "CryoSleep"] = False

    # Group-level context aggregations
    df["GroupMeanTotalSpend"] = df.groupby("GroupID")["TotalSpend"].transform("mean")
    df["Spend_Diff_GroupMean"] = df["TotalSpend"] - df["GroupMeanTotalSpend"]
    df["Spend_Ratio_GroupMean"] = df["TotalSpend"] / (df["GroupMeanTotalSpend"] + 1.0)
    df["CryoSleep_num"] = df["CryoSleep"].map({True: 1.0, False: 0.0})
    df["GroupCryoSleepRatio"] = (
        df.groupby("GroupID")["CryoSleep_num"].transform("mean").fillna(0.0)
    )
    df.drop(columns=["CryoSleep_num"], inplace=True)

    # Origin and deck normalized expenditure deviations
    if "TotalSpend" in manifest.columns:
        deck_mean_spend = manifest.groupby("CabinDeck")["TotalSpend"].mean()
        planet_mean_spend = manifest.groupby("HomePlanet")["TotalSpend"].mean()
    else:
        deck_mean_spend = df.groupby("CabinDeck")["TotalSpend"].mean()
        planet_mean_spend = df.groupby("HomePlanet")["TotalSpend"].mean()
    df["Spend_Diff_DeckMean"] = df["TotalSpend"] - df["CabinDeck"].map(deck_mean_spend).fillna(0.0)
    df["Spend_Diff_PlanetMean"] = df["TotalSpend"] - df["HomePlanet"].map(planet_mean_spend).fillna(0.0)

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

    return df


def prepare_datasets():
    """Load, engineer features, split, encode, and return datasets and metadata."""
    os.makedirs("./working", exist_ok=True)

    train_path = find_data_file("train.csv")
    test_path = find_data_file("test.csv")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # Process joint feature manifest without target labels across splits
    joint_manifest = pd.concat(
        [train_raw.drop(columns=["Transported"], errors="ignore"), test_raw],
        ignore_index=True,
    )
    joint_manifest["GroupID"] = joint_manifest["PassengerId"].str.split("_", expand=True)[0]

    # Deterministically propagate Cabin and Destination within GroupID on joint manifest
    group_cabin = joint_manifest.dropna(subset=["Cabin"]).groupby("GroupID")["Cabin"].first()
    joint_manifest["Cabin"] = joint_manifest["Cabin"].fillna(joint_manifest["GroupID"].map(group_cabin))
    group_dest = joint_manifest.dropna(subset=["Destination"]).groupby("GroupID")["Destination"].first()
    joint_manifest["Destination"] = joint_manifest["Destination"].fillna(joint_manifest["GroupID"].map(group_dest))

    c_split_m = joint_manifest["Cabin"].str.split("/", expand=True)
    joint_manifest["CabinDeck"] = c_split_m[0]
    joint_manifest["CabinNum"] = pd.to_numeric(c_split_m[1], errors="coerce")
    joint_manifest["CabinSide"] = c_split_m[2]

    group_deck = joint_manifest.dropna(subset=["CabinDeck"]).groupby("GroupID")["CabinDeck"].first()
    joint_manifest["CabinDeck"] = joint_manifest["CabinDeck"].fillna(joint_manifest["GroupID"].map(group_deck))
    group_side = joint_manifest.dropna(subset=["CabinSide"]).groupby("GroupID")["CabinSide"].first()
    joint_manifest["CabinSide"] = joint_manifest["CabinSide"].fillna(joint_manifest["GroupID"].map(group_side))
    group_num = joint_manifest.dropna(subset=["CabinNum"]).groupby("GroupID")["CabinNum"].first()
    joint_manifest["CabinNum"] = joint_manifest["CabinNum"].fillna(joint_manifest["GroupID"].map(group_num))

    amenity_cols_m = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in amenity_cols_m:
        joint_manifest[c] = joint_manifest[c].fillna(0.0)
    joint_manifest["TotalSpend"] = joint_manifest[amenity_cols_m].sum(axis=1)

    train_df = engineer_features(train_raw, is_train=True, joint_df=joint_manifest)
    test_df = engineer_features(test_raw, is_train=False, joint_df=joint_manifest)

    train_df["Transported"] = train_df["Transported"].astype(int)

    # Imputation using train statistics ONLY
    age_median = train_df["Age"].median()
    cabin_num_median = train_df["CabinNum"].median()
    ratio_median = train_df["CabinNum_Deck_Ratio"].median()

    train_df["Age"] = train_df["Age"].fillna(age_median)
    test_df["Age"] = test_df["Age"].fillna(age_median)

    train_df["CabinNum"] = train_df["CabinNum"].fillna(cabin_num_median)
    test_df["CabinNum"] = test_df["CabinNum"].fillna(cabin_num_median)

    train_df["CabinNum_Deck_Ratio"] = (
        train_df["CabinNum_Deck_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(ratio_median)
    )
    test_df["CabinNum_Deck_Ratio"] = (
        test_df["CabinNum_Deck_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(ratio_median)
    )

    train_df["SpendPerAge"] = train_df["TotalSpend"] / (train_df["Age"] + 1.0)
    test_df["SpendPerAge"] = test_df["TotalSpend"] / (test_df["Age"] + 1.0)

    # Deck-normalized cabin deciles updated with imputed ratios
    train_df["CabinRegion"] = (train_df["CabinNum_Deck_Ratio"] * 10).clip(0, 9).astype(int)
    test_df["CabinRegion"] = (test_df["CabinNum_Deck_Ratio"] * 10).clip(0, 9).astype(int)

    train_df["Deck_Side_Decile"] = train_df["Deck_Side"].astype(str) + "_" + train_df["CabinRegion"].astype(str)
    test_df["Deck_Side_Decile"] = test_df["Deck_Side"].astype(str) + "_" + test_df["CabinRegion"].astype(str)

    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "Deck_Side",
        "Deck_Side_Decile",
        "Route",
    ]
    for col in cat_columns:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)

    te_columns = ["Deck_Side_TE", "CabinDeck_TE", "Route_TE", "Deck_Side_Decile_TE"]
    for col in te_columns:
        train_df[col] = 0.0
        test_df[col] = 0.0

    # Frequency encoding fit on train (computed on string representations)
    deck_freq = train_df["CabinDeck"].value_counts(normalize=True).to_dict()
    dest_freq = train_df["Destination"].value_counts(normalize=True).to_dict()

    train_df["Deck_Freq"] = train_df["CabinDeck"].map(deck_freq).fillna(0.0).astype(float)
    test_df["Deck_Freq"] = test_df["CabinDeck"].map(deck_freq).fillna(0.0).astype(float)

    train_df["Dest_Freq"] = train_df["Destination"].map(dest_freq).fillna(0.0).astype(float)
    test_df["Dest_Freq"] = test_df["Destination"].map(dest_freq).fillna(0.0).astype(float)

    for col in cat_columns:
        categories = sorted(list(set(train_df[col].unique()).union(set(test_df[col].unique()))))
        cat_type = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(cat_type)
        test_df[col] = test_df[col].astype(cat_type)

    numeric_features = [
        "Age",
        "CabinNum",
        "CabinNum_Deck_Ratio",
        "GroupSize",
        "GroupNum",
        "FamilySize",
        "CabinSize",
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
        "Spend_Diff_GroupMean",
        "Spend_Ratio_GroupMean",
        "Spend_Diff_DeckMean",
        "Spend_Diff_PlanetMean",
        "GroupCryoSleepRatio",
        "NumAmenitiesUsed",
        "SpendPerAge",
        "CabinRegion",
        "Deck_Side_TE",
        "CabinDeck_TE",
        "Route_TE",
        "Deck_Side_Decile_TE",
    ]

    binary_features = [
        "IsSolo",
        "HasFamily",
        "IsSoloCabin",
        "ZeroSpend",
        "Has_RoomService",
        "Has_FoodCourt",
        "Has_ShoppingMall",
        "Has_Spa",
        "Has_VRDeck",
        "Age_is_na",
        "IsInfant",
        "IsChild",
        "IsTeen",
        "IsSenior",
    ]

    all_features = numeric_features + binary_features + cat_columns

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
        "categorical_features": cat_columns,
        "categorical_encoded_features": cat_columns,
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
        num_leaves=28,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=25,
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
    """Create a regularized XGBoost classifier with native categorical histogram partitioning."""
    try:
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.70,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=50,
        )
    except TypeError:
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.70,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            enable_categorical=True,
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


def create_catboost_model(seed: int = 42) -> CatBoostClassifier:
    """Create an oblivious-tree CatBoost classifier with categorical support and L2 leaf regularization."""
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="Logloss",
        random_seed=seed,
        verbose=False,
    )


def fit_catboost(
    model: CatBoostClassifier, X_train, y_train, X_val, y_val, cat_features=None
):
    """Fit CatBoost with early stopping on validation loss."""
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
            cat_features=cat_features,
        )
    except Exception:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            cat_features=cat_features,
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
        "Deck_Side_Decile",
        "Route",
    ]
    cat_cols_for_models = [c for c in cat_columns if c in features]
    te_target_cols = ["Deck_Side", "CabinDeck", "Route", "Deck_Side_Decile"]

    X_test_lgb = test_df[features].copy()
    X_test_xgb = test_df[features].copy()
    X_test_cat = test_df[features].copy()
    for col in cat_cols_for_models:
        X_test_cat[col] = X_test_cat[col].astype(str)

    test_te_avg = {f"{c}_TE": np.zeros(len(test_df), dtype=np.float64) for c in te_target_cols}

    oof_preds_lgb = np.zeros(len(full_train_df), dtype=np.float64)
    oof_preds_xgb = np.zeros(len(full_train_df), dtype=np.float64)
    oof_preds_cat = np.zeros(len(full_train_df), dtype=np.float64)
    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_cat = np.zeros(len(test_df), dtype=np.float64)
    oof_targets = full_train_df[target].values.astype(int)

    for fold in range(num_folds):
        print(f"\n--- Training Fold {fold}/{num_folds} ---")
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold
        val_indices = np.where(val_mask.to_numpy())[0]

        X_train_lgb = full_train_df.loc[train_mask, features].copy()
        y_train = full_train_df.loc[train_mask, target].values.astype(int)

        X_val_lgb = full_train_df.loc[val_mask, features].copy()
        y_val = full_train_df.loc[val_mask, target].values.astype(int)

        # Leak-free Bayesian Out-of-Fold Target Encoding (m=10.0) strictly on training indices
        prior = float(y_train.mean())
        m = 10.0
        X_test_lgb_fold = X_test_lgb.copy()
        X_test_xgb_fold = X_test_xgb.copy()
        X_test_cat_fold = X_test_cat.copy()

        for c in te_target_cols:
            te_name = f"{c}_TE"
            c_train_str = full_train_df.loc[train_mask, c].astype(str)
            stats = full_train_df.loc[train_mask].groupby(c_train_str)[target].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            smooth = (counts * means + m * prior) / (counts + m)
            te_map = smooth.to_dict()

            c_train_mapped = c_train_str.map(te_map).fillna(prior).astype(float).values
            c_val_str = full_train_df.loc[val_mask, c].astype(str)
            c_val_mapped = c_val_str.map(te_map).fillna(prior).astype(float).values
            c_test_str = test_df[c].astype(str)
            c_test_mapped = c_test_str.map(te_map).fillna(prior).astype(float).values

            X_train_lgb[te_name] = c_train_mapped
            X_val_lgb[te_name] = c_val_mapped
            X_test_lgb_fold[te_name] = c_test_mapped
            X_test_xgb_fold[te_name] = c_test_mapped
            X_test_cat_fold[te_name] = c_test_mapped
            test_te_avg[te_name] += c_test_mapped / num_folds

        # 1. Train LightGBM
        lgb_model = create_lgbm_model(seed=42 + fold)
        lgb_model = fit_lgbm(
            lgb_model, X_train_lgb, y_train, X_val_lgb, y_val, cat_features=cat_cols_for_models
        )
        val_lgb_prob = lgb_model.predict_proba(X_val_lgb)[:, 1]
        oof_preds_lgb[val_indices] = val_lgb_prob
        test_preds_lgb += lgb_model.predict_proba(X_test_lgb_fold)[:, 1] / num_folds
        fold_lgb_acc = accuracy_score(y_val, (val_lgb_prob >= 0.5).astype(int))

        # 2. Train XGBoost (native categorical histogram partitioning)
        X_train_xgb = X_train_lgb.copy()
        X_val_xgb = X_val_lgb.copy()

        xgb_model = create_xgb_model(seed=42 + fold)
        xgb_model = fit_xgb(xgb_model, X_train_xgb, y_train, X_val_xgb, y_val)
        val_xgb_prob = xgb_model.predict_proba(X_val_xgb)[:, 1]
        oof_preds_xgb[val_indices] = val_xgb_prob
        test_preds_xgb += xgb_model.predict_proba(X_test_xgb_fold)[:, 1] / num_folds
        fold_xgb_acc = accuracy_score(y_val, (val_xgb_prob >= 0.5).astype(int))

        # 3. Train CatBoost (string encoding for native categorical feature support)
        X_train_cat = X_train_lgb.copy()
        X_val_cat = X_val_lgb.copy()
        for col in cat_cols_for_models:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)

        cat_model = create_catboost_model(seed=42 + fold)
        cat_model = fit_catboost(
            cat_model, X_train_cat, y_train, X_val_cat, y_val, cat_features=cat_cols_for_models
        )
        val_cat_prob = cat_model.predict_proba(X_val_cat)[:, 1]
        oof_preds_cat[val_indices] = val_cat_prob
        test_preds_cat += cat_model.predict_proba(X_test_cat_fold)[:, 1] / num_folds
        fold_cat_acc = accuracy_score(y_val, (val_cat_prob >= 0.5).astype(int))

        fold_blend_acc = accuracy_score(
            y_val,
            (((val_lgb_prob + val_xgb_prob + val_cat_prob) / 3.0) >= 0.5).astype(int),
        )
        print(
            f"Fold {fold} | LGBM Acc: {fold_lgb_acc:.4f} | XGB Acc: {fold_xgb_acc:.4f} | Cat Acc: {fold_cat_acc:.4f} | Equal Blend Acc: {fold_blend_acc:.4f}"
        )

    # 4. Out-of-fold 3-way Blend Weight and Classification Threshold Optimization
    best_acc = -1.0
    best_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    best_threshold = 0.5

    step = 0.02
    thresholds = np.linspace(0.45, 0.55, 101)
    targets_bool = (oof_targets == 1)

    for w_lgb in np.linspace(0.0, 1.0, 51):
        max_xgb = 1.0 - w_lgb
        n_xgb = int(round(max_xgb / step)) + 1
        for w_xgb in np.linspace(0.0, max_xgb, n_xgb):
            w_cat = max(0.0, 1.0 - w_lgb - w_xgb)
            blended_oof = (
                w_lgb * oof_preds_lgb
                + w_xgb * oof_preds_xgb
                + w_cat * oof_preds_cat
            )
            for t in thresholds:
                acc = ((blended_oof >= t) == targets_bool).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w_lgb, w_xgb, w_cat)
                    best_threshold = t

    lgb_oof_acc = accuracy_score(oof_targets, (oof_preds_lgb >= 0.5).astype(int))
    xgb_oof_acc = accuracy_score(oof_targets, (oof_preds_xgb >= 0.5).astype(int))
    cat_oof_acc = accuracy_score(oof_targets, (oof_preds_cat >= 0.5).astype(int))
    print(f"\n--- Cross-Validation Summary ---")
    print(f"LGBM OOF Accuracy (thresh 0.5): {lgb_oof_acc:.4f}")
    print(f"XGBoost OOF Accuracy (thresh 0.5): {xgb_oof_acc:.4f}")
    print(f"CatBoost OOF Accuracy (thresh 0.5): {cat_oof_acc:.4f}")
    print(
        f"Optimal Ensemble: {best_weights[0]:.2f} * LightGBM + {best_weights[1]:.2f} * XGBoost + {best_weights[2]:.2f} * CatBoost"
    )
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")

    # 5. Generate Test Submission
    final_test_prob = (
        best_weights[0] * test_preds_lgb
        + best_weights[1] * test_preds_xgb
        + best_weights[2] * test_preds_cat
    )
    test_predictions = (final_test_prob >= best_threshold).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    for c in te_target_cols:
        test_df[f"{c}_TE"] = test_te_avg[f"{c}_TE"]
    test_df.to_csv("./working/test_processed.csv", index=False)

    print(f"Final Validation Score: {best_acc}")


if __name__ == "__main__":
    main()
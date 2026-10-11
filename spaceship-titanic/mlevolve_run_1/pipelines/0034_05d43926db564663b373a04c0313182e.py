import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ------------------------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ------------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)


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

    # Deck-adaptive continuous spatial hull ratios and smooth midpoint offsets
    max_deck_cabin = manifest.groupby("CabinDeck")["CabinNum"].max().replace(0, np.nan)
    df["CabinNum_Deck_Ratio"] = df["CabinNum"] / df["CabinDeck"].map(max_deck_cabin)

    # Deck-side normalized relative position and continuous hull midpoint offsets
    manifest_deck_side = (
        manifest["CabinDeck"].fillna("Missing").astype(str)
        + "_"
        + manifest["CabinSide"].fillna("Missing").astype(str)
    )
    max_deck_side_cabin = (
        manifest.groupby(manifest_deck_side)["CabinNum"].max().replace(0, np.nan).to_dict()
    )
    df["CabinNum_Deck_Side_Ratio"] = df["CabinNum"] / df["Deck_Side"].map(max_deck_side_cabin)

    # Continuous smooth hull midpoint offsets
    df["Cabin_Midpoint_Offset"] = (df["CabinNum_Deck_Ratio"] - 0.5).abs()
    df["Cabin_Side_Midpoint_Offset"] = (df["CabinNum_Deck_Side_Ratio"] - 0.5).abs()

    # Track unobserved CryoSleep before supervised CV imputation
    df["CryoSleep_Raw_Missing"] = df["CryoSleep"].isna().astype(int)

    # 5. Financial & Expenditure Engineering
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = df[col].fillna(0.0)

    df["TotalSpend"] = df[amenity_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["NumAmenitiesUsed"] = (df[amenity_cols] > 0).sum(axis=1).astype(int)
    for col in amenity_cols:
        df[f"Has_{col}"] = (df[col] > 0).astype(int)

    # Group and family behavioral aggregations across joint manifest
    if "TotalSpend" in manifest.columns:
        manifest_group_mean_spend = manifest.groupby("GroupID")["TotalSpend"].mean()
        df["GroupMeanTotalSpend"] = df["GroupID"].map(manifest_group_mean_spend).fillna(df["TotalSpend"])
    else:
        df["GroupMeanTotalSpend"] = df.groupby("GroupID")["TotalSpend"].transform("mean")

    df["Spend_Diff_GroupMean"] = df["TotalSpend"] - df["GroupMeanTotalSpend"]
    df["Spend_Ratio_GroupMean"] = df["TotalSpend"] / (df["GroupMeanTotalSpend"] + 1.0)

    if "CryoSleep" in manifest.columns:
        manifest_cryo_num = manifest["CryoSleep"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        manifest_group_cryo = manifest_cryo_num.groupby(manifest["GroupID"]).mean()
        df["GroupCryoSleepRatio"] = df["GroupID"].map(manifest_group_cryo).fillna(0.0).astype(float)
        if "LastName" in manifest.columns:
            manifest_fam_cryo = manifest_cryo_num.groupby(manifest["LastName"]).mean()
            df["FamilyCryoRatio"] = df["LastName"].map(manifest_fam_cryo).fillna(0.0).astype(float)
        else:
            df["FamilyCryoRatio"] = 0.0
    else:
        df["GroupCryoSleepRatio"] = 0.0
        df["FamilyCryoRatio"] = 0.0

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
    df["SpendSpread"] = df["LuxurySpend"] - df["CommunalSpend"]

    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["CommunalRatio"] = df["CommunalSpend"] / (df["TotalSpend"] + 1.0)

    # Preferred amenity engineering
    best_amenity = df[amenity_cols].idxmax(axis=1)
    df["PreferredAmenity"] = best_amenity
    df.loc[df["TotalSpend"] == 0, "PreferredAmenity"] = "None"

    # Continuous 3D spatial coordinates
    deck_rank_map = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "T": 0.0}
    df["DeckRank"] = df["CabinDeck"].map(deck_rank_map).fillna(0.0).astype(float)
    side_sign_map = {"P": -1.0, "S": 1.0, "Missing": 0.0}
    df["SideSign"] = df["CabinSide"].map(side_sign_map).fillna(0.0).astype(float)

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

    name_split_m = joint_manifest["Name"].str.split(" ", expand=True)
    joint_manifest["FirstName"] = name_split_m[0]
    joint_manifest["LastName"] = name_split_m[1]

    amenity_cols_m = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in amenity_cols_m:
        joint_manifest[c] = joint_manifest[c].fillna(0.0)
    joint_manifest["TotalSpend"] = joint_manifest[amenity_cols_m].sum(axis=1)
    joint_manifest = apply_domain_cleaning(joint_manifest)

    train_df = engineer_features(train_raw, is_train=True, joint_df=joint_manifest)
    test_df = engineer_features(test_raw, is_train=False, joint_df=joint_manifest)

    train_df["Transported"] = train_df["Transported"].astype(int)

    # Imputation using train statistics ONLY
    age_median = train_df["Age"].median()
    cabin_num_median = train_df["CabinNum"].median()
    ratio_median = train_df["CabinNum_Deck_Ratio"].median()
    ratio_side_median = train_df["CabinNum_Deck_Side_Ratio"].median()

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

    train_df["CabinNum_Deck_Side_Ratio"] = (
        train_df["CabinNum_Deck_Side_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(ratio_side_median)
    )
    test_df["CabinNum_Deck_Side_Ratio"] = (
        test_df["CabinNum_Deck_Side_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(ratio_side_median)
    )

    train_df["Cabin_Midpoint_Offset"] = (train_df["CabinNum_Deck_Ratio"] - 0.5).abs()
    test_df["Cabin_Midpoint_Offset"] = (test_df["CabinNum_Deck_Ratio"] - 0.5).abs()

    train_df["Cabin_Side_Midpoint_Offset"] = (train_df["CabinNum_Deck_Side_Ratio"] - 0.5).abs()
    test_df["Cabin_Side_Midpoint_Offset"] = (test_df["CabinNum_Deck_Side_Ratio"] - 0.5).abs()

    train_df["Spatial_Neighborhood_Rate"] = 0.5
    test_df["Spatial_Neighborhood_Rate"] = 0.5

    train_df["CryoSleep_Prob"] = 0.5
    test_df["CryoSleep_Prob"] = 0.5

    train_df["SpendPerAge"] = train_df["TotalSpend"] / (train_df["Age"] + 1.0)
    test_df["SpendPerAge"] = test_df["TotalSpend"] / (test_df["Age"] + 1.0)

    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "Deck_Side",
        "Route",
        "PreferredAmenity",
    ]
    for col in cat_columns:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)

    te_columns = [
        "Deck_Side_TE",
        "CabinDeck_TE",
        "Route_TE",
    ]
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
        "CabinNum_Deck_Side_Ratio",
        "Cabin_Midpoint_Offset",
        "Cabin_Side_Midpoint_Offset",
        "Spatial_Neighborhood_Rate",
        "CryoSleep_Prob",
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
        "SpendSpread",
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
        "FamilyCryoRatio",
        "DeckRank",
        "SideSign",
        "NumAmenitiesUsed",
        "SpendPerAge",
        "Deck_Side_TE",
        "CabinDeck_TE",
        "Route_TE",
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
        "CryoSleep_Raw_Missing",
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
    """Create a regularized LightGBM classifier with categorical smoothing."""
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        cat_smooth=15.0,
        cat_l2=10.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=30,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def create_meta_model() -> LogisticRegression:
    """Create an L2-regularized Logistic Regression stacking meta-classifier."""
    return LogisticRegression(C=0.5, penalty="l2", max_iter=500, random_state=42)


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
    """Create a deep oblivious-tree CatBoost classifier with categorical support and L2 leaf regularization."""
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


def create_shallow_catboost_model(seed: int = 42) -> CatBoostClassifier:
    """Create a shallow oblivious-tree CatBoost classifier capturing smooth main effects."""
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        eval_metric="Logloss",
        random_seed=seed,
        verbose=False,
    )


def fit_shallow_catboost(
    model: CatBoostClassifier, X_train, y_train, X_val, y_val, cat_features=None
):
    """Fit shallow CatBoost with early stopping on validation loss."""
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
# 2.1 Spatial Rates, Logit Conversion & Supervised CryoSleep CV Imputer
# ------------------------------------------------------------------------------
def to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit transformation."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def compute_spatial_neighborhood_rates(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_eval: pd.DataFrame,
    k_neighbors: int = 25,
    m_smooth: float = 5.0,
) -> np.ndarray:
    """
    For passengers in df_eval, compute spatial neighborhood transport rate
    from df_train along (CabinDeck, CabinSide) corridors ordered by CabinNum.
    """
    prior = float(np.mean(y_train))
    n_eval = len(df_eval)
    eval_rates = np.full(n_eval, prior, dtype=np.float64)

    deck_sides = df_train["Deck_Side"].astype(str).values
    cabin_nums_train = df_train["CabinNum"].values.astype(float)
    eval_deck_sides = df_eval["Deck_Side"].astype(str).values
    eval_cabin_nums = df_eval["CabinNum"].values.astype(float)

    unique_corridors = np.unique(eval_deck_sides)
    for corridor in unique_corridors:
        if corridor in ("Missing_Missing", "Missing"):
            continue
        train_idx = np.where(deck_sides == corridor)[0]
        if len(train_idx) == 0:
            continue
        c_train = cabin_nums_train[train_idx]
        y_c_train = y_train[train_idx]

        sort_idx = np.argsort(c_train)
        c_train_sorted = c_train[sort_idx]
        y_c_train_sorted = y_c_train[sort_idx]
        n_pts = len(c_train_sorted)

        eval_idx = np.where(eval_deck_sides == corridor)[0]
        if len(eval_idx) == 0:
            continue
        c_eval_vals = eval_cabin_nums[eval_idx]

        half_k = k_neighbors // 2
        for i, c in enumerate(c_eval_vals):
            pos = np.searchsorted(c_train_sorted, c)
            start = max(0, pos - half_k)
            end = min(n_pts, start + k_neighbors)
            if end == n_pts:
                start = max(0, end - k_neighbors)
            w_y = y_c_train_sorted[start:end]
            rate = (w_y.sum() + m_smooth * prior) / (len(w_y) + m_smooth)
            eval_rates[eval_idx[i]] = rate

    return eval_rates


def impute_cryosleep_cv(
    train_df_full: pd.DataFrame,
    test_df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    train_groups: np.ndarray,
    seed: int = 42,
):
    """
    Train a lightweight auxiliary LightGBM model within cross-validation
    using expenditures, age, deck, and planet to impute missing CryoSleep values.
    """
    aux_num_features = [
        "TotalSpend",
        "LuxurySpend",
        "CommunalSpend",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "ZeroSpend",
        "NumAmenitiesUsed",
        "TotalSpend_log",
        "LuxurySpend_log",
        "Age",
        "SpendPerAge",
        "GroupSize",
        "CabinNum",
        "CabinNum_Deck_Ratio",
        "DeckRank",
    ]
    aux_cat_features = ["CabinDeck", "HomePlanet", "Destination"]
    aux_cols = aux_num_features + aux_cat_features

    cat_dtypes = {}
    for c in aux_cat_features:
        all_cats = sorted(
            list(
                set(train_df_full[c].dropna().astype(str)).union(
                    set(test_df[c].dropna().astype(str))
                )
            )
        )
        cat_dtypes[c] = pd.CategoricalDtype(categories=all_cats)

    def prep_aux(df_sub):
        out = df_sub[aux_cols].copy()
        for c in aux_cat_features:
            out[c] = out[c].astype(str).astype(cat_dtypes[c])
        return out

    train_sub = train_df_full.loc[train_mask]
    is_known_train = (train_sub["CryoSleep_Raw_Missing"] == 0).values
    is_missing_train = ~is_known_train

    y_cryo_train = (train_sub["CryoSleep"].astype(str) == "True").astype(int).values

    X_train_known = prep_aux(train_sub.iloc[is_known_train])
    y_known = y_cryo_train[is_known_train]
    groups_known = train_groups[is_known_train]

    inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    aux_models = []
    for inner_tr, inner_va in inner_sgkf.split(X_train_known, y_known, groups=groups_known):
        m = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_train_known.iloc[inner_tr],
            y_known[inner_tr],
            eval_set=[(X_train_known.iloc[inner_va], y_known[inner_va])],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            categorical_feature=aux_cat_features,
        )
        aux_models.append(m)

    pred_train_missing = np.zeros(np.sum(is_missing_train), dtype=np.float64)
    if np.sum(is_missing_train) > 0:
        X_train_miss = prep_aux(train_sub.iloc[is_missing_train])
        pred_train_missing = np.mean(
            [m.predict_proba(X_train_miss)[:, 1] for m in aux_models], axis=0
        )

    val_sub = train_df_full.loc[val_mask]
    is_known_val = (val_sub["CryoSleep_Raw_Missing"] == 0).values
    is_missing_val = ~is_known_val
    pred_val_missing = np.zeros(np.sum(is_missing_val), dtype=np.float64)
    if np.sum(is_missing_val) > 0:
        X_val_miss = prep_aux(val_sub.iloc[is_missing_val])
        pred_val_missing = np.mean(
            [m.predict_proba(X_val_miss)[:, 1] for m in aux_models], axis=0
        )

    is_known_test = (test_df["CryoSleep_Raw_Missing"] == 0).values
    is_missing_test = ~is_known_test
    pred_test_missing = np.zeros(np.sum(is_missing_test), dtype=np.float64)
    if np.sum(is_missing_test) > 0:
        X_test_miss = prep_aux(test_df.iloc[is_missing_test])
        pred_test_missing = np.mean(
            [m.predict_proba(X_test_miss)[:, 1] for m in aux_models], axis=0
        )

    cryo_prob_train = np.zeros(len(train_sub), dtype=np.float64)
    cryo_clean_train = np.empty(len(train_sub), dtype=object)
    cryo_prob_train[is_known_train] = y_known.astype(float)
    cryo_clean_train[is_known_train] = np.where(y_known == 1, "True", "False")
    if np.sum(is_missing_train) > 0:
        cryo_prob_train[is_missing_train] = pred_train_missing
        cryo_clean_train[is_missing_train] = np.where(
            pred_train_missing >= 0.5, "True", "False"
        )

    y_cryo_val = (val_sub["CryoSleep"].astype(str) == "True").astype(int).values
    cryo_prob_val = np.zeros(len(val_sub), dtype=np.float64)
    cryo_clean_val = np.empty(len(val_sub), dtype=object)
    cryo_prob_val[is_known_val] = y_cryo_val[is_known_val].astype(float)
    cryo_clean_val[is_known_val] = np.where(
        y_cryo_val[is_known_val] == 1, "True", "False"
    )
    if np.sum(is_missing_val) > 0:
        cryo_prob_val[is_missing_val] = pred_val_missing
        cryo_clean_val[is_missing_val] = np.where(
            pred_val_missing >= 0.5, "True", "False"
        )

    y_cryo_test = (test_df["CryoSleep"].astype(str) == "True").astype(int).values
    cryo_prob_test = np.zeros(len(test_df), dtype=np.float64)
    cryo_clean_test = np.empty(len(test_df), dtype=object)
    cryo_prob_test[is_known_test] = y_cryo_test[is_known_test].astype(float)
    cryo_clean_test[is_known_test] = np.where(
        y_cryo_test[is_known_test] == 1, "True", "False"
    )
    if np.sum(is_missing_test) > 0:
        cryo_prob_test[is_missing_test] = pred_test_missing
        cryo_clean_test[is_missing_test] = np.where(
            pred_test_missing >= 0.5, "True", "False"
        )

    return (
        cryo_prob_train,
        cryo_clean_train,
        cryo_prob_val,
        cryo_clean_val,
        cryo_prob_test,
        cryo_clean_test,
    )


# ------------------------------------------------------------------------------
# 3. Main Two-Stage Stacking Cross-Validation & Submission Pipeline
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
        "Route",
        "PreferredAmenity",
    ]
    cat_cols_for_models = [c for c in cat_columns if c in features]
    te_target_cols = [
        "Deck_Side",
        "CabinDeck",
        "Route",
    ]

    cat_dtypes = {}
    for col in cat_cols_for_models:
        all_unique = sorted(
            list(
                set(full_train_df[col].dropna().astype(str)).union(
                    set(test_df[col].dropna().astype(str))
                ).union({"True", "False"} if col == "CryoSleep" else set())
            )
        )
        cat_dtypes[col] = pd.CategoricalDtype(categories=all_unique)

    oof_targets = full_train_df[target].values.astype(int)

    oof_logits_lgb = np.zeros(len(full_train_df), dtype=np.float64)
    oof_logits_deep = np.zeros(len(full_train_df), dtype=np.float64)
    oof_logits_shallow = np.zeros(len(full_train_df), dtype=np.float64)

    oof_probs_lgb = np.zeros(len(full_train_df), dtype=np.float64)
    oof_probs_deep = np.zeros(len(full_train_df), dtype=np.float64)
    oof_probs_shallow = np.zeros(len(full_train_df), dtype=np.float64)

    test_logits_lgb_folds = []
    test_logits_deep_folds = []
    test_logits_shallow_folds = []

    print("\n=== Stage 1: Cross-Validation & Base Learners ===")
    for fold in range(num_folds):
        print(f"\n--- Training Fold {fold}/{num_folds} ---")
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold
        val_indices = np.where(val_mask.to_numpy())[0]

        X_train_df = full_train_df.loc[train_mask, features].copy()
        y_train = full_train_df.loc[train_mask, target].values.astype(int)
        train_groups = full_train_df.loc[train_mask, "GroupID"].values

        X_val_df = full_train_df.loc[val_mask, features].copy()
        y_val = full_train_df.loc[val_mask, target].values.astype(int)

        X_test_df_fold = test_df[features].copy()

        # 1. Supervised CryoSleep imputation within CV
        (
            tr_cryo_prob,
            tr_cryo_clean,
            va_cryo_prob,
            va_cryo_clean,
            te_cryo_prob,
            te_cryo_clean,
        ) = impute_cryosleep_cv(
            full_train_df, test_df, train_mask, val_mask, train_groups, seed=42 + fold
        )

        X_train_df["CryoSleep_Prob"] = tr_cryo_prob
        X_train_df["CryoSleep"] = tr_cryo_clean

        X_val_df["CryoSleep_Prob"] = va_cryo_prob
        X_val_df["CryoSleep"] = va_cryo_clean

        X_test_df_fold["CryoSleep_Prob"] = te_cryo_prob
        X_test_df_fold["CryoSleep"] = te_cryo_clean

        # 2. Continuous spatial neighborhood transport rates along deck corridors
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        inner_spatial_rates = np.full(
            len(X_train_df), float(np.mean(y_train)), dtype=np.float64
        )
        for inner_tr_idx, inner_va_idx in inner_sgkf.split(
            X_train_df, y_train, groups=train_groups
        ):
            rates = compute_spatial_neighborhood_rates(
                df_train=X_train_df.iloc[inner_tr_idx],
                y_train=y_train[inner_tr_idx],
                df_eval=X_train_df.iloc[inner_va_idx],
                k_neighbors=25,
                m_smooth=5.0,
            )
            inner_spatial_rates[inner_va_idx] = rates
        X_train_df["Spatial_Neighborhood_Rate"] = inner_spatial_rates

        X_val_df["Spatial_Neighborhood_Rate"] = compute_spatial_neighborhood_rates(
            df_train=X_train_df,
            y_train=y_train,
            df_eval=X_val_df,
            k_neighbors=25,
            m_smooth=5.0,
        )
        X_test_df_fold["Spatial_Neighborhood_Rate"] = compute_spatial_neighborhood_rates(
            df_train=X_train_df,
            y_train=y_train,
            df_eval=X_test_df_fold,
            k_neighbors=25,
            m_smooth=5.0,
        )

        # 3. Bayesian smoothed target encoding
        m = 10.0
        for c in te_target_cols:
            te_name = f"{c}_TE"
            X_train_df[te_name] = 0.0

        for inner_train_idx, inner_val_idx in inner_sgkf.split(
            X_train_df, y_train, groups=train_groups
        ):
            inner_prior = float(y_train[inner_train_idx].mean())
            for c in te_target_cols:
                te_name = f"{c}_TE"
                inner_df = pd.DataFrame(
                    {
                        "cat": X_train_df[c].iloc[inner_train_idx].astype(str),
                        "target": y_train[inner_train_idx],
                    }
                )
                inner_stats = inner_df.groupby("cat")["target"].agg(["count", "mean"])
                inner_smooth = (
                    inner_stats["count"] * inner_stats["mean"] + m * inner_prior
                ) / (inner_stats["count"] + m)
                inner_te_map = inner_smooth.to_dict()

                inner_val_series = X_train_df[c].iloc[inner_val_idx].astype(str)
                inner_mapped = (
                    inner_val_series.map(inner_te_map)
                    .fillna(inner_prior)
                    .astype(float)
                    .values
                )
                X_train_df.iloc[
                    inner_val_idx, X_train_df.columns.get_loc(te_name)
                ] = inner_mapped

        outer_prior = float(y_train.mean())
        for c in te_target_cols:
            te_name = f"{c}_TE"
            outer_df = pd.DataFrame(
                {
                    "cat": X_train_df[c].astype(str),
                    "target": y_train,
                }
            )
            outer_stats = outer_df.groupby("cat")["target"].agg(["count", "mean"])
            smooth = (outer_stats["count"] * outer_stats["mean"] + m * outer_prior) / (
                outer_stats["count"] + m
            )
            te_map = smooth.to_dict()

            X_val_df[te_name] = (
                X_val_df[c].astype(str).map(te_map).fillna(outer_prior).astype(float).values
            )
            X_test_df_fold[te_name] = (
                X_test_df_fold[c]
                .astype(str)
                .map(te_map)
                .fillna(outer_prior)
                .astype(float)
                .values
            )

        for col in cat_cols_for_models:
            X_train_df[col] = X_train_df[col].astype(str).astype(cat_dtypes[col])
            X_val_df[col] = X_val_df[col].astype(str).astype(cat_dtypes[col])
            X_test_df_fold[col] = X_test_df_fold[col].astype(str).astype(cat_dtypes[col])

        # Model 1: Retuned LightGBM
        X_train_lgb = X_train_df.copy()
        X_val_lgb = X_val_df.copy()
        X_test_lgb = X_test_df_fold.copy()

        lgb_model = create_lgbm_model(seed=42 + fold)
        lgb_model = fit_lgbm(
            lgb_model,
            X_train_lgb,
            y_train,
            X_val_lgb,
            y_val,
            cat_features=cat_cols_for_models,
        )
        val_lgb_prob = lgb_model.predict_proba(X_val_lgb)[:, 1]
        test_lgb_prob = lgb_model.predict_proba(X_test_lgb)[:, 1]

        val_lgb_logit = to_logit(val_lgb_prob)
        test_lgb_logit = to_logit(test_lgb_prob)

        oof_logits_lgb[val_indices] = val_lgb_logit
        oof_probs_lgb[val_indices] = val_lgb_prob
        test_logits_lgb_folds.append(test_lgb_logit)
        fold_lgb_acc = accuracy_score(y_val, (val_lgb_prob >= 0.5).astype(int))

        # Model 2: Deep CatBoost (depth=6, l2_leaf_reg=3.0)
        X_train_deep = X_train_df.copy()
        X_val_deep = X_val_df.copy()
        X_test_deep = X_test_df_fold.copy()
        for col in cat_cols_for_models:
            X_train_deep[col] = X_train_deep[col].astype(str)
            X_val_deep[col] = X_val_deep[col].astype(str)
            X_test_deep[col] = X_test_deep[col].astype(str)

        deep_cat_model = create_catboost_model(seed=42 + fold)
        deep_cat_model = fit_catboost(
            deep_cat_model,
            X_train_deep,
            y_train,
            X_val_deep,
            y_val,
            cat_features=cat_cols_for_models,
        )
        val_deep_prob = deep_cat_model.predict_proba(X_val_deep)[:, 1]
        test_deep_prob = deep_cat_model.predict_proba(X_test_deep)[:, 1]

        val_deep_logit = to_logit(val_deep_prob)
        test_deep_logit = to_logit(test_deep_prob)

        oof_logits_deep[val_indices] = val_deep_logit
        oof_probs_deep[val_indices] = val_deep_prob
        test_logits_deep_folds.append(test_deep_logit)
        fold_deep_acc = accuracy_score(y_val, (val_deep_prob >= 0.5).astype(int))

        # Model 3: Shallow CatBoost (depth=4, l2_leaf_reg=5.0)
        X_train_shallow = X_train_df.copy()
        X_val_shallow = X_val_df.copy()
        X_test_shallow = X_test_df_fold.copy()
        for col in cat_cols_for_models:
            X_train_shallow[col] = X_train_shallow[col].astype(str)
            X_val_shallow[col] = X_val_shallow[col].astype(str)
            X_test_shallow[col] = X_test_shallow[col].astype(str)

        shallow_cat_model = create_shallow_catboost_model(seed=42 + fold)
        shallow_cat_model = fit_shallow_catboost(
            shallow_cat_model,
            X_train_shallow,
            y_train,
            X_val_shallow,
            y_val,
            cat_features=cat_cols_for_models,
        )
        val_shallow_prob = shallow_cat_model.predict_proba(X_val_shallow)[:, 1]
        test_shallow_prob = shallow_cat_model.predict_proba(X_test_shallow)[:, 1]

        val_shallow_logit = to_logit(val_shallow_prob)
        test_shallow_logit = to_logit(test_shallow_prob)

        oof_logits_shallow[val_indices] = val_shallow_logit
        oof_probs_shallow[val_indices] = val_shallow_prob
        test_logits_shallow_folds.append(test_shallow_logit)
        fold_shallow_acc = accuracy_score(y_val, (val_shallow_prob >= 0.5).astype(int))

        print(
            f"Fold {fold} | LGBM (31 leaves): {fold_lgb_acc:.4f} | "
            f"Deep CatBoost: {fold_deep_acc:.4f} | Shallow CatBoost: {fold_shallow_acc:.4f}"
        )

    # === Stage 2: Stacking Meta-Learner & Threshold Optimization ===
    print("\n=== Stage 2: Stacking Meta-Learner Training & Calibration ===")
    X_oof_logits = np.column_stack(
        [oof_logits_lgb, oof_logits_deep, oof_logits_shallow]
    )

    oof_meta_probs = np.zeros(len(full_train_df), dtype=np.float64)
    test_meta_probs = np.zeros(len(test_df), dtype=np.float64)

    for fold in range(num_folds):
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold
        val_indices = np.where(val_mask.to_numpy())[0]

        meta_model = create_meta_model()
        meta_model.fit(X_oof_logits[train_mask], oof_targets[train_mask])

        val_meta_prob = meta_model.predict_proba(X_oof_logits[val_mask])[:, 1]
        oof_meta_probs[val_indices] = val_meta_prob

        X_test_logits_fold = np.column_stack(
            [
                test_logits_lgb_folds[fold],
                test_logits_deep_folds[fold],
                test_logits_shallow_folds[fold],
            ]
        )
        test_meta_probs += meta_model.predict_proba(X_test_logits_fold)[:, 1] / num_folds

        coefs = meta_model.coef_[0]
        print(
            f"Fold {fold} Meta-Learner: LGBM weight={coefs[0]:.3f}, "
            f"DeepCat weight={coefs[1]:.3f}, ShallowCat weight={coefs[2]:.3f}, "
            f"Intercept={meta_model.intercept_[0]:.3f}"
        )

    best_acc = -1.0
    best_threshold = 0.5
    thresholds = np.arange(0.45, 0.5501, 0.001)

    for t in thresholds:
        acc = accuracy_score(oof_targets, (oof_meta_probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    lgb_oof_acc = accuracy_score(oof_targets, (oof_probs_lgb >= 0.5).astype(int))
    deep_cat_oof_acc = accuracy_score(
        oof_targets, (oof_probs_deep >= 0.5).astype(int)
    )
    shallow_cat_oof_acc = accuracy_score(
        oof_targets, (oof_probs_shallow >= 0.5).astype(int)
    )
    meta_oof_acc_05 = accuracy_score(
        oof_targets, (oof_meta_probs >= 0.5).astype(int)
    )

    print("\n--- Cross-Validation Summary ---")
    print(f"LGBM OOF Accuracy (thresh 0.5): {lgb_oof_acc:.4f}")
    print(f"Deep CatBoost OOF Accuracy (thresh 0.5): {deep_cat_oof_acc:.4f}")
    print(f"Shallow CatBoost OOF Accuracy (thresh 0.5): {shallow_cat_oof_acc:.4f}")
    print(f"Stacking Meta-Learner OOF Accuracy (thresh 0.5): {meta_oof_acc_05:.4f}")
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")
    print(f"Optimal Stacking OOF Accuracy: {best_acc:.4f}")

    final_test_predictions = (test_meta_probs >= best_threshold).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": final_test_predictions}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {best_acc}")


if __name__ == "__main__":
    main()
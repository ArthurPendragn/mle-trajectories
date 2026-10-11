import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

    # Deck-side normalized relative position and fine-grained spatial corridors
    manifest_deck_side = (
        manifest["CabinDeck"].fillna("Missing").astype(str)
        + "_"
        + manifest["CabinSide"].fillna("Missing").astype(str)
    )
    max_deck_side_cabin = (
        manifest.groupby(manifest_deck_side)["CabinNum"].max().replace(0, np.nan).to_dict()
    )
    df["CabinNum_Deck_Side_Ratio"] = df["CabinNum"] / df["Deck_Side"].map(max_deck_side_cabin)
    df["CabinBlock_25"] = (df["CabinNum"] // 25).fillna(-1).astype(float)
    df["CabinBlock_50"] = (df["CabinNum"] // 50).fillna(-1).astype(float)
    df["Deck_Side_Block"] = (
        df["Deck_Side"].astype(str)
        + "_"
        + (df["CabinNum"] // 50).fillna(-1).astype(int).astype(str)
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
        manifest_cryo_numeric = manifest["CryoSleep"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        group_cryo_avg = manifest_cryo_numeric.groupby(manifest["GroupID"]).mean()
        zero_spend_na_cryo = df["CryoSleep"].isna() & (df["TotalSpend"] == 0)
        group_cryo_vote = df["GroupID"].map(group_cryo_avg)
        df.loc[zero_spend_na_cryo & (group_cryo_vote > 0.5), "CryoSleep"] = True
        df.loc[zero_spend_na_cryo & (group_cryo_vote < 0.5), "CryoSleep"] = False

    # Cryo-Spend mismatch flag
    df["Cryo_Spend_Mismatch"] = (
        (df["TotalSpend"] > 0) & (df["CryoSleep"] == True)
    ).astype(int)

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

    # Discrete financial profile archetype
    spend_conditions = [
        df["TotalSpend"] == 0,
        (df["LuxurySpend"] > 0) & (df["CommunalSpend"] == 0),
        (df["CommunalSpend"] > 0) & (df["LuxurySpend"] == 0),
        (df["LuxurySpend"] > 0) & (df["CommunalSpend"] > 0),
    ]
    spend_choices = ["Zero_Spend", "Luxury_Only", "Communal_Only", "Mixed_Spend"]
    df["Spend_Type"] = np.select(spend_conditions, spend_choices, default="Zero_Spend")

    # Preferred amenity engineering
    best_amenity = df[amenity_cols].idxmax(axis=1)
    df["PreferredAmenity"] = best_amenity
    df.loc[df["TotalSpend"] == 0, "PreferredAmenity"] = "None"

    # Continuous 3D spatial coordinates
    deck_rank_map = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "T": 0.0}
    df["DeckRank"] = df["CabinDeck"].map(deck_rank_map).fillna(0.0).astype(float)
    side_sign_map = {"P": -1.0, "S": 1.0, "Missing": 0.0}
    df["SideSign"] = df["CabinSide"].map(side_sign_map).fillna(0.0).astype(float)

    df["Coord_X"] = df["CabinSide"].map(side_sign_map).fillna(0.0).astype(float)
    df["Coord_Y"] = df["DeckRank"].fillna(0.0).astype(float)
    df["Coord_Z"] = df["CabinNum_Deck_Ratio"].fillna(0.5).astype(float)
    df["Anomaly_Dist_3D"] = np.sqrt(
        df["Coord_X"] ** 2 + (df["Coord_Y"] - 4.0) ** 2 + (df["Coord_Z"] - 0.5) ** 2
    )

    # Amenity spend Shannon entropy
    entropy = np.zeros(len(df), dtype=float)
    pos_spend = df["TotalSpend"] > 0
    if pos_spend.any():
        p_spend = df.loc[pos_spend, amenity_cols].div(
            df.loc[pos_spend, "TotalSpend"], axis=0
        ).values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p_spend > 0, np.log(p_spend + 1e-12), 0.0)
            entropy[pos_spend] = -np.sum(p_spend * log_p, axis=1)
    df["Amenity_Entropy"] = entropy

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


def compute_relational_meta_features(
    df: pd.DataFrame, base_preds: np.ndarray
) -> pd.DataFrame:
    """Compute leave-one-out relational companion meta-features for groups and cabins."""
    preds_arr = np.asarray(base_preds, dtype=np.float64)
    if preds_arr.ndim == 2:
        if preds_arr.shape[1] == 1:
            p = preds_arr.ravel()
        elif preds_arr.shape[1] == 2 and np.allclose(
            preds_arr.sum(axis=1), 1.0, atol=1e-3
        ):
            p = preds_arr[:, 1]
        else:
            p = np.mean(preds_arr, axis=1)
    else:
        p = preds_arr.ravel()

    global_mean = float(np.mean(p))
    b = (p >= 0.5).astype(np.float64)
    global_rate = float(np.mean(b))

    n_samples = len(df)
    group_ids = df["GroupID"].astype(str).values

    # Group-level leave-one-out aggregations
    df_grp = pd.DataFrame({"GroupID": group_ids, "p": p, "p2": p**2, "b": b})
    grp_counts = df_grp.groupby("GroupID")["p"].transform("count").values
    grp_sums = df_grp.groupby("GroupID")["p"].transform("sum").values
    grp_sum2s = df_grp.groupby("GroupID")["p2"].transform("sum").values
    grp_bsums = df_grp.groupby("GroupID")["b"].transform("sum").values

    loo_grp_counts = grp_counts - 1
    has_grp_companions = loo_grp_counts > 0

    group_mean_pred = np.full(n_samples, global_mean, dtype=np.float64)
    group_mean_pred[has_grp_companions] = (
        grp_sums[has_grp_companions] - p[has_grp_companions]
    ) / loo_grp_counts[has_grp_companions]

    group_rate_pred = np.full(n_samples, global_rate, dtype=np.float64)
    group_rate_pred[has_grp_companions] = (
        grp_bsums[has_grp_companions] - b[has_grp_companions]
    ) / loo_grp_counts[has_grp_companions]

    group_std_pred = np.zeros(n_samples, dtype=np.float64)
    multi_grp_companions = loo_grp_counts >= 2
    if np.any(multi_grp_companions):
        comp_sum = grp_sums[multi_grp_companions] - p[multi_grp_companions]
        comp_sum2 = grp_sum2s[multi_grp_companions] - p[multi_grp_companions] ** 2
        comp_n = loo_grp_counts[multi_grp_companions]
        sample_var = (comp_sum2 - (comp_sum**2) / comp_n) / (comp_n - 1)
        sample_var = np.maximum(0.0, sample_var)
        group_std_pred[multi_grp_companions] = np.sqrt(sample_var)

    group_diff_pred = p - group_mean_pred

    # Cabin-level leave-one-out aggregations
    cabins = df["Cabin"].fillna("Missing").astype(str).values
    valid_cabin_mask = (cabins != "Missing") & (cabins != "") & (cabins != "nan")

    cabin_mean_pred = np.full(n_samples, global_mean, dtype=np.float64)
    cabin_rate_pred = np.full(n_samples, global_rate, dtype=np.float64)
    cabin_std_pred = np.zeros(n_samples, dtype=np.float64)

    if np.any(valid_cabin_mask):
        valid_indices = np.where(valid_cabin_mask)[0]
        valid_cabins = cabins[valid_indices]
        valid_p = p[valid_indices]
        valid_p2 = (p**2)[valid_indices]
        valid_b = b[valid_indices]

        df_cab = pd.DataFrame(
            {"Cabin": valid_cabins, "p": valid_p, "p2": valid_p2, "b": valid_b}
        )
        cab_counts = df_cab.groupby("Cabin")["p"].transform("count").values
        cab_sums = df_cab.groupby("Cabin")["p"].transform("sum").values
        cab_sum2s = df_cab.groupby("Cabin")["p2"].transform("sum").values
        cab_bsums = df_cab.groupby("Cabin")["b"].transform("sum").values

        loo_cab_counts = cab_counts - 1
        has_cab_companions = loo_cab_counts > 0

        sub_mean = np.full(len(valid_indices), global_mean, dtype=np.float64)
        sub_mean[has_cab_companions] = (
            cab_sums[has_cab_companions] - valid_p[has_cab_companions]
        ) / loo_cab_counts[has_cab_companions]
        cabin_mean_pred[valid_indices] = sub_mean

        sub_rate = np.full(len(valid_indices), global_rate, dtype=np.float64)
        sub_rate[has_cab_companions] = (
            cab_bsums[has_cab_companions] - valid_b[has_cab_companions]
        ) / loo_cab_counts[has_cab_companions]
        cabin_rate_pred[valid_indices] = sub_rate

        multi_cab_companions = loo_cab_counts >= 2
        if np.any(multi_cab_companions):
            c_comp_sum = cab_sums[multi_cab_companions] - valid_p[multi_cab_companions]
            c_comp_sum2 = (
                cab_sum2s[multi_cab_companions] - valid_p[multi_cab_companions] ** 2
            )
            c_comp_n = loo_cab_counts[multi_cab_companions]
            c_sample_var = (c_comp_sum2 - (c_comp_sum**2) / c_comp_n) / (
                c_comp_n - 1
            )
            c_sample_var = np.maximum(0.0, c_sample_var)
            sub_std = np.zeros(len(valid_indices), dtype=np.float64)
            sub_std[multi_cab_companions] = np.sqrt(c_sample_var)
            cabin_std_pred[valid_indices] = sub_std

    cabin_diff_pred = p - cabin_mean_pred

    relational_features = pd.DataFrame(
        {
            "Group_Companion_Mean_Pred": group_mean_pred,
            "Group_Companion_Std": group_std_pred,
            "Group_Companion_Rate": group_rate_pred,
            "Group_Companion_Diff": group_diff_pred,
            "Cabin_Companion_Mean_Pred": cabin_mean_pred,
            "Cabin_Companion_Rate": cabin_rate_pred,
            "Cabin_Companion_Std": cabin_std_pred,
            "Cabin_Companion_Diff": cabin_diff_pred,
        },
        index=df.index,
    )
    return relational_features


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

    train_df["CabinBlock_25"] = (train_df["CabinNum"] // 25).astype(float)
    test_df["CabinBlock_25"] = (test_df["CabinNum"] // 25).astype(float)

    train_df["CabinBlock_50"] = (train_df["CabinNum"] // 50).astype(float)
    test_df["CabinBlock_50"] = (test_df["CabinNum"] // 50).astype(float)

    train_df["Deck_Side_Block"] = (
        train_df["Deck_Side"].astype(str)
        + "_"
        + (train_df["CabinNum"] // 50).astype(int).astype(str)
    )
    test_df["Deck_Side_Block"] = (
        test_df["Deck_Side"].astype(str)
        + "_"
        + (test_df["CabinNum"] // 50).astype(int).astype(str)
    )

    train_df["SpendPerAge"] = train_df["TotalSpend"] / (train_df["Age"] + 1.0)
    test_df["SpendPerAge"] = test_df["TotalSpend"] / (test_df["Age"] + 1.0)

    train_df["Coord_Z"] = train_df["CabinNum_Deck_Ratio"]
    test_df["Coord_Z"] = test_df["CabinNum_Deck_Ratio"]
    train_df["Anomaly_Dist_3D"] = np.sqrt(
        train_df["Coord_X"] ** 2 + (train_df["Coord_Y"] - 4.0) ** 2 + (train_df["Coord_Z"] - 0.5) ** 2
    )
    test_df["Anomaly_Dist_3D"] = np.sqrt(
        test_df["Coord_X"] ** 2 + (test_df["Coord_Y"] - 4.0) ** 2 + (test_df["Coord_Z"] - 0.5) ** 2
    )

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
        "Deck_Side_Block",
        "Spend_Type",
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
        "Deck_Side_Decile_TE",
        "Deck_Side_Block_TE",
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
        "CabinBlock_25",
        "CabinBlock_50",
        "Coord_X",
        "Coord_Y",
        "Coord_Z",
        "Anomaly_Dist_3D",
        "Amenity_Entropy",
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
        "CabinRegion",
        "Deck_Side_TE",
        "CabinDeck_TE",
        "Route_TE",
        "Deck_Side_Decile_TE",
        "Deck_Side_Block_TE",
    ]

    binary_features = [
        "IsSolo",
        "HasFamily",
        "IsSoloCabin",
        "ZeroSpend",
        "Cryo_Spend_Mismatch",
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
    """Create a regularized LightGBM classifier with categorical smoothing."""
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=24,
        max_depth=6,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        cat_smooth=15.0,
        cat_l2=10.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=35,
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
            subsample=0.80,
            colsample_bytree=0.65,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=5.0,
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
            subsample=0.80,
            colsample_bytree=0.65,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=5.0,
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
    try:
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            one_hot_max_size=10,
            subsample=0.85,
            bootstrap_type="Bernoulli",
            eval_metric="Logloss",
            random_seed=seed,
            verbose=False,
        )
    except Exception:
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            one_hot_max_size=10,
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
# 2.1 Stacking Meta-Learner Architecture
# ------------------------------------------------------------------------------
class StackingMetaLearner:
    """Stage-2 regularized stacking meta-learner wrapping LogisticRegression."""

    def __init__(self, C: float = 0.1, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict_proba(X_scaled)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


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
        "Deck_Side_Block",
        "Spend_Type",
        "Route",
        "PreferredAmenity",
    ]
    cat_cols_for_models = [c for c in cat_columns if c in features]
    te_target_cols = [
        "Deck_Side",
        "CabinDeck",
        "Route",
        "Deck_Side_Decile",
        "Deck_Side_Block",
    ]

    X_test_base = test_df[features].copy()
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

        X_train_df = full_train_df.loc[train_mask, features].copy()
        y_train = full_train_df.loc[train_mask, target].values.astype(int)
        train_groups = full_train_df.loc[train_mask, "GroupID"].values

        X_val_df = full_train_df.loc[val_mask, features].copy()
        y_val = full_train_df.loc[val_mask, target].values.astype(int)

        X_test_df_fold = X_test_base.copy()

        # Strict nested out-of-fold target encoding with Bayesian smoothing (m=10.0)
        m = 10.0
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

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
                        "cat": full_train_df.loc[train_mask, c]
                        .iloc[inner_train_idx]
                        .astype(str),
                        "target": y_train[inner_train_idx],
                    }
                )
                inner_stats = inner_df.groupby("cat")["target"].agg(["count", "mean"])
                inner_counts = inner_stats["count"]
                inner_means = inner_stats["mean"]
                inner_smooth = (inner_counts * inner_means + m * inner_prior) / (
                    inner_counts + m
                )
                inner_te_map = inner_smooth.to_dict()

                inner_val_series = (
                    full_train_df.loc[train_mask, c].iloc[inner_val_idx].astype(str)
                )
                inner_mapped = (
                    inner_val_series.map(inner_te_map)
                    .fillna(inner_prior)
                    .astype(float)
                    .values
                )
                X_train_df.iloc[
                    inner_val_idx, X_train_df.columns.get_loc(te_name)
                ] = inner_mapped

        # Outer target encoding fit across full outer training split
        outer_prior = float(y_train.mean())
        for c in te_target_cols:
            te_name = f"{c}_TE"
            outer_df = pd.DataFrame(
                {
                    "cat": full_train_df.loc[train_mask, c].astype(str),
                    "target": y_train,
                }
            )
            outer_stats = outer_df.groupby("cat")["target"].agg(["count", "mean"])
            counts = outer_stats["count"]
            means = outer_stats["mean"]
            smooth = (counts * means + m * outer_prior) / (counts + m)
            te_map = smooth.to_dict()

            c_val_str = full_train_df.loc[val_mask, c].astype(str)
            X_val_df[te_name] = (
                c_val_str.map(te_map).fillna(outer_prior).astype(float).values
            )

            c_test_str = test_df[c].astype(str)
            c_test_mapped = (
                c_test_str.map(te_map).fillna(outer_prior).astype(float).values
            )
            X_test_df_fold[te_name] = c_test_mapped
            test_te_avg[te_name] += c_test_mapped / num_folds

        # 1. Train LightGBM
        X_train_lgb = X_train_df.copy()
        X_val_lgb = X_val_df.copy()
        X_test_lgb_fold = X_test_df_fold.copy()

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
        oof_preds_lgb[val_indices] = val_lgb_prob
        test_preds_lgb += lgb_model.predict_proba(X_test_lgb_fold)[:, 1] / num_folds
        fold_lgb_acc = accuracy_score(y_val, (val_lgb_prob >= 0.5).astype(int))

        # 2. Train XGBoost
        X_train_xgb = X_train_df.copy()
        X_val_xgb = X_val_df.copy()
        X_test_xgb_fold = X_test_df_fold.copy()
        for col in cat_cols_for_models:
            X_train_xgb[col] = X_train_xgb[col].astype("category")
            X_val_xgb[col] = X_val_xgb[col].astype("category")
            X_test_xgb_fold[col] = X_test_xgb_fold[col].astype("category")

        xgb_model = create_xgb_model(seed=42 + fold)
        xgb_model = fit_xgb(
            xgb_model,
            X_train_xgb,
            y_train,
            X_val_xgb,
            y_val,
        )
        val_xgb_prob = xgb_model.predict_proba(X_val_xgb)[:, 1]
        oof_preds_xgb[val_indices] = val_xgb_prob
        test_preds_xgb += xgb_model.predict_proba(X_test_xgb_fold)[:, 1] / num_folds
        fold_xgb_acc = accuracy_score(y_val, (val_xgb_prob >= 0.5).astype(int))

        # 3. Train CatBoost
        X_train_cat = X_train_df.copy()
        X_val_cat = X_val_df.copy()
        X_test_cat_fold = X_test_df_fold.copy()
        for col in cat_cols_for_models:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)
            X_test_cat_fold[col] = X_test_cat_fold[col].astype(str)

        cat_model = create_catboost_model(seed=42 + fold)
        cat_model = fit_catboost(
            cat_model,
            X_train_cat,
            y_train,
            X_val_cat,
            y_val,
            cat_features=cat_cols_for_models,
        )
        val_cat_prob = cat_model.predict_proba(X_val_cat)[:, 1]
        oof_preds_cat[val_indices] = val_cat_prob
        test_preds_cat += cat_model.predict_proba(X_test_cat_fold)[:, 1] / num_folds
        fold_cat_acc = accuracy_score(y_val, (val_cat_prob >= 0.5).astype(int))

        fold_triad_acc = accuracy_score(
            y_val,
            (((val_lgb_prob + val_xgb_prob + val_cat_prob) / 3.0) >= 0.5).astype(int),
        )
        print(
            f"Fold {fold} | LGBM: {fold_lgb_acc:.4f} | XGB: {fold_xgb_acc:.4f} | "
            f"CatBoost: {fold_cat_acc:.4f} | Triad Mean: {fold_triad_acc:.4f}"
        )

    lgb_oof_acc = accuracy_score(oof_targets, (oof_preds_lgb >= 0.5).astype(int))
    xgb_oof_acc = accuracy_score(oof_targets, (oof_preds_xgb >= 0.5).astype(int))
    cat_oof_acc = accuracy_score(oof_targets, (oof_preds_cat >= 0.5).astype(int))
    triad_oof_acc = accuracy_score(
        oof_targets,
        (((oof_preds_lgb + oof_preds_xgb + oof_preds_cat) / 3.0) >= 0.5).astype(int),
    )

    print(f"\n--- Level-1 GBDT Triad Summary ---")
    print(f"LGBM OOF Accuracy: {lgb_oof_acc:.4f}")
    print(f"XGBoost OOF Accuracy: {xgb_oof_acc:.4f}")
    print(f"CatBoost OOF Accuracy: {cat_oof_acc:.4f}")
    print(f"Triad Mean OOF Accuracy: {triad_oof_acc:.4f}")

    # Stage 2: Relational Companion Meta-Feature Construction
    print(f"\n--- Constructing Stage-2 Relational Meta-Features ---")
    L1_train = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat])
    L1_test = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])

    train_relational = compute_relational_meta_features(full_train_df, L1_train)
    test_relational = compute_relational_meta_features(test_df, L1_test)

    X_meta_train = np.hstack([L1_train, train_relational.values])
    X_meta_test = np.hstack([L1_test, test_relational.values])

    print(
        f"Stage-2 Feature Matrix Dimensions: Train={X_meta_train.shape}, Test={X_meta_test.shape}"
    )

    # Stage 2 Hierarchical Cross-Validation with StackingMetaLearner
    oof_meta_probs = np.zeros(len(full_train_df), dtype=np.float64)
    test_meta_probs_folds = np.zeros(len(test_df), dtype=np.float64)

    for fold in range(num_folds):
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold
        val_indices = np.where(val_mask.to_numpy())[0]

        meta_learner_fold = StackingMetaLearner(C=0.1, random_state=42 + fold)
        meta_learner_fold.fit(X_meta_train[train_mask], oof_targets[train_mask])

        val_meta_prob = meta_learner_fold.predict_proba(X_meta_train[val_mask])[:, 1]
        oof_meta_probs[val_indices] = val_meta_prob
        test_meta_probs_folds += (
            meta_learner_fold.predict_proba(X_meta_test)[:, 1] / num_folds
        )

    # Refit StackingMetaLearner across full out-of-fold meta-features
    meta_learner_full = StackingMetaLearner(C=0.1, random_state=42)
    meta_learner_full.fit(X_meta_train, oof_targets)
    test_meta_probs_full = meta_learner_full.predict_proba(X_meta_test)[:, 1]

    # Combine fold-averaged and full-fit meta predictions
    test_meta_probs = 0.5 * test_meta_probs_folds + 0.5 * test_meta_probs_full

    meta_feature_names = [
        "Pred_LGBM",
        "Pred_XGB",
        "Pred_CatBoost",
    ] + list(train_relational.columns)
    print("\nStage-2 Meta-Learner Learned Weights:")
    for feat_name, coef_val in zip(meta_feature_names, meta_learner_full.model.coef_[0]):
        print(f"  {feat_name:28s}: {coef_val:+.4f}")
    print(f"  Intercept                     : {meta_learner_full.model.intercept_[0]:+.4f}")

    # Direct 1D Accuracy Threshold Calibration: [0.40, 0.60] with 0.005 step
    thresholds = np.arange(0.40, 0.60001, 0.005)
    best_acc = -1.0
    best_threshold = 0.5
    targets_bool = (oof_targets == 1)

    for t in thresholds:
        acc = ((oof_meta_probs >= t) == targets_bool).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    default_meta_acc = ((oof_meta_probs >= 0.50) == targets_bool).mean()
    print(f"\n--- Stage-2 Stacking Evaluation ---")
    print(f"Stacking Meta OOF Accuracy (thresh 0.50): {default_meta_acc:.4f}")
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")
    print(f"Optimized Stacking OOF Accuracy: {best_acc:.4f}")

    # Generate Test Predictions
    test_predictions = (test_meta_probs >= best_threshold).astype(bool)

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
import copy
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from lightgbm import LGBMClassifier
try:
    from lightgbm import early_stopping, log_evaluation
except ImportError:
    early_stopping = None
    log_evaluation = None
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)


# ---------------------------------------------------------------------------
# Step 1: Data Processing & Feature Engineering
# ---------------------------------------------------------------------------
def locate_input_file(filename):
    """Search possible paths for the input file."""
    candidates = [
        os.path.join("./input", filename),
        os.path.join("./input/input", filename),
        os.path.join(".", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"File {filename} not found in expected locations.")


def extract_domain_features(df, global_manifest=None):
    """Extract deterministic structural, relational, and domain features prior to stateful imputation."""
    df = df.copy()

    # Build or parse global manifest to allow global demographic and relational propagation
    if global_manifest is None:
        try:
            tr_p = locate_input_file("train.csv")
            te_p = locate_input_file("test.csv")
            tr_df = pd.read_csv(tr_p)
            te_df = pd.read_csv(te_p)
            gm = pd.concat(
                [tr_df.drop(columns=["Transported"], errors="ignore"), te_df],
                ignore_index=True,
            )
        except Exception:
            gm = df.copy()
    else:
        gm = global_manifest.copy()

    # Ensure structural identity columns exist on global manifest
    if "GroupId" not in gm.columns:
        gm_p_parts = gm["PassengerId"].str.split("_", expand=True)
        gm["GroupId"] = gm_p_parts[0]
        gm["GroupNum"] = gm_p_parts[1].astype(int)
    if "CabinDeck" not in gm.columns:
        gm_c_parts = gm["Cabin"].str.split("/", expand=True)
        gm["CabinDeck"] = gm_c_parts[0]
        gm["CabinNum"] = pd.to_numeric(gm_c_parts[1], errors="coerce")
        gm["CabinSide"] = gm_c_parts[2]
    if "Surname" not in gm.columns:
        gm_names = gm["Name"].str.split(" ", expand=True)
        gm["Surname"] = gm_names[1].fillna("Unknown")

    # 1. Structural passenger decomposition on local df
    p_parts = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = p_parts[0]
    df["GroupNum"] = p_parts[1].astype(int)

    c_parts = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = c_parts[0]
    df["CabinNum"] = pd.to_numeric(c_parts[1], errors="coerce")
    df["CabinSide"] = c_parts[2]

    names = df["Name"].str.split(" ", expand=True)
    df["Surname"] = names[1].fillna("Unknown")

    # Global manifest demographic counts and group leader flag
    global_group_counts = gm["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(global_group_counts).fillna(1).astype(int)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["IsGroupLeader"] = (df["GroupNum"] == 1).astype(int)

    valid_surnames = gm[gm["Surname"] != "Unknown"]["Surname"]
    global_surname_counts = valid_surnames.value_counts()
    df["FamilySize"] = df["Surname"].map(global_surname_counts).fillna(1).astype(int)
    df.loc[df["Surname"] == "Unknown", "FamilySize"] = 1

    # 2. Relational co-traveler and deductive imputation
    # Deterministic physical deck rules: Decks A, B, C, T are Europa; Deck G is Earth
    gm.loc[gm["CabinDeck"].isin(["A", "B", "C", "T"]) & gm["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    gm.loc[(gm["CabinDeck"] == "G") & gm["HomePlanet"].isna(), "HomePlanet"] = "Earth"
    df.loc[df["CabinDeck"].isin(["A", "B", "C", "T"]) & df["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    df.loc[(df["CabinDeck"] == "G") & df["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Propagate HomePlanet via GroupId and Surname
    hp_valid = gm.dropna(subset=["HomePlanet"])
    hp_group = hp_valid.groupby("GroupId")["HomePlanet"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["GroupId"].map(hp_group))

    hp_sur_valid = hp_valid[hp_valid["Surname"] != "Unknown"]
    hp_surname = hp_sur_valid.groupby("Surname")["HomePlanet"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["Surname"].map(hp_surname))

    # Re-apply deck rules if Deck was known
    df.loc[df["CabinDeck"].isin(["A", "B", "C", "T"]) & df["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    df.loc[(df["CabinDeck"] == "G") & df["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Propagate Destination via GroupId and Surname
    dest_valid = gm.dropna(subset=["Destination"])
    dest_group = dest_valid.groupby("GroupId")["Destination"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["Destination"] = df["Destination"].fillna(df["GroupId"].map(dest_group))

    dest_sur_valid = dest_valid[dest_valid["Surname"] != "Unknown"]
    dest_surname = dest_sur_valid.groupby("Surname")["Destination"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["Destination"] = df["Destination"].fillna(df["Surname"].map(dest_surname))

    # Propagate CabinDeck via GroupId and Surname
    deck_valid = gm.dropna(subset=["CabinDeck"])
    deck_valid = deck_valid[~deck_valid["CabinDeck"].isin(["Missing", ""])]
    deck_group = deck_valid.groupby("GroupId")["CabinDeck"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinDeck"] = df["CabinDeck"].fillna(df["GroupId"].map(deck_group))

    deck_sur_valid = deck_valid[deck_valid["Surname"] != "Unknown"]
    deck_surname = deck_sur_valid.groupby("Surname")["CabinDeck"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinDeck"] = df["CabinDeck"].fillna(df["Surname"].map(deck_surname))

    # Propagate CabinSide via GroupId and Surname
    side_valid = gm.dropna(subset=["CabinSide"])
    side_valid = side_valid[~side_valid["CabinSide"].isin(["Missing", ""])]
    side_group = side_valid.groupby("GroupId")["CabinSide"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinSide"] = df["CabinSide"].fillna(df["GroupId"].map(side_group))

    side_sur_valid = side_valid[side_valid["Surname"] != "Unknown"]
    side_surname = side_sur_valid.groupby("Surname")["CabinSide"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinSide"] = df["CabinSide"].fillna(df["Surname"].map(side_surname))

    # Propagate CabinNum via GroupId and Surname
    num_valid = gm.dropna(subset=["CabinNum"])
    num_group = num_valid.groupby("GroupId")["CabinNum"].median()
    df["CabinNum"] = df["CabinNum"].fillna(df["GroupId"].map(num_group))

    num_sur_valid = num_valid[num_valid["Surname"] != "Unknown"]
    num_surname = num_sur_valid.groupby("Surname")["CabinNum"].median()
    df["CabinNum"] = df["CabinNum"].fillna(df["Surname"].map(num_surname))

    df["CabinDeck"] = df["CabinDeck"].fillna("Missing")
    df["CabinSide"] = df["CabinSide"].fillna("Missing")

    # Spatial hull features
    df["CabinDeck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]
    df["CabinZone"] = df["CabinNum"] // 300
    df["CabinNum_Odd"] = (df["CabinNum"] % 2)

    # Cabin room occupancy counting from global manifest
    cabin_counts = gm["Cabin"].dropna().value_counts()
    df["Cabin_Occupancy"] = df["Cabin"].map(cabin_counts).fillna(1).astype(float)

    # 3. Amenity expenditure, luxury vs communal spending, and constraints
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physics constraint: Confirmed CryoSleep passengers spend 0 on luxury amenities
    is_cryo = df["CryoSleep"] == True
    for col in amenity_cols:
        df.loc[is_cryo & df[col].isna(), col] = 0.0

    # Physics constraint: Children under 13 cannot spend on amenities
    is_child = df["Age"] < 13
    for col in amenity_cols:
        df.loc[is_child & df[col].isna(), col] = 0.0

    # Total, Luxury, and Communal spending aggregations
    df["TotalSpending"] = df[amenity_cols].sum(axis=1, skipna=True)
    df["LuxurySpending"] = (
        df["RoomService"].fillna(0.0)
        + df["Spa"].fillna(0.0)
        + df["VRDeck"].fillna(0.0)
    )
    df["CommunalSpending"] = (
        df["FoodCourt"].fillna(0.0)
        + df["ShoppingMall"].fillna(0.0)
    )
    df["Communal_Minus_Luxury"] = df["CommunalSpending"] - df["LuxurySpending"]

    # Group-level expenditure aggregations using global manifest
    for col in amenity_cols:
        gm[col] = pd.to_numeric(gm[col], errors="coerce")
    gm.loc[gm["CryoSleep"] == True, amenity_cols] = 0.0
    gm.loc[gm["Age"] < 13, amenity_cols] = 0.0
    gm_spend = gm[amenity_cols].sum(axis=1, skipna=True)
    group_total_map = gm_spend.groupby(gm["GroupId"]).sum()
    group_mean_map = gm_spend.groupby(gm["GroupId"]).mean()

    df["Group_TotalSpend"] = df["GroupId"].map(group_total_map).fillna(df["TotalSpending"])
    df["Group_MeanSpend"] = df["GroupId"].map(group_mean_map).fillna(df["TotalSpending"])
    df["Spend_To_Group_Ratio"] = df["TotalSpending"] / (df["Group_TotalSpend"] + 1.0)

    # Explicit spend and zero-spend flags
    df["HasSpent"] = (df["TotalSpending"] > 0).astype(int)
    df["ZeroSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["ZeroLuxury"] = (df["LuxurySpending"] == 0).astype(int)
    df["ZeroCommunal"] = (df["CommunalSpending"] == 0).astype(int)

    # Per-amenity zero expenditure flags
    for col in amenity_cols:
        df[f"Zero_{col}"] = (df[col].fillna(0.0) == 0.0).astype(int)

    # Inverse physics constraint: Anyone who spent money was NOT in CryoSleep
    spent_money = df["TotalSpending"] > 0
    df.loc[spent_money & df["CryoSleep"].isna(), "CryoSleep"] = False

    # Group CryoSleep priors from manifest
    gm_spent = gm_spend > 0
    gm.loc[gm_spent & gm["CryoSleep"].isna(), "CryoSleep"] = False
    gm_cryo_num = gm["CryoSleep"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    group_cryo_map = gm_cryo_num.groupby(gm["GroupId"]).mean()
    df["Group_Cryo_Ratio"] = df["GroupId"].map(group_cryo_map).fillna(gm_cryo_num.mean())

    # Proportional and log transformations
    for col in amenity_cols:
        df[f"{col}_Ratio"] = df[col].fillna(0.0) / (df["TotalSpending"] + 1.0)
        df[f"Log_{col}"] = np.log1p(np.maximum(df[col].fillna(0.0), 0.0))

    df["Log_TotalSpending"] = np.log1p(np.maximum(df["TotalSpending"], 0.0))
    df["Log_LuxurySpending"] = np.log1p(np.maximum(df["LuxurySpending"], 0.0))
    df["Log_CommunalSpending"] = np.log1p(np.maximum(df["CommunalSpending"], 0.0))

    df["Luxury_Ratio"] = df["LuxurySpending"] / (df["TotalSpending"] + 1.0)
    df["Communal_Ratio"] = df["CommunalSpending"] / (df["TotalSpending"] + 1.0)

    # Number of amenities used
    df["AmenitiesUsedCount"] = (df[amenity_cols].fillna(0.0) > 0).sum(axis=1)

    # VIP deductive constraints
    df.loc[df["Age"] < 18, "VIP"] = False
    df.loc[df["HomePlanet"] == "Earth", "VIP"] = False
    df.loc[df["CabinDeck"].isin(["G", "T"]), "VIP"] = False

    return df


def prepare_datasets():
    os.makedirs("./working", exist_ok=True)

    train_path = locate_input_file("train.csv")
    test_path = locate_input_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df[["PassengerId"]].copy()

    # Extract target from train
    y = train_df["Transported"].astype(int).values

    # Unified global passenger manifest (excluding labels)
    full_manifest = pd.concat(
        [train_df.drop(columns=["Transported"], errors="ignore"), test_df],
        ignore_index=True,
    )

    # Stratified hold-out split indices for validation preservation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(train_df, y))

    # Apply deterministic and relational feature extraction with global manifest
    X_train_raw = extract_domain_features(train_df, global_manifest=full_manifest)
    X_test_raw = extract_domain_features(test_df, global_manifest=full_manifest)

    # Categorical and numerical columns definition
    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "CabinDeck_Side",
    ]
    numeric_cols = [
        "Age",
        "CabinNum",
        "CabinZone",
        "GroupSize",
        "GroupNum",
        "IsAlone",
        "IsGroupLeader",
        "FamilySize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpending",
        "LuxurySpending",
        "CommunalSpending",
        "HasSpent",
        "ZeroSpending",
        "ZeroLuxury",
        "ZeroCommunal",
        "AmenitiesUsedCount",
        "Log_RoomService",
        "Log_FoodCourt",
        "Log_ShoppingMall",
        "Log_Spa",
        "Log_VRDeck",
        "Log_TotalSpending",
        "Log_LuxurySpending",
        "Log_CommunalSpending",
        "RoomService_Ratio",
        "FoodCourt_Ratio",
        "ShoppingMall_Ratio",
        "Spa_Ratio",
        "VRDeck_Ratio",
        "Luxury_Ratio",
        "Communal_Ratio",
        "Group_TotalSpend",
        "Group_MeanSpend",
        "Spend_To_Group_Ratio",
        "Cabin_Occupancy",
        "CabinNum_Odd",
        "Group_Cryo_Ratio",
        "Communal_Minus_Luxury",
        "Zero_RoomService",
        "Zero_FoodCourt",
        "Zero_ShoppingMall",
        "Zero_Spa",
        "Zero_VRDeck",
    ]

    # STRICT ISOLATION: Learn imputers strictly from training split
    num_medians = {col: X_train_raw.iloc[train_idx][col].median() for col in numeric_cols}
    for col in numeric_cols:
        if pd.isna(num_medians[col]):
            num_medians[col] = 0.0

    cat_modes = {}
    for col in categorical_cols:
        mode_val = X_train_raw.iloc[train_idx][col].mode(dropna=True)
        cat_modes[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "Missing"

    def apply_imputation(df):
        df_imputed = df.copy()
        for col in numeric_cols:
            df_imputed[col] = df_imputed[col].fillna(num_medians[col])
        for col in categorical_cols:
            df_imputed[col] = df_imputed[col].fillna(cat_modes[col]).astype(str)
        return df_imputed

    X_train_imp = apply_imputation(X_train_raw)
    X_test_imp = apply_imputation(X_test_raw)

    # Ordinal encoding: Map categoricals into clean non-negative integers
    cat_encodings = {}
    for col in categorical_cols:
        train_cats = sorted(X_train_imp.iloc[train_idx][col].unique().tolist())
        cat_encodings[col] = {val: idx for idx, val in enumerate(train_cats)}

    def apply_ordinal_encoding(df):
        df_encoded = df.copy()
        for col in categorical_cols:
            mapping = cat_encodings[col]
            fallback_val = len(mapping)
            df_encoded[col] = (
                df_encoded[col]
                .astype(str)
                .map(lambda x: mapping.get(x, fallback_val))
                .astype(int)
            )
        return df_encoded

    feature_cols = numeric_cols + categorical_cols
    X_train_encoded = apply_ordinal_encoding(X_train_imp)[feature_cols].copy()
    X_test_encoded = apply_ordinal_encoding(X_test_imp)[feature_cols].copy()

    X_train_final = X_train_encoded
    X_test_final = X_test_encoded
    X_val_final = X_train_final.iloc[val_idx].reset_index(drop=True)

    y_train_df = pd.DataFrame({"Transported": y})
    y_val_df = pd.DataFrame({"Transported": y[val_idx]})

    # Save prepared data to ./working
    X_train_final.to_csv("./working/X_train.csv", index=False)
    y_train_df.to_csv("./working/y_train.csv", index=False)
    X_val_final.to_csv("./working/X_val.csv", index=False)
    y_val_df.to_csv("./working/y_val.csv", index=False)
    X_test_final.to_csv("./working/X_test.csv", index=False)
    test_ids.to_csv("./working/test_ids.csv", index=False)

    return (
        X_train_final,
        y_train_df,
        X_val_final,
        y_val_df,
        X_test_final,
        test_ids,
    )


# ---------------------------------------------------------------------------
# Step 2: Model Design (LightGBM & XGBoost Ensemble)
# ---------------------------------------------------------------------------
def get_lgbm_model(random_state=42):
    """Instantiate LightGBM classifier with conservative regularized hyperparameters."""
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state=42):
    """Instantiate XGBoost classifier with depth-wise regularized growth."""
    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_third_model(random_state=42):
    """Instantiate CatBoost classifier (depth=6, l2_leaf_reg=3.0) with seamless fallback to HistGradientBoosting."""
    if CatBoostClassifier is not None:
        try:
            return CatBoostClassifier(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                eval_metric="Logloss",
                random_seed=random_state,
                verbose=False,
                early_stopping_rounds=50,
            )
        except Exception:
            pass
    return HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=6,
        l2_regularization=3.0,
        early_stopping=True,
        n_iter_no_change=50,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Step 3: Training, Evaluation & Test Inference (5-Fold Stratified CV)
# ---------------------------------------------------------------------------
def main():
    X_train_final, y_train_df, X_val_final, y_val_df, X_test_final, test_ids = (
        prepare_datasets()
    )

    X_train = X_train_final.values
    y_train = y_train_df["Transported"].values
    X_test = X_test_final.values

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(y_train), dtype=np.float64)
    oof_preds_xgb = np.zeros(len(y_train), dtype=np.float64)
    oof_preds_third = np.zeros(len(y_train), dtype=np.float64)

    test_preds_lgb = np.zeros(len(X_test), dtype=np.float64)
    test_preds_xgb = np.zeros(len(X_test), dtype=np.float64)
    test_preds_third = np.zeros(len(X_test), dtype=np.float64)

    print(f"Starting {n_splits}-fold Stratified Cross-Validation on {len(X_train)} records...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # 1. Fit LightGBM
        lgb_model = get_lgbm_model(random_state=42 + fold)
        if early_stopping is not None:
            lgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
            )
        else:
            try:
                lgb_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="binary_logloss",
                    early_stopping_rounds=50,
                    verbose=False,
                )
            except Exception:
                lgb_model.fit(X_tr, y_tr)

        lgb_val_prob = lgb_model.predict_proba(X_va)[:, 1]
        lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]

        # 2. Fit XGBoost
        xgb_model = get_xgb_model(random_state=42 + fold)
        try:
            xgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
        except TypeError:
            xgb_model = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42 + fold,
                n_jobs=-1,
                tree_method="hist",
            )
            xgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=50,
                verbose=False,
            )

        xgb_val_prob = xgb_model.predict_proba(X_va)[:, 1]
        xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]

        # 3. Fit Third Model (CatBoost / HistGB)
        third_model = get_third_model(random_state=42 + fold)
        try:
            third_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                verbose=False,
            )
        except TypeError:
            third_model.fit(X_tr, y_tr)
        except Exception:
            third_model.fit(X_tr, y_tr)

        third_val_prob = third_model.predict_proba(X_va)[:, 1]
        third_test_prob = third_model.predict_proba(X_test)[:, 1]

        # Accumulate out-of-fold and test predictions
        oof_preds_lgb[val_idx] = lgb_val_prob
        oof_preds_xgb[val_idx] = xgb_val_prob
        oof_preds_third[val_idx] = third_val_prob

        test_preds_lgb += lgb_test_prob / n_splits
        test_preds_xgb += xgb_test_prob / n_splits
        test_preds_third += third_test_prob / n_splits

        fold_equal_blend = (lgb_val_prob + xgb_val_prob + third_val_prob) / 3.0
        fold_acc = accuracy_score(y_va, (fold_equal_blend >= 0.5).astype(int))
        lgb_acc = accuracy_score(y_va, (lgb_val_prob >= 0.5).astype(int))
        xgb_acc = accuracy_score(y_va, (xgb_val_prob >= 0.5).astype(int))
        third_acc = accuracy_score(y_va, (third_val_prob >= 0.5).astype(int))
        print(
            f"Fold {fold + 1}/{n_splits} | Equal: {fold_acc:.4f} | LGB: {lgb_acc:.4f} | XGB: {xgb_acc:.4f} | Third: {third_acc:.4f}"
        )

    # Bounded scipy optimization for out-of-fold probability ensemble weights
    def brier_or_logloss(w_candidates):
        w = np.array(w_candidates)
        s = np.sum(w)
        if s <= 0:
            return 1e5
        w = w / s
        p = (
            w[0] * oof_preds_lgb
            + w[1] * oof_preds_xgb
            + w[2] * oof_preds_third
        )
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))

    init_w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        brier_or_logloss,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if opt_res.success and np.sum(opt_res.x) > 0:
        opt_weights = opt_res.x / np.sum(opt_res.x)
    else:
        opt_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    print(
        f"Optimized Ensemble Weights: LGB={opt_weights[0]:.4f}, "
        f"XGB={opt_weights[1]:.4f}, Third={opt_weights[2]:.4f}"
    )

    oof_blend = (
        opt_weights[0] * oof_preds_lgb
        + opt_weights[1] * oof_preds_xgb
        + opt_weights[2] * oof_preds_third
    )

    # 1D grid search for optimal decision threshold maximizing validation accuracy
    best_threshold = 0.5
    best_score = accuracy_score(y_train, (oof_blend >= 0.5).astype(int))

    threshold_candidates = np.linspace(0.40, 0.60, 201)
    for t in threshold_candidates:
        score = accuracy_score(y_train, (oof_blend >= t).astype(int))
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Optimal Decision Threshold: {best_threshold:.4f} (Validation Accuracy: {best_score:.4f})")

    final_score = best_score
    print(f"Final Validation Score: {final_score}")

    # Apply identical learned weights and threshold to test probabilities
    test_blend = (
        opt_weights[0] * test_preds_lgb
        + opt_weights[1] * test_preds_xgb
        + opt_weights[2] * test_preds_third
    )
    test_binary = (test_blend >= best_threshold).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "PassengerId": test_ids["PassengerId"],
            "Transported": test_binary,
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path} ({len(submission_df)} rows)")


if __name__ == "__main__":
    main()

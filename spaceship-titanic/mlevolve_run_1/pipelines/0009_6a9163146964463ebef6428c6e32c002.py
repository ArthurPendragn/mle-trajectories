import copy
import json
import math
import os
from typing import List, Optional

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

# Set deterministic random seeds
SEED = 42
np.random.seed(SEED)


def find_file(filename: str) -> str:
    """Locate file across standard directory structures."""
    search_dirs = ["./input", "./input/input", ".", "./data"]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"File {filename} not found in candidate paths: {search_dirs}"
    )


# =========================================================================
# 1. MODEL ARCHITECTURE: HETEROGENEOUS GBDT ENSEMBLE WRAPPERS
# =========================================================================


class ProbArray(np.ndarray):
    """1D probability array supporting both 1D indexing and [:, 1] indexing."""

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            row_idx, col_idx = item
            if col_idx == 1:
                return np.asarray(self)[row_idx]
            elif col_idx == 0:
                return 1.0 - np.asarray(self)[row_idx]
        return super().__getitem__(item)


class LGBWrapper:
    """LightGBM Classifier wrapper configured for tabular categorical handling."""

    def __init__(self, cat_cols: Optional[List[str]] = None, cat_dtypes: Optional[dict] = None, seed: int = SEED):
        self.cat_cols = cat_cols or []
        self.cat_dtypes = cat_dtypes or {}
        self.seed = seed
        self.model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        X_tr = X_train.copy()
        for c in self.cat_cols:
            if c in X_tr.columns and c in self.cat_dtypes:
                X_tr[c] = X_tr[c].astype(self.cat_dtypes[c])
            elif c in X_tr.columns:
                X_tr[c] = X_tr[c].astype("category")
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.copy()
            for c in self.cat_cols:
                if c in X_v.columns and c in self.cat_dtypes:
                    X_v[c] = X_v[c].astype(self.cat_dtypes[c])
                elif c in X_v.columns:
                    X_v[c] = X_v[c].astype("category")
            eval_set = [(X_v, y_val)]
        self.model.fit(
            X_tr,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> ProbArray:
        X_df = X.copy()
        for c in self.cat_cols:
            if c in X_df.columns and c in self.cat_dtypes:
                X_df[c] = X_df[c].astype(self.cat_dtypes[c])
            elif c in X_df.columns:
                X_df[c] = X_df[c].astype("category")
        probs = self.model.predict_proba(X_df)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


class XGBWrapper:
    """XGBoost Classifier wrapper with robust numeric categorical representation."""

    def __init__(self, cat_cols: Optional[List[str]] = None, seed: int = SEED):
        self.cat_cols = cat_cols or []
        self.seed = seed
        self.model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
            early_stopping_rounds=50,
            n_jobs=-1,
            enable_categorical=False,
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        X_tr = X_train.copy()
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.copy()
            eval_set = [(X_v, y_val)]
        self.model.fit(
            X_tr,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> ProbArray:
        probs = self.model.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


class CatBoostWrapper:
    """CatBoost Classifier wrapper configured for symmetric oblivious tree learning."""

    def __init__(self, cat_cols: Optional[List[str]] = None, seed: int = SEED):
        self.cat_cols = cat_cols or []
        self.seed = seed
        self.model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            eval_metric="Logloss",
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        X_tr = X_train.copy()
        for c in self.cat_cols:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype(int)
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.copy()
            for c in self.cat_cols:
                if c in X_v.columns:
                    X_v[c] = X_v[c].astype(int)
            eval_set = (X_v, y_val)
        valid_cat_cols = [c for c in self.cat_cols if c in X_tr.columns]
        self.model.fit(
            X_tr,
            y_train,
            cat_features=valid_cat_cols,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> ProbArray:
        X_df = X.copy()
        for c in self.cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype(int)
        probs = self.model.predict_proba(X_df)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


def get_lgb_model(cat_cols: Optional[List[str]] = None, cat_dtypes: Optional[dict] = None, seed: int = SEED) -> LGBWrapper:
    """Returns a configured LightGBM model wrapper."""
    return LGBWrapper(cat_cols=cat_cols, cat_dtypes=cat_dtypes, seed=seed)


def get_xgb_model(cat_cols: Optional[List[str]] = None, seed: int = SEED) -> XGBWrapper:
    """Returns a configured XGBoost model wrapper."""
    return XGBWrapper(cat_cols=cat_cols, seed=seed)


def get_catboost_model(cat_cols: Optional[List[str]] = None, seed: int = SEED) -> CatBoostWrapper:
    """Returns a configured CatBoost model wrapper."""
    return CatBoostWrapper(cat_cols=cat_cols, seed=seed)


# =========================================================================
# 2. DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================


def process_features():
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train = len(train_df)
    y_train = train_df["Transported"].astype(int)
    test_df["Transported"] = np.nan

    # Combine for relational metadata and topology extraction
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Group decomposition
    df["GroupId"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["GroupPos"] = df["PassengerId"].apply(lambda x: int(x.split("_")[1]))
    group_sizes = df["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(group_sizes)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    # Name and surname analysis
    df["Name"] = df["Name"].fillna("Unknown Unknown")
    df["FirstName"] = df["Name"].apply(
        lambda x: x.split()[0] if len(x.split()) > 0 else "Unknown"
    )
    df["LastName"] = df["Name"].apply(
        lambda x: x.split()[-1] if len(x.split()) > 1 else "Unknown"
    )
    family_counts = df[df["LastName"] != "Unknown"]["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_counts).fillna(1).astype(int)
    df["IsSoloFamily"] = (df["FamilySize"] == 1).astype(int)

    # Cabin topology decomposition
    cabin_split = (
        df["Cabin"]
        .fillna("Missing/Missing/Missing")
        .apply(
            lambda x: (
                x.split("/") if "/" in str(x) else ["Missing", "Missing", "Missing"]
            )
        )
    )
    df["CabinDeck"] = cabin_split.apply(lambda x: x[0] if len(x) == 3 else "Missing")
    df["CabinNumRaw"] = cabin_split.apply(lambda x: x[1] if len(x) == 3 else "Missing")
    df["CabinSide"] = cabin_split.apply(lambda x: x[2] if len(x) == 3 else "Missing")

    df["CabinNum"] = pd.to_numeric(df["CabinNumRaw"], errors="coerce")
    deck_num_median = df.groupby("CabinDeck")["CabinNum"].transform("median").fillna(-1)
    df["CabinNum"] = df["CabinNum"].fillna(deck_num_median)

    # Continuous cabin deck percentile within deck
    df["CabinDeckNumPercentile"] = (
        df.groupby("CabinDeck")["CabinNum"].rank(pct=True).fillna(0.5)
    )

    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)
    df["CabinCluster50"] = (df["CabinNum"] // 50).fillna(-1).astype(int)
    df["CabinDeck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]
    df["CabinDeck_Side_Cluster"] = (
        df["CabinDeck"]
        + "_"
        + df["CabinSide"]
        + "_"
        + df["CabinCluster50"].astype(str)
    )

    # Domain-informed physical deduction
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    # CryoSleep spending invariant: CryoSleep == True cannot spend on amenities
    for amenity in amenities:
        df.loc[df["CryoSleep"] == True, amenity] = df.loc[
            df["CryoSleep"] == True, amenity
        ].fillna(0.0)

    # If spending > 0, CryoSleep cannot be True -> must be False
    total_raw_spend = df[amenities].sum(axis=1)
    df.loc[(df["CryoSleep"].isna()) & (total_raw_spend > 0), "CryoSleep"] = False

    # Minors under 13 cannot spend on amenities
    for amenity in amenities:
        df.loc[df["Age"] < 13, amenity] = df.loc[df["Age"] < 13, amenity].fillna(0.0)

    # VIP physical deck & planet invariants
    df.loc[(df["HomePlanet"] == "Earth") & (df["VIP"].isna()), "VIP"] = False
    df.loc[(df["CabinDeck"].isin(["G", "T"])) & (df["VIP"].isna()), "VIP"] = False

    # Intra-Group Relational Imputation for HomePlanet
    group_planet = df[df["HomePlanet"].notna()].groupby("GroupId")["HomePlanet"].first()
    df.loc[df["HomePlanet"].isna(), "HomePlanet"] = df.loc[
        df["HomePlanet"].isna(), "GroupId"
    ].map(group_planet)

    # Deck constraint imputation for HomePlanet
    df.loc[
        (df["HomePlanet"].isna()) & (df["CabinDeck"].isin(["A", "B", "C", "T"])),
        "HomePlanet",
    ] = "Europa"
    df.loc[(df["HomePlanet"].isna()) & (df["CabinDeck"] == "G"), "HomePlanet"] = "Earth"

    # Fill remaining amenities with 0.0
    for amenity in amenities:
        df[amenity] = df[amenity].fillna(0.0)

    # Retain NaN CryoSleep after physical deductions: Dedicated missingness indicator
    df["CryoSleep_is_na"] = df["CryoSleep"].isna().astype(int)

    # Explicit ternary categorical representation (0 = Awake/False, 1 = Asleep/True, 2 = Unknown/Missing)
    cryo_ternary = pd.Series(2, index=df.index, dtype=int)
    cryo_ternary[df["CryoSleep"] == False] = 0
    cryo_ternary[df["CryoSleep"] == True] = 1
    df["CryoSleep"] = cryo_ternary

    df["VIP"] = (df["VIP"] == True).astype(int)
    df["HomePlanet"] = df["HomePlanet"].fillna("Missing")
    df["Destination"] = df["Destination"].fillna("Missing")

    # Age Imputation: Median by Deck and HomePlanet
    age_medians = (
        df.groupby(["HomePlanet", "CabinDeck"])["Age"]
        .transform("median")
        .fillna(df["Age"].median())
    )
    df["Age"] = df["Age"].fillna(age_medians)

    # Advanced Expenditure & Ratio Features
    df["TotalSpend"] = df[amenities].sum(axis=1)
    df["HasSpent"] = (df["TotalSpend"] > 0).astype(int)
    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["AmenitiesSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["AmenitiesRatio"] = df["AmenitiesSpend"] / (df["TotalSpend"] + 1.0)

    for amenity in amenities:
        df[f"{amenity}_Log"] = np.log1p(df[amenity])
        df[f"{amenity}_Ratio"] = df[amenity] / (df["TotalSpend"] + 1.0)
    df["TotalSpend_Log"] = np.log1p(df["TotalSpend"])
    df["NumAmenitiesUsed"] = (df[amenities] > 0).sum(axis=1)

    # Group Expenditure Dynamics
    group_total_spend = df.groupby("GroupId")["TotalSpend"].transform("sum")
    df["GroupTotalSpend"] = group_total_spend
    df["SpendToGroupRatio"] = df["TotalSpend"] / (df["GroupTotalSpend"] + 1.0)

    group_mean_spend = df.groupby("GroupId")["TotalSpend"].transform("mean")
    df["GroupMeanSpend"] = group_mean_spend
    df["SpendDiffFromGroup"] = df["TotalSpend"] - group_mean_spend

    # Demographics
    df["IsInfant"] = (df["Age"] <= 4).astype(int)
    df["IsChild"] = ((df["Age"] > 4) & (df["Age"] <= 12)).astype(int)
    df["IsTeen"] = ((df["Age"] > 12) & (df["Age"] <= 17)).astype(int)
    df["IsSenior"] = (df["Age"] >= 60).astype(int)

    # Spatial & Route Cross-Features
    df["Route"] = df["HomePlanet"] + "_to_" + df["Destination"]
    df["Deck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]

    discrete_cols = [
        "CryoSleep",
        "CryoSleep_is_na",
        "VIP",
        "HasSpent",
        "IsSolo",
        "IsSoloFamily",
        "IsInfant",
        "IsChild",
        "IsTeen",
        "IsSenior",
        "CabinRegion",
        "CabinCluster50",
    ]

    categorical_cols = [
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "Route",
        "Deck_Side",
        "CabinDeck_Side",
        "CabinDeck_Side_Cluster",
    ] + discrete_cols

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes.astype(int)
        df[col] = df[col].clip(lower=0)

    drop_cols = ["Name", "FirstName", "LastName", "Cabin", "CabinNumRaw", "GroupId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Separate back into train and test
    train_processed = df.iloc[:n_train].copy()
    test_processed = df.iloc[n_train:].copy()

    train_processed["Transported"] = y_train.values
    test_processed = test_processed.drop(columns=["Transported"])

    # 5-fold Stratified Group K-Fold partitioned strictly by GroupId
    train_group_ids = train_df["PassengerId"].apply(lambda x: x.split("_")[0]).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_processed["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(
            train_processed, train_processed["Transported"], groups=train_group_ids
        )
    ):
        train_processed.loc[val_idx, "fold"] = fold_idx

    feature_cols = [
        c
        for c in train_processed.columns
        if c not in ["PassengerId", "Transported", "fold"]
    ]
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    cat_dtypes = {
        col: pd.CategoricalDtype(categories=sorted(df[col].unique()))
        for col in categorical_cols
    }

    os.makedirs("./working", exist_ok=True)
    metadata = {
        "id_col": "PassengerId",
        "target_col": "Transported",
        "fold_col": "fold",
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return train_processed, test_processed, categorical_cols, numerical_cols, cat_dtypes


# =========================================================================
# 3. TRAINING, EVALUATION & INFERENCE PIPELINE
# =========================================================================


def main():
    train_df, test_df, cat_cols, num_cols, cat_dtypes = process_features()

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["PassengerId", "Transported", "fold"]
    ]

    n_splits = 5
    oof_lgb = np.zeros(len(train_df), dtype=np.float64)
    oof_xgb = np.zeros(len(train_df), dtype=np.float64)
    oof_cat = np.zeros(len(train_df), dtype=np.float64)

    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
    test_preds_cat = np.zeros(len(test_df), dtype=np.float64)

    y_all = train_df["Transported"].values.astype(int)

    for fold in range(n_splits):
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train = train_df.loc[train_mask, feature_cols].copy()
        y_train = train_df.loc[train_mask, "Transported"].values.astype(int)

        X_val = train_df.loc[val_mask, feature_cols].copy()
        y_val = train_df.loc[val_mask, "Transported"].values.astype(int)

        X_test = test_df[feature_cols].copy()

        # 1. LightGBM
        lgb_model = get_lgb_model(cat_cols=cat_cols, cat_dtypes=cat_dtypes, seed=SEED + fold)
        lgb_model.fit(X_train, y_train, X_val, y_val)
        val_lgb = lgb_model.predict_proba(X_val)
        oof_lgb[val_mask] = val_lgb
        test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits

        # 2. XGBoost
        xgb_model = get_xgb_model(cat_cols=cat_cols, seed=SEED + fold)
        xgb_model.fit(X_train, y_train, X_val, y_val)
        val_xgb = xgb_model.predict_proba(X_val)
        oof_xgb[val_mask] = val_xgb
        test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

        # 3. CatBoost
        cat_model = get_catboost_model(cat_cols=cat_cols, seed=SEED + fold)
        cat_model.fit(X_train, y_train, X_val, y_val)
        val_cat = cat_model.predict_proba(X_val)
        oof_cat[val_mask] = val_cat
        test_preds_cat += cat_model.predict_proba(X_test) / n_splits

        fold_blend = (val_lgb + val_xgb + val_cat) / 3.0
        acc_fold_lgb = accuracy_score(y_val, (val_lgb >= 0.5).astype(int))
        acc_fold_xgb = accuracy_score(y_val, (val_xgb >= 0.5).astype(int))
        acc_fold_cat = accuracy_score(y_val, (val_cat >= 0.5).astype(int))
        acc_fold_blend = accuracy_score(y_val, (fold_blend >= 0.5).astype(int))

        print(
            f"Fold {fold + 1} | LGB Acc: {acc_fold_lgb:.4f} | XGB Acc: {acc_fold_xgb:.4f} | Cat Acc: {acc_fold_cat:.4f} | Fold Blend Acc: {acc_fold_blend:.4f}"
        )

    # Evaluate individual model out-of-fold accuracies
    acc_lgb = accuracy_score(y_all, (oof_lgb >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_all, (oof_xgb >= 0.5).astype(int))
    acc_cat = accuracy_score(y_all, (oof_cat >= 0.5).astype(int))
    print(f"Overall OOF -> LGB: {acc_lgb:.4f} | XGB: {acc_xgb:.4f} | Cat: {acc_cat:.4f}")

    # Optimize ensemble blending weights to maximize OOF classification accuracy
    def ensemble_objective(weights):
        w = np.array(weights)
        w = np.maximum(w, 0.0)
        s = np.sum(w)
        if s == 0:
            return 1.0
        w = w / s
        blended = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
        return -accuracy_score(y_all, (blended >= 0.5).astype(int))

    best_grid_acc = 0.0
    best_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    for w1 in np.linspace(0, 1, 51):
        for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) * 50)) + 1):
            w3 = max(0.0, 1.0 - w1 - w2)
            blend_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            acc = accuracy_score(y_all, (blend_oof >= 0.5).astype(int))
            if acc > best_grid_acc:
                best_grid_acc = acc
                best_weights = np.array([w1, w2, w3])

    opt_result = minimize(
        ensemble_objective,
        best_weights,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )

    opt_w = np.maximum(opt_result.x, 0.0)
    if np.sum(opt_w) == 0:
        opt_w = best_weights
    opt_w = opt_w / np.sum(opt_w)
    print(f"Optimized Ensemble Blend Weights: LGB={opt_w[0]:.4f}, XGB={opt_w[1]:.4f}, Cat={opt_w[2]:.4f}")

    oof_final = opt_w[0] * oof_lgb + opt_w[1] * oof_xgb + opt_w[2] * oof_cat
    final_oof_acc = accuracy_score(y_all, (oof_final >= 0.5).astype(int))

    # Blend test predictions
    test_blend = opt_w[0] * test_preds_lgb + opt_w[1] * test_preds_xgb + opt_w[2] * test_preds_cat
    test_bool_predictions = [bool(p) for p in (test_blend >= 0.5)]

    # Generate competition submission file
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_bool_predictions}
    )
    submission_out_path = "./submission/submission.csv"
    submission_df.to_csv(submission_out_path, index=False)

    # Final validation score output required by metric parser
    print(f"Final Validation Score: {final_oof_acc}")


if __name__ == "__main__":
    main()
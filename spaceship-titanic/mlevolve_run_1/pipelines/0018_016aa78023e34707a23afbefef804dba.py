import copy
import json
import math
import os
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
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
            num_leaves=24,
            max_depth=5,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.5,
            cat_l2=10.0,
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
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
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
            l2_leaf_reg=4.0,
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


class MetaClassifier:
    """Level-2 Logistic Regression meta-learner for disciplined probability stacking."""

    def __init__(self, C: float = 0.1, seed: int = SEED):
        self.C = C
        self.seed = seed
        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            random_state=seed,
            max_iter=1000,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> ProbArray:
        probs = self.model.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


def get_meta_model(C: float = 0.1, seed: int = SEED) -> MetaClassifier:
    """Returns a configured Level-2 MetaClassifier wrapper."""
    return MetaClassifier(C=C, seed=seed)


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
    cabin_parts = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]

    # Deterministic intra-group Cabin imputation within GroupId
    df["CabinDeck"] = df.groupby("GroupId")["CabinDeck"].transform(lambda s: s.ffill().bfill())
    df["CabinSide"] = df.groupby("GroupId")["CabinSide"].transform(lambda s: s.ffill().bfill())
    df["CabinNum"] = df.groupby("GroupId")["CabinNum"].transform(lambda s: s.ffill().bfill())

    # Fallback imputation for remaining missing cabin components
    df["CabinDeck"] = df["CabinDeck"].fillna("Missing")
    df["CabinSide"] = df["CabinSide"].fillna("Missing")
    deck_num_median = df.groupby("CabinDeck")["CabinNum"].transform("median").fillna(-1)
    df["CabinNum"] = df["CabinNum"].fillna(deck_num_median)

    # Continuous cabin deck percentile within deck
    df["CabinDeckNumPercentile"] = (
        df.groupby("CabinDeck")["CabinNum"].rank(pct=True).fillna(0.5)
    )

    # Deck-normalized cabin position
    deck_min = df.groupby("CabinDeck")["CabinNum"].transform("min")
    deck_max = df.groupby("CabinDeck")["CabinNum"].transform("max")
    deck_range = (deck_max - deck_min).replace(0, 1.0)
    df["DeckNormalizedCabinPos"] = ((df["CabinNum"] - deck_min) / deck_range).fillna(0.5)

    # Cabin corridor parity
    df["CabinNumParity"] = (df["CabinNum"].astype(int) % 2).abs()

    # Cabin occupancy frequency
    cabin_full_id = (
        df["CabinDeck"].astype(str)
        + "/"
        + df["CabinNum"].astype(str)
        + "/"
        + df["CabinSide"].astype(str)
    )
    df["CabinOccupancy"] = cabin_full_id.map(cabin_full_id.value_counts()).fillna(1).astype(int)

    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)
    df["CabinCluster50"] = (df["CabinNum"] // 50).fillna(-1).astype(int)
    df["CabinSector25"] = (df["CabinNum"] // 25).fillna(-1).astype(int)
    df["CabinDeck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]
    df["CabinDeck_Side_Cluster"] = (
        df["CabinDeck"]
        + "_"
        + df["CabinSide"]
        + "_"
        + df["CabinCluster50"].astype(str)
    )
    df["CabinDeck_Side_Bin100"] = (
        df["CabinDeck"]
        + "_"
        + df["CabinSide"]
        + "_"
        + (df["CabinNum"] // 100).astype(int).astype(str)
    )
    df["CabinDeck_Side_Sector25"] = (
        df["CabinDeck"]
        + "_"
        + df["CabinSide"]
        + "_"
        + df["CabinSector25"].astype(str)
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

    # Surname-based HomePlanet lookup across matching LastNames
    known_lastnames = df[df["HomePlanet"].notna() & (df["LastName"] != "Unknown")]
    lastname_planet = (
        known_lastnames.groupby("LastName")["HomePlanet"]
        .agg(lambda s: s.mode()[0] if not s.empty else np.nan)
    )
    df.loc[df["HomePlanet"].isna(), "HomePlanet"] = df.loc[
        df["HomePlanet"].isna(), "LastName"
    ].map(lastname_planet)

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

    # Intra-group Destination imputation within GroupId
    df["Destination"] = df.groupby("GroupId")["Destination"].transform(lambda s: s.ffill().bfill())
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

    # Surname-level family spend aggregations
    known_family_mask = (df["LastName"] != "Unknown") & (df["LastName"] != "")
    family_mean_spend = df[known_family_mask].groupby("LastName")["TotalSpend"].mean()
    df["FamilyMeanSpend"] = df["LastName"].map(family_mean_spend).fillna(df["TotalSpend"])
    df["FamilyMeanSpend_Log"] = np.log1p(df["FamilyMeanSpend"])
    df["SpendDiffFromFamily"] = df["TotalSpend"] - df["FamilyMeanSpend"]
    family_total_spend = df[known_family_mask].groupby("LastName")["TotalSpend"].sum()
    df["FamilyTotalSpend"] = df["LastName"].map(family_total_spend).fillna(df["TotalSpend"])
    df["FamilyTotalSpend_Log"] = np.log1p(df["FamilyTotalSpend"])

    # Planetary spending profiles and relative expenditure ratios
    df["MarsSpend"] = df["RoomService"] + df["ShoppingMall"]
    df["EuropaSpend"] = df["FoodCourt"] + df["Spa"] + df["VRDeck"]
    df["MarsSpendRatio"] = df["MarsSpend"] / (df["TotalSpend"] + 1.0)
    df["EuropaSpendRatio"] = df["EuropaSpend"] / (df["TotalSpend"] + 1.0)
    df["MarsSpend_Log"] = np.log1p(df["MarsSpend"])
    df["EuropaSpend_Log"] = np.log1p(df["EuropaSpend"])

    for amenity in amenities:
        df[f"{amenity}_Log"] = np.log1p(df[amenity])
        df[f"{amenity}_Ratio"] = df[amenity] / (df["TotalSpend"] + 1.0)
    df["TotalSpend_Log"] = np.log1p(df["TotalSpend"])
    df["NumAmenitiesUsed"] = (df[amenities] > 0).sum(axis=1)

    # Non-linear logarithmic luxury spend transforms and ratios
    df["LuxurySpend_Log"] = np.log1p(df["LuxurySpend"])
    df["AmenitiesSpend_Log"] = np.log1p(df["AmenitiesSpend"])
    df["LuxuryToAmenities_LogRatio"] = df["LuxurySpend_Log"] / (df["AmenitiesSpend_Log"] + 1.0)
    df["LuxuryToTotal_LogRatio"] = df["LuxurySpend_Log"] / (df["TotalSpend_Log"] + 1.0)
    df["AmenitiesToTotal_LogRatio"] = df["AmenitiesSpend_Log"] / (df["TotalSpend_Log"] + 1.0)

    # Group Expenditure Dynamics and Behavioral Aggregates
    group_total_spend = df.groupby("GroupId")["TotalSpend"].transform("sum")
    df["GroupTotalSpend"] = group_total_spend
    df["SpendToGroupRatio"] = df["TotalSpend"] / (df["GroupTotalSpend"] + 1.0)

    group_mean_spend = df.groupby("GroupId")["TotalSpend"].transform("mean")
    df["GroupMeanSpend"] = group_mean_spend
    df["SpendDiffFromGroup"] = df["TotalSpend"] - group_mean_spend

    cryo_known = df["CryoSleep"].replace(2, np.nan).astype(float)
    df["GroupCryoSleepRatio"] = cryo_known.groupby(df["GroupId"]).transform("mean").fillna(cryo_known.mean())

    is_zero_spend = (df["TotalSpend"] == 0).astype(float)
    df["GroupZeroSpendRate"] = is_zero_spend.groupby(df["GroupId"]).transform("mean")

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
        "CabinNumParity",
    ]

    base_categorical_cols = [
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "Route",
        "Deck_Side",
        "CabinDeck_Side",
        "CabinRegion",
        "CabinCluster50",
        "CabinSector25",
        "CabinDeck_Side_Cluster",
        "CabinDeck_Side_Bin100",
        "CabinDeck_Side_Sector25",
    ]

    categorical_cols = base_categorical_cols + discrete_cols

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

    categorical_cols = [c for c in categorical_cols if c in train_processed.columns]

    feature_cols = [
        c
        for c in train_processed.columns
        if c not in ["PassengerId", "Transported", "fold"]
    ]
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    cat_dtypes = {
        col: pd.CategoricalDtype(
            categories=sorted(pd.concat([train_processed[col], test_processed[col]]).unique())
        )
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


def optimize_ensemble(oof_preds: List[np.ndarray], y_true: np.ndarray):
    """Directly optimize ensemble blend weights and decision threshold via Nelder-Mead on classification accuracy."""
    n_models = len(oof_preds)

    def objective(params):
        weights = params[:n_models]
        th = params[n_models]
        w_sum = np.sum(np.abs(weights))
        if w_sum < 1e-7 or th < 0.25 or th > 0.75:
            return 0.0
        w = np.abs(weights) / w_sum
        blend = np.zeros_like(y_true, dtype=np.float64)
        for i in range(n_models):
            blend += w[i] * oof_preds[i]
        acc = accuracy_score(y_true, (blend >= th).astype(int))
        return -acc

    best_score = -1.0
    best_weights = [1.0 / n_models] * n_models
    best_th = 0.50

    init_points = [
        [1.0 / n_models] * n_models + [0.50],
        [0.20, 0.40, 0.40, 0.50],
        [0.15, 0.35, 0.50, 0.50],
        [0.25, 0.35, 0.40, 0.49],
        [0.10, 0.45, 0.45, 0.51],
        [0.30, 0.30, 0.40, 0.50],
        [0.15, 0.30, 0.55, 0.48],
    ]

    for init in init_points:
        res = minimize(
            objective,
            init,
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            w_raw = np.abs(res.x[:n_models])
            best_weights = (w_raw / np.sum(w_raw)).tolist()
            best_th = float(res.x[n_models])

    # Fine-grained 1D decision threshold scan to verify/polish threshold
    w = np.array(best_weights)
    blend = np.zeros_like(y_true, dtype=np.float64)
    for i in range(n_models):
        blend += w[i] * oof_preds[i]

    for th_candidate in np.arange(0.40, 0.60001, 0.002):
        acc = accuracy_score(y_true, (blend >= th_candidate).astype(int))
        if acc > best_score:
            best_score = acc
            best_th = float(th_candidate)

    return best_weights, best_th, best_score


def main():
    train_df, test_df, cat_cols, num_cols, cat_dtypes = process_features()

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["PassengerId", "Transported", "fold"]
    ]

    n_splits = 5
    y_all = train_df["Transported"].values.astype(int)

    # =========================================================================
    # PASS 1: 5-FOLD STRATIFIED GROUP CV BASELINE MODELS
    # =========================================================================
    print("--- Starting Pass 1: Supervised 5-Fold Training ---")
    oof_lgb_p1 = np.zeros(len(train_df), dtype=np.float64)
    oof_xgb_p1 = np.zeros(len(train_df), dtype=np.float64)
    oof_cat_p1 = np.zeros(len(train_df), dtype=np.float64)

    test_preds_lgb_p1 = np.zeros(len(test_df), dtype=np.float64)
    test_preds_xgb_p1 = np.zeros(len(test_df), dtype=np.float64)
    test_preds_cat_p1 = np.zeros(len(test_df), dtype=np.float64)

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
        oof_lgb_p1[val_mask] = val_lgb
        test_preds_lgb_p1 += lgb_model.predict_proba(X_test) / n_splits

        # 2. XGBoost
        xgb_model = get_xgb_model(cat_cols=cat_cols, seed=SEED + fold)
        xgb_model.fit(X_train, y_train, X_val, y_val)
        val_xgb = xgb_model.predict_proba(X_val)
        oof_xgb_p1[val_mask] = val_xgb
        test_preds_xgb_p1 += xgb_model.predict_proba(X_test) / n_splits

        # 3. CatBoost
        cat_model = get_catboost_model(cat_cols=cat_cols, seed=SEED + fold)
        cat_model.fit(X_train, y_train, X_val, y_val)
        val_cat = cat_model.predict_proba(X_val)
        oof_cat_p1[val_mask] = val_cat
        test_preds_cat_p1 += cat_model.predict_proba(X_test) / n_splits

        fold_blend = (val_lgb + val_xgb + val_cat) / 3.0
        acc_fold_lgb = accuracy_score(y_val, (val_lgb >= 0.5).astype(int))
        acc_fold_xgb = accuracy_score(y_val, (val_xgb >= 0.5).astype(int))
        acc_fold_cat = accuracy_score(y_val, (val_cat >= 0.5).astype(int))
        acc_fold_blend = accuracy_score(y_val, (fold_blend >= 0.5).astype(int))

        print(
            f"Pass 1 Fold {fold + 1} | LGB: {acc_fold_lgb:.4f} | XGB: {acc_fold_xgb:.4f} | Cat: {acc_fold_cat:.4f} | Blend: {acc_fold_blend:.4f}"
        )

    acc_lgb_p1 = accuracy_score(y_all, (oof_lgb_p1 >= 0.5).astype(int))
    acc_xgb_p1 = accuracy_score(y_all, (oof_xgb_p1 >= 0.5).astype(int))
    acc_cat_p1 = accuracy_score(y_all, (oof_cat_p1 >= 0.5).astype(int))
    print(
        f"Pass 1 Overall OOF -> LGB: {acc_lgb_p1:.4f} | XGB: {acc_xgb_p1:.4f} | Cat: {acc_cat_p1:.4f}"
    )

    p1_weights, p1_th, p1_acc = optimize_ensemble([oof_lgb_p1, oof_xgb_p1, oof_cat_p1], y_all)
    print(
        f"Pass 1 Nelder-Mead Blend -> Acc: {p1_acc:.4f} | Weights (LGB, XGB, Cat): {p1_weights} | Thresh: {p1_th:.3f}"
    )

    # Pass 1 blended test probabilities
    test_prob_p1 = (
        p1_weights[0] * test_preds_lgb_p1
        + p1_weights[1] * test_preds_xgb_p1
        + p1_weights[2] * test_preds_cat_p1
    )

    # =========================================================================
    # PASS 2: SEMI-SUPERVISED PSEUDO-LABELING SELF-TRAINING
    # =========================================================================
    high_conf_pos = test_prob_p1 >= 0.90
    high_conf_neg = test_prob_p1 <= 0.10
    pseudo_mask = high_conf_pos | high_conf_neg
    n_pseudo = int(np.sum(pseudo_mask))

    print(
        f"--- Pseudo-Labeling Filter: {n_pseudo} confident samples ({np.sum(high_conf_pos)} pos, {np.sum(high_conf_neg)} neg) out of {len(test_df)} ({n_pseudo / len(test_df) * 100:.1f}%) ---"
    )

    if n_pseudo > 0:
        pseudo_X = test_df.loc[pseudo_mask, feature_cols].copy()
        pseudo_y = np.where(high_conf_pos[pseudo_mask], 1, 0).astype(int)

        print("--- Starting Pass 2: Semi-Supervised Augmented 5-Fold Training ---")
        oof_lgb_p2 = np.zeros(len(train_df), dtype=np.float64)
        oof_xgb_p2 = np.zeros(len(train_df), dtype=np.float64)
        oof_cat_p2 = np.zeros(len(train_df), dtype=np.float64)

        test_preds_lgb_p2 = np.zeros(len(test_df), dtype=np.float64)
        test_preds_xgb_p2 = np.zeros(len(test_df), dtype=np.float64)
        test_preds_cat_p2 = np.zeros(len(test_df), dtype=np.float64)

        for fold in range(n_splits):
            train_mask = train_df["fold"] != fold
            val_mask = train_df["fold"] == fold

            X_tr_orig = train_df.loc[train_mask, feature_cols].copy()
            y_tr_orig = train_df.loc[train_mask, "Transported"].values.astype(int)

            # Hold-out validation is strictly from real training set (zero leakage)
            X_val = train_df.loc[val_mask, feature_cols].copy()
            y_val = train_df.loc[val_mask, "Transported"].values.astype(int)

            X_train_aug = pd.concat([X_tr_orig, pseudo_X], ignore_index=True)
            y_train_aug = np.concatenate([y_tr_orig, pseudo_y])

            X_test = test_df[feature_cols].copy()

            # 1. LightGBM
            lgb_p2 = get_lgb_model(cat_cols=cat_cols, cat_dtypes=cat_dtypes, seed=SEED + fold)
            lgb_p2.fit(X_train_aug, y_train_aug, X_val, y_val)
            val_lgb_p2 = lgb_p2.predict_proba(X_val)
            oof_lgb_p2[val_mask] = val_lgb_p2
            test_preds_lgb_p2 += lgb_p2.predict_proba(X_test) / n_splits

            # 2. XGBoost
            xgb_p2 = get_xgb_model(cat_cols=cat_cols, seed=SEED + fold)
            xgb_p2.fit(X_train_aug, y_train_aug, X_val, y_val)
            val_xgb_p2 = xgb_p2.predict_proba(X_val)
            oof_xgb_p2[val_mask] = val_xgb_p2
            test_preds_xgb_p2 += xgb_p2.predict_proba(X_test) / n_splits

            # 3. CatBoost
            cat_p2 = get_catboost_model(cat_cols=cat_cols, seed=SEED + fold)
            cat_p2.fit(X_train_aug, y_train_aug, X_val, y_val)
            val_cat_p2 = cat_p2.predict_proba(X_val)
            oof_cat_p2[val_mask] = val_cat_p2
            test_preds_cat_p2 += cat_p2.predict_proba(X_test) / n_splits

            fold_blend_p2 = (val_lgb_p2 + val_xgb_p2 + val_cat_p2) / 3.0
            acc_fold_lgb_p2 = accuracy_score(y_val, (val_lgb_p2 >= 0.5).astype(int))
            acc_fold_xgb_p2 = accuracy_score(y_val, (val_xgb_p2 >= 0.5).astype(int))
            acc_fold_cat_p2 = accuracy_score(y_val, (val_cat_p2 >= 0.5).astype(int))
            acc_fold_blend_p2 = accuracy_score(y_val, (fold_blend_p2 >= 0.5).astype(int))

            print(
                f"Pass 2 Fold {fold + 1} | LGB: {acc_fold_lgb_p2:.4f} | XGB: {acc_fold_xgb_p2:.4f} | Cat: {acc_fold_cat_p2:.4f} | Blend: {acc_fold_blend_p2:.4f}"
            )

        acc_lgb_p2 = accuracy_score(y_all, (oof_lgb_p2 >= 0.5).astype(int))
        acc_xgb_p2 = accuracy_score(y_all, (oof_xgb_p2 >= 0.5).astype(int))
        acc_cat_p2 = accuracy_score(y_all, (oof_cat_p2 >= 0.5).astype(int))
        print(
            f"Pass 2 Overall OOF -> LGB: {acc_lgb_p2:.4f} | XGB: {acc_xgb_p2:.4f} | Cat: {acc_cat_p2:.4f}"
        )

        p2_weights, p2_th, p2_acc = optimize_ensemble([oof_lgb_p2, oof_xgb_p2, oof_cat_p2], y_all)
        print(
            f"Pass 2 Nelder-Mead Blend -> Acc: {p2_acc:.4f} | Weights (LGB, XGB, Cat): {p2_weights} | Thresh: {p2_th:.3f}"
        )

        if p2_acc >= p1_acc:
            print(f"Selecting Pass 2 Ensemble (Pass 2 Acc: {p2_acc:.4f} >= Pass 1 Acc: {p1_acc:.4f})")
            final_weights = p2_weights
            final_th = p2_th
            final_score = p2_acc
            final_test_probs = (
                final_weights[0] * test_preds_lgb_p2
                + final_weights[1] * test_preds_xgb_p2
                + final_weights[2] * test_preds_cat_p2
            )
        else:
            print(f"Selecting Pass 1 Ensemble (Pass 1 Acc: {p1_acc:.4f} > Pass 2 Acc: {p2_acc:.4f})")
            final_weights = p1_weights
            final_th = p1_th
            final_score = p1_acc
            final_test_probs = (
                final_weights[0] * test_preds_lgb_p1
                + final_weights[1] * test_preds_xgb_p1
                + final_weights[2] * test_preds_cat_p1
            )
    else:
        final_weights = p1_weights
        final_th = p1_th
        final_score = p1_acc
        final_test_probs = test_prob_p1

    test_bool_predictions = [bool(p) for p in (final_test_probs >= final_th)]

    # Generate competition submission file
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_bool_predictions}
    )
    submission_out_path = "./submission/submission.csv"
    submission_df.to_csv(submission_out_path, index=False)

    # Final validation score output required by metric parser
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
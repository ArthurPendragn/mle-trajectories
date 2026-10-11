import copy
import json
import math
import os
import re
from typing import List, Optional

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

# Set deterministic random seeds
SEED = 42
np.random.seed(SEED)

pd.set_option("future.no_silent_downcasting", True)


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

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
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
            sample_weight=sample_weight,
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

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        X_tr = X_train.copy()
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.copy()
            eval_set = [(X_v, y_val)]
        self.model.fit(
            X_tr,
            y_train,
            sample_weight=sample_weight,
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

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
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
            sample_weight=sample_weight,
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


class HistGBMWrapper:
    """HistGradientBoosting Classifier wrapper with native categorical handling."""

    def __init__(self, cat_cols: Optional[List[str]] = None, seed: int = SEED):
        self.cat_cols = cat_cols or []
        self.seed = seed
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        X_tr = X_train.copy()
        for c in self.cat_cols:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype(int)
        cat_indices = [i for i, c in enumerate(X_tr.columns) if c in self.cat_cols]
        cat_features_arg = cat_indices if len(cat_indices) > 0 else None

        self.model = HistGradientBoostingClassifier(
            categorical_features=cat_features_arg,
            max_iter=1000,
            learning_rate=0.03,
            max_leaf_nodes=31,
            max_depth=6,
            l2_regularization=1.5,
            early_stopping=True,
            random_state=self.seed,
        )
        self.model.fit(X_tr, y_train, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> ProbArray:
        X_df = X.copy()
        for c in self.cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype(int)
        probs = self.model.predict_proba(X_df)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


def get_histgbm_model(cat_cols: Optional[List[str]] = None, seed: int = SEED) -> HistGBMWrapper:
    """Returns a configured HistGradientBoosting model wrapper."""
    return HistGBMWrapper(cat_cols=cat_cols, seed=seed)


class GlobalSimplexEnsembleBlender:
    """Exact 4-model global simplex ensemble blender optimizing barycentric weights and decision threshold."""

    def __init__(
        self,
        step: float = 0.05,
        th_min: float = 0.42,
        th_max: float = 0.58,
        th_step: float = 0.002,
    ):
        self.step = step
        self.th_min = th_min
        self.th_max = th_max
        self.th_step = th_step
        self.best_weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.best_threshold: float = 0.50
        self.best_score: float = 0.0

    def fit(self, oof_preds: np.ndarray, y_true: np.ndarray):
        oof_preds = np.asarray(oof_preds, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=int)
        n_samples = len(y_true)

        n_steps = int(round(1.0 / self.step))
        quadruples = []
        for i in range(n_steps + 1):
            for j in range(n_steps + 1 - i):
                for k in range(n_steps + 1 - i - j):
                    l = n_steps - i - j - k
                    quadruples.append((i * self.step, j * self.step, k * self.step, l * self.step))

        coarse_thresholds = np.arange(self.th_min, self.th_max + 1e-6, 0.005)

        best_score = -1.0
        best_w = (0.25, 0.25, 0.25, 0.25)
        best_th = 0.50

        p0 = oof_preds[:, 0]
        p1 = oof_preds[:, 1]
        p2 = oof_preds[:, 2]
        p3 = oof_preds[:, 3]

        for w0, w1, w2, w3 in quadruples:
            blended = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
            for th in coarse_thresholds:
                score = np.count_nonzero((blended >= th) == y_true) / n_samples
                if score > best_score:
                    best_score = score
                    best_w = (w0, w1, w2, w3)
                    best_th = th

        fine_thresholds = np.arange(self.th_min, self.th_max + 1e-6, self.th_step)
        delta_range = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
        fine_quadruples = []
        for d0 in delta_range:
            for d1 in delta_range:
                for d2 in delta_range:
                    d3 = -(d0 + d1 + d2)
                    if abs(d3) <= 0.045:
                        w0 = best_w[0] + d0
                        w1 = best_w[1] + d1
                        w2 = best_w[2] + d2
                        w3 = best_w[3] + d3
                        if w0 >= -1e-7 and w1 >= -1e-7 and w2 >= -1e-7 and w3 >= -1e-7:
                            fine_quadruples.append((max(0.0, w0), max(0.0, w1), max(0.0, w2), max(0.0, w3)))

        for w0, w1, w2, w3 in fine_quadruples:
            s = w0 + w1 + w2 + w3
            if s <= 0:
                continue
            w0, w1, w2, w3 = w0 / s, w1 / s, w2 / s, w3 / s
            blended = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
            for th in fine_thresholds:
                score = np.count_nonzero((blended >= th) == y_true) / n_samples
                if score > best_score:
                    best_score = score
                    best_w = (w0, w1, w2, w3)
                    best_th = th

        self.best_weights = np.array(best_w, dtype=np.float64)
        self.best_threshold = float(best_th)
        self.best_score = float(best_score)
        return self

    def predict_proba(self, test_preds: np.ndarray) -> ProbArray:
        test_preds = np.asarray(test_preds, dtype=np.float64)
        probs = (
            test_preds[:, 0] * self.best_weights[0]
            + test_preds[:, 1] * self.best_weights[1]
            + test_preds[:, 2] * self.best_weights[2]
            + test_preds[:, 3] * self.best_weights[3]
        )
        return np.asarray(probs, dtype=np.float64).view(ProbArray)

    def predict(self, test_preds: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(test_preds)
        return (probs >= self.best_threshold).astype(int)


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

    # Cabin manifest occupancy frequency
    cabin_occupancy = df["Cabin"].value_counts()
    df["CabinOccupancy"] = df["Cabin"].map(cabin_occupancy).fillna(0).astype(int)

    # Cabin topology decomposition
    cabin_parts = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]

    # CabinNum missingness indicator prior to imputation
    df["CabinNum_is_na"] = df["CabinNum"].isna().astype(int)

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

    # 3D physical vessel spatial features: ordinal DeckLevel, interaction tokens, and sector bins
    deck_level_map = {"T": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "Missing": 8}
    df["DeckLevel"] = df["CabinDeck"].map(deck_level_map).fillna(8).astype(int)
    df["CabinDeck_Side"] = df["CabinDeck"].astype(str) + "_" + df["CabinSide"].astype(str)
    df["CabinSector50"] = (np.maximum(df["CabinNum"].values, 0) // 50).astype(int)

    # Domain-informed spatial hazard feature: Starboard side indicator
    df["IsStarboard"] = (df["CabinSide"] == "S").astype(int)

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

    # High-impact spatial hazard & consciousness interaction
    df["CryoSleep_x_Starboard"] = df["CryoSleep"] * 2 + df["IsStarboard"]

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
    df["LuxuryToSpendRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["AmenitiesRatio"] = df["AmenitiesSpend"] / (df["TotalSpend"] + 1.0)
    df["LuxuryToEssentialRatio"] = df["LuxurySpend"] / (df["AmenitiesSpend"] + 1.0)

    # Spend-to-Age interaction ratios
    df["TotalSpendToAge"] = df["TotalSpend"] / (df["Age"] + 1.0)
    df["LuxurySpendToAge"] = df["LuxurySpend"] / (df["Age"] + 1.0)
    df["AmenitiesSpendToAge"] = df["AmenitiesSpend"] / (df["Age"] + 1.0)

    for amenity in amenities:
        df[f"{amenity}_Log"] = np.log1p(df[amenity])
        df[f"{amenity}_Ratio"] = df[amenity] / (df["TotalSpend"] + 1.0)
    df["TotalSpend_Log"] = np.log1p(df["TotalSpend"])
    df["NumAmenitiesUsed"] = (df[amenities] > 0).sum(axis=1)

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

    # Route cross-feature
    df["Route"] = df["HomePlanet"] + "_to_" + df["Destination"]

    # Native integer categorical encoding for nominal variables (preserving native multi-split power)
    nominal_cols = ["HomePlanet", "Destination", "CabinDeck", "CabinSide", "CabinDeck_Side", "Route"]
    for col in nominal_cols:
        df[col] = df[col].astype(str)
        sorted_unique = sorted(df[col].unique())
        val_map = {val: idx for idx, val in enumerate(sorted_unique)}
        df[col] = df[col].map(val_map).astype(int)

    categorical_cols = [
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "CabinDeck_Side",
        "Route",
        "DeckLevel",
        "CabinSector50",
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
        "IsStarboard",
        "CryoSleep_x_Starboard",
        "CabinNum_is_na",
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(int)

    drop_cols = ["Name", "FirstName", "LastName", "Cabin", "CabinNumRaw", "GroupId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Separate back into train and test
    train_processed = df.iloc[:n_train].copy().reset_index(drop=True)
    test_processed = df.iloc[n_train:].copy().reset_index(drop=True)

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

    for c in numerical_cols:
        train_processed[c] = train_processed[c].fillna(0.0)
        test_processed[c] = test_processed[c].fillna(0.0)

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
    n_train = len(train_df)
    n_test = len(test_df)
    n_models = 4

    y_all = train_df["Transported"].values.astype(int)

    print("=== 5-Fold Heterogeneous 4-Model Committee Training ===")
    oof_preds = np.zeros((n_train, n_models), dtype=np.float64)
    test_preds = np.zeros((n_test, n_models), dtype=np.float64)

    for fold in range(n_splits):
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train = train_df.loc[train_mask, feature_cols].copy()
        y_train = train_df.loc[train_mask, "Transported"].values.astype(int)

        X_val = train_df.loc[val_mask, feature_cols].copy()
        y_val = train_df.loc[val_mask, "Transported"].values.astype(int)

        X_test = test_df[feature_cols].copy()

        # 1. LightGBM (Leaf-wise histogram)
        lgb_model = get_lgb_model(cat_cols=cat_cols, cat_dtypes=cat_dtypes, seed=SEED + fold)
        lgb_model.fit(X_train, y_train, X_val, y_val)
        val_lgb = lgb_model.predict_proba(X_val)
        oof_preds[val_mask, 0] = val_lgb
        test_preds[:, 0] += lgb_model.predict_proba(X_test) / n_splits

        # 2. XGBoost (Depth-wise regularized)
        xgb_model = get_xgb_model(cat_cols=cat_cols, seed=SEED + fold)
        xgb_model.fit(X_train, y_train, X_val, y_val)
        val_xgb = xgb_model.predict_proba(X_val)
        oof_preds[val_mask, 1] = val_xgb
        test_preds[:, 1] += xgb_model.predict_proba(X_test) / n_splits

        # 3. CatBoost (Symmetric oblivious trees)
        cat_model = get_catboost_model(cat_cols=cat_cols, seed=SEED + fold)
        cat_model.fit(X_train, y_train, X_val, y_val)
        val_cat = cat_model.predict_proba(X_val)
        oof_preds[val_mask, 2] = val_cat
        test_preds[:, 2] += cat_model.predict_proba(X_test) / n_splits

        # 4. HistGradientBoosting (Orthogonal independent binning)
        hist_model = get_histgbm_model(cat_cols=cat_cols, seed=SEED + fold)
        hist_model.fit(X_train, y_train, X_val, y_val)
        val_hist = hist_model.predict_proba(X_val)
        oof_preds[val_mask, 3] = val_hist
        test_preds[:, 3] += hist_model.predict_proba(X_test) / n_splits

        acc_fold_lgb = accuracy_score(y_val, (val_lgb >= 0.5).astype(int))
        acc_fold_xgb = accuracy_score(y_val, (val_xgb >= 0.5).astype(int))
        acc_fold_cat = accuracy_score(y_val, (val_cat >= 0.5).astype(int))
        acc_fold_hist = accuracy_score(y_val, (val_hist >= 0.5).astype(int))
        acc_fold_blend = accuracy_score(
            y_val, ((val_lgb + val_xgb + val_cat + val_hist) / 4.0 >= 0.5).astype(int)
        )

        print(
            f"Fold {fold + 1} | LGB: {acc_fold_lgb:.4f} | XGB: {acc_fold_xgb:.4f} | Cat: {acc_fold_cat:.4f} | Hist: {acc_fold_hist:.4f} | Blend: {acc_fold_blend:.4f}"
        )

    # Evaluate individual base model out-of-fold accuracies
    acc_lgb = accuracy_score(y_all, (oof_preds[:, 0] >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_all, (oof_preds[:, 1] >= 0.5).astype(int))
    acc_cat = accuracy_score(y_all, (oof_preds[:, 2] >= 0.5).astype(int))
    acc_hist = accuracy_score(y_all, (oof_preds[:, 3] >= 0.5).astype(int))
    print(
        f"Overall OOF -> LGB: {acc_lgb:.4f} | XGB: {acc_xgb:.4f} | Cat: {acc_cat:.4f} | Hist: {acc_hist:.4f}"
    )

    # Fit 4-model Global Simplex Ensemble Blender on pure OOF predictions
    blender = GlobalSimplexEnsembleBlender()
    blender.fit(oof_preds, y_all)
    final_oof_acc = blender.best_score

    print(
        f"Optimal Decision Boundary -> Weights: LGB={blender.best_weights[0]:.2f}, "
        f"XGB={blender.best_weights[1]:.2f}, Cat={blender.best_weights[2]:.2f}, "
        f"Hist={blender.best_weights[3]:.2f} | "
        f"Threshold: {blender.best_threshold:.3f} | OOF Acc: {final_oof_acc:.4f}"
    )

    # Predict test probabilities with blender and apply optimal threshold
    test_final_preds = blender.predict(test_preds)
    test_bool_predictions = [bool(p == 1) for p in test_final_preds]

    # Generate competition submission file
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"].values, "Transported": test_bool_predictions}
    )
    submission_out_path = "./submission/submission.csv"
    submission_df.to_csv(submission_out_path, index=False)

    # Final validation score output required by metric parser
    print(f"Final Validation Score: {final_oof_acc}")


if __name__ == "__main__":
    main()
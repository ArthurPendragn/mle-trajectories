import copy
import json
import math
import os
from typing import List, Optional

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set deterministic random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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


class TabularNN(nn.Module):
    """3-layer MLP with entity embeddings and continuous batch normalization."""

    def __init__(
        self,
        cat_dims: dict,
        cat_cols: List[str],
        num_cols: List[str],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for col in cat_cols:
            c_dim = cat_dims.get(col, 100) + 2
            emb_dim = min(32, max(4, int(math.ceil(math.sqrt(c_dim))) + 1))
            self.embeddings.append(nn.Embedding(c_dim, emb_dim))
            total_emb_dim += emb_dim

        self.num_cont = len(num_cols)
        if self.num_cont > 0:
            self.bn_cont = nn.BatchNorm1d(self.num_cont)
        else:
            self.bn_cont = nn.Identity()

        in_dim = total_emb_dim + self.num_cont

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0:
            emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_emb = torch.cat(emb_list, dim=1)
        else:
            x_emb = None

        if self.num_cont > 0:
            x_norm = self.bn_cont(x_cont)
            if x_emb is not None:
                x = torch.cat([x_emb, x_norm], dim=1)
            else:
                x = x_norm
        else:
            x = x_emb

        logits = self.mlp(x).squeeze(-1)
        return logits


class TabularNNWrapper:
    """PyTorch Tabular Neural Network combining entity embeddings and continuous MLP."""

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        cat_dims: Optional[dict] = None,
        seed: int = SEED,
        epochs: int = 25,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.cat_dims = cat_dims or {}
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TabularNN] = None
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.num_cols:
            X_tr_cont = self.scaler.fit_transform(X_train[self.num_cols].fillna(0.0).values)
            X_tr_cont = np.nan_to_num(X_tr_cont, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_tr_cont = np.empty((len(X_train), 0))

        if self.cat_cols:
            X_tr_cat = X_train[self.cat_cols].fillna(0).values.astype(np.int64)
            X_tr_cat = np.clip(X_tr_cat, 0, None)
        else:
            X_tr_cat = np.empty((len(X_train), 0), dtype=np.int64)

        y_tr = y_train.astype(np.float32)

        self.model = TabularNN(
            cat_dims=self.cat_dims,
            cat_cols=self.cat_cols,
            num_cols=self.num_cols,
        ).to(self.device)

        train_ds = TensorDataset(
            torch.tensor(X_tr_cat, dtype=torch.long),
            torch.tensor(X_tr_cont, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            if self.num_cols:
                X_val_cont = self.scaler.transform(X_val[self.num_cols].fillna(0.0).values)
                X_val_cont = np.nan_to_num(X_val_cont, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X_val_cont = np.empty((len(X_val), 0))

            if self.cat_cols:
                X_val_cat = X_val[self.cat_cols].fillna(0).values.astype(np.int64)
                X_val_cat = np.clip(X_val_cat, 0, None)
            else:
                X_val_cat = np.empty((len(X_val), 0), dtype=np.int64)

            val_cat_t = torch.tensor(X_val_cat, dtype=torch.long).to(self.device)
            val_cont_t = torch.tensor(X_val_cont, dtype=torch.float32).to(self.device)
            val_y_t = torch.tensor(y_val.astype(np.float32), dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            self.model.train()
            for b_cat, b_cont, b_y in train_loader:
                b_cat = b_cat.to(self.device)
                b_cont = b_cont.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(b_cat, b_cont)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_cat_t, val_cont_t)
                    val_loss = criterion(val_logits, val_y_t).item()
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

        if has_val:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: pd.DataFrame) -> ProbArray:
        self.model.eval()
        if self.num_cols:
            X_cont = self.scaler.transform(X[self.num_cols].fillna(0.0).values)
            X_cont = np.nan_to_num(X_cont, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_cont = np.empty((len(X), 0))

        if self.cat_cols:
            X_cat = X[self.cat_cols].fillna(0).values.astype(np.int64)
            X_cat = np.clip(X_cat, 0, None)
        else:
            X_cat = np.empty((len(X), 0), dtype=np.int64)

        ds = TensorDataset(
            torch.tensor(X_cat, dtype=torch.long),
            torch.tensor(X_cont, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        preds = []
        with torch.no_grad():
            for b_cat, b_cont in loader:
                b_cat = b_cat.to(self.device)
                b_cont = b_cont.to(self.device)
                logits = self.model(b_cat, b_cont)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)

        all_probs = np.concatenate(preds, axis=0) if preds else np.zeros(len(X))
        return np.asarray(all_probs, dtype=np.float64).view(ProbArray)


class StackingMetaLearner:
    """Level-2 Stacking Meta-Learner via regularized Logistic Regression."""

    def __init__(self, C: float = 0.1, solver: str = "lbfgs", seed: int = SEED):
        self.C = C
        self.solver = solver
        self.seed = seed
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            random_state=self.seed,
            max_iter=1000,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> ProbArray:
        probs = self.model.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64).view(ProbArray)


def get_tabular_nn_model(
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    cat_dims: Optional[dict] = None,
    seed: int = SEED,
) -> TabularNNWrapper:
    """Returns a configured Tabular Neural Network wrapper."""
    return TabularNNWrapper(
        cat_cols=cat_cols, num_cols=num_cols, cat_dims=cat_dims, seed=seed
    )


def get_stacking_meta_learner(
    C: float = 0.1, solver: str = "lbfgs", seed: int = SEED
) -> StackingMetaLearner:
    """Returns a configured Stacking Meta-Learner."""
    return StackingMetaLearner(C=C, solver=solver, seed=seed)


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

    # High-resolution spatial corridor bins & 25-cabin corridor clusters
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
        "CabinSector25",
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
        "CabinDeck_Side_Bin100",
        "CabinDeck_Side_Sector25",
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

    # Level-1 Out-Of-Fold and Test meta-feature matrices
    oof_meta = np.zeros((n_train, n_models), dtype=np.float64)
    test_meta = np.zeros((n_test, n_models), dtype=np.float64)

    y_all = train_df["Transported"].values.astype(int)

    # Derive categorical dimensions for embedding initialization
    cat_dims = {col: int(train_df[col].max() + 1) for col in cat_cols}
    for col in cat_cols:
        cat_dims[col] = max(cat_dims[col], int(test_df[col].max() + 1))

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
        oof_meta[val_mask, 0] = val_lgb
        test_meta[:, 0] += lgb_model.predict_proba(X_test) / n_splits

        # 2. XGBoost
        xgb_model = get_xgb_model(cat_cols=cat_cols, seed=SEED + fold)
        xgb_model.fit(X_train, y_train, X_val, y_val)
        val_xgb = xgb_model.predict_proba(X_val)
        oof_meta[val_mask, 1] = val_xgb
        test_meta[:, 1] += xgb_model.predict_proba(X_test) / n_splits

        # 3. CatBoost
        cat_model = get_catboost_model(cat_cols=cat_cols, seed=SEED + fold)
        cat_model.fit(X_train, y_train, X_val, y_val)
        val_cat = cat_model.predict_proba(X_val)
        oof_meta[val_mask, 2] = val_cat
        test_meta[:, 2] += cat_model.predict_proba(X_test) / n_splits

        # 4. Tabular Neural Network
        nn_model = get_tabular_nn_model(
            cat_cols=cat_cols, num_cols=num_cols, cat_dims=cat_dims, seed=SEED + fold
        )
        nn_model.fit(X_train, y_train, X_val, y_val)
        val_nn = nn_model.predict_proba(X_val)
        oof_meta[val_mask, 3] = val_nn
        test_meta[:, 3] += nn_model.predict_proba(X_test) / n_splits

        fold_blend = (val_lgb + val_xgb + val_cat + val_nn) / 4.0
        acc_fold_lgb = accuracy_score(y_val, (val_lgb >= 0.5).astype(int))
        acc_fold_xgb = accuracy_score(y_val, (val_xgb >= 0.5).astype(int))
        acc_fold_cat = accuracy_score(y_val, (val_cat >= 0.5).astype(int))
        acc_fold_nn = accuracy_score(y_val, (val_nn >= 0.5).astype(int))
        acc_fold_blend = accuracy_score(y_val, (fold_blend >= 0.5).astype(int))

        print(
            f"Fold {fold + 1} | LGB: {acc_fold_lgb:.4f} | XGB: {acc_fold_xgb:.4f} | Cat: {acc_fold_cat:.4f} | NN: {acc_fold_nn:.4f} | Fold Blend: {acc_fold_blend:.4f}"
        )

    # Evaluate individual model out-of-fold accuracies
    acc_lgb = accuracy_score(y_all, (oof_meta[:, 0] >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_all, (oof_meta[:, 1] >= 0.5).astype(int))
    acc_cat = accuracy_score(y_all, (oof_meta[:, 2] >= 0.5).astype(int))
    acc_nn = accuracy_score(y_all, (oof_meta[:, 3] >= 0.5).astype(int))
    print(
        f"Overall OOF -> LGB: {acc_lgb:.4f} | XGB: {acc_xgb:.4f} | Cat: {acc_cat:.4f} | NN: {acc_nn:.4f}"
    )

    # Level-2 Stacking Meta-Learner training on OOF meta-features
    meta_learner = get_stacking_meta_learner(C=0.1, solver="lbfgs", seed=SEED)
    meta_learner.fit(oof_meta, y_all)
    print(
        f"Stacking Meta-Learner Weights: LGB={meta_learner.model.coef_[0][0]:.4f}, "
        f"XGB={meta_learner.model.coef_[0][1]:.4f}, Cat={meta_learner.model.coef_[0][2]:.4f}, "
        f"NN={meta_learner.model.coef_[0][3]:.4f} | Intercept={meta_learner.model.intercept_[0]:.4f}"
    )

    oof_final = meta_learner.predict_proba(oof_meta)

    # 1D post-processing decision threshold scan over [0.40, 0.60] interval with step 0.005
    threshold_candidates = np.arange(0.40, 0.60001, 0.005)
    best_threshold = 0.50
    best_oof_acc = 0.0
    for th in threshold_candidates:
        acc = accuracy_score(y_all, (oof_final >= th).astype(int))
        if acc > best_oof_acc:
            best_oof_acc = acc
            best_threshold = th

    print(f"Optimal Decision Threshold: {best_threshold:.3f} (OOF Acc: {best_oof_acc:.4f})")
    final_oof_acc = best_oof_acc

    # Predict test probabilities with Level-2 meta-learner and apply optimal threshold
    test_final = meta_learner.predict_proba(test_meta)
    test_bool_predictions = [bool(p) for p in (test_final >= best_threshold)]

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
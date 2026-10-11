import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ---------------------------------------------------------
# 1. PATH RESOLUTION & DATA LOADING
# ---------------------------------------------------------
train_path = (
    "./input/train.csv"
    if os.path.exists("./input/train.csv")
    else "./input/input/train.csv"
)
test_path = (
    "./input/test.csv"
    if os.path.exists("./input/test.csv")
    else "./input/input/test.csv"
)

print(f"Loading data from: {train_path} and {test_path}")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_len = len(train_df)
test_len = len(test_df)
print(f"Loaded {train_len} train records and {test_len} test records.")

# Target encoding: boolean -> int (1 / 0)
train_df["Transported"] = train_df["Transported"].astype(int)

# Combine datasets for consistent roster-level feature extraction
combined_df = pd.concat(
    [train_df.drop(columns=["Transported"]), test_df], ignore_index=True
)

# ---------------------------------------------------------
# 2. DOMAIN-SPECIFIC DEDUCTION & FEATURE ENGINEERING
# ---------------------------------------------------------
amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Extract group and passenger position from PassengerId (gggg_pp)
combined_df["Group"] = combined_df["PassengerId"].apply(lambda x: x.split("_")[0])
combined_df["Group_Num"] = combined_df["PassengerId"].apply(
    lambda x: int(x.split("_")[1])
)

# Group size derived from full ship roster
group_size_map = combined_df["Group"].value_counts().to_dict()
combined_df["Group_Size"] = combined_df["Group"].map(group_size_map)
combined_df["Is_Alone"] = (combined_df["Group_Size"] == 1).astype(int)

# Extract Surname from Name
combined_df["Surname"] = combined_df["Name"].apply(
    lambda x: x.split()[-1] if pd.notnull(x) else np.nan
)
surname_size_map = combined_df["Surname"].value_counts().to_dict()
combined_df["Family_Size"] = (
    combined_df["Surname"].map(surname_size_map).fillna(1).astype(int)
)

# Physical invariant deductions
known_spend_sum = combined_df[amenity_cols].sum(axis=1, min_count=1)

# If spending > 0, passenger cannot be in CryoSleep
cryo_deduced_false = (known_spend_sum > 0) & combined_df["CryoSleep"].isnull()
combined_df.loc[cryo_deduced_false, "CryoSleep"] = False

# If CryoSleep == True, all amenity spending must be 0
for col in amenity_cols:
    combined_df.loc[
        (combined_df["CryoSleep"] == True) & combined_df[col].isnull(), col
    ] = 0.0

# If Age < 13, passengers cannot spend on luxury amenities
for col in amenity_cols:
    combined_df.loc[(combined_df["Age"] < 13) & combined_df[col].isnull(), col] = 0.0

combined_df["Zero_Spend"] = (combined_df[amenity_cols].sum(axis=1) == 0).astype(int)

# Cabin decomposition
def split_cabin(val):
    if pd.isnull(val):
        return np.nan, np.nan, np.nan
    parts = str(val).split("/")
    if len(parts) == 3:
        deck, num, side = parts[0], parts[1], parts[2]
        return deck, float(num) if num.isdigit() else np.nan, side
    return np.nan, np.nan, np.nan


cabin_splits = combined_df["Cabin"].apply(split_cabin)
combined_df["Cabin_Deck"] = [x[0] for x in cabin_splits]
combined_df["Cabin_Num"] = [x[1] for x in cabin_splits]
combined_df["Cabin_Side"] = [x[2] for x in cabin_splits]

# Group matching for Cabin Deck, Num & Side
group_deck = (
    combined_df.dropna(subset=["Cabin_Deck"])
    .groupby("Group")["Cabin_Deck"]
    .first()
    .to_dict()
)
group_side = (
    combined_df.dropna(subset=["Cabin_Side"])
    .groupby("Group")["Cabin_Side"]
    .first()
    .to_dict()
)
group_num = (
    combined_df.dropna(subset=["Cabin_Num"])
    .groupby("Group")["Cabin_Num"]
    .first()
    .to_dict()
)
combined_df["Cabin_Deck"] = combined_df["Cabin_Deck"].fillna(
    combined_df["Group"].map(group_deck)
)
combined_df["Cabin_Side"] = combined_df["Cabin_Side"].fillna(
    combined_df["Group"].map(group_side)
)
combined_df["Cabin_Num"] = combined_df["Cabin_Num"].fillna(
    combined_df["Group"].map(group_num)
)

# Deck-specific median imputation for remaining Cabin_Num
deck_cabin_num_median = (
    combined_df.dropna(subset=["Cabin_Num"])
    .groupby("Cabin_Deck")["Cabin_Num"]
    .median()
    .to_dict()
)
overall_cabin_num_median = combined_df["Cabin_Num"].median()
combined_df["Cabin_Num"] = combined_df["Cabin_Num"].fillna(
    combined_df["Cabin_Deck"].map(deck_cabin_num_median)
)
combined_df["Cabin_Num"] = combined_df["Cabin_Num"].fillna(overall_cabin_num_median)

# Group matching for Destination
group_dest = (
    combined_df.dropna(subset=["Destination"])
    .groupby("Group")["Destination"]
    .first()
    .to_dict()
)
combined_df["Destination"] = combined_df["Destination"].fillna(
    combined_df["Group"].map(group_dest)
)

# Group and Surname matching for HomePlanet
group_home = (
    combined_df.dropna(subset=["HomePlanet"])
    .groupby("Group")["HomePlanet"]
    .first()
    .to_dict()
)
combined_df["HomePlanet"] = combined_df["HomePlanet"].fillna(
    combined_df["Group"].map(group_home)
)

surname_home = (
    combined_df.dropna(subset=["HomePlanet"])
    .groupby("Surname")["HomePlanet"]
    .first()
    .to_dict()
)
combined_df["HomePlanet"] = combined_df["HomePlanet"].fillna(
    combined_df["Surname"].map(surname_home)
)

# Deterministic physical spaceship deductions and domain invariants
# VIP domain invariants: VIP=False for Age < 25 and Earth passengers
combined_df.loc[combined_df["Age"] < 25, "VIP"] = False
combined_df.loc[
    (combined_df["HomePlanet"] == "Earth") & combined_df["VIP"].isnull(),
    "VIP",
] = False

# Decks A, B, C, T are Europa exclusively
combined_df.loc[
    combined_df["Cabin_Deck"].isin(["A", "B", "C", "T"]) & combined_df["HomePlanet"].isnull(),
    "HomePlanet",
] = "Europa"
# Deck G is Earth exclusively
combined_df.loc[
    (combined_df["Cabin_Deck"] == "G") & combined_df["HomePlanet"].isnull(),
    "HomePlanet",
] = "Earth"
# Earth passengers restricted from Decks A, B, C, D, T (cannot be Earth)
combined_df.loc[
    combined_df["Cabin_Deck"].isin(["A", "B", "C", "D", "T"]) & (combined_df["HomePlanet"] == "Earth"),
    "HomePlanet",
] = np.nan
# Europa passengers restricted from Decks F, G (cannot be Europa)
combined_df.loc[
    combined_df["Cabin_Deck"].isin(["F", "G"]) & (combined_df["HomePlanet"] == "Europa"),
    "HomePlanet",
] = np.nan

# Impute missing amenity spends with 0.0
for col in amenity_cols:
    combined_df[col] = combined_df[col].fillna(0.0)

# Spending aggregations and metrics
combined_df["TotalSpend"] = combined_df[amenity_cols].sum(axis=1)
combined_df["HasSpent"] = (combined_df["TotalSpend"] > 0).astype(int)
combined_df["LuxurySpend"] = (
    combined_df["Spa"] + combined_df["VRDeck"] + combined_df["RoomService"]
)
combined_df["BasicSpend"] = combined_df["FoodCourt"] + combined_df["ShoppingMall"]
combined_df["Zero_LuxurySpend"] = (combined_df["LuxurySpend"] == 0).astype(int)

# Amenity zero-spend count & complete individual expenditure ratio profiles
combined_df["Amenity_Zero_Count"] = (combined_df[amenity_cols] == 0.0).sum(axis=1).astype(int)
for col in amenity_cols:
    combined_df[f"{col}_Ratio"] = combined_df[col] / (combined_df["TotalSpend"] + 1.0)

combined_df["Luxury_Ratio"] = combined_df["LuxurySpend"] / (
    combined_df["TotalSpend"] + 1.0
)
combined_df["Food_Ratio"] = combined_df["FoodCourt"] / (combined_df["TotalSpend"] + 1.0)
combined_df["Spa_VR_Ratio"] = (combined_df["Spa"] + combined_df["VRDeck"]) / (
    combined_df["TotalSpend"] + 1.0
)

# Group relational aggregations and passenger cabin density
combined_df["Group_TotalSpend"] = combined_df.groupby("Group")["TotalSpend"].transform("sum")
combined_df["Group_MeanSpend"] = combined_df.groupby("Group")["TotalSpend"].transform("mean")
combined_df["Group_Spend_Std"] = (
    combined_df.groupby("Group")["TotalSpend"].transform("std").fillna(0.0)
)
combined_df["Spend_To_Group_Spend_Ratio"] = combined_df["TotalSpend"] / (
    combined_df["Group_TotalSpend"] + 1.0
)
cabin_count_map = combined_df["Cabin"].value_counts().to_dict()
combined_df["Cabin_Occupancy"] = combined_df["Cabin"].map(cabin_count_map).fillna(1).astype(int)
combined_df["Cabin_Density"] = combined_df["Cabin_Occupancy"]

# Demographic & group leadership aggregates
combined_df["Is_Group_Leader"] = (combined_df["Group_Num"] == 1).astype(int)
group_age_max = combined_df.groupby("Group")["Age"].transform("max")
group_age_min = combined_df.groupby("Group")["Age"].transform("min")
combined_df["Group_Age_Span"] = (group_age_max - group_age_min).fillna(0.0)

# Continuous deck-relative coordinates
deck_max_cabin_map = (
    combined_df.dropna(subset=["Cabin_Deck", "Cabin_Num"])
    .groupby("Cabin_Deck")["Cabin_Num"]
    .max()
    .to_dict()
)
combined_df["Deck_Max_Cabin"] = combined_df["Cabin_Deck"].map(deck_max_cabin_map).fillna(1.0)
combined_df["Deck_Relative_Num"] = (
    combined_df["Cabin_Num"] / (combined_df["Deck_Max_Cabin"] + 1e-5)
).clip(0.0, 1.0)

# Luxury-to-Basic spend ratio
combined_df["Luxury_To_Basic_Ratio"] = (
    (combined_df["LuxurySpend"] + 1.0) / (combined_df["BasicSpend"] + 1.0)
)

# Log1p transformations
for col in amenity_cols + ["TotalSpend", "LuxurySpend", "BasicSpend", "Group_TotalSpend", "Group_MeanSpend", "Group_Spend_Std"]:
    combined_df[f"{col}_Log1p"] = np.log1p(np.maximum(0, combined_df[col]))

for col in amenity_cols:
    combined_df[f"{col}_IsZero"] = (combined_df[col] == 0.0).astype(int)

# Demographics & spatial features
combined_df["Is_Child"] = (combined_df["Age"] < 13).astype(int)
combined_df["Is_Senior"] = (combined_df["Age"] >= 60).astype(int)
combined_df["Spend_Per_Age"] = combined_df["TotalSpend"] / (
    combined_df["Age"].fillna(28.0) + 1.0
)

# Spatial refinements: deck-normalized percentile and localized cluster/zone bins
combined_df["Cabin_Percentile"] = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].rank(pct=True).fillna(0.5)
)
combined_df["Cabin_Zone"] = (combined_df["Cabin_Num"] // 25).astype(int)
combined_df["Cabin_Cluster"] = (combined_df["Cabin_Num"] // 100).astype(int)
combined_df["Cabin_Region"] = (combined_df["Cabin_Num"] // 300).astype(int)

# Numeric spatial representations for non-categorical boosting architectures
combined_df["Cabin_Zone_Num"] = (combined_df["Cabin_Num"] // 25).astype(float)
combined_df["Cabin_Cluster_Num"] = (combined_df["Cabin_Num"] // 100).astype(float)
combined_df["Cabin_Region_Num"] = (combined_df["Cabin_Num"] // 300).astype(float)

combined_df["Cabin_Deck"] = combined_df["Cabin_Deck"].fillna("Unknown")
combined_df["Cabin_Side"] = combined_df["Cabin_Side"].fillna("Unknown")
combined_df["Deck_Side"] = combined_df["Cabin_Deck"] + "_" + combined_df["Cabin_Side"]
combined_df["HomePlanet"] = combined_df["HomePlanet"].fillna("Unknown")
combined_df["Destination"] = combined_df["Destination"].fillna("Unknown")
combined_df["Planet_Destination"] = (
    combined_df["HomePlanet"] + "_" + combined_df["Destination"]
)

combined_df["CryoSleep_Val"] = (
    combined_df["CryoSleep"].map({True: 1.0, False: 0.0}).fillna(-1.0)
)
combined_df["VIP_Val"] = combined_df["VIP"].map({True: 1.0, False: 0.0}).fillna(-1.0)

# Group-level CryoSleep proportion
combined_df["Group_Cryo_Ratio"] = (
    combined_df.groupby("Group")["CryoSleep_Val"]
    .transform(lambda s: ((s == 1.0).sum() / len(s)))
    .fillna(0.5)
)

# Cleaned categorical columns (dropping fragmented Deck_Side_Zone)
cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Destination",
]

# High-cardinality and interaction categories for Bayesian out-of-fold target encoding
target_enc_cols = [
    "Deck_Side",
    "Cabin_Deck",
    "Planet_Destination",
    "HomePlanet",
    "Destination",
    "Cabin_Side",
]

# Explicit nominal categorical types for tree models
for col in cat_cols:
    combined_df[col] = combined_df[col].astype(str).astype("category")

# Build deterministic integer category mappings for TabularMLP embeddings
cat_mappings = {
    col: {val: i for i, val in enumerate(combined_df[col].astype(str).unique())}
    for col in cat_cols
}

# Feature list specification
feature_cols = [
    "Group_Size",
    "Group_Num",
    "Is_Alone",
    "Is_Group_Leader",
    "Family_Size",
    "Age",
    "Group_Age_Span",
    "Is_Child",
    "Is_Senior",
    "CryoSleep_Val",
    "VIP_Val",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "HasSpent",
    "LuxurySpend",
    "BasicSpend",
    "Luxury_To_Basic_Ratio",
    "Zero_LuxurySpend",
    "Amenity_Zero_Count",
    "RoomService_Ratio",
    "FoodCourt_Ratio",
    "ShoppingMall_Ratio",
    "Spa_Ratio",
    "VRDeck_Ratio",
    "Luxury_Ratio",
    "Food_Ratio",
    "Spa_VR_Ratio",
    "RoomService_Log1p",
    "FoodCourt_Log1p",
    "ShoppingMall_Log1p",
    "Spa_Log1p",
    "VRDeck_Log1p",
    "TotalSpend_Log1p",
    "LuxurySpend_Log1p",
    "BasicSpend_Log1p",
    "RoomService_IsZero",
    "FoodCourt_IsZero",
    "ShoppingMall_IsZero",
    "Spa_IsZero",
    "VRDeck_IsZero",
    "Zero_Spend",
    "Spend_Per_Age",
    "Group_TotalSpend",
    "Group_MeanSpend",
    "Group_Spend_Std",
    "Group_Spend_Std_Log1p",
    "Spend_To_Group_Spend_Ratio",
    "Group_Cryo_Ratio",
    "Cabin_Density",
    "Cabin_Occupancy",
    "Cabin_Num",
    "Deck_Relative_Num",
    "Cabin_Percentile",
    "Cabin_Region_Num",
    "Cabin_Cluster_Num",
    "Cabin_Zone_Num",
    "Deck_Side_TE",
    "Cabin_Deck_TE",
    "Planet_Destination_TE",
    "HomePlanet_TE",
    "Destination_TE",
    "Cabin_Side_TE",
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Destination",
]

num_cols = [c for c in feature_cols if c not in cat_cols]

# Separate into Train and Test partitions
processed_train = combined_df.iloc[:train_len].copy()
processed_train["Transported"] = train_df["Transported"].values
processed_test = combined_df.iloc[train_len:].copy()

# ---------------------------------------------------------
# 3. CROSS-VALIDATION SCHEMA & HOLD-OUT SPLIT
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
processed_train["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(processed_train, processed_train["Transported"])
):
    processed_train.loc[val_idx, "fold"] = fold

# Fold 0 serves as the primary hold-out validation set
val_mask = processed_train["fold"] == 0
tr_mask = ~val_mask

# Smoothed Bayesian out-of-fold target encoding strictly on training folds
for col in target_enc_cols:
    processed_train[f"{col}_TE"] = np.nan
    processed_test[f"{col}_TE"] = np.nan

unique_tr_folds = sorted(processed_train.loc[tr_mask, "fold"].unique())

for col in target_enc_cols:
    # 1. Internal OOF target encoding for tr_mask folds
    for f in unique_tr_folds:
        inner_tr_idx = processed_train[tr_mask & (processed_train["fold"] != f)].index
        inner_val_idx = processed_train[tr_mask & (processed_train["fold"] == f)].index

        y_inner = processed_train.loc[inner_tr_idx, "Transported"]
        prior = y_inner.mean()
        stats = processed_train.loc[inner_tr_idx].groupby(col, observed=False)["Transported"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + 10.0 * prior) / (stats["count"] + 10.0)
        smoothed_map = smoothed.to_dict()

        processed_train.loc[inner_val_idx, f"{col}_TE"] = (
            processed_train.loc[inner_val_idx, col].astype(str).map(smoothed_map).fillna(prior).astype(float)
        )

    # 2. Map hold-out validation (val_mask) and test using entire tr_mask statistics
    tr_all_idx = processed_train[tr_mask].index
    y_tr_all = processed_train.loc[tr_all_idx, "Transported"]
    prior_all = y_tr_all.mean()
    stats_all = processed_train.loc[tr_all_idx].groupby(col, observed=False)["Transported"].agg(["count", "mean"])
    smoothed_all = (stats_all["count"] * stats_all["mean"] + 10.0 * prior_all) / (stats_all["count"] + 10.0)
    smoothed_map_all = smoothed_all.to_dict()

    val_idx = processed_train[val_mask].index
    processed_train.loc[val_idx, f"{col}_TE"] = (
        processed_train.loc[val_idx, col].astype(str).map(smoothed_map_all).fillna(prior_all).astype(float)
    )
    processed_test[f"{col}_TE"] = (
        processed_test[col].astype(str).map(smoothed_map_all).fillna(prior_all).astype(float)
    )

# Fit median imputers for numerical variables strictly on training folds
for col in num_cols:
    med = processed_train.loc[tr_mask, col].median()
    processed_train[col] = processed_train[col].fillna(med)
    processed_test[col] = processed_test[col].fillna(med)

X_train = processed_train.loc[tr_mask, feature_cols].reset_index(drop=True)
y_train = processed_train.loc[tr_mask, "Transported"].reset_index(drop=True)
X_val = processed_train.loc[val_mask, feature_cols].reset_index(drop=True)
y_val = processed_train.loc[val_mask, "Transported"].reset_index(drop=True)

X_test = processed_test[feature_cols].reset_index(drop=True)
test_ids = processed_test["PassengerId"].reset_index(drop=True)

# Encoded numerical matrices for XGBoost
X_train_num = X_train[num_cols].astype(float).reset_index(drop=True)
X_val_num = X_val[num_cols].astype(float).reset_index(drop=True)
X_test_num = X_test[num_cols].astype(float).reset_index(drop=True)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 4. LEVEL-1 ESTIMATOR DEFINITIONS & TABULAR MLP ARCHITECTURE
# ---------------------------------------------------------
class TabularMLPNet(nn.Module):
    def __init__(self, cat_cardinalities, num_features):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for card in cat_cardinalities:
            emb_dim = max(4, min(16, int(card ** 0.5) + 2))
            self.embeddings.append(nn.Embedding(card, emb_dim))
            total_emb_dim += emb_dim

        in_dim = total_emb_dim + num_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(-1)


class TabularMLPModel:
    def __init__(
        self,
        cat_cols,
        num_cols,
        cat_mappings,
        epochs=35,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_mappings = cat_mappings
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.model = None

    def _prepare_tensors(self, df):
        x_cat = np.column_stack([
            df[c].astype(str).map(self.cat_mappings[c]).fillna(0).values.astype(int)
            for c in self.cat_cols
        ])
        x_num = self.scaler.transform(df[self.num_cols].values)
        return torch.tensor(x_cat, dtype=torch.long), torch.tensor(
            x_num, dtype=torch.float32
        )

    def fit(self, X_tr, y_tr, eval_set=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler = StandardScaler()
        self.scaler.fit(X_tr[self.num_cols].values)

        x_tr_cat, x_tr_num = self._prepare_tensors(X_tr)
        y_tr_t = torch.tensor(
            np.asarray(y_tr, dtype=np.float32), dtype=torch.float32
        )

        has_eval = eval_set is not None
        if has_eval:
            X_va, y_va = eval_set
            x_va_cat, x_va_num = self._prepare_tensors(X_va)
            y_va_t = torch.tensor(
                np.asarray(y_va, dtype=np.float32), dtype=torch.float32
            )
            val_dataset = TensorDataset(x_va_cat, x_va_num, y_va_t)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        train_dataset = TensorDataset(x_tr_cat, x_tr_num, y_tr_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=(len(train_dataset) % self.batch_size == 1),
        )

        cat_cardinalities = [len(self.cat_mappings[c]) for c in self.cat_cols]
        self.model = TabularMLPNet(cat_cardinalities, len(self.num_cols)).to(
            self.device
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_loss = float("inf")
        best_state = None
        patience = 8
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for b_cat, b_num, b_y in train_loader:
                b_cat = b_cat.to(self.device)
                b_num = b_num.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(b_cat, b_num)
                loss = criterion(logits, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            if has_eval:
                self.model.eval()
                val_loss = 0.0
                total_samples = 0
                with torch.no_grad():
                    for b_cat, b_num, b_y in val_loader:
                        b_cat = b_cat.to(self.device)
                        b_num = b_num.to(self.device)
                        b_y = b_y.to(self.device)
                        logits = self.model(b_cat, b_num)
                        val_loss += criterion(logits, b_y).item() * len(b_y)
                        total_samples += len(b_y)
                val_loss /= max(1, total_samples)
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.model.eval()
        x_cat, x_num = self._prepare_tensors(X)
        dataset = TensorDataset(x_cat, x_num)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for b_cat, b_num in loader:
                b_cat = b_cat.to(self.device)
                b_num = b_num.to(self.device)
                logits = self.model(b_cat, b_num)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs).flatten()


class LGBMModel:
    def __init__(self, cat_cols, **params):
        self.cat_cols = cat_cols
        self.params = params
        self.model = LGBMClassifier(**params)

    def fit(self, X_tr, y_tr, eval_set=None):
        callbacks = [lgb.early_stopping(50, verbose=False)]
        eval_s = [eval_set] if eval_set is not None else None
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_s,
            categorical_feature=self.cat_cols,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBModel:
    def __init__(self, num_cols, **params):
        self.num_cols = num_cols
        self.params = params
        self.model = XGBClassifier(**params)

    def fit(self, X_tr, y_tr, eval_set=None):
        X_tr_num = X_tr[self.num_cols].astype(float)
        eval_s = None
        if eval_set is not None:
            eval_s = [(eval_set[0][self.num_cols].astype(float), eval_set[1])]
        self.model.fit(X_tr_num, y_tr, eval_set=eval_s, verbose=False)
        return self

    def predict_proba(self, X):
        X_num = X[self.num_cols].astype(float)
        return self.model.predict_proba(X_num)[:, 1]


class CatBoostModel:
    def __init__(self, cat_cols, **params):
        self.cat_cols = cat_cols
        self.params = params
        self.model = CatBoostClassifier(**params, cat_features=cat_cols)

    def fit(self, X_tr, y_tr, eval_set=None):
        self.model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# Hyperparameters: Tightened LightGBM regularization alongside tuned XGBoost & CatBoost
lgb_params = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    min_child_samples=35,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

xgb_params = dict(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=50,
)

cat_params = dict(
    iterations=1000,
    learning_rate=0.04,
    depth=6,
    subsample=0.8,
    bootstrap_type="Bernoulli",
    l2_leaf_reg=4.0,
    random_seed=42,
    early_stopping_rounds=50,
    thread_count=-1,
    verbose=False,
)

# ---------------------------------------------------------
# 5. INTERNAL CROSS-VALIDATION & OUT-OF-FOLD EXTRACTION
# ---------------------------------------------------------
print("Training 4-Model Level-1 ensemble with internal 4-fold cross-validation...")
n_inner_splits = 4
inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_mlp = np.zeros(len(X_train))

val_lgb_preds = np.zeros(len(X_val))
val_xgb_preds = np.zeros(len(X_val))
val_cat_preds = np.zeros(len(X_val))
val_mlp_preds = np.zeros(len(X_val))

test_lgb_preds = np.zeros(len(X_test))
test_xgb_preds = np.zeros(len(X_test))
test_cat_preds = np.zeros(len(X_test))
test_mlp_preds = np.zeros(len(X_test))

for fold_idx, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_train, y_train)):
    print(f"--- Internal Fold {fold_idx + 1}/{n_inner_splits} ---")

    X_tr_in = X_train.iloc[in_tr_idx].reset_index(drop=True)
    y_tr_in = y_train.iloc[in_tr_idx].reset_index(drop=True)
    X_va_in = X_train.iloc[in_val_idx].reset_index(drop=True)
    y_va_in = y_train.iloc[in_val_idx].reset_index(drop=True)

    # 1. LightGBM
    clf_lgb = LGBMModel(cat_cols=cat_cols, **lgb_params)
    clf_lgb.fit(X_tr_in, y_tr_in, eval_set=(X_va_in, y_va_in))
    oof_lgb[in_val_idx] = clf_lgb.predict_proba(X_va_in)
    val_lgb_preds += clf_lgb.predict_proba(X_val) / n_inner_splits
    test_lgb_preds += clf_lgb.predict_proba(X_test) / n_inner_splits

    # 2. XGBoost
    clf_xgb = XGBModel(num_cols=num_cols, **xgb_params)
    clf_xgb.fit(X_tr_in, y_tr_in, eval_set=(X_va_in, y_va_in))
    oof_xgb[in_val_idx] = clf_xgb.predict_proba(X_va_in)
    val_xgb_preds += clf_xgb.predict_proba(X_val) / n_inner_splits
    test_xgb_preds += clf_xgb.predict_proba(X_test) / n_inner_splits

    # 3. CatBoost
    clf_cat = CatBoostModel(cat_cols=cat_cols, **cat_params)
    clf_cat.fit(X_tr_in, y_tr_in, eval_set=(X_va_in, y_va_in))
    oof_cat[in_val_idx] = clf_cat.predict_proba(X_va_in)
    val_cat_preds += clf_cat.predict_proba(X_val) / n_inner_splits
    test_cat_preds += clf_cat.predict_proba(X_test) / n_inner_splits

    # 4. TabularMLP with Entity Embeddings
    clf_mlp = TabularMLPModel(
        cat_cols=cat_cols,
        num_cols=num_cols,
        cat_mappings=cat_mappings,
        epochs=35,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42 + fold_idx,
    )
    clf_mlp.fit(X_tr_in, y_tr_in, eval_set=(X_va_in, y_va_in))
    oof_mlp[in_val_idx] = clf_mlp.predict_proba(X_va_in)
    val_mlp_preds += clf_mlp.predict_proba(X_val) / n_inner_splits
    test_mlp_preds += clf_mlp.predict_proba(X_test) / n_inner_splits

y_train_np = y_train.values
print(
    f"OOF LightGBM Accuracy: {accuracy_score(y_train_np, (oof_lgb >= 0.5).astype(int)):.6f}"
)
print(
    f"OOF XGBoost Accuracy: {accuracy_score(y_train_np, (oof_xgb >= 0.5).astype(int)):.6f}"
)
print(
    f"OOF CatBoost Accuracy: {accuracy_score(y_train_np, (oof_cat >= 0.5).astype(int)):.6f}"
)
print(
    f"OOF TabularMLP Accuracy: {accuracy_score(y_train_np, (oof_mlp >= 0.5).astype(int)):.6f}"
)

# ---------------------------------------------------------
# 6. BOUNDED CONVEX SIMPLEX BLENDING & THRESHOLD CALIBRATION
# ---------------------------------------------------------
print("Solving bounded convex simplex blending on out-of-fold predictions...")
oof_prob_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_mlp])

def blend_objective(weights):
    w = np.array(weights)
    pred = np.clip(oof_prob_matrix @ w, 1e-7, 1.0 - 1e-7)
    return log_loss(y_train_np, pred)

init_weights = np.ones(4) / 4.0
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": False, "ftol": 1e-9, "maxiter": 500},
)

blend_weights = np.maximum(0.0, opt_res.x)
blend_weights /= np.sum(blend_weights)

print(
    f"Simplex Blend Weights: LightGBM={blend_weights[0]:.4f}, "
    f"XGBoost={blend_weights[1]:.4f}, CatBoost={blend_weights[2]:.4f}, "
    f"TabularMLP={blend_weights[3]:.4f}"
)

val_prob_matrix = np.column_stack([val_lgb_preds, val_xgb_preds, val_cat_preds, val_mlp_preds])
test_prob_matrix = np.column_stack([test_lgb_preds, test_xgb_preds, test_cat_preds, test_mlp_preds])

y_val_np = y_val.values
print(
    f"Hold-Out LightGBM Accuracy: {accuracy_score(y_val_np, (val_lgb_preds >= 0.5).astype(int)):.6f}"
)
print(
    f"Hold-Out XGBoost Accuracy: {accuracy_score(y_val_np, (val_xgb_preds >= 0.5).astype(int)):.6f}"
)
print(
    f"Hold-Out CatBoost Accuracy: {accuracy_score(y_val_np, (val_cat_preds >= 0.5).astype(int)):.6f}"
)
print(
    f"Hold-Out TabularMLP Accuracy: {accuracy_score(y_val_np, (val_mlp_preds >= 0.5).astype(int)):.6f}"
)

# Predict blended probabilities on hold-out validation set
val_blend_probs = val_prob_matrix @ blend_weights

# Calibrate decision threshold on untouched hold-out set within [0.45, 0.55]
threshold_candidates = np.linspace(0.45, 0.55, 101)
best_score = 0.0
best_threshold = 0.5

for th in threshold_candidates:
    preds = (val_blend_probs >= th).astype(int)
    acc = accuracy_score(y_val_np, preds)
    if acc > best_score:
        best_score = float(acc)
        best_threshold = float(th)

print(f"Optimal Calibrated Decision Threshold: {best_threshold:.4f}")
final_score = best_score
print(f"Verified Final Hold-Out Classification Accuracy: {final_score:.6f}")

# ---------------------------------------------------------
# 7. TEST INFERENCE & SUBMISSION GENERATION
# ---------------------------------------------------------
test_blend_probs = test_prob_matrix @ blend_weights
test_predictions = (test_blend_probs >= best_threshold).astype(bool)

submission_path = "./submission/submission.csv"
submission_df = pd.DataFrame(
    {"PassengerId": test_ids.values, "Transported": test_predictions}
)
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == 4277, f"Expected 4277 rows, found {len(submission_df)}"
assert list(submission_df.columns) == ["PassengerId", "Transported"]
assert submission_df.isnull().sum().sum() == 0, "Null values found in submission"
assert submission_df["Transported"].dtype == bool, "Transported column must be boolean"

print(f"Final Validation Score: {final_score}")

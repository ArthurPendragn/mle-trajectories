import os
import copy
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Deterministic physical spaceship deductions
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
# Earth passengers are never VIP
combined_df.loc[
    (combined_df["HomePlanet"] == "Earth") & combined_df["VIP"].isnull(),
    "VIP",
] = False

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
combined_df["Cabin_Density"] = combined_df["Cabin"].map(cabin_count_map).fillna(1).astype(int)

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

# Spatial refinements: deck-normalized percentile, deck-relative coordinate, and clusters
combined_df["Cabin_Percentile"] = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].rank(pct=True).fillna(0.5)
)
deck_max_cabin = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].transform("max").replace(0, 1.0)
)
combined_df["Deck_Relative_Num"] = (
    combined_df["Cabin_Num"] / deck_max_cabin
).fillna(0.0)
combined_df["Cabin_Zone"] = (combined_df["Cabin_Num"] // 25).astype(int)
combined_df["Cabin_Cluster"] = (combined_df["Cabin_Num"] // 100).astype(int)
combined_df["Cabin_Region"] = (combined_df["Cabin_Num"] // 300).astype(int)

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

cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Destination",
    "Cabin_Region",
    "Cabin_Cluster",
    "Cabin_Zone",
]

# Explicit nominal categorical types for tree models
for col in cat_cols:
    combined_df[col] = combined_df[col].astype(str).astype("category")

# Feature list specification
feature_cols = [
    "Group_Size",
    "Group_Num",
    "Is_Alone",
    "Family_Size",
    "Age",
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
    "Zero_LuxurySpend",
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
    "Cabin_Num",
    "Cabin_Percentile",
    "Deck_Relative_Num",
    "Cabin_Region",
    "Cabin_Cluster",
    "Cabin_Zone",
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

# Fit median imputers for numerical variables strictly on Train
for col in num_cols:
    med = processed_train[col].median()
    processed_train[col] = processed_train[col].fillna(med)
    processed_test[col] = processed_test[col].fillna(med)

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

X_train = processed_train.loc[tr_mask, feature_cols].reset_index(drop=True)
y_train = processed_train.loc[tr_mask, "Transported"].reset_index(drop=True)
X_val = processed_train.loc[val_mask, feature_cols].reset_index(drop=True)
y_val = processed_train.loc[val_mask, "Transported"].reset_index(drop=True)

X_test = processed_test[feature_cols].reset_index(drop=True)
test_ids = processed_test["PassengerId"].reset_index(drop=True)

# Continuous feature scaling fitted strictly on training partition
scaler = StandardScaler()
X_train_cont_scaled = scaler.fit_transform(X_train[num_cols])
X_val_cont_scaled = scaler.transform(X_val[num_cols])
X_test_cont_scaled = scaler.transform(X_test[num_cols])
X_train_cont = X_train_cont_scaled
X_val_cont = X_val_cont_scaled
X_test_cont = X_test_cont_scaled

# Categorical mapping to zero-indexed integer arrays for neural entity embeddings
cat_maps = {}
cat_dims = []
for c in cat_cols:
    uniques = sorted(list(combined_df[c].astype(str).unique()))
    cat_maps[c] = {val: idx for idx, val in enumerate(uniques)}
    cat_dims.append(len(uniques))

X_train_cat_idx = np.stack(
    [X_train[c].astype(str).map(cat_maps[c]).fillna(0).astype(int).values for c in cat_cols],
    axis=1,
)
X_val_cat_idx = np.stack(
    [X_val[c].astype(str).map(cat_maps[c]).fillna(0).astype(int).values for c in cat_cols],
    axis=1,
)
X_test_cat_idx = np.stack(
    [X_test[c].astype(str).map(cat_maps[c]).fillna(0).astype(int).values for c in cat_cols],
    axis=1,
)
X_train_cat = X_train_cat_idx
X_val_cat = X_val_cat_idx
X_test_cat = X_test_cat_idx

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 4. HYBRID MULTI-PARADIGM MODEL ARCHITECTURES
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.SiLU()  # Swish activation
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.SiLU()  # Swish activation
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.ln2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return residual + out


class TabularNet(nn.Module):
    def __init__(self, cat_dims, num_cont_features, cont_proj_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(16, max(4, int((card ** 0.5) * 2))))
            for card in cat_dims
        ])
        total_emb_dim = sum(
            min(16, max(4, int((card ** 0.5) * 2))) for card in cat_dims
        )
        self.cont_proj = nn.Linear(num_cont_features, cont_proj_dim)
        hidden_dim = total_emb_dim + cont_proj_dim
        self.block1 = ResidualBlock(hidden_dim, dropout=0.25)
        self.block2 = ResidualBlock(hidden_dim, dropout=0.25)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_cont, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_rep = torch.cat(emb_outs, dim=1)
        cont_rep = self.cont_proj(x_cont)
        x = torch.cat([cat_rep, cont_rep], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class TabularNetClassifier:
    def __init__(
        self,
        cat_dims,
        num_cont_features,
        cont_proj_dim=64,
        epochs=40,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
        random_state=42,
    ):
        self.cat_dims = cat_dims
        self.num_cont_features = num_cont_features
        self.cont_proj_dim = cont_proj_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model = None

    def _extract_inputs(self, X):
        if isinstance(X, tuple):
            return torch.tensor(X[0], dtype=torch.float32), torch.tensor(X[1], dtype=torch.long)
        if isinstance(X, pd.DataFrame):
            cont_data = scaler.transform(X[num_cols])
            cat_data = np.stack(
                [X[c].astype(str).map(cat_maps[c]).fillna(0).astype(int).values for c in cat_cols],
                axis=1,
            )
            return torch.tensor(cont_data, dtype=torch.float32), torch.tensor(cat_data, dtype=torch.long)
        raise ValueError("Unsupported input format for TabularNet")

    def fit(self, X, y, eval_set=None):
        seed_everything(self.random_state)
        self.model = TabularNet(
            self.cat_dims, self.num_cont_features, self.cont_proj_dim
        ).to(self.device)

        X_cont_t, X_cat_t = self._extract_inputs(X)
        y_t = torch.tensor(np.array(y), dtype=torch.long)

        train_dataset = TensorDataset(X_cont_t, X_cat_t, y_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if eval_set is not None:
            X_val_data, y_val_data = eval_set[0]
            val_cont_t, val_cat_t = self._extract_inputs(X_val_data)
            val_y_t = torch.tensor(np.array(y_val_data), dtype=torch.long)
            val_dataset = TensorDataset(val_cont_t, val_cat_t, val_y_t)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for b_cont, b_cat, b_y in train_loader:
                b_cont = b_cont.to(self.device)
                b_cat = b_cat.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(b_cont, b_cat)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                total_val = 0
                with torch.no_grad():
                    for b_cont, b_cat, b_y in val_loader:
                        b_cont = b_cont.to(self.device)
                        b_cat = b_cat.to(self.device)
                        b_y = b_y.to(self.device)
                        logits = self.model(b_cont, b_cat)
                        loss = criterion(logits, b_y)
                        val_loss += loss.item() * len(b_y)
                        total_val += len(b_y)
                val_loss /= total_val

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_cont_t, X_cat_t = self._extract_inputs(X)
        dataset = TensorDataset(X_cont_t, X_cat_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for b_cont, b_cat in loader:
                b_cont = b_cont.to(self.device)
                b_cat = b_cat.to(self.device)
                logits = self.model(b_cont, b_cat)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.vstack(probs_list)


# LightGBM: leaf-wise expansion with categorical handling and regularization
lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# XGBoost: depth-wise regularized trees with subsampling
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=50,
)

# CatBoost: symmetric oblivious trees with native categorical handling
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.04,
    depth=6,
    subsample=0.8,
    bootstrap_type="Bernoulli",
    l2_leaf_reg=3.0,
    random_seed=42,
    early_stopping_rounds=50,
    thread_count=-1,
    cat_features=cat_cols,
    verbose=False,
)

# TabularNet: regularized neural net with entity embeddings
tab_model = TabularNetClassifier(
    cat_dims=cat_dims,
    num_cont_features=len(num_cols),
    cont_proj_dim=64,
    epochs=40,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    random_state=42,
)

# ---------------------------------------------------------
# 5. MULTI-PARADIGM COMMITTEE TRAINING WITH EARLY STOPPING
# ---------------------------------------------------------
print("Training hybrid multi-paradigm ensemble (LightGBM, XGBoost, CatBoost, TabularNet)...")

# Train LightGBM
print("Training LightGBM model...")
try:
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
except (TypeError, Exception):
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

# Train XGBoost
print("Training XGBoost model...")
xgb_needs_encoded = False
try:
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
except Exception:
    xgb_needs_encoded = True
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()
    for c in cat_cols:
        X_train_xgb[c] = X_train_xgb[c].cat.codes
        X_val_xgb[c] = X_val_xgb[c].cat.codes
        X_test_xgb[c] = X_test_xgb[c].cat.codes
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)

# Train CatBoost
print("Training CatBoost model...")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

# Train TabularNet
print("Training TabularNet model (40 epochs with AdamW & CosineAnnealingLR)...")
tab_model.fit(
    (X_train_cont_scaled, X_train_cat_idx),
    y_train,
    eval_set=[((X_val_cont_scaled, X_val_cat_idx), y_val)],
)

# ---------------------------------------------------------
# 6. HOLD-OUT SOFT-VOTING BLENDING & THRESHOLD CALIBRATION
# ---------------------------------------------------------
# Validation probability extraction
lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = (
    xgb_model.predict_proba(X_val_xgb if xgb_needs_encoded else X_val)[:, 1]
)
cat_val_probs = cat_model.predict_proba(X_val)[:, 1]
tab_val_probs = tab_model.predict_proba(
    (X_val_cont_scaled, X_val_cat_idx)
)[:, 1]

# Test probability extraction
lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]
xgb_test_probs = (
    xgb_model.predict_proba(X_test_xgb if xgb_needs_encoded else X_test)[:, 1]
)
cat_test_probs = cat_model.predict_proba(X_test)[:, 1]
tab_test_probs = tab_model.predict_proba(
    (X_test_cont_scaled, X_test_cat_idx)
)[:, 1]

y_val_np = y_val.values
print(
    f"LightGBM Validation Accuracy: {accuracy_score(y_val_np, (lgb_val_probs >= 0.5).astype(int)):.6f}"
)
print(
    f"XGBoost Validation Accuracy: {accuracy_score(y_val_np, (xgb_val_probs >= 0.5).astype(int)):.6f}"
)
print(
    f"CatBoost Validation Accuracy: {accuracy_score(y_val_np, (cat_val_probs >= 0.5).astype(int)):.6f}"
)
print(
    f"TabularNet Validation Accuracy: {accuracy_score(y_val_np, (tab_val_probs >= 0.5).astype(int)):.6f}"
)

# Stack 4-model probabilities for bounded ensemble search
val_prob_matrix = np.column_stack(
    [lgb_val_probs, xgb_val_probs, cat_val_probs, tab_val_probs]
)
test_prob_matrix = np.column_stack(
    [lgb_test_probs, xgb_test_probs, cat_test_probs, tab_test_probs]
)

# Generate bounded simplex blending weights to preserve model diversity
step = 0.05
weight_candidates = []
for w1 in np.arange(0.0, 1.01, step):
    for w2 in np.arange(0.0, 1.01 - w1, step):
        for w3 in np.arange(0.0, 1.01 - w1 - w2, step):
            w4 = round(1.0 - w1 - w2 - w3, 4)
            if w4 >= -1e-6:
                w4 = max(0.0, w4)
                if abs(w1 + w2 + w3 + w4 - 1.0) < 1e-4:
                    weight_candidates.append([round(w1, 4), round(w2, 4), round(w3, 4), round(w4, 4)])

# Also include continuous log-loss optimal weights on simplex
def nll_objective(w):
    w_norm = np.maximum(0, w)
    s = np.sum(w_norm)
    if s == 0:
        return 999.0
    w_norm = w_norm / s
    p = np.clip(np.dot(val_prob_matrix, w_norm), 1e-7, 1.0 - 1e-7)
    return log_loss(y_val_np, p)

opt_res = minimize(
    nll_objective,
    [0.25, 0.25, 0.25, 0.25],
    method="SLSQP",
    bounds=[(0.0, 1.0)] * 4,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)
if opt_res.success:
    w_slsqp = opt_res.x / np.sum(opt_res.x)
    weight_candidates.append(list(w_slsqp))

weight_candidates.append([0.25, 0.25, 0.25, 0.25])
weights_arr = np.array(weight_candidates)

# Vectorized simplex prediction combinations
val_blends = np.dot(weights_arr, val_prob_matrix.T)

# Decision threshold calibrated strictly in balanced prior interval [0.45, 0.55]
threshold_candidates = np.linspace(0.45, 0.55, 41)
best_score = 0.0
best_weights = np.array([0.25, 0.25, 0.25, 0.25])
best_threshold = 0.5

for th in threshold_candidates:
    preds = (val_blends >= th).astype(int)
    accs = (preds == y_val_np).mean(axis=1)
    best_idx = np.argmax(accs)
    if accs[best_idx] > best_score:
        best_score = float(accs[best_idx])
        best_weights = weights_arr[best_idx]
        best_threshold = float(th)

best_weights = best_weights / np.sum(best_weights)
print(
    f"Optimal Ensemble Blending Weights: LightGBM={best_weights[0]:.2f}, "
    f"XGBoost={best_weights[1]:.2f}, CatBoost={best_weights[2]:.2f}, "
    f"TabularNet={best_weights[3]:.2f}"
)
print(f"Optimal Decision Threshold: {best_threshold:.4f}")

# Final validation evaluation
final_val_probs = np.dot(val_prob_matrix, best_weights)
final_val_preds = (final_val_probs >= best_threshold).astype(int)
final_score = accuracy_score(y_val_np, final_val_preds)
print(f"Verified Final Hold-Out Classification Accuracy: {final_score:.6f}")

# ---------------------------------------------------------
# 7. TEST INFERENCE & SUBMISSION GENERATION
# ---------------------------------------------------------
final_test_probs = np.dot(test_prob_matrix, best_weights)
test_predictions = (final_test_probs >= best_threshold).astype(bool)

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

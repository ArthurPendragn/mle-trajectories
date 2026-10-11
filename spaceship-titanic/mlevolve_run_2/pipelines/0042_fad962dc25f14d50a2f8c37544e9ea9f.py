import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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

# Complete 5-way amenity expenditure profiles
combined_df["Amenity_Zero_Count"] = (
    (combined_df[amenity_cols] == 0.0).sum(axis=1).astype(int)
)
for col in amenity_cols:
    combined_df[f"{col}_Ratio"] = combined_df[col] / (combined_df["TotalSpend"] + 1.0)

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

# Spatial refinements: smooth deck-relative coordinates and localized cluster/zone bins
deck_max_cabin = combined_df.groupby("Cabin_Deck")["Cabin_Num"].transform("max")
combined_df["Deck_Relative_Pos"] = combined_df["Cabin_Num"] / (
    deck_max_cabin.replace(0, 1.0) + 1e-5
)
combined_df["Cabin_Percentile"] = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].rank(pct=True).fillna(0.5)
)
overall_max_cabin = combined_df["Cabin_Num"].max()
combined_df["Cabin_Num_Norm"] = combined_df["Cabin_Num"] / (
    overall_max_cabin if overall_max_cabin > 0 else 1.0
)
combined_df["Cabin_Is_Even"] = (combined_df["Cabin_Num"] % 2 == 0).astype(int)
combined_df["Num_Amenities_Used"] = (
    (combined_df[amenity_cols] > 0).sum(axis=1).astype(int)
)
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

# Group-level CryoSleep concordance metrics (ratio and uniformity of known sleepers)
cryo_known = combined_df[combined_df["CryoSleep"].notnull()]
group_cryo_sum = cryo_known.groupby("Group")["CryoSleep"].sum()
group_cryo_count = cryo_known.groupby("Group")["CryoSleep"].count()
group_cryo_ratio = (group_cryo_sum / group_cryo_count).to_dict()

combined_df["Group_Cryo_Ratio"] = (
    combined_df["Group"].map(group_cryo_ratio).fillna(0.5)
)
combined_df["Group_Cryo_Uniformity"] = (
    np.abs(combined_df["Group_Cryo_Ratio"] - 0.5) * 2.0
)
combined_df["Group_Cryo_Concordance"] = (
    (combined_df["Group_Cryo_Ratio"] == 1.0) | (combined_df["Group_Cryo_Ratio"] == 0.0)
).astype(int)

# Family luxury expenditure dispersion metrics
combined_df["Family_Luxury_Var"] = (
    combined_df.groupby("Surname")["LuxurySpend"].transform("var").fillna(0.0)
)
combined_df["Family_Luxury_Std"] = (
    combined_df.groupby("Surname")["LuxurySpend"].transform("std").fillna(0.0)
)
combined_df["Family_Luxury_Mean"] = (
    combined_df.groupby("Surname")["LuxurySpend"].transform("mean").fillna(0.0)
)
combined_df["Family_Luxury_Dispersion"] = combined_df["Family_Luxury_Std"] / (
    combined_df["Family_Luxury_Mean"] + 1.0
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
    "RoomService_Ratio",
    "FoodCourt_Ratio",
    "ShoppingMall_Ratio",
    "Spa_Ratio",
    "VRDeck_Ratio",
    "Amenity_Zero_Count",
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
    "Group_Cryo_Uniformity",
    "Group_Cryo_Concordance",
    "Family_Luxury_Var",
    "Family_Luxury_Std",
    "Family_Luxury_Dispersion",
    "Cabin_Density",
    "Cabin_Num",
    "Cabin_Num_Norm",
    "Cabin_Is_Even",
    "Num_Amenities_Used",
    "Cabin_Percentile",
    "Deck_Relative_Pos",
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

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 4. TABULAR MLP ARCHITECTURE & INPUT REPRESENTATIONS
# ---------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, emb_cardinalities, num_features_count):
        super(TabularMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=min(32, max(4, int(card ** 0.5) + 2)))
            for card in emb_cardinalities
        ])
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = total_emb_dim + num_features_count

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if len(emb_outs) > 0:
            x_cat_emb = torch.cat(emb_outs, dim=1)
            x = torch.cat([x_cat_emb, x_num], dim=1)
        else:
            x = x_num
        return self.net(x).squeeze(-1)


class TabularMLPClassifier:
    def __init__(
        self,
        cat_cols,
        num_cols,
        epochs=30,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cat_mappings = {}
        self.emb_cardinalities = []
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_inputs(self, X, is_fit=False):
        cat_list = []
        for col in self.cat_cols:
            if is_fit:
                unique_vals = X[col].astype(str).unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                mapping["__unknown__"] = len(unique_vals)
                self.cat_mappings[col] = mapping
                self.emb_cardinalities.append(len(mapping))

            mapping = self.cat_mappings[col]
            unknown_idx = mapping["__unknown__"]
            codes = X[col].astype(str).map(lambda v: mapping.get(v, unknown_idx)).values
            cat_list.append(codes)

        x_cat = (
            np.column_stack(cat_list)
            if len(cat_list) > 0
            else np.empty((len(X), 0), dtype=int)
        )

        if is_fit:
            x_num = self.scaler.fit_transform(X[self.num_cols].fillna(0.0))
        else:
            x_num = self.scaler.transform(X[self.num_cols].fillna(0.0))

        return torch.tensor(x_cat, dtype=torch.long), torch.tensor(
            x_num, dtype=torch.float32
        )

    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.cat_mappings = {}
        self.emb_cardinalities = []
        x_tr_cat, x_tr_num = self._prepare_inputs(X, is_fit=True)
        y_tr = torch.tensor(np.asarray(y), dtype=torch.float32)

        dataset = TensorDataset(x_tr_cat, x_tr_num, y_tr)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        x_va_cat, x_va_num, y_va = None, None, None
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            x_va_cat, x_va_num = self._prepare_inputs(X_val, is_fit=False)
            y_va = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        self.model = TabularMLP(self.emb_cardinalities, len(self.num_cols)).to(
            self.device
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for b_cat, b_num, b_y in loader:
                b_cat = b_cat.to(self.device)
                b_num = b_num.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                out = self.model(b_cat, b_num)
                loss = criterion(out, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            if eval_set is not None:
                self.model.eval()
                with torch.no_grad():
                    va_cat = x_va_cat.to(self.device)
                    va_num = x_va_num.to(self.device)
                    va_y = y_va.to(self.device)
                    val_out = self.model(va_cat, va_num)
                    val_loss = criterion(val_out, va_y).item()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.model.eval()
        x_cat, x_num = self._prepare_inputs(X, is_fit=False)
        dataset = TensorDataset(x_cat, x_num)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        probs = []
        with torch.no_grad():
            for b_cat, b_num in loader:
                b_cat = b_cat.to(self.device)
                b_num = b_num.to(self.device)
                out = self.model(b_cat, b_num)
                p1 = torch.sigmoid(out).cpu().numpy()
                probs.extend(p1.tolist())

        p1 = np.array(probs)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


# 1. CatBoost representations: cat_cols as string
X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

# 2. LightGBM and XGBoost representations: cat_cols as category
X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_val_lgb[col] = X_val_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# 3. Numeric representation for XGBoost fallback if needed
X_train_num = X_train.copy()
X_val_num = X_val.copy()
X_test_num = X_test.copy()
for col in cat_cols:
    X_train_num[col] = X_train[col].astype("category").cat.codes.astype(float)
    X_val_num[col] = X_val[col].astype("category").cat.codes.astype(float)
    X_test_num[col] = X_test[col].astype("category").cat.codes.astype(float)

# ---------------------------------------------------------
# 5. MULTI-STAGE OUT-OF-FOLD (OOF) STACKING & BAGGED INFERENCE
# ---------------------------------------------------------
n_inner_splits = 5
inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train), 4), dtype=float)
val_preds_bagged = np.zeros((len(X_val), 4), dtype=float)
test_preds_bagged = np.zeros((len(X_test), 4), dtype=float)

print("Starting internal 5-fold cross-validation stacking pipeline on folds 1-4...")

for inner_fold, (in_tr_idx, in_va_idx) in enumerate(
    inner_skf.split(X_train, y_train)
):
    print(f"--- Training Stack Fold {inner_fold + 1}/{n_inner_splits} ---")
    y_in_tr = y_train.iloc[in_tr_idx].values
    y_in_va = y_train.iloc[in_va_idx].values

    # Base Model 1: LightGBM with histogram categorical splits
    lgb_inst = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + inner_fold,
        n_jobs=-1,
        verbose=-1,
    )
    try:
        lgb_inst.fit(
            X_train_lgb.iloc[in_tr_idx],
            y_in_tr,
            eval_set=[(X_train_lgb.iloc[in_va_idx], y_in_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    except (TypeError, Exception):
        lgb_inst.fit(
            X_train_lgb.iloc[in_tr_idx],
            y_in_tr,
            eval_set=[(X_train_lgb.iloc[in_va_idx], y_in_va)],
            early_stopping_rounds=50,
            verbose=False,
        )
    oof_preds[in_va_idx, 0] = lgb_inst.predict_proba(
        X_train_lgb.iloc[in_va_idx]
    )[:, 1]
    val_preds_bagged[:, 0] += (
        lgb_inst.predict_proba(X_val_lgb)[:, 1] / n_inner_splits
    )
    test_preds_bagged[:, 0] += (
        lgb_inst.predict_proba(X_test_lgb)[:, 1] / n_inner_splits
    )

    # Base Model 2: XGBoost with hist method and categorical enablement
    try:
        try:
            xgb_inst = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                tree_method="hist",
                enable_categorical=True,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42 + inner_fold,
                n_jobs=-1,
                eval_metric="logloss",
                early_stopping_rounds=50,
            )
            xgb_inst.fit(
                X_train_lgb.iloc[in_tr_idx],
                y_in_tr,
                eval_set=[(X_train_lgb.iloc[in_va_idx], y_in_va)],
                verbose=False,
            )
        except TypeError:
            xgb_inst = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                tree_method="hist",
                enable_categorical=True,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42 + inner_fold,
                n_jobs=-1,
                eval_metric="logloss",
            )
            xgb_inst.fit(
                X_train_lgb.iloc[in_tr_idx],
                y_in_tr,
                eval_set=[(X_train_lgb.iloc[in_va_idx], y_in_va)],
                early_stopping_rounds=50,
                verbose=False,
            )
        oof_preds[in_va_idx, 1] = xgb_inst.predict_proba(
            X_train_lgb.iloc[in_va_idx]
        )[:, 1]
        val_preds_bagged[:, 1] += (
            xgb_inst.predict_proba(X_val_lgb)[:, 1] / n_inner_splits
        )
        test_preds_bagged[:, 1] += (
            xgb_inst.predict_proba(X_test_lgb)[:, 1] / n_inner_splits
        )
    except Exception:
        # Robust numeric fallback if categorical enablement encounters version constraints
        xgb_inst = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + inner_fold,
            n_jobs=-1,
            eval_metric="logloss",
        )
        xgb_inst.fit(
            X_train_num.iloc[in_tr_idx],
            y_in_tr,
            eval_set=[(X_train_num.iloc[in_va_idx], y_in_va)],
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_preds[in_va_idx, 1] = xgb_inst.predict_proba(
            X_train_num.iloc[in_va_idx]
        )[:, 1]
        val_preds_bagged[:, 1] += (
            xgb_inst.predict_proba(X_val_num)[:, 1] / n_inner_splits
        )
        test_preds_bagged[:, 1] += (
            xgb_inst.predict_proba(X_test_num)[:, 1] / n_inner_splits
        )

    # Base Model 3: CatBoost with native categorical features
    cat_inst = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        l2_leaf_reg=3.0,
        random_seed=42 + inner_fold,
        early_stopping_rounds=50,
        thread_count=-1,
        cat_features=cat_cols,
        verbose=False,
    )
    cat_inst.fit(
        X_train_cat.iloc[in_tr_idx],
        y_in_tr,
        eval_set=(X_train_cat.iloc[in_va_idx], y_in_va),
        verbose=False,
    )
    oof_preds[in_va_idx, 2] = cat_inst.predict_proba(
        X_train_cat.iloc[in_va_idx]
    )[:, 1]
    val_preds_bagged[:, 2] += (
        cat_inst.predict_proba(X_val_cat)[:, 1] / n_inner_splits
    )
    test_preds_bagged[:, 2] += (
        cat_inst.predict_proba(X_test_cat)[:, 1] / n_inner_splits
    )

    # Base Model 4: Regularized PyTorch Tabular MLP with entity embeddings
    mlp_inst = TabularMLPClassifier(
        cat_cols=cat_cols,
        num_cols=num_cols,
        epochs=30,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        random_state=42 + inner_fold,
    )
    mlp_inst.fit(
        X_train.iloc[in_tr_idx],
        y_in_tr,
        eval_set=[(X_train.iloc[in_va_idx], y_in_va)],
    )
    oof_preds[in_va_idx, 3] = mlp_inst.predict_proba(
        X_train.iloc[in_va_idx]
    )[:, 1]
    val_preds_bagged[:, 3] += (
        mlp_inst.predict_proba(X_val)[:, 1] / n_inner_splits
    )
    test_preds_bagged[:, 3] += (
        mlp_inst.predict_proba(X_test)[:, 1] / n_inner_splits
    )

# ---------------------------------------------------------
# 6. STAGE-2 BOUNDED CONVEX SIMPLEX BLENDING & EVALUATION
# ---------------------------------------------------------
model_names = ["LightGBM", "XGBoost", "CatBoost", "TabularMLP"]
for idx, name in enumerate(model_names):
    oof_acc = accuracy_score(y_train.values, (oof_preds[:, idx] >= 0.5).astype(int))
    val_acc = accuracy_score(
        y_val.values, (val_preds_bagged[:, idx] >= 0.5).astype(int)
    )
    print(f"{name} - OOF Accuracy: {oof_acc:.6f} | Bagged Val Accuracy: {val_acc:.6f}")


# Solve for optimal non-negative blending weights summing to 1.0 using SLSQP
def blending_objective(weights):
    weights = np.array(weights)
    blend = np.dot(oof_preds, weights)
    blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
    return -np.mean(
        y_train.values * np.log(blend) + (1.0 - y_train.values) * np.log(1.0 - blend)
    )


n_models = len(model_names)
initial_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blending_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 200, "ftol": 1e-9},
)

opt_weights = np.maximum(0.0, opt_res.x)
opt_weights = opt_weights / np.sum(opt_weights)

weight_summary = dict(zip(model_names, [round(float(w), 4) for w in opt_weights]))
print(f"Optimal Blending Simplex Weights: {weight_summary}")

# Calibrate decision threshold strictly on internal OOF predictions within [0.45, 0.55]
oof_blended_probs = np.dot(oof_preds, opt_weights)
threshold_candidates = np.linspace(0.45, 0.55, 101)
best_threshold = 0.5
best_oof_score = 0.0

for th in threshold_candidates:
    preds = (oof_blended_probs >= th).astype(int)
    acc = accuracy_score(y_train.values, preds)
    if acc > best_oof_score:
        best_oof_score = float(acc)
        best_threshold = float(th)

print(
    f"Calibrated OOF Decision Threshold: {best_threshold:.4f} (OOF Accuracy: {best_oof_score:.6f})"
)

# Evaluate on hold-out fold 0 before generating test predictions
val_blended_probs = np.dot(val_preds_bagged, opt_weights)
val_final_preds = (val_blended_probs >= best_threshold).astype(int)
final_val_score = float(accuracy_score(y_val.values, val_final_preds))
print(f"Verified Final Hold-Out Classification Accuracy: {final_val_score:.6f}")

# ---------------------------------------------------------
# 7. TEST INFERENCE & SUBMISSION GENERATION
# ---------------------------------------------------------
test_blended_probs = np.dot(test_preds_bagged, opt_weights)
test_predictions = (test_blended_probs >= best_threshold).astype(bool)

submission_path = "./submission/submission.csv"
submission_df = pd.DataFrame(
    {"PassengerId": test_ids.values, "Transported": test_predictions}
)
submission_df.to_csv(submission_path, index=False)

assert len(submission_df) == 4277, f"Expected 4277 rows, found {len(submission_df)}"
assert list(submission_df.columns) == ["PassengerId", "Transported"]
assert submission_df.isnull().sum().sum() == 0, "Null values found in submission"
assert submission_df["Transported"].dtype == bool, "Transported column must be boolean"

print(f"Final Validation Score: {final_val_score}")

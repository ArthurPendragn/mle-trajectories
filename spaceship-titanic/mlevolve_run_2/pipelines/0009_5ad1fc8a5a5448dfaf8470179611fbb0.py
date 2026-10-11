import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Group matching for Cabin Deck & Side
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
combined_df["Cabin_Deck"] = combined_df["Cabin_Deck"].fillna(
    combined_df["Group"].map(group_deck)
)
combined_df["Cabin_Side"] = combined_df["Cabin_Side"].fillna(
    combined_df["Group"].map(group_side)
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

# Spatial refinements: deck-normalized percentile and localized cluster bins
combined_df["Cabin_Percentile"] = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].rank(pct=True).fillna(0.5)
)
combined_df["Cabin_Cluster"] = (combined_df["Cabin_Num"] // 100).fillna(-1).astype(int)
combined_df["Cabin_Region"] = (combined_df["Cabin_Num"] // 300).fillna(-1).astype(int)

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
]

# Tree-optimized label encoding for categorical attributes
for col in cat_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

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
    "Cabin_Region",
    "Cabin_Cluster",
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
# 4. HETEROGENEOUS MODEL DEFINITIONS (GBDT & TABULAR MLP)
# ---------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, cat_cardinalities, num_continuous, dropout=0.25):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=min(50, max(4, (c + 1) // 2)))
            for c in cat_cardinalities
        ])
        total_embed_dim = sum(min(50, max(4, (c + 1) // 2)) for c in cat_cardinalities)
        in_dim = total_embed_dim + num_continuous

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat, x_num):
        if len(self.embeddings) > 0 and x_cat is not None:
            emb_outs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
            x_emb = torch.cat(emb_outs, dim=1)
            x = torch.cat([x_emb, x_num], dim=1)
        else:
            x = x_num
        return self.mlp(x).squeeze(-1)


# LightGBM: leaf-wise expansion with feature & data subsampling and regularization
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

# CatBoost: symmetric oblivious trees with Bernoulli subsampling and L2 regularization
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
    verbose=False,
)

# ---------------------------------------------------------
# 5. MULTI-MODEL TRAINING WITH EARLY STOPPING
# ---------------------------------------------------------
print("Training heterogeneous model ensemble (GBDTs & TabularMLP)...")

# Train LightGBM
print("Training LightGBM model...")
try:
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
except TypeError:
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

# Train XGBoost
print("Training XGBoost model...")
try:
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
except TypeError:
    xgb_model_fallback = XGBClassifier(
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
    )
    xgb_model_fallback.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    xgb_model = xgb_model_fallback

# Train CatBoost
print("Training CatBoost model...")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

# Train TabularMLP
print("Training TabularMLP neural network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[num_cols].values)
X_val_num_scaled = scaler.transform(X_val[num_cols].values)
X_test_num_scaled = scaler.transform(X_test[num_cols].values)

X_train_cat_arr = X_train[cat_cols].values.astype(np.int64)
X_val_cat_arr = X_val[cat_cols].values.astype(np.int64)
X_test_cat_arr = X_test[cat_cols].values.astype(np.int64)

y_train_arr = y_train.values.astype(np.float32)
y_val_arr = y_val.values.astype(np.float32)

train_dataset = TensorDataset(
    torch.tensor(X_train_cat_arr, dtype=torch.long),
    torch.tensor(X_train_num_scaled, dtype=torch.float32),
    torch.tensor(y_train_arr, dtype=torch.float32),
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

val_cat_tensor = torch.tensor(X_val_cat_arr, dtype=torch.long).to(device)
val_num_tensor = torch.tensor(X_val_num_scaled, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(y_val_arr, dtype=torch.float32).to(device)

test_cat_tensor = torch.tensor(X_test_cat_arr, dtype=torch.long).to(device)
test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32).to(device)

cat_cardinalities = [int(combined_df[c].max() + 1) for c in cat_cols]
mlp_model = TabularMLP(
    cat_cardinalities=cat_cardinalities,
    num_continuous=len(num_cols),
    dropout=0.25,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

best_val_loss = float("inf")
best_mlp_weights = None
patience = 12
patience_counter = 0

for epoch in range(60):
    mlp_model.train()
    for batch_cat, batch_num, batch_y in train_loader:
        batch_cat = batch_cat.to(device)
        batch_num = batch_num.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = mlp_model(batch_cat, batch_num)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    scheduler.step()

    mlp_model.eval()
    with torch.no_grad():
        val_logits = mlp_model(val_cat_tensor, val_num_tensor)
        val_loss = criterion(val_logits, val_y_tensor).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_mlp_weights = copy.deepcopy(mlp_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_mlp_weights is not None:
    mlp_model.load_state_dict(best_mlp_weights)

# ---------------------------------------------------------
# 6. HOLD-OUT SOFT-VOTING BLENDING & THRESHOLD CALIBRATION
# ---------------------------------------------------------
# Validation probability extraction
lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
cat_val_probs = cat_model.predict_proba(X_val)[:, 1]

mlp_model.eval()
with torch.no_grad():
    mlp_val_probs = torch.sigmoid(mlp_model(val_cat_tensor, val_num_tensor)).cpu().numpy()
    mlp_test_probs = torch.sigmoid(mlp_model(test_cat_tensor, test_num_tensor)).cpu().numpy()

# Test probability extraction
lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
cat_test_probs = cat_model.predict_proba(X_test)[:, 1]

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
    f"TabularMLP Validation Accuracy: {accuracy_score(y_val_np, (mlp_val_probs >= 0.5).astype(int)):.6f}"
)

# Stack 4-model probabilities for fast vector search
val_prob_matrix = np.column_stack([lgb_val_probs, xgb_val_probs, cat_val_probs, mlp_val_probs])
test_prob_matrix = np.column_stack([lgb_test_probs, xgb_test_probs, cat_test_probs, mlp_test_probs])

# Generate simplex weight grid (sum = 1) across 4 models
weight_candidates = []
for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) * 20)) + 1):
        for w3 in np.linspace(0, 1 - w1 - w2, int(round((1 - w1 - w2) * 20)) + 1):
            w4 = max(0.0, 1.0 - w1 - w2 - w3)
            if abs(w1 + w2 + w3 + w4 - 1.0) < 1e-5:
                weight_candidates.append([w1, w2, w3, w4])
weights_arr = np.array(weight_candidates)

# Vectorized simplex prediction combinations
val_blends = np.dot(weights_arr, val_prob_matrix.T)

threshold_candidates = np.linspace(0.40, 0.60, 41)
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

print(
    f"Optimal Ensemble Blending Weights: LightGBM={best_weights[0]:.2f}, "
    f"XGBoost={best_weights[1]:.2f}, CatBoost={best_weights[2]:.2f}, "
    f"TabularMLP={best_weights[3]:.2f}"
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

import os
import copy
import random
import numpy as np
import pandas as pd
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

# Spatial refinements: deck-normalized percentile and localized cluster/zone bins
combined_df["Cabin_Percentile"] = (
    combined_df.groupby("Cabin_Deck")["Cabin_Num"].rank(pct=True).fillna(0.5)
)
deck_max_num = combined_df.groupby("Cabin_Deck")["Cabin_Num"].transform("max")
combined_df["Cabin_Relative_Pos"] = combined_df["Cabin_Num"] / (
    deck_max_num.replace(0, 1.0) + 1e-5
)
combined_df["Cabin_Zone"] = (combined_df["Cabin_Num"] // 25).astype(int)
combined_df["Cabin_Cluster"] = (combined_df["Cabin_Num"] // 100).astype(int)
combined_df["Cabin_Region"] = (combined_df["Cabin_Num"] // 300).astype(int)

combined_df["Cabin_Deck"] = combined_df["Cabin_Deck"].fillna("Unknown")
combined_df["Cabin_Side"] = combined_df["Cabin_Side"].fillna("Unknown")
combined_df["Deck_Side"] = combined_df["Cabin_Deck"] + "_" + combined_df["Cabin_Side"]
combined_df["Deck_Side_Zone"] = (
    combined_df["Deck_Side"] + "_" + combined_df["Cabin_Zone"].astype(str)
)
combined_df["Deck_Side_Compartment"] = (
    combined_df["Deck_Side"] + "_" + (combined_df["Cabin_Num"] // 20).astype(str)
)
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
    "Deck_Side_Zone",
    "Deck_Side_Compartment",
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
    "Cabin_Density",
    "Cabin_Num",
    "Cabin_Percentile",
    "Cabin_Relative_Pos",
    "Cabin_Region",
    "Cabin_Cluster",
    "Cabin_Zone",
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Deck_Side_Zone",
    "Deck_Side_Compartment",
    "Planet_Destination",
    "Group_Target",
    "Has_Group_Target",
    "Cabin_Target",
    "Has_Cabin_Target",
]

num_cols = [c for c in feature_cols if c not in cat_cols]

# Separate into Train and Test partitions
processed_train = combined_df.iloc[:train_len].copy()
processed_train["Transported"] = train_df["Transported"].values
processed_test = combined_df.iloc[train_len:].copy()

# Fit median imputers for numerical variables strictly on Train
for col in [c for c in num_cols if c in processed_train.columns]:
    med = processed_train[col].median()
    processed_train[col] = processed_train[col].fillna(med)
    processed_test[col] = processed_test[col].fillna(med)

# Pre-initialize cross-validation target encoding feature columns
for target_feat in ["Group_Target", "Has_Group_Target", "Cabin_Target", "Has_Cabin_Target"]:
    processed_train[target_feat] = 0.0
    processed_test[target_feat] = 0.0

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

# Calculate leak-free LOO target proportions on training split
tr_targets = processed_train.loc[tr_mask, "Transported"].values
global_mean = float(tr_targets.mean())

# 1. Multi-passenger group LOO target signal
tr_groups = processed_train.loc[tr_mask, "Group"].values
group_sums = pd.Series(tr_targets).groupby(tr_groups).transform("sum").values
group_counts = pd.Series(tr_targets).groupby(tr_groups).transform("count").values
has_group_tr = (group_counts > 1).astype(float)
loo_group_tr = np.where(
    has_group_tr == 1.0,
    (group_sums - tr_targets) / (group_counts - 1),
    global_mean,
)

# 2. Shared cabin LOO target signal
tr_cabins = processed_train.loc[tr_mask, "Cabin"].values
valid_cabins = pd.notnull(tr_cabins) & (tr_cabins != "")
cabin_sums = np.zeros(len(tr_targets), dtype=float)
cabin_counts = np.zeros(len(tr_targets), dtype=float)

valid_cabins_series = pd.Series(tr_cabins)
valid_idx = valid_cabins_series[valid_cabins].index
c_sum = (
    pd.Series(tr_targets[valid_cabins], index=valid_idx)
    .groupby(valid_cabins_series[valid_cabins])
    .transform("sum")
)
c_count = (
    pd.Series(tr_targets[valid_cabins], index=valid_idx)
    .groupby(valid_cabins_series[valid_cabins])
    .transform("count")
)
cabin_sums[valid_cabins] = c_sum.values
cabin_counts[valid_cabins] = c_count.values

has_cabin_tr = ((cabin_counts > 1) & valid_cabins).astype(float)
loo_cabin_tr = np.where(
    has_cabin_tr == 1.0,
    (cabin_sums - tr_targets) / np.maximum(cabin_counts - 1, 1),
    global_mean,
)

# Propagate training priors to validation and test
group_prior_stats = (
    pd.DataFrame({"Group": tr_groups, "target": tr_targets})
    .groupby("Group")["target"]
    .agg(["sum", "count"])
    .to_dict("index")
)

cabin_prior_stats = (
    pd.DataFrame(
        {"Cabin": tr_cabins[valid_cabins], "target": tr_targets[valid_cabins]}
    )
    .groupby("Cabin")["target"]
    .agg(["sum", "count"])
    .to_dict("index")
)


def map_priors(keys, stats_dict, default_val):
    target_arr = np.full(len(keys), default_val, dtype=float)
    has_flag = np.zeros(len(keys), dtype=float)
    for i, k in enumerate(keys):
        if pd.notnull(k) and k != "" and k in stats_dict:
            target_arr[i] = stats_dict[k]["sum"] / stats_dict[k]["count"]
            has_flag[i] = 1.0
    return target_arr, has_flag


val_groups = processed_train.loc[val_mask, "Group"].values
val_cabins = processed_train.loc[val_mask, "Cabin"].values
val_group_target, val_has_group = map_priors(
    val_groups, group_prior_stats, global_mean
)
val_cabin_target, val_has_cabin = map_priors(
    val_cabins, cabin_prior_stats, global_mean
)

test_groups = processed_test["Group"].values
test_cabins = processed_test["Cabin"].values
test_group_target, test_has_group = map_priors(
    test_groups, group_prior_stats, global_mean
)
test_cabin_target, test_has_cabin = map_priors(
    test_cabins, cabin_prior_stats, global_mean
)

processed_train.loc[tr_mask, "Group_Target"] = loo_group_tr
processed_train.loc[tr_mask, "Has_Group_Target"] = has_group_tr
processed_train.loc[tr_mask, "Cabin_Target"] = loo_cabin_tr
processed_train.loc[tr_mask, "Has_Cabin_Target"] = has_cabin_tr

processed_train.loc[val_mask, "Group_Target"] = val_group_target
processed_train.loc[val_mask, "Has_Group_Target"] = val_has_group
processed_train.loc[val_mask, "Cabin_Target"] = val_cabin_target
processed_train.loc[val_mask, "Has_Cabin_Target"] = val_has_cabin

processed_test["Group_Target"] = test_group_target
processed_test["Has_Group_Target"] = test_has_group
processed_test["Cabin_Target"] = test_cabin_target
processed_test["Has_Cabin_Target"] = test_has_cabin

X_train = processed_train.loc[tr_mask, feature_cols].reset_index(drop=True)
y_train = processed_train.loc[tr_mask, "Transported"].reset_index(drop=True)
X_val = processed_train.loc[val_mask, feature_cols].reset_index(drop=True)
y_val = processed_train.loc[val_mask, "Transported"].reset_index(drop=True)

X_test = processed_test[feature_cols].reset_index(drop=True)
test_ids = processed_test["PassengerId"].reset_index(drop=True)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 4. GBDT ENSEMBLE MODEL DEFINITIONS
# ---------------------------------------------------------
# LightGBM: leaf-wise expansion receiving clean category dtypes directly
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

# XGBoost: hist tree method with depth-wise regularization
try:
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
except TypeError:
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
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

# ---------------------------------------------------------
# 5. GBDT COMMITTEE TRAINING WITH EARLY STOPPING
# ---------------------------------------------------------
print("Training GBDT model ensemble (LightGBM, XGBoost, CatBoost)...")

# Train LightGBM with early stopping on fold 0 validation
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

# Prepare frequency-encoded and numerical feature matrices for XGBoost
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()

for col in cat_cols:
    s_tr = X_train[col].astype(str)
    s_val = X_val[col].astype(str)
    s_test = X_test[col].astype(str)

    freq_map = s_tr.value_counts(normalize=True).to_dict()
    X_train_xgb[f"{col}_Freq"] = s_tr.map(freq_map).fillna(0.0).astype(float)
    X_val_xgb[f"{col}_Freq"] = s_val.map(freq_map).fillna(0.0).astype(float)
    X_test_xgb[f"{col}_Freq"] = s_test.map(freq_map).fillna(0.0).astype(float)

    unique_cats = {val: idx for idx, val in enumerate(s_tr.unique())}
    X_train_xgb[col] = s_tr.map(unique_cats).fillna(-1.0).astype(float)
    X_val_xgb[col] = s_val.map(unique_cats).fillna(-1.0).astype(float)
    X_test_xgb[col] = s_test.map(unique_cats).fillna(-1.0).astype(float)

print("Training XGBoost model...")
try:
    xgb_model.fit(
        X_train_xgb,
        y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_model.fit(
        X_train_xgb,
        y_train,
        eval_set=[(X_val_xgb, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

# Prepare native categorical string features for CatBoost
X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

print("Training CatBoost model...")
cat_model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val), verbose=False)

# ---------------------------------------------------------
# 6. HOLD-OUT SOFT-VOTING BLENDING & THRESHOLD CALIBRATION
# ---------------------------------------------------------
# Validation probability extraction
lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val_xgb)[:, 1]
cat_val_probs = cat_model.predict_proba(X_val_cat)[:, 1]

# Test probability extraction
lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
cat_test_probs = cat_model.predict_proba(X_test_cat)[:, 1]

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

# Stack 3-model probabilities for bounded ensemble search
val_prob_matrix = np.column_stack([lgb_val_probs, xgb_val_probs, cat_val_probs])
test_prob_matrix = np.column_stack([lgb_test_probs, xgb_test_probs, cat_test_probs])

# Bounded convex probability blending search (w >= 0.10) to preserve diversity
weight_candidates = []
for w1_int in range(10, 81, 2):
    w1 = w1_int / 100.0
    for w2_int in range(10, 101 - w1_int - 10, 2):
        w2 = w2_int / 100.0
        w3 = round(1.0 - w1 - w2, 4)
        if w3 >= 0.10:
            weight_candidates.append([w1, w2, w3])
weight_candidates.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
weights_arr = np.array(weight_candidates)

# Vectorized simplex prediction combinations
val_blends = np.dot(weights_arr, val_prob_matrix.T)

# Decision threshold calibrated strictly in balanced prior interval [0.45, 0.55]
threshold_candidates = np.linspace(0.45, 0.55, 41)
best_score = 0.0
best_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
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
    f"XGBoost={best_weights[1]:.2f}, CatBoost={best_weights[2]:.2f}"
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

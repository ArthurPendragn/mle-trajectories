import os
import copy
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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
combined_df["Deck_Side_Zone"] = (
    combined_df["Deck_Side"] + "_" + combined_df["Cabin_Zone"].astype(str)
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
    "Deck_Side_Zone",
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
    "Cabin_Num",
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
    "Deck_Side_Zone",
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
        stats = processed_train.loc[inner_tr_idx].groupby(col)["Transported"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + 10.0 * prior) / (stats["count"] + 10.0)
        smoothed_map = smoothed.to_dict()

        processed_train.loc[inner_val_idx, f"{col}_TE"] = (
            processed_train.loc[inner_val_idx, col].astype(str).map(smoothed_map).fillna(prior).astype(float)
        )

    # 2. Map hold-out validation (val_mask) and test using entire tr_mask statistics
    tr_all_idx = processed_train[tr_mask].index
    y_tr_all = processed_train.loc[tr_all_idx, "Transported"]
    prior_all = y_tr_all.mean()
    stats_all = processed_train.loc[tr_all_idx].groupby(col)["Transported"].agg(["count", "mean"])
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
# 4. LEVEL-1 & LEVEL-2 ESTIMATOR DEFINITIONS
# ---------------------------------------------------------
# LightGBM: leaf-wise regularized gradient boosting
lgb_params = dict(
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
lgb_model = LGBMClassifier(**lgb_params)

# XGBoost: depth-wise regularized trees with target-encoded inputs
xgb_params = dict(
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
xgb_model = XGBClassifier(**xgb_params)

# CatBoost: symmetric oblivious trees with native categorical handling
cat_params = dict(
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
cat_model = CatBoostClassifier(**cat_params)

# Level-2 Stacking Meta-Learner: L2-regularized logistic regression
meta_learner = LogisticRegression(C=0.1, penalty="l2", random_state=42)

# ---------------------------------------------------------
# 5. INTERNAL CROSS-VALIDATION & OUT-OF-FOLD EXTRACTION
# ---------------------------------------------------------
print("Training Level-1 ensemble with internal 4-fold cross-validation...")
n_inner_splits = 4
inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

val_lgb_preds = np.zeros(len(X_val))
val_xgb_preds = np.zeros(len(X_val))
val_cat_preds = np.zeros(len(X_val))

test_lgb_preds = np.zeros(len(X_test))
test_xgb_preds = np.zeros(len(X_test))
test_cat_preds = np.zeros(len(X_test))

for fold_idx, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_train, y_train)):
    print(f"--- Internal Fold {fold_idx + 1}/{n_inner_splits} ---")

    X_tr_in = X_train.iloc[in_tr_idx].reset_index(drop=True)
    y_tr_in = y_train.iloc[in_tr_idx].reset_index(drop=True)
    X_va_in = X_train.iloc[in_val_idx].reset_index(drop=True)
    y_va_in = y_train.iloc[in_val_idx].reset_index(drop=True)

    X_tr_in_num = X_train_num.iloc[in_tr_idx].reset_index(drop=True)
    X_va_in_num = X_train_num.iloc[in_val_idx].reset_index(drop=True)

    # 1. LightGBM
    clf_lgb = LGBMClassifier(**lgb_params)
    clf_lgb.fit(
        X_tr_in,
        y_tr_in,
        eval_set=[(X_va_in, y_va_in)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[in_val_idx] = clf_lgb.predict_proba(X_va_in)[:, 1]
    val_lgb_preds += clf_lgb.predict_proba(X_val)[:, 1] / n_inner_splits
    test_lgb_preds += clf_lgb.predict_proba(X_test)[:, 1] / n_inner_splits

    # 2. XGBoost
    clf_xgb = XGBClassifier(**xgb_params)
    clf_xgb.fit(
        X_tr_in_num,
        y_tr_in,
        eval_set=[(X_va_in_num, y_va_in)],
        verbose=False,
    )
    oof_xgb[in_val_idx] = clf_xgb.predict_proba(X_va_in_num)[:, 1]
    val_xgb_preds += clf_xgb.predict_proba(X_val_num)[:, 1] / n_inner_splits
    test_xgb_preds += clf_xgb.predict_proba(X_test_num)[:, 1] / n_inner_splits

    # 3. CatBoost
    clf_cat = CatBoostClassifier(**cat_params)
    clf_cat.fit(
        X_tr_in,
        y_tr_in,
        eval_set=(X_va_in, y_va_in),
        verbose=False,
    )
    oof_cat[in_val_idx] = clf_cat.predict_proba(X_va_in)[:, 1]
    val_cat_preds += clf_cat.predict_proba(X_val)[:, 1] / n_inner_splits
    test_cat_preds += clf_cat.predict_proba(X_test)[:, 1] / n_inner_splits

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

# ---------------------------------------------------------
# 6. LEVEL-2 STACKING META-LEARNER TRAINING & CALIBRATION
# ---------------------------------------------------------
print("Fitting L2-regularized Stacking Meta-Learner...")
oof_prob_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
meta_learner.fit(oof_prob_matrix, y_train_np)

print(
    f"Meta-Learner Weights: LightGBM={meta_learner.coef_[0][0]:.4f}, "
    f"XGBoost={meta_learner.coef_[0][1]:.4f}, CatBoost={meta_learner.coef_[0][2]:.4f}, "
    f"Intercept={meta_learner.intercept_[0]:.4f}"
)

val_prob_matrix = np.column_stack([val_lgb_preds, val_xgb_preds, val_cat_preds])
test_prob_matrix = np.column_stack([test_lgb_preds, test_xgb_preds, test_cat_preds])

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

# Predict probabilities from meta-learner on hold-out validation set
val_meta_probs = meta_learner.predict_proba(val_prob_matrix)[:, 1]

# Calibrate decision threshold on untouched hold-out set within [0.45, 0.55]
threshold_candidates = np.linspace(0.45, 0.55, 101)
best_score = 0.0
best_threshold = 0.5

for th in threshold_candidates:
    preds = (val_meta_probs >= th).astype(int)
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
test_meta_probs = meta_learner.predict_proba(test_prob_matrix)[:, 1]
test_predictions = (test_meta_probs >= best_threshold).astype(bool)

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

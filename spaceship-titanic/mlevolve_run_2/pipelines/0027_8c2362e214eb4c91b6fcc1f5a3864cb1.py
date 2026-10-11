import copy
import os
import random
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------
# 0. Reproducibility Configuration
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# 1. Path Resolution & Data Ingestion
# -----------------------------------------------------------------------------
input_dir = "./input"
train_path = os.path.join(input_dir, "train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(input_dir, "input", "train.csv")

test_path = os.path.join(input_dir, "test.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(input_dir, "input", "test.csv")

os.makedirs("./working", exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

test_ids = df_test["PassengerId"].copy()
y = df_train["Transported"].astype(int).values

df_train_raw = df_train.drop(columns=["Transported"]).copy()
df_test_raw = df_test.copy()

n_train = len(df_train_raw)
combined = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)

# -----------------------------------------------------------------------------
# 2. Domain-Specific Feature Engineering & Physical Constraint Resolution
# -----------------------------------------------------------------------------
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Parse PassengerId into Group and Position
combined["Group"] = combined["PassengerId"].apply(lambda x: x.split("_")[0])
combined["GroupNum"] = combined["PassengerId"].apply(lambda x: int(x.split("_")[1]))

# Manifest-level group size
group_sizes = combined["Group"].value_counts().to_dict()
combined["GroupSize"] = combined["Group"].map(group_sizes)
combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
combined["GroupPosRatio"] = (combined["GroupNum"] / combined["GroupSize"]).astype(float)

# Extract Surname for family group identification
combined["Surname"] = combined["Name"].apply(
    lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
)
surname_counts = combined["Surname"].value_counts().to_dict()
combined["FamilySize"] = combined["Surname"].map(surname_counts).fillna(1)

# Deduce HomePlanet from group members if missing
group_homeplanet = combined.groupby("Group")["HomePlanet"].first().dropna().to_dict()
combined["HomePlanet"] = combined["HomePlanet"].fillna(
    combined["Group"].map(group_homeplanet)
)

# Extract Cabin components: Deck / Num / Side
combined["CabinDeck"] = combined["Cabin"].apply(
    lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan
)
combined["CabinNum"] = combined["Cabin"].apply(
    lambda x: float(str(x).split("/")[1]) if pd.notna(x) else np.nan
)
combined["CabinSide"] = combined["Cabin"].apply(
    lambda x: str(x).split("/")[2] if pd.notna(x) else np.nan
)

# Deduce CabinDeck and CabinSide from group members if missing
group_deck = combined.groupby("Group")["CabinDeck"].first().dropna().to_dict()
combined["CabinDeck"] = combined["CabinDeck"].fillna(combined["Group"].map(group_deck))

group_side = combined.groupby("Group")["CabinSide"].first().dropna().to_dict()
combined["CabinSide"] = combined["CabinSide"].fillna(combined["Group"].map(group_side))

# Deduce HomePlanet from CabinDeck domain invariants (Decks A, B, C, T to Europa; Deck G to Earth)
combined.loc[
    combined["HomePlanet"].isna() & combined["CabinDeck"].isin(["A", "B", "C", "T"]),
    "HomePlanet",
] = "Europa"
combined.loc[
    combined["HomePlanet"].isna() & (combined["CabinDeck"] == "G"), "HomePlanet"
] = "Earth"

# Deterministic physical constraint resolution:
# 1) If in CryoSleep, luxury spending is strictly 0
for col in spend_cols:
    combined.loc[combined["CryoSleep"] == True, col] = combined.loc[
        combined["CryoSleep"] == True, col
    ].fillna(0.0)

# 2) If total spend > 0, CryoSleep must be False
known_spend_sum = combined[spend_cols].fillna(0).sum(axis=1)
combined.loc[(combined["CryoSleep"].isna()) & (known_spend_sum > 0), "CryoSleep"] = (
    False
)

# 3) Children (Age < 13) cannot spend on amenities
child_mask = combined["Age"] < 13
for col in spend_cols:
    combined.loc[child_mask, col] = combined.loc[child_mask, col].fillna(0.0)

# Impute remaining missing values for spend columns with 0.0 (baseline non-spenders)
for col in spend_cols:
    combined[col] = combined[col].fillna(0.0)

# Deduce CryoSleep from group consensus first before fallback
group_cryo = (
    combined.groupby("Group")["CryoSleep"]
    .agg(lambda s: s.mode().iloc[0] if not s.dropna().empty else np.nan)
    .dropna()
    .to_dict()
)
combined["CryoSleep"] = combined["CryoSleep"].fillna(combined["Group"].map(group_cryo))
combined["CryoSleep"] = (combined["CryoSleep"] == True).astype(int)
combined["VIP"] = (combined["VIP"] == True).astype(int)

# Expenditure aggregation and non-linear ratios
combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
combined["ZeroSpend"] = (combined["TotalSpend"] == 0).astype(int)
combined["LuxurySpend"] = combined["RoomService"] + combined["Spa"] + combined["VRDeck"]
combined["ZeroLuxurySpend"] = (combined["LuxurySpend"] == 0).astype(int)
combined["BasicSpend"] = combined["FoodCourt"] + combined["ShoppingMall"]
combined["AmenityCount"] = (combined[spend_cols] > 0).sum(axis=1)
combined["Luxury_to_Total"] = combined["LuxurySpend"] / (combined["TotalSpend"] + 1.0)

for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
    combined[f"Log_{col}"] = np.log1p(combined[col])

for col in spend_cols:
    combined[f"Pct_{col}"] = combined[col] / (combined["TotalSpend"] + 1.0)

# Cabin spatial regions and localized blocks
combined["CabinRegion"] = (combined["CabinNum"] // 300).fillna(-1)
combined["CabinBlock"] = (combined["CabinNum"] // 50).fillna(-1)
combined["CabinParity"] = (combined["CabinNum"] % 2).fillna(-1)

# Deck-relative normalized cabin positions
deck_max = combined.groupby("CabinDeck")["CabinNum"].transform("max")
combined["CabinDeck_RelPos"] = (combined["CabinNum"] / deck_max).fillna(0.0)

# Route interaction & CabinDeck_CabinSide interaction
combined["HomePlanet"] = combined["HomePlanet"].fillna("Missing")
combined["Destination"] = combined["Destination"].fillna("Missing")
combined["Route"] = combined["HomePlanet"] + "_" + combined["Destination"]
combined["CabinDeck_CabinSide"] = (
    combined["CabinDeck"].fillna("Missing").astype(str)
    + "_"
    + combined["CabinSide"].fillna("Missing").astype(str)
)
combined["CryoSleep_CabinSide"] = (
    combined["CryoSleep"].astype(str)
    + "_"
    + combined["CabinSide"].fillna("Missing").astype(str)
)
combined["HomePlanet_CabinDeck"] = (
    combined["HomePlanet"].fillna("Missing").astype(str)
    + "_"
    + combined["CabinDeck"].fillna("Missing").astype(str)
)

# Uncontaminated leave-one-out peer statistics
group_cryo_sum = combined.groupby("Group")["CryoSleep"].transform("sum")
group_spend_sum = combined.groupby("Group")["TotalSpend"].transform("sum")
has_peers = combined["GroupSize"] > 1

combined["Peer_CryoSleep_Rate"] = 0.0
combined.loc[has_peers, "Peer_CryoSleep_Rate"] = (
    (group_cryo_sum[has_peers] - combined.loc[has_peers, "CryoSleep"])
    / (combined.loc[has_peers, "GroupSize"] - 1)
)

combined["Peer_TotalSpend"] = 0.0
combined.loc[has_peers, "Peer_TotalSpend"] = (
    (group_spend_sum[has_peers] - combined.loc[has_peers, "TotalSpend"])
    / (combined.loc[has_peers, "GroupSize"] - 1)
)
combined["Log_Peer_TotalSpend"] = np.log1p(combined["Peer_TotalSpend"])

# Age lifecycle buckets
combined["AgeGroup"] = pd.cut(
    combined["Age"].fillna(-1),
    bins=[-2, 0, 12, 18, 30, 50, 70, 100],
    labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
).astype(str)

# -----------------------------------------------------------------------------
# 3. GBDT Feature Preparation: Ordinal Encoding & Column Sanitization
# -----------------------------------------------------------------------------
categorical_cols = [
    "HomePlanet",
    "Destination",
    "CabinDeck",
    "CabinSide",
    "CabinDeck_CabinSide",
    "CryoSleep_CabinSide",
    "HomePlanet_CabinDeck",
    "Route",
    "AgeGroup",
    "CryoSleep",
    "VIP",
    "ZeroSpend",
    "ZeroLuxurySpend",
    "IsAlone",
    "CabinBlock",
    "CabinParity",
]

for cat_col in categorical_cols:
    combined[cat_col] = combined[cat_col].fillna("Missing").astype(str)

drop_cols = ["PassengerId", "Cabin", "Name", "Group", "Surname"]
combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])

# Encode categorical variables with OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined[categorical_cols] = encoder.fit_transform(combined[categorical_cols])

# Sanitize column names: replace non-alphanumeric characters with underscores
combined.columns = [re.sub(r"[^\w]", "_", c) for c in combined.columns]

# Split back to train and test
train_features = combined.iloc[:n_train].copy().reset_index(drop=True)
test_features = combined.iloc[n_train:].copy().reset_index(drop=True)

# Retain unscaled numeric features, imputing missing numerical values with train medians
numerical_cols = [c for c in train_features.columns if c not in categorical_cols]
num_medians = train_features[numerical_cols].median().to_dict()
train_features[numerical_cols] = train_features[numerical_cols].fillna(num_medians)
test_features[numerical_cols] = test_features[numerical_cols].fillna(num_medians)

# -----------------------------------------------------------------------------
# 4. 5-Fold Cross-Validation, Heterogeneous GBDT Ensemble & Soft Voting
# -----------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds_ensemble = np.zeros(len(train_features))
oof_preds_lgb = np.zeros(len(train_features))
oof_preds_xgb = np.zeros(len(train_features))
oof_preds_cb = np.zeros(len(train_features))

test_preds_total = np.zeros(len(test_features))
fold_0_acc = None

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
    X_train_f, y_train_f = train_features.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = train_features.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM (leaf-wise growth)
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        max_depth=6,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Model 2: XGBoost (depth-wise growth)
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )

    # Model 3: CatBoost (symmetric oblivious trees)
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_state=SEED + fold,
        verbose=0,
    )
    cb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        verbose=0,
    )

    # Validation predictions per base estimator
    p_lgb_val = lgb_model.predict_proba(X_val_f)[:, 1]
    p_xgb_val = xgb_model.predict_proba(X_val_f)[:, 1]
    p_cb_val = cb_model.predict_proba(X_val_f)[:, 1]

    # Equal-weight soft-voting probability ensemble
    p_ens_val = (p_lgb_val + p_xgb_val + p_cb_val) / 3.0

    oof_preds_lgb[val_idx] = p_lgb_val
    oof_preds_xgb[val_idx] = p_xgb_val
    oof_preds_cb[val_idx] = p_cb_val
    oof_preds_ensemble[val_idx] = p_ens_val

    fold_acc = accuracy_score(y_val_f, (p_ens_val >= 0.5).astype(int))
    print(
        f"Fold {fold} | LGB: {accuracy_score(y_val_f, (p_lgb_val >= 0.5).astype(int)):.4f} "
        f"| XGB: {accuracy_score(y_val_f, (p_xgb_val >= 0.5).astype(int)):.4f} "
        f"| CB: {accuracy_score(y_val_f, (p_cb_val >= 0.5).astype(int)):.4f} "
        f"| Ensemble Val Acc: {fold_acc:.4f}"
    )

    if fold == 0:
        fold_0_acc = fold_acc

    # Test set probability inference and fold accumulation
    p_lgb_test = lgb_model.predict_proba(test_features)[:, 1]
    p_xgb_test = xgb_model.predict_proba(test_features)[:, 1]
    p_cb_test = cb_model.predict_proba(test_features)[:, 1]

    fold_ens_test = (p_lgb_test + p_xgb_test + p_cb_test) / 3.0
    test_preds_total += fold_ens_test / 5.0

# Overall Out-of-Fold Evaluation
oof_acc = accuracy_score(y, (oof_preds_ensemble >= 0.5).astype(int))
print(f"OOF Validation Accuracy: {oof_acc:.4f}")
print(f"Fold 0 Validation Accuracy (Benchmark): {fold_0_acc:.4f}")

# -----------------------------------------------------------------------------
# 5. Submission Generation & Output Validation
# -----------------------------------------------------------------------------
test_preds = (test_preds_total >= 0.5).astype(bool)

os.makedirs("./submission", exist_ok=True)
sub = pd.DataFrame(
    {"PassengerId": test_ids.values, "Transported": test_preds}
)
sub_path = "./submission/submission.csv"
sub.to_csv(sub_path, index=False)

# Verify submission integrity
assert os.path.exists(sub_path), "Submission file does not exist!"
assert len(sub) == 4277, f"Expected 4277 rows, got {len(sub)}"
assert list(sub.columns) == ["PassengerId", "Transported"], f"Unexpected columns: {sub.columns}"
assert sub["Transported"].dtype == bool, f"Expected bool dtype, got {sub['Transported'].dtype}"
assert sub.isna().sum().sum() == 0, "Submission contains NaN values!"

print(f"Final Validation Score: {fold_0_acc}")
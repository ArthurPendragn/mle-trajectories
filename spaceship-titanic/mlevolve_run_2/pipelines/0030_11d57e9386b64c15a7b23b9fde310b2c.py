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

# Track raw missing status for CryoSleep before deductive imputation
combined["CryoSleep_IsMissing"] = combined["CryoSleep"].isna().astype(int)

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

# Engineer high-signal domain interaction features
combined["CryoSleep_ZeroSpend"] = (
    (combined["CryoSleep"] == 1) & (combined["ZeroSpend"] == 1)
).astype(int)
combined["NoSpendingAdult"] = (
    (combined["Age"].fillna(-1) >= 18)
    & (combined["ZeroSpend"] == 1)
    & (combined["CryoSleep"] == 0)
).astype(int)
combined["Spend_Spread"] = combined[spend_cols].std(axis=1).fillna(0.0)

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
    "CryoSleep_ZeroSpend",
    "NoSpendingAdult",
    "CryoSleep_IsMissing",
]

for cat_col in categorical_cols:
    combined[cat_col] = combined[cat_col].fillna("Missing").astype(str)

drop_cols = ["PassengerId", "Cabin", "Name", "Group", "Surname"]
combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])

# Encode categorical variables with OrdinalEncoder as contiguous non-negative integers
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined[categorical_cols] = encoder.fit_transform(combined[categorical_cols]).astype(int)

# Sanitize column names: replace non-alphanumeric characters with underscores
combined.columns = [re.sub(r"[^\w]", "_", c) for c in combined.columns]
categorical_cols = [re.sub(r"[^\w]", "_", c) for c in categorical_cols]

# Split back to train and test
train_features = combined.iloc[:n_train].copy().reset_index(drop=True)
test_features = combined.iloc[n_train:].copy().reset_index(drop=True)

# Retain unscaled numeric features, imputing missing numerical values with train medians
numerical_cols = [c for c in train_features.columns if c not in categorical_cols]
num_medians = train_features[numerical_cols].median().to_dict()
train_features[numerical_cols] = train_features[numerical_cols].fillna(num_medians)
test_features[numerical_cols] = test_features[numerical_cols].fillna(num_medians)

# Explicitly ensure categorical columns are contiguous integers for LightGBM/XGBoost
for c in categorical_cols:
    train_features[c] = train_features[c].astype(int)
    test_features[c] = test_features[c].astype(int)

# Dual-representation: preserve discrete string copies for CatBoost native categorical handling
train_features_cb = train_features.copy()
test_features_cb = test_features.copy()
for c in categorical_cols:
    train_features_cb[c] = train_features_cb[c].astype(str)
    test_features_cb[c] = test_features_cb[c].astype(str)

# -----------------------------------------------------------------------------
# 4. Two-Stage Semi-Supervised Learning & Threshold-Calibrated Ensemble
# -----------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ----------------- Stage 1: Initial Cross-Validation & Test Probability Estimation -----------------
print("=== Stage 1: Base Cross-Validation with Native Categorical Ensembles ===")
stage1_test_preds = np.zeros(len(test_features))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
    X_train_f, y_train_f = train_features.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = train_features.iloc[val_idx], y[val_idx]

    X_train_f_cb = train_features_cb.iloc[train_idx]
    X_val_f_cb = train_features_cb.iloc[val_idx]

    # Model 1: LightGBM with native categorical features and histogram splits
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.75,
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
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Model 2: XGBoost with histogram tree method
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.75,
        subsample=0.8,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
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

    # Model 3: CatBoost with native string cat_features and L2 regularization
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        cat_features=categorical_cols,
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_state=SEED + fold,
        verbose=0,
    )
    cb_model.fit(
        X_train_f_cb,
        y_train_f,
        eval_set=(X_val_f_cb, y_val_f),
        verbose=0,
    )

    p_lgb_test = lgb_model.predict_proba(test_features)[:, 1]
    p_xgb_test = xgb_model.predict_proba(test_features)[:, 1]
    p_cb_test = cb_model.predict_proba(test_features_cb)[:, 1]

    fold_test_ens = (p_lgb_test + p_xgb_test + p_cb_test) / 3.0
    stage1_test_preds += fold_test_ens / 5.0

# ----------------- Stage 1 Filtering: High-Confidence Pseudo-Label Extraction -----------------
high_conf_pos = stage1_test_preds >= 0.95
high_conf_neg = stage1_test_preds <= 0.05
high_conf_mask = high_conf_pos | high_conf_neg
pseudo_indices = np.where(high_conf_mask)[0]

pseudo_X = test_features.iloc[pseudo_indices].copy().reset_index(drop=True)
pseudo_X_cb = test_features_cb.iloc[pseudo_indices].copy().reset_index(drop=True)
pseudo_y = np.where(high_conf_pos[pseudo_indices], 1, 0)

print(
    f"Stage 1 Filter: Extracted {len(pseudo_y)} high-confidence pseudo-labeled test instances "
    f"({int(high_conf_pos.sum())} positive, {int(high_conf_neg.sum())} negative)"
)

# ----------------- Stage 2: Pseudo-Label Augmented Retraining & Untouched Holdout Evaluation -----------------
print("=== Stage 2: Semi-Supervised Retraining on Augmented Manifests ===")
oof_preds_lgb = np.zeros(len(train_features))
oof_preds_xgb = np.zeros(len(train_features))
oof_preds_cb = np.zeros(len(train_features))

stage2_test_lgb = np.zeros(len(test_features))
stage2_test_xgb = np.zeros(len(test_features))
stage2_test_cb = np.zeros(len(test_features))

val_idx_fold0 = None

for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
    if fold == 0:
        val_idx_fold0 = val_idx

    X_train_orig, y_train_orig = train_features.iloc[train_idx], y[train_idx]
    X_train_orig_cb = train_features_cb.iloc[train_idx]

    # Strictly untouched holdout validation fold
    X_val_f, y_val_f = train_features.iloc[val_idx], y[val_idx]
    X_val_f_cb = train_features_cb.iloc[val_idx]

    # Augment fold training data with pseudo-labeled test samples
    if len(pseudo_y) > 0:
        X_train_f = pd.concat([X_train_orig, pseudo_X], axis=0, ignore_index=True)
        X_train_f_cb = pd.concat([X_train_orig_cb, pseudo_X_cb], axis=0, ignore_index=True)
        y_train_f = np.concatenate([y_train_orig, pseudo_y], axis=0)
    else:
        X_train_f = X_train_orig
        X_train_f_cb = X_train_orig_cb
        y_train_f = y_train_orig

    # Stage 2 LightGBM
    lgb_model_2 = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.75,
        subsample=0.8,
        subsample_freq=1,
        max_depth=6,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold + 100,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model_2.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Stage 2 XGBoost
    xgb_model_2 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        colsample_bytree=0.75,
        subsample=0.8,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=SEED + fold + 100,
        n_jobs=-1,
    )
    xgb_model_2.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )

    # Stage 2 CatBoost
    cb_model_2 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        cat_features=categorical_cols,
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_state=SEED + fold + 100,
        verbose=0,
    )
    cb_model_2.fit(
        X_train_f_cb,
        y_train_f,
        eval_set=(X_val_f_cb, y_val_f),
        verbose=0,
    )

    p_lgb_val = lgb_model_2.predict_proba(X_val_f)[:, 1]
    p_xgb_val = xgb_model_2.predict_proba(X_val_f)[:, 1]
    p_cb_val = cb_model_2.predict_proba(X_val_f_cb)[:, 1]

    oof_preds_lgb[val_idx] = p_lgb_val
    oof_preds_xgb[val_idx] = p_xgb_val
    oof_preds_cb[val_idx] = p_cb_val

    p_ens_val = (p_lgb_val + p_xgb_val + p_cb_val) / 3.0
    fold_acc = accuracy_score(y_val_f, (p_ens_val >= 0.5).astype(int))
    print(
        f"Stage 2 Fold {fold} | LGB: {accuracy_score(y_val_f, (p_lgb_val >= 0.5).astype(int)):.4f} "
        f"| XGB: {accuracy_score(y_val_f, (p_xgb_val >= 0.5).astype(int)):.4f} "
        f"| CB: {accuracy_score(y_val_f, (p_cb_val >= 0.5).astype(int)):.4f} "
        f"| Equal Val Acc: {fold_acc:.4f}"
    )

    stage2_test_lgb += lgb_model_2.predict_proba(test_features)[:, 1] / 5.0
    stage2_test_xgb += xgb_model_2.predict_proba(test_features)[:, 1] / 5.0
    stage2_test_cb += cb_model_2.predict_proba(test_features_cb)[:, 1] / 5.0

# ----------------- Optimal Blend Weight Search & Decision Threshold Calibration -----------------
print("=== Simplex Weight Search & Decision Threshold Calibration ===")
best_score = -1.0
best_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
best_threshold = 0.5

weight_candidates = []
for w1 in np.linspace(0.0, 1.0, 21):
    for w2 in np.linspace(0.0, 1.0 - w1, 21):
        w3 = round(1.0 - w1 - w2, 4)
        if w3 >= -1e-6:
            weight_candidates.append((round(w1, 2), round(w2, 2), max(0.0, round(w3, 2))))

threshold_candidates = np.linspace(0.44, 0.56, 25)

for w_lgb, w_xgb, w_cb in weight_candidates:
    blend_oof = w_lgb * oof_preds_lgb + w_xgb * oof_preds_xgb + w_cb * oof_preds_cb
    for th in threshold_candidates:
        score = accuracy_score(y, (blend_oof >= th).astype(int))
        if score > best_score:
            best_score = score
            best_weights = (w_lgb, w_xgb, w_cb)
            best_threshold = th

print(f"Optimal Blend Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
print(f"Optimal Decision Threshold: {best_threshold:.4f}")
print(f"Calibrated OOF Validation Accuracy: {best_score:.4f}")

# Calculate calibrated Fold 0 validation accuracy benchmark
p_val_fold0 = (
    best_weights[0] * oof_preds_lgb[val_idx_fold0]
    + best_weights[1] * oof_preds_xgb[val_idx_fold0]
    + best_weights[2] * oof_preds_cb[val_idx_fold0]
)
fold_0_acc = accuracy_score(y[val_idx_fold0], (p_val_fold0 >= best_threshold).astype(int))
print(f"Fold 0 Validation Accuracy (Benchmark): {fold_0_acc:.4f}")

# -----------------------------------------------------------------------------
# 5. Submission Generation & Output Validation
# -----------------------------------------------------------------------------
test_preds_total = (
    best_weights[0] * stage2_test_lgb
    + best_weights[1] * stage2_test_xgb
    + best_weights[2] * stage2_test_cb
)
test_preds = (test_preds_total >= best_threshold).astype(bool)

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
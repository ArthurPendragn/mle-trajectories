import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# 2. Locate and load raw data robustly
def get_data_path(filename):
    for candidate in [
        f"./input/{filename}",
        f"./input/input/{filename}",
        f"./{filename}",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in ./input or ./input/input")


train_path = get_data_path("train.csv")
test_path = get_data_path("test.csv")

df_train_raw = pd.read_csv(train_path)
df_test_raw = pd.read_csv(test_path)

# 3. Extract manifest-level relational mappings (Group and Family counts)
all_pids = pd.concat(
    [df_train_raw["PassengerId"], df_test_raw["PassengerId"]], ignore_index=True
)
all_groups = all_pids.str.split("_").str[0]
group_counts = all_groups.value_counts().to_dict()

all_names = pd.concat(
    [df_train_raw["Name"].dropna(), df_test_raw["Name"].dropna()], ignore_index=True
)
all_surnames = all_names.str.split().str[-1]
surname_counts = all_surnames.value_counts().to_dict()

# 4. Manifest Relational Imputation & Domain Feature Extraction
df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

# Passenger ID structural features
df_all["Group"] = df_all["PassengerId"].str.split("_").str[0]
df_all["Group_Num"] = df_all["PassengerId"].str.split("_").str[1].astype(int)
df_all["Group_Size"] = df_all["Group"].map(group_counts).fillna(1).astype(int)
df_all["Is_Alone"] = (df_all["Group_Size"] == 1).astype(int)
df_all["Surname"] = df_all["Name"].str.split().str[-1]
df_all["Family_Size"] = df_all["Surname"].map(surname_counts).fillna(1).astype(int)

# Cabin decomposition
cabin_split = df_all["Cabin"].str.split("/", expand=True)
df_all["Cabin_Deck"] = cabin_split[0]
df_all["Cabin_Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
df_all["Cabin_Side"] = cabin_split[2]

# Amenity spending missingness indicator flags
amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in amenity_cols:
    df_all[f"{col}_Missing"] = df_all[col].isna().astype(float)

# Physical consistency: spending > 0 implies passenger cannot be in CryoSleep
has_spending = (df_all[amenity_cols] > 0).any(axis=1)
df_all.loc[has_spending, "CryoSleep"] = False

# Physical consistency: decks A/B/C/T belong to Europa, deck G belongs to Earth
df_all.loc[
    df_all["Cabin_Deck"].isin(["A", "B", "C", "T"]) & df_all["HomePlanet"].isna(),
    "HomePlanet",
] = "Europa"
df_all.loc[(df_all["Cabin_Deck"] == "G") & df_all["HomePlanet"].isna(), "HomePlanet"] = (
    "Earth"
)

# Relational propagation within identical travel groups (near-perfect fidelity)
group_hp = df_all.dropna(subset=["HomePlanet"]).groupby("Group")["HomePlanet"].first()
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(df_all["Group"].map(group_hp))

group_deck = (
    df_all.dropna(subset=["Cabin_Deck"]).groupby("Group")["Cabin_Deck"].first()
)
df_all["Cabin_Deck"] = df_all["Cabin_Deck"].fillna(df_all["Group"].map(group_deck))

group_side = (
    df_all.dropna(subset=["Cabin_Side"]).groupby("Group")["Cabin_Side"].first()
)
df_all["Cabin_Side"] = df_all["Cabin_Side"].fillna(df_all["Group"].map(group_side))

group_cnum = df_all.dropna(subset=["Cabin_Num"]).groupby("Group")["Cabin_Num"].first()
df_all["Cabin_Num"] = df_all["Cabin_Num"].fillna(df_all["Group"].map(group_cnum))

# Relational propagation within identical family surnames
surname_hp = (
    df_all.dropna(subset=["HomePlanet"])
    .groupby("Surname")["HomePlanet"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(df_all["Surname"].map(surname_hp))

surname_deck = (
    df_all.dropna(subset=["Cabin_Deck"])
    .groupby("Surname")["Cabin_Deck"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["Cabin_Deck"] = df_all["Cabin_Deck"].fillna(df_all["Surname"].map(surname_deck))

surname_side = (
    df_all.dropna(subset=["Cabin_Side"])
    .groupby("Surname")["Cabin_Side"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["Cabin_Side"] = df_all["Cabin_Side"].fillna(df_all["Surname"].map(surname_side))

# Re-apply physical deck-planet deduction after deck imputation
df_all.loc[
    df_all["Cabin_Deck"].isin(["A", "B", "C", "T"]) & df_all["HomePlanet"].isna(),
    "HomePlanet",
] = "Europa"
df_all.loc[(df_all["Cabin_Deck"] == "G") & df_all["HomePlanet"].isna(), "HomePlanet"] = (
    "Earth"
)

# Physical VIP consistency rules: Earth passengers and minors under 18 are never VIP
df_all.loc[df_all["HomePlanet"] == "Earth", "VIP"] = False
df_all.loc[df_all["Age"] < 18, "VIP"] = False

# Reconstruct full Cabin before computing occupancy and sharing features
valid_reconstructed_cabin = (
    df_all["Cabin_Deck"].notna()
    & df_all["Cabin_Num"].notna()
    & df_all["Cabin_Side"].notna()
)
df_all.loc[valid_reconstructed_cabin, "Cabin"] = (
    df_all.loc[valid_reconstructed_cabin, "Cabin_Deck"]
    + "/"
    + df_all.loc[valid_reconstructed_cabin, "Cabin_Num"].astype(int).astype(str)
    + "/"
    + df_all.loc[valid_reconstructed_cabin, "Cabin_Side"]
)

# Spatial deck and side interaction
df_all["Cabin_Deck_Side"] = df_all.apply(
    lambda r: (
        f"{r['Cabin_Deck']}_{r['Cabin_Side']}"
        if pd.notna(r["Cabin_Deck"]) and pd.notna(r["Cabin_Side"])
        else np.nan
    ),
    axis=1,
)

# Spatial cabin zone binning (longitudinal regions)
df_all["Cabin_Region"] = (df_all["Cabin_Num"] // 300).apply(
    lambda v: f"Region_{int(v)}" if pd.notna(v) else np.nan
)

# Domain rules: Passengers in CryoSleep or children under 13 spend nothing
for col in amenity_cols:
    df_all.loc[df_all["CryoSleep"] == True, col] = 0.0
    df_all.loc[df_all["Age"] < 13, col] = 0.0

# Spending breakdown and interaction features
df_all["TotalSpend"] = df_all[amenity_cols].sum(axis=1, min_count=1).fillna(0.0)
df_all["ZeroSpend"] = (df_all["TotalSpend"] == 0).astype(float)
df_all["Amenity_Count"] = (df_all[amenity_cols] > 0).sum(axis=1).astype(float)

df_all["LuxurySpend"] = (
    df_all[["Spa", "VRDeck", "RoomService"]].sum(axis=1, min_count=1).fillna(0.0)
)
df_all["EssentialSpend"] = (
    df_all[["FoodCourt", "ShoppingMall"]].sum(axis=1, min_count=1).fillna(0.0)
)

for col in amenity_cols:
    df_all[f"{col}_Log"] = np.log1p(df_all[col].clip(lower=0))
    df_all[f"{col}_Ratio"] = (df_all[col] / (df_all["TotalSpend"] + 1.0)).fillna(0.0)

df_all["TotalSpend_Log"] = np.log1p(df_all["TotalSpend"].clip(lower=0))
df_all["LuxurySpend_Log"] = np.log1p(df_all["LuxurySpend"].clip(lower=0))
df_all["EssentialSpend_Log"] = np.log1p(df_all["EssentialSpend"].clip(lower=0))

df_all["LuxurySpend_Ratio"] = (
    df_all["LuxurySpend"] / (df_all["TotalSpend"] + 1.0)
).fillna(0.0)
df_all["EssentialSpend_Ratio"] = (
    df_all["EssentialSpend"] / (df_all["TotalSpend"] + 1.0)
).fillna(0.0)

# Missingness indicators for continuous variables
df_all["Age_Missing"] = df_all["Age"].isna().astype(float)
df_all["Cabin_Num_Missing"] = df_all["Cabin_Num"].isna().astype(float)

# Cabin occupancy and spatial sharing features on reconstructed Cabin
cabin_counts = df_all["Cabin"].dropna().value_counts().to_dict()
df_all["Cabin_Occupancy"] = df_all["Cabin"].map(cabin_counts).fillna(1.0).astype(float)
df_all["Is_Cabin_Shared"] = (df_all["Cabin_Occupancy"] > 1).astype(float)

# Group-level spending and CryoSleep aggregations
df_all["CryoSleep_Num"] = df_all["CryoSleep"].map({True: 1.0, False: 0.0})
cryo_global_mean = float(df_all["CryoSleep_Num"].mean())
if np.isnan(cryo_global_mean):
    cryo_global_mean = 0.5
df_all["Group_CryoSleep_Rate"] = (
    df_all.groupby("Group")["CryoSleep_Num"].transform("mean").fillna(cryo_global_mean)
)
df_all["Group_CryoSleep_Count"] = (
    df_all.groupby("Group")["CryoSleep_Num"].transform("sum").fillna(0.0)
)
df_all["Group_TotalSpend"] = (
    df_all.groupby("Group")["TotalSpend"].transform("sum").fillna(0.0)
)
df_all["Group_MeanSpend"] = (
    df_all.groupby("Group")["TotalSpend"].transform("mean").fillna(0.0)
)
df_all["Spend_To_Group_Mean_Ratio"] = (
    df_all["TotalSpend"] / (df_all["Group_MeanSpend"] + 1.0)
).fillna(0.0)
df_all["Spend_Diff_Group_Mean"] = (
    df_all["TotalSpend"] - df_all["Group_MeanSpend"]
).fillna(0.0)

# Age bins
df_all["Age_Group"] = pd.cut(
    df_all["Age"],
    bins=[-np.inf, 12, 18, 30, 50, np.inf],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
).astype(str)
df_all.loc[df_all["Age"].isna(), "Age_Group"] = np.nan

# Define Native Categoricals and Numerics
cat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Cabin_Deck",
    "Cabin_Side",
    "Cabin_Deck_Side",
    "Age_Group",
    "Cabin_Region",
]

num_cols = [
    "Age",
    "Age_Missing",
    "Group_Num",
    "Group_Size",
    "Is_Alone",
    "Family_Size",
    "Cabin_Num",
    "Cabin_Num_Missing",
    "Cabin_Occupancy",
    "Is_Cabin_Shared",
    "Group_TotalSpend",
    "Group_MeanSpend",
    "Group_CryoSleep_Rate",
    "Group_CryoSleep_Count",
    "Spend_To_Group_Mean_Ratio",
    "Spend_Diff_Group_Mean",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "ZeroSpend",
    "Amenity_Count",
    "LuxurySpend",
    "LuxurySpend_Ratio",
    "LuxurySpend_Log",
    "EssentialSpend",
    "EssentialSpend_Ratio",
    "EssentialSpend_Log",
    "TotalSpend_Log",
    "RoomService_Log",
    "FoodCourt_Log",
    "ShoppingMall_Log",
    "Spa_Log",
    "VRDeck_Log",
    "RoomService_Ratio",
    "FoodCourt_Ratio",
    "ShoppingMall_Ratio",
    "Spa_Ratio",
    "VRDeck_Ratio",
    "RoomService_Missing",
    "FoodCourt_Missing",
    "ShoppingMall_Missing",
    "Spa_Missing",
    "VRDeck_Missing",
]

# Convert all discrete categoricals into pandas category dtype
for col in cat_cols:
    df_all[col] = (
        df_all[col]
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown"})
        .astype("category")
    )

# Separate back into df_train_feat and df_test_feat
n_train = len(df_train_raw)
df_train_feat = df_all.iloc[:n_train].copy().reset_index(drop=True)
df_test_feat = df_all.iloc[n_train:].copy().reset_index(drop=True)

y = df_train_feat["Transported"].astype(int).values
test_ids = df_test_feat["PassengerId"].values

# Impute Numerics strictly using training medians
num_medians = df_train_feat[num_cols].median()
df_train_feat[num_cols] = df_train_feat[num_cols].fillna(num_medians).astype(np.float32)
df_test_feat[num_cols] = df_test_feat[num_cols].fillna(num_medians).astype(np.float32)

feature_cols = cat_cols + num_cols
X_train = df_train_feat[feature_cols].copy()
X_test = df_test_feat[feature_cols].copy()

# Dedicated formatted views for CatBoost and HistGradientBoosting
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

X_train_hgb = X_train.copy()
X_test_hgb = X_test.copy()
for col in cat_cols:
    X_train_hgb[col] = X_train_hgb[col].cat.codes.astype(np.int32)
    X_test_hgb[col] = X_test_hgb[col].cat.codes.astype(np.int32)

hgb_cat_mask = [c in cat_cols for c in feature_cols]


# 5. Heterogeneous Level-1 Model Constructors
def create_lgb():
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def create_cb():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=0.03,
        depth=6,
        subsample=0.8,
        iterations=1000,
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
        cat_features=cat_cols,
    )


def create_xgb():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.03,
        max_depth=5,
        colsample_bytree=0.8,
        subsample=0.8,
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
    )


def create_hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=1000,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=50,
        random_state=SEED,
        categorical_features=hgb_cat_mask,
    )


# 6. 5-Fold Stratified Stacking Cross-Validation Pipeline
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_hgb = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_hgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    X_tr_cb_f = X_train_cb.iloc[train_idx]
    X_va_cb_f = X_train_cb.iloc[val_idx]

    X_tr_hgb_f = X_train_hgb.iloc[train_idx]
    X_va_hgb_f = X_train_hgb.iloc[val_idx]

    # Model 1: Native-Categorical LightGBM
    model_lgb = create_lgb()
    try:
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    except TypeError:
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            early_stopping_rounds=50,
            verbose=False,
        )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    # Model 2: Native-Categorical CatBoost
    model_cb = create_cb()
    model_cb.fit(
        X_tr_cb_f,
        y_tr,
        eval_set=(X_va_cb_f, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = model_cb.predict_proba(X_va_cb_f)[:, 1]
    test_preds_cb += model_cb.predict_proba(X_test_cb)[:, 1] / n_splits

    # Model 3: Native Histogram XGBoost
    try:
        model_xgb = create_xgb()
        model_xgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
    except TypeError:
        model_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.03,
            max_depth=5,
            colsample_bytree=0.8,
            subsample=0.8,
            n_estimators=1000,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
        )
        model_xgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False,
        )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # Model 4: HistGradientBoosting
    model_hgb = create_hgb()
    model_hgb.fit(X_tr_hgb_f, y_tr)
    oof_hgb[val_idx] = model_hgb.predict_proba(X_va_hgb_f)[:, 1]
    test_preds_hgb += model_hgb.predict_proba(X_test_hgb)[:, 1] / n_splits

    fold_ens_probs = (
        oof_lgb[val_idx] + oof_cb[val_idx] + oof_xgb[val_idx] + oof_hgb[val_idx]
    ) / 4.0
    fold_ens_preds = (fold_ens_probs >= 0.5).astype(int)
    fold_acc = accuracy_score(y_va, fold_ens_preds)
    print(f"Fold {fold + 1}/{n_splits} Ensemble Validation Accuracy: {fold_acc:.4f}")

# 7. Level-2 Stacking Meta-Learner Fitting
oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb, oof_hgb])
test_matrix = np.column_stack([test_preds_lgb, test_preds_cb, test_preds_xgb, test_preds_hgb])

meta_model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=SEED)
meta_model.fit(oof_matrix, y)

oof_meta_probs = meta_model.predict_proba(oof_matrix)[:, 1]
test_meta_probs = meta_model.predict_proba(test_matrix)[:, 1]

print(f"OOF LightGBM Accuracy: {accuracy_score(y, (oof_lgb >= 0.5).astype(int)):.4f}")
print(f"OOF CatBoost Accuracy: {accuracy_score(y, (oof_cb >= 0.5).astype(int)):.4f}")
print(f"OOF XGBoost Accuracy:  {accuracy_score(y, (oof_xgb >= 0.5).astype(int)):.4f}")
print(f"OOF HistGB Accuracy:   {accuracy_score(y, (oof_hgb >= 0.5).astype(int)):.4f}")
print(f"Meta-Learner Weights:  {meta_model.coef_[0]}, Intercept: {meta_model.intercept_[0]:.4f}")
print(f"OOF Stacking (thresh=0.50) Accuracy: {accuracy_score(y, (oof_meta_probs >= 0.5).astype(int)):.4f}")

# 8. Decision Threshold Calibration on Out-Of-Fold Predictions
best_threshold = 0.50
best_val_acc = float(accuracy_score(y, (oof_meta_probs >= 0.50).astype(int)))

threshold_candidates = np.linspace(0.40, 0.60, 201)
for th in threshold_candidates:
    score = float(accuracy_score(y, (oof_meta_probs >= th).astype(int)))
    if score > best_val_acc:
        best_val_acc = score
        best_threshold = th

print(f"Calibrated Threshold: {best_threshold:.4f} with OOF Accuracy: {best_val_acc:.4f}")

# 9. Generate Calibrated Test Predictions and Save Submission
test_preds_bool = test_meta_probs >= best_threshold

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"PassengerId": test_ids, "Transported": test_preds_bool.astype(bool)}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

# Submission Integrity Assertions
assert len(submission_df) == len(
    df_test_feat
), f"Expected {len(df_test_feat)} rows, found {len(submission_df)}"
assert list(submission_df.columns) == [
    "PassengerId",
    "Transported",
], "Incorrect columns"
assert not submission_df["Transported"].isna().any(), "Found NaN in predictions"
assert submission_df["Transported"].dtype == bool, "Target must be boolean"

# 10. Final Validation Metric Output
print(f"Final Validation Score: {best_val_acc}")

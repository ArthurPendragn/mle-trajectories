import copy
import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Reproducibility & Setup
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def find_data_file(filename):
    for candidate in [
        os.path.join(".", "input", filename),
        os.path.join(".", "input", "input", filename),
        os.path.join("input", filename),
        os.path.join("input", "input", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {filename}")


train_path = find_data_file("train.csv")
test_path = find_data_file("test.csv")
sub_sample_path = find_data_file("sample_submission.csv")

raw_train = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_sample_path)

# ---------------------------------------------------------
# 2. Structural Parsing Across Static Voyage Manifest
# ---------------------------------------------------------
all_manifest = pd.concat([raw_train, raw_test], ignore_index=True)

all_manifest["GroupId"] = all_manifest["PassengerId"].apply(lambda x: x.split("_")[0])
all_manifest["GroupNum"] = all_manifest["PassengerId"].apply(
    lambda x: int(x.split("_")[1])
)
all_manifest["LastName"] = all_manifest["Name"].apply(
    lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
)

group_sizes = all_manifest["GroupId"].value_counts().to_dict()
family_sizes = all_manifest["LastName"].dropna().value_counts().to_dict()
cabin_occupancy = all_manifest["Cabin"].dropna().value_counts().to_dict()

all_manifest["Cabin_Deck"] = all_manifest["Cabin"].apply(
    lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan
)
all_manifest["Cabin_Num"] = pd.to_numeric(
    all_manifest["Cabin"].apply(
        lambda x: (
            str(x).split("/")[1]
            if pd.notna(x) and len(str(x).split("/")) > 1
            else np.nan
        )
    ),
    errors="coerce",
)
deck_max_cabin_num = all_manifest.groupby("Cabin_Deck")["Cabin_Num"].max().to_dict()


def parse_structural_features(df):
    d = df.copy()
    d["GroupId"] = d["PassengerId"].apply(lambda x: x.split("_")[0])
    d["GroupNum"] = d["PassengerId"].apply(lambda x: int(x.split("_")[1]))
    d["GroupSize"] = d["GroupId"].map(group_sizes).fillna(1).astype(int)
    d["IsAlone"] = (d["GroupSize"] == 1).astype(int)
    d["GroupPosRatio"] = d["GroupNum"] / d["GroupSize"]

    d["LastName"] = d["Name"].apply(
        lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
    )
    d["FamilySize"] = d["LastName"].map(family_sizes).fillna(1).astype(int)
    d["IsFamilyAlone"] = (d["FamilySize"] == 1).astype(int)

    # Cabin decomposition
    d["Cabin_Deck"] = d["Cabin"].apply(
        lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan
    )
    d["Cabin_Num"] = pd.to_numeric(
        d["Cabin"].apply(
            lambda x: (
                str(x).split("/")[1]
                if pd.notna(x) and len(str(x).split("/")) > 1
                else np.nan
            )
        ),
        errors="coerce",
    )
    d["Cabin_Side"] = d["Cabin"].apply(
        lambda x: (
            str(x).split("/")[2]
            if pd.notna(x) and len(str(x).split("/")) > 2
            else np.nan
        )
    )
    d["Cabin_Region"] = d["Cabin_Num"].apply(
        lambda x: int(x // 300) if pd.notna(x) else np.nan
    )
    d["Cabin_Count"] = d["Cabin"].map(cabin_occupancy).fillna(0).astype(int)
    d["Deck_Max_Num"] = d["Cabin_Deck"].map(deck_max_cabin_num).fillna(1.0).astype(float)
    d["Cabin_Pos_Ratio"] = (d["Cabin_Num"] / (d["Deck_Max_Num"] + 1e-5)).fillna(0.0)
    return d


parsed_train = parse_structural_features(raw_train)
parsed_test = parse_structural_features(raw_test)

# ---------------------------------------------------------
# 3. Dataset Configuration & Relational Knowledge Mappings
# ---------------------------------------------------------
train_df = parsed_train.copy().reset_index(drop=True)
test_df = parsed_test.copy().reset_index(drop=True)

# Leak-free relational knowledge mappings derived strictly from training data
train_known_planet = train_df.dropna(subset=["HomePlanet"])
group_planet_map = train_known_planet.groupby("GroupId")["HomePlanet"].first().to_dict()
family_planet_map = (
    train_known_planet.dropna(subset=["LastName"])
    .groupby("LastName")["HomePlanet"]
    .first()
    .to_dict()
)

train_known_deck = train_df.dropna(subset=["Cabin_Deck"])
group_deck_map = train_known_deck.groupby("GroupId")["Cabin_Deck"].first().to_dict()

train_known_side = train_df.dropna(subset=["Cabin_Side"])
group_side_map = train_known_side.groupby("GroupId")["Cabin_Side"].first().to_dict()

train_median_cabin_num = train_df["Cabin_Num"].median()
train_median_age = train_df["Age"].median()

deck_spend_map = (
    train_df.groupby("Cabin_Deck")[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ]
    .sum()
    .sum(axis=1)
    .to_dict()
)

# ---------------------------------------------------------
# 5. Domain Invariant Resolution & Feature Engineering
# ---------------------------------------------------------
amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def apply_domain_resolution_and_features(df):
    d = df.copy()

    # Physical Invariant 1: Suspended passengers have zero expenditure
    cryo_mask = d["CryoSleep"] == True
    for col in amenities:
        d.loc[cryo_mask & d[col].isna(), col] = 0.0

    # Physical Invariant 2: Active spenders cannot be in cryo-sleep
    partial_spend = d[amenities].fillna(0).sum(axis=1)
    has_spend = partial_spend > 0
    d.loc[has_spend & d["CryoSleep"].isna(), "CryoSleep"] = False

    # Physical Invariant 3: Children under 13 have zero luxury amenity expenditures
    child_mask = d["Age"] < 13
    for col in amenities:
        d.loc[child_mask & d[col].isna(), col] = 0.0

    # Physical Invariant 4: Earth passengers are never VIP
    earth_mask = d["HomePlanet"] == "Earth"
    d.loc[earth_mask & d["VIP"].isna(), "VIP"] = False

    # Physical Invariant 5: Minors are never VIP
    minor_mask = d["Age"] < 18
    d.loc[minor_mask & d["VIP"].isna(), "VIP"] = False

    # Physical Invariant 6: Specific cabin decks map to origin planets
    d.loc[
        d["Cabin_Deck"].isin(["A", "B", "C", "T"]) & d["HomePlanet"].isna(),
        "HomePlanet",
    ] = "Europa"
    d.loc[(d["Cabin_Deck"] == "G") & d["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Relational Knowledge Deduction via Group and Family mappings
    local_group_planet = (
        d.dropna(subset=["HomePlanet"])
        .groupby("GroupId")["HomePlanet"]
        .first()
        .to_dict()
    )
    d["HomePlanet"] = d["HomePlanet"].fillna(d["GroupId"].map(local_group_planet))
    d["HomePlanet"] = d["HomePlanet"].fillna(d["GroupId"].map(group_planet_map))
    d["HomePlanet"] = d["HomePlanet"].fillna(d["LastName"].map(family_planet_map))

    local_group_side = (
        d.dropna(subset=["Cabin_Side"])
        .groupby("GroupId")["Cabin_Side"]
        .first()
        .to_dict()
    )
    d["Cabin_Side"] = d["Cabin_Side"].fillna(d["GroupId"].map(local_group_side))
    d["Cabin_Side"] = d["Cabin_Side"].fillna(d["GroupId"].map(group_side_map))

    local_group_deck = (
        d.dropna(subset=["Cabin_Deck"])
        .groupby("GroupId")["Cabin_Deck"]
        .first()
        .to_dict()
    )
    d["Cabin_Deck"] = d["Cabin_Deck"].fillna(d["GroupId"].map(local_group_deck))
    d["Cabin_Deck"] = d["Cabin_Deck"].fillna(d["GroupId"].map(group_deck_map))

    for col in amenities:
        d[col] = d[col].fillna(0.0)

    d["TotalSpend"] = d[amenities].sum(axis=1)
    d["HasSpent"] = (d["TotalSpend"] > 0).astype(int)
    d["LuxurySpend"] = d["Spa"] + d["VRDeck"] + d["RoomService"]
    d["SustenanceSpend"] = d["FoodCourt"] + d["ShoppingMall"]
    d["LuxuryRatio"] = d["LuxurySpend"] / (d["TotalSpend"] + 1.0)
    d["SustenanceRatio"] = d["SustenanceSpend"] / (d["TotalSpend"] + 1.0)

    for col in amenities:
        d[f"{col}_Ratio"] = d[col] / (d["TotalSpend"] + 1.0)
        d[f"Log_{col}"] = np.log1p(d[col])

    d["Log_TotalSpend"] = np.log1p(d["TotalSpend"])
    d["Log_LuxurySpend"] = np.log1p(d["LuxurySpend"])
    d["Log_SustenanceSpend"] = np.log1p(d["SustenanceSpend"])
    d["NumAmenitiesUsed"] = sum((d[col] > 0).astype(int) for col in amenities)
    d["SpendMax"] = d[amenities].max(axis=1)
    d["SpendStd"] = d[amenities].std(axis=1)

    d["Age_isna"] = d["Age"].isna().astype(int)
    d["Age"] = d["Age"].fillna(train_median_age)
    d["IsChild"] = (d["Age"] < 13).astype(int)
    d["IsTeen"] = ((d["Age"] >= 13) & (d["Age"] < 18)).astype(int)
    d["IsAdult"] = ((d["Age"] >= 18) & (d["Age"] < 60)).astype(int)
    d["IsSenior"] = (d["Age"] >= 60).astype(int)

    age_bins = [-1, 12, 17, 25, 35, 50, 65, 120]
    d["AgeGroup"] = pd.cut(
        d["Age"], bins=age_bins, labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(int)

    d["Cabin_Num_isna"] = d["Cabin_Num"].isna().astype(int)
    d["Cabin_Num"] = d["Cabin_Num"].fillna(train_median_cabin_num)
    d["Cabin_Region"] = d["Cabin_Num"] // 300

    deck_max = d["Cabin_Deck"].map(deck_max_cabin_num).fillna(1.0).astype(float)
    d["Cabin_Pos_Ratio"] = (d["Cabin_Num"] / (deck_max + 1e-5)).fillna(0.0)

    d["Deck_Side"] = d["Cabin_Deck"].astype(str) + "_" + d["Cabin_Side"].astype(str)
    d["Planet_Dest"] = d["HomePlanet"].astype(str) + "_" + d["Destination"].astype(str)
    d["Deck_Planet"] = d["Cabin_Deck"].astype(str) + "_" + d["HomePlanet"].astype(str)
    d["Cryo_Deck"] = d["CryoSleep"].astype(str) + "_" + d["Cabin_Deck"].astype(str)

    d["DeckSpendBenchmark"] = (
        d["Cabin_Deck"].map(deck_spend_map).fillna(0.0).astype(float)
    )

    categorical_cols_to_fill = [
        "HomePlanet",
        "Destination",
        "Cabin_Deck",
        "Cabin_Side",
    ]
    for col in categorical_cols_to_fill:
        d[col] = d[col].fillna("Unknown")

    d["CryoSleep"] = d["CryoSleep"].fillna(False).astype(int)
    d["VIP"] = d["VIP"].fillna(False).astype(int)

    # Group-level relational aggregations and zero-spend amenity counts
    d["ZeroSpendCount"] = sum((d[col] == 0.0).astype(int) for col in amenities)
    d["Group_TotalSpend"] = d.groupby("GroupId")["TotalSpend"].transform("sum")
    d["Group_CryoFraction"] = d.groupby("GroupId")["CryoSleep"].transform("mean")
    d["Group_CryoStd"] = d.groupby("GroupId")["CryoSleep"].transform("std").fillna(0.0)
    d["Group_CryoUniform"] = (
        (d["Group_CryoFraction"] == 0.0) | (d["Group_CryoFraction"] == 1.0)
    ).astype(int)
    d["Passenger_To_Group_SpendRatio"] = d["TotalSpend"] / (d["Group_TotalSpend"] + 1.0)
    d["Group_SpendRatio"] = d["Passenger_To_Group_SpendRatio"]

    return d


X_train_df = apply_domain_resolution_and_features(train_df)
X_test_df = apply_domain_resolution_and_features(test_df)

y = train_df["Transported"].astype(int).values

cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Dest",
    "Deck_Planet",
    "Cryo_Deck",
]

drop_cols = ["PassengerId", "Cabin", "Name", "GroupId", "LastName"]
if "Transported" in X_train_df.columns:
    drop_cols.append("Transported")

feature_cols = [c for c in X_train_df.columns if c not in drop_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

cat_cardinalities = []
for col in cat_cols:
    unique_cats = sorted(
        list(
            set(
                X_train_df[col].astype(str).tolist()
                + X_test_df[col].astype(str).tolist()
            )
        )
    )
    cat_to_idx = {cat_val: idx for idx, cat_val in enumerate(unique_cats)}
    X_train_df[col] = X_train_df[col].astype(str).map(cat_to_idx).fillna(0).astype(np.int64)
    X_test_df[col] = X_test_df[col].astype(str).map(cat_to_idx).fillna(0).astype(np.int64)
    cat_cardinalities.append(len(unique_cats))

for col in num_cols:
    X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce").fillna(0.0).astype(np.float32)
    X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce").fillna(0.0).astype(np.float32)


# ---------------------------------------------------------
# 6. Unscaled Feature Matrices for Tree Ensembles
# ---------------------------------------------------------
X = X_train_df[feature_cols].copy()
X_test = X_test_df[feature_cols].copy()
X_train = X.copy()
y_train = y.copy()

# ---------------------------------------------------------
# 7. Heterogeneous Tripartite Ensemble & Stratified 5-Fold CV
# ---------------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X), dtype=np.float64)
test_preds = np.zeros(len(X_test), dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM Classifier
    lgb_model = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        n_estimators=1000,
        verbose=-1,
    )
    try:
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    except Exception:
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            categorical_feature=cat_cols,
            early_stopping_rounds=50,
            verbose=False,
        )

    # Model 2: XGBoost Classifier
    xgb_model = XGBClassifier(
        max_depth=5,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=SEED + fold,
        n_estimators=1000,
    )
    try:
        xgb_model.set_params(early_stopping_rounds=50)
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
    except Exception:
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False,
        )

    # Model 3: CatBoost Classifier
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=4.0,
        eval_metric="Logloss",
        random_seed=SEED + fold,
        verbose=False,
        early_stopping_rounds=50,
        cat_features=cat_cols,
    )
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )

    # Fold validation predictions
    lgb_val_p = lgb_model.predict_proba(X_va)[:, 1]
    xgb_val_p = xgb_model.predict_proba(X_va)[:, 1]
    cat_val_p = cat_model.predict_proba(X_va)[:, 1]
    fold_val_p = (lgb_val_p + xgb_val_p + cat_val_p) / 3.0
    oof_preds[val_idx] = fold_val_p

    # Test predictions from all 15 estimators
    lgb_test_p = lgb_model.predict_proba(X_test)[:, 1]
    xgb_test_p = xgb_model.predict_proba(X_test)[:, 1]
    cat_test_p = cat_model.predict_proba(X_test)[:, 1]
    fold_test_p = (lgb_test_p + xgb_test_p + cat_test_p) / 3.0
    test_preds += fold_test_p / n_splits

# ---------------------------------------------------------
# 8. Out-of-Fold Threshold Calibration & Test Inference
# ---------------------------------------------------------
best_threshold = 0.5
best_val_score = -1.0
threshold_grid = np.linspace(0.45, 0.55, 101)
for th in threshold_grid:
    score = accuracy_score(y, (oof_preds >= th).astype(int))
    if score > best_val_score:
        best_val_score = score
        best_threshold = th

final_val_score = best_val_score

test_predictions = (test_preds >= best_threshold).astype(bool)

submission_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
)

submission_file = "./submission/submission.csv"
submission_df.to_csv(submission_file, index=False)

assert os.path.exists(submission_file), "Submission file does not exist"
assert len(submission_df) == len(
    raw_test
), f"Expected {len(raw_test)} rows, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "PassengerId",
    "Transported",
], "Incorrect column headers"
assert submission_df["Transported"].dtype == bool, "Transported must be boolean"
assert (
    submission_df["Transported"].isna().sum() == 0
), "NaN values detected in predictions"

print(f"Final Validation Score: {final_val_score:.5f}")

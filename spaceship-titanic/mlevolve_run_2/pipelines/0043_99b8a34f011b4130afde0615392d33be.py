import copy
import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    return d


parsed_train = parse_structural_features(raw_train)
parsed_test = parse_structural_features(raw_test)

# ---------------------------------------------------------
# 3. Strict Stratified Train / Validation Split
# ---------------------------------------------------------
y_all = parsed_train["Transported"].astype(int)
train_df, val_df = train_test_split(
    parsed_train, test_size=0.20, random_state=SEED, stratify=y_all
)
test_df = parsed_test.copy()

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ---------------------------------------------------------
# 4. Leak-Free Relational Knowledge Mappings (From Train Only)
# ---------------------------------------------------------
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
    d["SpendStd"] = d[amenities].std(axis=1).fillna(0.0)

    # Leak-free group spending aggregations
    d["Group_TotalSpend"] = d.groupby("GroupId")["TotalSpend"].transform("sum")
    d["Group_MeanSpend"] = d.groupby("GroupId")["TotalSpend"].transform("mean")
    d["Group_SpendRatio"] = d["TotalSpend"] / (d["Group_TotalSpend"] + 1.0)

    d["Age_isna"] = d["Age"].isna().astype(int)
    d["Age"] = d["Age"].fillna(train_median_age)
    d["Group_AgeMean"] = d.groupby("GroupId")["Age"].transform("mean")
    d["Group_AgeDiff"] = d["Age"] - d["Group_AgeMean"]

    d["Spend_Zero_Count"] = sum((d[col] == 0).astype(int) for col in amenities)
    d["Spend_to_Age_Ratio"] = d["TotalSpend"] / (d["Age"] + 1.0)

    d["IsChild"] = (d["Age"] < 13).astype(int)
    d["IsTeen"] = ((d["Age"] >= 13) & (d["Age"] < 18)).astype(int)
    d["IsAdult"] = ((d["Age"] >= 18) & (d["Age"] < 60)).astype(int)
    d["IsSenior"] = (d["Age"] >= 60).astype(int)

    age_bins = [-1, 12, 17, 25, 35, 50, 65, 120]
    d["AgeGroup"] = pd.cut(
        d["Age"], bins=age_bins, labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(int)

    # Spatial corridor and cabin decomposition features
    d["Cabin_Num_isna"] = d["Cabin_Num"].isna().astype(int)
    d["Cabin_Num"] = d["Cabin_Num"].fillna(train_median_cabin_num)
    d["Cabin_Region"] = (d["Cabin_Num"] // 300).astype(int)
    d["Cabin_Corridor"] = (d["Cabin_Num"] % 100).astype(int)

    d["Deck_Side"] = d["Cabin_Deck"].astype(str) + "_" + d["Cabin_Side"].astype(str)
    d["Planet_Dest"] = d["HomePlanet"].astype(str) + "_" + d["Destination"].astype(str)
    d["Deck_Planet"] = d["Cabin_Deck"].astype(str) + "_" + d["HomePlanet"].astype(str)
    d["Cryo_Deck"] = d["CryoSleep"].astype(str) + "_" + d["Cabin_Deck"].astype(str)
    d["Deck_Region"] = d["Cabin_Deck"].astype(str) + "_" + d["Cabin_Region"].astype(str)

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

    return d


X_train_df = apply_domain_resolution_and_features(train_df)
X_val_df = apply_domain_resolution_and_features(val_df)
X_test_df = apply_domain_resolution_and_features(test_df)

y_train = train_df["Transported"].astype(int).values
y_val = val_df["Transported"].astype(int).values

cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Dest",
    "Deck_Planet",
    "Cryo_Deck",
    "Deck_Region",
]

for col in cat_cols:
    le = LabelEncoder()
    all_vals = sorted(
        list(
            set(
                X_train_df[col].astype(str).tolist()
                + X_val_df[col].astype(str).tolist()
                + X_test_df[col].astype(str).tolist()
            )
        )
    )
    le.fit(all_vals)
    X_train_df[col] = le.transform(X_train_df[col].astype(str))
    X_val_df[col] = le.transform(X_val_df[col].astype(str))
    X_test_df[col] = le.transform(X_test_df[col].astype(str))

drop_cols = ["PassengerId", "Cabin", "Name", "GroupId", "LastName"]
if "Transported" in X_train_df.columns:
    drop_cols.append("Transported")

feature_cols = [c for c in X_train_df.columns if c not in drop_cols]
X_tr = X_train_df[feature_cols].copy()
X_va = X_val_df[feature_cols].copy()
X_te = X_test_df[feature_cols].copy()


# ---------------------------------------------------------
# 6. Complementary Dual GBDT Architecture (LightGBM + XGBoost)
# ---------------------------------------------------------
print("Training complementary LightGBM and XGBoost models...")

lgb_model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    objective="binary",
    verbose=-1,
    n_jobs=-1,
)

lgb_callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
lgb_model.fit(
    X_tr,
    y_train,
    eval_set=[(X_va, y_val)],
    eval_metric="binary_logloss",
    callbacks=lgb_callbacks,
)

xgb_model = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=SEED,
    eval_metric="logloss",
    early_stopping_rounds=50,
    n_jobs=-1,
)

xgb_model.fit(
    X_tr,
    y_train,
    eval_set=[(X_va, y_val)],
    verbose=False,
)

# ---------------------------------------------------------
# 7. Validation Probability Blending & Threshold Calibration
# ---------------------------------------------------------
lgb_val_probs = lgb_model.predict_proba(X_va)[:, 1]
lgb_test_probs = lgb_model.predict_proba(X_te)[:, 1]

xgb_val_probs = xgb_model.predict_proba(X_va)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_te)[:, 1]

lgb_val_acc = accuracy_score(y_val, (lgb_val_probs >= 0.5).astype(int))
xgb_val_acc = accuracy_score(y_val, (xgb_val_probs >= 0.5).astype(int))
print(f"LightGBM Validation Accuracy (th=0.5): {lgb_val_acc:.4f}")
print(f"XGBoost Validation Accuracy (th=0.5): {xgb_val_acc:.4f}")

weights = np.linspace(0.0, 1.0, 101)
thresholds = np.linspace(0.42, 0.58, 81)

best_score = -1.0
best_w = 0.5
best_th = 0.5

for w in weights:
    blend_val = w * lgb_val_probs + (1.0 - w) * xgb_val_probs
    preds_matrix = (blend_val[None, :] >= thresholds[:, None]).astype(int)
    scores = (preds_matrix == y_val[None, :]).mean(axis=1)
    max_idx = np.argmax(scores)
    if scores[max_idx] > best_score:
        best_score = float(scores[max_idx])
        best_w = float(w)
        best_th = float(thresholds[max_idx])

final_val_score = best_score
print(f"Optimal ensemble weight (LightGBM): {best_w:.2f}")
print(f"Optimal decision threshold: {best_th:.4f}")

# ---------------------------------------------------------
# 8. Test Inference & Submission Verification
# ---------------------------------------------------------
blend_test = best_w * lgb_test_probs + (1.0 - best_w) * xgb_test_probs
test_predictions = (blend_test >= best_th).astype(bool)

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

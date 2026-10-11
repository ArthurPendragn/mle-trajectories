import copy
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)


# ---------------------------------------------------------------------------
# Step 1: Data Processing & Feature Engineering
# ---------------------------------------------------------------------------
def locate_input_file(filename):
    """Search possible paths for the input file."""
    candidates = [
        os.path.join("./input", filename),
        os.path.join("./input/input", filename),
        os.path.join(".", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"File {filename} not found in expected locations.")


def extract_domain_features(df):
    """Extract deterministic structural and domain features prior to stateful imputation."""
    df = df.copy()

    # 1. Passenger group features from PassengerId (gggg_pp)
    passenger_parts = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = passenger_parts[0]
    df["GroupNum"] = passenger_parts[1].astype(int)

    # Group size calculation
    group_counts = df["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(group_counts)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # 2. Cabin structural decomposition: deck / num / side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0].fillna("Missing")
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2].fillna("Missing")

    # 3. Amenity expenditure features
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physics constraint: Confirmed CryoSleep passengers spend 0 on luxury amenities
    is_cryo = df["CryoSleep"] == True
    for col in amenity_cols:
        df.loc[is_cryo & df[col].isna(), col] = 0.0

    # Physics constraint: Children under 13 cannot spend on amenities
    is_child = df["Age"] < 13
    for col in amenity_cols:
        df.loc[is_child & df[col].isna(), col] = 0.0

    # Total and relative spending
    df["TotalSpending"] = df[amenity_cols].sum(axis=1, skipna=True)
    df["HasSpent"] = (df["TotalSpending"] > 0).astype(int)

    # Inverse physics constraint: Anyone who spent money was NOT in CryoSleep
    spent_money = df["TotalSpending"] > 0
    df.loc[spent_money & df["CryoSleep"].isna(), "CryoSleep"] = False

    # Proportional amenity spending
    for col in amenity_cols:
        df[f"{col}_Ratio"] = df[col] / (df["TotalSpending"] + 1.0)
        df[f"Log_{col}"] = np.log1p(np.maximum(df[col].fillna(0.0), 0.0))
    df["Log_TotalSpending"] = np.log1p(np.maximum(df["TotalSpending"], 0.0))

    # Number of amenities used
    df["AmenitiesUsedCount"] = (df[amenity_cols].fillna(0.0) > 0).sum(axis=1)

    # 4. Family surname extraction from Name
    surnames = df["Name"].str.split(" ", expand=True)[1]
    df["Surname"] = surnames.fillna("Unknown")
    surname_counts = df["Surname"].value_counts()
    df["FamilySize"] = df["Surname"].map(surname_counts)
    df.loc[df["Surname"] == "Unknown", "FamilySize"] = 1

    # 5. Deterministic HomePlanet deductive rules from Cabin Deck
    df.loc[
        df["CabinDeck"].isin(["A", "B", "C", "T"]) & df["HomePlanet"].isna(),
        "HomePlanet",
    ] = "Europa"
    df.loc[(df["CabinDeck"] == "G") & df["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    return df


def prepare_datasets():
    os.makedirs("./working", exist_ok=True)

    train_path = locate_input_file("train.csv")
    test_path = locate_input_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df[["PassengerId"]].copy()

    # Extract target from train
    y = train_df["Transported"].astype(int).values

    # Stratified hold-out split (80% train, 20% validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(train_df, y))

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Apply deterministic feature extraction
    X_train_raw = extract_domain_features(train_split)
    X_val_raw = extract_domain_features(val_split)
    X_test_raw = extract_domain_features(test_df)

    # Categorical and numerical columns definition
    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
    ]
    numeric_cols = [
        "Age",
        "CabinNum",
        "GroupSize",
        "GroupNum",
        "IsAlone",
        "FamilySize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpending",
        "HasSpent",
        "AmenitiesUsedCount",
        "Log_RoomService",
        "Log_FoodCourt",
        "Log_ShoppingMall",
        "Log_Spa",
        "Log_VRDeck",
        "Log_TotalSpending",
        "RoomService_Ratio",
        "FoodCourt_Ratio",
        "ShoppingMall_Ratio",
        "Spa_Ratio",
        "VRDeck_Ratio",
    ]

    # STRICT ISOLATION: Learn imputers strictly from training split
    num_medians = {col: X_train_raw[col].median() for col in numeric_cols}
    for col in numeric_cols:
        if pd.isna(num_medians[col]):
            num_medians[col] = 0.0

    cat_modes = {}
    for col in categorical_cols:
        mode_val = X_train_raw[col].mode(dropna=True)
        cat_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"

    def apply_imputation(df):
        df_imputed = df.copy()
        for col in numeric_cols:
            df_imputed[col] = df_imputed[col].fillna(num_medians[col])
        for col in categorical_cols:
            df_imputed[col] = df_imputed[col].fillna(cat_modes[col]).astype(str)
        return df_imputed

    X_train_imp = apply_imputation(X_train_raw)
    X_val_imp = apply_imputation(X_val_raw)
    X_test_imp = apply_imputation(X_test_raw)

    # One-hot encoding fitted strictly on train categories
    encoded_train_list = [X_train_imp[numeric_cols].reset_index(drop=True)]
    encoded_val_list = [X_val_imp[numeric_cols].reset_index(drop=True)]
    encoded_test_list = [X_test_imp[numeric_cols].reset_index(drop=True)]

    for col in categorical_cols:
        train_cats = sorted(X_train_imp[col].unique().tolist())
        for cat in train_cats:
            col_name = f"{col}_{cat}"
            encoded_train_list.append(
                pd.Series((X_train_imp[col] == cat).astype(float), name=col_name)
            )
            encoded_val_list.append(
                pd.Series((X_val_imp[col] == cat).astype(float), name=col_name)
            )
            encoded_test_list.append(
                pd.Series((X_test_imp[col] == cat).astype(float), name=col_name)
            )

    X_train_final = pd.concat(encoded_train_list, axis=1)
    X_val_final = pd.concat(encoded_val_list, axis=1)
    X_test_final = pd.concat(encoded_test_list, axis=1)

    # Standard scale continuous features (strictly fit on train)
    scaler = StandardScaler()
    continuous_features = [
        col for col in numeric_cols if col not in ["IsAlone", "HasSpent"]
    ]
    X_train_final[continuous_features] = scaler.fit_transform(
        X_train_final[continuous_features]
    )
    X_val_final[continuous_features] = scaler.transform(
        X_val_final[continuous_features]
    )
    X_test_final[continuous_features] = scaler.transform(
        X_test_final[continuous_features]
    )

    y_train_df = pd.DataFrame({"Transported": y_train})
    y_val_df = pd.DataFrame({"Transported": y_val})

    # Save prepared data to ./working
    X_train_final.to_csv("./working/X_train.csv", index=False)
    y_train_df.to_csv("./working/y_train.csv", index=False)
    X_val_final.to_csv("./working/X_val.csv", index=False)
    y_val_df.to_csv("./working/y_val.csv", index=False)
    X_test_final.to_csv("./working/X_test.csv", index=False)
    test_ids.to_csv("./working/test_ids.csv", index=False)

    return (
        X_train_final,
        y_train_df,
        X_val_final,
        y_val_df,
        X_test_final,
        test_ids,
    )


# ---------------------------------------------------------------------------
# Step 2: Model Design (Heterogeneous Gradient Boosted Ensemble)
# ---------------------------------------------------------------------------
def build_models():
    """Instantiate heterogeneous ensemble of gradient boosted tree classifiers."""
    return {
        "lgbm": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_samples=10,
            reg_lambda=1.5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=10,
            reg_lambda=1.5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "hist": HistGradientBoostingClassifier(
            max_iter=300,
            min_samples_leaf=10,
            l2_regularization=1.5,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Step 3: Training, Evaluation & Test Inference
# ---------------------------------------------------------------------------
def main():
    X_train_final, y_train_df, X_val_final, y_val_df, X_test_final, test_ids = (
        prepare_datasets()
    )

    # Sanitize feature column names by replacing spaces and non-alphanumeric characters with underscores
    clean_columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in X_train_final.columns
    ]
    X_train_final.columns = clean_columns
    X_val_final.columns = clean_columns
    X_test_final.columns = clean_columns

    models = build_models()
    val_probs_list = []
    test_probs_list = []

    y_train = y_train_df["Transported"]

    for name, model in models.items():
        model.fit(X_train_final, y_train)
        val_probs_list.append(model.predict_proba(X_val_final)[:, 1])
        test_probs_list.append(model.predict_proba(X_test_final)[:, 1])

    # Validation predictions via ensemble soft probability averaging
    val_probs = np.mean(val_probs_list, axis=0)
    val_preds = (val_probs >= 0.5).astype(bool)
    val_targets = y_val_df["Transported"].astype(bool).values

    final_score = accuracy_score(val_targets, val_preds)

    # Test predictions via ensemble soft probability averaging
    test_probs = np.mean(test_probs_list, axis=0)
    test_preds = (test_probs >= 0.5).astype(bool)

    # Generate submission file matching sample_submission format
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "PassengerId": test_ids["PassengerId"],
            "Transported": test_preds,
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()

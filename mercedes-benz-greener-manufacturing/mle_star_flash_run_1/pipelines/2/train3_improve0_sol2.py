import itertools
import os
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 1. Load Data
train_df = pd.read_csv("./input/train.csv")
test_path = "./input/test.csv"
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Separate features and target
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

X_test = None
if test_df is not None:
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )

# Identify categorical features
cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]

# 2. High-Signal Sparsity Aggregations & Categorical Pair Interactions
num_cols = [c for c in X.columns if c not in cat_features]

X_work_train = X.copy()
X_work_test = X_test.copy() if X_test is not None else None

# Row-wise sparsity and summary aggregations across numerical/binary features
if len(num_cols) > 0:
    train_num_mat = X_work_train[num_cols].fillna(0).values
    X_work_train["row_active_sum"] = np.sum(train_num_mat, axis=1)
    X_work_train["row_active_count"] = np.sum(train_num_mat > 0, axis=1)
    X_work_train["row_zero_count"] = np.sum(train_num_mat == 0, axis=1)
    X_work_train["row_mean"] = np.mean(train_num_mat, axis=1)
    X_work_train["row_std"] = np.std(train_num_mat, axis=1)

    if X_work_test is not None:
        test_num_mat = X_work_test[num_cols].fillna(0).values
        X_work_test["row_active_sum"] = np.sum(test_num_mat, axis=1)
        X_work_test["row_active_count"] = np.sum(test_num_mat > 0, axis=1)
        X_work_test["row_zero_count"] = np.sum(test_num_mat == 0, axis=1)
        X_work_test["row_mean"] = np.mean(test_num_mat, axis=1)
        X_work_test["row_std"] = np.std(test_num_mat, axis=1)

# Pair interactions for categorical features
cat_interaction_cols = []
if len(cat_features) >= 2:
    max_cat_pairs = min(len(cat_features), 5)
    selected_cats = cat_features[:max_cat_pairs]
    for col1, col2 in itertools.combinations(selected_cats, 2):
        inter_col = f"inter_{col1}_{col2}"
        cat_interaction_cols.append(inter_col)
        X_work_train[inter_col] = (
            X_work_train[col1].astype(str) + "_" + X_work_train[col2].astype(str)
        )
        if X_work_test is not None:
            X_work_test[inter_col] = (
                X_work_test[col1].astype(str) + "_" + X_work_test[col2].astype(str)
            )

all_cat_features = cat_features + cat_interaction_cols

# --- For CatBoost (Native Categorical Strings + Sparsity Statistics) ---
X_cb = X_work_train.copy()
for col in all_cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_work_test is not None:
    X_test_cb = X_work_test.copy()
    for col in all_cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Clean Label Encoding + Sparsity Statistics) ---
X_xgb = X_work_train.copy()
X_test_xgb = X_work_test.copy() if X_work_test is not None else None

for col in all_cat_features:
    le = LabelEncoder()
    if X_test_xgb is not None and col in X_test_xgb.columns:
        combined_vals = list(
            set(X_work_train[col].astype(str)).union(
                set(X_work_test[col].astype(str))
            )
        )
        le.fit(combined_vals)
        X_xgb[col] = le.transform(X_xgb[col].astype(str))
        X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    else:
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))

X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)
if X_test_xgb is not None:
    X_test_xgb = X_test_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train/Validation Split
X_train_cb, X_val_cb, y_train, y_val = train_test_split(
    X_cb, y, test_size=0.2, random_state=42
)
X_train_xgb, X_val_xgb, _, _ = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42
)

# 3. Model Training
# --- Train CatBoost ---
cb_model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    random_seed=42,
    cat_features=all_cat_features,
    eval_metric="RMSE",
    early_stopping_rounds=50,
    verbose=False,
)
cb_model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), verbose=False)

# --- Train XGBoost ---
dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
dval = xgb.DMatrix(X_val_xgb, label=y_val)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.008,
    "max_depth": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "alpha": 2.0,
    "lambda": 5.0,
    "random_state": 42,
    "tree_method": "hist",
}
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# 4. Ensemble Validation Predictions
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)
ensemble_val_preds = 0.5 * val_preds_cb + 0.5 * val_preds_xgb

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions
if X_test is not None:
    test_preds_cb = cb_model.predict(X_test_cb)
    dtest = xgb.DMatrix(X_test_xgb)
    test_preds_xgb = xgb_model.predict(dtest)
    final_test_preds = 0.5 * test_preds_cb + 0.5 * test_preds_xgb

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
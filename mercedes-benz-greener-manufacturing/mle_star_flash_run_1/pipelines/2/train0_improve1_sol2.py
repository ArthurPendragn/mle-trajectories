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

# 2. Feature Preprocessing
# --- For CatBoost (Native Categorical Strings) ---
X_cb = X.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = X_test.copy()
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded) ---
X_xgb = X.copy()
if X_test is not None:
    X_test_xgb = X_test.copy()
else:
    X_test_xgb = None

for col in cat_features:
    le = LabelEncoder()
    if X_test is not None and col in X_test.columns:
        combined_vals = list(
            set(X[col].astype(str)).union(set(X_test[col].astype(str)))
        )
        le.fit(combined_vals)
        X_xgb[col] = le.transform(X_xgb[col].astype(str))
        X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    else:
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))

X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)
if X_test_xgb is not None:
    X_test_xgb = X_test_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Feature Engineering Pipeline
# Convert to DataFrames for preprocessing
X_df = pd.DataFrame(X_xgb).copy()
X_test_df = (
    pd.DataFrame(X_test_xgb).copy()
    if (X_test is not None and X_test_xgb is not None)
    else None
)

# Eliminate zero-variance features
zero_var_cols = [col for col in X_df.columns if X_df[col].nunique() <= 1]
X_df.drop(columns=zero_var_cols, inplace=True)
if X_test_df is not None:
    X_test_df.drop(columns=zero_var_cols, inplace=True, errors="ignore")

# Eliminate duplicate features
dup_mask = X_df.T.duplicated()
dup_cols = X_df.columns[dup_mask].tolist()
X_df.drop(columns=dup_cols, inplace=True)
if X_test_df is not None:
    X_test_df.drop(columns=dup_cols, inplace=True, errors="ignore")

# Add row-wise summary statistics
for df in [X_df] + ([X_test_df] if X_test_df is not None else []):
    num_data = df.select_dtypes(include=[np.number])
    df["row_sum"] = num_data.sum(axis=1)
    df["row_mean"] = num_data.mean(axis=1)
    df["row_std"] = num_data.std(axis=1).fillna(0.0)

y_arr = np.array(y)
n_samples = len(X_df)
n_test = len(X_test_df) if X_test_df is not None else 0

# 2. 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_lgb = np.zeros(n_samples)
oof_ridge = np.zeros(n_samples)

if X_test_df is not None:
    test_preds_cb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    test_preds_lgb = np.zeros(n_test)
    test_preds_ridge = np.zeros(n_test)

# 3. Model Training across Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_arr)):
    X_tr, y_tr = X_df.iloc[train_idx], y_arr[train_idx]
    X_va, y_va = X_df.iloc[val_idx], y_arr[val_idx]

    # --- Train CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_va)
    if X_test_df is not None:
        test_preds_cb += cb_model.predict(X_test_df) / kf.n_splits

    # --- Train XGBoost ---
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.008,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "alpha": 2.0,
        "lambda": 5.0,
        "random_state": 42 + fold,
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
    oof_xgb[val_idx] = xgb_model.predict(dval)
    if X_test_df is not None:
        dtest = xgb.DMatrix(X_test_df)
        test_preds_xgb += xgb_model.predict(dtest) / kf.n_splits

    # --- Train LightGBM ---
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "random_state": 42 + fold,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    if X_test_df is not None:
        test_preds_lgb += (
            lgb_model.predict(X_test_df, num_iteration=lgb_model.best_iteration)
            / kf.n_splits
        )

    # --- Train Regularized Linear Model (RidgeCV) ---
    ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 50))
    ridge_model.fit(X_tr, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_va)
    if X_test_df is not None:
        test_preds_ridge += ridge_model.predict(X_test_df) / kf.n_splits

# 4. Meta-Learner Stacking (Non-Negative Linear Regression)
oof_matrix = np.column_stack([oof_cb, oof_xgb, oof_lgb, oof_ridge])
meta_learner = LinearRegression(positive=True)
meta_learner.fit(oof_matrix, y_arr)

final_oof_preds = meta_learner.predict(oof_matrix)
final_validation_score = r2_score(y_arr, final_oof_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions
if X_test_df is not None:
    test_matrix = np.column_stack(
        [test_preds_cb, test_preds_xgb, test_preds_lgb, test_preds_ridge]
    )
    final_test_preds = meta_learner.predict(test_matrix)

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
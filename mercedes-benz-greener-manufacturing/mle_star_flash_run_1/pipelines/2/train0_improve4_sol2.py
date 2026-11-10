import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import nnls

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
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

# --- For XGBoost / LightGBM / Ridge (Label Encoded) ---
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

# Compute row-level aggregation features
binary_cols = [
    col
    for col in X_xgb.columns
    if set(X_xgb[col].dropna().unique()).issubset({0, 1})
]
if len(binary_cols) > 0:
    X_xgb["row_binary_sum"] = (X_xgb[binary_cols] == 1).sum(axis=1)
    if X_test_xgb is not None:
        X_test_xgb["row_binary_sum"] = (X_test_xgb[binary_cols] == 1).sum(
            axis=1
        )

X_xgb["row_sparsity"] = (X_xgb == 0).sum(axis=1) / X_xgb.shape[1]
if X_test_xgb is not None:
    X_test_xgb["row_sparsity"] = (X_test_xgb == 0).sum(axis=1) / X_test_xgb.shape[1]

# Filter zero-variance columns across train dataset
non_zero_var_cols = [c for c in X_xgb.columns if X_xgb[c].nunique() > 1]
X_xgb_proc = X_xgb[non_zero_var_cols].copy()
if X_test_xgb is not None:
    X_test_xgb_proc = X_test_xgb[non_zero_var_cols].copy()

# 3. Outlier-Robust Target Transformation
y_arr = np.array(y, dtype=np.float64)
y_min = np.min(y_arr)
y_shift = 0.0 if y_min > 0 else (np.abs(y_min) + 1.0)
y_trans = np.log(y_arr + y_shift)
inv_transform = lambda preds: np.exp(preds) - y_shift

# 4. Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(len(y_arr))
oof_xgb = np.zeros(len(y_arr))
oof_lgb = np.zeros(len(y_arr))
oof_ridge = np.zeros(len(y_arr))

if X_test is not None:
    test_preds_cb = np.zeros(len(X_test_cb))
    test_preds_xgb = np.zeros(len(X_test_xgb_proc))
    test_preds_lgb = np.zeros(len(X_test_xgb_proc))
    test_preds_ridge = np.zeros(len(X_test_xgb_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_cb, y_trans)):
    # Slice data for CatBoost
    X_tr_cb = X_cb.iloc[train_idx]
    X_va_cb = X_cb.iloc[val_idx]

    # Slice data for XGB / LGB / Ridge
    X_tr_xgb = X_xgb_proc.iloc[train_idx]
    X_va_xgb = X_xgb_proc.iloc[val_idx]

    y_tr, y_va = y_trans[train_idx], y_trans[val_idx]

    # --- CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        cat_features=cat_features,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_va_cb)
    if X_test is not None:
        test_preds_cb += cb_model.predict(X_test_cb) / n_splits

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr)
    dval = xgb.DMatrix(X_va_xgb, label=y_va)
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
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
        num_boost_round=1800,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(dval)
    if X_test is not None:
        dtest = xgb.DMatrix(X_test_xgb_proc)
        test_preds_xgb += xgb_model.predict(dtest) / n_splits

    # --- LightGBM ---
    lgb_train = lgb.Dataset(X_tr_xgb, label=y_tr)
    lgb_val = lgb.Dataset(X_va_xgb, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.015,
        "num_leaves": 15,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "random_state": 42 + fold,
        "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1800,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_xgb)
    if X_test is not None:
        test_preds_lgb += lgb_model.predict(X_test_xgb_proc) / n_splits

    # --- Ridge Regression ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_xgb.fillna(0))
    X_va_scaled = scaler.transform(X_va_xgb.fillna(0))

    ridge_model = RidgeCV(alphas=np.logspace(-2, 4, 20))
    ridge_model.fit(X_tr_scaled, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_va_scaled)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test_xgb_proc.fillna(0))
        test_preds_ridge += ridge_model.predict(X_test_scaled) / n_splits

# 5. Invert Transformation on OOF and Test Predictions
oof_cb_inv = inv_transform(oof_cb)
oof_xgb_inv = inv_transform(oof_xgb)
oof_lgb_inv = inv_transform(oof_lgb)
oof_ridge_inv = inv_transform(oof_ridge)

# 6. Non-Negative Least Squares (NNLS) Blending
oof_matrix = np.column_stack([oof_cb_inv, oof_xgb_inv, oof_lgb_inv, oof_ridge_inv])
weights, _ = nnls(oof_matrix, y_arr)
if np.sum(weights) > 0:
    weights /= np.sum(weights)
else:
    weights = np.ones(4) / 4

# Compute Ensembled OOF Predictions and Clip to Empirical Bounds
oof_ensemble = np.dot(oof_matrix, weights)
oof_ensemble_clipped = np.clip(oof_ensemble, np.min(y_arr), np.max(y_arr))

final_validation_score = r2_score(y_arr, oof_ensemble_clipped)
print(f"Final Validation Performance: {final_validation_score}")

# 7. Generate Test Predictions & Save Submission
if X_test is not None:
    test_cb_inv = inv_transform(test_preds_cb)
    test_xgb_inv = inv_transform(test_preds_xgb)
    test_lgb_inv = inv_transform(test_preds_lgb)
    test_ridge_inv = inv_transform(test_preds_ridge)

    test_matrix = np.column_stack(
        [test_cb_inv, test_xgb_inv, test_lgb_inv, test_ridge_inv]
    )
    final_test_preds = np.dot(test_matrix, weights)
    final_test_preds = np.clip(final_test_preds, np.min(y_arr), np.max(y_arr))

    pred = final_test_preds
    predictions = final_test_preds

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
else:
    pred = oof_ensemble_clipped
    predictions = oof_ensemble_clipped
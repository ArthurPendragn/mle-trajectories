import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
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

# 3. Feature Engineering & Decomposition on Binary Features
binary_cols = [
    col
    for col in X_xgb.columns
    if X_xgb[col].nunique() <= 2 and col not in cat_features
]
cat_cols = list(cat_features) if cat_features is not None else []

# Decomposition on binary features
n_comp = min(10, len(binary_cols)) if len(binary_cols) > 10 else len(binary_cols)
if n_comp > 0:
    decomposers = {
        "pca": PCA(n_components=n_comp, random_state=42),
        "ica": FastICA(n_components=n_comp, random_state=42, max_iter=1000),
        "tsvd": TruncatedSVD(n_components=n_comp, random_state=42),
        "grp": GaussianRandomProjection(n_components=n_comp, random_state=42),
        "srp": SparseRandomProjection(
            n_components=n_comp, random_state=42, dense_output=True
        ),
    }

    decomp_train_list = []
    decomp_test_list = []
    for name, model in decomposers.items():
        train_decomp = model.fit_transform(X_xgb[binary_cols])
        decomp_train_list.append(
            pd.DataFrame(
                train_decomp,
                columns=[f"{name}_{i}" for i in range(n_comp)],
                index=X_xgb.index,
            )
        )
        if X_test is not None:
            test_decomp = model.transform(X_test_xgb[binary_cols])
            decomp_test_list.append(
                pd.DataFrame(
                    test_decomp,
                    columns=[f"{name}_{i}" for i in range(n_comp)],
                    index=X_test_xgb.index,
                )
            )

    decomp_train_df = pd.concat(decomp_train_list, axis=1)
    X_xgb_extended = pd.concat([X_xgb, decomp_train_df], axis=1)
    if X_test is not None:
        decomp_test_df = pd.concat(decomp_test_list, axis=1)
        X_test_xgb_extended = pd.concat([X_test_xgb, decomp_test_df], axis=1)
    else:
        X_test_xgb_extended = None
else:
    X_xgb_extended = X_xgb.copy()
    X_test_xgb_extended = X_test_xgb.copy() if X_test is not None else None

# 4. Setup 5-Fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF predictions and Test prediction arrays
oof_cb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_lasso = np.zeros(len(y))
oof_ridge = np.zeros(len(y))
oof_enet = np.zeros(len(y))

if X_test is not None:
    test_preds_cb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lasso = np.zeros(len(X_test))
    test_preds_ridge = np.zeros(len(X_test))
    test_preds_enet = np.zeros(len(X_test))

# 5. K-Fold Training with Target Encoding and Shallow Models
y_arr = np.array(y)
global_mean = np.mean(y_arr)
smooth_weight = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb_extended, y_arr)):
    X_tr_xgb = X_xgb_extended.iloc[train_idx].copy()
    X_va_xgb = X_xgb_extended.iloc[val_idx].copy()
    X_te_xgb = (
        X_test_xgb_extended.copy() if X_test_xgb_extended is not None else None
    )

    y_tr = y_arr[train_idx]
    y_va = y_arr[val_idx]

    for col in cat_cols:
        if col in X_tr_xgb.columns:
            stats = pd.DataFrame({"cat": X_tr_xgb[col], "target": y_tr})
            agg = stats.groupby("cat")["target"].agg(["count", "mean"])
            smooth_te = (agg["count"] * agg["mean"] + smooth_weight * global_mean) / (
                agg["count"] + smooth_weight
            )
            te_dict = smooth_te.to_dict()

            X_tr_xgb[f"{col}_te"] = X_tr_xgb[col].map(te_dict).fillna(global_mean)
            X_va_xgb[f"{col}_te"] = X_va_xgb[col].map(te_dict).fillna(global_mean)
            if X_te_xgb is not None:
                X_te_xgb[f"{col}_te"] = X_te_xgb[col].map(te_dict).fillna(global_mean)

    # CatBoost features
    X_tr_cb = X_cb.iloc[train_idx]
    X_va_cb = X_cb.iloc[val_idx]
    X_te_cb = X_test_cb if X_test is not None else None

    # Numeric features for Linear Models
    scaler = StandardScaler()
    num_cols = [c for c in X_tr_xgb.columns if c not in cat_cols]
    X_tr_dense = scaler.fit_transform(X_tr_xgb[num_cols].fillna(0))
    X_va_dense = scaler.transform(X_va_xgb[num_cols].fillna(0))
    if X_te_xgb is not None:
        X_te_dense = scaler.transform(X_te_xgb[num_cols].fillna(0))

    # --- Train CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.015,
        depth=3,
        l2_leaf_reg=6.0,
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
        test_preds_cb += cb_model.predict(X_te_cb) / n_splits

    # --- Train XGBoost ---
    xgb_features = [c for c in X_tr_xgb.columns if c not in cat_cols]
    dtrain = xgb.DMatrix(X_tr_xgb[xgb_features], label=y_tr)
    dval = xgb.DMatrix(X_va_xgb[xgb_features], label=y_va)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 2,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "alpha": 3.0,
        "lambda": 8.0,
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
    if X_test is not None:
        dtest = xgb.DMatrix(X_te_xgb[xgb_features])
        test_preds_xgb += xgb_model.predict(dtest) / n_splits

    # --- Train Regularized Linear Models ---
    # Lasso
    lasso = Lasso(alpha=0.05, max_iter=2000, random_state=42 + fold)
    lasso.fit(X_tr_dense, y_tr)
    oof_lasso[val_idx] = lasso.predict(X_va_dense)
    if X_test is not None:
        test_preds_lasso += lasso.predict(X_te_dense) / n_splits

    # Ridge
    ridge = Ridge(alpha=10.0, random_state=42 + fold)
    ridge.fit(X_tr_dense, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_va_dense)
    if X_test is not None:
        test_preds_ridge += ridge.predict(X_te_dense) / n_splits

    # ElasticNet
    enet = ElasticNet(
        alpha=0.05, l1_ratio=0.5, max_iter=2000, random_state=42 + fold
    )
    enet.fit(X_tr_dense, y_tr)
    oof_enet[val_idx] = enet.predict(X_va_dense)
    if X_test is not None:
        test_preds_enet += enet.predict(X_te_dense) / n_splits

# 6. Determine Optimal Ensemble Weights via Constrained Optimization
oof_matrix = np.column_stack(
    [oof_cb, oof_xgb, oof_lasso, oof_ridge, oof_enet]
)


def loss_func(weights):
    weights = np.array(weights)
    pred_vals = oof_matrix @ weights
    return -r2_score(y_arr, pred_vals)


initial_weights = np.ones(5) / 5.0
bounds = [(0, 1) for _ in range(5)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = opt_res.x

ensemble_oof_preds = oof_matrix @ optimal_weights
final_validation_score = r2_score(y_arr, ensemble_oof_preds)
print(f"Optimal Ensemble Weights (CB, XGB, Lasso, Ridge, ENet): {np.round(optimal_weights, 4)}")
print(f"Final Validation Performance: {final_validation_score}")

# 7. Generate Test Predictions and Export
if X_test is not None:
    test_matrix = np.column_stack(
        [
            test_preds_cb,
            test_preds_xgb,
            test_preds_lasso,
            test_preds_ridge,
            test_preds_enet,
        ]
    )
    final_test_preds = test_matrix @ optimal_weights
    pred = final_test_preds
    predictions = final_test_preds

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
else:
    pred = ensemble_oof_preds
    predictions = ensemble_oof_preds
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

from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

# 2. Feature Preprocessing & Decomposition

# Identify non-categorical / binary / numerical columns for latent space extraction
decomp_cols = [c for c in X.columns if c not in cat_features]
decomp_train_df = None
decomp_test_df = None

if len(decomp_cols) > 0:
    X_decomp_input = (
        X[decomp_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    if X_test is not None:
        X_test_decomp_input = (
            X_test[decomp_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        )
    else:
        X_test_decomp_input = None

    n_comp = min(6, len(decomp_cols))
    if n_comp >= 2:
        # Linear and non-linear latent decomposition methods
        pca = PCA(n_components=n_comp, random_state=42)
        pca_train = pca.fit_transform(X_decomp_input)
        pca_test = (
            pca.transform(X_test_decomp_input)
            if X_test_decomp_input is not None
            else None
        )

        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_train = svd.fit_transform(X_decomp_input)
        svd_test = (
            svd.transform(X_test_decomp_input)
            if X_test_decomp_input is not None
            else None
        )

        try:
            ica = FastICA(
                n_components=n_comp, random_state=42, max_iter=500, tol=1e-3
            )
            ica_train = ica.fit_transform(X_decomp_input)
            ica_test = (
                ica.transform(X_test_decomp_input)
                if X_test_decomp_input is not None
                else None
            )
        except Exception:
            ica_train, ica_test = None, None

        grp = GaussianRandomProjection(n_components=n_comp, random_state=42)
        grp_train = grp.fit_transform(X_decomp_input)
        grp_test = (
            grp.transform(X_test_decomp_input)
            if X_test_decomp_input is not None
            else None
        )

        decomp_dfs_train = []
        decomp_dfs_test = []

        for name, tr_arr, te_arr in [
            ("pca", pca_train, pca_test),
            ("svd", svd_train, svd_test),
            ("ica", ica_train, ica_test),
            ("grp", grp_train, grp_test),
        ]:
            if tr_arr is not None:
                cols = [f"{name}_{i}" for i in range(tr_arr.shape[1])]
                decomp_dfs_train.append(
                    pd.DataFrame(tr_arr, columns=cols, index=X.index)
                )
                if te_arr is not None:
                    decomp_dfs_test.append(
                        pd.DataFrame(te_arr, columns=cols, index=X_test.index)
                    )

        if decomp_dfs_train:
            decomp_train_df = pd.concat(decomp_dfs_train, axis=1)
        if decomp_dfs_test:
            decomp_test_df = pd.concat(decomp_dfs_test, axis=1)

# --- For CatBoost (Native Categorical Strings + Decomposed Latent Features) ---
X_cb = X.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = X_test.copy()
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

if decomp_train_df is not None:
    X_cb = pd.concat([X_cb, decomp_train_df], axis=1)
if decomp_test_df is not None and X_test_cb is not None:
    X_test_cb = pd.concat([X_test_cb, decomp_test_df], axis=1)

# --- For XGBoost (Label Encoded + Decomposed Latent Features) ---
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

if decomp_train_df is not None:
    X_xgb = pd.concat([X_xgb, decomp_train_df], axis=1)
if decomp_test_df is not None and X_test_xgb is not None:
    X_test_xgb = pd.concat([X_test_xgb, decomp_test_df], axis=1)

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
    subsample=0.8,
    random_seed=42,
    cat_features=cat_features,
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
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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# ==========================================
# 1. Strict Feature Hygiene
# ==========================================
# A. Remove zero-variance (constant) columns
constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=constant_cols, errors="ignore")
    cat_features = [c for c in cat_features if c not in constant_cols]

# B. Remove duplicate columns (including duplicate binary columns)
# Transpose sample or full dataframe to check duplicated column signatures
dup_mask = X.T.duplicated(keep="first")
duplicate_cols = X.columns[dup_mask].tolist()
if duplicate_cols:
    X = X.drop(columns=duplicate_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=duplicate_cols, errors="ignore")
    cat_features = [c for c in cat_features if c not in duplicate_cols]

num_cols = [c for c in X.columns if c not in cat_features]

# ==========================================
# 2. Row-wise Active Feature Counts
# ==========================================
df_counts_train = pd.DataFrame(index=X.index)
df_counts_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

if len(num_cols) > 0:
    df_counts_train["active_non_zero_count"] = (X[num_cols].fillna(0) != 0).sum(axis=1)
    df_counts_train["active_positive_count"] = (X[num_cols].fillna(0) > 0).sum(axis=1)
    if X_test is not None:
        df_counts_test["active_non_zero_count"] = (X_test[num_cols].fillna(0) != 0).sum(axis=1)
        df_counts_test["active_positive_count"] = (X_test[num_cols].fillna(0) > 0).sum(axis=1)

# ==========================================
# 3. Frequency & Out-Of-Fold Regularized Target Encodings
# ==========================================
df_enc_train = pd.DataFrame(index=X.index)
df_enc_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

# Frequency Encodings on primary categorical features
for col in cat_features:
    freq = X[col].astype(str).value_counts(normalize=True)
    df_enc_train[f"{col}_freq"] = X[col].astype(str).map(freq).fillna(0)
    if X_test is not None:
        df_enc_test[f"{col}_freq"] = X_test[col].astype(str).map(freq).fillna(0)

# Out-of-fold Regularized Target Encoding (Smoothing prior: weight = 10)
if "y" in globals() and y is not None and len(cat_features) > 0:
    global_mean = np.mean(y)
    smooth_weight = 10.0
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in cat_features:
        col_str = X[col].astype(str)
        oof_te = np.zeros(len(X))
        
        for tr_idx, val_idx in kf_te.split(X, y):
            X_tr_col, y_tr = col_str.iloc[tr_idx], y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
            stats = pd.DataFrame({"cat": X_tr_col, "target": y_tr}).groupby("cat")["target"].agg(["count", "mean"])
            smoothed_map = (stats["count"] * stats["mean"] + smooth_weight * global_mean) / (stats["count"] + smooth_weight)
            oof_te[val_idx] = col_str.iloc[val_idx].map(smoothed_map).fillna(global_mean)
        
        df_enc_train[f"{col}_oof_te"] = oof_te

        if X_test is not None:
            full_stats = pd.DataFrame({"cat": col_str, "target": y}).groupby("cat")["target"].agg(["count", "mean"])
            full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth_weight * global_mean) / (full_stats["count"] + smooth_weight)
            df_enc_test[f"{col}_oof_te"] = X_test[col].astype(str).map(full_smoothed).fillna(global_mean)

# ==========================================
# 4. Multi-View Linear Decomposition Suite
# ==========================================
decomp_train_parts = []
decomp_test_parts = []

if len(num_cols) > 0:
    scaler = StandardScaler()
    X_num_clean = X[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
    X_num_scaled = scaler.fit_transform(X_num_clean)
    decomp_train_parts.append(X_num_scaled)

    if X_test is not None:
        X_test_num_clean = X_test[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
        X_test_num_scaled = scaler.transform(X_test_num_clean)
        decomp_test_parts.append(X_test_num_scaled)

if len(cat_features) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_str = X[cat_features].astype(str)
    X_cat_ohe = ohe.fit_transform(X_cat_str)
    decomp_train_parts.append(X_cat_ohe)

    if X_test is not None:
        X_test_cat_str = X_test[cat_features].astype(str)
        X_test_cat_ohe = ohe.transform(X_test_cat_str)
        decomp_test_parts.append(X_test_cat_ohe)

if len(decomp_train_parts) > 0:
    Z_train = np.hstack(decomp_train_parts)
    Z_test = np.hstack(decomp_test_parts) if X_test is not None else None
else:
    Z_train = np.zeros((len(X), 1))
    Z_test = np.zeros((len(X_test), 1)) if X_test is not None else None

df_decomp_train = {}
df_decomp_test = {}
n_comp = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca_tr = pca.fit_transform(Z_train)
for i in range(n_comp):
    df_decomp_train[f"latent_pca_{i}"] = pca_tr[:, i]
if Z_test is not None:
    pca_te = pca.transform(Z_test)
    for i in range(n_comp):
        df_decomp_test[f"latent_pca_{i}"] = pca_te[:, i]

# TruncatedSVD
svd = TruncatedSVD(n_components=n_comp, random_state=42)
svd_tr = svd.fit_transform(Z_train)
for i in range(n_comp):
    df_decomp_train[f"latent_svd_{i}"] = svd_tr[:, i]
if Z_test is not None:
    svd_te = svd.transform(Z_test)
    for i in range(n_comp):
        df_decomp_test[f"latent_svd_{i}"] = svd_te[:, i]

# FastICA
ica = FastICA(n_components=n_comp, random_state=42, max_iter=400, tol=1e-3)
ica_tr = ica.fit_transform(Z_train)
for i in range(n_comp):
    df_decomp_train[f"latent_ica_{i}"] = ica_tr[:, i]
if Z_test is not None:
    ica_te = ica.transform(Z_test)
    for i in range(n_comp):
        df_decomp_test[f"latent_ica_{i}"] = ica_te[:, i]

# FactorAnalysis
fa = FactorAnalysis(n_components=n_comp, random_state=42, max_iter=300)
fa_tr = fa.fit_transform(Z_train)
for i in range(n_comp):
    df_decomp_train[f"latent_fa_{i}"] = fa_tr[:, i]
if Z_test is not None:
    fa_te = fa.transform(Z_test)
    for i in range(n_comp):
        df_decomp_test[f"latent_fa_{i}"] = fa_te[:, i]

df_decomp_train = pd.DataFrame(df_decomp_train, index=X.index)
df_decomp_test = pd.DataFrame(df_decomp_test, index=X_test.index) if X_test is not None else None

# Combine all engineered continuous representations
dense_engineered_tr = pd.concat([df_counts_train, df_enc_train, df_decomp_train], axis=1)
dense_engineered_te = pd.concat([df_counts_test, df_enc_test, df_decomp_test], axis=1) if X_test is not None else None

# ==========================================
# 5. Model Matrix Construction
# ==========================================
# --- For CatBoost (Native Categorical Strings + Continuous Decompositions) ---
X_cb = pd.concat([X.copy(), dense_engineered_tr], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), dense_engineered_te], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost & LightGBM & Ridge (Numeric Encoded + Continuous Decompositions) ---
X_num_all = pd.concat([X.copy(), dense_engineered_tr], axis=1)
X_test_num_all = pd.concat([X_test.copy(), dense_engineered_te], axis=1) if X_test is not None else None

for col in cat_features:
    le = LabelEncoder()
    if X_test is not None and col in X_test.columns:
        combined_vals = list(set(X[col].astype(str)).union(set(X_test[col].astype(str))))
        le.fit(combined_vals)
        X_num_all[col] = le.transform(X_num_all[col].astype(str))
        X_test_num_all[col] = le.transform(X_test_num_all[col].astype(str))
    else:
        X_num_all[col] = le.fit_transform(X_num_all[col].astype(str))

X_xgb = X_num_all.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test_xgb = X_test_num_all.apply(pd.to_numeric, errors="coerce").fillna(0) if X_test_num_all is not None else None

X_lgb = X_xgb.copy()
X_test_lgb = X_test_xgb.copy() if X_test_xgb is not None else None

# Standard scaled matrix for Ridge
ridge_scaler = StandardScaler()
X_ridge = pd.DataFrame(ridge_scaler.fit_transform(X_xgb), index=X_xgb.index, columns=X_xgb.columns)
X_test_ridge = pd.DataFrame(ridge_scaler.transform(X_test_xgb), index=X_test_xgb.index, columns=X_test_xgb.columns) if X_test_xgb is not None else None

# ==========================================
# 6. Out-Of-Fold Ensembling: LightGBM, XGBoost, CatBoost, Ridge
# ==========================================
if "y" in globals() and y is not None:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cb = np.zeros(len(X))
    oof_ridge = np.zeros(len(X))

    test_preds_lgb = np.zeros(len(X_test)) if X_test is not None else None
    test_preds_xgb = np.zeros(len(X_test)) if X_test is not None else None
    test_preds_cb = np.zeros(len(X_test)) if X_test is not None else None
    test_preds_ridge = np.zeros(len(X_test)) if X_test is not None else None

    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for tr_idx, val_idx in kf.split(X, y):
        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        model_lgb.fit(X_lgb.iloc[tr_idx], y_arr[tr_idx], eval_set=[(X_lgb.iloc[val_idx], y_arr[val_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict(X_lgb.iloc[val_idx])
        if X_test is not None:
            test_preds_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, tree_method="hist")
        model_xgb.fit(X_xgb.iloc[tr_idx], y_arr[tr_idx], eval_set=[(X_xgb.iloc[val_idx], y_arr[val_idx])], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_xgb.iloc[val_idx])
        if X_test is not None:
            test_preds_xgb += model_xgb.predict(X_test_xgb) / kf.n_splits

        # 3. CatBoost
        model_cb = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, subsample=0.8, random_seed=42, cat_features=cat_features, early_stopping_rounds=50, verbose=False)
        model_cb.fit(X_cb.iloc[tr_idx], y_arr[tr_idx], eval_set=(X_cb.iloc[val_idx], y_arr[val_idx]), verbose=False)
        oof_cb[val_idx] = model_cb.predict(X_cb.iloc[val_idx])
        if X_test is not None:
            test_preds_cb += model_cb.predict(X_test_cb) / kf.n_splits

        # 4. Ridge Regression
        model_ridge = Ridge(alpha=10.0, random_state=42)
        model_ridge.fit(X_ridge.iloc[tr_idx], y_arr[tr_idx])
        oof_ridge[val_idx] = model_ridge.predict(X_ridge.iloc[val_idx])
        if X_test is not None:
            test_preds_ridge += model_ridge.predict(X_test_ridge) / kf.n_splits

    # Meta Ensemble (Weighted Blend)
    oof_predictions = 0.35 * oof_lgb + 0.35 * oof_xgb + 0.20 * oof_cb + 0.10 * oof_ridge
    if X_test is not None:
        test_predictions = 0.35 * test_preds_lgb + 0.35 * test_preds_xgb + 0.20 * test_preds_cb + 0.10 * test_preds_ridge

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
import itertools
import os
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
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

# 2. Base Feature Identification
num_cols = [c for c in X.columns if c not in cat_features]
X_enhanced_train = X.copy()
X_enhanced_test = X_test.copy() if X_test is not None else None

# 3. Row-wise Statistical Aggregations Across Numerical Features
if len(num_cols) > 0:
    train_num = X[num_cols]
    test_num = X_test[num_cols] if X_test is not None else None

    row_stats_train = pd.DataFrame(index=X.index)
    row_stats_train["row_null_count"] = train_num.isnull().sum(axis=1)
    row_stats_train["row_mean"] = train_num.mean(axis=1)
    row_stats_train["row_std"] = train_num.std(axis=1).fillna(0)
    row_stats_train["row_min"] = train_num.min(axis=1)
    row_stats_train["row_max"] = train_num.max(axis=1)
    row_stats_train["row_skew"] = train_num.skew(axis=1).fillna(0)

    X_enhanced_train = pd.concat([X_enhanced_train, row_stats_train], axis=1)

    if X_test is not None:
        row_stats_test = pd.DataFrame(index=X_test.index)
        row_stats_test["row_null_count"] = test_num.isnull().sum(axis=1)
        row_stats_test["row_mean"] = test_num.mean(axis=1)
        row_stats_test["row_std"] = test_num.std(axis=1).fillna(0)
        row_stats_test["row_min"] = test_num.min(axis=1)
        row_stats_test["row_max"] = test_num.max(axis=1)
        row_stats_test["row_skew"] = test_num.skew(axis=1).fillna(0)
        X_enhanced_test = pd.concat([X_enhanced_test, row_stats_test], axis=1)

    for col in num_cols:
        if X[col].isnull().sum() > 0 or (X_test is not None and X_test[col].isnull().sum() > 0):
            X_enhanced_train[f"{col}_isnan"] = X[col].isnull().astype(float)
            if X_test is not None:
                X_enhanced_test[f"{col}_isnan"] = X_test[col].isnull().astype(float)

# 4. Non-linear Polynomial & Ratio Interactions Between Top Numerical Features
if len(num_cols) >= 2:
    top_num_cols = num_cols[:min(5, len(num_cols))]
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        v1_tr = X[col1].fillna(X[col1].median(numeric_only=True)).fillna(0)
        v2_tr = X[col2].fillna(X[col2].median(numeric_only=True)).fillna(0)

        X_enhanced_train[f"poly_prod_{col1}_{col2}"] = v1_tr * v2_tr
        X_enhanced_train[f"poly_diff_{col1}_{col2}"] = v1_tr - v2_tr
        X_enhanced_train[f"poly_ratio_{col1}_{col2}"] = v1_tr / (np.abs(v2_tr) + 1e-5)

        if X_test is not None:
            v1_te = X_test[col1].fillna(X[col1].median(numeric_only=True)).fillna(0)
            v2_te = X_test[col2].fillna(X[col2].median(numeric_only=True)).fillna(0)
            X_enhanced_test[f"poly_prod_{col1}_{col2}"] = v1_te * v2_te
            X_enhanced_test[f"poly_diff_{col1}_{col2}"] = v1_te - v2_te
            X_enhanced_test[f"poly_ratio_{col1}_{col2}"] = v1_te / (np.abs(v2_te) + 1e-5)

# 5. Frequency Encodings for Categorical Features
if len(cat_features) > 0:
    for col in cat_features:
        if X_test is not None:
            combined_series = pd.concat([X[col], X_test[col]]).astype(str)
        else:
            combined_series = X[col].astype(str)
        freq_map = combined_series.value_counts(normalize=True).to_dict()

        X_enhanced_train[f"{col}_freq"] = X[col].astype(str).map(freq_map).fillna(0)
        if X_test is not None:
            X_enhanced_test[f"{col}_freq"] = X_test[col].astype(str).map(freq_map).fillna(0)

# 6. Out-Of-Fold Smoothed Target Encoding (Single & Interaction Pairs)
target_enc_cols = list(cat_features)
cat_pairs = []
if len(cat_features) >= 2:
    for c1, c2 in itertools.combinations(cat_features[:min(4, len(cat_features))], 2):
        pair_col = f"{c1}_x_{c2}"
        X_enhanced_train[pair_col] = X[c1].astype(str) + "_" + X[c2].astype(str)
        if X_test is not None:
            X_enhanced_test[pair_col] = X_test[c1].astype(str) + "_" + X_test[c2].astype(str)
        cat_pairs.append(pair_col)

all_target_enc_cols = target_enc_cols + cat_pairs
all_cat_cols = list(set(cat_features + cat_pairs))

target_series = pd.Series(y).reset_index(drop=True)
global_mean = float(target_series.mean())
smoothing_weight = 10.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in all_target_enc_cols:
    col_series = X_enhanced_train[col].astype(str).reset_index(drop=True)
    oof_te = np.full(len(X_enhanced_train), global_mean, dtype=float)

    for train_idx, val_idx in kf.split(X_enhanced_train, target_series):
        tr_col, tr_target = col_series.iloc[train_idx], target_series.iloc[train_idx]
        stats = tr_target.groupby(tr_col).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing_weight * global_mean) / (
            stats["count"] + smoothing_weight
        )
        oof_te[val_idx] = col_series.iloc[val_idx].map(smoothed).fillna(global_mean).values

    X_enhanced_train[f"{col}_te"] = oof_te

    if X_test is not None:
        full_stats = target_series.groupby(col_series).agg(["count", "mean"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smoothing_weight * global_mean) / (
            full_stats["count"] + smoothing_weight
        )
        X_enhanced_test[f"{col}_te"] = (
            X_enhanced_test[col].astype(str).map(full_smoothed).fillna(global_mean).values
        )

# 7. Geometric & Latent Manifold Engineering (SVD, NMF, MiniBatchKMeans)
ohe_train_parts = []
ohe_test_parts = []

if len(num_cols) > 0:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_num_clean = X[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
    X_num_scaled = scaler.fit_transform(X_num_clean)
    ohe_train_parts.append(X_num_scaled)

    if X_test is not None:
        X_test_num_clean = X_test[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
        X_test_num_scaled = scaler.transform(X_test_num_clean)
        ohe_test_parts.append(X_test_num_scaled)

if len(cat_features) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_str = X[cat_features].astype(str)
    X_cat_ohe = ohe.fit_transform(X_cat_str)
    ohe_train_parts.append(X_cat_ohe)

    if X_test is not None:
        X_test_cat_str = X_test[cat_features].astype(str)
        X_test_cat_ohe = ohe.transform(X_test_cat_str)
        ohe_test_parts.append(X_test_cat_ohe)

if len(ohe_train_parts) > 0:
    Z_train = np.hstack(ohe_train_parts)
    Z_test = np.hstack(ohe_test_parts) if X_test is not None else None
else:
    Z_train = np.zeros((len(X), 1))
    Z_test = np.zeros((len(X_test), 1)) if X_test is not None else None

geom_features_train = {}
geom_features_test = {}

n_comp_svd = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_comp_svd, random_state=42)
svd_train = svd.fit_transform(Z_train)
for i in range(n_comp_svd):
    geom_features_train[f"latent_svd_{i}"] = svd_train[:, i]

if Z_test is not None:
    svd_test = svd.transform(Z_test)
    for i in range(n_comp_svd):
        geom_features_test[f"latent_svd_{i}"] = svd_test[:, i]

n_comp_nmf = min(4, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
nmf = NMF(n_components=n_comp_nmf, init="nndsvda", max_iter=400, random_state=42)
nmf_train = nmf.fit_transform(np.clip(Z_train, 0, None))
for i in range(n_comp_nmf):
    geom_features_train[f"latent_nmf_{i}"] = nmf_train[:, i]

if Z_test is not None:
    nmf_test = nmf.transform(np.clip(Z_test, 0, None))
    for i in range(n_comp_nmf):
        geom_features_test[f"latent_nmf_{i}"] = nmf_test[:, i]

k_granularities = [4, 8, 16]
for k in k_granularities:
    actual_k = min(k, len(Z_train))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_k,
        batch_size=1024,
        max_iter=100,
        random_state=42 + actual_k,
        n_init="auto",
    )
    dist_train = kmeans.fit_transform(Z_train)
    for c_idx in range(actual_k):
        geom_features_train[f"dist_kmeans_k{actual_k}_c{c_idx}"] = dist_train[:, c_idx]
    geom_features_train[f"dist_kmeans_k{actual_k}_min"] = dist_train.min(axis=1)

    if Z_test is not None:
        dist_test = kmeans.transform(Z_test)
        for c_idx in range(actual_k):
            geom_features_test[f"dist_kmeans_k{actual_k}_c{c_idx}"] = dist_test[:, c_idx]
        geom_features_test[f"dist_kmeans_k{actual_k}_min"] = dist_test.min(axis=1)

df_geom_train = pd.DataFrame(geom_features_train, index=X.index)
df_geom_test = pd.DataFrame(geom_features_test, index=X_test.index) if X_test is not None else None

# Combine all engineered representations
X_enhanced_train = pd.concat([X_enhanced_train, df_geom_train], axis=1)
if X_test is not None:
    X_enhanced_test = pd.concat([X_enhanced_test, df_geom_test], axis=1)

# 8. Model-Specific Dataset Preparation
# --- For CatBoost (Native Categoricals Preserved as Strings) ---
cb_cat_cols = [c for c in all_cat_cols if c in X_enhanced_train.columns]

X_cb = X_enhanced_train.copy()
for col in cb_cat_cols:
    X_cb[col] = X_cb[col].astype(str)

if X_enhanced_test is not None:
    X_test_cb = X_enhanced_test.copy()
    for col in cb_cat_cols:
        if col in X_test_cb.columns:
            X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded Categoricals + Standardized Numerics) ---
X_xgb = X_enhanced_train.copy()
X_test_xgb = X_enhanced_test.copy() if X_enhanced_test is not None else None

for col in cb_cat_cols:
    le = LabelEncoder()
    if X_test_xgb is not None and col in X_test_xgb.columns:
        combined_vals = list(set(X_xgb[col].astype(str)).union(set(X_test_xgb[col].astype(str))))
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

# 9. Model Training
# --- Train CatBoost ---
cb_model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    subsample=0.8,
    random_seed=42,
    cat_features=cb_cat_cols,
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

# 10. Ensemble Validation Predictions
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)
ensemble_val_preds = 0.5 * val_preds_cb + 0.5 * val_preds_xgb

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 11. Generate Test Predictions
if X_test is not None:
    test_preds_cb = cb_model.predict(X_test_cb)
    dtest = xgb.DMatrix(X_test_xgb)
    test_preds_xgb = xgb_model.predict(dtest)
    final_test_preds = 0.5 * test_preds_cb + 0.5 * test_preds_xgb

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
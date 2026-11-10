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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# 1. Prune zero-variance and duplicate features
constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])
    cat_features = [c for c in cat_features if c not in constant_cols]

dup_mask = X.T.duplicated()
if dup_mask.any():
    dup_cols = X.columns[dup_mask].tolist()
    X = X.drop(columns=dup_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=[c for c in dup_cols if c in X_test.columns])
    cat_features = [c for c in cat_features if c not in dup_cols]

num_cols = [c for c in X.columns if c not in cat_features]

# 2. Frequency encoding for categorical features
freq_features_train = {}
freq_features_test = {}
for col in cat_features:
    freq = X[col].astype(str).value_counts(normalize=True)
    freq_features_train[f"{col}_freq"] = X[col].astype(str).map(freq).fillna(0).values
    if X_test is not None:
        freq_features_test[f"{col}_freq"] = X_test[col].astype(str).map(freq).fillna(0).values

# 3. Scaled dense representation for row-wise summary stats and latent decompositions
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

# 4. Row-wise summary statistics
stat_features_train = {}
stat_features_test = {}

stat_features_train["row_active_sum"] = (Z_train > 0).sum(axis=1)
stat_features_train["row_mean"] = Z_train.mean(axis=1)
stat_features_train["row_var"] = Z_train.var(axis=1)
stat_features_train["row_sparsity"] = (Z_train == 0).mean(axis=1)

if Z_test is not None:
    stat_features_test["row_active_sum"] = (Z_test > 0).sum(axis=1)
    stat_features_test["row_mean"] = Z_test.mean(axis=1)
    stat_features_test["row_var"] = Z_test.var(axis=1)
    stat_features_test["row_sparsity"] = (Z_test == 0).mean(axis=1)

# 5. Streamlined decomposition suite (PCA, SVD) & single low-k KMeans
decomp_features_train = {}
decomp_features_test = {}

n_comp = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1

svd = TruncatedSVD(n_components=n_comp, random_state=42)
svd_train = svd.fit_transform(Z_train)
for i in range(n_comp):
    decomp_features_train[f"latent_svd_{i}"] = svd_train[:, i]
if Z_test is not None:
    svd_test = svd.transform(Z_test)
    for i in range(n_comp):
        decomp_features_test[f"latent_svd_{i}"] = svd_test[:, i]

pca = PCA(n_components=n_comp, random_state=42)
pca_train = pca.fit_transform(Z_train)
for i in range(n_comp):
    decomp_features_train[f"latent_pca_{i}"] = pca_train[:, i]
if Z_test is not None:
    pca_test = pca.transform(Z_test)
    for i in range(n_comp):
        decomp_features_test[f"latent_pca_{i}"] = pca_test[:, i]

k_kmeans = min(6, len(Z_train))
kmeans = MiniBatchKMeans(
    n_clusters=k_kmeans,
    batch_size=1024,
    max_iter=100,
    random_state=42,
    n_init="auto"
)
dist_train = kmeans.fit_transform(Z_train)
for c_idx in range(k_kmeans):
    decomp_features_train[f"dist_kmeans_c{c_idx}"] = dist_train[:, c_idx]
decomp_features_train["dist_kmeans_min"] = dist_train.min(axis=1)

if Z_test is not None:
    dist_test = kmeans.transform(Z_test)
    for c_idx in range(k_kmeans):
        decomp_features_test[f"dist_kmeans_c{c_idx}"] = dist_test[:, c_idx]
    decomp_features_test["dist_kmeans_min"] = dist_test.min(axis=1)

# Combine engineered features
engineered_train = {**freq_features_train, **stat_features_train, **decomp_features_train}
engineered_test = {**freq_features_test, **stat_features_test, **decomp_features_test} if X_test is not None else None

df_eng_train = pd.DataFrame(engineered_train, index=X.index)
df_eng_test = pd.DataFrame(engineered_test, index=X_test.index) if X_test is not None else None

# --- For CatBoost (Native Categoricals + Compact Engineered Manifolds) ---
X_cb = pd.concat([X.copy(), df_eng_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), df_eng_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded + Compact Engineered Manifolds) ---
X_xgb = pd.concat([X.copy(), df_eng_train], axis=1)
if X_test is not None:
    X_test_xgb = pd.concat([X_test.copy(), df_eng_test], axis=1)
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
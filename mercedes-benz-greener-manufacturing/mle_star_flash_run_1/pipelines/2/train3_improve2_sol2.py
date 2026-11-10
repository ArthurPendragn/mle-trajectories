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

from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.random_projection import GaussianRandomProjection

# 2. Feature Preprocessing & Comprehensive Latent Manifold / Interaction Engineering
num_cols = [c for c in X.columns if c not in cat_features]

# Prepare scaled and one-hot representations for geometric & decomposition feature extraction
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

# --- 1. Row-level Active Bit and Distributional Statistics ---
geom_features_train["row_active_sum"] = Z_train.sum(axis=1)
geom_features_train["row_active_mean"] = Z_train.mean(axis=1)
geom_features_train["row_active_std"] = Z_train.std(axis=1)
geom_features_train["row_active_nonzero_cnt"] = (np.abs(Z_train) > 1e-5).sum(axis=1)
geom_features_train["row_active_skew"] = skew(Z_train, axis=1, nan_policy="omit")
geom_features_train["row_active_kurt"] = kurtosis(Z_train, axis=1, nan_policy="omit")

if Z_test is not None:
    geom_features_test["row_active_sum"] = Z_test.sum(axis=1)
    geom_features_test["row_active_mean"] = Z_test.mean(axis=1)
    geom_features_test["row_active_std"] = Z_test.std(axis=1)
    geom_features_test["row_active_nonzero_cnt"] = (np.abs(Z_test) > 1e-5).sum(axis=1)
    geom_features_test["row_active_skew"] = skew(Z_test, axis=1, nan_policy="omit")
    geom_features_test["row_active_kurt"] = kurtosis(Z_test, axis=1, nan_policy="omit")

# --- 2. Hierarchical Group-level Aggregation Statistics ---
row_sum_train_series = pd.Series(geom_features_train["row_active_sum"], index=X.index)
for col in cat_features:
    grp_mean = row_sum_train_series.groupby(X[col].astype(str)).mean()
    grp_std = row_sum_train_series.groupby(X[col].astype(str)).std().fillna(0)

    geom_features_train[f"grp_{col}_rowsum_mean"] = X[col].astype(str).map(grp_mean).fillna(grp_mean.mean()).values
    geom_features_train[f"grp_{col}_rowsum_std"] = X[col].astype(str).map(grp_std).fillna(0).values

    if X_test is not None:
        geom_features_test[f"grp_{col}_rowsum_mean"] = X_test[col].astype(str).map(grp_mean).fillna(grp_mean.mean()).values
        geom_features_test[f"grp_{col}_rowsum_std"] = X_test[col].astype(str).map(grp_std).fillna(0).values

# --- 3. Pairwise Categorical Interactions with Frequency Encoding ---
if len(cat_features) >= 2:
    for col1, col2 in combinations(cat_features[:6], 2):
        comb_train = X[col1].astype(str) + "_" + X[col2].astype(str)
        comb_freq = comb_train.value_counts(normalize=True)

        geom_features_train[f"freq_inter_{col1}_{col2}"] = comb_train.map(comb_freq).fillna(0).values
        if X_test is not None:
            comb_test = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
            geom_features_test[f"freq_inter_{col1}_{col2}"] = comb_test.map(comb_freq).fillna(0).values

# --- 4. Multi-View Latent Decompositions ---
d_features = Z_train.shape[1]

# TruncatedSVD
n_comp_svd = min(6, max(2, d_features - 1)) if d_features > 1 else 1
svd = TruncatedSVD(n_components=n_comp_svd, random_state=42)
svd_train = svd.fit_transform(Z_train)
for i in range(n_comp_svd):
    geom_features_train[f"latent_svd_{i}"] = svd_train[:, i]
if Z_test is not None:
    svd_test = svd.transform(Z_test)
    for i in range(n_comp_svd):
        geom_features_test[f"latent_svd_{i}"] = svd_test[:, i]

# NMF
n_comp_nmf = min(4, max(2, d_features - 1)) if d_features > 1 else 1
nmf = NMF(n_components=n_comp_nmf, init="nndsvda", max_iter=400, random_state=42)
nmf_train = nmf.fit_transform(np.clip(Z_train, 0, None))
for i in range(n_comp_nmf):
    geom_features_train[f"latent_nmf_{i}"] = nmf_train[:, i]
if Z_test is not None:
    nmf_test = nmf.transform(np.clip(Z_test, 0, None))
    for i in range(n_comp_nmf):
        geom_features_test[f"latent_nmf_{i}"] = nmf_test[:, i]

# PCA
n_comp_pca = min(4, max(2, d_features - 1)) if d_features > 1 else 1
pca = PCA(n_components=n_comp_pca, random_state=42)
pca_train = pca.fit_transform(Z_train)
for i in range(n_comp_pca):
    geom_features_train[f"latent_pca_{i}"] = pca_train[:, i]
if Z_test is not None:
    pca_test = pca.transform(Z_test)
    for i in range(n_comp_pca):
        geom_features_test[f"latent_pca_{i}"] = pca_test[:, i]

# FastICA
try:
    n_comp_ica = min(4, max(2, d_features - 1)) if d_features > 1 else 1
    ica = FastICA(n_components=n_comp_ica, random_state=42, max_iter=300, tol=1e-3)
    ica_train = ica.fit_transform(Z_train)
    for i in range(n_comp_ica):
        geom_features_train[f"latent_ica_{i}"] = ica_train[:, i]
    if Z_test is not None:
        ica_test = ica.transform(Z_test)
        for i in range(n_comp_ica):
            geom_features_test[f"latent_ica_{i}"] = ica_test[:, i]
except Exception:
    pass

# Factor Analysis
try:
    n_comp_fa = min(4, max(2, d_features - 1)) if d_features > 1 else 1
    fa = FactorAnalysis(n_components=n_comp_fa, random_state=42, max_iter=200)
    fa_train = fa.fit_transform(Z_train)
    for i in range(n_comp_fa):
        geom_features_train[f"latent_fa_{i}"] = fa_train[:, i]
    if Z_test is not None:
        fa_test = fa.transform(Z_test)
        for i in range(n_comp_fa):
            geom_features_test[f"latent_fa_{i}"] = fa_test[:, i]
except Exception:
    pass

# Gaussian Random Projection (GRP)
n_comp_grp = min(4, max(2, d_features - 1)) if d_features > 1 else 1
grp = GaussianRandomProjection(n_components=n_comp_grp, random_state=42)
grp_train = grp.fit_transform(Z_train)
for i in range(n_comp_grp):
    geom_features_train[f"latent_grp_{i}"] = grp_train[:, i]
if Z_test is not None:
    grp_test = grp.transform(Z_test)
    for i in range(n_comp_grp):
        geom_features_test[f"latent_grp_{i}"] = grp_test[:, i]

# --- 5. Multi-granularity Euclidean Cluster Distances via MiniBatchKMeans ---
k_granularities = [4, 8, 16]
for k in k_granularities:
    actual_k = min(k, len(Z_train))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_k,
        batch_size=1024,
        max_iter=100,
        random_state=42 + actual_k,
        n_init="auto"
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

df_geom_train = pd.DataFrame(geom_features_train, index=X.index).fillna(0)
df_geom_test = pd.DataFrame(geom_features_test, index=X_test.index).fillna(0) if X_test is not None else None

# --- Assemble CatBoost Representation (Native Categorical Strings + Latent/Manifolds) ---
X_cb = pd.concat([X.copy(), df_geom_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), df_geom_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- Assemble XGBoost Representation (Label Encoded + Latent/Manifolds) ---
X_xgb = pd.concat([X.copy(), df_geom_train], axis=1)
if X_test is not None:
    X_test_xgb = pd.concat([X_test.copy(), df_geom_test], axis=1)
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
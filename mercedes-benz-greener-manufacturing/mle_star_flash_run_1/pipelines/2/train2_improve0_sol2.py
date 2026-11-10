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
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# 2. Feature Preprocessing & Latent Manifold / Dense Summary Engineering
num_cols = [c for c in X.columns if c not in cat_features]

# Prepare scaled representation for latent manifold feature extraction
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

# 1. Row-wise Dense Summary Statistics (Sparsity and Activation Density)
geom_features_train["row_stat_sum"] = np.sum(Z_train, axis=1)
geom_features_train["row_stat_mean"] = np.mean(Z_train, axis=1)
geom_features_train["row_stat_std"] = np.std(Z_train, axis=1)
geom_features_train["row_stat_zeros"] = np.sum(np.isclose(Z_train, 0.0), axis=1)

if Z_test is not None:
    geom_features_test["row_stat_sum"] = np.sum(Z_test, axis=1)
    geom_features_test["row_stat_mean"] = np.mean(Z_test, axis=1)
    geom_features_test["row_stat_std"] = np.std(Z_test, axis=1)
    geom_features_test["row_stat_zeros"] = np.sum(np.isclose(Z_test, 0.0), axis=1)

# 2. Dense latent topic components via TruncatedSVD
n_comp_svd = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_comp_svd, random_state=42)
svd_train = svd.fit_transform(Z_train)
for i in range(n_comp_svd):
    geom_features_train[f"latent_svd_{i}"] = svd_train[:, i]

if Z_test is not None:
    svd_test = svd.transform(Z_test)
    for i in range(n_comp_svd):
        geom_features_test[f"latent_svd_{i}"] = svd_test[:, i]

# 3. Dense latent topic components via NMF (Non-negative Matrix Factorization)
n_comp_nmf = min(4, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
nmf = NMF(n_components=n_comp_nmf, init="nndsvda", max_iter=400, random_state=42)
nmf_train = nmf.fit_transform(np.clip(Z_train, 0, None))
for i in range(n_comp_nmf):
    geom_features_train[f"latent_nmf_{i}"] = nmf_train[:, i]

if Z_test is not None:
    nmf_test = nmf.transform(np.clip(Z_test, 0, None))
    for i in range(n_comp_nmf):
        geom_features_test[f"latent_nmf_{i}"] = nmf_test[:, i]

# 4. Orthogonal Variance Latent Components via PCA
n_comp_pca = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
pca = PCA(n_components=n_comp_pca, random_state=42)
pca_train = pca.fit_transform(Z_train)
for i in range(n_comp_pca):
    geom_features_train[f"latent_pca_{i}"] = pca_train[:, i]

if Z_test is not None:
    pca_test = pca.transform(Z_test)
    for i in range(n_comp_pca):
        geom_features_test[f"latent_pca_{i}"] = pca_test[:, i]

# 5. Statistically Independent Signal Components via FastICA
n_comp_ica = min(4, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
ica = FastICA(n_components=n_comp_ica, max_iter=500, random_state=42, tol=1e-3)
ica_train = ica.fit_transform(Z_train)
for i in range(n_comp_ica):
    geom_features_train[f"latent_ica_{i}"] = ica_train[:, i]

if Z_test is not None:
    ica_test = ica.transform(Z_test)
    for i in range(n_comp_ica):
        geom_features_test[f"latent_ica_{i}"] = ica_test[:, i]

df_geom_train = pd.DataFrame(geom_features_train, index=X.index)
df_geom_test = pd.DataFrame(geom_features_test, index=X_test.index) if X_test is not None else None

# --- For CatBoost (Native Categorical Strings + Latent Manifolds) ---
X_cb = pd.concat([X.copy(), df_geom_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), df_geom_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded + Latent Manifolds) ---
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
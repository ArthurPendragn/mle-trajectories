import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LinearRegression,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_csv("./input/train.csv")
test_path = "./input/test.csv"
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

X_test = None
if test_df is not None:
    X_test = test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()

# Split into Train and Holdout Validation sets
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_raw = X_train_raw.reset_index(drop=True)
X_val_raw = X_val_raw.reset_index(drop=True)
if X_test is not None:
    X_test_raw = X_test.reset_index(drop=True)
else:
    X_test_raw = None

# Identify categorical and numerical columns
cat_features = [col for col in X_train_raw.columns if not pd.api.types.is_numeric_dtype(X_train_raw[col])]
num_cols = [c for c in X_train_raw.columns if c not in cat_features]

# Prune constant and duplicated columns based on X_train_raw
constant_cols = [c for c in num_cols if X_train_raw[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

dup_cols = []
seen_series = {}
for col in valid_num_cols:
    vals_tuple = tuple(X_train_raw[col].values)
    if vals_tuple in seen_series:
        dup_cols.append(col)
    else:
        seen_series[vals_tuple] = col

cleaned_num_cols = [c for c in valid_num_cols if c not in dup_cols]
cols_to_keep = [c for c in X_train_raw.columns if c not in constant_cols and c not in dup_cols]

X_tr_filt = X_train_raw[cols_to_keep].copy()
X_va_filt = X_val_raw[cols_to_keep].copy()
X_te_filt = X_test_raw[cols_to_keep].copy() if X_test_raw is not None else None

# Row-wise active statistics on cleaned binary/numeric space
for split_df, raw_filt in zip(
    [X_tr_filt, X_va_filt] + ([X_te_filt] if X_te_filt is not None else []),
    [X_train_raw[cleaned_num_cols], X_val_raw[cleaned_num_cols]] + ([X_test_raw[cleaned_num_cols]] if X_te_filt is not None else [])
):
    split_df["row_active_sum"] = raw_filt.sum(axis=1).values
    split_df["row_active_mean"] = raw_filt.mean(axis=1).values
    split_df["row_active_std"] = raw_filt.std(axis=1).values

# Manifold Decompositions
n_decomp = min(10, max(2, len(cleaned_num_cols) - 1))
X_tr_num = X_train_raw[cleaned_num_cols].fillna(0)
X_va_num = X_val_raw[cleaned_num_cols].fillna(0)
X_te_num = X_test_raw[cleaned_num_cols].fillna(0) if X_test_raw is not None else None

decomp_models = {
    "pca": PCA(n_components=n_decomp, random_state=42),
    "svd": TruncatedSVD(n_components=n_decomp, random_state=42),
    "grp": GaussianRandomProjection(n_components=n_decomp, random_state=42),
    "srp": SparseRandomProjection(n_components=n_decomp, random_state=42),
}

decomp_train_dict = {}
decomp_val_dict = {}
decomp_test_dict = {}

for name, model in decomp_models.items():
    tr_feat = model.fit_transform(X_tr_num)
    va_feat = model.transform(X_va_num)
    te_feat = model.transform(X_te_num) if X_te_num is not None else None
    for i in range(n_decomp):
        decomp_train_dict[f"{name}_{i}"] = tr_feat[:, i]
        decomp_val_dict[f"{name}_{i}"] = va_feat[:, i]
        if te_feat is not None:
            decomp_test_dict[f"{name}_{i}"] = te_feat[:, i]

try:
    ica = FastICA(n_components=n_decomp, random_state=42, max_iter=1000, tol=1e-3)
    tr_ica = ica.fit_transform(X_tr_num)
    va_ica = ica.transform(X_va_num)
    te_ica = ica.transform(X_te_num) if X_te_num is not None else None
    for i in range(n_decomp):
        decomp_train_dict[f"ica_{i}"] = tr_ica[:, i]
        decomp_val_dict[f"ica_{i}"] = va_ica[:, i]
        if te_ica is not None:
            decomp_test_dict[f"ica_{i}"] = te_ica[:, i]
except Exception:
    pass

df_decomp_train = pd.DataFrame(decomp_train_dict)
df_decomp_val = pd.DataFrame(decomp_val_dict)
df_decomp_test = pd.DataFrame(decomp_test_dict) if X_te_filt is not None else None

# Frequency Encodings
freq_train = {}
freq_val = {}
freq_test = {}
for col in cat_features:
    all_cats = X_train_raw[col].astype(str)
    if X_test_raw is not None:
        all_cats = pd.concat([all_cats, X_test_raw[col].astype(str)], axis=0)
    freq_map = all_cats.value_counts()
    freq_train[f"freq_{col}"] = X_train_raw[col].astype(str).map(freq_map).fillna(0).values
    freq_val[f"freq_{col}"] = X_val_raw[col].astype(str).map(freq_map).fillna(0).values
    if X_test_raw is not None:
        freq_test[f"freq_{col}"] = X_test_raw[col].astype(str).map(freq_map).fillna(0).values

df_freq_train = pd.DataFrame(freq_train)
df_freq_val = pd.DataFrame(freq_val)
df_freq_test = pd.DataFrame(freq_test) if X_test_raw is not None else None

# 5-fold Out-Of-Fold Target Encoding on X_tr_filt for high-cardinality categorical columns
target_cols = [c for c in ["X0", "X1", "X2", "X5"] if c in cat_features]
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y_train.mean()

oof_te_mean = {f"te_mean_{c}": np.zeros(len(X_train_raw)) for c in target_cols}
oof_te_std = {f"te_std_{c}": np.zeros(len(X_train_raw)) for c in target_cols}
val_te_mean = {f"te_mean_{c}": np.zeros(len(X_val_raw)) for c in target_cols}
val_te_std = {f"te_std_{c}": np.zeros(len(X_val_raw)) for c in target_cols}
test_te_mean = {f"te_mean_{c}": np.zeros(len(X_test_raw)) for c in target_cols} if X_test_raw is not None else {}
test_te_std = {f"te_std_{c}": np.zeros(len(X_test_raw)) for c in target_cols} if X_test_raw is not None else {}

for c in target_cols:
    smooth = 10
    # Global target encoding for val/test
    grp = pd.DataFrame({"cat": X_train_raw[c].astype(str), "target": y_train}).groupby("cat")["target"].agg(["count", "mean", "std"])
    smoothed_mean = (grp["count"] * grp["mean"] + smooth * global_mean) / (grp["count"] + smooth)
    std_map = grp["std"].fillna(0)

    val_te_mean[f"te_mean_{c}"] = X_val_raw[c].astype(str).map(smoothed_mean).fillna(global_mean).values
    val_te_std[f"te_std_{c}"] = X_val_raw[c].astype(str).map(std_map).fillna(0).values

    if X_test_raw is not None:
        test_te_mean[f"te_mean_{c}"] = X_test_raw[c].astype(str).map(smoothed_mean).fillna(global_mean).values
        test_te_std[f"te_std_{c}"] = X_test_raw[c].astype(str).map(std_map).fillna(0).values

    # OOF on train
    for tr_idx, val_idx in kf_te.split(X_train_raw):
        f_tr_cat = X_train_raw.iloc[tr_idx][c].astype(str)
        f_tr_y = y_train[tr_idx]
        f_val_cat = X_train_raw.iloc[val_idx][c].astype(str)

        f_grp = pd.DataFrame({"cat": f_tr_cat, "target": f_tr_y}).groupby("cat")["target"].agg(["count", "mean", "std"])
        f_smoothed_mean = (f_grp["count"] * f_grp["mean"] + smooth * global_mean) / (f_grp["count"] + smooth)
        f_std_map = f_grp["std"].fillna(0)

        oof_te_mean[f"te_mean_{c}"][val_idx] = f_val_cat.map(f_smoothed_mean).fillna(global_mean).values
        oof_te_std[f"te_std_{c}"][val_idx] = f_val_cat.map(f_std_map).fillna(0).values

df_te_train = pd.DataFrame({**oof_te_mean, **oof_te_std})
df_te_val = pd.DataFrame({**val_te_mean, **val_te_std})
df_te_test = pd.DataFrame({**test_te_mean, **test_te_std}) if X_test_raw is not None else None

# Prepare Tree Feature Space
# Label encode categoricals for XGB / LGBM / CB
X_train_tree = pd.concat([X_tr_filt, df_decomp_train, df_freq_train, df_te_train], axis=1)
X_val_tree = pd.concat([X_va_filt, df_decomp_val, df_freq_val, df_te_val], axis=1)
X_test_tree = pd.concat([X_te_filt, df_decomp_test, df_freq_test, df_te_test], axis=1) if X_te_filt is not None else None

for c in cat_features:
    if c in X_train_tree.columns:
        le = LabelEncoder()
        if X_test_tree is not None:
            all_v = pd.concat([X_train_tree[c].astype(str), X_val_tree[c].astype(str), X_test_tree[c].astype(str)])
        else:
            all_v = pd.concat([X_train_tree[c].astype(str), X_val_tree[c].astype(str)])
        le.fit(all_v)
        X_train_tree[c] = le.transform(X_train_tree[c].astype(str))
        X_val_tree[c] = le.transform(X_val_tree[c].astype(str))
        if X_test_tree is not None:
            X_test_tree[c] = le.transform(X_test_tree[c].astype(str))

# Prepare Linear / Kernel Feature Space (One-Hot Encoded categoricals + Scaled numeric & decomposition)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_train = ohe.fit_transform(X_train_raw[cat_features].astype(str))
ohe_val = ohe.transform(X_val_raw[cat_features].astype(str))
ohe_test = ohe.transform(X_test_raw[cat_features].astype(str)) if X_test_raw is not None else None

num_feats_train = pd.concat([X_tr_filt[cleaned_num_cols], df_decomp_train, df_freq_train, df_te_train], axis=1)
num_feats_val = pd.concat([X_va_filt[cleaned_num_cols], df_decomp_val, df_freq_val, df_te_val], axis=1)
num_feats_test = pd.concat([X_te_filt[cleaned_num_cols], df_decomp_test, df_freq_test, df_te_test], axis=1) if X_te_filt is not None else None

scaler = RobustScaler()
num_scaled_train = scaler.fit_transform(num_feats_train)
num_scaled_val = scaler.transform(num_feats_val)
num_scaled_test = scaler.transform(num_feats_test) if num_feats_test is not None else None

X_train_lin = np.hstack([ohe_train, num_scaled_train])
X_val_lin = np.hstack([ohe_val, num_scaled_val])
X_test_lin = np.hstack([ohe_test, num_scaled_test]) if num_scaled_test is not None else None

# Define 8 Base Estimators
def get_base_models():
    return {
        "xgb": (
            "tree",
            xgb.XGBRegressor(
                n_estimators=350,
                learning_rate=0.015,
                max_depth=3,
                subsample=0.75,
                colsample_bytree=0.7,
                reg_alpha=2.0,
                reg_lambda=5.0,
                random_state=42,
                tree_method="hist",
                verbosity=0,
            ),
        ),
        "lgb": (
            "tree",
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.015,
                max_depth=3,
                num_leaves=7,
                colsample_bytree=0.6,
                subsample=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1,
            ),
        ),
        "cb": (
            "tree",
            CatBoostRegressor(
                iterations=450,
                learning_rate=0.02,
                depth=4,
                l2_leaf_reg=5.0,
                random_seed=42,
                verbose=False,
            ),
        ),
        "bayesian_ridge": (
            "linear",
            BayesianRidge(
                max_iter=1000,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
            ),
        ),
        "huber": (
            "linear",
            HuberRegressor(
                epsilon=1.35,
                alpha=10.0,
                max_iter=1000,
            ),
        ),
        "elastic_net": (
            "linear",
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
                cv=5,
                random_state=42,
                max_iter=2000,
            ),
        ),
        "krr": (
            "linear",
            KernelRidge(
                alpha=0.8,
                kernel="polynomial",
                degree=2,
                coef0=2.5,
            ),
        ),
        "svr": (
            "linear",
            SVR(
                C=2.0,
                epsilon=0.1,
                kernel="rbf",
            ),
        ),
    }

base_model_configs = get_base_models()
model_keys = list(base_model_configs.keys())
n_models = len(model_keys)

# 5-fold CV to collect out-of-fold predictions on Train set & generate fold predictions for Val/Test
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_matrix = np.zeros((len(X_train_raw), n_models))
val_preds_matrix = np.zeros((len(X_val_raw), n_models))
test_preds_matrix = np.zeros((len(X_test_raw), n_models)) if X_test_raw is not None else None

for m_idx, key in enumerate(model_keys):
    m_type, _ = base_model_configs[key]
    X_tr_data = X_train_tree.values if m_type == "tree" else X_train_lin
    X_va_data = X_val_tree.values if m_type == "tree" else X_val_lin
    X_te_data = (X_test_tree.values if m_type == "tree" else X_test_lin) if X_test_raw is not None else None

    fold_val_preds = np.zeros(len(X_val_raw))
    fold_test_preds = np.zeros(len(X_test_raw)) if X_test_raw is not None else None

    for train_fold_idx, oof_fold_idx in kf.split(X_tr_data):
        _, model = get_base_models()[key]
        X_fold_tr, y_fold_tr = X_tr_data[train_fold_idx], y_train[train_fold_idx]
        X_fold_oof = X_tr_data[oof_fold_idx]

        model.fit(X_fold_tr, y_fold_tr)
        oof_matrix[oof_fold_idx, m_idx] = model.predict(X_fold_oof)
        fold_val_preds += model.predict(X_va_data) / 5.0
        if X_test_raw is not None:
            fold_test_preds += model.predict(X_te_data) / 5.0

    val_preds_matrix[:, m_idx] = fold_val_preds
    if X_test_raw is not None:
        test_preds_matrix[:, m_idx] = fold_test_preds

# 3. Robust Huber-Loss SLSQP Optimization
def huber_loss(w, X_mat, y_true, delta=1.5):
    pred = X_mat @ w
    diff = np.abs(y_true - pred)
    loss = np.where(diff <= delta, 0.5 * (diff ** 2), delta * (diff - 0.5 * delta))
    return np.mean(loss)

init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    huber_loss,
    init_weights,
    args=(oof_matrix, y_train, 1.5),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9},
)
convex_weights = opt_res.x

oof_convex = oof_matrix @ convex_weights
val_convex = val_preds_matrix @ convex_weights
test_convex = test_preds_matrix @ convex_weights if test_preds_matrix is not None else None

# 4. Dual Meta-Estimator Stacking (Bayesian Ridge)
bayes_meta = BayesianRidge(alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-4)
bayes_meta.fit(oof_matrix, y_train)

oof_bayes = bayes_meta.predict(oof_matrix)
val_bayes = bayes_meta.predict(val_preds_matrix)
test_bayes = bayes_meta.predict(test_preds_matrix) if test_preds_matrix is not None else None

# Blend: 0.65 * Convex + 0.35 * Bayes
oof_blended = 0.65 * oof_convex + 0.35 * oof_bayes
val_blended = 0.65 * val_convex + 0.35 * val_bayes
test_blended = (0.65 * test_convex + 0.35 * test_bayes) if test_preds_matrix is not None else None

# 5. Post-Ensemble OOF Linear Calibration & Domain Guardrails
calibrator = LinearRegression()
calibrator.fit(oof_blended.reshape(-1, 1), y_train)

val_final = calibrator.predict(val_blended.reshape(-1, 1))
if test_blended is not None:
    test_final = calibrator.predict(test_blended.reshape(-1, 1))
else:
    test_final = None

# Domain bounding strictly within training limits [y_min, y_max]
y_min, y_max = y_train.min(), y_train.max()
val_final = np.clip(val_final, y_min, y_max)
if test_final is not None:
    test_final = np.clip(test_final, y_min, y_max)

final_validation_score = r2_score(y_val, val_final)
print(f"Final Validation Performance: {final_validation_score}")

# Generate submission if test set is present
if test_final is not None:
    submission = pd.DataFrame({"y": test_final})
    submission.to_csv("submission.csv", index=False)
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    RidgeCV,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import xgboost as xgb

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

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify Categorical and Numerical Columns
cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# 2. Two-Track Unified Representation Pipeline
# Prune zero-variance and duplicate binary/numerical columns based on X_train
constant_cols = [c for c in num_cols if X_train[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

dup_cols = []
seen_series = {}
for col in valid_num_cols:
    vals_tuple = tuple(X_train[col].values)
    if vals_tuple in seen_series:
        dup_cols.append(col)
    else:
        seen_series[vals_tuple] = col

cleaned_num_cols = [c for c in valid_num_cols if c not in dup_cols]

# Extract cleaned binary/numerical subsets
X_tr_num = X_train[cleaned_num_cols].fillna(0).astype(np.float32)
X_va_num = X_val[cleaned_num_cols].fillna(0).astype(np.float32)
X_te_num = X_test[cleaned_num_cols].fillna(0).astype(np.float32) if X_test is not None else None

# Row-wise active statistics
tr_act_sum = X_tr_num.sum(axis=1).values.reshape(-1, 1)
tr_act_mean = X_tr_num.mean(axis=1).values.reshape(-1, 1)
tr_act_std = X_tr_num.std(axis=1).values.reshape(-1, 1)
stats_train = np.hstack([tr_act_sum, tr_act_mean, tr_act_std])

va_act_sum = X_va_num.sum(axis=1).values.reshape(-1, 1)
va_act_mean = X_va_num.mean(axis=1).values.reshape(-1, 1)
va_act_std = X_va_num.std(axis=1).values.reshape(-1, 1)
stats_val = np.hstack([va_act_sum, va_act_mean, va_act_std])

if X_te_num is not None:
    te_act_sum = X_te_num.sum(axis=1).values.reshape(-1, 1)
    te_act_mean = X_te_num.mean(axis=1).values.reshape(-1, 1)
    te_act_std = X_te_num.std(axis=1).values.reshape(-1, 1)
    stats_test = np.hstack([te_act_sum, te_act_mean, te_act_std])

# Multi-dimensional Decomposition Manifold
n_decomp = min(10, max(2, len(cleaned_num_cols) - 1))
pca = PCA(n_components=n_decomp, random_state=42)
svd = TruncatedSVD(n_components=n_decomp, random_state=42)
ica = FastICA(n_components=n_decomp, random_state=42, max_iter=1000, tol=1e-3)
grp = GaussianRandomProjection(n_components=n_decomp, random_state=42)
srp = SparseRandomProjection(n_components=n_decomp, random_state=42)

decomp_tr_list = [
    pca.fit_transform(X_tr_num),
    svd.fit_transform(X_tr_num),
    ica.fit_transform(X_tr_num),
    grp.fit_transform(X_tr_num),
    srp.fit_transform(X_tr_num),
]
decomp_va_list = [
    pca.transform(X_va_num),
    svd.transform(X_va_num),
    ica.transform(X_va_num),
    grp.transform(X_va_num),
    srp.transform(X_va_num),
]
decomp_train = np.hstack(decomp_tr_list)
decomp_val = np.hstack(decomp_va_list)

if X_te_num is not None:
    decomp_te_list = [
        pca.transform(X_te_num),
        svd.transform(X_te_num),
        ica.transform(X_te_num),
        grp.transform(X_te_num),
        srp.transform(X_te_num),
    ]
    decomp_test = np.hstack(decomp_te_list)

# Categorical Encodings
# 1. Frequency Encoding
freq_tr, freq_va, freq_te = [], [], []
for col in cat_cols:
    freq_map = X_train[col].astype(str).value_counts()
    freq_tr.append(X_train[col].astype(str).map(freq_map).fillna(0).values.reshape(-1, 1))
    freq_va.append(X_val[col].astype(str).map(freq_map).fillna(0).values.reshape(-1, 1))
    if X_test is not None:
        freq_te.append(X_test[col].astype(str).map(freq_map).fillna(0).values.reshape(-1, 1))

freq_train = np.hstack(freq_tr) if freq_tr else np.empty((len(X_train), 0))
freq_val = np.hstack(freq_va) if freq_va else np.empty((len(X_val), 0))
freq_test = np.hstack(freq_te) if freq_te and X_test is not None else (np.empty((len(X_test), 0)) if X_test is not None else None)

# 2. Ordinal Encoding for Trees
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
ord_train = ord_enc.fit_transform(X_train[cat_cols].astype(str)) if cat_cols else np.empty((len(X_train), 0))
ord_val = ord_enc.transform(X_val[cat_cols].astype(str)) if cat_cols else np.empty((len(X_val), 0))
ord_test = ord_enc.transform(X_test[cat_cols].astype(str)) if (cat_cols and X_test is not None) else (np.empty((len(X_test), 0)) if X_test is not None else None)

# 3. One-Hot Encoding for Linear/Kernel models
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_train = ohe.fit_transform(X_train[cat_cols].astype(str)) if cat_cols else np.empty((len(X_train), 0))
ohe_val = ohe.transform(X_val[cat_cols].astype(str)) if cat_cols else np.empty((len(X_val), 0))
ohe_test = ohe.transform(X_test[cat_cols].astype(str)) if (cat_cols and X_test is not None) else (np.empty((len(X_test), 0)) if X_test is not None else None)

# Track A: Tree Space Representation
X_tr_tree = np.hstack([X_tr_num.values, ord_train, freq_train, stats_train, decomp_train]).astype(np.float32)
X_va_tree = np.hstack([X_va_num.values, ord_val, freq_val, stats_val, decomp_val]).astype(np.float32)
X_te_tree = np.hstack([X_te_num.values, ord_test, freq_test, stats_test, decomp_test]).astype(np.float32) if X_test is not None else None

# Track B: Linear/Kernel Space Representation
scaler = RobustScaler()
dense_linear_tr = np.hstack([X_tr_num.values, stats_train, decomp_train])
dense_linear_va = np.hstack([X_va_num.values, stats_val, decomp_val])
dense_linear_tr_scaled = scaler.fit_transform(dense_linear_tr)
dense_linear_va_scaled = scaler.transform(dense_linear_va)

X_tr_linear = np.hstack([ohe_train, dense_linear_tr_scaled]).astype(np.float32)
X_va_linear = np.hstack([ohe_val, dense_linear_va_scaled]).astype(np.float32)

if X_test is not None:
    dense_linear_te = np.hstack([X_te_num.values, stats_test, decomp_test])
    dense_linear_te_scaled = scaler.transform(dense_linear_te)
    X_te_linear = np.hstack([ohe_test, dense_linear_te_scaled]).astype(np.float32)
else:
    X_te_linear = None

# 3. 5-Fold OOF Prediction Generation across 7-Learner Zoo
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_names_tree = ["xgb", "lgb", "cb"]
model_names_linear = ["krr", "huber", "bayesian_ridge", "elasticnet"]
all_model_names = model_names_tree + model_names_linear

oof_preds = {m: np.zeros(len(X_train)) for m in all_model_names}
val_preds_dict = {m: np.zeros(len(X_val)) for m in all_model_names}
test_preds_dict = {m: np.zeros(len(X_test)) if X_test is not None else None for m in all_model_names}

def get_base_models():
    return {
        "xgb": xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.015,
            max_depth=3,
            colsample_bytree=0.6,
            subsample=0.8,
            reg_alpha=2.0,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        ),
        "lgb": lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.018,
            max_depth=3,
            num_leaves=7,
            colsample_bytree=0.6,
            subsample=0.8,
            reg_alpha=1.5,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
        ),
        "cb": CatBoostRegressor(
            iterations=550,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=5.0,
            subsample=0.8,
            random_seed=42,
            verbose=False,
        ),
        "krr": KernelRidge(
            alpha=0.6, kernel="polynomial", degree=2, coef0=2.5
        ),
        "huber": HuberRegressor(
            epsilon=1.35, alpha=10.0, max_iter=1000
        ),
        "bayesian_ridge": BayesianRidge(
            max_iter=1000,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
        ),
        "elasticnet": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            cv=5,
            max_iter=2000,
            random_state=42,
        ),
    }

for fold, (trn_idx, oof_idx) in enumerate(kf.split(X_tr_tree, y_train)):
    models = get_base_models()
    
    # Train Tree Models
    X_f_tr_tree, y_f_tr = X_tr_tree[trn_idx], y_train[trn_idx]
    X_f_oof_tree = X_tr_tree[oof_idx]
    
    for m_name in model_names_tree:
        model = models[m_name]
        model.fit(X_f_tr_tree, y_f_tr)
        oof_preds[m_name][oof_idx] = model.predict(X_f_oof_tree)
        val_preds_dict[m_name] += model.predict(X_va_tree) / kf.n_splits
        if X_test is not None:
            test_preds_dict[m_name] += model.predict(X_te_tree) / kf.n_splits
            
    # Train Linear Models
    X_f_tr_lin = X_tr_linear[trn_idx]
    X_f_oof_lin = X_tr_linear[oof_idx]
    
    for m_name in model_names_linear:
        model = models[m_name]
        model.fit(X_f_tr_lin, y_f_tr)
        oof_preds[m_name][oof_idx] = model.predict(X_f_oof_lin)
        val_preds_dict[m_name] += model.predict(X_va_linear) / kf.n_splits
        if X_test is not None:
            test_preds_dict[m_name] += model.predict(X_te_linear) / kf.n_splits

# 4. Hierarchical Two-Tier Intra-Family and Cross-Family Meta-Blending
def optimize_convex_weights(pred_matrix, y_true):
    n_models = pred_matrix.shape[1]
    init_w = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    def loss_func(w):
        p = np.dot(pred_matrix, w)
        return np.mean((y_true - p) ** 2)
        
    res = minimize(loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else init_w

# Stage 1: Intra-Family Optimization
OOF_tree = np.column_stack([oof_preds[m] for m in model_names_tree])
VAL_tree = np.column_stack([val_preds_dict[m] for m in model_names_tree])
TE_tree = np.column_stack([test_preds_dict[m] for m in model_names_tree]) if X_test is not None else None

w_tree = optimize_convex_weights(OOF_tree, y_train)
P_tree_oof = np.dot(OOF_tree, w_tree)
P_tree_val = np.dot(VAL_tree, w_tree)
P_tree_te = np.dot(TE_tree, w_tree) if X_test is not None else None

OOF_linear = np.column_stack([oof_preds[m] for m in model_names_linear])
VAL_linear = np.column_stack([val_preds_dict[m] for m in model_names_linear])
TE_linear = np.column_stack([test_preds_dict[m] for m in model_names_linear]) if X_test is not None else None

w_linear = optimize_convex_weights(OOF_linear, y_train)
P_linear_oof = np.dot(OOF_linear, w_linear)
P_linear_val = np.dot(VAL_linear, w_linear)
P_linear_te = np.dot(TE_linear, w_linear) if X_test is not None else None

# Stage 2: Cross-Family Adaptive Blending
OOF_families = np.column_stack([P_tree_oof, P_linear_oof])
VAL_families = np.column_stack([P_tree_val, P_linear_val])
TE_families = np.column_stack([P_tree_te, P_linear_te]) if X_test is not None else None

w_families = optimize_convex_weights(OOF_families, y_train)
P_hier_val = np.dot(VAL_families, w_families)
P_hier_te = np.dot(TE_families, w_families) if X_test is not None else None

# Stage 3: Meta-Regularization across all 7 base streams
OOF_all = np.column_stack([oof_preds[m] for m in all_model_names])
VAL_all = np.column_stack([val_preds_dict[m] for m in all_model_names])
TE_all = np.column_stack([test_preds_dict[m] for m in all_model_names]) if X_test is not None else None

meta_ridge = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=5)
meta_ridge.fit(OOF_all, y_train)
P_meta_val = meta_ridge.predict(VAL_all)
P_meta_te = meta_ridge.predict(TE_all) if X_test is not None else None

# Combine Hierarchical and Meta-regularized Streams (0.60 * Hier + 0.40 * Meta)
final_val_preds = 0.60 * P_hier_val + 0.40 * P_meta_val

# 5. Domain Calibration and Boundary Safeguards
y_min, y_max = np.min(y_train), np.max(y_train)
final_val_preds = np.clip(final_val_preds, y_min, y_max)

final_validation_score = r2_score(y_val, final_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 6. Generate and Save Test Predictions
if X_test is not None:
    final_test_preds = 0.60 * P_hier_te + 0.40 * P_meta_te
    final_test_preds = np.clip(final_test_preds, y_min, y_max)
    submission = pd.DataFrame({"y": final_test_preds})
    if "ID" in test_df.columns:
        submission.insert(0, "ID", test_df["ID"])
    submission.to_csv("submission.csv", index=False)
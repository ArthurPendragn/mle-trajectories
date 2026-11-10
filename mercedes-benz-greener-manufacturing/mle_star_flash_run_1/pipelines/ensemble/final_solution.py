import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import NMF, PCA, FastICA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from sklearn.svm import SVR
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_csv("./input/train.csv")
test_path = "./input/test.csv"
test_df = pd.read_csv(test_path)

y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])
X_test = test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()

# Identify categorical features
cat_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
num_cols = [c for c in X.columns if c not in cat_features]

# 2. Prune Constant and Duplicated Columns
constant_cols = [c for c in num_cols if X[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

dup_cols = []
seen_series = {}
for col in valid_num_cols:
    vals_tuple = tuple(X[col].values)
    if vals_tuple in seen_series:
        dup_cols.append(col)
    else:
        seen_series[vals_tuple] = col

cleaned_num_cols = [c for c in valid_num_cols if c not in dup_cols]

# 3. Feature Engineering: Statistics, Decompositions & Frequency Encodings
engineered_train = {}
engineered_test = {}

if len(cleaned_num_cols) > 0:
    X_num_clean = X[cleaned_num_cols].fillna(0).astype(np.float32)
    engineered_train["row_active_sum"] = X_num_clean.sum(axis=1).values
    engineered_train["row_active_mean"] = X_num_clean.mean(axis=1).values
    engineered_train["row_active_std"] = X_num_clean.std(axis=1).values

    X_test_num_clean = X_test[cleaned_num_cols].fillna(0).astype(np.float32)
    engineered_test["row_active_sum"] = X_test_num_clean.sum(axis=1).values
    engineered_test["row_active_mean"] = X_test_num_clean.mean(axis=1).values
    engineered_test["row_active_std"] = X_test_num_clean.std(axis=1).values

    # Multi-resolution linear manifold projections
    for n_comp, prefix in [(5, "5"), (10, "10")]:
        actual_comp = min(n_comp, max(2, len(cleaned_num_cols) - 1))

        # PCA
        pca = PCA(n_components=actual_comp, random_state=42)
        pca_tr = pca.fit_transform(X_num_clean)
        pca_te = pca.transform(X_test_num_clean)
        for i in range(actual_comp):
            engineered_train[f"pca_{prefix}_{i}"] = pca_tr[:, i]
            engineered_test[f"pca_{prefix}_{i}"] = pca_te[:, i]

        # TruncatedSVD
        svd = TruncatedSVD(n_components=actual_comp, random_state=42)
        svd_tr = svd.fit_transform(X_num_clean)
        svd_te = svd.transform(X_test_num_clean)
        for i in range(actual_comp):
            engineered_train[f"svd_{prefix}_{i}"] = svd_tr[:, i]
            engineered_test[f"svd_{prefix}_{i}"] = svd_te[:, i]

        # FastICA
        ica = FastICA(n_components=actual_comp, random_state=42, max_iter=500, tol=1e-3)
        ica_tr = ica.fit_transform(X_num_clean)
        ica_te = ica.transform(X_test_num_clean)
        for i in range(actual_comp):
            engineered_train[f"ica_{prefix}_{i}"] = ica_tr[:, i]
            engineered_test[f"ica_{prefix}_{i}"] = ica_te[:, i]

        # Gaussian Random Projection
        grp = GaussianRandomProjection(n_components=actual_comp, eps=0.1, random_state=42)
        grp_tr = grp.fit_transform(X_num_clean)
        grp_te = grp.transform(X_test_num_clean)
        for i in range(actual_comp):
            engineered_train[f"grp_{prefix}_{i}"] = grp_tr[:, i]
            engineered_test[f"grp_{prefix}_{i}"] = grp_te[:, i]

        # Sparse Random Projection
        srp = SparseRandomProjection(n_components=actual_comp, random_state=42)
        srp_tr = srp.fit_transform(X_num_clean)
        srp_te = srp.transform(X_test_num_clean)
        for i in range(actual_comp):
            engineered_train[f"srp_{prefix}_{i}"] = srp_tr[:, i]
            engineered_test[f"srp_{prefix}_{i}"] = srp_te[:, i]

        # Non-Negative Matrix Factorization (NMF)
        nmf = NMF(n_components=actual_comp, random_state=42, max_iter=500, init="nndsvda")
        nmf_tr = nmf.fit_transform(np.clip(X_num_clean.values, 0, None))
        nmf_te = nmf.transform(np.clip(X_test_num_clean.values, 0, None))
        for i in range(actual_comp):
            engineered_train[f"nmf_{prefix}_{i}"] = nmf_tr[:, i]
            engineered_test[f"nmf_{prefix}_{i}"] = nmf_te[:, i]

# Category Frequency Encodings
for col in cat_features:
    combined_cat = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
    freq_map = combined_cat.value_counts()
    engineered_train[f"freq_{col}"] = X[col].astype(str).map(freq_map).fillna(0).values
    engineered_test[f"freq_{col}"] = X_test[col].astype(str).map(freq_map).fillna(0).values

df_eng_train = pd.DataFrame(engineered_train, index=X.index)
df_eng_test = pd.DataFrame(engineered_test, index=X_test.index)

# Filter base DataFrame
cols_to_keep = [c for c in X.columns if c not in constant_cols and c not in dup_cols]
X_filtered = X[cols_to_keep].copy()
X_test_filtered = X_test[cols_to_keep].copy()

# Construct unified numerical matrix with Label Encoded categoricals
X_all = pd.concat([X_filtered, df_eng_train], axis=1)
X_test_all = pd.concat([X_test_filtered, df_eng_test], axis=1)

for col in cat_features:
    if col in X_all.columns:
        le = LabelEncoder()
        combined_vals = list(set(X[col].astype(str)).union(set(X_test[col].astype(str))))
        le.fit(combined_vals)
        X_all[col] = le.transform(X_all[col].astype(str))
        X_test_all[col] = le.transform(X_test_all[col].astype(str))

X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
X_test_all = X_test_all.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

# Hold-out Validation Split (80/20)
X_train_full, X_val_full, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# Prepare Scaled representation for Linear/Kernel/Distance estimators
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val_full)
X_test_scaled = scaler.transform(X_test_all)

# Convert to arrays
X_train_arr = X_train_full.values
X_val_arr = X_val_full.values
X_test_arr = X_test_all.values

n_train = len(y_train)
n_val = len(y_val)
n_test = len(X_test_arr)

# 4. Define 10 Diverse Base Learners
N_MODELS = 10
oof_train_preds = np.zeros((n_train, N_MODELS), dtype=np.float64)
oof_val_preds = np.zeros((n_val, N_MODELS), dtype=np.float64)
oof_test_preds = np.zeros((n_test, N_MODELS), dtype=np.float64)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_arr, y_train)):
    # Slice features
    X_tr_tree, y_tr = X_train_arr[trn_idx], y_train[trn_idx]
    X_va_tree, y_va = X_train_arr[val_idx], y_train[val_idx]
    X_tr_sc = X_train_scaled[trn_idx]
    X_va_sc = X_train_scaled[val_idx]

    # 1. XGBoost
    dtrn = xgb.DMatrix(X_tr_tree, label=y_tr)
    dval_fold = xgb.DMatrix(X_va_tree, label=y_va)
    dval_holdout = xgb.DMatrix(X_val_arr)
    dtest = xgb.DMatrix(X_test_arr)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.015,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.65,
        "alpha": 1.5,
        "lambda": 3.0,
        "random_state": 42 + fold,
        "tree_method": "hist",
    }
    m_xgb = xgb.train(
        xgb_params,
        dtrn,
        num_boost_round=600,
        evals=[(dval_fold, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_train_preds[val_idx, 0] = m_xgb.predict(dval_fold)
    oof_val_preds[:, 0] += m_xgb.predict(dval_holdout) / 5.0
    oof_test_preds[:, 0] += m_xgb.predict(dtest) / 5.0

    # 2. LightGBM
    m_lgb = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.015,
        max_depth=3,
        num_leaves=7,
        colsample_bytree=0.65,
        subsample=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42 + fold,
        verbose=-1,
    )
    m_lgb.fit(X_tr_tree, y_tr)
    oof_train_preds[val_idx, 1] = m_lgb.predict(X_va_tree)
    oof_val_preds[:, 1] += m_lgb.predict(X_val_arr) / 5.0
    oof_test_preds[:, 1] += m_lgb.predict(X_test_arr) / 5.0

    # 3. CatBoost
    m_cb = CatBoostRegressor(
        iterations=700,
        learning_rate=0.02,
        depth=3,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        verbose=False,
    )
    m_cb.fit(X_tr_tree, y_tr, eval_set=(X_va_tree, y_va), early_stopping_rounds=50, verbose=False)
    oof_train_preds[val_idx, 2] = m_cb.predict(X_va_tree)
    oof_val_preds[:, 2] += m_cb.predict(X_val_arr) / 5.0
    oof_test_preds[:, 2] += m_cb.predict(X_test_arr) / 5.0

    # 4. ExtraTrees Regressor
    m_et = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        max_features=0.6,
        random_state=42 + fold,
        n_jobs=-1,
    )
    m_et.fit(X_tr_tree, y_tr)
    oof_train_preds[val_idx, 3] = m_et.predict(X_va_tree)
    oof_val_preds[:, 3] += m_et.predict(X_val_arr) / 5.0
    oof_test_preds[:, 3] += m_et.predict(X_test_arr) / 5.0

    # 5. RandomForest Regressor
    m_rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=4,
        max_features=0.6,
        random_state=42 + fold,
        n_jobs=-1,
    )
    m_rf.fit(X_tr_tree, y_tr)
    oof_train_preds[val_idx, 4] = m_rf.predict(X_va_tree)
    oof_val_preds[:, 4] += m_rf.predict(X_val_arr) / 5.0
    oof_test_preds[:, 4] += m_rf.predict(X_test_arr) / 5.0

    # 6. Huber Regressor
    m_huber = HuberRegressor(epsilon=1.35, alpha=15.0, max_iter=1000)
    m_huber.fit(X_tr_sc, y_tr)
    oof_train_preds[val_idx, 5] = m_huber.predict(X_va_sc)
    oof_val_preds[:, 5] += m_huber.predict(X_val_scaled) / 5.0
    oof_test_preds[:, 5] += m_huber.predict(X_test_scaled) / 5.0

    # 7. Bayesian Ridge
    m_bridge = BayesianRidge(max_iter=1000, alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-4)
    m_bridge.fit(X_tr_sc, y_tr)
    oof_train_preds[val_idx, 6] = m_bridge.predict(X_va_sc)
    oof_val_preds[:, 6] += m_bridge.predict(X_val_scaled) / 5.0
    oof_test_preds[:, 6] += m_bridge.predict(X_test_scaled) / 5.0

    # 8. ElasticNetCV
    m_enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=42 + fold, max_iter=2000)
    m_enet.fit(X_tr_sc, y_tr)
    oof_train_preds[val_idx, 7] = m_enet.predict(X_va_sc)
    oof_val_preds[:, 7] += m_enet.predict(X_val_scaled) / 5.0
    oof_test_preds[:, 7] += m_enet.predict(X_test_scaled) / 5.0

    # 9. Kernel Ridge
    m_krr = KernelRidge(alpha=1.0, kernel="polynomial", degree=2, coef0=1.0)
    m_krr.fit(X_tr_sc, y_tr)
    oof_train_preds[val_idx, 8] = m_krr.predict(X_va_sc)
    oof_val_preds[:, 8] += m_krr.predict(X_val_scaled) / 5.0
    oof_test_preds[:, 8] += m_krr.predict(X_test_scaled) / 5.0

    # 10. SVR (Support Vector Regressor RBF)
    m_svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    m_svr.fit(X_tr_sc, y_tr)
    oof_train_preds[val_idx, 9] = m_svr.predict(X_va_sc)
    oof_val_preds[:, 9] += m_svr.predict(X_val_scaled) / 5.0
    oof_test_preds[:, 9] += m_svr.predict(X_test_scaled) / 5.0

# 5. Tri-Stream Meta-Ensemble Fusion

# Stream 1: Trimmed Regularized Huber SLSQP Convex Optimization with Prior Shrinkage
def huber_loss(res, delta=3.0):
    abs_res = np.abs(res)
    huber = np.where(abs_res <= delta, 0.5 * (res ** 2), delta * (abs_res - 0.5 * delta))
    return np.mean(huber)

uniform_prior = np.ones(N_MODELS) / N_MODELS
shrinkage_lambda = 0.05

def objective(weights):
    w = np.array(weights)
    pred = oof_train_preds @ w
    res = y_train - pred
    loss = huber_loss(res, delta=3.0) + shrinkage_lambda * np.sum((w - uniform_prior) ** 2)
    return loss

init_weights = np.ones(N_MODELS) / N_MODELS
bounds = [(0.0, 1.0) for _ in range(N_MODELS)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-9},
)

convex_weights = opt_res.x if opt_res.success else uniform_prior
pred_stream1_train = oof_train_preds @ convex_weights
pred_stream1_val = oof_val_preds @ convex_weights
pred_stream1_test = oof_test_preds @ convex_weights

# Stream 2: High-regularization RidgeCV Meta-Learner
meta_scaler = StandardScaler()
oof_train_std = meta_scaler.fit_transform(oof_train_preds)
oof_val_std = meta_scaler.transform(oof_val_preds)
oof_test_std = meta_scaler.transform(oof_test_preds)

ridge_meta = RidgeCV(alphas=np.logspace(0, 5, 50), cv=5)
ridge_meta.fit(oof_train_std, y_train)

pred_stream2_train = ridge_meta.predict(oof_train_std)
pred_stream2_val = ridge_meta.predict(oof_val_std)
pred_stream2_test = ridge_meta.predict(oof_test_std)

# Stream 3: Non-Negative ElasticNet Meta-Learner
enet_meta = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], positive=True, cv=5, random_state=42, max_iter=2000)
enet_meta.fit(oof_train_preds, y_train)

pred_stream3_train = enet_meta.predict(oof_train_preds)
pred_stream3_val = enet_meta.predict(oof_val_preds)
pred_stream3_test = enet_meta.predict(oof_test_preds)

# Blend 3 Meta-Streams: 0.50 * Convex + 0.30 * RidgeCV + 0.20 * ElasticNet
blended_train = 0.50 * pred_stream1_train + 0.30 * pred_stream2_train + 0.20 * pred_stream3_train
blended_val = 0.50 * pred_stream1_val + 0.30 * pred_stream2_val + 0.20 * pred_stream3_val
blended_test = 0.50 * pred_stream1_test + 0.30 * pred_stream2_test + 0.20 * pred_stream3_test

# 6. OOF Residual Calibration and Target Bounds Guardrail
calibrator = LinearRegression()
calibrator.fit(blended_train.reshape(-1, 1), y_train)

calibrated_val = calibrator.predict(blended_val.reshape(-1, 1))
calibrated_test = calibrator.predict(blended_test.reshape(-1, 1))

# Guardrail clamping
y_min, y_max = np.min(y_train), np.max(y_train)
final_val_preds = np.clip(calibrated_val, y_min, y_max)

final_validation_score = r2_score(y_val, final_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 7. Generate Submission
os.makedirs("./final", exist_ok=True)
final_test_preds = np.clip(calibrated_test, y_min, y_max)
submission = pd.DataFrame({"y": final_test_preds})
submission.to_csv("./final/submission.csv", index=False)
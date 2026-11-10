import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import xgboost as xgb

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Separate features and target
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

X_test = None
if test_df is not None:
    X_test = test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()

# Identify categorical and numerical features
cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
num_cols = [col for col in X.columns if col not in cat_cols]

# 2. Unified Feature Engineering
# Filter zero-variance and identical duplicate numerical/binary columns
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

# Compute row-wise active statistics on cleaned binary/numeric space
engineered_train = {}
engineered_test = {}

if len(cleaned_num_cols) > 0:
    X_num_clean = X[cleaned_num_cols].fillna(0).astype(np.float32)
    engineered_train["row_active_sum"] = X_num_clean.sum(axis=1).values
    engineered_train["row_active_mean"] = X_num_clean.mean(axis=1).values
    engineered_train["row_active_std"] = X_num_clean.std(axis=1).values

    if X_test is not None:
        X_test_num_clean = X_test[cleaned_num_cols].fillna(0).astype(np.float32)
        engineered_test["row_active_sum"] = X_test_num_clean.sum(axis=1).values
        engineered_test["row_active_mean"] = X_test_num_clean.mean(axis=1).values
        engineered_test["row_active_std"] = X_test_num_clean.std(axis=1).values

    # Linear and Manifold Decomposition Embeddings
    n_decomp = min(10, max(2, len(cleaned_num_cols) - 1)) if len(cleaned_num_cols) > 1 else 1

    # PCA
    pca = PCA(n_components=n_decomp, random_state=42)
    pca_train = pca.fit_transform(X_num_clean)
    for i in range(n_decomp):
        engineered_train[f"pca_{i}"] = pca_train[:, i]
    if X_test is not None:
        pca_test = pca.transform(X_test_num_clean)
        for i in range(n_decomp):
            engineered_test[f"pca_{i}"] = pca_test[:, i]

    # TruncatedSVD
    svd = TruncatedSVD(n_components=n_decomp, random_state=42)
    svd_train = svd.fit_transform(X_num_clean)
    for i in range(n_decomp):
        engineered_train[f"svd_{i}"] = svd_train[:, i]
    if X_test is not None:
        svd_test = svd.transform(X_test_num_clean)
        for i in range(n_decomp):
            engineered_test[f"svd_{i}"] = svd_test[:, i]

    # FastICA
    try:
        ica = FastICA(n_components=n_decomp, random_state=42, max_iter=1000, tol=1e-3)
        ica_train = ica.fit_transform(X_num_clean)
        for i in range(n_decomp):
            engineered_train[f"ica_{i}"] = ica_train[:, i]
        if X_test is not None:
            ica_test = ica.transform(X_test_num_clean)
            for i in range(n_decomp):
                engineered_test[f"ica_{i}"] = ica_test[:, i]
    except Exception:
        pass

    # Gaussian Random Projection
    grp = GaussianRandomProjection(n_components=n_decomp, eps=0.1, random_state=42)
    grp_train = grp.fit_transform(X_num_clean)
    for i in range(n_decomp):
        engineered_train[f"grp_{i}"] = grp_train[:, i]
    if X_test is not None:
        grp_test = grp.transform(X_test_num_clean)
        for i in range(n_decomp):
            engineered_test[f"grp_{i}"] = grp_test[:, i]

    # Sparse Random Projection
    srp = SparseRandomProjection(n_components=n_decomp, random_state=42, dense_output=True)
    srp_train = srp.fit_transform(X_num_clean)
    for i in range(n_decomp):
        engineered_train[f"srp_{i}"] = srp_train[:, i]
    if X_test is not None:
        srp_test = srp.transform(X_test_num_clean)
        for i in range(n_decomp):
            engineered_test[f"srp_{i}"] = srp_test[:, i]

# Categorical Frequency Encoding
for col in cat_cols:
    if X_test is not None:
        combined_cat = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
        freq_map = combined_cat.value_counts()
    else:
        freq_map = X[col].astype(str).value_counts()
    engineered_train[f"freq_{col}"] = X[col].astype(str).map(freq_map).fillna(0).values
    if X_test is not None:
        engineered_test[f"freq_{col}"] = X_test[col].astype(str).map(freq_map).fillna(0).values

df_eng_train = pd.DataFrame(engineered_train, index=X.index)
df_eng_test = pd.DataFrame(engineered_test, index=X_test.index) if X_test is not None else None

# Filter base dataframe
cols_to_keep = [c for c in X.columns if c not in constant_cols and c not in dup_cols]
X_filtered = X[cols_to_keep].copy()
X_test_filtered = X_test[cols_to_keep].copy() if X_test is not None else None

# 3. Dual Representation Spaces
# --- Representation A: Tree Space ---
X_tree = pd.concat([X_filtered, df_eng_train], axis=1).copy()
X_test_tree = pd.concat([X_test_filtered, df_eng_test], axis=1).copy() if X_test_filtered is not None else None

for col in cat_cols:
    if col in X_tree.columns:
        le = LabelEncoder()
        if X_test_tree is not None and col in X_test_tree.columns:
            combined_vals = list(set(X[col].astype(str)).union(set(X_test[col].astype(str))))
            le.fit(combined_vals)
            X_tree[col] = le.transform(X_tree[col].astype(str))
            X_test_tree[col] = le.transform(X_test_tree[col].astype(str))
        else:
            X_tree[col] = le.fit_transform(X_tree[col].astype(str))

X_tree = X_tree.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
if X_test_tree is not None:
    X_test_tree = X_test_tree.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

# --- Representation B: Linear Space ---
if len(cat_cols) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_train = ohe.fit_transform(X[cat_cols].astype(str))
    ohe_train_df = pd.DataFrame(ohe_train, index=X.index, columns=ohe.get_feature_names_out(cat_cols))
    if X_test is not None:
        ohe_test = ohe.transform(X_test[cat_cols].astype(str))
        ohe_test_df = pd.DataFrame(ohe_test, index=X_test.index, columns=ohe.get_feature_names_out(cat_cols))
    else:
        ohe_test_df = None
else:
    ohe_train_df = pd.DataFrame(index=X.index)
    ohe_test_df = pd.DataFrame(index=X_test.index) if X_test is not None else None

num_filtered_cols = [c for c in cleaned_num_cols if c in X_filtered.columns]
X_linear = pd.concat([ohe_train_df, X_filtered[num_filtered_cols], df_eng_train], axis=1).astype(np.float32)
if X_test is not None:
    X_test_linear = pd.concat([ohe_test_df, X_test_filtered[num_filtered_cols], df_eng_test], axis=1).astype(np.float32)
else:
    X_test_linear = None

# 4. Train / Validation Split
indices = np.arange(len(y))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_tree = X_tree.iloc[train_idx].reset_index(drop=True)
X_val_tree = X_tree.iloc[val_idx].reset_index(drop=True)

X_train_linear_raw = X_linear.iloc[train_idx].reset_index(drop=True)
X_val_linear_raw = X_linear.iloc[val_idx].reset_index(drop=True)

y_train = y[train_idx]
y_val = y[val_idx]

# RobustScale linear space on train split
scaler = RobustScaler()
X_train_linear = scaler.fit_transform(X_train_linear_raw)
X_val_linear = scaler.transform(X_val_linear_raw)
X_test_linear_scaled = scaler.transform(X_test_linear) if X_test_linear is not None else None

# Latent meta features for passthrough
latent_cols = [c for c in ["pca_0", "pca_1", "pca_2", "row_active_sum"] if c in df_eng_train.columns]
latent_train = df_eng_train.iloc[train_idx][latent_cols].values
latent_val = df_eng_train.iloc[val_idx][latent_cols].values
latent_test = df_eng_test[latent_cols].values if df_eng_test is not None else None

# 5. 5-Fold Cross-Validation for Base Learners Zoo
kf = KFold(n_splits=5, shuffle=True, random_state=42)
num_models = 6
oof_preds = np.zeros((len(train_idx), num_models), dtype=np.float32)
val_preds_base = np.zeros((len(val_idx), num_models), dtype=np.float32)
test_preds_base = np.zeros((len(X_test), num_models), dtype=np.float32) if X_test is not None else None

for fold, (f_train_idx, f_val_idx) in enumerate(kf.split(X_train_tree, y_train)):
    # Tree fold data
    f_X_tr_tree, f_y_tr = X_train_tree.iloc[f_train_idx], y_train[f_train_idx]
    f_X_va_tree = X_train_tree.iloc[f_val_idx]
    
    # Linear fold data
    f_X_tr_lin = X_train_linear[f_train_idx]
    f_X_va_lin = X_train_linear[f_val_idx]

    # Model 0: XGBoost
    xgb_m = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.015,
        max_depth=3,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42 + fold,
        tree_method="hist",
        verbosity=0,
    )
    xgb_m.fit(f_X_tr_tree, f_y_tr)
    oof_preds[f_val_idx, 0] = xgb_m.predict(f_X_va_tree)
    val_preds_base[:, 0] += xgb_m.predict(X_val_tree) / kf.n_splits
    if X_test_tree is not None:
        test_preds_base[:, 0] += xgb_m.predict(X_test_tree) / kf.n_splits

    # Model 1: CatBoost
    cb_m = CatBoostRegressor(
        iterations=500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        verbose=False,
    )
    cb_m.fit(f_X_tr_tree, f_y_tr)
    oof_preds[f_val_idx, 1] = cb_m.predict(f_X_va_tree)
    val_preds_base[:, 1] += cb_m.predict(X_val_tree) / kf.n_splits
    if X_test_tree is not None:
        test_preds_base[:, 1] += cb_m.predict(X_test_tree) / kf.n_splits

    # Model 2: LightGBM
    lgb_m = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=7,
        colsample_bytree=0.6,
        subsample=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_m.fit(f_X_tr_tree, f_y_tr)
    oof_preds[f_val_idx, 2] = lgb_m.predict(f_X_va_tree)
    val_preds_base[:, 2] += lgb_m.predict(X_val_tree) / kf.n_splits
    if X_test_tree is not None:
        test_preds_base[:, 2] += lgb_m.predict(X_test_tree) / kf.n_splits

    # Model 3: Kernel Ridge
    kr_m = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    kr_m.fit(f_X_tr_lin, f_y_tr)
    oof_preds[f_val_idx, 3] = kr_m.predict(f_X_va_lin)
    val_preds_base[:, 3] += kr_m.predict(X_val_linear) / kf.n_splits
    if X_test_linear_scaled is not None:
        test_preds_base[:, 3] += kr_m.predict(X_test_linear_scaled) / kf.n_splits

    # Model 4: Bayesian Ridge
    br_m = BayesianRidge(max_iter=1000, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
    br_m.fit(f_X_tr_lin, f_y_tr)
    oof_preds[f_val_idx, 4] = br_m.predict(f_X_va_lin)
    val_preds_base[:, 4] += br_m.predict(X_val_linear) / kf.n_splits
    if X_test_linear_scaled is not None:
        test_preds_base[:, 4] += br_m.predict(X_test_linear_scaled) / kf.n_splits

    # Model 5: Huber Regressor
    hb_m = HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=1000)
    hb_m.fit(f_X_tr_lin, f_y_tr)
    oof_preds[f_val_idx, 5] = hb_m.predict(f_X_va_lin)
    val_preds_base[:, 5] += hb_m.predict(X_val_linear) / kf.n_splits
    if X_test_linear_scaled is not None:
        test_preds_base[:, 5] += hb_m.predict(X_test_linear_scaled) / kf.n_splits

# 6. Feature-Augmented RidgeCV Meta-Learner with Latent Passthrough
scaler_latent = RobustScaler()
latent_train_scaled = scaler_latent.fit_transform(latent_train)
latent_val_scaled = scaler_latent.transform(latent_val)

augmented_oof_train = np.column_stack([oof_preds, latent_train_scaled])
augmented_meta_val = np.column_stack([val_preds_base, latent_val_scaled])

meta_learner = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=5)
meta_learner.fit(augmented_oof_train, y_train)
meta_val_preds = meta_learner.predict(augmented_meta_val)

# 7. Metric-Direct Non-Negative Bounded Blend
def loss_func(weights):
    pred = oof_preds @ weights
    return np.mean((y_train - pred) ** 2)

initial_weights = np.ones(num_models) / num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
opt_weights = opt_res.x if opt_res.success else initial_weights

blend_val_preds = val_preds_base @ opt_weights

# Final Blended Predictions and Target Domain Calibration
final_val_preds = 0.70 * meta_val_preds + 0.30 * blend_val_preds
target_min, target_max = y_train.min(), y_train.max()
final_val_preds = np.clip(final_val_preds, target_min, target_max)

final_validation_score = r2_score(y_val, final_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 8. Test Predictions and Submission Generation
if X_test is not None:
    latent_test_scaled = scaler_latent.transform(latent_test)
    augmented_meta_test = np.column_stack([test_preds_base, latent_test_scaled])
    
    meta_test_preds = meta_learner.predict(augmented_meta_test)
    blend_test_preds = test_preds_base @ opt_weights
    
    final_test_preds = 0.70 * meta_test_preds + 0.30 * blend_test_preds
    final_test_preds = np.clip(final_test_preds, target_min, target_max)

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
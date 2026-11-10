import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
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
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )

# Hold-out split first to maintain a pure validation set
train_idx, val_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42
)

X_train_raw = X.iloc[train_idx].reset_index(drop=True)
y_train = y[train_idx]
X_val_raw = X.iloc[val_idx].reset_index(drop=True)
y_val = y[val_idx]

if X_test is not None:
    X_test_raw = X_test.copy().reset_index(drop=True)
else:
    X_test_raw = None

# Identify categorical features
cat_cols = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]
num_cols = [c for c in X.columns if c not in cat_cols]

# Eliminate zero-variance and duplicate columns based on train fold
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

# 2. Dual-Path Feature Engineering
def build_features(df_train, df_val, df_test=None):
    train_eng = {}
    val_eng = {}
    test_eng = {} if df_test is not None else None

    # Row-wise active sum and aggregate statistics
    if len(cleaned_num_cols) > 0:
        X_num_train = df_train[cleaned_num_cols].fillna(0).values
        X_num_val = df_val[cleaned_num_cols].fillna(0).values

        train_eng["row_active_sum"] = np.sum(X_num_train, axis=1)
        train_eng["row_active_mean"] = np.mean(X_num_train, axis=1)
        train_eng["row_active_std"] = np.std(X_num_train, axis=1)

        val_eng["row_active_sum"] = np.sum(X_num_val, axis=1)
        val_eng["row_active_mean"] = np.mean(X_num_val, axis=1)
        val_eng["row_active_std"] = np.std(X_num_val, axis=1)

        if df_test is not None:
            X_num_test = df_test[cleaned_num_cols].fillna(0).values
            test_eng["row_active_sum"] = np.sum(X_num_test, axis=1)
            test_eng["row_active_mean"] = np.mean(X_num_test, axis=1)
            test_eng["row_active_std"] = np.std(X_num_test, axis=1)

        # Decompositions
        n_comp = min(10, max(2, len(cleaned_num_cols) - 1))

        # PCA
        pca = PCA(n_components=n_comp, random_state=42)
        train_pca = pca.fit_transform(X_num_train)
        val_pca = pca.transform(X_num_val)
        for i in range(n_comp):
            train_eng[f"pca_{i}"] = train_pca[:, i]
            val_eng[f"pca_{i}"] = val_pca[:, i]
        if df_test is not None:
            test_pca = pca.transform(X_num_test)
            for i in range(n_comp):
                test_eng[f"pca_{i}"] = test_pca[:, i]

        # SVD
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        train_svd = svd.fit_transform(X_num_train)
        val_svd = svd.transform(X_num_val)
        for i in range(n_comp):
            train_eng[f"svd_{i}"] = train_svd[:, i]
            val_eng[f"svd_{i}"] = val_svd[:, i]
        if df_test is not None:
            test_svd = svd.transform(X_num_test)
            for i in range(n_comp):
                test_eng[f"svd_{i}"] = test_svd[:, i]

        # FastICA
        try:
            ica = FastICA(
                n_components=n_comp, random_state=42, max_iter=1000, tol=1e-3
            )
            train_ica = ica.fit_transform(X_num_train)
            val_ica = ica.transform(X_num_val)
            for i in range(n_comp):
                train_eng[f"ica_{i}"] = train_ica[:, i]
                val_eng[f"ica_{i}"] = val_ica[:, i]
            if df_test is not None:
                test_ica = ica.transform(X_num_test)
                for i in range(n_comp):
                    test_eng[f"ica_{i}"] = test_ica[:, i]
        except Exception:
            pass

        # GaussianRandomProjection
        grp = GaussianRandomProjection(
            n_components=n_comp, eps=0.1, random_state=42
        )
        train_grp = grp.fit_transform(X_num_train)
        val_grp = grp.transform(X_num_val)
        for i in range(n_comp):
            train_eng[f"grp_{i}"] = train_grp[:, i]
            val_eng[f"grp_{i}"] = val_grp[:, i]
        if df_test is not None:
            test_grp = grp.transform(X_num_test)
            for i in range(n_comp):
                test_eng[f"grp_{i}"] = test_grp[:, i]

        # SparseRandomProjection
        srp = SparseRandomProjection(
            n_components=n_comp, dense_output=True, random_state=42
        )
        train_srp = srp.fit_transform(X_num_train)
        val_srp = srp.transform(X_num_val)
        for i in range(n_comp):
            train_eng[f"srp_{i}"] = train_srp[:, i]
            val_eng[f"srp_{i}"] = val_srp[:, i]
        if df_test is not None:
            test_srp = srp.transform(X_num_test)
            for i in range(n_comp):
                test_eng[f"srp_{i}"] = test_srp[:, i]

    # Frequency encodings
    for col in cat_cols:
        freq_map = df_train[col].astype(str).value_counts()
        train_eng[f"freq_{col}"] = (
            df_train[col].astype(str).map(freq_map).fillna(0).values
        )
        val_eng[f"freq_{col}"] = (
            df_val[col].astype(str).map(freq_map).fillna(0).values
        )
        if df_test is not None:
            test_eng[f"freq_{col}"] = (
                df_test[col].astype(str).map(freq_map).fillna(0).values
            )

    df_eng_train = pd.DataFrame(train_eng, index=df_train.index)
    df_eng_val = pd.DataFrame(val_eng, index=df_val.index)
    df_eng_test = (
        pd.DataFrame(test_eng, index=df_test.index)
        if df_test is not None
        else None
    )

    cols_to_keep = [
        c for c in df_train.columns if c not in constant_cols and c not in dup_cols
    ]

    # --- Tree representation space (Label Encoded + Filtered + Decomposition & Row Aggs) ---
    X_tree_tr = pd.concat([df_train[cols_to_keep], df_eng_train], axis=1)
    X_tree_va = pd.concat([df_val[cols_to_keep], df_eng_val], axis=1)
    X_tree_te = (
        pd.concat([df_test[cols_to_keep], df_eng_test], axis=1)
        if df_test is not None
        else None
    )

    for col in cat_cols:
        if col in X_tree_tr.columns:
            le = LabelEncoder()
            all_vals = list(df_train[col].astype(str).unique()) + list(
                df_val[col].astype(str).unique()
            )
            if df_test is not None:
                all_vals += list(df_test[col].astype(str).unique())
            le.fit(list(set(all_vals)))
            X_tree_tr[col] = le.transform(df_train[col].astype(str))
            X_tree_va[col] = le.transform(df_val[col].astype(str))
            if X_tree_te is not None:
                X_tree_te[col] = le.transform(df_test[col].astype(str))

    X_tree_tr = X_tree_tr.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_tree_va = X_tree_va.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    if X_tree_te is not None:
        X_tree_te = X_tree_te.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    # --- Linear representation space (One-Hot Encoded + Robust Scaled) ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if len(cat_cols) > 0:
        ohe_tr = ohe.fit_transform(df_train[cat_cols].astype(str))
        ohe_va = ohe.transform(df_val[cat_cols].astype(str))
        ohe_te = ohe.transform(df_test[cat_cols].astype(str)) if df_test is not None else None
    else:
        ohe_tr = np.empty((len(df_train), 0))
        ohe_va = np.empty((len(df_val), 0))
        ohe_te = np.empty((len(df_test), 0)) if df_test is not None else None

    scaler = RobustScaler()
    tree_num_tr = scaler.fit_transform(X_tree_tr)
    tree_num_va = scaler.transform(X_tree_va)
    tree_num_te = scaler.transform(X_tree_te) if X_tree_te is not None else None

    X_lin_tr = np.hstack([ohe_tr, tree_num_tr]).astype(np.float32)
    X_lin_va = np.hstack([ohe_va, tree_num_va]).astype(np.float32)
    X_lin_te = np.hstack([ohe_te, tree_num_te]).astype(np.float32) if df_test is not None else None

    return (
        X_tree_tr,
        X_tree_va,
        X_tree_te,
        X_lin_tr,
        X_lin_va,
        X_lin_te,
    )


(
    X_tree_train,
    X_tree_val,
    X_tree_test,
    X_lin_train,
    X_lin_val,
    X_lin_test,
) = build_features(X_train_raw, X_val_raw, X_test_raw)

# 3. Model Zoo Out-of-Fold (OOF) Prediction Generation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

num_models = 6
oof_preds = np.zeros((len(y_train), num_models))
val_preds_models = np.zeros((len(y_val), num_models))
test_preds_models = (
    np.zeros((len(X_tree_test), num_models)) if X_tree_test is not None else None
)

for fold, (trn_idx, oof_idx) in enumerate(kf.split(X_tree_train)):
    # Tree fold data
    X_tr_tree, X_oof_tree = (
        X_tree_train.iloc[trn_idx],
        X_tree_train.iloc[oof_idx],
    )
    # Linear fold data
    X_tr_lin, X_oof_lin = X_lin_train[trn_idx], X_lin_train[oof_idx]
    y_tr, y_oof = y_train[trn_idx], y_train[oof_idx]

    # 1. XGBoost
    xgb_mod = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.015,
        max_depth=3,
        colsample_bytree=0.6,
        subsample=0.8,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42 + fold,
        tree_method="hist",
        verbosity=0,
    )
    xgb_mod.fit(X_tr_tree, y_tr)
    oof_preds[oof_idx, 0] = xgb_mod.predict(X_oof_tree)
    val_preds_models[:, 0] += xgb_mod.predict(X_tree_val) / n_splits
    if X_tree_test is not None:
        test_preds_models[:, 0] += xgb_mod.predict(X_tree_test) / n_splits

    # 2. CatBoost
    cb_mod = CatBoostRegressor(
        iterations=500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        verbose=False,
    )
    cb_mod.fit(X_tr_tree, y_tr)
    oof_preds[oof_idx, 1] = cb_mod.predict(X_oof_tree)
    val_preds_models[:, 1] += cb_mod.predict(X_tree_val) / n_splits
    if X_tree_test is not None:
        test_preds_models[:, 1] += cb_mod.predict(X_tree_test) / n_splits

    # 3. LightGBM
    lgb_mod = lgb.LGBMRegressor(
        n_estimators=350,
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
    lgb_mod.fit(X_tr_tree, y_tr)
    oof_preds[oof_idx, 2] = lgb_mod.predict(X_oof_tree)
    val_preds_models[:, 2] += lgb_mod.predict(X_tree_val) / n_splits
    if X_tree_test is not None:
        test_preds_models[:, 2] += lgb_mod.predict(X_tree_test) / n_splits

    # 4. Kernel Ridge
    krr_mod = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    krr_mod.fit(X_tr_lin, y_tr)
    oof_preds[oof_idx, 3] = krr_mod.predict(X_oof_lin)
    val_preds_models[:, 3] += krr_mod.predict(X_lin_val) / n_splits
    if X_lin_test is not None:
        test_preds_models[:, 3] += krr_mod.predict(X_lin_test) / n_splits

    # 5. Bayesian Ridge
    br_mod = BayesianRidge(
        max_iter=1000,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
    )
    br_mod.fit(X_tr_lin, y_tr)
    oof_preds[oof_idx, 4] = br_mod.predict(X_oof_lin)
    val_preds_models[:, 4] += br_mod.predict(X_lin_val) / n_splits
    if X_lin_test is not None:
        test_preds_models[:, 4] += br_mod.predict(X_lin_test) / n_splits

    # 6. Huber Regressor
    huber_mod = HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=1000)
    huber_mod.fit(X_tr_lin, y_tr)
    oof_preds[oof_idx, 5] = huber_mod.predict(X_oof_lin)
    val_preds_models[:, 5] += huber_mod.predict(X_lin_val) / n_splits
    if X_lin_test is not None:
        test_preds_models[:, 5] += huber_mod.predict(X_lin_test) / n_splits


# 4. Constrained Non-Negative Meta-Blending
def loss_func(weights):
    weights = np.array(weights)
    pred = np.dot(oof_preds, weights)
    return np.mean((y_train - pred) ** 2)


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x

# Meta-prediction with residual balancing shrinkage (0.75 * Meta + 0.25 * Simple Blend of top tree models)
val_meta_pred = np.dot(val_preds_models, best_weights)
val_simple_blend = 0.5 * val_preds_models[:, 0] + 0.5 * val_preds_models[:, 1]
final_val_preds = 0.75 * val_meta_pred + 0.25 * val_simple_blend

final_validation_score = r2_score(y_val, final_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Submission if test dataset is available
if test_preds_models is not None:
    test_meta_pred = np.dot(test_preds_models, best_weights)
    test_simple_blend = (
        0.5 * test_preds_models[:, 0] + 0.5 * test_preds_models[:, 1]
    )
    final_test_preds = 0.75 * test_meta_pred + 0.25 * test_simple_blend

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
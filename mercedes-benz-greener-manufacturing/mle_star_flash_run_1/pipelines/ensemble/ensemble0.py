import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    HuberRegressor,
    RidgeCV,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import xgboost as xgb

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

X_test = None
if test_df is not None:
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )

# 2. Unified Feature Engineering
cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]
num_cols = [c for c in X.columns if c not in cat_features]

# Eliminate zero-variance / constant columns
constant_cols = [c for c in num_cols if X[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

# Eliminate duplicated columns
dup_cols = []
seen_series = {}
for col in valid_num_cols:
    vals_tuple = tuple(X[col].values)
    if vals_tuple in seen_series:
        dup_cols.append(col)
    else:
        seen_series[vals_tuple] = col

cleaned_num_cols = [c for c in valid_num_cols if c not in dup_cols]

engineered_train = {}
engineered_test = {}

if len(cleaned_num_cols) > 0:
    X_num_clean = X[cleaned_num_cols].fillna(0)
    engineered_train["row_active_sum"] = X_num_clean.sum(axis=1).values
    engineered_train["row_active_mean"] = X_num_clean.mean(axis=1).values
    engineered_train["row_active_std"] = X_num_clean.std(axis=1).values

    if X_test is not None:
        X_test_num_clean = X_test[cleaned_num_cols].fillna(0)
        engineered_test["row_active_sum"] = X_test_num_clean.sum(axis=1).values
        engineered_test["row_active_mean"] = X_test_num_clean.mean(axis=1).values
        engineered_test["row_active_std"] = X_test_num_clean.std(axis=1).values

    n_decomp_comp = min(10, max(2, len(cleaned_num_cols) - 1)) if len(cleaned_num_cols) > 1 else 1

    # TruncatedSVD
    svd = TruncatedSVD(n_components=n_decomp_comp, random_state=42)
    svd_train = svd.fit_transform(X_num_clean)
    for i in range(n_decomp_comp):
        engineered_train[f"svd_{i}"] = svd_train[:, i]
    if X_test is not None:
        svd_test = svd.transform(X_test_num_clean)
        for i in range(n_decomp_comp):
            engineered_test[f"svd_{i}"] = svd_test[:, i]

    # PCA
    pca = PCA(n_components=n_decomp_comp, random_state=42)
    pca_train = pca.fit_transform(X_num_clean)
    for i in range(n_decomp_comp):
        engineered_train[f"pca_{i}"] = pca_train[:, i]
    if X_test is not None:
        pca_test = pca.transform(X_test_num_clean)
        for i in range(n_decomp_comp):
            engineered_test[f"pca_{i}"] = pca_test[:, i]

    # FastICA
    try:
        ica = FastICA(n_components=n_decomp_comp, random_state=42, max_iter=1000, tol=1e-3)
        ica_train = ica.fit_transform(X_num_clean)
        for i in range(n_decomp_comp):
            engineered_train[f"ica_{i}"] = ica_train[:, i]
        if X_test is not None:
            ica_test = ica.transform(X_test_num_clean)
            for i in range(n_decomp_comp):
                engineered_test[f"ica_{i}"] = ica_test[:, i]
    except Exception:
        pass

    # Gaussian Random Projection
    grp = GaussianRandomProjection(n_components=n_decomp_comp, eps=0.1, random_state=42)
    grp_train = grp.fit_transform(X_num_clean)
    for i in range(n_decomp_comp):
        engineered_train[f"grp_{i}"] = grp_train[:, i]
    if X_test is not None:
        grp_test = grp.transform(X_test_num_clean)
        for i in range(n_decomp_comp):
            engineered_test[f"grp_{i}"] = grp_test[:, i]

    # Sparse Random Projection
    srp = SparseRandomProjection(n_components=n_decomp_comp, dense_output=True, random_state=42)
    srp_train = srp.fit_transform(X_num_clean)
    for i in range(n_decomp_comp):
        engineered_train[f"srp_{i}"] = srp_train[:, i]
    if X_test is not None:
        srp_test = srp.transform(X_test_num_clean)
        for i in range(n_decomp_comp):
            engineered_test[f"srp_{i}"] = srp_test[:, i]

# Category Frequency Encodings
for col in cat_features:
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

# Filter out constant and duplicate features
cols_to_keep = [c for c in X.columns if c not in constant_cols and c not in dup_cols]
X_filtered = X[cols_to_keep].copy()
X_test_filtered = X_test[cols_to_keep].copy() if X_test is not None else None

# Label Encode categorical columns for models
X_all = pd.concat([X_filtered, df_eng_train], axis=1)
if X_test_filtered is not None:
    X_test_all = pd.concat([X_test_filtered, df_eng_test], axis=1)
else:
    X_test_all = None

for col in cat_features:
    if col in X_all.columns:
        le = LabelEncoder()
        if X_test_all is not None and col in X_test_all.columns:
            combined_vals = list(
                set(X[col].astype(str)).union(set(X_test[col].astype(str)))
            )
            le.fit(combined_vals)
            X_all[col] = le.transform(X_all[col].astype(str))
            X_test_all[col] = le.transform(X_test_all[col].astype(str))
        else:
            X_all[col] = le.fit_transform(X_all[col].astype(str))

X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
if X_test_all is not None:
    X_test_all = X_test_all.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# 3. Build Stacking Ensemble
linear_preprocessor = Pipeline(
    [
        ("scale", RobustScaler()),
    ]
)

estimators = [
    (
        "lgb",
        lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=7,
            colsample_bytree=0.6,
            subsample=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        ),
    ),
    (
        "xgb",
        xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=3,
            colsample_bytree=0.7,
            subsample=0.75,
            reg_alpha=2.0,
            reg_lambda=5.0,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        ),
    ),
    (
        "cb",
        CatBoostRegressor(
            iterations=600,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=5.0,
            subsample=0.8,
            random_seed=42,
            verbose=False,
        ),
    ),
    (
        "krr",
        Pipeline(
            [
                ("prep", linear_preprocessor),
                (
                    "model",
                    KernelRidge(
                        alpha=0.6, kernel="polynomial", degree=2, coef0=2.5
                    ),
                ),
            ]
        ),
    ),
    (
        "huber",
        Pipeline(
            [
                ("prep", linear_preprocessor),
                (
                    "model",
                    HuberRegressor(
                        epsilon=1.35, alpha=10.0, max_iter=1000
                    ),
                ),
            ]
        ),
    ),
    (
        "bayesian_ridge",
        Pipeline(
            [
                ("prep", linear_preprocessor),
                (
                    "model",
                    BayesianRidge(
                        max_iter=1000,
                        alpha_1=1e-6,
                        alpha_2=1e-6,
                        lambda_1=1e-6,
                        lambda_2=1e-6,
                    ),
                ),
            ]
        ),
    ),
]

meta_learner = RidgeCV(
    alphas=np.logspace(-1, 4, 30),
    cv=5,
)

stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False,
)

# Fit Stacking Regressor
stacking_reg.fit(X_train, y_train)

# Fit standalone tuned GBDTs for direct blended regularization
cb_standalone = CatBoostRegressor(
    iterations=800,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    subsample=0.8,
    random_seed=42,
    verbose=False,
)
cb_standalone.fit(X_train, y_train)

xgb_standalone = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.01,
    max_depth=3,
    colsample_bytree=0.7,
    subsample=0.75,
    reg_alpha=2.0,
    reg_lambda=5.0,
    random_state=42,
    tree_method="hist",
    verbosity=0,
)
xgb_standalone.fit(X_train, y_train)

# 4. Predict and Evaluate on Validation Set
stack_val_preds = stacking_reg.predict(X_val)
cb_val_preds = cb_standalone.predict(X_val)
xgb_val_preds = xgb_standalone.predict(X_val)

gbdt_val_preds = 0.5 * cb_val_preds + 0.5 * xgb_val_preds
ensemble_val_preds = 0.6 * stack_val_preds + 0.4 * gbdt_val_preds

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions if test set exists
if X_test_all is not None:
    stack_test_preds = stacking_reg.predict(X_test_all)
    cb_test_preds = cb_standalone.predict(X_test_all)
    xgb_test_preds = xgb_standalone.predict(X_test_all)
    
    gbdt_test_preds = 0.5 * cb_test_preds + 0.5 * xgb_test_preds
    final_test_preds = 0.6 * stack_test_preds + 0.4 * gbdt_test_preds

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
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
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection

# Identify numerical/binary and categorical columns
num_cols = [c for c in X.columns if c not in cat_features]

# 1. Eliminate zero-variance and duplicated binary/numerical columns
constant_cols = [c for c in num_cols if X[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

# Find duplicated columns on train
dup_cols = []
seen_series = {}
for col in valid_num_cols:
    vals_tuple = tuple(X[col].values)
    if vals_tuple in seen_series:
        dup_cols.append(col)
    else:
        seen_series[vals_tuple] = col

cleaned_num_cols = [c for c in valid_num_cols if c not in dup_cols]

# 2. Row-wise active sum and aggregate statistics on cleaned numeric/binary columns
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

    # 3. Linear decomposition embeddings on cleaned binary/numeric space
    n_decomp_comp = min(8, max(2, len(cleaned_num_cols) - 1)) if len(cleaned_num_cols) > 1 else 1

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
        ica = FastICA(n_components=n_decomp_comp, random_state=42, max_iter=500, tol=1e-3)
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

# 4. Category Frequency / Count Encodings
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

# Filter out constant and duplicate features from base DataFrame
cols_to_keep = [c for c in X.columns if c not in constant_cols and c not in dup_cols]
X_filtered = X[cols_to_keep].copy()
X_test_filtered = X_test[cols_to_keep].copy() if X_test is not None else None

# --- For CatBoost (Native Categorical Strings + Filtered & Decomposition Features) ---
X_cb = pd.concat([X_filtered, df_eng_train], axis=1)
for col in cat_features:
    if col in X_cb.columns:
        X_cb[col] = X_cb[col].astype(str)

if X_test_filtered is not None:
    X_test_cb = pd.concat([X_test_filtered, df_eng_test], axis=1)
    for col in cat_features:
        if col in X_test_cb.columns:
            X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded + Filtered & Decomposition Features) ---
X_xgb = pd.concat([X_filtered, df_eng_train], axis=1)
if X_test_filtered is not None:
    X_test_xgb = pd.concat([X_test_filtered, df_eng_test], axis=1)
else:
    X_test_xgb = None

for col in cat_features:
    if col in X_xgb.columns:
        le = LabelEncoder()
        if X_test_xgb is not None and col in X_test_xgb.columns:
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
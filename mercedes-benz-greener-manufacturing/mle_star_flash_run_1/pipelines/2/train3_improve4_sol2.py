import os
from itertools import combinations
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Separate features and target
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"]) if "ID" in train_df.columns else train_df.drop(columns=["y"])

X_test = None
if test_df is not None:
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )

# Identify categorical features
cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]

# 2. Prune zero-variance and duplicate columns
constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if len(constant_cols) > 0:
    X = X.drop(columns=constant_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

dup_cols = []
cols_to_check = list(X.columns)
for i in range(len(cols_to_check)):
    for j in range(i + 1, len(cols_to_check)):
        col_i = cols_to_check[i]
        col_j = cols_to_check[j]
        if col_j not in dup_cols and X[col_i].equals(X[col_j]):
            dup_cols.append(col_j)

if len(dup_cols) > 0:
    X = X.drop(columns=dup_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=[c for c in dup_cols if c in X_test.columns])

cat_features = [c for c in cat_features if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_features]

# 3. Categorical Interaction Features
active_cat_cols = list(cat_features)
cat_pairs = list(combinations(cat_features, 2))[:10]
inter_train_dict = {}
inter_test_dict = {}

for col1, col2 in cat_pairs:
    pair_name = f"{col1}_x_{col2}"
    inter_train_dict[pair_name] = X[col1].astype(str) + "_" + X[col2].astype(str)
    if X_test is not None:
        inter_test_dict[pair_name] = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
    active_cat_cols.append(pair_name)

if inter_train_dict:
    X = pd.concat([X, pd.DataFrame(inter_train_dict, index=X.index)], axis=1)
    if X_test is not None:
        X_test = pd.concat([X_test, pd.DataFrame(inter_test_dict, index=X_test.index)], axis=1)

# 4. Feature Engineering (Frequency, Target Encoding, Row Statistics, Decompositions)
eng_train_dict = {}
eng_test_dict = {}

# Frequency Encodings
for col in active_cat_cols:
    freq_map = X[col].astype(str).value_counts(normalize=True).to_dict()
    eng_train_dict[f"{col}_freq"] = X[col].astype(str).map(freq_map).fillna(0).values
    if X_test is not None:
        eng_test_dict[f"{col}_freq"] = X_test[col].astype(str).map(freq_map).fillna(0).values

# Smooth Out-Of-Fold Target Encoding
y_series = pd.Series(y, index=X.index).astype(float)
global_mean = y_series.mean()
smooth_weight = 10.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in active_cat_cols:
    col_str = X[col].astype(str)
    oof_te = np.full(len(X), global_mean)
    
    for tr_idx, val_idx in kf.split(X, y_series):
        X_tr_col, y_tr = col_str.iloc[tr_idx], y_series.iloc[tr_idx]
        stats = y_tr.groupby(X_tr_col).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smooth_weight * global_mean) / (stats["count"] + smooth_weight)
        oof_te[val_idx] = col_str.iloc[val_idx].map(smoothed).fillna(global_mean).values

    eng_train_dict[f"{col}_target_enc"] = oof_te

    if X_test is not None:
        full_stats = y_series.groupby(col_str).agg(["count", "mean"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth_weight * global_mean) / (full_stats["count"] + smooth_weight)
        eng_test_dict[f"{col}_target_enc"] = X_test[col].astype(str).map(full_smoothed).fillna(global_mean).values

# Row-level Binary Sparsity & Activation Statistics
numeric_clean_train = X[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0) if len(num_cols) > 0 else pd.DataFrame(index=X.index)
if len(num_cols) > 0:
    eng_train_dict["row_sum"] = numeric_clean_train.sum(axis=1).values
    eng_train_dict["row_mean"] = numeric_clean_train.mean(axis=1).values
    eng_train_dict["row_std"] = numeric_clean_train.std(axis=1).fillna(0).values
    eng_train_dict["row_zero_count"] = (numeric_clean_train == 0).sum(axis=1).values
    eng_train_dict["row_sparsity"] = (numeric_clean_train == 0).mean(axis=1).values

    if X_test is not None:
        numeric_clean_test = X_test[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
        eng_test_dict["row_sum"] = numeric_clean_test.sum(axis=1).values
        eng_test_dict["row_mean"] = numeric_clean_test.mean(axis=1).values
        eng_test_dict["row_std"] = numeric_clean_test.std(axis=1).fillna(0).values
        eng_test_dict["row_zero_count"] = (numeric_clean_test == 0).sum(axis=1).values
        eng_test_dict["row_sparsity"] = (numeric_clean_test == 0).mean(axis=1).values

# Dimensionality Reductions (PCA, TruncatedSVD, FastICA)
scaler = StandardScaler()
scaled_train_num = scaler.fit_transform(numeric_clean_train)
if X_test is not None:
    scaled_test_num = scaler.transform(numeric_clean_test)

n_components = min(6, max(2, scaled_train_num.shape[1] - 1)) if scaled_train_num.shape[1] > 1 else 1

pca = PCA(n_components=n_components, random_state=42)
pca_tr = pca.fit_transform(scaled_train_num)
for i in range(n_components):
    eng_train_dict[f"decomp_pca_{i}"] = pca_tr[:, i]
if X_test is not None:
    pca_te = pca.transform(scaled_test_num)
    for i in range(n_components):
        eng_test_dict[f"decomp_pca_{i}"] = pca_te[:, i]

svd = TruncatedSVD(n_components=n_components, random_state=42)
svd_tr = svd.fit_transform(scaled_train_num)
for i in range(n_components):
    eng_train_dict[f"decomp_svd_{i}"] = svd_tr[:, i]
if X_test is not None:
    svd_te = svd.transform(scaled_test_num)
    for i in range(n_components):
        eng_test_dict[f"decomp_svd_{i}"] = svd_te[:, i]

try:
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500, tol=1e-3)
    ica_tr = ica.fit_transform(scaled_train_num)
    for i in range(n_components):
        eng_train_dict[f"decomp_ica_{i}"] = ica_tr[:, i]
    if X_test is not None:
        ica_te = ica.transform(scaled_test_num)
        for i in range(n_components):
            eng_test_dict[f"decomp_ica_{i}"] = ica_te[:, i]
except Exception:
    pass

engineered_train = pd.DataFrame(eng_train_dict, index=X.index)
engineered_test = pd.DataFrame(eng_test_dict, index=X_test.index) if X_test is not None else None

# 5. Assemble Datasets
X_cb = pd.concat([X.copy(), engineered_train], axis=1)
for col in active_cat_cols:
    X_cb[col] = X_cb[col].astype(str)

non_cat_cb_cols = [c for c in X_cb.columns if c not in active_cat_cols]
for c in non_cat_cb_cols:
    X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce").fillna(0)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), engineered_test], axis=1)
    for col in active_cat_cols:
        X_test_cb[col] = X_test_cb[col].astype(str)
    for c in non_cat_cb_cols:
        X_test_cb[c] = pd.to_numeric(X_test_cb[c], errors="coerce").fillna(0)
else:
    X_test_cb = None

# Dataset for XGBoost
X_xgb = pd.concat([X.copy(), engineered_train], axis=1)
if X_test is not None:
    X_test_xgb = pd.concat([X_test.copy(), engineered_test], axis=1)
else:
    X_test_xgb = None

for col in active_cat_cols:
    le = LabelEncoder()
    if X_test is not None and col in X_test.columns:
        combined_vals = list(set(X[col].astype(str)).union(set(X_test[col].astype(str))))
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

# 6. Model Training
# --- CatBoost ---
cb_model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    subsample=0.8,
    random_seed=42,
    cat_features=active_cat_cols,
    eval_metric="RMSE",
    early_stopping_rounds=50,
    verbose=False,
)
cb_model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), verbose=False)

# --- XGBoost ---
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

# 7. Ensemble Validation Predictions
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)
ensemble_val_preds = 0.5 * val_preds_cb + 0.5 * val_preds_xgb

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 8. Generate Test Predictions
if X_test is not None:
    test_preds_cb = cb_model.predict(X_test_cb)
    dtest = xgb.DMatrix(X_test_xgb)
    test_preds_xgb = xgb_model.predict(dtest)
    final_test_preds = 0.5 * test_preds_cb + 0.5 * test_preds_xgb

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
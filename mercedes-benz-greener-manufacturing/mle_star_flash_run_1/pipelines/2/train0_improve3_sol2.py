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

# 2. Feature Preprocessing
# --- For CatBoost (Native Categorical Strings) ---
X_cb = X.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = X_test.copy()
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded) ---
X_xgb = X.copy()
if X_test is not None:
    X_test_xgb = X_test.copy()
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

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize, nnls
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 1. Data Preparation and Cleaning
# Use X_cb or X_xgb base dataframe to construct enriched feature sets
if isinstance(X_cb, pd.DataFrame):
    df_train = X_cb.copy()
else:
    df_train = pd.DataFrame(X_cb)

if X_test is not None:
    if isinstance(X_test_cb if "X_test_cb" in locals() else X_test, pd.DataFrame):
        df_test = (X_test_cb if "X_test_cb" in locals() else X_test).copy()
    else:
        df_test = pd.DataFrame(X_test_cb if "X_test_cb" in locals() else X_test)
else:
    df_test = None

y_arr = np.array(y).ravel()

# Identify categorical and numerical columns
cat_cols = [
    col
    for col in df_train.columns
    if df_train[col].dtype == "object"
    or str(col).startswith("X")
    and df_train[col].dtype not in [np.number, int, float]
    or (
        isinstance(col, str)
        and col
        in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8", "cat_features"]
    )
]
if "cat_features" in locals() and isinstance(cat_features, (list, tuple)):
    for cf in cat_features:
        if cf in df_train.columns and cf not in cat_cols:
            cat_cols.append(cf)

# Apply Label Encoding and Frequency Encoding on categorical features
df_train_proc = pd.DataFrame(index=df_train.index)
df_test_proc = pd.DataFrame(index=df_test.index) if df_test is not None else None

for c in cat_cols:
    le = LabelEncoder()
    train_vals = df_train[c].astype(str)
    if df_test is not None:
        test_vals = df_test[c].astype(str)
        le.fit(list(train_vals) + list(test_vals))
        df_test_proc[f"{c}_le"] = le.transform(test_vals)
    else:
        le.fit(train_vals)
    df_train_proc[f"{c}_le"] = le.transform(train_vals)

    # Frequency encoding
    freq = train_vals.value_counts(normalize=True).to_dict()
    df_train_proc[f"{c}_freq"] = train_vals.map(freq).fillna(0).astype(float)
    if df_test is not None:
        df_test_proc[f"{c}_freq"] = test_vals.map(freq).fillna(0).astype(float)

# Copy numerical/binary features
num_cols = [c for c in df_train.columns if c not in cat_cols]
for c in num_cols:
    df_train_proc[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(0)
    if df_test is not None:
        df_test_proc[c] = pd.to_numeric(df_test[c], errors="coerce").fillna(0)

# Remove zero-variance (constant) columns
non_const_cols = [
    c for c in df_train_proc.columns if df_train_proc[c].std() > 1e-7
]
df_train_proc = df_train_proc[non_const_cols]
if df_test_proc is not None:
    df_test_proc = df_test_proc[non_const_cols]

# Remove duplicate binary/numerical columns
transposed = df_train_proc.T
unique_cols = transposed.drop_duplicates().T.columns.tolist()
df_train_proc = df_train_proc[unique_cols]
if df_test_proc is not None:
    df_test_proc = df_test_proc[unique_cols]

# Low-dimensional decomposition features (PCA, FastICA, TruncatedSVD)
n_comp = min(10, df_train_proc.shape[1])
pca = PCA(n_components=n_comp, random_state=42)
ica = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=1e-3)
tsvd = TruncatedSVD(n_components=n_comp, random_state=42)

pca_train = pca.fit_transform(df_train_proc)
ica_train = ica.fit_transform(df_train_proc)
tsvd_train = tsvd.fit_transform(df_train_proc)

decomp_train = pd.DataFrame(
    np.hstack([pca_train, ica_train, tsvd_train]),
    columns=[f"pca_{i}" for i in range(n_comp)]
    + [f"ica_{i}" for i in range(n_comp)]
    + [f"tsvd_{i}" for i in range(n_comp)],
    index=df_train_proc.index,
)
X_all = pd.concat([df_train_proc, decomp_train], axis=1)

if df_test_proc is not None:
    pca_test = pca.transform(df_test_proc)
    ica_test = ica.transform(df_test_proc)
    tsvd_test = tsvd.transform(df_test_proc)
    decomp_test = pd.DataFrame(
        np.hstack([pca_test, ica_test, tsvd_test]),
        columns=[f"pca_{i}" for i in range(n_comp)]
        + [f"ica_{i}" for i in range(n_comp)]
        + [f"tsvd_{i}" for i in range(n_comp)],
        index=df_test_proc.index,
    )
    X_test_all = pd.concat([df_test_proc, decomp_test], axis=1)
else:
    X_test_all = None

# 2. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(y_arr))
oof_lgb = np.zeros(len(y_arr))
oof_cb = np.zeros(len(y_arr))
oof_et = np.zeros(len(y_arr))

test_preds_xgb = np.zeros(len(X_test_all)) if X_test_all is not None else None
test_preds_lgb = np.zeros(len(X_test_all)) if X_test_all is not None else None
test_preds_cb = np.zeros(len(X_test_all)) if X_test_all is not None else None
test_preds_et = np.zeros(len(X_test_all)) if X_test_all is not None else None

# 3. Model Training with Shallow, Regularized Ensembles
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_arr)):
    X_tr, y_tr = X_all.iloc[train_idx], y_arr[train_idx]
    X_va, y_va = X_all.iloc[val_idx], y_arr[val_idx]

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.008,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "alpha": 2.0,
        "lambda": 5.0,
        "random_state": 42 + fold,
        "tree_method": "hist",
    }
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(dval)
    if X_test_all is not None:
        dtest = xgb.DMatrix(X_test_all)
        test_preds_xgb += xgb_model.predict(dtest) / n_splits

    # --- LightGBM ---
    lgb_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.008,
        max_depth=3,
        num_leaves=7,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    if X_test_all is not None:
        test_preds_lgb += lgb_model.predict(X_test_all) / n_splits

    # --- CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.01,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_va)
    if X_test_all is not None:
        test_preds_cb += cb_model.predict(X_test_all) / n_splits

    # --- ExtraTrees ---
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        max_features=0.6,
        min_samples_leaf=3,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict(X_va)
    if X_test_all is not None:
        test_preds_et += et_model.predict(X_test_all) / n_splits

# 4. Optimal Non-Negative Weighting Optimization
OOF_matrix = np.column_stack([oof_xgb, oof_lgb, oof_cb, oof_et])


def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(0, weights)
    if weights.sum() > 0:
        weights /= weights.sum()
    pred = OOF_matrix @ weights
    return -r2_score(y_arr, pred)


init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)

weights = opt_res.x
if weights.sum() > 0:
    weights = weights / weights.sum()

oof_ensemble = OOF_matrix @ weights
lower_bound, upper_bound = np.percentile(y_arr, 0.05), np.percentile(
    y_arr, 99.95
)
oof_ensemble_clipped = np.clip(oof_ensemble, lower_bound, upper_bound)

final_validation_score = r2_score(y_arr, oof_ensemble_clipped)
print(f"Ensemble Model Weights (XGB, LGB, CB, ET): {np.round(weights, 4)}")
print(f"Final Validation Performance (5-Fold CV R2): {final_validation_score}")

# 5. Generate Test Predictions with Boundary Clipping
if X_test_all is not None:
    Test_matrix = np.column_stack(
        [test_preds_xgb, test_preds_lgb, test_preds_cb, test_preds_et]
    )
    final_test_preds = Test_matrix @ weights
    final_test_preds = np.clip(final_test_preds, lower_bound, upper_bound)

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
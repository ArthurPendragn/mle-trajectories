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
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Dimensionality Reduction Features Extraction
# Identify numeric/binary feature columns (excluding categorical columns)
if isinstance(X_cb, pd.DataFrame):
    num_cols = [c for c in X_cb.columns if c not in cat_features]
    train_num_data = X_cb[num_cols].values
    test_num_data = (
        X_test_cb[num_cols].values if X_test is not None else None
    )
else:
    train_num_data = X_cb
    test_num_data = X_test_cb if X_test is not None else None

n_comp = min(10, train_num_data.shape[1])
pca = PCA(n_components=n_comp, random_state=42)
ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
tsvd = TruncatedSVD(n_components=n_comp, random_state=42)

pca_train = pca.fit_transform(train_num_data)
ica_train = ica.fit_transform(train_num_data)
tsvd_train = tsvd.fit_transform(train_num_data)

decomp_train = pd.DataFrame(
    np.hstack([pca_train, ica_train, tsvd_train]),
    columns=[f"pca_{i}" for i in range(n_comp)]
    + [f"ica_{i}" for i in range(n_comp)]
    + [f"tsvd_{i}" for i in range(n_comp)],
    index=X_cb.index if isinstance(X_cb, pd.DataFrame) else None,
)

if isinstance(X_cb, pd.DataFrame):
    X_cb_enhanced = pd.concat([X_cb, decomp_train], axis=1)
else:
    X_cb_enhanced = np.hstack([X_cb, decomp_train.values])

if isinstance(X_xgb, pd.DataFrame):
    X_xgb_enhanced = pd.concat([X_xgb, decomp_train], axis=1)
else:
    X_xgb_enhanced = np.hstack([X_xgb, decomp_train.values])

if X_test is not None:
    pca_test = pca.transform(test_num_data)
    ica_test = ica.transform(test_num_data)
    tsvd_test = tsvd.transform(test_num_data)

    decomp_test = pd.DataFrame(
        np.hstack([pca_test, ica_test, tsvd_test]),
        columns=[f"pca_{i}" for i in range(n_comp)]
        + [f"ica_{i}" for i in range(n_comp)]
        + [f"tsvd_{i}" for i in range(n_comp)],
        index=X_test_cb.index if isinstance(X_test_cb, pd.DataFrame) else None,
    )

    if isinstance(X_test_cb, pd.DataFrame):
        X_test_cb_enhanced = pd.concat([X_test_cb, decomp_test], axis=1)
    else:
        X_test_cb_enhanced = np.hstack([X_test_cb, decomp_test.values])

    if isinstance(X_test_xgb, pd.DataFrame):
        X_test_xgb_enhanced = pd.concat([X_test_xgb, decomp_test], axis=1)
    else:
        X_test_xgb_enhanced = np.hstack([X_test_xgb, decomp_test.values])
else:
    X_test_cb_enhanced = None
    X_test_xgb_enhanced = None

# 2. 5-Fold Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_cb = np.zeros(len(y))
oof_preds_xgb = np.zeros(len(y))

if X_test is not None:
    test_preds_cb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    dtest = xgb.DMatrix(X_test_xgb_enhanced)

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

y_values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

# 3. Model Training across Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_cb_enhanced, y_values)):
    # Slice datasets
    if isinstance(X_cb_enhanced, pd.DataFrame):
        X_train_cb_fold = X_cb_enhanced.iloc[train_idx]
        X_val_cb_fold = X_cb_enhanced.iloc[val_idx]
        X_train_xgb_fold = X_xgb_enhanced.iloc[train_idx]
        X_val_xgb_fold = X_xgb_enhanced.iloc[val_idx]
    else:
        X_train_cb_fold = X_cb_enhanced[train_idx]
        X_val_cb_fold = X_cb_enhanced[val_idx]
        X_train_xgb_fold = X_xgb_enhanced[train_idx]
        X_val_xgb_fold = X_xgb_enhanced[val_idx]

    y_train_fold = y_values[train_idx]
    y_val_fold = y_values[val_idx]

    # --- CatBoost ---
    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42 + fold,
        cat_features=cat_features,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(
        X_train_cb_fold,
        y_train_fold,
        eval_set=(X_val_cb_fold, y_val_fold),
        verbose=False,
    )

    oof_preds_cb[val_idx] = cb_model.predict(X_val_cb_fold)
    if X_test is not None:
        test_preds_cb += cb_model.predict(X_test_cb_enhanced) / n_splits

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_train_xgb_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_xgb_fold, label=y_val_fold)

    xgb_fold_params = xgb_params.copy()
    xgb_fold_params["random_state"] = 42 + fold

    xgb_model = xgb.train(
        xgb_fold_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    oof_preds_xgb[val_idx] = xgb_model.predict(dval)
    if X_test is not None:
        test_preds_xgb += xgb_model.predict(dtest) / n_splits


# 4. Optimize Ensemble Blending Weights
def loss_func(weights):
    w_cb, w_xgb = weights
    pred = w_cb * oof_preds_cb + w_xgb * oof_preds_xgb
    return -r2_score(y_values, pred)


initial_weights = [0.5, 0.5]
bounds = [(0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    loss_func,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_w_cb, best_w_xgb = opt_res.x

ensemble_oof_preds = best_w_cb * oof_preds_cb + best_w_xgb * oof_preds_xgb
final_validation_score = r2_score(y_values, ensemble_oof_preds)
print(f"Optimal Weights: CatBoost={best_w_cb:.4f}, XGBoost={best_w_xgb:.4f}")
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions
if X_test is not None:
    final_test_preds = best_w_cb * test_preds_cb + best_w_xgb * test_preds_xgb
    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, PowerTransformer

# 2. Robust Non-linear Transforms, Quantile Binning & Interaction Engineering
num_cols = [c for c in X.columns if c not in cat_features]
has_test = X_test is not None

engineered_train = pd.DataFrame(index=X.index)
engineered_test = pd.DataFrame(index=X_test.index) if has_test else None

if len(num_cols) > 0:
    # Handle missing values robustly for transformation
    X_num_train = X[num_cols].copy()
    X_num_test = X_test[num_cols].copy() if has_test else None

    med_vals = X_num_train.median(numeric_only=True).fillna(0)
    X_num_train_clean = X_num_train.fillna(med_vals)
    X_num_test_clean = X_num_test.fillna(med_vals) if has_test else None

    # 1. Non-linear Yeo-Johnson / Power Transformations on Skewed Numerical Features
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    try:
        pt_train = pt.fit_transform(X_num_train_clean)
        for i, col in enumerate(num_cols):
            engineered_train[f"{col}_yj"] = pt_train[:, i]
            if has_test:
                engineered_test[f"{col}_yj"] = pt.transform(X_num_test_clean)[:, i]
    except Exception:
        # Fallback to log1p if power transformer encounters numerical issues
        for col in num_cols:
            min_val = X_num_train_clean[col].min()
            shift = max(0.0, -min_val + 1e-5)
            engineered_train[f"{col}_log1p"] = np.log1p(X_num_train_clean[col] + shift)
            if has_test:
                engineered_test[f"{col}_log1p"] = np.log1p(X_num_test_clean[col] + shift)

    # 2. Robust Rank/Percentile Encodings
    for col in num_cols:
        train_vals = X_num_train_clean[col].values
        # Compute percentile rank based on training distribution
        engineered_train[f"{col}_pct_rank"] = pd.Series(train_vals, index=X.index).rank(pct=True).values
        if has_test:
            test_vals = X_num_test_clean[col].values
            # Target-agnostic percentile mapping from training quantiles
            engineered_test[f"{col}_pct_rank"] = np.searchsorted(np.sort(train_vals), test_vals) / float(len(train_vals))

    # 3. Quantile Binning & Categorical-Continuous Interactions
    n_bins = min(8, max(3, len(X) // 100))
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", random_state=42)
    try:
        binned_train = kbd.fit_transform(X_num_train_clean)
        for i, col in enumerate(num_cols):
            engineered_train[f"{col}_qbin"] = binned_train[:, i]
            if has_test:
                engineered_test[f"{col}_qbin"] = kbd.transform(X_num_test_clean)[:, i]
    except Exception:
        pass

    # 4. Pairwise Numerical Ratios/Differences for high-variance pairs
    top_num_cols = num_cols[: min(5, len(num_cols))]
    for i in range(len(top_num_cols)):
        for j in range(i + 1, len(top_num_cols)):
            col1, col2 = top_num_cols[i], top_num_cols[j]
            engineered_train[f"{col1}_sub_{col2}"] = X_num_train_clean[col1] - X_num_train_clean[col2]
            engineered_train[f"{col1}_ratio_{col2}"] = X_num_train_clean[col1] / (np.abs(X_num_train_clean[col2]) + 1e-6)
            if has_test:
                engineered_test[f"{col1}_sub_{col2}"] = X_num_test_clean[col1] - X_num_test_clean[col2]
                engineered_test[f"{col1}_ratio_{col2}"] = X_num_test_clean[col1] / (np.abs(X_num_test_clean[col2]) + 1e-6)

# 5. Filter out collinear or near-zero variance engineered features
if engineered_train.shape[1] > 0:
    engineered_train_clean = engineered_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    vt = VarianceThreshold(threshold=1e-4)
    vt.fit(engineered_train_clean)
    selected_cols = engineered_train.columns[vt.get_support()]

    engineered_train = engineered_train_clean[selected_cols]
    if has_test:
        engineered_test_clean = engineered_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        engineered_test = engineered_test_clean[selected_cols]

# --- For CatBoost (Native Categorical Strings + Transformed Numerical Features) ---
X_cb = pd.concat([X.copy(), engineered_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if has_test:
    X_test_cb = pd.concat([X_test.copy(), engineered_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded Categoricals + Transformed Numerical Features) ---
X_xgb = pd.concat([X.copy(), engineered_train], axis=1)
if has_test:
    X_test_xgb = pd.concat([X_test.copy(), engineered_test], axis=1)
else:
    X_test_xgb = None

for col in cat_features:
    le = LabelEncoder()
    if has_test and col in X_test.columns:
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
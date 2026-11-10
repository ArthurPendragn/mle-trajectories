import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 1. Load Training Data
train_df = pd.read_csv("./input/train.csv")
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]

# 2. Feature Preprocessing
# --- For CatBoost (Native Categorical Strings) ---
X_cb = X.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

# --- For XGBoost (Label Encoded) ---
X_xgb = X.copy()
for col in cat_features:
    le = LabelEncoder()
    X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))

X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train/Validation Split
X_train_cb, X_val_cb, y_train, y_val = train_test_split(
    X_cb, y, test_size=0.2, random_state=42
)
X_train_xgb, X_val_xgb, _, _ = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42
)

# 3. Baseline Model Training
# CatBoost
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

# XGBoost
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

# 4. Predictions & Ablation Evaluations
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)

# Full Pipeline (CatBoost + XGBoost Ensemble)
full_ensemble_preds = 0.5 * val_preds_cb + 0.5 * val_preds_xgb
baseline_r2 = r2_score(y_val, full_ensemble_preds)

# Ablation 1: Disable XGBoost (CatBoost only)
cb_only_r2 = r2_score(y_val, val_preds_cb)
cb_diff = baseline_r2 - cb_only_r2

# Ablation 2: Disable CatBoost (XGBoost only)
xgb_only_r2 = r2_score(y_val, val_preds_xgb)
xgb_diff = baseline_r2 - xgb_only_r2

# Print Ablation Study Results
print("=== Ablation Study Results ===")
print(f"Full Ensemble (Baseline) R2: {baseline_r2:.5f}")
print(f"Ablation 1 (CatBoost Only - Disabling XGBoost) R2: {cb_only_r2:.5f} (Delta: {-cb_diff:+.5f})")
print(f"Ablation 2 (XGBoost Only - Disabling CatBoost) R2: {xgb_only_r2:.5f} (Delta: {-xgb_diff:+.5f})")

# Determine which component contributes the most
# The component whose removal causes the largest drop in R2 contributes the most
contributions = {
    "XGBoost Model Component": cb_diff,
    "CatBoost Model Component": xgb_diff,
}

top_component = max(contributions, key=contributions.get)
print("\n=== Summary ===")
print(f"The most critical component contributing to overall performance is: {top_component} (Performance drop without it: {contributions[top_component]:.5f} R2)")
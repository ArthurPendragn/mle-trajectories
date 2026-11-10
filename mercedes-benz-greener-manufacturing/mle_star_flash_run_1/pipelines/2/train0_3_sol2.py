import os
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
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

# --- For Stacking Regressor (One-Hot Encoded) ---
if X_test is not None:
    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    combined_ohe = pd.get_dummies(combined, drop_first=True)
    X_ohe = combined_ohe.iloc[: len(X)].values.astype(np.float64)
    X_test_ohe = combined_ohe.iloc[len(X) :].values.astype(np.float64)
else:
    X_ohe = pd.get_dummies(X, drop_first=True).values.astype(np.float64)
    X_test_ohe = None

# Train/Validation Split
X_train_cb, X_val_cb, y_train, y_val = train_test_split(
    X_cb, y, test_size=0.2, random_state=42
)
X_train_xgb, X_val_xgb, _, _ = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42
)
X_train_ohe, X_val_ohe, _, _ = train_test_split(
    X_ohe, y, test_size=0.2, random_state=42
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

# --- Train Stacking Regressor ---
base_models = [
    (
        "elastic",
        Pipeline(
            [
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=25, random_state=42)),
                (
                    "reg",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42
                    ),
                ),
            ]
        ),
    ),
    (
        "kernel_ridge",
        Pipeline(
            [
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=20, random_state=42)),
                ("reg", KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
            ]
        ),
    ),
]

stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=ElasticNetCV(cv=5, random_state=42),
    cv=5,
    n_jobs=-1,
)
stacking_reg.fit(X_train_ohe, y_train)

# 4. Ensemble Validation Predictions
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)
val_preds_stack = stacking_reg.predict(X_val_ohe)

ensemble_val_preds = (val_preds_cb + val_preds_xgb + val_preds_stack) / 3.0

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions
if X_test is not None:
    test_preds_cb = cb_model.predict(X_test_cb)
    dtest = xgb.DMatrix(X_test_xgb)
    test_preds_xgb = xgb_model.predict(dtest)
    test_preds_stack = stacking_reg.predict(X_test_ohe)

    final_test_preds = (
        test_preds_cb + test_preds_xgb + test_preds_stack
    ) / 3.0

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
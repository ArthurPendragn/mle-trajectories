import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
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

# Identify categorical and binary/numeric features
cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]
binary_cols = [c for c in X.columns if c not in cat_features]

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

# --- For XGBoost & LightGBM (Label Encoded) ---
X_encoded = X.copy()
if X_test is not None:
    X_test_encoded = X_test.copy()
else:
    X_test_encoded = None

for col in cat_features:
    le = LabelEncoder()
    if X_test is not None and col in X_test.columns:
        combined_vals = list(
            set(X[col].astype(str)).union(set(X_test[col].astype(str)))
        )
        le.fit(combined_vals)
        X_encoded[col] = le.transform(X_encoded[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
    else:
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
if X_test_encoded is not None:
    X_test_encoded = X_test_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train/Validation Split
X_train_cb, X_val_cb, y_train, y_val = train_test_split(
    X_cb, y, test_size=0.2, random_state=42
)
X_train_enc, X_val_enc, _, _ = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# --- Dimensionality Reduction on Binary Features ---
decomp_union = FeatureUnion(
    [
        ("pca", PCA(n_components=10, random_state=42)),
        ("ica", FastICA(n_components=10, random_state=42, max_iter=1000)),
        ("tsvd", TruncatedSVD(n_components=10, random_state=42)),
    ]
)

decomp_train = decomp_union.fit_transform(
    X_train_enc[binary_cols].values.astype(np.float64)
)
decomp_val = decomp_union.transform(
    X_val_enc[binary_cols].values.astype(np.float64)
)

X_train_lgb = np.hstack([X_train_enc.values.astype(np.float64), decomp_train])
X_val_lgb = np.hstack([X_val_enc.values.astype(np.float64), decomp_val])

if X_test_encoded is not None:
    decomp_test = decomp_union.transform(
        X_test_encoded[binary_cols].values.astype(np.float64)
    )
    X_test_lgb = np.hstack(
        [X_test_encoded.values.astype(np.float64), decomp_test]
    )
else:
    X_test_lgb = None

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
dtrain = xgb.DMatrix(X_train_enc, label=y_train)
dval = xgb.DMatrix(X_val_enc, label=y_val)

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

# --- Train LightGBM ---
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(
    X_train_lgb,
    y_train,
    eval_set=[(X_val_lgb, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# 4. Ensemble Validation Predictions
val_preds_cb = cb_model.predict(X_val_cb)
val_preds_xgb = xgb_model.predict(dval)
val_preds_lgb = lgb_model.predict(X_val_lgb)

ensemble_val_preds = (
    val_preds_cb * (1 / 3) + val_preds_xgb * (1 / 3) + val_preds_lgb * (1 / 3)
)

final_validation_score = r2_score(y_val, ensemble_val_preds)
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Test Predictions
if X_test is not None:
    test_preds_cb = cb_model.predict(X_test_cb)
    dtest = xgb.DMatrix(X_test_encoded)
    test_preds_xgb = xgb_model.predict(dtest)
    test_preds_lgb = lgb_model.predict(X_test_lgb)

    final_test_preds = (
        test_preds_cb * (1 / 3)
        + test_preds_xgb * (1 / 3)
        + test_preds_lgb * (1 / 3)
    )

    submission = pd.DataFrame({"y": final_test_preds})
    submission.to_csv("submission.csv", index=False)
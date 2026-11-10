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

# 2. Feature Engineering & Preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Identify numerical features
num_features = [col for col in X.columns if col not in cat_features]

# Combine train and test distributions for consistent statistics and encoding
if X_test is not None:
    df_all = pd.concat([X, X_test], axis=0, ignore_index=True)
else:
    df_all = X.copy()

# Ensure numerical columns in df_all are numeric
if num_features:
    df_all_num = df_all[num_features].apply(pd.to_numeric, errors="coerce")
    X_num = X[num_features].apply(pd.to_numeric, errors="coerce")
    if X_test is not None:
        X_test_num = X_test[num_features].apply(pd.to_numeric, errors="coerce")
else:
    df_all_num = pd.DataFrame(index=df_all.index)
    X_num = pd.DataFrame(index=X.index)
    X_test_num = pd.DataFrame(index=X_test.index) if X_test is not None else None

# --- A. Frequency Encoding for Categorical Features ---
freq_feats_train = pd.DataFrame(index=X.index)
freq_feats_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

for col in cat_features:
    freq_map = df_all[col].astype(str).value_counts(normalize=True).to_dict()
    freq_feats_train[f"{col}_freq"] = X[col].astype(str).map(freq_map).fillna(0).astype(np.float32)
    if X_test is not None:
        freq_feats_test[f"{col}_freq"] = X_test[col].astype(str).map(freq_map).fillna(0).astype(np.float32)

# --- B. Row-wise Statistical Aggregations on Numerical Features ---
row_stats_train = pd.DataFrame(index=X.index)
row_stats_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

if len(num_features) > 1:
    row_stats_train["row_mean"] = X_num.mean(axis=1).astype(np.float32)
    row_stats_train["row_std"] = X_num.std(axis=1).fillna(0).astype(np.float32)
    row_stats_train["row_sum"] = X_num.sum(axis=1).astype(np.float32)
    row_stats_train["row_min"] = X_num.min(axis=1).astype(np.float32)
    row_stats_train["row_max"] = X_num.max(axis=1).astype(np.float32)
    row_stats_train["row_nonzero_count"] = (X_num != 0).sum(axis=1).astype(np.float32)
    row_stats_train["row_null_count"] = X_num.isna().sum(axis=1).astype(np.float32)

    if X_test is not None:
        row_stats_test["row_mean"] = X_test_num.mean(axis=1).astype(np.float32)
        row_stats_test["row_std"] = X_test_num.std(axis=1).fillna(0).astype(np.float32)
        row_stats_test["row_sum"] = X_test_num.sum(axis=1).astype(np.float32)
        row_stats_test["row_min"] = X_test_num.min(axis=1).astype(np.float32)
        row_stats_test["row_max"] = X_test_num.max(axis=1).astype(np.float32)
        row_stats_test["row_nonzero_count"] = (X_test_num != 0).sum(axis=1).astype(np.float32)
        row_stats_test["row_null_count"] = X_test_num.isna().sum(axis=1).astype(np.float32)

# --- C. Cross-Feature Group Aggregations ---
grp_feats_train = pd.DataFrame(index=X.index)
grp_feats_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

# Select key categorical and numerical features for aggregation
key_cats = [col for col in cat_features if df_all[col].nunique() < 100][:4]
key_nums = num_features[:4]

for c_col in key_cats:
    c_str_all = df_all[c_col].astype(str)
    c_str_train = X[c_col].astype(str)
    c_str_test = X_test[c_col].astype(str) if X_test is not None else None
    for n_col in key_nums:
        mean_map = df_all_num[n_col].groupby(c_str_all).mean().to_dict()
        std_map = df_all_num[n_col].groupby(c_str_all).std().fillna(0).to_dict()

        col_mean_name = f"grp_{c_col}_{n_col}_mean"
        col_std_name = f"grp_{c_col}_{n_col}_std"

        grp_feats_train[col_mean_name] = c_str_train.map(mean_map).fillna(0).astype(np.float32)
        grp_feats_train[col_std_name] = c_str_train.map(std_map).fillna(0).astype(np.float32)

        if X_test is not None:
            grp_feats_test[col_mean_name] = c_str_test.map(mean_map).fillna(0).astype(np.float32)
            grp_feats_test[col_std_name] = c_str_test.map(std_map).fillna(0).astype(np.float32)

# Combine all engineered tabular features
eng_train = pd.concat([freq_feats_train, row_stats_train, grp_feats_train], axis=1)
eng_test = pd.concat([freq_feats_test, row_stats_test, grp_feats_test], axis=1) if X_test is not None else None

# --- D. Finalize CatBoost Datasets (Native Categoricals + Engineered Numerics) ---
X_cb = pd.concat([X.copy(), eng_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.copy(), eng_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- E. Finalize XGBoost Datasets (Label Encoded + Engineered Numerics) ---
X_xgb = pd.concat([X.copy(), eng_train], axis=1)
if X_test is not None:
    X_test_xgb = pd.concat([X_test.copy(), eng_test], axis=1)
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
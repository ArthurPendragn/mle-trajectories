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

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 2. Feature Engineering & Preprocessing
num_features = [col for col in X.columns if col not in cat_features]

X_feat = X.copy()
if X_test is not None:
    X_test_feat = X_test.copy()
else:
    X_test_feat = None

# --- A. Row-level Statistical Summaries ---
if len(num_features) > 0:
    for df in [X_feat, X_test_feat]:
        if df is not None:
            df["row_null_count"] = df[num_features].isna().sum(axis=1)
            df["row_num_mean"] = df[num_features].mean(axis=1)
            df["row_num_std"] = df[num_features].std(axis=1).fillna(0)
            df["row_num_l2"] = np.sqrt((df[num_features].fillna(0) ** 2).sum(axis=1))

# --- B. Frequency Encoding & Pairwise Categorical Interactions ---
new_cat_features = list(cat_features)

for col in cat_features:
    freq_map = X_feat[col].astype(str).value_counts(normalize=True).to_dict()
    X_feat[f"{col}_freq"] = X_feat[col].astype(str).map(freq_map).fillna(0)
    if X_test_feat is not None:
        X_test_feat[f"{col}_freq"] = X_test_feat[col].astype(str).map(freq_map).fillna(0)

# Pairwise categorical interactions (limit to top pairs to avoid explosion)
top_cats = cat_features[: min(len(cat_features), 5)]
for col1, col2 in itertools.combinations(top_cats, 2):
    inter_col = f"{col1}_x_{col2}"
    X_feat[inter_col] = X_feat[col1].astype(str) + "_" + X_feat[col2].astype(str)
    if X_test_feat is not None:
        X_test_feat[inter_col] = (
            X_test_feat[col1].astype(str) + "_" + X_test_feat[col2].astype(str)
        )
    new_cat_features.append(inter_col)

# --- C. Grouped Numerical Aggregations ---
if len(cat_features) > 0 and len(num_features) > 0:
    group_cats = cat_features[: min(len(cat_features), 3)]
    target_nums = num_features[: min(len(num_features), 6)]
    for cat_col in group_cats:
        for num_col in target_nums:
            agg_stats = (
                X_feat.groupby(cat_col)[num_col]
                .agg(["mean", "std"])
                .rename(
                    columns={
                        "mean": f"{num_col}_grp_{cat_col}_mean",
                        "std": f"{num_col}_grp_{cat_col}_std",
                    }
                )
            )
            mean_col = f"{num_col}_grp_{cat_col}_mean"
            std_col = f"{num_col}_grp_{cat_col}_std"

            X_feat = X_feat.merge(
                agg_stats, on=cat_col, how="left"
            )
            X_feat[std_col] = X_feat[std_col].fillna(0)
            X_feat[f"{num_col}_diff_grp_{cat_col}"] = X_feat[num_col] - X_feat[mean_col]
            X_feat[f"{num_col}_ratio_grp_{cat_col}"] = X_feat[num_col] / (
                X_feat[mean_col].abs() + 1e-6
            )

            if X_test_feat is not None:
                X_test_feat = X_test_feat.merge(
                    agg_stats, on=cat_col, how="left"
                )
                X_test_feat[std_col] = X_test_feat[std_col].fillna(0)
                X_test_feat[f"{num_col}_diff_grp_{cat_col}"] = (
                    X_test_feat[num_col] - X_test_feat[mean_col]
                )
                X_test_feat[f"{num_col}_ratio_grp_{cat_col}"] = (
                    X_test_feat[num_col] / (X_test_feat[mean_col].abs() + 1e-6)
                )

# --- For CatBoost (Native Categorical Strings) ---
X_cb = X_feat.copy()
for col in new_cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test_feat is not None:
    X_test_cb = X_test_feat.copy()
    for col in new_cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded & Fully Numeric) ---
X_xgb = X_feat.copy()
if X_test_feat is not None:
    X_test_xgb = X_test_feat.copy()
else:
    X_test_xgb = None

for col in new_cat_features:
    le = LabelEncoder()
    if X_test_xgb is not None and col in X_test_xgb.columns:
        combined_vals = list(
            set(X_xgb[col].astype(str)).union(set(X_test_xgb[col].astype(str)))
        )
        le.fit(combined_vals)
        X_xgb[col] = le.transform(X_xgb[col].astype(str))
        X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    else:
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))

X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)
if X_test_xgb is not None:
    X_test_xgb = X_test_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

cat_features = new_cat_features

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
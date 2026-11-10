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
from sklearn.preprocessing import LabelEncoder

# 2. Feature Engineering & Preprocessing
num_features = [c for c in X.columns if c not in cat_features]

X_feat = X.copy()
X_test_feat = X_test.copy() if X_test is not None else None

# Combine temporarily for consistent reference statistics and transformations
combined_df = (
    pd.concat([X_feat, X_test_feat], axis=0, ignore_index=True)
    if X_test_feat is not None
    else X_feat.copy()
)

# --- 2.1 Frequency & Frequency-Rank Encodings for Categorical Variables ---
new_cat_derived_cols = []
for col in cat_features:
    freq = combined_df[col].value_counts(normalize=True)
    freq_rank = freq.rank(pct=True)

    X_feat[f"{col}_freq"] = X_feat[col].map(freq).fillna(0).astype(np.float32)
    X_feat[f"{col}_freq_rank"] = (
        X_feat[col].map(freq_rank).fillna(0).astype(np.float32)
    )

    if X_test_feat is not None:
        X_test_feat[f"{col}_freq"] = (
            X_test_feat[col].map(freq).fillna(0).astype(np.float32)
        )
        X_test_feat[f"{col}_freq_rank"] = (
            X_test_feat[col].map(freq_rank).fillna(0).astype(np.float32)
        )

# --- 2.2 Hierarchical Multi-Granularity Distributional Aggregations ---
# Select primary numerical features for group aggregations
top_num_cols = (
    num_features
    if len(num_features) <= 8
    else combined_df[num_features].var().nlargest(8).index.tolist()
)

for cat_col in cat_features:
    # Group aggregations on combined reference data
    grouped = combined_df.groupby(cat_col)[top_num_cols]

    agg_medians = grouped.median()
    agg_mins = grouped.min()
    agg_maxs = grouped.max()
    agg_q25 = grouped.quantile(0.25)
    agg_q75 = grouped.quantile(0.75)
    agg_iqr = agg_q75 - agg_q25

    for num_col in top_num_cols:
        med_map = agg_medians[num_col]
        min_map = agg_mins[num_col]
        max_map = agg_maxs[num_col]
        iqr_map = agg_iqr[num_col]

        for df_target in [X_feat, X_test_feat]:
            if df_target is None:
                continue
            cat_series = df_target[cat_col]
            med_val = cat_series.map(med_map).astype(np.float32)
            min_val = cat_series.map(min_map).astype(np.float32)
            max_val = cat_series.map(max_map).astype(np.float32)
            iqr_val = cat_series.map(iqr_map).astype(np.float32)

            df_target[f"{cat_col}_{num_col}_median"] = med_val
            df_target[f"{cat_col}_{num_col}_min"] = min_val
            df_target[f"{cat_col}_{num_col}_max"] = max_val
            df_target[f"{cat_col}_{num_col}_iqr"] = iqr_val

            # Relational distance and ratio features relative to group center
            df_target[f"{cat_col}_{num_col}_diff_median"] = (
                df_target[num_col] - med_val
            ).astype(np.float32)
            df_target[f"{cat_col}_{num_col}_ratio_median"] = (
                (df_target[num_col] + 1e-5) / (med_val.abs() + 1e-5)
            ).astype(np.float32)

# --- 2.3 Quantile-Ranked Representations for Continuous Features ---
for num_col in num_features:
    combined_rank = combined_df[num_col].rank(pct=True, method="average")
    rank_mapping = combined_rank.iloc[: len(X_feat)]
    X_feat[f"{num_col}_qrank"] = rank_mapping.astype(np.float32)
    if X_test_feat is not None:
        X_test_feat[f"{num_col}_qrank"] = combined_rank.iloc[
            len(X_feat) :
        ].values.astype(np.float32)

# --- 2.4 Pairwise Numerical Interactions on Variance-Rich Continuous Features ---
top_interact_cols = (
    num_features
    if len(num_features) <= 5
    else combined_df[num_features].var().nlargest(5).index.tolist()
)

for i in range(len(top_interact_cols)):
    for j in range(i + 1, len(top_interact_cols)):
        col1 = top_interact_cols[i]
        col2 = top_interact_cols[j]

        # Differences, products, and non-linear ratios
        X_feat[f"{col1}_diff_{col2}"] = (
            X_feat[col1] - X_feat[col2]
        ).astype(np.float32)
        X_feat[f"{col1}_mult_{col2}"] = (
            X_feat[col1] * X_feat[col2]
        ).astype(np.float32)
        X_feat[f"{col1}_ratio_{col2}"] = (
            (X_feat[col1] + 1e-5) / (X_feat[col2].abs() + 1e-5)
        ).astype(np.float32)

        if X_test_feat is not None:
            X_test_feat[f"{col1}_diff_{col2}"] = (
                X_test_feat[col1] - X_test_feat[col2]
            ).astype(np.float32)
            X_test_feat[f"{col1}_mult_{col2}"] = (
                X_test_feat[col1] * X_test_feat[col2]
            ).astype(np.float32)
            X_test_feat[f"{col1}_ratio_{col2}"] = (
                (X_test_feat[col1] + 1e-5) / (X_test_feat[col2].abs() + 1e-5)
            ).astype(np.float32)

# --- 2.5 Pipeline Formatting ---
# --- For CatBoost (Native Categorical Strings) ---
X_cb = X_feat.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test_feat is not None:
    X_test_cb = X_test_feat.copy()
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# --- For XGBoost (Label Encoded and Imputed) ---
X_xgb = X_feat.copy()
if X_test_feat is not None:
    X_test_xgb = X_test_feat.copy()
else:
    X_test_xgb = None

for col in cat_features:
    le = LabelEncoder()
    if X_test_xgb is not None and col in X_test_xgb.columns:
        combined_vals = list(
            set(X_feat[col].astype(str)).union(
                set(X_test_xgb[col].astype(str))
            )
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
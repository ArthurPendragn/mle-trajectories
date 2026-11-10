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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# 1. Base Feature Identification
num_cols = [c for c in X.columns if c not in cat_features]
train_len = len(X)

# Combine datasets temporarily for consistent feature engineering
df_all = pd.concat([X, X_test], axis=0, ignore_index=True) if X_test is not None else X.copy()

engineered_features = pd.DataFrame(index=df_all.index)

# 2. Categorical Frequency / Count Encodings
for cat_col in cat_features:
    freq_map = df_all[cat_col].astype(str).value_counts(normalize=True)
    count_map = df_all[cat_col].astype(str).value_counts()
    engineered_features[f"{cat_col}_freq"] = df_all[cat_col].astype(str).map(freq_map).astype(np.float32)
    engineered_features[f"{cat_col}_count"] = df_all[cat_col].astype(str).map(count_map).astype(np.float32)

# 3. High-Frequency Categorical Interactions (Pairwise combinations)
if len(cat_features) >= 2:
    cat_pairs = list(itertools.combinations(cat_features[:6], 2))
    for col_a, col_b in cat_pairs:
        inter_series = df_all[col_a].astype(str) + "___" + df_all[col_b].astype(str)
        inter_freq = inter_series.value_counts(normalize=True)
        engineered_features[f"inter_{col_a}_{col_b}_freq"] = inter_series.map(inter_freq).astype(np.float32)

# 4. Group-Level Numerical Aggregations & Difference/Ratio Features
if len(cat_features) > 0 and len(num_cols) > 0:
    top_cats = [c for c in cat_features if df_all[c].nunique() > 1][:4]
    top_nums = num_cols[:8]

    for cat_col in top_cats:
        for num_col in top_nums:
            # Compute group aggregations on training subset to avoid distribution shifts
            train_cat_series = X[cat_col].astype(str)
            train_num_series = X[num_col]

            group_stats = pd.DataFrame({
                "cat": train_cat_series,
                "val": train_num_series
            }).groupby("cat")["val"].agg(["mean", "std"]).reset_index()

            stat_dict_mean = group_stats.set_index("cat")["mean"].to_dict()
            stat_dict_std = group_stats.set_index("cat")["std"].to_dict()

            cat_as_str = df_all[cat_col].astype(str)
            grp_mean = cat_as_str.map(stat_dict_mean).astype(np.float32)
            grp_std = cat_as_str.map(stat_dict_std).fillna(0.0).astype(np.float32)

            orig_val = df_all[num_col].astype(np.float32)
            diff_feat = orig_val - grp_mean
            ratio_feat = orig_val / (grp_mean.abs() + 1e-5)

            engineered_features[f"grp_{cat_col}_{num_col}_mean"] = grp_mean
            engineered_features[f"grp_{cat_col}_{num_col}_std"] = grp_std
            engineered_features[f"grp_{cat_col}_{num_col}_diff"] = diff_feat
            engineered_features[f"grp_{cat_col}_{num_col}_ratio"] = ratio_feat

# 5. Compact Latent Manifold Features via TruncatedSVD
svd_parts_train = []
svd_parts_test = []

if len(num_cols) > 0:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_num_clean = X[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
    X_num_scaled = scaler.fit_transform(X_num_clean)
    svd_parts_train.append(X_num_scaled)

    if X_test is not None:
        X_test_num_clean = X_test[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
        X_test_num_scaled = scaler.transform(X_test_num_clean)
        svd_parts_test.append(X_test_num_scaled)

if len(cat_features) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_str = X[cat_features].astype(str)
    X_cat_ohe = ohe.fit_transform(X_cat_str)
    svd_parts_train.append(X_cat_ohe)

    if X_test is not None:
        X_test_cat_str = X_test[cat_features].astype(str)
        X_test_cat_ohe = ohe.transform(X_test_cat_str)
        svd_parts_test.append(X_test_cat_ohe)

if len(svd_parts_train) > 0:
    Z_train = np.hstack(svd_parts_train)
    Z_test = np.hstack(svd_parts_test) if X_test is not None else None

    n_comp_svd = min(5, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp_svd, random_state=42)
    svd_train_mat = svd.fit_transform(Z_train)
    svd_test_mat = svd.transform(Z_test) if Z_test is not None else None

    for i in range(n_comp_svd):
        col_name = f"latent_svd_{i}"
        if Z_test is not None:
            combined_svd = np.concatenate([svd_train_mat[:, i], svd_test_mat[:, i]], axis=0)
        else:
            combined_svd = svd_train_mat[:, i]
        engineered_features[col_name] = combined_svd

# Split engineered features back into train and test sets
df_eng_train = engineered_features.iloc[:train_len].reset_index(drop=True)
df_eng_test = engineered_features.iloc[train_len:].reset_index(drop=True) if X_test is not None else None

# 6. For CatBoost (Native Categorical Strings + Domain & SVD Features)
X_cb = pd.concat([X.reset_index(drop=True), df_eng_train], axis=1)
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)

if X_test is not None:
    X_test_cb = pd.concat([X_test.reset_index(drop=True), df_eng_test], axis=1)
    for col in cat_features:
        X_test_cb[col] = X_test_cb[col].astype(str)
else:
    X_test_cb = None

# 7. For XGBoost (Label Encoded Categoricals + Domain & SVD Features)
X_xgb = pd.concat([X.reset_index(drop=True), df_eng_train], axis=1)
if X_test is not None:
    X_test_xgb = pd.concat([X_test.reset_index(drop=True), df_eng_test], axis=1)
else:
    X_test_xgb = None

for col in cat_features:
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
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# 1. Load Data
train_df = pd.read_csv("./input/train.csv")
y_raw = train_df["y"].values.astype(np.float64)
X_raw = train_df.drop(columns=["ID", "y"])

cat_features = [
    col for col in X_raw.columns if not pd.api.types.is_numeric_dtype(X_raw[col])
]


def prepare_datasets(use_feature_eng=True):
    # CatBoost features
    X_cb = X_raw.copy()
    for col in cat_features:
        X_cb[col] = X_cb[col].astype(str)

    # XGB/LGB/Ridge features
    X_num = X_raw.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_num[col] = le.fit_transform(X_num[col].astype(str))
    X_num = X_num.apply(pd.to_numeric, errors="coerce").fillna(0)

    if use_feature_eng:
        binary_cols = [
            col
            for col in X_num.columns
            if set(X_num[col].dropna().unique()).issubset({0, 1})
        ]
        if len(binary_cols) > 0:
            X_num["row_binary_sum"] = (X_num[binary_cols] == 1).sum(axis=1)
        X_num["row_sparsity"] = (X_num == 0).sum(axis=1) / X_num.shape[1]

        non_zero_var_cols = [c for c in X_num.columns if X_num[c].nunique() > 1]
        X_num = X_num[non_zero_var_cols].copy()

    return X_cb, X_num


def run_pipeline(
    use_target_transform=True, use_feature_eng=True, use_nnls=True
):
    X_cb, X_xgb = prepare_datasets(use_feature_eng=use_feature_eng)

    if use_target_transform:
        y_min = np.min(y_raw)
        y_shift = 0.0 if y_min > 0 else (np.abs(y_min) + 1.0)
        y_train_target = np.log(y_raw + y_shift)
        inv_transform = lambda preds: np.exp(preds) - y_shift
    else:
        y_train_target = y_raw.copy()
        inv_transform = lambda preds: preds

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cb = np.zeros(len(y_raw))
    oof_xgb = np.zeros(len(y_raw))
    oof_lgb = np.zeros(len(y_raw))
    oof_ridge = np.zeros(len(y_raw))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cb, y_train_target)):
        X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
        X_tr_xgb, X_va_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        y_tr, y_va = y_train_target[train_idx], y_train_target[val_idx]

        # 1. CatBoost
        cb_model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            subsample=0.8,
            random_seed=42 + fold,
            cat_features=cat_features,
            eval_metric="RMSE",
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), verbose=False)
        oof_cb[val_idx] = cb_model.predict(X_va_cb)

        # 2. XGBoost
        dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr)
        dval = xgb.DMatrix(X_va_xgb, label=y_va)
        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.01,
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
            num_boost_round=1800,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_xgb[val_idx] = xgb_model.predict(dval)

        # 3. LightGBM
        lgb_train = lgb.Dataset(X_tr_xgb, label=y_tr)
        lgb_val = lgb.Dataset(X_va_xgb, label=y_va, reference=lgb_train)
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.015,
            "num_leaves": 15,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
            "random_state": 42 + fold,
            "verbose": -1,
        }
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1800,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va_xgb)

        # 4. Ridge
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_xgb.fillna(0))
        X_va_scaled = scaler.transform(X_va_xgb.fillna(0))
        ridge_model = RidgeCV(alphas=np.logspace(-2, 4, 20))
        ridge_model.fit(X_tr_scaled, y_tr)
        oof_ridge[val_idx] = ridge_model.predict(X_va_scaled)

    # Invert transformation
    oof_cb_inv = inv_transform(oof_cb)
    oof_xgb_inv = inv_transform(oof_xgb)
    oof_lgb_inv = inv_transform(oof_lgb)
    oof_ridge_inv = inv_transform(oof_ridge)

    oof_matrix = np.column_stack(
        [oof_cb_inv, oof_xgb_inv, oof_lgb_inv, oof_ridge_inv]
    )

    if use_nnls:
        weights, _ = nnls(oof_matrix, y_raw)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(4) / 4
    else:
        weights = np.ones(4) / 4

    oof_ensemble = np.dot(oof_matrix, weights)
    oof_ensemble_clipped = np.clip(oof_ensemble, np.min(y_raw), np.max(y_raw))
    return r2_score(y_raw, oof_ensemble_clipped)


# Execute Ablation Experiments
print("Running Ablation Experiments...\n")

baseline_score = run_pipeline(
    use_target_transform=True, use_feature_eng=True, use_nnls=True
)
print(
    f"Baseline (Full Pipeline with Log Transform, Feature Eng, NNLS): R2 = {baseline_score:.5f}"
)

# Ablation 1: No Target Transformation (Raw Target Training)
score_no_log = run_pipeline(
    use_target_transform=False, use_feature_eng=True, use_nnls=True
)
delta_no_log = score_no_log - baseline_score
print(
    f"Ablation 1 (Without Log Target Transform): R2 = {score_no_log:.5f} (Delta R2: {delta_no_log:+.5f})"
)

# Ablation 2: No Feature Engineering / Filtering
score_no_fe = run_pipeline(
    use_target_transform=True, use_feature_eng=False, use_nnls=True
)
delta_no_fe = score_no_fe - baseline_score
print(
    f"Ablation 2 (Without Row Aggregations & Zero-Variance Filtering): R2 = {score_no_fe:.5f} (Delta R2: {delta_no_fe:+.5f})"
)

# Ablation 3: Equal Average Blending Instead of NNLS
score_no_nnls = run_pipeline(
    use_target_transform=True, use_feature_eng=True, use_nnls=False
)
delta_no_nnls = score_no_nnls - baseline_score
print(
    f"Ablation 3 (Without NNLS Optimization / Uniform Average Blend): R2 = {score_no_nnls:.5f} (Delta R2: {delta_no_nnls:+.5f})"
)

# Determine component with highest contribution
impacts = {
    "Log Target Transformation": -delta_no_log,
    "Feature Engineering (Row Stats & Variance Filtering)": -delta_no_fe,
    "NNLS Stacking Optimization": -delta_no_nnls,
}

top_component = max(impacts, key=impacts.get)
print("\n--- Ablation Study Conclusion ---")
print(
    f"The component contributing the most to overall performance is: '{top_component}' with an impact of +{impacts[top_component]:.5f} R2."
)
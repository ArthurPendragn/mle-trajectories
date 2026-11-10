import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load Training Data
train_df = pd.read_csv("./input/train.csv")
y = train_df["y"].values
X_raw = train_df.drop(columns=["ID", "y"])

cat_cols_raw = [col for col in X_raw.columns if not pd.api.types.is_numeric_dtype(X_raw[col])]

# 2. Pipeline Feature Builder function supporting modular ablation toggles
def build_features(
    df,
    target,
    enable_hygiene=True,
    enable_target_freq_enc=True,
    enable_multiview_decomp=True,
):
    X = df.copy()
    cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # --- A. Strict Feature Hygiene (Constant & Duplicate Columns Removal) ---
    if enable_hygiene:
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            cat_features = [c for c in cat_features if c not in constant_cols]

        dup_mask = X.T.duplicated(keep="first")
        duplicate_cols = X.columns[dup_mask].tolist()
        if duplicate_cols:
            X = X.drop(columns=duplicate_cols)
            cat_features = [c for c in cat_features if c not in duplicate_cols]

    num_cols = [c for c in X.columns if c not in cat_features]
    engineered_blocks = []

    # Row-wise active counts
    if len(num_cols) > 0:
        df_counts = pd.DataFrame(index=X.index)
        df_counts["active_non_zero_count"] = (X[num_cols].fillna(0) != 0).sum(axis=1)
        df_counts["active_positive_count"] = (X[num_cols].fillna(0) > 0).sum(axis=1)
        engineered_blocks.append(df_counts)

    # --- B. Frequency & Out-Of-Fold Regularized Target Encodings ---
    if enable_target_freq_enc and len(cat_features) > 0:
        df_enc = pd.DataFrame(index=X.index)
        global_mean = float(np.mean(target))
        smooth_weight = 10.0
        kf_te = KFold(n_splits=5, shuffle=True, random_state=42)

        for col in cat_features:
            # Frequency encoding
            freq = X[col].astype(str).value_counts(normalize=True)
            df_enc[f"{col}_freq"] = X[col].astype(str).map(freq).fillna(0)

            # Regularized OOF Target encoding
            col_str = X[col].astype(str)
            oof_te = np.zeros(len(X))
            for tr_idx, val_idx in kf_te.split(X, target):
                X_tr_col, y_tr = col_str.iloc[tr_idx], target[tr_idx]
                stats = pd.DataFrame({"cat": X_tr_col, "target": y_tr}).groupby("cat")["target"].agg(["count", "mean"])
                smoothed_map = (stats["count"] * stats["mean"] + smooth_weight * global_mean) / (stats["count"] + smooth_weight)
                oof_te[val_idx] = col_str.iloc[val_idx].map(smoothed_map).fillna(global_mean)
            df_enc[f"{col}_oof_te"] = oof_te

        engineered_blocks.append(df_enc)

    # --- C. Multi-View Linear Decomposition Suite (PCA, TruncatedSVD, FastICA, FactorAnalysis) ---
    if enable_multiview_decomp:
        decomp_parts = []
        if len(num_cols) > 0:
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X[num_cols].fillna(0))
            decomp_parts.append(X_num_scaled)

        if len(cat_features) > 0:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_ohe = ohe.fit_transform(X[cat_features].astype(str))
            decomp_parts.append(X_cat_ohe)

        if len(decomp_parts) > 0:
            Z = np.hstack(decomp_parts)
            n_comp = min(5, max(2, Z.shape[1] - 1)) if Z.shape[1] > 1 else 1
            df_decomp = {}

            # PCA
            pca = PCA(n_components=n_comp, random_state=42)
            pca_feats = pca.fit_transform(Z)
            for i in range(n_comp):
                df_decomp[f"latent_pca_{i}"] = pca_feats[:, i]

            # TruncatedSVD
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_feats = svd.fit_transform(Z)
            for i in range(n_comp):
                df_decomp[f"latent_svd_{i}"] = svd_feats[:, i]

            # FastICA
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=300, tol=1e-3)
            ica_feats = ica.fit_transform(Z)
            for i in range(n_comp):
                df_decomp[f"latent_ica_{i}"] = ica_feats[:, i]

            # Factor Analysis
            fa = FactorAnalysis(n_components=n_comp, random_state=42, max_iter=200)
            fa_feats = fa.fit_transform(Z)
            for i in range(n_comp):
                df_decomp[f"latent_fa_{i}"] = fa_feats[:, i]

            engineered_blocks.append(pd.DataFrame(df_decomp, index=X.index))

    # Combine full feature representations
    if engineered_blocks:
        dense_feats = pd.concat(engineered_blocks, axis=1)
        X_all = pd.concat([X, dense_feats], axis=1)
    else:
        X_all = X.copy()

    # Create model-ready matrices
    # CatBoost matrix:
    X_cb = X_all.copy()
    for col in cat_features:
        X_cb[col] = X_cb[col].astype(str)

    # XGBoost matrix (label-encoded):
    X_xgb = X_all.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))
    X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X_cb, X_xgb, cat_features


# 3. Model Training & Evaluation Routine
def evaluate_pipeline(X_cb, X_xgb, cat_features, y_target):
    X_tr_cb, X_val_cb, y_train, y_val = train_test_split(
        X_cb, y_target, test_size=0.2, random_state=42
    )
    X_tr_xgb, X_val_xgb, _, _ = train_test_split(
        X_xgb, y_target, test_size=0.2, random_state=42
    )

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42,
        cat_features=cat_features,
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_tr_cb, y_train, eval_set=(X_val_cb, y_val), verbose=False)
    preds_cb = cb_model.predict(X_val_cb)

    # XGBoost
    dtrain = xgb.DMatrix(X_tr_xgb, label=y_train)
    dval = xgb.DMatrix(X_val_xgb, label=y_val)
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.015,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "alpha": 2.0,
        "lambda": 5.0,
        "random_state": 42,
        "tree_method": "hist",
    }
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1200,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    preds_xgb = xgb_model.predict(dval)

    # Ensemble prediction
    ensemble_preds = 0.5 * preds_cb + 0.5 * preds_xgb
    return r2_score(y_val, ensemble_preds)


# 4. Run Ablation Study
print("--- Starting Ablation Study ---")

# Baseline: Full Pipeline
X_cb_full, X_xgb_full, cats_full = build_features(
    X_raw, y, enable_hygiene=True, enable_target_freq_enc=True, enable_multiview_decomp=True
)
score_baseline = evaluate_pipeline(X_cb_full, X_xgb_full, cats_full, y)
print(f"Full Baseline (All Components Enabled) R2: {score_baseline:.5f}")

# Ablation 1: Disable Feature Hygiene (keep duplicate and zero-variance columns)
X_cb_ab1, X_xgb_ab1, cats_ab1 = build_features(
    X_raw, y, enable_hygiene=False, enable_target_freq_enc=True, enable_multiview_decomp=True
)
score_ab1 = evaluate_pipeline(X_cb_ab1, X_xgb_ab1, cats_ab1, y)
delta_ab1 = score_ab1 - score_baseline
print(f"Ablation 1 (Without Feature Hygiene) R2: {score_ab1:.5f} (Delta: {delta_ab1:+.5f})")

# Ablation 2: Disable Frequency & OOF Target Encodings
X_cb_ab2, X_xgb_ab2, cats_ab2 = build_features(
    X_raw, y, enable_hygiene=True, enable_target_freq_enc=False, enable_multiview_decomp=True
)
score_ab2 = evaluate_pipeline(X_cb_ab2, X_xgb_ab2, cats_ab2, y)
delta_ab2 = score_ab2 - score_baseline
print(f"Ablation 2 (Without Freq & OOF Target Encodings) R2: {score_ab2:.5f} (Delta: {delta_ab2:+.5f})")

# Ablation 3: Disable Multi-View Linear Decompositions (PCA, ICA, FA, SVD)
X_cb_ab3, X_xgb_ab3, cats_ab3 = build_features(
    X_raw, y, enable_hygiene=True, enable_target_freq_enc=True, enable_multiview_decomp=False
)
score_ab3 = evaluate_pipeline(X_cb_ab3, X_xgb_ab3, cats_ab3, y)
delta_ab3 = score_ab3 - score_baseline
print(f"Ablation 3 (Without Multi-View Decompositions) R2: {score_ab3:.5f} (Delta: {delta_ab3:+.5f})")

# 5. Determine and Print Most Influential Component
ablation_impacts = {
    "Strict Feature Hygiene (Zero-Variance & Duplicate Pruning)": score_baseline - score_ab1,
    "Frequency & OOF Regularized Target Encodings": score_baseline - score_ab2,
    "Multi-View Linear Decomposition Suite (PCA, SVD, ICA, FA)": score_baseline - score_ab3,
}

top_component = max(ablation_impacts, key=ablation_impacts.get)
print("\n--- Ablation Study Summary ---")
for component, impact in ablation_impacts.items():
    print(f"Component: {component} | Contribution Impact (+R2): {impact:+.5f}")

print(f"\nThe component contributing the most to overall performance is: '{top_component}' with an impact of {ablation_impacts[top_component]:+.5f} R2.")
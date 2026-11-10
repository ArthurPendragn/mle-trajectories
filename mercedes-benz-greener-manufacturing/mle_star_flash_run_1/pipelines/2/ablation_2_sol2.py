import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import xgboost as xgb

# 1. Load Training Data
train_df = pd.read_csv("./input/train.csv")
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]
num_cols = [c for c in X.columns if c not in cat_features]

# 2. Geometric Feature Extraction
ohe_train_parts = []
if len(num_cols) > 0:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_num_clean = X[num_cols].fillna(X[num_cols].median(numeric_only=True)).fillna(0)
    X_num_scaled = scaler.fit_transform(X_num_clean)
    ohe_train_parts.append(X_num_scaled)

if len(cat_features) > 0:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_str = X[cat_features].astype(str)
    X_cat_ohe = ohe.fit_transform(X_cat_str)
    ohe_train_parts.append(X_cat_ohe)

if len(ohe_train_parts) > 0:
    Z_train = np.hstack(ohe_train_parts)
else:
    Z_train = np.zeros((len(X), 1))

# TruncatedSVD Latent Features
svd_features = {}
n_comp_svd = min(6, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_comp_svd, random_state=42)
svd_train = svd.fit_transform(Z_train)
for i in range(n_comp_svd):
    svd_features[f"latent_svd_{i}"] = svd_train[:, i]
df_svd = pd.DataFrame(svd_features, index=X.index)

# NMF Latent Features
nmf_features = {}
n_comp_nmf = min(4, max(2, Z_train.shape[1] - 1)) if Z_train.shape[1] > 1 else 1
nmf = NMF(n_components=n_comp_nmf, init="nndsvda", max_iter=400, random_state=42)
nmf_train = nmf.fit_transform(np.clip(Z_train, 0, None))
for i in range(n_comp_nmf):
    nmf_features[f"latent_nmf_{i}"] = nmf_train[:, i]
df_nmf = pd.DataFrame(nmf_features, index=X.index)

# MiniBatchKMeans Cluster Distance Features
kmeans_features = {}
k_granularities = [4, 8, 16]
for k in k_granularities:
    actual_k = min(k, len(Z_train))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_k,
        batch_size=1024,
        max_iter=100,
        random_state=42 + actual_k,
        n_init="auto",
    )
    dist_train = kmeans.fit_transform(Z_train)
    for c_idx in range(actual_k):
        kmeans_features[f"dist_kmeans_k{actual_k}_c{c_idx}"] = dist_train[:, c_idx]
    kmeans_features[f"dist_kmeans_k{actual_k}_min"] = dist_train.min(axis=1)
df_kmeans = pd.DataFrame(kmeans_features, index=X.index)


# Helper function to evaluate feature sets
def evaluate_feature_set(extra_features_list):
    if len(extra_features_list) > 0:
        X_augmented = pd.concat([X.copy()] + extra_features_list, axis=1)
    else:
        X_augmented = X.copy()

    # Prep CatBoost Data
    X_cb = X_augmented.copy()
    for col in cat_features:
        X_cb[col] = X_cb[col].astype(str)

    # Prep XGBoost Data
    X_xgb = X_augmented.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))
    X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Split
    X_train_cb, X_val_cb, y_train, y_val = train_test_split(
        X_cb, y, test_size=0.2, random_state=42
    )
    X_train_xgb, X_val_xgb, _, _ = train_test_split(
        X_xgb, y, test_size=0.2, random_state=42
    )

    # Fit CatBoost
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

    # Fit XGBoost
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

    # Blend Predictions
    val_preds_cb = cb_model.predict(X_val_cb)
    val_preds_xgb = xgb_model.predict(dval)
    val_preds = 0.5 * val_preds_cb + 0.5 * val_preds_xgb

    return r2_score(y_val, val_preds)


# 3. Running Ablation Experiments
print("=== Running Ablation Studies on Geometric & Latent Manifold Features ===")

# Baseline: All features (SVD + NMF + KMeans)
baseline_score = evaluate_feature_set([df_svd, df_nmf, df_kmeans])
print(f"Baseline (Full Manifold: SVD + NMF + KMeans) R2 Score: {baseline_score:.5f}")

# Ablation 1: Remove Latent Decomposition (No SVD, No NMF; keep KMeans)
ablation_1_score = evaluate_feature_set([df_kmeans])
delta_1 = baseline_score - ablation_1_score
print(
    f"Ablation 1 (Without SVD & NMF Features) R2 Score: {ablation_1_score:.5f} (Delta vs Baseline: {-delta_1:+.5f})"
)

# Ablation 2: Remove Multi-Granularity KMeans Cluster Distances (keep SVD & NMF)
ablation_2_score = evaluate_feature_set([df_svd, df_nmf])
delta_2 = baseline_score - ablation_2_score
print(
    f"Ablation 2 (Without KMeans Cluster Distances) R2 Score: {ablation_2_score:.5f} (Delta vs Baseline: {-delta_2:+.5f})"
)

# Ablation 3: Remove All Geometric Features (Raw Baseline Features Only)
ablation_3_score = evaluate_feature_set([])
delta_3 = baseline_score - ablation_3_score
print(
    f"Ablation 3 (Without Any Geometric/Latent Features) R2 Score: {ablation_3_score:.5f} (Delta vs Baseline: {-delta_3:+.5f})"
)

# 4. Summary and Contribution Analysis
print("\n=== Ablation Summary & Impact Analysis ===")
feature_impacts = {
    "Latent Decompositions (SVD & NMF)": delta_1,
    "KMeans Cluster Distance Features": delta_2,
    "Combined Geometric Feature Suite": delta_3,
}

for component, impact in feature_impacts.items():
    print(f"Contribution of {component}: {impact:+.5f} R2")

top_contributor = max(
    [
        ("Latent Decompositions (SVD & NMF)", delta_1),
        ("KMeans Cluster Distance Features", delta_2),
    ],
    key=lambda x: x[1],
)

print(
    f"\nConclusion: Among the evaluated sub-components, '{top_contributor[0]}' contributes the most to the overall performance with a contribution of {top_contributor[1]:+.5f} R2."
)
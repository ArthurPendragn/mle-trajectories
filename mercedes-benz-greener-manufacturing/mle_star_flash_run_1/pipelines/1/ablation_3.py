from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import xgboost as xgb

# 1. Load Data
train_df = pd.read_csv('./input/train.csv')
y = train_df['y'].values
X = train_df.drop(columns=['ID', 'y'])

# Identify categorical columns and apply Ordinal Encoding
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

X = X.astype(np.float32)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def get_decomp_union(n_components=10, include_random_projections=True):
    transformers = [
        ("identity", Pipeline([("passthrough", "passthrough")])),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("tsvd", TruncatedSVD(n_components=n_components, random_state=42)),
        (
            "ica",
            FastICA(
                n_components=n_components,
                random_state=42,
                max_iter=1000,
                tol=1e-3,
            ),
        ),
    ]
    if include_random_projections:
        transformers.extend(
            [
                (
                    "grp",
                    GaussianRandomProjection(
                        n_components=n_components, random_state=42
                    ),
                ),
                (
                    "srp",
                    SparseRandomProjection(
                        n_components=n_components, random_state=42
                    ),
                ),
            ]
        )
    return FeatureUnion(transformers)


def build_stacking_ensemble(
    n_decomp_components=10,
    include_random_projections=True,
    model_family="hybrid",  # "hybrid" or "tree_only"
):
    decomp_union = get_decomp_union(
        n_components=n_decomp_components,
        include_random_projections=include_random_projections,
    )

    linear_preprocessor = Pipeline(
        [
            ("decomp", decomp_union),
            ("scale", RobustScaler()),
        ]
    )

    tree_preprocessor = Pipeline([("decomp", decomp_union)])

    tree_estimators = [
        (
            "lgb",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        lgb.LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.02,
                            max_depth=3,
                            num_leaves=7,
                            colsample_bytree=0.6,
                            subsample=0.8,
                            reg_alpha=1.0,
                            reg_lambda=1.0,
                            random_state=42,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "xgb",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        xgb.XGBRegressor(
                            n_estimators=300,
                            learning_rate=0.02,
                            max_depth=3,
                            colsample_bytree=0.6,
                            subsample=0.8,
                            reg_alpha=1.0,
                            reg_lambda=1.0,
                            random_state=42,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        ),
        (
            "cb",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        CatBoostRegressor(
                            iterations=500,
                            learning_rate=0.02,
                            depth=3,
                            l2_leaf_reg=5.0,
                            random_seed=42,
                            verbose=False,
                        ),
                    ),
                ]
            ),
        ),
    ]

    linear_estimators = [
        (
            "krr",
            Pipeline(
                [
                    ("prep", linear_preprocessor),
                    (
                        "model",
                        KernelRidge(
                            alpha=0.6, kernel="polynomial", degree=2, coef0=2.5
                        ),
                    ),
                ]
            ),
        ),
        (
            "huber",
            Pipeline(
                [
                    ("prep", linear_preprocessor),
                    (
                        "model",
                        HuberRegressor(
                            epsilon=1.35, alpha=10.0, max_iter=1000
                        ),
                    ),
                ]
            ),
        ),
        (
            "bayesian_ridge",
            Pipeline(
                [
                    ("prep", linear_preprocessor),
                    (
                        "model",
                        BayesianRidge(
                            max_iter=1000,
                            alpha_1=1e-6,
                            alpha_2=1e-6,
                            lambda_1=1e-6,
                            lambda_2=1e-6,
                        ),
                    ),
                ]
            ),
        ),
    ]

    if model_family == "tree_only":
        estimators = tree_estimators
    else:
        estimators = tree_estimators + linear_estimators

    meta_learner = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=5)

    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking_reg


# ==========================================
# Run Baseline and Ablation Experiments
# ==========================================
experiments = {}

# 1. Baseline Full Stacking Model
print("Running Baseline Model...")
baseline_model = build_stacking_ensemble(
    n_decomp_components=10,
    include_random_projections=True,
    model_family="hybrid",
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_val)
baseline_score = r2_score(y_val, baseline_pred)
experiments["Baseline (Full Ensemble: Hybrid + All Decomps @ 10)"] = (
    baseline_score
)

# 2. Ablation 1: Removing Random Projections (Deterministic Decomps only: PCA, TSVD, ICA)
print("Running Ablation 1: Without Random Projections (GRP/SRP)...")
ablation_1_model = build_stacking_ensemble(
    n_decomp_components=10,
    include_random_projections=False,
    model_family="hybrid",
)
ablation_1_model.fit(X_train, y_train)
ablation_1_pred = ablation_1_model.predict(X_val)
ablation_1_score = r2_score(y_val, ablation_1_pred)
experiments["Ablation 1 (Without Random Projections: PCA/TSVD/ICA only)"] = (
    ablation_1_score
)

# 3. Ablation 2: Model Diversity Ablation (Tree Models Only)
print("Running Ablation 2: Tree Models Only (Removing Linear/Kernel Models)...")
ablation_2_model = build_stacking_ensemble(
    n_decomp_components=10,
    include_random_projections=True,
    model_family="tree_only",
)
ablation_2_model.fit(X_train, y_train)
ablation_2_pred = ablation_2_model.predict(X_val)
ablation_2_score = r2_score(y_val, ablation_2_pred)
experiments["Ablation 2 (Tree Models Only: LGB + XGB + CB)"] = ablation_2_score

# 4. Ablation 3: Low-Capacity Latent Dimension (n_components=4 instead of 10)
print("Running Ablation 3: Low Decomposition Capacity (n_components=4)...")
ablation_3_model = build_stacking_ensemble(
    n_decomp_components=4,
    include_random_projections=True,
    model_family="hybrid",
)
ablation_3_model.fit(X_train, y_train)
ablation_3_pred = ablation_3_model.predict(X_val)
ablation_3_score = r2_score(y_val, ablation_3_pred)
experiments["Ablation 3 (Low Decomposition Capacity: n_components=4)"] = (
    ablation_3_score
)

# ==========================================
# Report Results
# ==========================================
print("\n" + "=" * 80)
print("ABLATION STUDY RESULTS SUMMARY")
print("=" * 80)
print(f"{'Configuration':<60} | {'Validation R2':<15} | {'Delta R2'}")
print("-" * 80)

deltas = {}
for name, score in experiments.items():
    delta = score - baseline_score
    deltas[name] = abs(delta)
    if name.startswith("Baseline"):
        print(f"{name:<60} | {score:<15.5f} | Baseline")
    else:
        print(f"{name:<60} | {score:<15.5f} | {delta:+.5f}")

print("=" * 80)

# Identify most influential component among ablations
ablation_deltas = {k: v for k, v in deltas.items() if not k.startswith("Baseline")}
most_impactful_component = max(ablation_deltas, key=ablation_deltas.get)
print(f"\nMost influential component studied: '{most_impactful_component}'")
print(
    f"Performance impact: Delta R2 = {experiments[most_impactful_component] - baseline_score:+.5f}"
)
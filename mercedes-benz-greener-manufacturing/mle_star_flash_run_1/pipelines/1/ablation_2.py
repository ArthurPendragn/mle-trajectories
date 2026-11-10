import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)

# Load data
train_df = pd.read_csv("./input/train.csv")

# Separate target and features
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

# Categorical encoding upstream
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

X = X.astype(np.float32)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def get_decomp_union():
    return FeatureUnion(
        [
            ("identity", FunctionTransformer()),
            ("pca", PCA(n_components=10, random_state=42)),
            ("tsvd", TruncatedSVD(n_components=10, random_state=42)),
            ("ica", FastICA(n_components=10, random_state=42, max_iter=1000)),
            ("grp", GaussianRandomProjection(n_components=10, random_state=42)),
            ("srp", SparseRandomProjection(n_components=10, random_state=42)),
        ]
    )


def build_stacking_ensemble(
    tree_use_decomp=True,
    scaler_type="robust",
    passthrough_meta=False,
):
    categorical_selector = make_column_selector(
        dtype_include=["object", "category"]
    )
    numerical_selector = make_column_selector(dtype_include=["number"])

    encoder_ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_selector,
            ),
            ("num", "passthrough", numerical_selector),
        ],
        remainder="passthrough",
    )

    decomp_union = get_decomp_union()

    # Scaler selection for linear pipeline
    if scaler_type == "robust":
        scaler_step = ("scale", RobustScaler())
    elif scaler_type == "standard":
        scaler_step = ("scale", StandardScaler())
    else:
        scaler_step = ("scale", FunctionTransformer())

    linear_preprocessor = Pipeline(
        [
            ("encode", encoder_ct),
            ("decomp", decomp_union),
            scaler_step,
        ]
    )

    if tree_use_decomp:
        tree_preprocessor = Pipeline([("decomp", decomp_union)])
    else:
        tree_preprocessor = Pipeline([("identity", FunctionTransformer())])

    estimators = [
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

    meta_learner = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=5)

    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=passthrough_meta,
    )
    return stacking_reg


def evaluate_config(name, **kwargs):
    model = build_stacking_ensemble(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)
    return score


# 1. Baseline Full Model
print("Running Baseline and Ablation Experiments...")
baseline_score = evaluate_config("Baseline")
print(f"Baseline Validation R2 Score: {baseline_score:.5f}")

# 2. Ablation 1: Raw features for tree models (no decomposition features in tree preprocessors)
score_ablation_trees_no_decomp = evaluate_config(
    "Tree Models without Decomposition",
    tree_use_decomp=False,
    scaler_type="robust",
    passthrough_meta=False,
)
delta_trees_no_decomp = score_ablation_trees_no_decomp - baseline_score
print(
    f"Ablation 1 (Tree Models on Raw Features Only): R2 = {score_ablation_trees_no_decomp:.5f} (Delta: {delta_trees_no_decomp:+.5f})"
)

# 3. Ablation 2: Meta-Learner Passthrough (passing original features directly to RidgeCV meta-learner)
score_ablation_passthrough = evaluate_config(
    "Meta-Learner Passthrough True",
    tree_use_decomp=True,
    scaler_type="robust",
    passthrough_meta=True,
)
delta_passthrough = score_ablation_passthrough - baseline_score
print(
    f"Ablation 2 (Stacking Meta Passthrough = True): R2 = {score_ablation_passthrough:.5f} (Delta: {delta_passthrough:+.5f})"
)

# 4. Ablation 3: Linear Pipeline Scaling (StandardScaler vs RobustScaler)
score_ablation_standard_scaler = evaluate_config(
    "Linear Preprocessor StandardScaler",
    tree_use_decomp=True,
    scaler_type="standard",
    passthrough_meta=False,
)
delta_scaler = score_ablation_standard_scaler - baseline_score
print(
    f"Ablation 3 (StandardScaler instead of RobustScaler in Linear Prep): R2 = {score_ablation_standard_scaler:.5f} (Delta: {delta_scaler:+.5f})"
)

# Determine the most influential modification
ablation_impacts = {
    "Tree Preprocessor Decomposition Features": abs(delta_trees_no_decomp),
    "Stacking Meta Passthrough Setting": abs(delta_passthrough),
    "Linear Pipeline RobustScaler vs StandardScaler": abs(delta_scaler),
}

most_influential_part = max(ablation_impacts, key=ablation_impacts.get)
print("\n--- Ablation Study Summary ---")
print(
    f"The component contributing the most to overall performance variance is: '{most_influential_part}' with an absolute score shift of {ablation_impacts[most_influential_part]:.5f}."
)
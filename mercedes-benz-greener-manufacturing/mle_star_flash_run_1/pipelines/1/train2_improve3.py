from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    RidgeCV,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import xgboost as xgb

# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y'].values
X = train_df.drop(columns=['ID', 'y'])

# Train/validation split prior to any feature engineering to strictly avoid data leakage
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.copy()
X_val = X_val.copy()

# Identify categorical and binary/numeric columns
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
binary_cols = [c for c in X_train.columns if c not in cat_cols]

# Compute row-wise binary summary statistics before column pruning
if binary_cols:
    for df in (X_train, X_val):
        df['bin_row_sum'] = df[binary_cols].sum(axis=1)
        df['bin_row_mean'] = df[binary_cols].mean(axis=1)
        df['bin_row_std'] = df[binary_cols].std(axis=1).fillna(0)
        df['bin_active_count'] = (df[binary_cols] > 0).sum(axis=1)

# Prune zero-variance and duplicate/collinear binary columns fitted strictly on train
if binary_cols:
    # 1. Zero-variance pruning
    train_var = X_train[binary_cols].var()
    non_zero_var_cols = train_var[train_var > 0].index.tolist()
    
    # 2. Redundant collinear / duplicate column pruning
    unique_bin_df = X_train[non_zero_var_cols].T.drop_duplicates().T
    kept_bin_cols = unique_bin_df.columns.tolist()
    
    # Drop pruned columns from both sets
    cols_to_drop = list(set(binary_cols) - set(kept_bin_cols))
    X_train.drop(columns=cols_to_drop, inplace=True)
    X_val.drop(columns=cols_to_drop, inplace=True)

# Process categorical features: Ordinal Encoding and Frequency Encoding
if cat_cols:
    # Ordinal Encoding
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
    X_val[cat_cols] = encoder.transform(X_val[cat_cols].astype(str))

    # Frequency Encoding fitted on train partition
    for col in cat_cols:
        freq_map = X_train[col].value_counts(normalize=True).to_dict()
        X_train[f'{col}_freq'] = X_train[col].map(freq_map).fillna(0).astype(np.float32)
        X_val[f'{col}_freq'] = X_val[col].map(freq_map).fillna(0).astype(np.float32)

# Convert all features to float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)


def build_stacking_ensemble():
    categorical_selector = make_column_selector(dtype_include=["object", "category"])
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

    decomp_union = FeatureUnion(
        [
            ("identity", Pipeline([("passthrough", "passthrough")])),
            (
                "pca",
                PCA(n_components=10, random_state=42),
            ),
            (
                "tsvd",
                TruncatedSVD(n_components=10, random_state=42),
            ),
            (
                "ica",
                FastICA(n_components=10, random_state=42, max_iter=1000),
            ),
            (
                "grp",
                GaussianRandomProjection(n_components=10, random_state=42),
            ),
            (
                "srp",
                SparseRandomProjection(n_components=10, random_state=42),
            ),
        ]
    )

    linear_preprocessor = Pipeline(
        [
            ("encode", encoder_ct),
            ("decomp", decomp_union),
            ("scale", RobustScaler()),
        ]
    )

    tree_preprocessor = Pipeline(
        [
            ("decomp", decomp_union),
        ]
    )

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

    meta_learner = RidgeCV(
        alphas=np.logspace(-1, 4, 30),
        cv=5,
    )

    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking_reg


# Instantiate and fit model
model = build_stacking_ensemble()
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
final_validation_score = r2_score(y_val, y_val_pred)

print(f'Final Validation Performance: {final_validation_score}')
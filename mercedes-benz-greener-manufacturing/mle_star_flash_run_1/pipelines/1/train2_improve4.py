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

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Separate target and initial features
y = train_df['y'].values
drop_cols = [col for col in ['ID', 'y'] if col in train_df.columns]
X = train_df.drop(columns=drop_cols)

# Train/validation split strictly prior to feature engineering to prevent leakage
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.copy().reset_index(drop=True)
X_val = X_val.copy().reset_index(drop=True)

# Identify base categorical and binary/numerical columns
base_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
bin_cols = [col for col in X_train.columns if col not in base_cat_cols]

# Create interaction pair features between dominant categorical columns
interaction_cols = []
if len(base_cat_cols) > 1:
    dominant_cat_cols = base_cat_cols[:min(6, len(base_cat_cols))]
    for i in range(len(dominant_cat_cols)):
        for j in range(i + 1, len(dominant_cat_cols)):
            col_a, col_b = dominant_cat_cols[i], dominant_cat_cols[j]
            pair_name = f"{col_a}_{col_b}"
            X_train[pair_name] = (
                X_train[col_a].astype(str) + "_" + X_train[col_b].astype(str)
            )
            X_val[pair_name] = (
                X_val[col_a].astype(str) + "_" + X_val[col_b].astype(str)
            )
            interaction_cols.append(pair_name)

all_cat_cols = base_cat_cols + interaction_cols

# Regularized Target Encoding with Out-Of-Fold computation and Empirical Bayes smoothing
if all_cat_cols:
    smoothing_weight = 10.0
    global_mean = np.mean(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in all_cat_cols:
        te_col_name = f"{col}_te"
        X_train[te_col_name] = np.nan
        X_val[te_col_name] = np.nan

        # Out-of-fold target encoding for training split
        for train_idx, val_idx in kf.split(X_train):
            fold_train_x, fold_train_y = (
                X_train.iloc[train_idx][col],
                y_train[train_idx],
            )
            stats = pd.DataFrame({'cat': fold_train_x, 'target': fold_train_y}).groupby('cat')['target'].agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + smoothing_weight * global_mean) / (stats['count'] + smoothing_weight)
            X_train.iloc[val_idx, X_train.columns.get_loc(te_col_name)] = (
                X_train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
            )

        # Full train target encoding mapping for validation split
        full_stats = pd.DataFrame({'cat': X_train[col], 'target': y_train}).groupby('cat')['target'].agg(['count', 'mean'])
        full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing_weight * global_mean) / (full_stats['count'] + smoothing_weight)
        X_val[te_col_name] = X_val[col].map(full_smoothed).fillna(global_mean).values

    # Apply Ordinal Encoding to categorical columns
    ord_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X_train[all_cat_cols] = ord_encoder.fit_transform(
        X_train[all_cat_cols].astype(str)
    )
    X_val[all_cat_cols] = ord_encoder.transform(X_val[all_cat_cols].astype(str))

# Row-level binary sparsity aggregates
if bin_cols:
    X_train_bin = X_train[bin_cols].astype(np.float32)
    X_val_bin = X_val[bin_cols].astype(np.float32)

    X_train['bin_sum'] = X_train_bin.sum(axis=1)
    X_train['bin_mean'] = X_train_bin.mean(axis=1)
    X_train['bin_std'] = X_train_bin.std(axis=1)

    X_val['bin_sum'] = X_val_bin.sum(axis=1)
    X_val['bin_mean'] = X_val_bin.mean(axis=1)
    X_val['bin_std'] = X_val_bin.std(axis=1)

    # Latent continuous representations via linear decomposition techniques
    n_comp = min(10, max(2, len(bin_cols) // 10))

    pca = PCA(n_components=n_comp, random_state=42)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    ica = FastICA(
        n_components=n_comp,
        random_state=42,
        max_iter=1000,
        tol=1e-3,
    )

    pca_train = pca.fit_transform(X_train_bin)
    pca_val = pca.transform(X_val_bin)

    svd_train = svd.fit_transform(X_train_bin)
    svd_val = svd.transform(X_val_bin)

    ica_train = ica.fit_transform(X_train_bin)
    ica_val = ica.transform(X_val_bin)

    for i in range(n_comp):
        X_train[f'pca_{i}'] = pca_train[:, i]
        X_val[f'pca_{i}'] = pca_val[:, i]
        X_train[f'svd_{i}'] = svd_train[:, i]
        X_val[f'svd_{i}'] = svd_val[:, i]
        X_train[f'ica_{i}'] = ica_train[:, i]
        X_val[f'ica_{i}'] = ica_val[:, i]

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
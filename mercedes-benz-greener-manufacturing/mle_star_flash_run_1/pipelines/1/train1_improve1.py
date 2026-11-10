from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
import xgboost as xgb

# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y'].values
X = train_df.drop(columns=['ID', 'y'])

# Identify categorical columns and apply Ordinal Encoding
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

# Convert all features to float32
X = X.astype(np.float32)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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
                PCA(n_components=12, random_state=42),
            ),
            (
                "tsvd",
                TruncatedSVD(n_components=12, random_state=42),
            ),
            (
                "ica",
                FastICA(n_components=12, random_state=42, max_iter=1000),
            ),
            (
                "grp",
                GaussianRandomProjection(n_components=12, random_state=42),
            ),
        ]
    )

    linear_preprocessor = Pipeline(
        [
            ("encode", encoder_ct),
            ("variance_thresh", VarianceThreshold()),
            ("decomp", decomp_union),
            ("scale", RobustScaler()),
        ]
    )

    tree_preprocessor = Pipeline(
        [
            ("variance_thresh", VarianceThreshold()),
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
            "et",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=200,
                            max_depth=6,
                            min_samples_split=4,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
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
            "enet",
            Pipeline(
                [
                    ("prep", linear_preprocessor),
                    (
                        "model",
                        ElasticNetCV(
                            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                            cv=5,
                            random_state=42,
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
        ),
    ]

    meta_learner = RidgeCV(
        alphas=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
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
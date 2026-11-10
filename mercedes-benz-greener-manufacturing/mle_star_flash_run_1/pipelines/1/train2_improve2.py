from itertools import combinations
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
    HuberRegressor,
    RidgeCV,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import xgboost as xgb

# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y'].values
X = train_df.drop(columns=['ID', 'y'], errors='ignore')

# Train/validation split first to ensure strict data isolation
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_raw = X_train_raw.copy().reset_index(drop=True)
X_val_raw = X_val_raw.copy().reset_index(drop=True)

# Identify categorical and binary/numerical columns
cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
bin_cols = [col for col in X_train_raw.columns if col not in cat_cols]

# 1. Pre-filter zero-variance and duplicate (perfectly collinear) binary columns on train
bin_train = X_train_raw[bin_cols].astype(np.float32)
bin_val = X_val_raw[bin_cols].astype(np.float32)

# Filter zero variance
non_zero_var_mask = bin_train.std(axis=0) > 0
valid_bin_cols = bin_train.columns[non_zero_var_mask].tolist()
bin_train = bin_train[valid_bin_cols]
bin_val = bin_val[valid_bin_cols]

# Filter duplicate/collinear columns
unique_bin_df = bin_train.T.drop_duplicates().T
pruned_bin_cols = unique_bin_df.columns.tolist()
bin_train = bin_train[pruned_bin_cols]
bin_val = bin_val[pruned_bin_cols]

# 2. Row-level statistical summaries strictly on the pruned binary matrix
bin_train_stats = pd.DataFrame({
    'bin_row_sum': bin_train.sum(axis=1),
    'bin_row_mean': bin_train.mean(axis=1),
    'bin_row_std': bin_train.std(axis=1)
})
bin_val_stats = pd.DataFrame({
    'bin_row_sum': bin_val.sum(axis=1),
    'bin_row_mean': bin_val.mean(axis=1),
    'bin_row_std': bin_val.std(axis=1)
})

# 3. Dimensionality reduction (PCA and TruncatedSVD) on pruned binary features
n_comp = min(5, bin_train.shape[1])
pca = PCA(n_components=n_comp, random_state=42)
svd = TruncatedSVD(n_components=n_comp, random_state=42)

pca_train = pd.DataFrame(
    pca.fit_transform(bin_train),
    columns=[f'pca_{i}' for i in range(n_comp)]
)
pca_val = pd.DataFrame(
    pca.transform(bin_val),
    columns=[f'pca_{i}' for i in range(n_comp)]
)

svd_train = pd.DataFrame(
    svd.fit_transform(bin_train),
    columns=[f'svd_{i}' for i in range(n_comp)]
)
svd_val = pd.DataFrame(
    svd.transform(bin_val),
    columns=[f'svd_{i}' for i in range(n_comp)]
)

# 4. Categorical interactions, Ordinal Encoding, and K-Fold Smoothed OOF Target Encoding
X_train_cat = pd.DataFrame(index=X_train_raw.index)
X_val_cat = pd.DataFrame(index=X_val_raw.index)

if cat_cols:
    # Build pairwise categorical interactions
    all_cat_features = list(cat_cols)
    for c1, c2 in combinations(cat_cols, 2):
        pair_name = f"{c1}_{c2}"
        X_train_raw[pair_name] = X_train_raw[c1].astype(str) + "_" + X_train_raw[c2].astype(str)
        X_val_raw[pair_name] = X_val_raw[c1].astype(str) + "_" + X_val_raw[c2].astype(str)
        all_cat_features.append(pair_name)

    # Ordinal encoding for original categorical columns
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat[cat_cols] = ord_enc.fit_transform(X_train_raw[cat_cols].astype(str))
    X_val_cat[cat_cols] = ord_enc.transform(X_val_raw[cat_cols].astype(str))

    # K-Fold Out-of-Fold Target Encoding with smoothing
    global_mean = float(np.mean(y_train))
    smoothing_weight = 10.0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in all_cat_features:
        oof_te = np.zeros(len(X_train_raw), dtype=np.float32)
        train_s = X_train_raw[col].astype(str)
        val_s = X_val_raw[col].astype(str)

        for train_idx, val_idx in kf.split(X_train_raw, y_train):
            tr_cat = train_s.iloc[train_idx]
            tr_target = pd.Series(y_train[train_idx], index=tr_cat.index)
            stats = tr_target.groupby(tr_cat).agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + smoothing_weight * global_mean) / (stats['count'] + smoothing_weight)
            oof_te[val_idx] = train_s.iloc[val_idx].map(smoothed).fillna(global_mean).values

        # Full train mapping for validation split
        full_stats = pd.Series(y_train, index=train_s.index).groupby(train_s).agg(['count', 'mean'])
        full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing_weight * global_mean) / (full_stats['count'] + smoothing_weight)

        X_train_cat[f'{col}_te'] = oof_te
        X_val_cat[f'{col}_te'] = val_s.map(full_smoothed).fillna(global_mean).values.astype(np.float32)

# 5. Assemble final training and validation feature sets
X_train = pd.concat([bin_train, bin_train_stats, pca_train, svd_train, X_train_cat], axis=1).astype(np.float32)
X_val = pd.concat([bin_val, bin_val_stats, pca_val, svd_val, X_val_cat], axis=1).astype(np.float32)


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
            ("identity", FunctionTransformer()),
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
                FastICA(n_components=10, random_state=42, max_iter=1000, tol=0.01),
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
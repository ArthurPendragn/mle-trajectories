import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
import xgboost as xgb

# 1. Load Data & Prepare Splits
train_df = pd.read_csv('./input/train.csv')
y = train_df['y'].values
X = train_df.drop(columns=['ID', 'y'])

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

X = X.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. Modular Ensemble Builder for Ablation Testing
def build_pipeline_ensemble(
    use_decomp=True, use_ohe=True, meta_learner_type='lasso_pos'
):
    categorical_selector = make_column_selector(
        dtype_include=['object', 'category']
    )
    numerical_selector = make_column_selector(dtype_include=['number'])

    if use_ohe:
        encoder_ct = ColumnTransformer(
            transformers=[
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    categorical_selector,
                ),
                ('num', 'passthrough', numerical_selector),
            ],
            remainder='passthrough',
        )
    else:
        encoder_ct = 'passthrough'

    if use_decomp:
        decomp_union = FeatureUnion(
            [
                ('identity', Pipeline([('passthrough', 'passthrough')])),
                ('pca', PCA(n_components=8, random_state=42)),
                ('tsvd', TruncatedSVD(n_components=8, random_state=42)),
                ('ica', FastICA(n_components=8, random_state=42, max_iter=1000)),
                ('grp', GaussianRandomProjection(n_components=8, random_state=42)),
            ]
        )
    else:
        decomp_union = 'passthrough'

    linear_steps = []
    if encoder_ct != 'passthrough':
        linear_steps.append(('encode', encoder_ct))
    if decomp_union != 'passthrough':
        linear_steps.append(('decomp', decomp_union))
    linear_steps.append(('scale', RobustScaler()))

    linear_preprocessor = (
        Pipeline(linear_steps)
        if len(linear_steps) > 1
        else Pipeline([('scale', RobustScaler())])
    )

    if decomp_union != 'passthrough':
        tree_preprocessor = Pipeline([('decomp', decomp_union)])
    else:
        tree_preprocessor = 'passthrough'

    def make_tree_pipeline(model):
        if tree_preprocessor != 'passthrough':
            return Pipeline([('prep', tree_preprocessor), ('model', model)])
        return model

    estimators = [
        (
            'lgb',
            make_tree_pipeline(
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
                )
            ),
        ),
        (
            'xgb',
            make_tree_pipeline(
                xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.02,
                    max_depth=3,
                    colsample_bytree=0.6,
                    subsample=0.8,
                    reg_alpha=1.0,
                    random_state=42,
                    verbosity=0,
                )
            ),
        ),
        (
            'cb',
            make_tree_pipeline(
                CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.02,
                    depth=3,
                    l2_leaf_reg=5.0,
                    random_seed=42,
                    verbose=False,
                )
            ),
        ),
        (
            'krr',
            Pipeline(
                [
                    ('prep', linear_preprocessor),
                    (
                        'model',
                        KernelRidge(
                            alpha=0.6, kernel='polynomial', degree=2, coef0=2.5
                        ),
                    ),
                ]
            ),
        ),
        (
            'huber',
            Pipeline(
                [
                    ('prep', linear_preprocessor),
                    (
                        'model',
                        HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=1000),
                    ),
                ]
            ),
        ),
        (
            'enet',
            Pipeline(
                [
                    ('prep', linear_preprocessor),
                    (
                        'model',
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

    if meta_learner_type == 'lasso_pos':
        meta_learner = LassoCV(
            alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            positive=True,
            cv=5,
            random_state=42,
            max_iter=2000,
        )
    elif meta_learner_type == 'ridge':
        meta_learner = RidgeCV(
            alphas=np.logspace(-2, 3, 30),
        )

    return StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


# 3. Perform Ablation Experiments
ablation_configs = {
    'Full Baseline (Decomp + OHE + LassoCV Positive)': {
        'use_decomp': True,
        'use_ohe': True,
        'meta_learner_type': 'lasso_pos',
    },
    'Ablation 1 (Without Decomposition Features)': {
        'use_decomp': False,
        'use_ohe': True,
        'meta_learner_type': 'lasso_pos',
    },
    'Ablation 2 (Meta-Learner: RidgeCV instead of LassoCV Positive)': {
        'use_decomp': True,
        'use_ohe': True,
        'meta_learner_type': 'ridge',
    },
    'Ablation 3 (Without One-Hot Encoding in Linear Preprocessor)': {
        'use_decomp': True,
        'use_ohe': False,
        'meta_learner_type': 'lasso_pos',
    },
}

results = {}

print('--- Starting Ablation Study ---')
for name, config in ablation_configs.items():
    model = build_pipeline_ensemble(**config)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = r2_score(y_val, preds)
    results[name] = score
    print(f'{name}: R2 = {score:.5f}')

# 4. Print Summary and Deltas
baseline_score = results['Full Baseline (Decomp + OHE + LassoCV Positive)']
print('\n--- Ablation Study Summary & Relative Impact ---')
impacts = {}
for name, score in results.items():
    if name == 'Full Baseline (Decomp + OHE + LassoCV Positive)':
        continue
    delta = score - baseline_score
    impacts[name] = delta
    print(f'{name} -> Delta R2: {delta:+.5f} (Score: {score:.5f})')

# Determine most impactful component
most_impactful = min(impacts.items(), key=lambda x: x[1])
print(
    f'\nMost critical component identified: {most_impactful[0]} (Disabling/changing it causes largest drop: {most_impactful[1]:.5f} R2)'
)
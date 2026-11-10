from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
import xgboost as xgb

# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y'].values
X_raw = train_df.drop(columns=['ID', 'y'])

# Identify categorical columns
cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

# Train/validation split on raw features
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# 1. Base feature preprocessing (Ordinal Encoding)
X_train_base = X_train_raw.copy()
X_val_base = X_val_raw.copy()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X_train_base[cat_cols] = encoder.fit_transform(
        X_train_base[cat_cols].astype(str)
    )
    X_val_base[cat_cols] = encoder.transform(X_val_base[cat_cols].astype(str))

X_train_base = X_train_base.astype(np.float32)
X_val_base = X_val_base.astype(np.float32)

# 2. Reference feature preprocessing (One-Hot Encoding + Dimensionality Reduction)
X_tr_enc = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
X_val_enc = pd.get_dummies(
    X_val_raw, columns=cat_cols, drop_first=True
).reindex(columns=X_tr_enc.columns, fill_value=0)

dim_reduction = FeatureUnion(
    [
        ('pca', PCA(n_components=10, random_state=42)),
        (
            'ica',
            FastICA(n_components=10, random_state=42, max_iter=1000, tol=1e-3),
        ),
        ('tsvd', TruncatedSVD(n_components=10, random_state=42)),
    ]
)

decomp_tr = dim_reduction.fit_transform(X_tr_enc)
decomp_val = dim_reduction.transform(X_val_enc)

X_train_decomp = np.hstack(
    [X_tr_enc.values.astype(np.float32), decomp_tr.astype(np.float32)]
)
X_val_decomp = np.hstack(
    [X_val_enc.values.astype(np.float32), decomp_val.astype(np.float32)]
)


# Build Base Stacking Ensemble
def build_stacking_ensemble():
    estimators = [
        (
            'lgb',
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=3,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            'xgb',
            xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=3,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                random_state=42,
                verbosity=0,
            ),
        ),
        (
            'cb',
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.02,
                depth=4,
                l2_leaf_reg=5.0,
                random_seed=42,
                verbose=False,
            ),
        ),
        (
            'krr',
            make_pipeline(
                RobustScaler(),
                KernelRidge(
                    alpha=0.6, kernel='polynomial', degree=2, coef0=2.5
                ),
            ),
        ),
        (
            'enet',
            make_pipeline(
                RobustScaler(),
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, random_state=42
                ),
            ),
        ),
    ]

    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking_reg


# Train Base Stacking Ensemble
base_stacking_model = build_stacking_ensemble()
base_stacking_model.fit(X_train_base, y_train)
y_val_pred_base = base_stacking_model.predict(X_val_base)

# Train Reference XGBoost Regressor
ref_xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.005,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse',
)

ref_xgb_model.fit(
    X_train_decomp,
    y_train,
    eval_set=[(X_val_decomp, y_val)],
    verbose=False,
)
y_val_pred_ref = ref_xgb_model.predict(X_val_decomp)

# Blend Predictions
y_val_pred = 0.55 * y_val_pred_base + 0.45 * y_val_pred_ref
final_validation_score = r2_score(y_val, y_val_pred)

print(f'Final Validation Performance: {final_validation_score}')
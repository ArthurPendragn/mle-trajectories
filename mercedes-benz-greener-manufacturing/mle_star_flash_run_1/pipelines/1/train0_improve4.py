import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.svm import SVR

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
    estimators = [
        (
            'lgb',
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=3,
                num_leaves=7,
                colsample_bytree=0.6,
                subsample=0.8,
                subsample_freq=1,
                reg_alpha=1.0,
                reg_lambda=1.0,
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
                subsample=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
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
                subsample=0.8,
                random_seed=42,
                verbose=False,
            ),
        ),
        (
            'krr',
            make_pipeline(
                VarianceThreshold(threshold=0.0),
                SelectPercentile(score_func=f_regression, percentile=60),
                RobustScaler(),
                KernelRidge(
                    alpha=0.6, kernel='polynomial', degree=2, coef0=2.5
                ),
            ),
        ),
        (
            'lasso',
            make_pipeline(
                VarianceThreshold(threshold=0.0),
                SelectPercentile(score_func=f_regression, percentile=60),
                RobustScaler(),
                LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, random_state=42),
            ),
        ),
        (
            'svr',
            make_pipeline(
                VarianceThreshold(threshold=0.0),
                SelectPercentile(score_func=f_regression, percentile=60),
                RobustScaler(),
                SVR(C=10.0, epsilon=0.1, kernel='rbf'),
            ),
        ),
        (
            'enet',
            make_pipeline(
                VarianceThreshold(threshold=0.0),
                SelectPercentile(score_func=f_regression, percentile=60),
                RobustScaler(),
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, random_state=42
                ),
            ),
        ),
    ]

    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(
            alphas=np.logspace(-2, 3, 30),
        ),
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
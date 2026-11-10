from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
import xgboost as xgb

# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y'].values
X_raw = train_df.drop(columns=['ID', 'y'])

# Identify categorical columns
cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for Base Stacking Ensemble (Ordinal Encoding + float32)
X_stack = X_raw.copy()
if cat_cols:
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    X_stack[cat_cols] = encoder.fit_transform(X_stack[cat_cols].astype(str))
X_stack = X_stack.astype(np.float32)

# Preprocessing for Reference LightGBM (Native Category Dtype)
X_ref = X_raw.copy()
for col in cat_cols:
    X_ref[col] = X_ref[col].astype('category')

# Train/validation split for both feature representations
X_train_stack, X_val_stack, y_train, y_val = train_test_split(
    X_stack, y, test_size=0.2, random_state=42
)
X_train_ref, X_val_ref, _, _ = train_test_split(
    X_ref, y, test_size=0.2, random_state=42
)


# Build Stacking Ensemble integrating CatBoost, LightGBM, XGBoost, KRR, and ElasticNet
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
stacking_model = build_stacking_ensemble()
stacking_model.fit(X_train_stack, y_train)
y_val_pred_stack = stacking_model.predict(X_val_stack)

# Train Reference LightGBM Regressor with early stopping & native categorical handling
ref_lgb_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=15,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=2.0,
    reg_lambda=5.0,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

ref_lgb_model.fit(
    X_train_ref,
    y_train,
    eval_set=[(X_val_ref, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
y_val_pred_ref = ref_lgb_model.predict(X_val_ref)

# Ensemble Predictions via Blending
y_val_pred_ensemble = 0.6 * y_val_pred_stack + 0.4 * y_val_pred_ref

# Evaluate on validation set
final_validation_score = r2_score(y_val, y_val_pred_ensemble)

print(f'Final Validation Performance: {final_validation_score}')
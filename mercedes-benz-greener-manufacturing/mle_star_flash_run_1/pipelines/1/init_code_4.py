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


# Build Stacking Ensemble
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


# Instantiate and fit model
model = build_stacking_ensemble()
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
final_validation_score = r2_score(y_val, y_val_pred)

print(f'Final Validation Performance: {final_validation_score}')
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y']
X = train_df.drop(columns=['ID', 'y'])

# Identify categorical features and convert to category dtype
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype('category')

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize LightGBM Regressor with feature subsampling & native categorical handling
model = lgb.LGBMRegressor(
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
    verbose=-1
)

# Fit model with early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# Predict on validation set and evaluate R2
y_val_pred = model.predict(X_val)
final_validation_score = r2_score(y_val, y_val_pred)

print(f"Final Validation Performance: {final_validation_score}")
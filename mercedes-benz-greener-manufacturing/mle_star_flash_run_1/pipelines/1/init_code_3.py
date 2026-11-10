import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load training data
train_df = pd.read_csv('./input/train.csv')

# Separate target and features
y = train_df['y']
X = train_df.drop(columns=['ID', 'y'])

# Identify categorical features and ensure string type
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    cat_features=cat_cols,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=False
)

# Train model
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict on validation set and evaluate R2
y_val_pred = model.predict(X_val)
final_validation_score = r2_score(y_val, y_val_pred)

print(f"Final Validation Performance: {final_validation_score}")
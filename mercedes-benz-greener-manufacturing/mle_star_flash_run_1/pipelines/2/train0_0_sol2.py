import os
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("./input/train.csv")
test_path = "./input/test.csv"
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Separate features and target
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

# Identify categorical features
cat_features = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]

# Convert categorical columns to string type for CatBoost native handling
for col in cat_features:
    X[col] = X[col].astype(str)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize CatBoostRegressor with native categorical support
model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    subsample=0.8,
    random_seed=42,
    cat_features=cat_features,
    eval_metric="RMSE",
    early_stopping_rounds=50,
    verbose=False,
)

# Fit model
model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

# Validate model
val_preds = model.predict(X_val)
final_validation_score = r2_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")

# Generate test predictions
if test_df is not None:
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )
    for col in cat_features:
        X_test[col] = X_test[col].astype(str)
    test_preds = model.predict(X_test)
    submission = pd.DataFrame({"y": test_preds})
    submission.to_csv("submission.csv", index=False)
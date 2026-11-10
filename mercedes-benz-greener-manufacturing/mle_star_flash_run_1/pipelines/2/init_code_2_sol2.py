import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load training data
train_df = pd.read_csv("./input/train.csv")

# Separate target and features
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

# Identify categorical features
categorical_cols = [
    col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
]

# Check if test data exists to ensure consistent label encoding
test_path = "./input/test.csv"
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    X_test = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )
else:
    test_df = None
    X_test = None

X_encoded = X.copy()
if X_test is not None:
    X_test_encoded = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    if X_test is not None and col in X_test.columns:
        combined_vals = list(
            set(X[col].astype(str)).union(set(X_test[col].astype(str)))
        )
        le.fit(combined_vals)
        X_encoded[col] = le.transform(X_encoded[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
    else:
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# Ensure all columns are numeric
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
if X_test is not None:
    X_test_encoded = X_test_encoded.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# XGBoost hyperparameters with constrained depth and strong subsampling
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.008,
    "max_depth": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "alpha": 2.0,
    "lambda": 5.0,
    "random_state": 42,
    "tree_method": "hist",
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# Predict on validation set
val_preds = model.predict(dval)
final_validation_score = r2_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")

# Generate test predictions if test set exists
if X_test is not None:
    dtest = xgb.DMatrix(X_test_encoded)
    test_preds = model.predict(dtest)
    submission = pd.DataFrame({"y": test_preds})
    submission.to_csv("submission.csv", index=False)
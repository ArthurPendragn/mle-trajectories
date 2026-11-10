import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Load data
train_df = pd.read_csv("./input/train.csv")
test_path = "./input/test.csv"
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Separate target and features
y = train_df["y"].values
X_train_raw = train_df.drop(columns=["ID", "y"])

if test_df is not None:
    X_test_raw = (
        test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    )
    combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    X_encoded = combined_encoded.iloc[: len(train_df)].values.astype(np.float64)
    X_test_encoded = combined_encoded.iloc[len(train_df) :].values.astype(
        np.float64
    )
else:
    X_encoded = (
        pd.get_dummies(X_train_raw, drop_first=True).values.astype(np.float64)
    )
    X_test_encoded = None

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Base models
base_models = [
    (
        "elastic",
        Pipeline(
            [
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=25, random_state=42)),
                (
                    "reg",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42
                    ),
                ),
            ]
        ),
    ),
    (
        "kernel_ridge",
        Pipeline(
            [
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=20, random_state=42)),
                ("reg", KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
            ]
        ),
    ),
]

# Stacking Regressor
stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=ElasticNetCV(cv=5, random_state=42),
    cv=5,
    n_jobs=-1,
)

# Fit model
stacking_reg.fit(X_train, y_train)

# Predict and evaluate on validation set
val_preds = stacking_reg.predict(X_val)
final_validation_score = r2_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")

# Generate test predictions
if X_test_encoded is not None:
    test_preds = stacking_reg.predict(X_test_encoded)
    submission = pd.DataFrame({"y": test_preds})
    submission.to_csv("submission.csv", index=False)
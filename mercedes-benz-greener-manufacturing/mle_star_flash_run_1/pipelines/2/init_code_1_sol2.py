import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load data
train_df = pd.read_csv("./input/train.csv")

# Separate target and ID
y = train_df["y"].values
X = train_df.drop(columns=["ID", "y"])

# Identify categorical columns reliably
categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

# Encode categorical columns using LabelEncoder
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# Ensure all columns are numeric
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train / Validation Split
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Dimensionality reduction on non-categorical features
binary_cols = [c for c in X.columns if c not in categorical_cols]

decomp_union = FeatureUnion(
    [
        ("pca", PCA(n_components=10, random_state=42)),
        ("ica", FastICA(n_components=10, random_state=42, max_iter=1000)),
        ("tsvd", TruncatedSVD(n_components=10, random_state=42)),
    ]
)

decomp_train = decomp_union.fit_transform(X_train_df[binary_cols].values.astype(np.float64))
decomp_val = decomp_union.transform(X_val_df[binary_cols].values.astype(np.float64))

X_tr_feat = np.hstack([X_train_df.values.astype(np.float64), decomp_train])
X_va_feat = np.hstack([X_val_df.values.astype(np.float64), decomp_val])

# Initialize and train LightGBM Regressor
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_tr_feat,
    y_train,
    eval_set=[(X_va_feat, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Evaluate on validation set
val_preds = model.predict(X_va_feat)
final_validation_score = r2_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")

# If test.csv exists, generate test predictions
test_path = "./input/test.csv"
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["ID"]) if "ID" in test_df.columns else test_df.copy()
    X_test_encoded = X_test.copy()
    for col in categorical_cols:
        if col in X_test_encoded.columns:
            le = LabelEncoder()
            # Fit on combined train and test categories to handle unseen values safely
            unique_vals = list(set(X[col].astype(str)).union(set(X_test[col].astype(str))))
            le.fit(unique_vals)
            X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
    X_test_encoded = X_test_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    decomp_test = decomp_union.transform(X_test_encoded[binary_cols].values.astype(np.float64))
    X_te_feat = np.hstack([X_test_encoded.values.astype(np.float64), decomp_test])
    test_preds = model.predict(X_te_feat)
    submission = pd.DataFrame({"y": test_preds})
    submission.to_csv("submission.csv", index=False)
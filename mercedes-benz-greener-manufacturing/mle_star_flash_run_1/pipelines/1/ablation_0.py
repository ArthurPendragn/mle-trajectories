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

# Load training data
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


# Base estimators definition
def get_base_estimators():
    return {
        'lgb': lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            random_state=42,
            verbose=-1,
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            random_state=42,
            verbosity=0,
        ),
        'cb': CatBoostRegressor(
            iterations=500,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=5.0,
            random_seed=42,
            verbose=False,
        ),
        'krr': make_pipeline(
            RobustScaler(),
            KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
        ),
        'enet': make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, random_state=42
            ),
        ),
    }


def evaluate_stacking(estimator_dict):
    estimators = [(name, est) for name, est in estimator_dict.items()]
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    stacking_reg.fit(X_train, y_train)
    y_pred = stacking_reg.predict(X_val)
    return r2_score(y_val, y_pred)


# Run baseline evaluation
full_estimators = get_base_estimators()
baseline_score = evaluate_stacking(full_estimators)
print(f'Baseline (All Estimators) R2: {baseline_score:.5f}')

# Run leave-one-out ablations for each component
ablation_results = {}
performance_drops = {}

for name in full_estimators.keys():
    ablated_estimators = {k: v for k, v in full_estimators.items() if k != name}
    score = evaluate_stacking(ablated_estimators)
    drop = baseline_score - score
    ablation_results[f'Without {name}'] = score
    performance_drops[name] = drop
    print(
        f'Ablation [Without {name:4s}] -> Validation R2: {score:.5f} (Delta: {-drop:+.5f})'
    )

# Determine most impactful component
most_impactful_component = max(performance_drops, key=performance_drops.get)
max_drop = performance_drops[most_impactful_component]

print('\n--- Ablation Study Summary ---')
print(
    f'The component contributing the most to overall performance is "{most_impactful_component}" '
    f'(Removing it causes the largest performance drop of {max_drop:.5f} R2).'
)
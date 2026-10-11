import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"


def build():
    df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(bool)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_marked = X.skb.mark_as_X(cv=cv, split_kwargs={})
    y_marked = y.skb.mark_as_y()

    audit = {
        "population_check": skrub.as_data_op(
            {
                "all_train_rows": {
                    "row_count": 8693,
                    "adversarial_auc": 0.644,
                }
            }
        ),
        "learning_curve": skrub.as_data_op(
            {
                "model": "HistGradientBoostingClassifier",
                "sizes": [1738, 3477, 5215, 6954],
                "val_accuracy": [0.767, 0.783, 0.800, 0.798],
                "fit_time_seconds": [0.55, 0.61, 0.65, 0.69],
                "pool_size_chosen": 8693,
                "rationale": (
                    "The full training pool of 8693 rows fits in under 1 second per fold, "
                    "achieving ~80% accuracy and leaving generous time budget for model exploration."
                ),
            }
        ),
    }

    return {
        "X": X_marked,
        "y": y_marked,
        "scoring": "accuracy",
        "audit": audit,
    }
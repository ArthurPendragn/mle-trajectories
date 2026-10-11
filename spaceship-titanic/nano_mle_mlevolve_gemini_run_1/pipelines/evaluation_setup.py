import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/test.csv"
TARGET_COL = "Transported"


def build():
    train_raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test_raw = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_raw = train_raw.drop(columns=[TARGET_COL])
    y_raw = train_raw[TARGET_COL]

    X = X_raw.skb.mark_as_X(cv=cv, split_kwargs={})
    y = y_raw.skb.mark_as_y()

    audit = {
        "train_rows": train_raw.skb.apply_func(len),
        "test_rows": test_raw.skb.apply_func(len),
    }

    return {
        "X": X,
        "y": y,
        "scoring": "accuracy",
        "audit": audit,
    }
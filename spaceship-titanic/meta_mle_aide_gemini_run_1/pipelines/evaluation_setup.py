import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/test.csv"
TARGET = "Transported"


def build():
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    raw_X = train.drop(columns=[TARGET])
    raw_y = train[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_kwargs = {}

    if hasattr(raw_X.skb, "mark_as_X"):
        X = raw_X.skb.mark_as_X(cv=cv, split_kwargs=split_kwargs)
    else:
        X = getattr(skrub, "mark_as_X")(raw_X, cv=cv, split_kwargs=split_kwargs)

    if hasattr(raw_y.skb, "mark_as_y"):
        y = raw_y.skb.mark_as_y()
    else:
        y = getattr(skrub, "mark_as_y")(raw_y)

    row_keys = train["PassengerId"]

    audit = {
        "train_rows": train.skb.apply_func(len),
        "test_rows": test.skb.apply_func(len),
    }

    return {
        "X": X,
        "y": y,
        "scoring": "accuracy",
        "row_keys": row_keys,
        "audit": audit,
    }
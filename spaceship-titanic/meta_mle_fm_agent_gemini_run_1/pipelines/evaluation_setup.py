import pandas as pd
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"


def build():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = train_df.drop(columns=["Transported"])
    y = train_df["Transported"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    marked_X = X.skb.mark_as_X(cv=cv, split_kwargs={})
    marked_y = y.skb.mark_as_y()

    return {
        "X": marked_X,
        "y": marked_y,
        "scoring": "accuracy",
    }
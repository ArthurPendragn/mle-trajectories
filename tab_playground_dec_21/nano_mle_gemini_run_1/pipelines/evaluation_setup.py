import pandas as pd
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"


def mark_X(data, cv, split_kwargs=None):
    if split_kwargs is None:
        split_kwargs = {}
    if hasattr(data, "skb") and hasattr(data.skb, "mark_as_X"):
        return data.skb.mark_as_X(cv=cv, split_kwargs=split_kwargs)
    if hasattr(skrub, "mark_as_X"):
        return skrub.mark_as_X(data, cv=cv, split_kwargs=split_kwargs)
    return data.mark_as_X(cv=cv, split_kwargs=split_kwargs)


def mark_y(data):
    if hasattr(data, "skb") and hasattr(data.skb, "mark_as_y"):
        return data.skb.mark_as_y()
    if hasattr(skrub, "mark_as_y"):
        return skrub.mark_as_y(data)
    return data.mark_as_y()


def build_evaluation():
    raw_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = raw_df.drop(columns=["Cover_Type", "Id"])
    y = raw_df["Cover_Type"]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    split_kwargs = {}

    X_marked = mark_X(X, cv=cv, split_kwargs=split_kwargs)
    y_marked = mark_y(y)

    audit = {
        "class_counts": (
            raw_df[["Cover_Type"]]
            .groupby("Cover_Type")
            .size()
            .reset_index(name="count")
        )
    }

    return {
        "X": X_marked,
        "y": y_marked,
        "scoring": "accuracy",
        "audit": audit,
    }


def build():
    return build_evaluation()
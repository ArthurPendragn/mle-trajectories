import pandas as pd
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"


def build_evaluation():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)

    y = train_df["Cover_Type"].skb.mark_as_y()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = train_df.drop(columns=["Cover_Type", "Id"]).skb.mark_as_X(
        cv=cv, split_kwargs={}
    )

    audit = {
        "target_distribution": train_df.groupby("Cover_Type")
        .size()
        .reset_index(name="count"),
        "train_head": train_df.head(5),
    }

    return {
        "X": X,
        "y": y,
        "scoring": "accuracy",
        "audit": audit,
    }


def build():
    return build_evaluation()
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"


def build_evaluation():
    raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = raw.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(cv=cv, split_kwargs={})
    y = raw["Cover_Type"].skb.mark_as_y()
    row_keys = raw["Id"]
    return {
        "X": X,
        "y": y,
        "scoring": "accuracy",
        "row_keys": row_keys,
        "audit": {
            "target_counts": raw["Cover_Type"].value_counts().to_frame().reset_index(),
        },
    }


def build():
    return build_evaluation()
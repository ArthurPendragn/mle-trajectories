import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
CLASS_COUNTS = {
    1: 1468136,
    2: 2262087,
    3: 195712,
    4: 377,
    5: 1,
    6: 11426,
    7: 62261,
}


def build():
    data = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    data = data.sort_values("Id").reset_index(drop=True)

    # A single generator is shared by the ascending-label selection nodes.
    rng = skrub.as_data_op(42).skb.apply_func(np.random.default_rng)
    pieces = []
    for label, count in CLASS_COUNTS.items():
        sample_size = max(1, int(np.floor(0.1 * count + 0.5)))
        positions = rng.choice(count, size=sample_size, replace=False)
        class_rows = data[data["Cover_Type"] == label]
        pieces.append(class_rows.iloc[positions])

    rows = pieces[0].skb.concat(pieces[1:], axis=0)
    rows = rows.sort_values("Id").reset_index(drop=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = rows.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=cv, split_kwargs={}
    )
    y = rows["Cover_Type"].skb.mark_as_y()
    return {
        "X": X,
        "y": y,
        "scoring": "accuracy",
        "row_keys": rows["Id"],
        "audit": {
            "class_counts": rows["Cover_Type"].value_counts().sort_index(),
            "sample_shape": rows.shape,
            "id_bounds": rows["Id"].agg(["min", "max"]),
        },
    }
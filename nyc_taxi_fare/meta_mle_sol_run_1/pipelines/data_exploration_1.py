import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
]
COLS = ["pickup_datetime", "passenger_count"] + COORDS


def features(rows, rounded):
    x = rows[COORDS + ["passenger_count"]].copy()
    if rounded:
        for col in COORDS:
            x = x.assign(**{col: x[col].round(4)})
    dt = rows["pickup_datetime"].skb.apply_func(
        pd.to_datetime, utc=True, errors="coerce"
    )
    x = x.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
        second=dt.dt.second,
    )
    return x.fillna(-999)


def valid_mask(rows):
    return (
        rows["pickup_longitude"].between(-75, -72)
        & rows["dropoff_longitude"].between(-75, -72)
        & rows["pickup_latitude"].between(40, 42)
        & rows["dropoff_latitude"].between(40, 42)
        & rows["passenger_count"].between(1, 6)
    ).fillna(False)


def build():
    full = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv,
        usecols=COLS + ["fare_amount"],
        dtype={c: "float32" for c in COORDS + ["passenger_count", "fare_amount"]},
    )
    train = full.sample(n=250000, random_state=718)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=COLS,
        dtype={c: "float32" for c in COORDS + ["passenger_count"]},
    )
    out = {
        "source_row_count": full.shape[0],
        "train_summary": train.describe(include="all"),
        "test_summary": test.describe(include="all"),
        "train_missing": train.isna().sum(),
        "test_missing": test.isna().sum(),
        "train_passenger_counts": train["passenger_count"].value_counts(dropna=False),
        "test_passenger_counts": test["passenger_count"].value_counts(dropna=False),
        "test_valid_count": valid_mask(test).sum(),
    }
    dated = train.assign(
        source_block=train.index // 5000000,
        year=train["pickup_datetime"].str.slice(0, 4),
    )
    out["source_order_years"] = dated.groupby(["source_block", "year"]).size()
    out["source_order_means"] = dated.groupby("source_block")[
        COORDS + ["fare_amount", "passenger_count"]
    ].mean()
    out["test_years"] = test["pickup_datetime"].str.slice(0, 4).value_counts()
    out["fare_quantiles"] = train["fare_amount"].quantile(
        [0, .0001, .001, .01, .05, .5, .95, .99, .999, .9999, 1]
    )
    fare_bins = [-np.inf, 0, 2.5, 5, 10, 20, 50, 100, 200, np.inf]
    fare_band = train["fare_amount"].skb.apply_func(pd.cut, bins=fare_bins)
    train = train.assign(fare_band=fare_band)
    fare = train["fare_amount"]
    valid = valid_mask(train)
    rules = {
        "all": fare.notna() | fare.isna(),
        "observable_valid": valid,
        "fare_ge_0": fare.ge(0),
        "fare_ge_2_5": fare.ge(2.5),
        "fare_ge_5": fare.ge(5),
        "fare_le_100": fare.le(100),
        "fare_le_200": fare.le(200),
        "fare_0_100": fare.between(0, 100),
        "fare_2_5_100": fare.between(2.5, 100),
        "fare_2_5_200": fare.between(2.5, 200),
        "valid_fare_ge_2_5": valid & fare.ge(2.5),
        "valid_fare_le_100": valid & fare.le(100),
        "valid_fare_2_5_100": valid & fare.between(2.5, 100),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=371)
    for name, mask in rules.items():
        selected = train[mask.fillna(False)]
        out[name + "_sample_count"] = selected.shape[0]
        out[name + "_estimated_source_count"] = (
            selected.shape[0] * full.shape[0] / 250000
        )
        out[name + "_fare_bands"] = selected.groupby(
            "fare_band", observed=False
        ).size()
        sample = selected.sample(n=9914, random_state=391)
        for normalized in [False, True]:
            a = features(sample, normalized).assign(origin=0)
            b = features(test, normalized).assign(origin=1)
            both = a.skb.concat([b], axis=0).reset_index(drop=True)
            model = skrub.as_data_op(
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=15,
                    learning_rate=0.08,
                    l2_regularization=10,
                    random_state=41,
                )
            )
            suffix = "_rounded" if normalized else "_raw"
            out[name + suffix + "_auc_folds"] = model.skb.apply_func(
                cross_val_score,
                both.drop(columns=["origin"]),
                both["origin"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,
            )
    for name, frame in [("train", train), ("test", test)]:
        precision = frame[COORDS].copy()
        for col in COORDS:
            precision = precision.assign(
                **{col: (frame[col] - frame[col].round(4)).abs()}
            )
        out[name + "_rounding_residuals"] = precision.describe()
    return out
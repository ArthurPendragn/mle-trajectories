import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
]
INPUTS = ["pickup_datetime"] + COORDS + ["passenger_count"]
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=1729)

# Sampling: read the complete source, then uniform sampling without replacement.
# This does not assume random source ordering and can be repeated at larger sizes.
# No keys, timestamp frequencies, or source indices enter the classifier.
def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    return rows[COORDS + ["passenger_count"]].assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
        second=dt.dt.second,
    )

def auc(train_features, test_features):
    # Each comparison uses the same number of source and prediction rows,
    # the same sampling seed, and the same stratified folds.
    a = train_features.sample(n=9914, random_state=7301).assign(domain=0)
    b = test_features.assign(domain=1)
    both = a.skb.concat([b], axis=0).reset_index(drop=True)
    model = LGBMClassifier(
        n_estimators=180, num_leaves=15, learning_rate=0.05,
        min_child_samples=60, reg_lambda=5.0,
        n_jobs=8, verbosity=-1, random_state=1729,
    )
    return skrub.as_data_op(model).skb.apply_func(
        cross_val_score,
        both.drop(columns="domain"),
        both["domain"],
        cv=CV, scoring="roc_auc", n_jobs=1,
    )

def build():
    full = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    train = full.sample(n=250000, random_state=2026).reset_index(drop=True)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=INPUTS
    )
    tf = features(train)
    qf = features(test)
    # Broad observable validity: both endpoints in a large NYC-region box,
    # and passenger counts present in the prediction source. No fare filtering.
    valid = (
        train["pickup_longitude"].between(-75, -72)
        & train["dropoff_longitude"].between(-75, -72)
        & train["pickup_latitude"].between(40, 42)
        & train["dropoff_latitude"].between(40, 42)
        & train["passenger_count"].between(1, 6)
    ).fillna(False)
    test_valid = (
        test["pickup_longitude"].between(-75, -72)
        & test["dropoff_longitude"].between(-75, -72)
        & test["pickup_latitude"].between(40, 42)
        & test["dropoff_latitude"].between(40, 42)
        & test["passenger_count"].between(1, 6)
    ).fillna(False)
    fare = train["fare_amount"]
    rules = {
        "all": fare.notna(),
        "observable_valid": valid,
        "fare_ge_0": fare.ge(0).fillna(False),
        "fare_ge_2_5": fare.ge(2.5).fillna(False),
        "fare_ge_5": fare.ge(5).fillna(False),
        "fare_le_100": fare.le(100).fillna(False),
        "fare_le_200": fare.le(200).fillna(False),
        "fare_0_100": fare.between(0, 100).fillna(False),
        "fare_2_5_100": fare.between(2.5, 100).fillna(False),
        "fare_2_5_200": fare.between(2.5, 200).fillna(False),
        "valid_ge_2_5": valid & fare.ge(2.5).fillna(False),
        "valid_le_100": valid & fare.le(100).fillna(False),
        "valid_2_5_100": valid & fare.between(2.5, 100).fillna(False),
    }
    out = {
        "source_count": full.shape[0],
        "sample_count": train.shape[0],
        "test_valid_count": test[test_valid].shape[0],
        "train_second_counts": tf["second"].value_counts().sort_index(),
        "test_second_counts": qf["second"].value_counts().sort_index(),
        "train_minute_counts": tf["minute"].value_counts().sort_index(),
        "test_minute_counts": qf["minute"].value_counts().sort_index(),
        "train_timestamp_multiplicity": train["pickup_datetime"].value_counts().describe(),
        "test_timestamp_multiplicity": test["pickup_datetime"].value_counts().describe(),
        "train_top_timestamps": train["pickup_datetime"].value_counts().head(20),
        "test_top_timestamps": test["pickup_datetime"].value_counts().head(20),
    }
    ablations = {
        "raw": COORDS + ["passenger_count", "year", "month", "day", "weekday", "hour", "minute", "second"],
        "minute_normalized": COORDS + ["passenger_count", "year", "month", "day", "weekday", "hour", "minute"],
        "hour_normalized": COORDS + ["passenger_count", "year", "month", "day", "weekday", "hour"],
        "coarse_calendar": COORDS + ["passenger_count", "year", "month", "weekday", "hour"],
        "locations_only": COORDS,
        "passengers_only": ["passenger_count"],
        "calendar_only": ["year", "month", "day", "weekday", "hour"],
        "clock_only": ["hour", "minute", "second"],
        "resolution_only": ["minute", "second"],
    }
    for name, cols in ablations.items():
        out["ablation_" + name] = auc(tf[cols], qf[cols])
    for name, mask in rules.items():
        selected = train[mask]
        sf = tf[mask]
        out[name + "_count"] = selected.shape[0]
        out[name + "_estimated_source_count"] = selected.shape[0] * full.shape[0] / 250000
        bands = selected["fare_amount"].skb.apply_func(
            pd.cut, bins=[-np.inf, 0, 2.5, 5, 10, 20, 50, 100, 200, np.inf]
        )
        out[name + "_fare_bands"] = bands.value_counts(sort=False)
        for resolution in ["minute_normalized", "hour_normalized"]:
            cols = ablations[resolution]
            out[name + "_" + resolution + "_auc"] = auc(sf[cols], qf[cols])
    return out
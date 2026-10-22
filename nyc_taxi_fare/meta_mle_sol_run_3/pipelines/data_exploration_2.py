import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
INPUTS = COORDS + ["passenger_count", "pickup_datetime"]
TIME = ["year", "month", "day", "weekday", "hour", "minute", "second"]

def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    return rows[COORDS + ["passenger_count"]].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour,
        minute=dt.dt.minute, second=dt.dt.second
    )

def auc(train, test, columns):
    a = train.sample(n=9914, random_state=714)[columns].assign(source_label=0)
    b = test[columns].assign(source_label=1)
    joined = a.skb.concat([b], axis=0).reset_index(drop=True)
    estimator = LGBMClassifier(
        n_estimators=180, num_leaves=15, learning_rate=0.05,
        max_depth=-1, min_child_samples=50, verbosity=-1,
        n_jobs=4, random_state=714
    )
    return skrub.as_data_op(estimator).skb.apply_func(
        cross_val_score, joined[columns], joined["source_label"],
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=714),
        n_jobs=1
    )

def build():
    raw = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    sample = raw.sample(n=500000, random_state=714)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=INPUTS
    )
    train_features = features(sample)
    test_features = features(test)
    all_columns = COORDS + ["passenger_count"] + TIME
    out = {}
    groups = {
        "coordinates": COORDS,
        "passengers": ["passenger_count"],
        "timestamp": TIME,
        "combined": all_columns,
        "calendar_date": ["year", "month", "day"],
        "clock": ["hour", "minute", "second"],
        "without_seconds": [c for c in all_columns if c != "second"],
        "without_day": [c for c in all_columns if c != "day"],
        "without_calendar_date": COORDS + ["passenger_count", "weekday", "hour", "minute", "second"],
    }
    for name, cols in groups.items():
        out[name + "_auc"] = auc(train_features, test_features, cols)
    for c in TIME:
        out["only_" + c + "_auc"] = auc(train_features, test_features, [c])
        out["train_" + c + "_counts"] = train_features[c].value_counts().sort_index()
        out["test_" + c + "_counts"] = test_features[c].value_counts().sort_index()
    for c in COORDS:
        out["train_" + c + "_rounding"] = (
            train_features[c] - train_features[c].round(4)
        ).abs().describe()
        out["test_" + c + "_rounding"] = (
            test_features[c] - test_features[c].round(4)
        ).abs().describe()

    valid = sample["passenger_count"].between(1, 6)
    for c in ["pickup_longitude", "dropoff_longitude"]:
        valid = valid & sample[c].between(-75, -72).fillna(False)
    for c in ["pickup_latitude", "dropoff_latitude"]:
        valid = valid & sample[c].between(40, 42).fillna(False)
    fare = sample["fare_amount"]
    masks = {
        "all": fare.notna(),
        "valid_inputs": valid,
        "positive": fare > 0,
        "at_least_2_5": fare >= 2.5,
        "at_least_5": fare >= 5,
        "at_most_100": fare <= 100,
        "at_most_200": fare <= 200,
        "band_2_5_100": fare.between(2.5, 100),
        "band_0_200": fare.between(0, 200),
        "valid_band_2_5_100": valid & fare.between(2.5, 100),
    }
    rounded_test = test_features.assign(
        **{c: test_features[c].round(4) for c in COORDS}
    )
    for name, mask in masks.items():
        rows = sample[mask.fillna(False)]
        f = features(rows)
        out[name + "_retained"] = rows.shape[0]
        out[name + "_fare_summary"] = rows["fare_amount"].describe(
            percentiles=[0.01, 0.5, 0.99]
        )
        out[name + "_raw_auc"] = auc(f, test_features, all_columns)
        rounded = f.assign(**{c: f[c].round(4) for c in COORDS})
        out[name + "_rounded_auc"] = auc(rounded, rounded_test, all_columns)

    dt_train = sample["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    dt_test = test["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    out["train_daily_counts"] = dt_train.dt.floor("D").value_counts().sort_index()
    out["test_daily_counts"] = dt_test.dt.floor("D").value_counts().sort_index()
    out["design"] = skrub.as_data_op({
        "sampling": "Read only task source columns; pandas sample(n=500000, random_state=714) over full source. For nested learning curves, sample maximum size once then use iloc prefixes of that randomly ordered sample.",
        "adversary": "LGBMClassifier, 180 trees, 15 leaves, learning_rate=.05, min_child_samples=50; no key or identifiers in features.",
        "folds": "Three shuffled stratified folds, random_state=714, sequential cross_val_score; 9914 deterministic candidate rows versus all 9914 test rows.",
        "valid_inputs": "Both longitudes in [-75,-72], both latitudes in [40,42], passenger_count in [1,6]; missing values fail.",
        "normalization": "Coordinates rounded to four decimals; independently report exclusion of second and day features without excluding rows.",
        "selection_caution": "No substantial fare or timestamp-based exclusion is recommended from adversarial evidence alone. Fare>=5 is diagnostic only, pending common-validation fare RMSE evidence."
    })
    return out
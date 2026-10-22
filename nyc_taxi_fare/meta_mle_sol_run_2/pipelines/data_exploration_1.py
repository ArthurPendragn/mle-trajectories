import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
INPUTS = [
    "pickup_datetime", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]

def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True, errors="coerce")
    out = rows[INPUTS[1:]].copy()
    out = out.assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        hour=dt.dt.hour, minute=dt.dt.minute, weekday=dt.dt.dayofweek,
        dx=rows["dropoff_longitude"] - rows["pickup_longitude"],
        dy=rows["dropoff_latitude"] - rows["pickup_latitude"]
    )
    out = out.assign(
        abs_dx=out["dx"].abs(),
        abs_dy=out["dy"].abs(),
        distance_squared=out["dx"] ** 2 + out["dy"] ** 2
    )
    return out.replace([np.inf, -np.inf], np.nan)

def valid_inputs(rows):
    return (
        rows["pickup_longitude"].between(-74.3, -73.6)
        & rows["dropoff_longitude"].between(-74.3, -73.6)
        & rows["pickup_latitude"].between(40.45, 41.0)
        & rows["dropoff_latitude"].between(40.45, 41.0)
        & rows["passenger_count"].between(1, 6)
    ).fillna(False)

def build():
    full = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(pd.read_csv)
    sample = full.sample(n=400000, random_state=1971).reset_index(drop=True)
    pool = sample.iloc[:350000]
    validation = sample.iloc[350000:]
    validation = validation[valid_inputs(validation)]
    fare = pool["fare_amount"]
    valid = valid_inputs(pool)
    rules = {
        "unfiltered": fare.notna(),
        "input_valid": valid,
        "fare_at_least_zero": fare.ge(0),
        "fare_at_least_2_5": fare.ge(2.5),
        "fare_at_most_100": fare.le(100),
        "fare_at_most_200": fare.le(200),
        "fare_band_0_200": fare.between(0, 200),
        "fare_band_2_5_100": fare.between(2.5, 100),
        "input_valid_fare_at_least_zero": valid & fare.ge(0),
        "input_valid_fare_at_most_200": valid & fare.le(200),
        "input_valid_fare_band_2_5_100": valid & fare.between(2.5, 100),
    }
    result = {
        "source_rows": full.shape[0],
        "sample_description": sample.describe(include="all"),
        "test_description": test.describe(include="all"),
        "sample_missing": sample.isna().mean(),
        "test_missing": test.isna().mean(),
        "fare_quantiles": sample["fare_amount"].quantile(
            [0, .0001, .001, .01, .5, .95, .99, .999, .9999, 1]
        ),
        "sample_duplicate_input_fraction": sample.duplicated(subset=INPUTS).mean(),
        "test_duplicate_input_fraction": test.duplicated(subset=INPUTS).mean(),
        "source_head": full.head(8),
        "source_tail": full.tail(8),
        "test_head": test.head(8),
        "sampling_protocol": skrub.as_data_op({
            "method": "Uniform deterministic pandas sample over the full source",
            "seed": 1971,
            "sample_size": 400000,
            "cache": "None; source-only graph recomputes samples",
            "validation": "Last 50000 sampled rows, input-valid only; unchanged across rules",
            "metric": "Negative RMSE; larger is better",
        }),
    }
    for name, mask in rules.items():
        selected = pool[mask.fillna(False)]
        result[name + "_count"] = selected.shape[0]
        result[name + "_retention"] = mask.fillna(False).mean()
        result[name + "_fare_summary"] = selected["fare_amount"].describe(
            percentiles=[.01, .5, .99, .999]
        )
        result[name + "_excluded_fares"] = pool[~mask.fillna(False)]["fare_amount"].describe()
        candidate = selected.sample(n=9914, random_state=29)
        a = features(candidate).assign(origin=0)
        b = features(test).assign(origin=1)
        domain = a.skb.concat([b], axis=0).reset_index(drop=True)
        clf = LGBMClassifier(
            n_estimators=140, num_leaves=15, learning_rate=.06,
            max_depth=-1, min_child_samples=70,
            colsample_bytree=.9, reg_lambda=4,
            n_jobs=8, random_state=81, verbosity=-1
        )
        result[name + "_adversarial_auc"] = skrub.as_data_op(clf).skb.apply_func(
            cross_val_score,
            domain.drop(columns="origin"), domain["origin"],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=81),
            scoring="roc_auc", n_jobs=1
        )
        train_rows = selected.sample(n=90000, random_state=47)
        train_features = features(train_rows).assign(
            target=train_rows["fare_amount"], fold=-1
        )
        val_features = features(validation).assign(
            target=validation["fare_amount"], fold=0
        )
        regression = train_features.skb.concat(
            [val_features], axis=0
        ).reset_index(drop=True)
        split = regression["fold"].skb.apply_func(PredefinedSplit)
        reg = LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=.06,
            min_child_samples=70, reg_lambda=5,
            n_jobs=8, random_state=47, verbosity=-1
        )
        result[name + "_shared_validation_negative_rmse"] = skrub.as_data_op(reg).skb.apply_func(
            cross_val_score,
            regression.drop(columns=["target", "fold"]),
            regression["target"], cv=split,
            scoring="neg_root_mean_squared_error", n_jobs=1
        )
    return result
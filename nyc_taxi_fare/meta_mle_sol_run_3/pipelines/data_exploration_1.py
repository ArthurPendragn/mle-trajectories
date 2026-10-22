import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = ["pickup_longitude", "pickup_latitude",
          "dropoff_longitude", "dropoff_latitude"]
COLS = ["pickup_datetime", "passenger_count"] + COORDS


def features(rows, precision=None):
    x = rows[COORDS + ["passenger_count"]].copy()
    if precision is not None:
        x = x.round(precision)
    dt = rows["pickup_datetime"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True)
    x = x.assign(year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
                 hour=dt.dt.hour, minute=dt.dt.minute,
                 second=dt.dt.second, weekday=dt.dt.dayofweek)
    x = x.assign(
        delta_lon=x["dropoff_longitude"] - x["pickup_longitude"],
        delta_lat=x["dropoff_latitude"] - x["pickup_latitude"])
    return x


def build():
    train = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=COLS + ["fare_amount"],
        dtype={c: "float64" for c in COORDS + ["fare_amount", "passenger_count"]})
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=COLS)
    sample = train.sample(n=500000, random_state=714)
    fare = sample["fare_amount"]
    valid = (
        sample["pickup_longitude"].between(-75, -72) &
        sample["dropoff_longitude"].between(-75, -72) &
        sample["pickup_latitude"].between(40, 42) &
        sample["dropoff_latitude"].between(40, 42) &
        sample["passenger_count"].between(1, 6)
    ).fillna(False)
    rules = {
        "all": sample,
        "valid_inputs": sample[valid],
        "fare_positive": sample[(fare > 0).fillna(False)],
        "fare_at_least_2_5": sample[(fare >= 2.5).fillna(False)],
        "fare_at_least_5": sample[(fare >= 5).fillna(False)],
        "fare_at_most_100": sample[(fare <= 100).fillna(False)],
        "fare_at_most_200": sample[(fare <= 200).fillna(False)],
        "fare_2_5_to_100": sample[fare.between(2.5, 100).fillna(False)],
        "fare_0_to_200": sample[fare.between(0, 200).fillna(False)],
        "valid_and_fare_2_5_to_100": sample[
            valid & fare.between(2.5, 100).fillna(False)],
    }
    out = {
        "source_row_count": train.shape[0],
        "sample_numeric_summary": sample[COORDS + ["passenger_count", "fare_amount"]].describe(
            percentiles=[0.001, 0.01, 0.5, 0.95, 0.99, 0.999]),
        "test_numeric_summary": test[COORDS + ["passenger_count"]].describe(
            percentiles=[0.001, 0.01, 0.5, 0.95, 0.99, 0.999]),
        "sample_missing": sample.isna().sum(),
        "test_missing": test.isna().sum(),
        "train_passengers": sample["passenger_count"].value_counts(dropna=False),
        "test_passengers": test["passenger_count"].value_counts(dropna=False),
        "sampling_design": skrub.as_data_op({
            "source": ROOT + "/train.csv.gz",
            "method": "pandas sample across the complete source, not a prefix",
            "random_state": 714,
            "n": 500000,
            "reusable_file": None,
            "note": "No derived file is written; reproduce this recorded source and sampling graph."
        }),
    }
    sf = features(sample)
    tf = features(test)
    out["train_years"] = sf["year"].value_counts(dropna=False)
    out["test_years"] = tf["year"].value_counts(dropna=False)
    out["train_months"] = sf.groupby(["year", "month"]).size()
    out["test_months"] = tf.groupby(["year", "month"]).size()
    ordered = sample.assign(source_segment=sample.index // 5000000,
                            year=sf["year"], month=sf["month"])
    out["source_order_years"] = ordered.groupby(["source_segment", "year"]).size()
    out["source_order_fares"] = ordered.groupby("source_segment")["fare_amount"].agg(
        ["count", "mean", "median", "min", "max"])
    for c in COORDS:
        out["train_precision_" + c] = (
            (sample[c] - sample[c].round(4)).abs().describe())
        out["test_precision_" + c] = (
            (test[c] - test[c].round(4)).abs().describe())

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=714)
    for name, population in rules.items():
        out[name + "_rows"] = population.shape[0]
        out[name + "_fares"] = population["fare_amount"].describe(
            percentiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        sub = population.sample(n=15000, random_state=715)
        for precision, suffix in [(None, "raw"), (4, "rounded4")]:
            a = features(sub, precision).assign(origin=0)
            b = features(test, precision).assign(origin=1)
            combined = a.skb.concat([b], axis=0).reset_index(drop=True)
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                HistGradientBoostingClassifier(
                    max_iter=120, max_leaf_nodes=15,
                    learning_rate=0.08, l2_regularization=10,
                    min_samples_leaf=80, random_state=716))
            out[name + "_" + suffix + "_auc"] = skrub.as_data_op(model).skb.apply_func(
                cross_val_score,
                combined.drop(columns="origin"),
                combined["origin"],
                cv=cv, scoring="roc_auc", n_jobs=1)
    return out
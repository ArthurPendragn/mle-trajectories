import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
SOURCE_COUNT = 55423856
SAMPLE_SIZE = 250000
SEED = 73191

def temporal(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    return rows.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        date=dt.dt.strftime("%Y-%m-%d"),
        hour_group=dt.dt.strftime("%Y-%m-%d %H"),
        stratum=dt.dt.month * 1000 + dt.dt.dayofweek * 100 + dt.dt.hour,
    )

def features(rows):
    cols = ["pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "passenger_count",
            "year", "month", "day", "weekday", "hour"]
    return rows[cols].clip(lower=-1000, upper=3000).fillna(-999).astype("float32")

def build():
    cols = ["pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "passenger_count"]
    dtypes = {c: "float32" for c in cols if c != "pickup_datetime"}
    train = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=cols + ["fare_amount"],
        dtype=dict(dtypes, fare_amount="float32")
    )
    sample = train.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=cols, dtype=dtypes
    )
    sample = temporal(sample)
    test = temporal(test)
    dt_test = test["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    near_hours = test[["hour_group"]]
    for shift in [-6, -3, -1, 1, 3, 6]:
        shifted = test[["hour_group"]].assign(
            hour_group=(dt_test + pd.Timedelta(hours=shift)).dt.strftime("%Y-%m-%d %H")
        )
        near_hours = near_hours.skb.concat([shifted], axis=0)

    fare = sample["fare_amount"]
    rules = {
        "all": sample["year"].notna(),
        "supported_years": sample["year"].isin(test["year"]),
        "supported_dates": sample["date"].isin(test["date"]),
        "supported_month_weekday_hour": sample["stratum"].isin(test["stratum"]),
        "supported_date_hours": sample["hour_group"].isin(test["hour_group"]),
        "supported_date_hour_neighborhood": sample["hour_group"].isin(near_hours["hour_group"]),
        "fare_ge_0": fare >= 0,
        "fare_ge_2_5": fare >= 2.5,
        "fare_ge_5": fare >= 5,
        "fare_le_100": fare <= 100,
        "fare_le_200": fare <= 200,
        "fare_0_100": fare.between(0, 100),
        "fare_2_5_100": fare.between(2.5, 100),
        "fare_2_5_200": fare.between(2.5, 200),
    }
    valid = (
        sample["pickup_longitude"].between(-75, -72) &
        sample["dropoff_longitude"].between(-75, -72) &
        sample["pickup_latitude"].between(40, 42) &
        sample["dropoff_latitude"].between(40, 42) &
        sample["passenger_count"].between(1, 6)
    ).fillna(False)
    rules["observable_valid"] = valid
    rules["valid_ge_2_5"] = valid & (fare >= 2.5)
    rules["valid_le_100"] = valid & (fare <= 100)
    rules["valid_2_5_100"] = valid & fare.between(2.5, 100)

    ordinary = StratifiedKFold(n_splits=3, shuffle=True, random_state=194)
    grouped = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=194)
    classifier = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, learning_rate=0.1,
        min_samples_leaf=40, l2_regularization=2,
        early_stopping=False, random_state=194
    )
    result = {}
    for name, mask in rules.items():
        selected = sample[mask.fillna(False)]
        result[name + "_count"] = selected.shape[0]
        result[name + "_estimated_source_count"] = selected.shape[0] * (SOURCE_COUNT / SAMPLE_SIZE)
        result[name + "_fare_bands"] = fare[mask.fillna(False)].skb.apply_func(
            pd.cut, bins=[-np.inf, 0, 2.5, 5, 10, 20, 50, 100, 200, np.inf]
        ).value_counts(sort=False)

        left = features(selected).assign(domain=0)
        right = features(test).assign(domain=1)
        combined = left.skb.concat([right], axis=0).reset_index(drop=True)
        X = combined.drop(columns="domain")
        y = combined["domain"]
        groups = selected[["hour_group"]].skb.concat(
            [test[["hour_group"]]], axis=0
        ).reset_index(drop=True)["hour_group"]
        result[name + "_ordinary_auc"] = skrub.as_data_op(classifier).skb.apply_func(
            cross_val_score, X, y, cv=ordinary, scoring="roc_auc", n_jobs=1
        )
        result[name + "_timestamp_grouped_auc"] = skrub.as_data_op(classifier).skb.apply_func(
            cross_val_score, X, y, cv=grouped, groups=groups,
            scoring="roc_auc", n_jobs=1
        )

    result["documentation"] = skrub.as_data_op(
        "Read full source, then pandas sample without replacement n=250000, seed=73191. "
        "No target exclusions except explicitly named comparison rules. Estimated counts "
        "multiply sample counts by 55423856/250000. Temporal membership rules use test "
        "year, exact calendar date, month-weekday-hour tuple, or exact calendar-date-hour. "
        "Neighborhood is union of observed date-hours shifted by 0,-6,-3,-1,1,3,6 hours "
        "(not every hour in a continuous window). Fare bands are right-closed. "
        "Observable validity: both longitudes in [-75,-72], both latitudes in [40,42], "
        "passengers in [1,6], missing fails. Adversarial features: four coordinates, "
        "passengers, year/month/day/weekday/hour; numeric clipping [-1000,3000], "
        "missing=-999, float32. No keys, seconds, minutes, or frequency features. "
        "Same HGB classifier and splitter seeds for every population; three ordinary "
        "stratified folds and three stratified grouped folds with date-hour groups shared "
        "across domains. Grouping holds entire normalized timestamp hours out, testing "
        "generalization beyond repeated timestamps. Selection on test-supported dates "
        "can itself inflate apparent matching: weigh AUC reduction against retained rows. "
        "Prefer all rows if its grouped AUC is within fold uncertainty of the minimum; "
        "do not interpret low ordinary AUC alone as proof of genuine selection. "
        "Temporal filters excluding most rows need predictive RMSE evidence before locking."
    )
    return result
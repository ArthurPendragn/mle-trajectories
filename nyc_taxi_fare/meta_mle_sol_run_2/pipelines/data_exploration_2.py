import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
INPUTS = ["pickup_datetime"] + COORDS + ["passenger_count"]


def features(rows, normalized=False, precision=False):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    x = rows[COORDS + ["passenger_count"]]
    if normalized:
        for c in COORDS:
            x = x.assign(**{c: rows[c].astype("float32").astype("float64").round(4)})
    x = x.assign(year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
                 weekday=dt.dt.dayofweek, hour=dt.dt.hour, minute=dt.dt.minute)
    if not normalized:
        x = x.assign(second=dt.dt.second)
    if precision:
        for c in COORDS:
            x = x.assign(**{
                c + "_round6_error": (rows[c] - rows[c].round(6)).abs(),
                c + "_float32_error": (rows[c] - rows[c].astype("float32").astype("float64")).abs()
            })
    dx = (x["dropoff_longitude"] - x["pickup_longitude"]) * 84.3
    dy = (x["dropoff_latitude"] - x["pickup_latitude"]) * 111.2
    return x.assign(dx=dx, dy=dy, distance=(dx ** 2 + dy ** 2) ** 0.5,
                    manhattan=dx.abs() + dy.abs())


def valid(rows):
    return (
        rows["pickup_longitude"].between(-75, -72)
        & rows["dropoff_longitude"].between(-75, -72)
        & rows["pickup_latitude"].between(40, 42)
        & rows["dropoff_latitude"].between(40, 42)
        & rows["passenger_count"].between(1, 6)
    ).fillna(False)


def adversarial(rows, test, normalized=True, columns=None, precision=False):
    a = rows.sample(n=9914, random_state=921).reset_index(drop=True)
    b = test.reset_index(drop=True)
    xa = features(a, normalized, precision).assign(domain=0)
    xb = features(b, normalized, precision).assign(domain=1)
    together = xa.skb.concat([xb], axis=0).reset_index(drop=True)
    x = together.drop(columns=["domain"])
    if columns is not None:
        x = x[columns]
    groups = a[["pickup_datetime"]].skb.concat(
        [b[["pickup_datetime"]]], axis=0
    ).reset_index(drop=True)["pickup_datetime"]
    model = LGBMClassifier(
        n_estimators=140, num_leaves=15, learning_rate=0.06,
        min_child_samples=60, n_jobs=8, random_state=321, verbosity=-1
    )
    return skrub.as_data_op(model).skb.apply_func(
        cross_val_score, x, together["domain"],
        groups=groups, cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=715),
        scoring="roc_auc", n_jobs=1
    )


def build():
    train = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(pd.read_csv)
    sample = train.sample(n=400000, random_state=1971).reset_index(drop=True)
    pool = sample.iloc[:300000]
    ok = valid(pool)
    fare = pool["fare_amount"]
    rules = {
        "unfiltered": fare.notna(),
        "input_valid": ok,
        "fare_ge0": fare.ge(0),
        "fare_ge2_5": fare.ge(2.5),
        "fare_le100": fare.le(100),
        "fare_le200": fare.le(200),
        "fare_band0_200": fare.between(0, 200),
        "fare_band2_5_100": fare.between(2.5, 100),
        "valid_ge0": ok & fare.ge(0),
        "valid_le200": ok & fare.le(200),
        "valid_band2_5_100": ok & fare.between(2.5, 100)
    }
    output = {
        "protocol": skrub.as_data_op({
            "sampling": "Full-source uniform sample, n=400000, seed=1971; first 300000 fitting, remaining 100000 validation.",
            "input_valid_definition": "Both longitudes in [-75,-72], both latitudes in [40,42], passenger_count in [1,6], all inclusive; missing comparisons false.",
            "prior_rule_limit": "Earlier source code is unavailable; this is an explicitly reproducible replacement, not a claim to recover undocumented thresholds.",
            "normalization": "Coordinates cast float32 then float64 and rounded to four decimals; seconds omitted. Applied equally to both domains.",
            "adversary": "LightGBM 140 trees, 15 leaves, lr .06, minimum leaf 60; balanced 9914 rows/domain; seed 921; three stratified exact-timestamp grouped folds, seed 715.",
            "validation": "Two fixed disjoint input-valid holdouts of up to 50000 rows, shared by all rules. Paired negative RMSE differences compare identical validation rows.",
            "interpretation": "AUC substantially above .55 remains evidence of shift. Normalisation ablations distinguish precision and timestamp effects; overlapping fold variation is not evidence for stronger exclusion."
        }),
        "source_count": train.shape[0],
        "sample_count": sample.shape[0],
        "test_calendar": features(test, False).describe(),
        "sample_calendar": features(sample, False).describe(),
        "test_precision": features(test, False, True).describe(),
        "sample_precision": features(sample, False, True).describe(),
        "test_timestamp_frequencies": test["pickup_datetime"].value_counts().describe(),
        "sample_timestamp_frequencies": sample["pickup_datetime"].value_counts().describe(),
        "auc_raw": adversarial(pool, test, False),
        "auc_raw_with_precision": adversarial(pool, test, False, precision=True)
    }
    calendar = ["year", "month", "day", "weekday", "hour", "minute"]
    spatial = COORDS + ["passenger_count", "dx", "dy", "distance", "manhattan"]
    output["auc_normalized_calendar_only"] = adversarial(pool, test, True, calendar)
    output["auc_normalized_spatial_only"] = adversarial(pool, test, True, spatial)
    output["auc_normalized_no_day_minute"] = adversarial(
        pool, test, True, spatial + ["year", "month", "weekday", "hour"]
    )
    validations = []
    for lo, hi in [(300000, 350000), (350000, 400000)]:
        v = sample.iloc[lo:hi]
        validations.append(v[valid(v)].reset_index(drop=True))

    rmse_results = {}
    for name, mask in rules.items():
        selected = pool[mask.fillna(False)].reset_index(drop=True)
        output[name + "_count"] = selected.shape[0]
        output[name + "_excluded_fares"] = pool[~mask.fillna(False)]["fare_amount"].describe()
        output[name + "_auc_normalized"] = adversarial(selected, test, True)
        scores = []
        for rep, val in enumerate(validations):
            combined = selected.skb.concat([val], axis=0).reset_index(drop=True)
            x = features(combined, True)
            y = combined["fare_amount"]
            n = selected.shape[0]
            total = combined.shape[0]
            tr_idx = skrub.as_data_op(0).skb.apply_func(np.arange, n)
            va_idx = n.skb.apply_func(np.arange, total)
            model = LGBMRegressor(
                n_estimators=350, num_leaves=31, learning_rate=0.06,
                min_child_samples=80, n_jobs=8, verbosity=-1,
                random_state=511
            )
            score = skrub.as_data_op(model).skb.apply_func(
                cross_val_score, x, y, cv=[(tr_idx, va_idx)],
                scoring="neg_root_mean_squared_error", n_jobs=1
            )
            output[name + "_negative_rmse_rep" + str(rep)] = score
            scores.append(score)
        rmse_results[name] = scores

    for name in rules:
        if name != "unfiltered":
            for rep in range(2):
                output[name + "_paired_gain_rep" + str(rep)] = (
                    rmse_results[name][rep] - rmse_results["unfiltered"][rep]
                )
    return output
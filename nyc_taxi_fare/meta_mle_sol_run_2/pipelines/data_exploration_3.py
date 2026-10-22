import numpy as np
import pandas as pd
import skrub
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
SIZES = (100000, 300000, 1000000)


def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(
        pd.to_datetime, utc=True, errors="coerce"
    )
    out = rows[[
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "passenger_count"
    ]].copy()
    dx = (out["dropoff_longitude"] - out["pickup_longitude"]) * 84.3
    dy = (out["dropoff_latitude"] - out["pickup_latitude"]) * 111.2
    out = out.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
        dx=dx,
        dy=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs(),
        rotated_manhattan=(dx * 0.78 + dy * 0.63).abs()
        + (dy * 0.78 - dx * 0.63).abs(),
    )
    for name, lon, lat in [
        ("jfk", -73.7781, 40.6413),
        ("lga", -73.8740, 40.7769),
        ("ewr", -74.1745, 40.6895),
    ]:
        for endpoint in ("pickup", "dropoff"):
            dist = (
                ((out[endpoint + "_longitude"] - lon) * 84.3) ** 2
                + ((out[endpoint + "_latitude"] - lat) * 111.2) ** 2
            ) ** 0.5
            out = out.assign(**{endpoint + "_" + name: dist})
    return out.astype("float32")


def build():
    rows = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv,
        usecols=[
            "fare_amount", "pickup_datetime", "pickup_longitude",
            "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
            "passenger_count",
        ],
    )
    sample = rows.sample(n=1400000, random_state=1971).reset_index(drop=True)
    validation = sample.iloc[-60000:].reset_index(drop=True)
    pool = sample.iloc[:-60000]
    pool = pool[
        ~pool["pickup_datetime"].isin(validation["pickup_datetime"])
    ].reset_index(drop=True)
    combined = pool.iloc[:1000000].skb.concat(
        [validation], axis=0
    ).reset_index(drop=True)
    X = features(combined)
    y = combined["fare_amount"]
    result = {
        "source_count": rows.shape[0],
        "eligible_training_count": pool.shape[0],
        "validation_count": validation.shape[0],
        "protocol": skrub.as_data_op({
            "population": "Unfiltered; no target or input validity exclusions.",
            "sampling": "Uniform full-source pandas sample of 1,400,000 rows, seed 1971.",
            "splitting": (
                "Last 60,000 sampled rows form three disjoint fixed 20,000-row "
                "validation sets. Every exact timestamp present in any validation "
                "set is excluded from the training pool. Nested training prefixes "
                "are prefixes of the randomly ordered sample, not of the source."
            ),
            "model": (
                "XGBoost squared-error, CUDA histogram, 600 trees, depth 8, "
                "learning rate .06, minimum child weight 20, subsample .85, "
                "column subsample .9, lambda 10, eight CPU threads."
            ),
            "timing_caveat": (
                "Reported experiment seconds sum cross_validate fit_time and "
                "score_time across three sequential fits. Source decoding and shared "
                "DataOp feature construction are outside these timers; add graph "
                "evaluation overhead when budgeting subsequent experiments. "
                "No unmeasured larger size is fitted."
            ),
            "budget_rule": (
                "Extrapolate three-fold time versus training size from these results. "
                "Use at most remaining_seconds/12 per subsequent experiment, with "
                "headroom for source reads, feature construction and larger models. "
                "If the last paired gain is positive beyond its standard error, "
                "do not interpret a smaller gain than between-fold spread as saturation. "
                "A fixed-holdout locked pool needs the chosen training size plus "
                "60,000 validation rows; ordinary three-fold CV needs about "
                "1.5 times the chosen per-fold training size."
            ),
        }),
    }
    previous = None
    for n in SIZES:
        folds = [
            (
                np.arange(n),
                np.arange(1000000 + k * 20000, 1000000 + (k + 1) * 20000),
            )
            for k in range(3)
        ]
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=600,
            max_depth=8,
            learning_rate=0.06,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=10,
            n_jobs=8,
            random_state=1971,
        )
        measured = skrub.as_data_op(model).skb.apply_func(
            cross_validate,
            X,
            y,
            cv=folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        rmse = -measured["test_score"]
        label = "n_" + str(n)
        result[label + "_rmse"] = rmse
        result[label + "_fit_seconds"] = measured["fit_time"]
        result[label + "_score_seconds"] = measured["score_time"]
        result[label + "_experiment_seconds"] = (
            measured["fit_time"] + measured["score_time"]
        ).skb.apply_func(np.sum)
        if previous is not None:
            gain = previous - rmse
            result[label + "_paired_gain"] = gain
            result[label + "_mean_gain"] = gain.skb.apply_func(np.mean)
            result[label + "_gain_se"] = gain.skb.apply_func(
                np.std, ddof=1
            ) / np.sqrt(3)
        previous = rmse
    return result
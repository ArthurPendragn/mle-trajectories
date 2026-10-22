import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor

TRAIN = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
COLS = [
    "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]


def build():
    rows = skrub.as_data_op(TRAIN).skb.apply_func(
        pd.read_csv, usecols=COLS
    ).sample(n=1060000, random_state=714).reset_index(drop=True)
    y = rows["fare_amount"]
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    X = rows.drop(columns=["fare_amount", "pickup_datetime"])
    dx = (X["dropoff_longitude"] - X["pickup_longitude"]) * 84.3
    dy = (X["dropoff_latitude"] - X["pickup_latitude"]) * 111.2
    X = X.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
        second=dt.dt.second,
        delta_x=dx,
        delta_y=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs(),
        rotated_manhattan=(dx * 0.788 + dy * 0.616).abs()
        + (dy * 0.788 - dx * 0.616).abs()
    ).astype("float32").skb.set_name("geography_and_calendar")
    train_indices = skrub.as_data_op(1000000).skb.apply_func(np.arange)
    folds = []
    for j in range(3):
        test_indices = skrub.as_data_op(1000000 + 20000 * j).skb.apply_func(
            np.arange, 1000000 + 20000 * (j + 1)
        )
        folds.append((train_indices, test_indices))
    model = XGBRegressor(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.06,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_jobs=16,
        random_state=714
    )
    curve = skrub.as_data_op(model).skb.apply_func(
        learning_curve, X, y,
        train_sizes=[100000, 300000, 1000000],
        cv=folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        shuffle=False,
        return_times=True,
        error_score="raise"
    )
    rmse = -curve[2]
    gains = rmse[:-1] - rmse[1:]
    total_fit_score_seconds = (curve[3] + curve[4]).sum(axis=1)
    return {
        "training_sizes": curve[0],
        "validation_rmse_by_size_and_fold": rmse,
        "mean_validation_rmse": rmse.mean(axis=1),
        "paired_rmse_improvements": gains,
        "mean_paired_improvements": gains.mean(axis=1),
        "paired_improvement_standard_errors": gains.std(axis=1, ddof=1) / np.sqrt(3),
        "fit_seconds_by_size_and_fold": curve[3],
        "score_seconds_by_size_and_fold": curve[4],
        "all_fold_fit_and_score_seconds": total_fit_score_seconds,
        "design": skrub.as_data_op({
            "sampling": "Full-source pandas sample(n=1060000, random_state=714), reset index; no fare or input filtering.",
            "training": "First 1000000 sampled rows; nested prefixes 100000, 300000, 1000000.",
            "validation": "Three disjoint 20000-row slices at offsets 1000000, 1020000, 1040000; all sizes use identical validation slices and identical nested training pools.",
            "model": "XGBoost squared error, CUDA hist, 700 trees, depth 8, learning rate .06, min_child_weight 20, lambda 10.",
            "timing": "Reported library times sum fitting and scoring over all three validation splits. Common source reading and graph feature construction are additional costs recorded by the exploration wall timer, not included in these arrays.",
            "selection": "Use paired gains and measured throughput to select the largest affordable pool; do not interpret between-validation-slice spread as uncertainty of paired gains. A positive final paired gain supports extrapolation beyond one million, not a saturation claim.",
            "throughput_constraint": "Reserve at least twelve full experiments in remaining wall time; project larger sizes from observed incremental timing rather than running an unbounded size in this exploration."
        })
    }
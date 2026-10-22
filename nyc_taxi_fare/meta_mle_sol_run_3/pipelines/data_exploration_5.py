import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COLS = ["pickup_datetime", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "passenger_count", "fare_amount"]


class RestrictionEvidence(TransformerMixin, BaseEstimator):
    def __init__(self, training_rows=300000):
        self.training_rows = training_rows

    def fit(self, X, y=None):
        frame = X.reset_index(drop=True)
        features = [c for c in frame.columns
                    if c not in ["fare_amount", "date_support", "hour_support"]]
        pool = frame.iloc[:-60000]
        validation = frame.iloc[-60000:]
        records = []
        predictions = {}
        for name, mask in [
            ("all", np.ones(len(pool), dtype=bool)),
            ("date_supported", pool["date_support"].to_numpy() > 0),
            ("date_hour_supported", pool["hour_support"].to_numpy() > 0),
        ]:
            train = pool.loc[mask].iloc[:self.training_rows]
            model = XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.06,
                min_child_weight=20, reg_lambda=10, objective="reg:squarederror",
                tree_method="hist", device="cuda", n_jobs=16, random_state=714)
            model.fit(train[features].astype("float32"), train["fare_amount"])
            predictions[name] = model.predict(validation[features].astype("float32"))
            for fold in range(3):
                start, end = fold * 20000, (fold + 1) * 20000
                subset = validation.iloc[start:end]
                supported = subset["hour_support"].to_numpy() > 0
                target = subset["fare_amount"].to_numpy()[supported]
                pred = predictions[name][start:end][supported]
                baseline = predictions["all"][start:end][supported]
                if len(target):
                    rmse = float(np.sqrt(mean_squared_error(target, pred)))
                    base = float(np.sqrt(mean_squared_error(target, baseline)))
                else:
                    rmse = base = np.nan
                records.append(dict(
                    rule=name, fold=fold, training_rows=len(train),
                    eligible_training_rows=int(np.sum(mask)),
                    common_validation_rows=len(target),
                    validation_coverage=float(np.mean(supported)),
                    rmse=rmse, improvement_over_all=base-rmse))
        self.result_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.result_


def temporal(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    return rows.assign(
        date=dt.dt.strftime("%Y-%m-%d"),
        date_hour=dt.dt.strftime("%Y-%m-%d %H"),
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour,
        minute=dt.dt.minute, second=dt.dt.second)


def build():
    raw = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=COLS)
    train = temporal(raw.sample(n=2000000, random_state=714).reset_index(drop=True))
    test = temporal(skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=[c for c in COLS if c != "fare_amount"]))

    date_counts = test.groupby("date").size().rename("test_date_count").reset_index()
    hour_counts = test.groupby("date_hour").size().rename("test_hour_count").reset_index()
    train_counts = train.groupby("date_hour").size().rename("train_hour_count").reset_index()
    tagged = train.merge(date_counts, on="date", how="left").merge(
        hour_counts, on="date_hour", how="left").merge(
        train_counts, on="date_hour", how="left")
    tagged = tagged.assign(
        date_support=tagged["test_date_count"].fillna(0),
        hour_support=tagged["test_hour_count"].fillna(0),
        matching_weight=tagged["test_hour_count"].fillna(0) / tagged["train_hour_count"])

    feature_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude",
                    "dropoff_latitude", "passenger_count", "year", "month",
                    "day", "weekday", "hour", "minute", "second"]
    fare = tagged["fare_amount"]
    rules = {
        "all": tagged,
        "date_supported": tagged[tagged["date_support"] > 0],
        "date_hour_supported": tagged[tagged["hour_support"] > 0],
        "positive": tagged[fare > 0],
        "at_least_2_5": tagged[fare >= 2.5],
        "at_least_5": tagged[fare >= 5],
        "at_most_100": tagged[fare <= 100],
        "at_most_200": tagged[fare <= 200],
        "band_2_5_100": tagged[fare.between(2.5, 100)],
        "band_0_200": tagged[fare.between(0, 200)],
    }
    result = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=714)
    classifier = LGBMClassifier(
        n_estimators=180, num_leaves=15, learning_rate=0.05,
        min_child_samples=50, n_jobs=16, verbosity=-1, random_state=714)
    for name, population in rules.items():
        candidate = population.sample(n=9914, random_state=714)
        a = candidate[feature_cols].assign(source=0)
        b = test[feature_cols].assign(source=1)
        combined = a.skb.concat([b], axis=0).reset_index(drop=True)
        result[name + "_auc"] = skrub.as_data_op(classifier).skb.apply_func(
            cross_val_score, combined[feature_cols], combined["source"],
            scoring="roc_auc", cv=cv, n_jobs=1)
        result[name + "_sample_retained"] = population.shape[0]

    weights = tagged["matching_weight"]
    probabilities = weights / weights.sum()
    matched_indices = tagged.shape[0].skb.apply_func(
        np.random.RandomState(714).choice,
        size=9914, replace=False, p=probabilities)
    matched = tagged.iloc[matched_indices]
    combined = matched[feature_cols].assign(source=0).skb.concat(
        [test[feature_cols].assign(source=1)], axis=0).reset_index(drop=True)
    result["date_hour_matched_auc"] = skrub.as_data_op(classifier).skb.apply_func(
        cross_val_score, combined[feature_cols], combined["source"],
        scoring="roc_auc", cv=cv, n_jobs=1)

    for name, rows in [("train", train), ("test", test)]:
        multiplicity = rows.groupby("pickup_datetime").size()
        result[name + "_timestamp_multiplicity"] = multiplicity.describe(
            percentiles=[0.5, 0.9, 0.99])
        result[name + "_duplicate_row_fraction"] = rows["pickup_datetime"].duplicated(
            keep=False).mean()
        result[name + "_joint_clock_counts"] = rows.groupby(
            ["hour", "minute", "second"]).size().sort_values(ascending=False).head(30)
        result[name + "_date_hour_counts"] = rows.groupby(
            "date_hour").size().sort_values(ascending=False).head(30)

    evidence = tagged[feature_cols + ["fare_amount", "date_support", "hour_support"]]
    dx = (evidence["dropoff_longitude"] - evidence["pickup_longitude"]) * 84.3
    dy = (evidence["dropoff_latitude"] - evidence["pickup_latitude"]) * 111.0
    evidence = evidence.assign(dx=dx, dy=dy, distance=(dx**2 + dy**2)**0.5,
                               manhattan=dx.abs()+dy.abs())
    result["common_validation_fare_comparison"] = evidence.skb.apply(
        RestrictionEvidence(), no_wrap=True)
    result["design"] = skrub.as_data_op({
        "sampling": "Full train source read; uniform 2,000,000-row sample, seed 714, reset index. All retained counts refer to this sample, not the full source.",
        "adversary": "Same 180-tree LightGBM and three stratified shuffled folds seed 714. Each rule samples 9914 rows seed 714; identifiers and fare excluded.",
        "support_rules": "Exact UTC calendar date or exact UTC date-hour appears in test. No fare restriction in either support rule.",
        "matched_rule": "NumPy RandomState(714).choice without replacement, weighted by test cell count / sampled training cell count in exact date-hour cells, normalized to sum one; zero for absent cells. This avoids pandas' conservative weighted-sampling guard.",
        "fare_evidence": "First 1,940,000 sample rows supply training; final 60,000 supply three disjoint common validation slices. Equal maximum training size 300,000 per rule; CUDA squared-error boosting.",
        "interpretation": "Support and matching are diagnostic, not automatic recommendations. Multiplicity measured on a sample understates full-source repetition. No normalization is justified merely by separability.",
        "provisional_validation": "Keep unrestricted training pool and raw fares. Report ordinary RMSE plus test date-hour-frequency weighted validation RMSE, using test-count / validation-count cell weights and reporting covered test mass. Exact-cell weighting does not establish absence of within-cell selection.",
        "scope_limit": "Full-source support counts are not computed; no claim of full-source retained counts is made."
    })
    return result
import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COORDS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
INPUTS = COORDS + ["passenger_count", "pickup_datetime"]


def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    dx = (rows["dropoff_longitude"] - rows["pickup_longitude"]) * 84.3
    dy = (rows["dropoff_latitude"] - rows["pickup_latitude"]) * 111.2
    return rows[COORDS + ["passenger_count"]].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour,
        minute=dt.dt.minute, second=dt.dt.second,
        dx=dx, dy=dy, distance=(dx * dx + dy * dy) ** 0.5,
        manhattan=dx.abs() + dy.abs(),
        mid_lon=(rows["pickup_longitude"] + rows["dropoff_longitude"]) / 2,
        mid_lat=(rows["pickup_latitude"] + rows["dropoff_latitude"]) / 2,
    ).replace([np.inf, -np.inf], np.nan)


def build():
    source = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    sampled = source.sample(n=1300000, random_state=714).reset_index(drop=True)
    rows = sampled.iloc[:300000].reset_index(drop=True)
    validation = sampled.iloc[300000:].reset_index(drop=True)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=INPUTS
    )
    f = features(rows)
    tf = features(test)
    y = rows["fare_amount"]
    learner = XGBRegressor(
        objective="reg:squarederror", device="cuda", tree_method="hist",
        n_estimators=450, max_depth=8, learning_rate=0.06,
        min_child_weight=20, reg_lambda=10, random_state=714, n_jobs=16
    )
    oof = skrub.as_data_op(learner).skb.apply_func(
        cross_val_predict, f, y,
        cv=KFold(n_splits=3, shuffle=True, random_state=714),
        n_jobs=1, method="predict"
    )
    residual = y - oof
    distance = f["distance"]
    ratio = y / distance.clip(lower=0.25)
    near_zero = (distance < 0.25).fillna(False)
    unknown_distance = distance.isna()
    special = near_zero | unknown_distance
    date = rows["pickup_datetime"].str.slice(0, 10)
    hour = rows["pickup_datetime"].str.slice(0, 13)
    rules = {
        "all": y.notna(),
        "fare_positive": (y > 0).fillna(False),
        "fare_at_least_2_5": (y >= 2.5).fillna(False),
        "fare_at_least_5": (y >= 5).fillna(False),
        "fare_at_most_100": (y <= 100).fillna(False),
        "fare_at_most_200": (y <= 200).fillna(False),
        "fare_band_2_5_100": y.between(2.5, 100).fillna(False),
        "test_date_supported": date.isin(test["pickup_datetime"].str.slice(0, 10)),
        "test_hour_supported": hour.isin(test["pickup_datetime"].str.slice(0, 13)),
    }
    for cutoff in [2, 4, 8]:
        rules["ratio_at_least_" + str(cutoff)] = special | (ratio >= cutoff).fillna(False)
        rules["ratio_at_most_" + str(cutoff)] = special | (ratio <= cutoff).fillna(False)
    rules["ratio_band_2_8"] = special | ratio.between(2, 8).fillna(False)
    rules["ratio_band_2_4"] = special | ratio.between(2, 4).fillna(False)
    for cutoff in [3, 5, 10]:
        rules["residual_at_least_minus_" + str(cutoff)] = (residual >= -cutoff).fillna(False)
        rules["residual_at_most_" + str(cutoff)] = (residual <= cutoff).fillna(False)
        rules["residual_band_" + str(cutoff)] = residual.between(-cutoff, cutoff).fillna(False)

    outputs = {}
    adversary = LGBMClassifier(
        n_estimators=180, num_leaves=15, learning_rate=0.05,
        min_child_samples=50, random_state=714, n_jobs=16, verbosity=-1
    )
    for name, mask in rules.items():
        candidate = f[mask]
        selected = candidate.sample(n=9914, random_state=714)
        a = selected.assign(origin=0).skb.concat(
            [tf.assign(origin=1)], axis=0
        ).reset_index(drop=True)
        outputs[name + "_retained"] = mask.sum()
        outputs[name + "_auc"] = skrub.as_data_op(adversary).skb.apply_func(
            cross_val_score, a.drop(columns=["origin"]), a["origin"],
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=714),
            n_jobs=1
        )
    outputs["near_zero_count"] = near_zero.sum()
    outputs["missing_distance_count"] = unknown_distance.sum()
    outputs["ratio_quantiles"] = ratio[~special].quantile([0, .01, .1, .5, .9, .99, 1])
    outputs["residual_quantiles"] = residual.quantile([0, .01, .1, .5, .9, .99, 1])

    tc = test.assign(cell=test["pickup_datetime"].str.slice(0, 13))
    tc = tc.groupby("cell").size().rename("test_count").reset_index()
    vc = validation.assign(cell=validation["pickup_datetime"].str.slice(0, 13))
    vc = vc.groupby("cell").size().rename("validation_count").reset_index()
    cells = tc.merge(vc, on="cell", how="left").fillna({"validation_count": 0})
    covered = cells[cells["validation_count"] > 0]
    covered = covered.assign(
        row_weight=covered["test_count"] / covered["validation_count"],
        squared_weight_sum=covered["test_count"] ** 2 / covered["validation_count"]
    )
    mass = covered["test_count"].sum()
    outputs["covered_test_mass"] = mass / 9914
    outputs["weighted_validation_rows"] = covered["validation_count"].sum()
    outputs["effective_sample_size"] = mass ** 2 / covered["squared_weight_sum"].sum()
    outputs["maximum_normalized_row_weight"] = covered["row_weight"].max() / mass
    outputs["weight_quantiles_by_cell"] = covered["row_weight"].quantile([0, .5, .9, .99, 1])
    outputs["largest_test_cells"] = cells.sort_values("test_count", ascending=False).head(30)
    outputs["design"] = skrub.as_data_op({
        "sampling": "Read full source, sample 1,300,000 rows seed 714, reset index. First 300,000 for population study; last 1,000,000 for independent coverage study. No fare exclusions.",
        "ratio": "Fare divided by planar distance km: sqrt((84.3*dlongitude)^2+(111.2*dlatitude)^2). All ratio rules retain distances below 0.25 km and missing distances; these counts are reported separately.",
        "residual": "Raw fare minus three-fold out-of-fold CUDA XGBoost prediction, with shuffled KFold seed 714. Residuals are selection diagnostics, never features.",
        "adversary": "Every population samples 9,914 rows seed 714, compared with all test rows using the identical 180-tree LightGBM, three shuffled stratified folds seed 714. Features are coordinates, passenger count, UTC calendar/clock and explicit planar geometry; no target, residual, ratio or identifier.",
        "restrictions": "No restrictive population is recommended by this study: no common-validation predictive evidence is supplied. A lower adversarial AUC alone cannot justify excluding rows on fare, ratio or residual.",
        "validation": "Keep unrestricted training and raw fares. Use deterministic random validation rows, report ordinary RMSE and date-hour-weighted RMSE. Each covered row weight is test cell count divided by validation cell count; normalize by covered test count. Report coverage and ESS, and do not treat uncovered cells as represented.",
        "limitations": "Counts are within the 300,000-row study sample, not full-source counts. Weighting does not resolve within-cell selection or prove test representativeness. Small AUC differences within fold variability are not improvements."
    })
    return outputs
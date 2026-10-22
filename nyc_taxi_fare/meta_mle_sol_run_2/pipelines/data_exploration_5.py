import pandas as pd
import numpy as np
import skrub

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
INPUTS = [
    "pickup_datetime", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]

def add_strata(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True, errors="coerce")
    dx = (rows["dropoff_longitude"] - rows["pickup_longitude"]) * 84.4
    dy = (rows["dropoff_latitude"] - rows["pickup_latitude"]) * 111.2
    distance = (dx ** 2 + dy ** 2) ** 0.5
    valid = (
        rows["pickup_longitude"].between(-75, -72)
        & rows["dropoff_longitude"].between(-75, -72)
        & rows["pickup_latitude"].between(40, 42)
        & rows["dropoff_latitude"].between(40, 42)
    ).fillna(False)
    result = rows.assign(
        distance_km=distance,
        distance_band=distance.skb.apply_func(
            pd.cut,
            bins=[-np.inf, 0.1, 1, 3, 5, 10, 20, 40, np.inf],
            labels=["near_zero", "0.1_1", "1_3", "3_5", "5_10", "10_20", "20_40", "over_40"]
        ).astype("string").fillna("missing"),
        coordinate_valid=valid,
        year=dt.dt.year,
        hour=dt.dt.hour,
    )
    for prefix, lon, lat in [
        ("jfk", -73.7822, 40.6442),
        ("lga", -73.8740, 40.7769),
        ("ewr", -74.1745, 40.6895),
    ]:
        pickup = (
            ((rows["pickup_longitude"] - lon) * 84.4) ** 2
            + ((rows["pickup_latitude"] - lat) * 111.2) ** 2
        ) ** 0.5
        dropoff = (
            ((rows["dropoff_longitude"] - lon) * 84.4) ** 2
            + ((rows["dropoff_latitude"] - lat) * 111.2) ** 2
        ) ** 0.5
        result = result.assign(**{
            prefix: ((pickup < 3) | (dropoff < 3)).fillna(False)
        })
    for endpoint in ["pickup", "dropoff"]:
        lon = rows[endpoint + "_longitude"]
        lat = rows[endpoint + "_latitude"]
        region = (
            lon.skb.apply_func(
                pd.cut,
                bins=[-np.inf, -74.04, -73.96, -73.88, np.inf],
                labels=["west", "central", "east", "outer_east"],
            ).astype("string").fillna("missing")
            + "_"
            + lat.skb.apply_func(
                pd.cut,
                bins=[-np.inf, 40.70, 40.78, np.inf],
                labels=["south", "mid", "north"],
            ).astype("string").fillna("missing")
        )
        result = result.assign(**{endpoint + "_region": region})
    return result.assign(
        corridor=result["pickup_region"] + "__" + result["dropoff_region"]
    )

def build():
    source = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv, usecols=INPUTS + ["fare_amount"]
    )
    sampled = source.sample(n=1400000, random_state=1971)
    validation = sampled.iloc[-60000:].reset_index(drop=True)
    validation = validation.assign(
        fold=validation.index // 20000,
        row=validation.index + 8300000,
    )
    validation = add_strata(validation)
    test = skrub.as_data_op(ROOT + "/test.csv").skb.apply_func(
        pd.read_csv, usecols=INPUTS
    )
    test = add_strata(test)
    outputs = {
        "limitations": skrub.as_data_op({
            "status": "Requested prediction-error audit cannot be computed from permitted inputs.",
            "reason": "Retained OOF predictions are previous-step outputs, which the source-only plan contract forbids reading. They are not supplied as an in-memory DataOp.",
            "no_refitting": True,
            "available": "Exact leading-candidate validation inputs and labels, per-fold strata and test coverage.",
            "interpretation": "Fare summaries and extreme fares are not residuals. No error attribution or modeling recommendation is justified from these summaries alone.",
            "required_followup": "The harness must join its retained OOF table on row to validation_rows and compute prediction minus fare_amount, squared error, and per-fold stratum summaries outside this source-only plan.",
            "definitions": "Airport involvement means either endpoint within 3 km of the named airport center. Coordinate validity tests both endpoints within longitude [-75,-72] and latitude [40,42]. Corridors are coarse coordinate-grid pairs, not administrative boroughs.",
        }),
        "validation_rows": validation,
        "largest_fares_not_largest_errors": validation.sort_values(
            "fare_amount", ascending=False
        ).head(40),
        "smallest_fares_not_largest_errors": validation.sort_values(
            "fare_amount", ascending=True
        ).head(20),
    }
    for column in [
        "distance_band", "jfk", "lga", "ewr", "corridor",
        "year", "hour", "coordinate_valid"
    ]:
        stats = validation.groupby(["fold", column], dropna=False)["fare_amount"].agg(
            ["count", "mean", "std", "min", "max"]
        ).reset_index()
        coverage = test.groupby(column, dropna=False).size().reset_index(name="test_count")
        coverage = coverage.assign(test_fraction=coverage["test_count"] / test.shape[0])
        stats = stats.assign(validation_fraction=stats["count"] / 20000)
        outputs[column + "_coverage_and_labels"] = stats.merge(
            coverage, on=column, how="outer"
        ).sort_values(["fold", "count"], ascending=[True, False])
    return outputs
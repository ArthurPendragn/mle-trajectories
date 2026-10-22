import numpy as np
import pandas as pd
import skrub

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
SEED = 1971
N_TRAIN = 8_300_000
N_VALID = 60_000
FEATURE_COLUMNS = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]


class FixedTimestampHoldouts:
    def get_n_splits(self, X=None, y=None, groups=None):
        return 3

    def split(self, X, y=None, groups=None):
        if len(X) != N_TRAIN + N_VALID:
            raise ValueError("Unexpected locked population size")
        training = np.arange(N_TRAIN, dtype=np.int64)
        for fold in range(3):
            start = N_TRAIN + fold * 20_000
            yield training, np.arange(start, start + 20_000, dtype=np.int64)


def build():
    source = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv,
        usecols=FEATURE_COLUMNS + ["fare_amount"],
    )

    initial_sample = source.sample(n=1_400_000, random_state=SEED)
    validation = initial_sample.iloc[-N_VALID:]
    validation_timestamps = validation["pickup_datetime"]

    eligible = source[
        ~source["pickup_datetime"].isin(validation_timestamps)
    ]
    training = eligible.sample(n=N_TRAIN, random_state=SEED)
    rows = training.skb.concat([validation], axis=0).reset_index(drop=True)

    X = rows[FEATURE_COLUMNS].skb.mark_as_X(
        cv=FixedTimestampHoldouts(), split_kwargs={}
    )
    y = rows["fare_amount"].skb.mark_as_y()

    comparisons = {
        "unfiltered": [300000, [0.53126, 0.53756, 0.49960]],
        "input_valid": [292477, [0.48007, 0.55583, 0.49055]],
        "fare_ge_zero": [299992, [0.50499, 0.48730, 0.52304]],
        "fare_ge_2_5": [299984, [0.52796, 0.54110, 0.55024]],
        "fare_le_100": [299891, [0.49469, 0.54942, 0.53377]],
        "fare_le_200": [299996, [0.49988, 0.50905, 0.54248]],
        "fare_band_zero_200": [299988, [0.49019, 0.54230, 0.53114]],
        "fare_band_2_5_100": [299875, [0.50280, 0.53866, 0.51587]],
        "valid_ge_zero": [292470, [0.54868, 0.52252, 0.48516]],
        "valid_le_200": [292474, [0.47625, 0.56143, 0.51969]],
        "valid_band_2_5_100": [292369, [0.56252, 0.55509, 0.50633]],
    }
    audit = {
        "training_rows": training.shape[0],
        "validation_rows": validation.shape[0],
        "population_rows": rows.shape[0],
        "timestamp_overlap_rows": training["pickup_datetime"].isin(
            validation_timestamps
        ).sum(),
        "validation_unique_timestamps": validation_timestamps.nunique(),
        "population_comparisons": skrub.as_data_op(comparisons),
        "population_decision": skrub.as_data_op({
            "rule": "No input or target filtering",
            "reason": "Least restrictive rule within adversarial fold noise; restrictive filters lacked consistent paired RMSE benefit.",
            "adversary": "Balanced LightGBM, three exact-timestamp-grouped stratified folds; normalized label-free geographic/calendar inputs.",
        }),
        "learning_curve": skrub.as_data_op({
            "model": "CUDA XGBoost squared-error, 600 trees, depth 8, learning_rate .06, min_child_weight 20, subsample .85, colsample_bytree .9, reg_lambda 10",
            "sizes": [100000, 300000, 1000000],
            "rmse": [
                [4.135986, 3.874339, 4.421140],
                [4.092864, 3.764580, 4.304618],
                [3.854782, 3.668225, 4.184640],
            ],
            "three_fit_feature_inclusive_seconds": [5.310838, 11.428575, 50.569760],
            "source_decode_seconds": 99.999396,
            "paired_gains": [
                [0.043123, 0.109760, 0.116522],
                [0.238081, 0.096355, 0.119978],
            ],
            "paired_gain_means": [0.089801, 0.151471],
            "paired_gain_standard_errors": [0.023421, 0.043839],
            "selected_training_rows": N_TRAIN,
            "estimated_experiment_seconds_with_headroom": 1492.307264,
            "reason": "Curve still improving; extrapolated runtime limit leaves room for twelve experiments. Larger-size memory feasibility is extrapolated, not measured.",
        }),
        "sampling": skrub.as_data_op({
            "validation": "Last 60000 rows of full-source uniform 1400000-row sample, seed 1971",
            "training": "Full-source uniform sample after excluding every validation timestamp, seed 1971",
            "folds": "Common training pool; three disjoint contiguous 20000-row slices of randomly sampled validation rows",
        }),
    }
    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
        "audit": audit,
    }
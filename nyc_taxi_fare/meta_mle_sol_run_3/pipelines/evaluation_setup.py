import numpy as np
import pandas as pd
import skrub

TRAIN = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SEED = 714
N_TRAIN = 25000000
N_TOTAL = 26000000
FEATURE_COLUMNS = [
    "pickup_datetime", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count",
]


class FixedPoolCV:
    def get_n_splits(self, X=None, y=None, groups=None):
        return 3

    def split(self, X, y=None, groups=None):
        if len(X) != N_TOTAL:
            raise ValueError("Unexpected locked population size")
        train = np.arange(N_TRAIN, dtype=np.int64)
        boundaries = [N_TRAIN, 25333333, 25666666, N_TOTAL]
        for i in range(3):
            yield train, np.arange(boundaries[i], boundaries[i + 1], dtype=np.int64)


def cell_weights(cells, test_cells):
    counts = cells.value_counts()
    count_table = counts.rename_axis("cell").reset_index(name="validation_count")
    test_table = test_cells.rename_axis("cell").reset_index(name="test_count")
    weighted = cells.rename("cell").to_frame().merge(
        count_table, on="cell", how="left", sort=False
    ).merge(test_table, on="cell", how="left", sort=False)
    return weighted["test_count"].fillna(0) / weighted["validation_count"]


def build():
    rows = skrub.as_data_op(TRAIN).skb.apply_func(
        pd.read_csv, usecols=FEATURE_COLUMNS + ["key", "fare_amount"]
    )
    rows = rows.sample(n=N_TOTAL, random_state=SEED).reset_index(drop=True)
    X = rows[FEATURE_COLUMNS].skb.mark_as_X(cv=FixedPoolCV(), split_kwargs={})
    y = rows["fare_amount"].skb.mark_as_y()
    keys = rows["key"]

    test = skrub.as_data_op(TEST).skb.apply_func(
        pd.read_csv, usecols=["pickup_datetime"]
    )
    test_cells = test["pickup_datetime"].str.slice(0, 13).value_counts()
    validation = rows.iloc[N_TRAIN:]
    cells = validation["pickup_datetime"].str.slice(0, 13)
    counts = cells.value_counts()
    weights = cell_weights(cells, test_cells)
    covered = test_cells.index.isin(counts.index)
    audit = {
        "validation_test_mass_covered": test_cells[covered].sum() / test_cells.sum(),
        "validation_effective_sample_size": weights.sum() ** 2 / (weights ** 2).sum(),
        "validation_maximum_normalized_weight": weights.max() / weights.sum(),
        "validation_supported_rows": (weights > 0).sum(),
        "validation_weight_quantiles": weights.quantile([0, .5, .9, .99, 1]),
        "population_selection": skrub.as_data_op({
            "rule": "Uniform full-source sample, no input or target exclusions.",
            "unresolved_shift": "Timestamp selection remains distinguishable; exact-hour weighting is diagnostic only.",
            "primary_metric": "Ordinary RMSE, reported with negative sign by sklearn.",
            "weighted_metric_recipe": "For each validation fold, weight squared errors by test-cell count / fold-cell count, then take sqrt of their weighted mean. Report coverage and ESS; do not treat absent cells as represented.",
        }),
        "population_study_300k": skrub.as_data_op({
            "counts_are": "Retained counts within the 300000-row study sample, not full-source counts.",
            "all": [300000, .87431760, .87116574, .88176840],
            "fare_positive": [299983, .87612955, .88029482, .88993280],
            "fare_ge_2_5": [299977, .88111956, .87774595, .87920340],
            "fare_ge_5": [258762, .88192904, .88055756, .89113851],
            "fare_le_100": [299887, .87791102, .87760236, .88243453],
            "fare_le_200": [299990, .87836666, .87501584, .88449612],
            "fare_band_2_5_100": [299864, .87138490, .88132800, .88177270],
            "test_date_supported": [153180, .87285784, .87397543, .87730244],
            "test_hour_supported": [9984, .86912279, .87250598, .86433572],
            "ratio_ge_2": [293219, .87362594, .88998399, .88719500],
            "ratio_le_2": [18464, .98700772, .98567711, .98552142],
            "ratio_ge_4": [140906, .91497749, .91643778, .91932394],
            "ratio_le_4": [170777, .89605325, .90081916, .90706868],
            "ratio_ge_8": [26364, .97768347, .97443543, .97351983],
            "ratio_le_8": [285319, .87976696, .88739968, .88188864],
            "ratio_band_2_8": [278538, .88039046, .87720665, .88595642],
            "ratio_band_2_4": [163996, .90843965, .90772795, .90373361],
            "residual_ge_minus_3": [283944, .87698975, .87855824, .88138826],
            "residual_le_3": [278506, .87354840, .87722918, .88245972],
            "residual_abs_le_3": [262450, .87618759, .88518387, .88710315],
            "residual_ge_minus_5": [292449, .87526615, .88277831, .88409254],
            "residual_le_5": [289380, .87154575, .88013209, .88472580],
            "residual_abs_le_5": [281829, .87138055, .87847491, .87746545],
            "residual_ge_minus_10": [297963, .87130007, .87870760, .88305712],
            "residual_le_10": [296255, .87741573, .88216191, .88130859],
            "residual_abs_le_10": [294218, .87410662, .88570851, .89113063],
            "ratio_definition": "fare / sqrt((84.3*delta_lon)^2+(111.2*delta_lat)^2); retain distance<0.25 and missing distance in every ratio rule.",
            "adversary": "9914 candidate rows seed 714 versus all test rows; identical 180-tree LightGBM, three shuffled stratified folds seed 714; no labels or identifiers.",
        }),
        "learning_curve": skrub.as_data_op({
            "model": "CUDA XGBoost squared error, 700 trees, depth 8, learning rate .06, child weight 20, lambda 10, seed 714; fold-fitted median imputation.",
            "sizes": [300000, 1000000, 3000000, 10000000],
            "mean_rmse": [4.211269, 4.154456, 4.065719, 4.032211],
            "three_split_seconds": [12.223692, 34.326595, 111.889997, 438.602508],
            "paired_gain": [.056812, .088738, .033508],
            "paired_gain_se": [.063337, .052033, .017003],
            "chosen_training_rows": N_TRAIN,
            "projected_experiment_seconds": 180 + 1.35 * 438.602508 * 2.5,
            "justification": "Extrapolated affordable pool; final measured gain positive in every fold, not established saturation. Twelve projected experiments fit remaining budget.",
            "timing_limitations": "Measured curve times exclude shared geographic/timestamp graph construction and source reads. The 180-second reserve is an assumption, not a separately measured feature-building cost; 25-million-row runtime is unmeasured.",
        }),
    }
    boundaries = [N_TRAIN, 25333333, 25666666, N_TOTAL]
    for i in range(3):
        fold = rows.iloc[boundaries[i]:boundaries[i + 1]]
        fc = fold["pickup_datetime"].str.slice(0, 13)
        fw = cell_weights(fc, test_cells)
        supported = test_cells.index.isin(fc.value_counts().index)
        audit["fold_%d_coverage" % i] = test_cells[supported].sum() / test_cells.sum()
        audit["fold_%d_ess" % i] = fw.sum() ** 2 / (fw ** 2).sum()
        audit["fold_%d_max_weight" % i] = fw.max() / fw.sum()
    return {
        "X": X, "y": y, "row_keys": keys,
        "scoring": "neg_root_mean_squared_error", "audit": audit,
    }
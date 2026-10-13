import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import BaseCrossValidator

ROOT = "gs://mle-trajectories-data/ttt-task/input/"
N_TRACKERS = 355
N_TRAIN = 160000
N_TEST = 3000
N_FOLDS = 3
POOL_SIZE = N_TRAIN + N_FOLDS * N_TEST


class DomainHoldoutCV(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        return N_FOLDS

    def split(self, X, y=None, groups=None):
        if len(X) != POOL_SIZE:
            raise ValueError("Unexpected locked population size")
        train = np.arange(N_TRAIN)
        for fold in range(N_FOLDS):
            start = N_TRAIN + fold * N_TEST
            yield train.copy(), np.arange(start, start + N_TEST)


def mean_domain_recall_at_10(estimator, X, y):
    """predict returns compact tracker IDs, with at most ten columns."""
    predicted = np.asarray(estimator.predict(X))
    truth = np.asarray(y)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if predicted.shape[0] != truth.shape[0]:
        raise ValueError("Prediction row alignment mismatch")
    if predicted.shape[1] > 10:
        predicted = predicted[:, :10]
    hits = np.zeros_like(truth, dtype=bool)
    for rank in range(predicted.shape[1]):
        values = predicted[:, rank]
        valid = np.isfinite(values) & (values >= 0) & (values < N_TRACKERS)
        valid &= values == np.floor(values)
        rows = np.flatnonzero(valid)
        hits[rows, values[valid].astype(int)] = True
    positives = truth > 0
    return float(
        ((hits & positives).sum(axis=1) / positives.sum(axis=1)).mean()
    )


def build():
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    )
    edges = edges.drop_duplicates(["domain_id", "tracker_id"])
    counts = edges.groupby("domain_id").size().rename("tracker_count").reset_index()
    eligible = counts[counts["tracker_count"] >= 2][["domain_id"]]
    pool = eligible.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)

    selected = edges.merge(pool, on="domain_id", how="inner")
    wide = selected.assign(present=1).pivot_table(
        index="domain_id",
        columns="tracker_id",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    wide = wide.reindex(columns=list(range(N_TRACKERS)), fill_value=0)
    aligned = pool.merge(wide, left_on="domain_id", right_index=True, how="left")
    y = aligned[list(range(N_TRACKERS))].astype("uint8").skb.mark_as_y()

    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"]
    )
    domains = domains[domains["domain_id"].isin(pool["domain_id"])]
    X = pool.merge(domains, on="domain_id", how="left")
    X = X.skb.mark_as_X(cv=DomainHoldoutCV(), split_kwargs={})

    comparisons = [
        ("all", 18682899, .573085, .003053),
        ("at_least_1", 18682899, .572887, .004462),
        ("at_most_1", 10047394, .646036, .008796),
        ("at_least_2", 8635505, .509694, .003625),
        ("at_most_2", 14369613, .613311, .007176),
        ("at_least_3", 4313286, .560059, .002390),
        ("at_most_3", 16544942, .589803, .010489),
        ("at_least_5", 1158047, .686705, .005335),
        ("at_most_5", 18055440, .585604, .006701),
        ("at_least_10", 90936, .869315, .005835),
        ("at_most_10", 18623021, .578305, .005423),
        ("at_least_20", 5362, .979073, .001884),
        ("at_most_20", 18678283, .568326, .009949),
        ("band_2_3", 6497548, .540302, .005133),
        ("band_2_5", 8008046, .507162, .002307),
        ("band_2_10", 8575627, .507661, .001767),
        ("band_3_5", 3685827, .533930, .005190),
        ("band_1_3", 16544942, .596685, .001067),
        ("has_outgoing_links", 13689331, .684827, .002973),
        ("has_incoming_links", 16347018, .622525, .004842),
    ]
    fixed_audit = {
        "population_comparisons": comparisons,
        "selection": "At least two distinct recorded trackers; seed-42 random pool.",
        "learning_curve": {
            "model": "CUDA MLP, 128 hidden units, 12 epochs, normalized-positive softmax",
            "training_sizes": [10000, 40000, 160000],
            "restricted_recall": [.724838, .747109, .749728],
            "unrestricted_recall": [.717513, .738669, .740450],
            "restricted_fold_sd": [.004857, .003951, .005195],
            "restricted_seconds_all_folds": [2.271069, 8.859656, 34.496739],
            "timing_excludes": "Source ingestion and graph joins",
            "size_reason": "160k retained; further measured gains are below fold variability.",
            "extrapolated_full_population_seconds": 2462,
            "reserved_source_seconds_per_experiment": 600,
        },
        "label_safety": (
            "All 9000 holdout domains are excluded from every training fold. "
            "Source-derived label features must exclude the entire locked pool "
            "or obtain labels through marked y."
        ),
        "limitations": "Target selection and recorded-label completeness remain unverified.",
        "prediction_contract": "Compact tracker_id rankings, at most ten columns.",
    }
    audit = {
        name: skrub.as_data_op(value)
        for name, value in fixed_audit.items()
    }
    audit["eligible_rows"] = eligible.shape[0]
    audit["locked_rows"] = pool.shape[0]

    return {
        "X": X,
        "y": y,
        "scoring": mean_domain_recall_at_10,
        "row_keys": X["domain_id"],
        "audit": audit,
    }
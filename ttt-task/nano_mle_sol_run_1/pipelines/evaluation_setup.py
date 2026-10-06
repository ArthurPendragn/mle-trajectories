import numpy as np
import pandas as pd
import skrub

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
N_TRAIN = 1_200_000
N_VALIDATION = 2_000
N_SPLITS = 3
N_RESERVED = N_VALIDATION * N_SPLITS
N_POOL = N_TRAIN + N_RESERVED
N_TRACKERS = 355


class ReservedDomainSplits:
    """All splits share a training set; validation domains never enter training."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return N_SPLITS

    def split(self, X, y=None, groups=None):
        if len(X) != N_POOL:
            raise ValueError("The locked population must contain exactly 1,206,000 rows.")
        training = np.arange(N_RESERVED, N_POOL, dtype=np.int64)
        for fold in range(N_SPLITS):
            start = fold * N_VALIDATION
            validation = np.arange(start, start + N_VALIDATION, dtype=np.int64)
            yield training.copy(), validation


def recall_at_10(estimator, X, y):
    """predict returns 355 ranking scores or up to ten compact tracker IDs."""
    truth = np.asarray(y)
    prediction = np.asarray(estimator.predict(X))
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if prediction.ndim != 2 or prediction.shape[0] != truth.shape[0]:
        raise ValueError("Predictions must have one row per validation domain.")

    if prediction.shape[1] == N_TRACKERS:
        scores = np.nan_to_num(
            prediction.astype(np.float64), nan=-np.inf,
            posinf=np.inf, neginf=-np.inf
        )
        guesses = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    elif prediction.shape[1] <= 10:
        guesses = prediction
    else:
        raise ValueError("Return either 355 scores or at most ten compact tracker IDs.")

    recalls = np.zeros(len(truth), dtype=np.float64)
    for i in range(len(truth)):
        selected = set()
        for value in guesses[i]:
            if pd.isna(value):
                continue
            try:
                tracker = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if tracker != value or not 0 <= tracker < N_TRACKERS:
                continue
            selected.add(tracker)
            if len(selected) == 10:
                break
        denominator = np.count_nonzero(truth[i])
        if denominator and selected:
            recalls[i] = np.count_nonzero(truth[i, list(selected)]) / denominator
    return float(recalls.mean())


def build_evaluation():
    labels = skrub.as_data_op(BASE + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    )
    labels = labels.drop_duplicates(["domain_id", "tracker_id"])
    counts = (
        labels.groupby("domain_id")["tracker_id"]
        .nunique()
        .rename("known_tracker_count")
        .reset_index()
    )
    eligible = counts[counts["known_tracker_count"] >= 2]
    pool = (
        eligible[["domain_id"]]
        .sort_values("domain_id")
        .sample(n=N_POOL, replace=False, random_state=42)
        .reset_index(drop=True)
    )

    row_keys = pool[["domain_id"]]
    X = pool[["domain_id"]].skb.mark_as_X(
        cv=ReservedDomainSplits(), split_kwargs={}
    )

    pool_labels = labels.merge(pool, on="domain_id", how="inner", sort=False)
    raw_y = (
        pool_labels.assign(present=1)
        .pivot_table(
            index="domain_id",
            columns="tracker_id",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(
            index=pool["domain_id"],
            columns=list(range(N_TRACKERS)),
            fill_value=0,
        )
        .fillna(0)
        .astype("uint8")
        .reset_index(drop=True)
    )
    y = raw_y.skb.mark_as_y()

    targets = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    )
    audit = {
        "eligible_population_shape": eligible.shape,
        "labelled_population_shape": counts.shape,
        "pool_shape": pool.shape,
        "target_label_overlap_shape": targets.merge(
            counts[["domain_id"]], on="domain_id", how="inner"
        ).shape,
        "population_comparison_evidence": skrub.as_data_op({
            "source": "exploration_e21b365199ae",
            "mean_adversarial_auc": {
                "all": 0.581334,
                "at_most_1": 0.657775,
                "at_least_1": 0.581334,
                "at_most_2": 0.625485,
                "at_least_2": 0.501100,
                "at_most_3": 0.604449,
                "at_least_3": 0.562517,
                "at_most_5": 0.583018,
                "at_least_5": 0.700142,
                "at_most_10": 0.578447,
                "at_least_10": 0.887880,
                "band_1_2": 0.625485,
                "band_2_3": 0.538921,
                "band_2_5": 0.515977,
                "band_3_10": 0.556774,
                "band_5_10": 0.692647,
                "classified": 0.990407,
                "unclassified": 0.584808,
                "has_outgoing": 0.693845,
                "no_outgoing": 0.882130,
                "has_incoming": 0.636778,
                "no_incoming": 0.945523,
            },
            "selected_fold_auc": [0.506585, 0.500998, 0.495717],
            "selected_auc_std": 0.005435,
            "interpretation": (
                "At least two trackers has the lowest observed AUC; no less "
                "restrictive rule is within fold noise. Observable similarity "
                "does not establish label completeness or conditional-label equivalence."
            ),
        }),
        "pool_sizing_evidence": skrub.as_data_op({
            "source": "exploration_e21b365199ae",
            "model": "CUDA character-ngram MLP, 256 hidden units, inverse-cardinality BCE",
            "training_sizes": [120000, 400000, 1200000],
            "mean_recall_at_10": [0.770724, 0.784351, 0.791325],
            "fold_std": [0.004780, 0.007046, 0.004877],
            "complete_three_split_seconds": [117.904947, 176.019659, 371.911776],
            "reason": (
                "Use the largest measured size while preserving allowance for "
                "more expensive families. Twelve measured experiments cost about "
                "4463 seconds. The conservative extrapolated limit is about "
                "4.95 million training domains; diminishing measured gains "
                "motivate the smaller pool, not a proven statistical plateau."
            ),
            "auxiliary_label_policy": (
                "Exclude every locked-pool domain from external tracker-label "
                "sources before any supervised join, aggregation, or graph walk. "
                "External domains remain available. Fit preprocessing on training "
                "rows only. All 6000 reserved domains are excluded from every fit."
            ),
            "sampling": (
                "Sort eligible domain IDs, sample without replacement with seed "
                "42, and reserve the first three consecutive 2000-row subsets. "
                "Recomputation of this recorded graph fixes membership and order; "
                "no output files are used as inputs."
            ),
        }),
    }
    return {
        "X": X,
        "y": y,
        "scoring": recall_at_10,
        "row_keys": row_keys,
        "audit": audit,
    }


def build():
    return build_evaluation()
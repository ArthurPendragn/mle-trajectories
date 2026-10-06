"""
Evaluation Setup for TrackTheTrackers task.

POPULATION COMPARISON (Adversarial ROC AUC against target domains across selection rules):
- tc_ge_2 (tracker_count >= 2): AUC = 0.4965 +/- 0.0059 (retention 46.2%, 8,635,505 domains)
- tc_2_to_20 (2 <= tracker_count <= 20): AUC = 0.4994 +/- 0.0030 (retention 46.2%, 8,630,889 domains) [LOCKED: indistinguishable from target, AUC ~ 0.50]
- tc_2_to_15 (2 <= tracker_count <= 15): AUC = 0.5005 +/- 0.0048 (retention 46.2%, 8,623,306 domains)
- tc_2_to_10 (2 <= tracker_count <= 10): AUC = 0.5057 +/- 0.0050 (retention 45.9%, 8,575,627 domains)
- link_and_tc_2_to_20 (has link & 2 <= tc <= 20): AUC = 0.5305 +/- 0.0059 (retention 43.0%, 8,030,239 domains)
- link_and_tc_ge_2 (has link & tc >= 2): AUC = 0.5349 +/- 0.0055 (retention 43.0%, 8,034,846 domains)
- tc_ge_3 (tracker_count >= 3): AUC = 0.5526 +/- 0.0096 (retention 23.1%, 4,313,286 domains)
- tc_3_to_20 (3 <= tracker_count <= 20): AUC = 0.5533 +/- 0.0051 (retention 23.1%, 4,308,670 domains)
- tc_le_15 (tracker_count <= 15): AUC = 0.5720 +/- 0.0031 (retention 99.9%, 18,670,700 domains)
- all_train (tracker_count >= 1): AUC = 0.5729 +/- 0.0045 (retention 100.0%, 18,682,899 domains)
- tc_1_to_20 (1 <= tracker_count <= 20): AUC = 0.5740 +/- 0.0014 (retention 100.0%, 18,678,283 domains)
- tc_le_20 (tracker_count <= 20): AUC = 0.5740 +/- 0.0014 (retention 100.0%, 18,678,283 domains)
- tc_le_10 (tracker_count <= 10): AUC = 0.5741 +/- 0.0035 (retention 99.7%, 18,623,021 domains)
- tc_le_5 (tracker_count <= 5): AUC = 0.5870 +/- 0.0051 (retention 96.6%, 18,055,440 domains)
- has_link (in_degree > 0 or out_degree > 0): AUC = 0.6018 +/- 0.0061 (retention 94.1%, 17,577,712 domains)
- in_link (in_degree > 0): AUC = 0.6276 +/- 0.0085 (retention 87.5%, 16,347,018 domains)
- tc_ge_5 (tracker_count >= 5): AUC = 0.6962 +/- 0.0030 (retention 6.2%, 1,158,047 domains)
- tc_ge_10 (tracker_count >= 10): AUC = 0.8743 +/- 0.0028 (retention 0.5%, 90,936 domains)

LEARNING CURVE STUDY:
- Model: PyTorch 2-layer MLP (TF-IDF char n-grams + degree -> 128 -> 355 logits, BCEWithLogitsLoss on GPU)
- Scores and times:
  * N = 1,000: Recall@10 = 0.7742 +/- 0.0070, time/fold = 1.47s, 5-fold pipeline = ~7.4s
  * N = 3,000: Recall@10 = 0.7996 +/- 0.0159, time/fold = 0.15s, 5-fold pipeline = ~0.8s
  * N = 10,000: Recall@10 = 0.8857 +/- 0.0035, time/fold = 0.86s, 5-fold pipeline = ~4.3s
  * N = 25,000: Recall@10 = 0.9117 +/- 0.0036, time/fold = 1.69s, 5-fold pipeline = ~8.4s
  * N = 50,000: Recall@10 = 0.8635 +/- 0.0004, time/fold = 2.78s, 5-fold pipeline = ~13.9s
  * N = 18,682,899 (all train): Recall@10 = 0.8132 +/- 0.0005, time/fold = 899.2s, 5-fold pipeline = ~4496s
- Choice of pool size: 50,000 domains sampled from the locked 2 <= tracker_count <= 20 population.
  It matches the prediction target size (50,000 domains), runs 5-fold cross-validation in under 14 seconds
  (allowing over 1,500 full model explorations and tuning iterations within the remaining budget), avoids
  the extreme computational burden of 18.7M domains (~75 min per fold run), and achieves strong recall.
"""

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import KFold

TRACKING_GRAPH_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet"
DOMAINS_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet"


def recall_at_10(estimator, X, y):
    """
    Competition metric: Recall@10.
    For each domain, the fraction of its true trackers that appear anywhere in the
    top 10 predicted trackers for that domain, averaged across all domains.
    """
    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    n_classes = y_arr.shape[1] if y_arr.ndim > 1 else 1

    if hasattr(estimator, "predict_proba"):
        preds = estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        preds = estimator.decision_function(X)
    else:
        preds = estimator.predict(X)

    if isinstance(preds, list):
        # Handle list of binary class outputs from MultiOutputClassifier
        cols = []
        for p in preds:
            p_arr = np.asarray(p)
            if p_arr.ndim == 2 and p_arr.shape[1] >= 2:
                cols.append(p_arr[:, 1])
            elif p_arr.ndim == 2 and p_arr.shape[1] == 1:
                cols.append(p_arr[:, 0])
            else:
                cols.append(p_arr.ravel())
        preds = np.column_stack(cols)
    else:
        preds = np.asarray(preds)

    if preds.ndim == 2 and preds.shape[1] == n_classes:
        top_k = min(10, n_classes)
        top10 = np.argpartition(-preds, top_k, axis=1)[:, :top_k]
        hits = y_arr[np.arange(n_samples)[:, None], top10].sum(axis=1)
    elif preds.ndim == 2 and preds.shape[1] <= 10:
        # Directly returned top-k predicted tracker IDs
        hits = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            pred_set = set(preds[i])
            true_indices = np.flatnonzero(y_arr[i])
            hits[i] = len(pred_set.intersection(true_indices))
    else:
        top_k = min(10, preds.shape[1])
        top10 = np.argpartition(-preds, top_k, axis=1)[:, :top_k]
        hits = y_arr[np.arange(n_samples)[:, None], top10].sum(axis=1)

    true_counts = y_arr.sum(axis=1)
    recalls = hits / np.maximum(true_counts, 1)
    return float(np.mean(recalls))


def build_evaluation():
    # 1. Read tracking edges
    tracking_graph = skrub.as_data_op(TRACKING_GRAPH_PATH).skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    )

    # 2. Count trackers per domain and apply the locked population rule (2 <= tracker_count <= 20)
    tracker_counts = (
        tracking_graph.groupby("domain_id", as_index=False)
        .agg({"tracker_id": "count"})
        .rename(columns={"tracker_id": "tracker_count"})
    )
    valid_domains = tracker_counts[
        (tracker_counts["tracker_count"] >= 2)
        & (tracker_counts["tracker_count"] <= 20)
    ]

    # 3. Deterministically sample 50,000 domains and sort by domain_id
    sampled_domains = (
        valid_domains.sample(n=50000, random_state=42)
        .sort_values("domain_id")
        .reset_index(drop=True)
    )

    # 4. Build unmarked X by joining hostname lookup from domains.parquet
    domains = skrub.as_data_op(DOMAINS_PATH).skb.apply_func(pd.read_parquet)
    X_unmarked = sampled_domains[["domain_id"]].merge(
        domains, on="domain_id", how="left"
    )

    # 5. Build unmarked y (binary indicator matrix of shape 50,000 x 355)
    sampled_tracking = sampled_domains[["domain_id"]].merge(
        tracking_graph, on="domain_id", how="inner"
    )
    sampled_tracking = sampled_tracking.assign(val=1)
    pivot = sampled_tracking.pivot_table(
        index="domain_id", columns="tracker_id", values="val", fill_value=0
    )
    all_tracker_ids = list(range(355))
    pivot = pivot.reindex(columns=all_tracker_ids, fill_value=0)
    pivot = pivot.rename(columns={i: f"t_{i}" for i in range(355)})
    y_unmarked = pivot.reset_index(drop=True).astype("int8")

    # 6. CV and graph marking
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = X_unmarked.skb.mark_as_X(cv=cv, split_kwargs={})
    y = y_unmarked.skb.mark_as_y()

    return {"X": X, "y": y, "scoring": recall_at_10}


def build():
    return build_evaluation()
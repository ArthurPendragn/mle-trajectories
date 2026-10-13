import time
import numpy as np
import pandas as pd
import skrub
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = "gs://mle-trajectories-data/ttt-task/input/"


class LearningCurve(TransformerMixin, BaseEstimator):
    def __init__(self, epochs=12):
        self.epochs = epochs

    def fit(self, X, y=None):
        torch.set_num_threads(16)
        numeric = ["host_length", "host_dots", "host_hyphens", "host_digits"]
        labels = [f"label_{i}" for i in range(355)]
        target = X[labels].to_numpy(dtype=np.float32)
        counts = target.sum(axis=1)
        rng = np.random.RandomState(418)
        eligible = np.flatnonzero(X["restricted_pool"].to_numpy() & (counts >= 2))
        rng.shuffle(eligible)
        test_pool = eligible[:9000]
        remaining = np.ones(len(X), dtype=bool)
        remaining[test_pool] = False
        all_pool = np.flatnonzero(remaining & X["all_pool"].to_numpy())
        restricted_pool = np.flatnonzero(remaining & X["restricted_pool"].to_numpy())
        rng.shuffle(all_pool)
        rng.shuffle(restricted_pool)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []
        for rule, pool in [("all_labelled", all_pool), ("at_least_two", restricted_pool)]:
            for size in [10000, 40000, 160000]:
                if len(pool) < size:
                    continue
                train_idx = pool[:size]
                scores = []
                feature_seconds = 0.0
                started = time.perf_counter()
                for fold in range(3):
                    torch.manual_seed(731 + fold)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(731 + fold)
                    test_idx = test_pool[fold * 3000:(fold + 1) * 3000]
                    feature_started = time.perf_counter()
                    encoder = OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        max_categories=128,
                        sparse_output=False,
                        dtype=np.float32,
                    )
                    scaler = StandardScaler()
                    train_cat = encoder.fit_transform(X.iloc[train_idx][["tld"]])
                    test_cat = encoder.transform(X.iloc[test_idx][["tld"]])
                    train_num = scaler.fit_transform(
                        X.iloc[train_idx][numeric].to_numpy(dtype=np.float32)
                    )
                    test_num = scaler.transform(
                        X.iloc[test_idx][numeric].to_numpy(dtype=np.float32)
                    )
                    train_features = np.concatenate([train_cat, train_num], axis=1)
                    test_features = np.concatenate([test_cat, test_num], axis=1)
                    tx = torch.as_tensor(train_features, device=device)
                    vx = torch.as_tensor(test_features, device=device)
                    ty = torch.as_tensor(
                        target[train_idx] / counts[train_idx, None], device=device
                    )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    feature_seconds += time.perf_counter() - feature_started
                    net = torch.nn.Sequential(
                        torch.nn.Linear(tx.shape[1], 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 355),
                    ).to(device)
                    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.001)
                    dataset = torch.utils.data.TensorDataset(tx, ty)
                    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
                    net.train()
                    for epoch in range(self.epochs):
                        total_loss = 0.0
                        for bx, by in loader:
                            optimizer.zero_grad()
                            loss = -(by * torch.log_softmax(net(bx), dim=1)).sum(dim=1).mean()
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss.detach()) * len(bx)
                        if epoch == self.epochs - 1:
                            print(rule, size, fold, "loss", total_loss / size, flush=True)
                    net.eval()
                    with torch.no_grad():
                        top = net(vx).topk(10, dim=1).indices.cpu().numpy()
                    truth = target[test_idx]
                    recall = np.take_along_axis(truth, top, axis=1).sum(axis=1) / counts[test_idx]
                    scores.append(float(recall.mean()))
                    if device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                results.append({
                    "rule": rule,
                    "training_rows_per_fold": size,
                    "test_rows_per_fold": 3000,
                    "model": "128-hidden-unit GPU MLP; normalized-positive softmax loss",
                    "epochs": self.epochs,
                    "device": device,
                    "recall_at_10": float(np.mean(scores)),
                    "recall_std": float(np.std(scores)),
                    "fold_scores": scores,
                    "seconds_all_folds": elapsed,
                    "feature_seconds_all_folds": feature_seconds,
                    "timing_scope": "fold feature fitting/building, model fitting, scoring; source graph excluded",
                })
                print(results[-1], flush=True)
        self.results_ = pd.DataFrame(results)
        self.model_ = net
        return self

    def transform(self, X):
        return self.results_.copy()


def build():
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    )
    counts = edges.groupby("domain_id").size().rename("tracker_count").reset_index()
    all_sample = counts.sample(n=220000, random_state=411)[["domain_id"]].assign(all_pool=True)
    restricted_sample = counts[counts["tracker_count"] >= 2].sample(
        n=200000, random_state=412
    )[["domain_id"]].assign(restricted_pool=True)
    pool = all_sample.merge(restricted_sample, on="domain_id", how="outer")
    pool = pool.assign(
        all_pool=pool["all_pool"].fillna(False),
        restricted_pool=pool["restricted_pool"].fillna(False),
    )
    selected_edges = edges[edges["domain_id"].isin(pool["domain_id"])].assign(present=1)
    labels = selected_edges.pivot_table(
        index="domain_id", columns="tracker_id", values="present",
        aggfunc="max", fill_value=0
    ).reindex(columns=list(range(355)), fill_value=0)
    labels = labels.rename(columns={i: f"label_{i}" for i in range(355)}).reset_index()
    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"]
    )
    domains = domains[domains["domain_id"].isin(pool["domain_id"])]
    features = domains.assign(
        tld=domains["domain"].str.rsplit(".", n=1).str[-1].fillna(""),
        host_length=domains["domain"].str.len(),
        host_dots=domains["domain"].str.count(r"\."),
        host_hyphens=domains["domain"].str.count("-"),
        host_digits=domains["domain"].str.count(r"\d"),
    ).drop(columns=["domain"])
    data = pool.merge(features, on="domain_id", how="left").merge(
        labels, on="domain_id", how="left"
    )
    curve = data.skb.apply(LearningCurve()).skb.set_name("measured_learning_curve")
    return {
        "learning_curve_and_paired_population_comparison": curve,
        "sample_counts": pool[["all_pool", "restricted_pool"]].sum(),
        "design": skrub.as_data_op({
            "held_out_population": "9000 randomly sampled domains with at least two recorded trackers; identical across every comparison; three disjoint 3000-row test sets",
            "training": "independent deterministic samples from each population; all 9000 held-out domains excluded from both training pools",
            "metric": "mean per-domain Recall@10",
            "objective": "cross entropy against positives normalized by domain tracker count; estimates expected per-tracker recall contribution",
            "pool_sizing": "Compare measured gains and full-three-fold times at 10k, 40k and 160k; extrapolate ceiling from largest timing, reserving source-read overhead and time for at least twelve experiments",
            "recommendation_pending_measurements": "At-least-two is the least restrictive near-chance adversarial population; paired recall evidence here determines whether the exclusion is beneficial",
            "graph_scope": "This curve uses hostnames only and no hyperlinks. Future graph features need only edges incident to modelled or prediction domains, plus explicitly justified further-hop edges",
            "label_safety": "No label-derived features in this experiment. Future graph-label aggregates must use marked training-fold labels or domains provably outside the entire locked population; never read held-out labels back from the tracking source",
            "source_timing": "Source ingestion and graph joins are outside reported model times and must be added when budgeting later experiments",
        }),
    }
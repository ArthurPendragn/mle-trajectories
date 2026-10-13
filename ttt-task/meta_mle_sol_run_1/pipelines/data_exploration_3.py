import time
import resource
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skrub
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

ROOT = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
REMAINING_SECONDS = 55727.0
SEED = 427
VALIDATION_SEED = 617
N_TRACKERS = 355


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, N_TRACKERS),
        )

    def forward(self, x):
        return self.layers(x)


class ExtendedCurve(TransformerMixin, BaseEstimator):
    def __init__(self, sizes=(120000, 400000, 1200000), epochs=6):
        self.sizes = sizes
        self.epochs = epochs

    def fit(self, X, y=None):
        torch.set_num_threads(16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph_finished = pd.Timestamp.now()
        source_seconds = float(
            (graph_finished - pd.Timestamp(X["started"].iloc[0])).total_seconds()
        )
        records = []

        numeric = [
            "hostname_length", "hostname_dots", "hostname_digits",
            "hostname_hyphens", "classified", "out_degree", "in_degree",
            "out_classified_fraction", "in_classified_fraction",
        ]
        categorical = ["tld", "category"]
        target = X[X["is_target"].eq(1)]
        adversarial_pool = X[X["adversarial_sample"].eq(1) & X["is_target"].eq(0)]
        adversarial_started = time.perf_counter()
        split = StratifiedKFold(n_splits=3, shuffle=True, random_state=617)
        rules = [("all", adversarial_pool)]
        for k in (1, 2, 3, 5, 10):
            rules.append(("at_most_" + str(k),
                          adversarial_pool[adversarial_pool["label_count"].le(k)]))
            rules.append(("at_least_" + str(k),
                          adversarial_pool[adversarial_pool["label_count"].ge(k)]))
        for lo, hi in ((1, 2), (2, 3), (2, 5), (3, 10), (5, 10)):
            rules.append(("band_%d_%d" % (lo, hi),
                          adversarial_pool[
                              adversarial_pool["label_count"].between(lo, hi)]))
        rules.extend([
            ("classified", adversarial_pool[adversarial_pool["classified"].eq(1)]),
            ("unclassified", adversarial_pool[adversarial_pool["classified"].eq(0)]),
            ("has_outgoing", adversarial_pool[adversarial_pool["out_degree"].gt(0)]),
            ("no_outgoing", adversarial_pool[adversarial_pool["out_degree"].eq(0)]),
            ("has_incoming", adversarial_pool[adversarial_pool["in_degree"].gt(0)]),
            ("no_incoming", adversarial_pool[adversarial_pool["in_degree"].eq(0)]),
        ])
        auc_rows = []
        for name, eligible in rules:
            n = min(15000, len(eligible), len(target))
            if n < 12:
                continue
            left = eligible.sample(n=n, random_state=427)
            right = target.sample(n=n, random_state=427)
            features = pd.concat(
                [left[numeric + categorical], right[numeric + categorical]],
                ignore_index=True,
            )
            for column in categorical:
                features[column] = features[column].fillna("missing").astype(str)
            features[numeric] = features[numeric].fillna(0).astype(float)
            labels = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
            scores = []
            for train, test in split.split(features, labels):
                model = CatBoostClassifier(
                    iterations=140, depth=5, learning_rate=0.08,
                    loss_function="Logloss", random_seed=617,
                    thread_count=16, verbose=False, allow_writing_files=False,
                )
                model.fit(features.iloc[train], labels[train],
                          cat_features=categorical)
                scores.append(roc_auc_score(
                    labels[test], model.predict_proba(features.iloc[test])[:, 1]))
            row = {
                "section": "adversarial", "rule": name,
                "balanced_per_class": n,
                "sampled_eligible": len(eligible),
                "retained_domains": eligible["retained_" + name].iloc[0],
                "auc_fold_1": scores[0], "auc_fold_2": scores[1],
                "auc_fold_3": scores[2],
                "auc_mean": float(np.mean(scores)),
                "auc_std": float(np.std(scores, ddof=1)),
            }
            records.append(row)
            auc_rows.append(row)
        adversarial_seconds = time.perf_counter() - adversarial_started

        lowest = min(auc_rows, key=lambda r: r["auc_mean"])
        tied = [r for r in auc_rows
                if r["auc_mean"] <= lowest["auc_mean"] + lowest["auc_std"]]
        selected = max(tied, key=lambda r: r["retained_domains"])

        pool = X[X["curve_row"].eq(1)].copy()
        validation = pool[pool["validation_fold"].ge(0)].copy()
        training = pool[pool["validation_fold"].eq(-1)].copy()
        training = training.sort_values("training_order")
        experiment_allowance = 0.8 * REMAINING_SECONDS / 12
        # Fourfold inflation of the measured variable cost, rather than
        # extrapolating the unusually small 40k measurement.
        initial_slope = 4.0 * 22.026534 / 120000.0
        initial_limit = min(
            8635505 - 6000,
            max(0, int((experiment_allowance - 51.194288) / initial_slope)),
        )
        current_limit = initial_limit
        previous_mean = None
        previous_std = None
        measured = []

        for size in self.sizes:
            if size > current_limit or size > len(training):
                records.append({
                    "section": "decision", "rule": "size_not_executed",
                    "training_domains_per_fold": size,
                    "timing_limit": current_limit,
                    "reason": "Size exceeds the current conservative extrapolated limit.",
                })
                break
            fold_scores = []
            vector_seconds = 0.0
            fit_score_seconds = 0.0
            gpu_peak = 0.0
            for fold in range(3):
                begin = time.perf_counter()
                tr = training.iloc[:size]
                va = validation[validation["validation_fold"].eq(fold)]
                text = TfidfVectorizer(
                    analyzer="char", ngram_range=(2, 4), max_features=4096,
                    min_df=3, sublinear_tf=True, dtype=np.float32,
                )
                enc = OneHotEncoder(handle_unknown="ignore",
                                    sparse_output=True, dtype=np.float32)
                scale = StandardScaler(with_mean=False)
                tr_num = tr[numeric].fillna(0).to_numpy(dtype=np.float32)
                va_num = va[numeric].fillna(0).to_numpy(dtype=np.float32)
                degree_columns = [numeric.index("out_degree"),
                                  numeric.index("in_degree")]
                tr_num[:, degree_columns] = np.log1p(tr_num[:, degree_columns])
                va_num[:, degree_columns] = np.log1p(va_num[:, degree_columns])
                tr_cat = tr[categorical].fillna("missing").astype(str)
                va_cat = va[categorical].fillna("missing").astype(str)
                tr_features = sp.hstack([
                    text.fit_transform(tr["host"].fillna("")),
                    enc.fit_transform(tr_cat),
                    sp.csr_matrix(scale.fit_transform(tr_num)),
                ], format="csr", dtype=np.float32)
                va_features = sp.hstack([
                    text.transform(va["host"].fillna("")),
                    enc.transform(va_cat),
                    sp.csr_matrix(scale.transform(va_num)),
                ], format="csr", dtype=np.float32)
                tr_y = np.zeros((len(tr), N_TRACKERS), dtype=np.float32)
                va_y = np.zeros((len(va), N_TRACKERS), dtype=np.float32)
                for i, ids in enumerate(tr["tracker_ids"]):
                    tr_y[i, np.asarray(ids, dtype=int)] = 1
                for i, ids in enumerate(va["tracker_ids"]):
                    va_y[i, np.asarray(ids, dtype=int)] = 1
                vector_seconds += time.perf_counter() - begin

                begin = time.perf_counter()
                torch.manual_seed(617 + fold)
                if device == "cuda":
                    torch.cuda.manual_seed_all(617 + fold)
                    torch.cuda.reset_peak_memory_stats()
                net = Network(tr_features.shape[1]).to(device)
                optimizer = torch.optim.AdamW(net.parameters(), lr=0.002,
                                              weight_decay=0.0001)
                generator = torch.Generator().manual_seed(617 + fold)
                dataset = TensorDataset(
                    torch.arange(len(tr)), torch.from_numpy(tr_y))
                loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                                    generator=generator, num_workers=0)
                net.train()
                for epoch in range(self.epochs):
                    total_loss = 0.0
                    for idx, truth in loader:
                        features = torch.from_numpy(
                            tr_features[idx.numpy()].toarray()).to(device)
                        truth = truth.to(device)
                        logits = net(features)
                        row_loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, truth, reduction="none").sum(dim=1)
                        loss = (row_loss / truth.sum(dim=1).clamp_min(1)).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(idx)
                    print("curve", size, "fold", fold, "epoch", epoch,
                          "weighted_bce", total_loss / len(tr), flush=True)
                net.eval()
                recovered = []
                with torch.no_grad():
                    for start in range(0, len(va), 1024):
                        end = min(start + 1024, len(va))
                        features = torch.from_numpy(
                            va_features[start:end].toarray()).to(device)
                        top = net(features).topk(10, dim=1).indices.cpu().numpy()
                        truth = va_y[start:end]
                        hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
                        recovered.extend((hits / truth.sum(axis=1)).tolist())
                if device == "cuda":
                    torch.cuda.synchronize()
                    gpu_peak = max(
                        gpu_peak, torch.cuda.max_memory_allocated() / 2**30)
                fit_score_seconds += time.perf_counter() - begin
                fold_scores.append(float(np.mean(recovered)))
                self.net_ = net
                self.text_ = text
                self.encoder_ = enc
                self.scaler_ = scale

            complete = source_seconds + vector_seconds + fit_score_seconds
            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores, ddof=1))
            conservative_slope = max(
                initial_slope, 3.0 * (vector_seconds + fit_score_seconds) / size)
            current_limit = min(
                8635505 - 6000,
                max(0, int((experiment_allowance - source_seconds)
                           / conservative_slope)),
            )
            row = {
                "section": "curve",
                "model": "CUDA char-ngram MLP, 256 hidden units, inverse-cardinality BCE",
                "training_domains_per_fold": size,
                "validation_domains_per_fold": 2000,
                "recall_fold_1": fold_scores[0],
                "recall_fold_2": fold_scores[1],
                "recall_fold_3": fold_scores[2],
                "recall_mean": mean, "recall_std": std,
                "shared_source_feature_seconds": source_seconds,
                "all_fold_fitted_feature_seconds": vector_seconds,
                "all_fold_fit_score_seconds": fit_score_seconds,
                "complete_experiment_seconds": complete,
                "process_peak_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "gpu_peak_allocated_gib": gpu_peak,
                "updated_extrapolated_training_limit": current_limit,
                "device": device,
            }
            records.append(row)
            measured.append(row)
            if previous_mean is not None:
                noise = max(std, previous_std)
                if mean - previous_mean <= noise:
                    records.append({
                        "section": "decision", "rule": "curve_flattened",
                        "reason": "Incremental gain is no larger than fold spread.",
                        "recall_gain": mean - previous_mean,
                        "fold_noise": noise,
                    })
                    break
            previous_mean, previous_std = mean, std

        recommendation = measured[-1] if measured else None
        records.append({
            "section": "recommendation",
            "selected_rule": selected["rule"],
            "lowest_auc_rule": lowest["rule"],
            "lowest_auc": lowest["auc_mean"],
            "tie_tolerance": lowest["auc_std"],
            "less_restrictive_tied_rules": ", ".join(
                r["rule"] for r in tied
                if r["retained_domains"] > lowest["retained_domains"]),
            "recommended_training_domains": (
                recommendation["training_domains_per_fold"]
                if recommendation else 0),
            "recommended_pool_domains": (
                recommendation["training_domains_per_fold"] + 6000
                if recommendation else 0),
            "initial_extrapolated_training_limit": initial_limit,
            "updated_extrapolated_training_limit": current_limit,
            "per_experiment_allowance_seconds": experiment_allowance,
            "adversarial_fit_score_seconds": adversarial_seconds,
            "reason": (
                "Choose the largest measured non-prohibited size. Reserve 20% runtime "
                "and inflate observed variable costs threefold to allow more expensive "
                "families; a timing limit is not a measured performance guarantee."
            ),
            "fold_reproducibility": (
                "Previous exploration source and exact validation IDs were not supplied. "
                "These are reproducible reconstructed subsets, not a claim of identical "
                "earlier folds. Re-measured 120k is the within-exploration baseline."
            ),
            "graph_slice": (
                "Outgoing hyperlinks with source in modelled/predicted domains and "
                "incoming hyperlinks with target in those domains. Neighbour categories "
                "use classification metadata only."
            ),
            "label_masking": (
                "This curve uses no supervised graph features. Future auxiliary label "
                "tables must exclude every locked-pool domain, including all validation "
                "subsets, before any join or walk. Labels outside the pool remain usable."
            ),
            "metric": (
                "Mean domain-level fraction of distinct true trackers recovered by "
                "ten unique tracker IDs; every validation domain enters the mean."
            ),
            "caveat": (
                "Observable similarity does not prove target label completeness or "
                "conditional-label equivalence. Historical unpublished filter identities "
                "cannot be recovered; all six complementary metadata/degree filters "
                "are explicitly recomputed here."
            ),
        })
        self.report_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.report_.copy()

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def build():
    started = skrub.as_data_op("now").skb.apply_func(pd.Timestamp)
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"])
    counts = edges.groupby("domain_id").size().reset_index(name="label_count")
    eligible = counts[counts["label_count"] >= 2]
    validation = eligible.sample(n=6000, random_state=VALIDATION_SEED).reset_index(drop=True)
    validation = validation.assign(
        validation_fold=validation.index // 2000,
        training_order=-1,
        curve_row=1,
    )
    training = eligible[
        ~eligible["domain_id"].isin(validation["domain_id"])
    ].sample(n=1200000, random_state=SEED).reset_index(drop=True)
    training = training.assign(
        training_order=training.index, validation_fold=-1, curve_row=1)
    curve_rows = training.skb.concat([validation], axis=0).reset_index(drop=True)
    adv = counts[counts["domain_id"] % 40 == 0][["domain_id"]]
    adv = adv.assign(adversarial_sample=1)
    target = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"])
    target = target.assign(is_target=1)
    ids = curve_rows[["domain_id"]].skb.concat(
        [adv[["domain_id"]], target[["domain_id"]]], axis=0).drop_duplicates()
    rows = ids.merge(counts, on="domain_id", how="left")
    rows = rows.merge(
        curve_rows[["domain_id", "curve_row", "validation_fold", "training_order"]],
        on="domain_id", how="left")
    rows = rows.merge(adv, on="domain_id", how="left")
    rows = rows.merge(target, on="domain_id", how="left")
    rows = rows.assign(started=started)

    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"])
    domains = domains[domains["domain_id"].isin(rows["domain_id"])]
    domains = domains.assign(
        host=domains["domain"].str.lower().str.replace(r"^www\.", "", regex=True))
    classifications = skrub.as_data_op(
        ROOT + "url-classification.csv").skb.apply_func(
            pd.read_csv, usecols=["url", "category"])
    classifications = classifications.assign(
        host=classifications["url"].str.lower()
        .str.replace(r"^[a-z]+://", "", regex=True)
        .str.split("/").str[0]
        .str.replace(r"^www\.", "", regex=True))
    classifications = classifications[["host", "category"]].drop_duplicates("host")
    domains = domains.merge(classifications, on="host", how="left")
    domains = domains.assign(
        classified=domains["category"].notna().astype("int8"),
        tld=domains["host"].str.rsplit(".", n=1).str[-1],
        hostname_length=domains["host"].str.len(),
        hostname_dots=domains["host"].str.count(r"\."),
        hostname_digits=domains["host"].str.count(r"\d"),
        hostname_hyphens=domains["host"].str.count("-"),
    )
    rows = rows.merge(domains, on="domain_id", how="left")
    rows = rows.assign(classified=rows["classified"].fillna(0))

    # Classification coverage of neighbours is label-free and does not require
    # retaining the full domain lookup in later fits.
    all_domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"])
    all_domains = all_domains.assign(
        host=all_domains["domain"].str.lower().str.replace(r"^www\.", "", regex=True))
    classified_ids = all_domains[
        all_domains["host"].isin(classifications["host"])
    ][["domain_id"]].drop_duplicates()
    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"])
    out_links = links[links["source_domain_id"].isin(ids["domain_id"])]
    in_links = links[links["target_domain_id"].isin(ids["domain_id"])]
    out_links = out_links.assign(
        neighbour_classified=out_links["target_domain_id"].isin(
            classified_ids["domain_id"]).astype("int8"))
    in_links = in_links.assign(
        neighbour_classified=in_links["source_domain_id"].isin(
            classified_ids["domain_id"]).astype("int8"))
    out_stats = out_links.groupby("source_domain_id").agg(
        out_degree=("target_domain_id", "size"),
        out_classified_fraction=("neighbour_classified", "mean"),
    ).reset_index().rename(columns={"source_domain_id": "domain_id"})
    in_stats = in_links.groupby("target_domain_id").agg(
        in_degree=("source_domain_id", "size"),
        in_classified_fraction=("neighbour_classified", "mean"),
    ).reset_index().rename(columns={"target_domain_id": "domain_id"})
    rows = rows.merge(out_stats, on="domain_id", how="left")
    rows = rows.merge(in_stats, on="domain_id", how="left")
    rows = rows.assign(
        out_degree=rows["out_degree"].fillna(0),
        in_degree=rows["in_degree"].fillna(0),
        out_classified_fraction=rows["out_classified_fraction"].fillna(0),
        in_classified_fraction=rows["in_classified_fraction"].fillna(0),
        is_target=rows["is_target"].fillna(0),
        curve_row=rows["curve_row"].fillna(0),
        adversarial_sample=rows["adversarial_sample"].fillna(0),
    )

    labels = edges[edges["domain_id"].isin(curve_rows["domain_id"])]
    labels = labels.groupby("domain_id").agg(
        tracker_ids=("tracker_id", list)).reset_index()
    rows = rows.merge(labels, on="domain_id", how="left")

    # Tracker-count rule sizes are exact. Coverage-filter population sizes
    # are estimated from the identical deterministic 1/40 labelled sample.
    rows = rows.assign(retained_all=counts.shape[0])
    for k in (1, 2, 3, 5, 10):
        rows = rows.assign(**{
            "retained_at_most_" + str(k):
                counts[counts["label_count"] <= k].shape[0],
            "retained_at_least_" + str(k):
                counts[counts["label_count"] >= k].shape[0],
        })
    for lo, hi in ((1, 2), (2, 3), (2, 5), (3, 10), (5, 10)):
        rows = rows.assign(**{
            "retained_band_%d_%d" % (lo, hi):
                counts[counts["label_count"].between(lo, hi)].shape[0],
        })
    sampled = rows[
        rows["adversarial_sample"].eq(1) & rows["is_target"].eq(0)]
    coverage = {
        "classified": sampled["classified"].eq(1),
        "unclassified": sampled["classified"].eq(0),
        "has_outgoing": sampled["out_degree"].gt(0),
        "no_outgoing": sampled["out_degree"].eq(0),
        "has_incoming": sampled["in_degree"].gt(0),
        "no_incoming": sampled["in_degree"].eq(0),
    }
    for name, mask in coverage.items():
        rows = rows.assign(**{
            "retained_" + name:
                sampled[mask].shape[0] * counts.shape[0] / sampled.shape[0],
        })
    report = rows.skb.apply(ExtendedCurve())
    comparisons = report[report["section"].eq("adversarial")]
    coverage_names = [
        "classified", "unclassified", "has_outgoing", "no_outgoing",
        "has_incoming", "no_incoming",
    ]
    return {
        "learning_curve": report[report["section"].eq("curve")].dropna(axis=1, how="all"),
        "count_rule_comparisons": comparisons[
            ~comparisons["rule"].isin(coverage_names)].dropna(axis=1, how="all"),
        "coverage_rule_comparisons": comparisons[
            comparisons["rule"].isin(coverage_names)].dropna(axis=1, how="all"),
        "recommendation": report[
            report["section"].eq("recommendation")].dropna(axis=1, how="all"),
        "stopping_decisions": report[
            report["section"].eq("decision")].dropna(axis=1, how="all"),
        "curve_pool_shape": curve_rows.shape,
        "outgoing_graph_slice_shape": out_links.shape,
        "incoming_graph_slice_shape": in_links.shape,
    }
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skrub
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"


class LinearTrackerNetwork(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.linear = nn.Linear(width, outputs)

    def forward(self, x):
        return self.linear(x)


class PopulationStudy(TransformerMixin, BaseEstimator):
    def __init__(self, adversarial_cap=12000, curve_sizes=(2000, 8000, 24000, 64000)):
        self.adversarial_cap = adversarial_cap
        self.curve_sizes = curve_sizes

    def fit(self, X, y=None):
        started = time.perf_counter()
        torch.set_num_threads(16)
        reports = []

        def record(section, name, value):
            reports.append({
                "section": section,
                "name": name,
                "value": json.dumps(value, default=str),
            })

        rows = X.reset_index(drop=True).copy()
        edges = y[["domain_id", "tracker_id"]].drop_duplicates()
        counts = rows["label_count"].fillna(0).to_numpy(dtype=int)
        target_mask = rows["is_target"].fillna(0).to_numpy() == 1
        labelled_mask = (counts > 0) & ~target_mask
        record("diagnostics", "population", {
            "rows": len(rows),
            "targets": int(target_mask.sum()),
            "labelled_non_targets": int(labelled_mask.sum()),
            "targets_with_known_labels": int(((counts > 0) & target_mask).sum()),
            "unique_label_edges": len(edges),
            "labelled_domains": int(edges.domain_id.nunique()),
            "trackers_observed": int(edges.tracker_id.nunique()),
            "count_quantiles_labelled": pd.Series(counts[labelled_mask]).quantile(
                [0, .1, .25, .5, .75, .9, .99, 1]).to_dict(),
            "count_quantiles_target_overlap": pd.Series(
                counts[target_mask & (counts > 0)]).quantile(
                    [0, .25, .5, .75, 1]).to_dict(),
        })

        cat_cols = ["tld", "category"]
        num_cols = [
            "hostname_length", "hostname_dots", "hostname_digits",
            "hostname_hyphens", "classified", "out_degree", "in_degree",
            "out_classified_neighbors", "in_classified_neighbors",
            "out_classified_fraction", "in_classified_fraction",
            "freedom_of_the_press",
        ]
        feature = rows[cat_cols + num_cols].copy()
        for col in cat_cols:
            feature[col] = feature[col].fillna("__missing__").astype(str)
        for col in num_cols:
            feature[col] = pd.to_numeric(feature[col], errors="coerce").fillna(-1)
        for col in [
            "out_degree", "in_degree", "out_classified_neighbors",
            "in_classified_neighbors",
        ]:
            feature[col] = np.log1p(feature[col].clip(lower=0))

        record("diagnostics", "observable_distributions", {
            "labelled": feature.loc[labelled_mask, num_cols].describe().to_dict(),
            "target": feature.loc[target_mask, num_cols].describe().to_dict(),
            "labelled_top_tlds": feature.loc[labelled_mask, "tld"].value_counts().head(20).to_dict(),
            "target_top_tlds": feature.loc[target_mask, "tld"].value_counts().head(20).to_dict(),
            "labelled_top_categories": feature.loc[labelled_mask, "category"].value_counts().head(20).to_dict(),
            "target_top_categories": feature.loc[target_mask, "category"].value_counts().head(20).to_dict(),
        })

        rng = np.random.default_rng(2401)
        order = rng.permutation(len(rows))
        target_ids = order[target_mask[order]][:self.adversarial_cap]
        rules = [("all_labelled", labelled_mask)]
        for threshold in [2, 3, 5, 8, 10, 15, 20, 30]:
            rules.append(("count_at_least_" + str(threshold),
                          labelled_mask & (counts >= threshold)))
            rules.append(("count_at_most_" + str(threshold),
                          labelled_mask & (counts <= threshold)))
        for lo, hi in [(2, 5), (3, 10), (5, 15), (8, 20), (10, 30), (2, 20)]:
            rules.append(("count_band_%d_%d" % (lo, hi),
                          labelled_mask & (counts >= lo) & (counts <= hi)))
        out_degree = rows.out_degree.fillna(0).to_numpy()
        in_degree = rows.in_degree.fillna(0).to_numpy()
        rules.extend([
            ("has_out_links", labelled_mask & (out_degree > 0)),
            ("has_in_links", labelled_mask & (in_degree > 0)),
            ("has_any_links", labelled_mask & ((out_degree + in_degree) > 0)),
            ("has_both_directions", labelled_mask & (out_degree > 0) & (in_degree > 0)),
            ("no_links", labelled_mask & ((out_degree + in_degree) == 0)),
            ("classified_only", labelled_mask & (rows.classified.to_numpy() > 0)),
            ("unclassified_only", labelled_mask & (rows.classified.to_numpy() == 0)),
        ])
        results = []
        for name, mask in rules:
            tick = time.perf_counter()
            ids = order[mask[order]][:self.adversarial_cap]
            n = min(len(ids), len(target_ids))
            if n < 150:
                record("adversarial", name, {
                    "population_size": int(mask.sum()), "status": "too_few_rows",
                    "sample_per_class": n,
                })
                continue
            ix = np.concatenate([ids[:n], target_ids[:n]])
            labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
            a = feature.iloc[ix].reset_index(drop=True)
            scores = []
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=71).split(a, labels):
                model = CatBoostClassifier(
                    iterations=160, depth=5, learning_rate=.08,
                    loss_function="Logloss", thread_count=16, verbose=False,
                    random_seed=71, allow_writing_files=False,
                )
                model.fit(a.iloc[tr], labels[tr], cat_features=cat_cols)
                scores.append(float(roc_auc_score(
                    labels[te], model.predict_proba(a.iloc[te])[:, 1])))
            result = {
                "rule": name, "population_size": int(mask.sum()),
                "sample_per_class": n, "fold_auc": scores,
                "mean_auc": float(np.mean(scores)),
                "std_auc": float(np.std(scores, ddof=1)),
                "elapsed_seconds": time.perf_counter() - tick,
            }
            results.append(result)
            record("adversarial", name, result)
            print("adversarial", name, result, flush=True)

        if not results:
            self.report_ = pd.DataFrame(reports)
            return self
        best = min(results, key=lambda r: r["mean_auc"])
        compatible = [
            r for r in results
            if r["mean_auc"] <= best["mean_auc"] + max(
                best["std_auc"], r["std_auc"], .002)
        ]
        selected = max(compatible, key=lambda r: r["population_size"])
        population = dict(rules)[selected["rule"]]
        pool = order[population[order]]
        record("recommendation", "population", {
            "lowest_auc_rule": best,
            "least_restrictive_within_fold_noise": selected,
            "meaningful_remaining_shift": selected["mean_auc"] > .55,
            "selection_caution": (
                "AUC above 0.55 is meaningful shift, not proof of population equivalence. "
                "Tracker-count filters are hypotheses about hidden target selection, "
                "not evidence of complete labels."
            ),
        })

        if len(pool) < 1000:
            record("learning_curve", "skipped", "Selected population has fewer than 1000 rows.")
            self.report_ = pd.DataFrame(reports)
            return self

        # No supervised graph statistics or validation labels enter features.
        index = pd.Index(rows.domain_id)
        label_row = index.get_indexer(edges.domain_id)
        good = label_row >= 0
        n_outputs = max(355, int(edges.tracker_id.max()) + 1)
        labels_sparse = sp.csr_matrix(
            (np.ones(int(good.sum()), dtype=np.float32),
             (label_row[good], edges.loc[good, "tracker_id"].to_numpy(dtype=int))),
            shape=(len(rows), n_outputs),
        )
        labels_sparse.data[:] = 1
        test_reservoir = pool[:min(6000, max(600, len(pool) // 5))]
        folds = list(KFold(3, shuffle=True, random_state=320).split(test_reservoir))
        available = len(pool) - max(len(te) for _, te in folds)
        sizes = sorted(set(min(int(s), available) for s in self.curve_sizes if s > 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        curve = []
        for size in sizes:
            tick = time.perf_counter()
            fold_scores = []
            for fold_number, (_, te_local) in enumerate(folds):
                test_ids = test_reservoir[te_local]
                train_ids = pool[~np.isin(pool, test_ids)][:size]
                feature_tick = time.perf_counter()
                train_frame = rows.iloc[train_ids]
                test_frame = rows.iloc[test_ids]
                hasher = HashingVectorizer(
                    analyzer="char", ngram_range=(2, 5), n_features=4096,
                    alternate_sign=False, norm="l2", dtype=np.float32,
                    lowercase=True,
                )
                train_text = hasher.transform(train_frame.domain.fillna("").astype(str))
                test_text = hasher.transform(test_frame.domain.fillna("").astype(str))
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                        dtype=np.float32, min_frequency=10)
                train_cat = encoder.fit_transform(feature.iloc[train_ids][cat_cols])
                test_cat = encoder.transform(feature.iloc[test_ids][cat_cols])
                scaler = StandardScaler()
                train_num = scaler.fit_transform(
                    feature.iloc[train_ids][num_cols]).astype(np.float32)
                test_num = scaler.transform(
                    feature.iloc[test_ids][num_cols]).astype(np.float32)
                train_x = sp.hstack(
                    [train_text, train_cat, sp.csr_matrix(train_num)], format="csr")
                test_x = sp.hstack(
                    [test_text, test_cat, sp.csr_matrix(test_num)], format="csr")
                train_y = labels_sparse[train_ids].toarray()
                test_y = labels_sparse[test_ids].toarray()
                feature_seconds = time.perf_counter() - feature_tick
                torch.manual_seed(77 + fold_number)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(77 + fold_number)
                net = LinearTrackerNetwork(train_x.shape[1], n_outputs).to(device)
                optimizer = torch.optim.AdamW(net.parameters(), lr=.012, weight_decay=.001)
                batch_rng = np.random.default_rng(77 + fold_number)
                # Equal positive reward per domain aligns with macro recall.
                for epoch in range(8):
                    net.train()
                    permutation = batch_rng.permutation(len(train_ids))
                    for begin in range(0, len(permutation), 512):
                        batch = permutation[begin:begin + 512]
                        bx = torch.from_numpy(train_x[batch].toarray()).to(device)
                        by = torch.from_numpy(train_y[batch]).to(device)
                        positive = by / by.sum(dim=1, keepdim=True).clamp(min=1)
                        negative = (1 - by) / (1 - by).sum(
                            dim=1, keepdim=True).clamp(min=1)
                        logits = net(bx)
                        loss = (
                            positive * torch.nn.functional.softplus(-logits)
                            + .3 * negative * torch.nn.functional.softplus(logits)
                        ).sum(dim=1).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                net.eval()
                recalls = []
                with torch.no_grad():
                    for begin in range(0, len(test_ids), 512):
                        end = min(begin + 512, len(test_ids))
                        bx = torch.from_numpy(test_x[begin:end].toarray()).to(device)
                        top = net(bx).topk(10, dim=1).indices.cpu().numpy()
                        truth = test_y[begin:end]
                        hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
                        recalls.extend((hits / np.maximum(truth.sum(axis=1), 1)).tolist())
                score = float(np.mean(recalls))
                fold_scores.append(score)
                print("curve", size, fold_number, score,
                      "feature_seconds", feature_seconds, flush=True)
                del net, optimizer, train_x, test_x, train_y, test_y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = time.perf_counter() - tick
            result = {
                "training_domains_per_fold": size,
                "test_domains_total": len(test_reservoir),
                "fold_scores": fold_scores,
                "mean_recall_at_10": float(np.mean(fold_scores)),
                "std_recall_at_10": float(np.std(fold_scores, ddof=1)),
                "seconds_all_folds_including_features": elapsed,
                "device": str(device),
                "model": "hashed hostname + categorical/numeric linear neural multilabel ranker",
                "objective": "inverse-label-count positive logistic loss plus negative logistic loss",
            }
            curve.append(result)
            record("learning_curve", "size_" + str(size), result)
            remaining = max(0, 57579 - (time.perf_counter() - started))
            if len(curve) >= 2:
                previous = curve[-2]
                slope = max(
                    (elapsed - previous["seconds_all_folds_including_features"]) /
                    max(size - previous["training_domains_per_fold"], 1),
                    elapsed / max(size, 1) * .5,
                )
                intercept = max(0, elapsed - slope * size)
                safe_limit = max(size, int((remaining / 12 - intercept) / slope))
                record("learning_curve", "extrapolation_" + str(size), {
                    "seconds_per_added_training_domain_all_folds": slope,
                    "remaining_seconds": remaining,
                    "twelve_experiment_pool_limit": min(available, safe_limit),
                    "not_measured_beyond_limit": True,
                })
                following = [s for s in sizes if s > size]
                if following and intercept + slope * following[0] > remaining / 12:
                    record("learning_curve", "stop", "Next size would leave fewer than twelve experiments.")
                    break
                gain = result["mean_recall_at_10"] - previous["mean_recall_at_10"]
                noise = max(result["std_recall_at_10"], previous["std_recall_at_10"])
                if gain >= 0 and gain < max(.002, noise * .5):
                    record("learning_curve", "stop", "Curve flattened within fold noise.")
                    break
        record("recommendation", "evaluation", {
            "pool_rule": selected["rule"],
            "measured_pool_limit": curve[-1]["training_domains_per_fold"] if curve else None,
            "cv": "Deterministic domain-level 3 folds; for a larger pool, use the same fixed test reservoir and disjoint training indices.",
            "metric": "For every domain: number of distinct true trackers among ten highest-scoring tracker ids divided by its number of distinct observed true trackers; average equally across domains. Domains with no submitted guesses contribute zero. Labelled evaluation domains all have at least one observed tracker.",
            "masking": "If adding supervised graph features, remove ALL tracker edges of each validation domain from the label-source table, not only individual held-out pairs. Training-domain self-return walks must not reintroduce their own labels. Edges from labelled domains outside the locked population may remain.",
            "needed_link_graph": "For one-hop features, retain hyperlink edges with either endpoint in modelled or target domains. Use the full node lookup only to annotate those neighbours. Two-hop features require additional edges touching that first-hop frontier.",
            "completeness": "These sources cannot establish that missing tracker edges are true negatives or that target selection is label independent. The curve uses observed trackers as ground truth.",
        })
        self.report_ = pd.DataFrame(reports)
        return self

    def transform(self, X):
        return self.report_


def read_parquet(name, columns):
    return skrub.as_data_op(BASE + name).skb.apply_func(pd.read_parquet, columns=columns)


def build():
    labels = read_parquet("tracking_graph_train.parquet",
                          ["domain_id", "tracking_domain_id", "tracker_id"])
    domains = read_parquet("domains.parquet", ["domain_id", "domain"])
    links = read_parquet("link-graph.parquet",
                         ["source_domain_id", "target_domain_id"])
    targets = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(pd.read_csv, sep="\t")
    classification = skrub.as_data_op(BASE + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"])
    press = skrub.as_data_op(BASE + "freedom-of-the-press.csv").skb.apply_func(
        pd.read_csv, sep=None, engine="python",
        usecols=["tld", "freedom_of_the_press"])
    trackers = skrub.as_data_op(BASE + "trackers.tsv").skb.apply_func(pd.read_csv, sep="\t")

    label_counts = labels.groupby("domain_id")["tracker_id"].nunique().rename(
        "label_count").reset_index()
    keys = label_counts[["domain_id"]].skb.concat(
        [targets[["domain_id"]]], axis=0).drop_duplicates().reset_index(drop=True)
    host = classification["url"].astype("string").str.lower().str.replace(
        r"^[a-z][a-z0-9+.-]*://", "", regex=True).str.split("/").str[0].str.split(
            ":").str[0].str.replace(r"^www\.", "", regex=True)
    classified = classification.assign(host=host)[["host", "category"]].drop_duplicates(
        subset=["host"])
    domain_host = domains["domain"].astype("string").str.lower().str.replace(
        r"^www\.", "", regex=True)
    nodes = domains.assign(host=domain_host).merge(classified, on="host", how="left")
    nodes = nodes.assign(classified=nodes["category"].notna().astype("int8"))
    rows = keys.merge(nodes, on="domain_id", how="left").merge(
        label_counts, on="domain_id", how="left")
    rows = rows.merge(targets[["domain_id"]].assign(is_target=1),
                      on="domain_id", how="left")
    name = rows["domain"].fillna("").astype("string")
    rows = rows.assign(
        tld=name.str.rsplit(".", n=1).str[-1],
        hostname_length=name.str.len(),
        hostname_dots=name.str.count(r"\."),
        hostname_digits=name.str.count(r"[0-9]"),
        hostname_hyphens=name.str.count("-"),
    )

    # These graph features are label-free.
    outgoing = links[links["source_domain_id"].isin(keys["domain_id"])]
    incoming = links[links["target_domain_id"].isin(keys["domain_id"])]
    out_degrees = outgoing.groupby("source_domain_id")["target_domain_id"].nunique().rename(
        "out_degree").reset_index().rename(columns={"source_domain_id": "domain_id"})
    in_degrees = incoming.groupby("target_domain_id")["source_domain_id"].nunique().rename(
        "in_degree").reset_index().rename(columns={"target_domain_id": "domain_id"})
    class_nodes = nodes[nodes["classified"] == 1][["domain_id"]]
    out_class = outgoing[outgoing["target_domain_id"].isin(class_nodes["domain_id"])]
    in_class = incoming[incoming["source_domain_id"].isin(class_nodes["domain_id"])]
    out_coverage = out_class.groupby("source_domain_id")["target_domain_id"].nunique().rename(
        "out_classified_neighbors").reset_index().rename(columns={"source_domain_id": "domain_id"})
    in_coverage = in_class.groupby("target_domain_id")["source_domain_id"].nunique().rename(
        "in_classified_neighbors").reset_index().rename(columns={"target_domain_id": "domain_id"})
    for frame in [out_degrees, in_degrees, out_coverage, in_coverage]:
        rows = rows.merge(frame, on="domain_id", how="left")
    rows = rows.fillna({
        "out_degree": 0, "in_degree": 0,
        "out_classified_neighbors": 0, "in_classified_neighbors": 0,
        "classified": 0, "is_target": 0, "label_count": 0,
    })
    rows = rows.assign(
        out_classified_fraction=rows["out_classified_neighbors"] /
        rows["out_degree"].clip(lower=1),
        in_classified_fraction=rows["in_classified_neighbors"] /
        rows["in_degree"].clip(lower=1),
    )
    rows = rows.merge(press.drop_duplicates(subset=["tld"]), on="tld", how="left")
    report = rows.skb.apply(PopulationStudy(), y=labels)
    return {
        "population_and_curve": report,
        "label_source_shape": labels.shape,
        "domain_source_shape": domains.shape,
        "link_source_shape": links.shape,
        "target_source_shape": targets.shape,
        "classification_source_shape": classification.shape,
        "label_duplicate_count": labels.shape[0] - labels.drop_duplicates(
            subset=["domain_id", "tracker_id"]).shape[0],
        "target_duplicate_count": targets.shape[0] - targets.drop_duplicates(
            subset=["domain_id"]).shape[0],
        "outgoing_relevant_shape": outgoing.shape,
        "incoming_relevant_shape": incoming.shape,
        "tracker_frequency": labels.groupby("tracker_id")["domain_id"].nunique().rename(
            "domain_count").reset_index().sort_values("domain_count", ascending=False),
        "tracker_metadata": trackers,
        "tracker_count_distribution": label_counts.groupby("label_count").size().rename(
            "domains").reset_index(),
        "classified_preview": classified.head(12),
        "target_preview": rows[rows["is_target"] == 1].head(12),
    }
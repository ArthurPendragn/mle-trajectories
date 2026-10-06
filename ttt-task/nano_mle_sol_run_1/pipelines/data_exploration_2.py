import time
import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier

ROOT = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
RULES = [("all", "all", 0, 0)]
for k in [1, 2, 3, 5, 10]:
    RULES += [("at_most_%d" % k, "upper", k, 0),
              ("at_least_%d" % k, "lower", k, 0)]
for a, b in [(1, 2), (2, 3), (2, 5), (3, 10), (5, 10)]:
    RULES.append(("band_%d_%d" % (a, b), "band", a, b))
RULES += [("classified", "classified", 0, 0),
          ("unclassified", "unclassified", 0, 0),
          ("has_outgoing", "out", 0, 0),
          ("has_incoming", "in", 0, 0),
          ("has_both", "both", 0, 0),
          ("no_outgoing", "no_out", 0, 0)]

REPORT_COLUMNS = {
    "adversarial": [
        "rule", "sampled_eligible", "balanced_per_class",
        "auc_fold_1", "auc_fold_2", "auc_fold_3", "auc_mean", "auc_std",
    ],
    "curve": [
        "model", "training_domains_per_fold", "validation_domains_per_fold",
        "recall_fold_1", "recall_fold_2", "recall_fold_3", "recall_mean",
        "recall_std", "all_fold_fitted_feature_seconds",
        "fold_feature_fit_score_seconds", "shared_source_feature_seconds",
        "complete_experiment_seconds", "device",
    ],
    "population_diagnostics": [
        "selected_rule", "lowest_auc_rule", "lowest_auc", "selected_auc",
        "tie_tolerance", "sampled_selected_domains", "adversarial_features",
        "adversarial_model", "balanced_protocol", "shift_interpretation",
        "shared_source_feature_seconds",
    ],
    "recommendation": [
        "population_rule", "recommended_train_pool",
        "timing_extrapolated_training_limit", "reason", "cv", "metric",
        "supervised_feature_leakage", "external_labels", "needed_graph",
        "graph_label_safety", "unknowns", "graph_filter_population_sizes",
    ],
}


def mask_for(frame, mode, a, b):
    if mode == "all":
        return frame["label_count"] >= 1
    if mode == "upper":
        return frame["label_count"] <= a
    if mode == "lower":
        return frame["label_count"] >= a
    if mode == "band":
        return (frame["label_count"] >= a) & (frame["label_count"] <= b)
    if mode == "classified":
        return frame["classified"] == 1
    if mode == "unclassified":
        return frame["classified"] == 0
    if mode == "out":
        return frame["out_degree"] > 0
    if mode == "in":
        return frame["in_degree"] > 0
    if mode == "both":
        return (frame["in_degree"] > 0) & (frame["out_degree"] > 0)
    return frame["out_degree"] == 0


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 192), nn.ReLU(),
                                    nn.Linear(192, 355))

    def forward(self, x):
        return self.layers(x)


class PopulationStudy(TransformerMixin, BaseEstimator):
    def __init__(self, remaining_seconds=55000):
        self.remaining_seconds = remaining_seconds

    def fit(self, X, y=None):
        began = time.perf_counter()
        torch.set_num_threads(16)
        frame = X.copy()
        numeric = ["hostname_length", "hostname_dots", "hostname_digits",
                   "hostname_hyphens", "classified", "out_degree", "in_degree",
                   "out_classified_fraction", "in_classified_fraction"]
        categorical = ["tld", "category"]
        feature_names = numeric + categorical
        frame[numeric] = frame[numeric].fillna(0).astype(float)
        frame[categorical] = frame[categorical].fillna("__missing__").astype(str)
        labelled = frame[frame.is_target == 0].reset_index(drop=True)
        target = frame[frame.is_target == 1].reset_index(drop=True)
        n = min(15000, len(target))
        target_ad = target.sample(n=n, random_state=427)
        rows = []
        for name, mode, a, b in RULES:
            eligible = labelled.loc[mask_for(labelled, mode, a, b)]
            count = min(n, len(eligible))
            if count < 100:
                continue
            left = eligible.sample(n=count, random_state=427)
            right = target_ad.iloc[:count]
            data = pd.concat([left[feature_names], right[feature_names]],
                             ignore_index=True)
            labels = np.r_[np.zeros(count), np.ones(count)]
            scores = []
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=617).split(data, labels):
                model = CatBoostClassifier(
                    iterations=140, depth=5, learning_rate=0.08,
                    loss_function="Logloss", verbose=False,
                    random_seed=617, thread_count=16,
                    allow_writing_files=False)
                model.fit(data.iloc[tr], labels[tr], cat_features=categorical)
                scores.append(roc_auc_score(labels[te],
                                            model.predict_proba(data.iloc[te])[:, 1]))
            rows.append(dict(rule=name, sampled_eligible=len(eligible),
                             balanced_per_class=count, auc_fold_1=scores[0],
                             auc_fold_2=scores[1], auc_fold_3=scores[2],
                             auc_mean=np.mean(scores), auc_std=np.std(scores, ddof=1)))
        adversarial = pd.DataFrame(rows)
        best = adversarial.loc[adversarial.auc_mean.idxmin()]
        tolerance = max(float(best.auc_std), 0.002)
        tied = adversarial[adversarial.auc_mean <= best.auc_mean + tolerance]
        selected = tied.sort_values("sampled_eligible", ascending=False).iloc[0]
        definition = next(r for r in RULES if r[0] == selected.rule)
        pool = labelled.loc[mask_for(labelled, *definition[1:])].sample(
            frac=1, random_state=291).reset_index(drop=True)
        graph_seconds = max(0.0, (pd.Timestamp.now() - pd.Timestamp(
            frame["started"].iloc[0])).total_seconds()
                            - (time.perf_counter() - began))
        diagnostics = pd.DataFrame([{
            "selected_rule": selected.rule,
            "lowest_auc_rule": best.rule,
            "lowest_auc": best.auc_mean,
            "selected_auc": selected.auc_mean,
            "tie_tolerance": tolerance,
            "sampled_selected_domains": len(pool),
            "adversarial_features": ", ".join(feature_names),
            "adversarial_model": "CatBoost 140 trees depth 5",
            "balanced_protocol": "same seed 427; up to 15000 per class; stratified three folds seed 617",
            "shift_interpretation": "AUC above 0.55 is meaningful observable shift, not proof of matching populations",
            "shared_source_feature_seconds": graph_seconds,
        }])
        label_edges = y[["domain_id", "tracker_id"]].drop_duplicates()
        vsize = min(2000, max(100, len(pool) // 10))
        validation = pool.iloc[:3 * vsize]
        training = pool.iloc[3 * vsize:]
        sizes = [s for s in [10000, 40000, 120000] if s <= len(training)]
        if len(sizes) < 2:
            sizes = sorted(set([max(100, len(training) // 3), len(training)]))
        sizes = [s for s in sizes if s > 0]
        curve = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for size in sizes:
            if curve:
                extrapolated = graph_seconds + curve[-1]["fold_feature_fit_score_seconds"] * (
                    size / curve[-1]["training_domains_per_fold"])
                if extrapolated > self.remaining_seconds / 12:
                    break
            tick = time.perf_counter()
            fold_scores = []
            feature_seconds = []
            for fold in range(3):
                train = training.iloc[:size].reset_index(drop=True)
                valid = validation.iloc[fold * vsize:(fold + 1) * vsize].reset_index(drop=True)
                ft = time.perf_counter()
                char = CountVectorizer(analyzer="char", ngram_range=(2, 3),
                                       max_features=768, dtype=np.float32)
                train_char = char.fit_transform(train.domain.fillna("")).toarray()
                valid_char = char.transform(valid.domain.fillna("")).toarray()
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                    dtype=np.float32)
                train_cat = enc.fit_transform(train[categorical])
                valid_cat = enc.transform(valid[categorical])
                scaler = StandardScaler()
                train_num = scaler.fit_transform(np.log1p(train[numeric])).astype(np.float32)
                valid_num = scaler.transform(np.log1p(valid[numeric])).astype(np.float32)
                tx = np.concatenate([np.log1p(train_char), train_cat, train_num], axis=1)
                vx = np.concatenate([np.log1p(valid_char), valid_cat, valid_num], axis=1)
                train_lookup = pd.Series(np.arange(len(train)), index=train.domain_id)
                valid_lookup = pd.Series(np.arange(len(valid)), index=valid.domain_id)
                te = label_edges[label_edges.domain_id.isin(train.domain_id)]
                ve = label_edges[label_edges.domain_id.isin(valid.domain_id)]
                ty = np.zeros((len(train), 355), dtype=np.float32)
                vy = np.zeros((len(valid), 355), dtype=np.float32)
                ty[te.domain_id.map(train_lookup).to_numpy(),
                   te.tracker_id.to_numpy(dtype=int)] = 1
                vy[ve.domain_id.map(valid_lookup).to_numpy(),
                   ve.tracker_id.to_numpy(dtype=int)] = 1
                feature_seconds.append(time.perf_counter() - ft)
                torch.manual_seed(617 + fold)
                if device == "cuda":
                    torch.cuda.manual_seed_all(617 + fold)
                net = Network(tx.shape[1]).to(device)
                optimizer = torch.optim.AdamW(net.parameters(), lr=0.002,
                                               weight_decay=0.001)
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(tx),
                                                   torch.from_numpy(ty)),
                    batch_size=1024, shuffle=True)
                net.train()
                for epoch in range(6):
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        logits = net(bx)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, by, reduction="none")
                        loss = (loss.mean(dim=1) / by.sum(dim=1).clamp_min(1)).mean()
                        loss.backward()
                        optimizer.step()
                net.eval()
                predictions = []
                with torch.no_grad():
                    for offset in range(0, len(vx), 2048):
                        predictions.append(net(torch.from_numpy(
                            vx[offset:offset + 2048]).to(device)).cpu().numpy())
                score_matrix = np.concatenate(predictions)
                top = np.argpartition(score_matrix, -10, axis=1)[:, -10:]
                recovered = np.take_along_axis(vy, top, axis=1).sum(axis=1)
                fold_scores.append(float(np.mean(recovered / vy.sum(axis=1).clip(min=1))))
                del net, optimizer
            elapsed = time.perf_counter() - tick
            curve.append(dict(
                model="GPU MLP char ngrams + TLD/category + graph degrees; inverse-cardinality weighted BCE",
                training_domains_per_fold=size, validation_domains_per_fold=vsize,
                recall_fold_1=fold_scores[0], recall_fold_2=fold_scores[1],
                recall_fold_3=fold_scores[2], recall_mean=np.mean(fold_scores),
                recall_std=np.std(fold_scores, ddof=1),
                all_fold_fitted_feature_seconds=sum(feature_seconds),
                fold_feature_fit_score_seconds=elapsed,
                shared_source_feature_seconds=graph_seconds,
                complete_experiment_seconds=elapsed + graph_seconds,
                device=device))
            if len(curve) >= 2:
                gain = curve[-1]["recall_mean"] - curve[-2]["recall_mean"]
                if gain < 0.003:
                    break
        curve_table = pd.DataFrame(curve, columns=REPORT_COLUMNS["curve"])
        if curve:
            last = curve[-1]
            feasible = int(last["training_domains_per_fold"] *
                           max(0, self.remaining_seconds / 12 - graph_seconds) /
                           max(1, last["fold_feature_fit_score_seconds"]))
            recommend_size = min(len(training), feasible,
                                 last["training_domains_per_fold"])
        else:
            feasible = 0
            recommend_size = 0
        recommendation = pd.DataFrame([{
            "population_rule": selected.rule,
            "recommended_train_pool": recommend_size + 3 * vsize,
            "timing_extrapolated_training_limit": feasible,
            "reason": "Use measured largest size unless curve flattens; extrapolated limit is not a measured score. Small modular sample bounds this exploration.",
            "cv": "Three disjoint validation subsets; common training pool excludes all three subsets. Deterministic domain-level splits.",
            "metric": "Mean over every validation domain of recovered distinct true trackers / distinct true trackers; top 10 unique tracker ids; zero predictions contribute zero.",
            "supervised_feature_leakage": "No tracker labels used in any feature. Validation tracker labels used only to score. Hostname vectorizer, categorical encoder, scaler, and network fit on each training subset only.",
            "external_labels": "All labelled domains outside a future locked pool may supply auxiliary label features if provably disjoint; none used here.",
            "needed_graph": "Only outgoing edges whose source is a sampled or target domain and incoming edges whose target is a sampled or target domain. Neighbour category coverage needs classified domain ids, not tracker labels.",
            "graph_label_safety": "Future supervised graph features must exclude ALL locked-domain tracker edges, not merely direct validation rows; avoids self-return paths.",
            "unknowns": "Observed labels need not be complete; adversarial similarity does not prove target selection or conditional-label equivalence.",
            "graph_filter_population_sizes": "Graph and metadata filter retained counts are sample estimates; tracker-count rules have exact separate counts.",
        }])
        results = dict(adversarial=adversarial, curve=curve_table,
                       population_diagnostics=diagnostics,
                       recommendation=recommendation)
        reports = []
        for section, table in results.items():
            report = table.rename(
                columns={name: "report_" + name for name in table.columns})
            reports.append(report.assign(report_section=section))
        self.report_ = pd.concat(reports, ignore_index=True, sort=False)
        return self

    def transform(self, X):
        # skrub's dataframe transformer wrapper expects dataframe output aligned
        # with its input. Pad the compact report with empty rows, then remove
        # those rows downstream using the explicit section marker.
        report = self.report_.copy()
        if len(report) > len(X):
            raise ValueError("The report has more rows than the study input.")
        report.index = X.index[:len(report)]
        return report.reindex(X.index)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def report_section(study, section):
    columns = REPORT_COLUMNS[section]
    report_columns = ["report_" + name for name in columns]
    selected = study[
        study["report_section"].eq(section).fillna(False)
    ][report_columns]
    return selected.rename(
        columns={prefixed: original
                 for prefixed, original in zip(report_columns, columns)}
    ).reset_index(drop=True)


def build():
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"])
    counts = edges.groupby("domain_id").agg(label_count=("tracker_id", "nunique")).reset_index()
    target = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(pd.read_csv, sep="\t")
    sample = counts[(counts["domain_id"] % 40) == 7].assign(is_target=0)
    target_rows = target.merge(counts, on="domain_id", how="left").assign(is_target=1)
    sample = sample.skb.concat([target_rows], axis=0).reset_index(drop=True)
    started = skrub.as_data_op(None).skb.apply_func(pd.Timestamp.now)
    sample = sample.assign(started=started)
    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"])
    domains = domains[domains["domain_id"].isin(sample["domain_id"])]
    categories = skrub.as_data_op(ROOT + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"])
    categories = categories.assign(
        domain=categories["url"].str.lower().str.replace(
            r"^https?://", "", regex=True).str.split("/").str[0].str.replace(
                r"^www\.", "", regex=True))
    categories = categories[["domain", "category"]].drop_duplicates("domain")
    domain_all = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"])
    domain_all = domain_all.assign(
        host=domain_all["domain"].str.lower().str.replace(r"^www\.", "", regex=True))
    classified_ids = domain_all[domain_all["host"].isin(categories["domain"])][["domain_id"]]
    domains = domains.assign(
        host=domains["domain"].str.lower().str.replace(r"^www\.", "", regex=True))
    rows = sample.merge(domains, on="domain_id", how="left")
    rows = rows.merge(categories.rename(columns={"domain": "host"}), on="host", how="left")
    rows = rows.assign(
        classified=rows["category"].notna().astype("int8"),
        tld=rows["domain"].str.split(".").str[-1],
        hostname_length=rows["domain"].str.len(),
        hostname_dots=rows["domain"].str.count(r"\."),
        hostname_digits=rows["domain"].str.count(r"\d"),
        hostname_hyphens=rows["domain"].str.count("-"))
    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"])
    outgoing = links[links["source_domain_id"].isin(sample["domain_id"])]
    incoming = links[links["target_domain_id"].isin(sample["domain_id"])]
    outgoing = outgoing.assign(
        neighbour_classified=outgoing["target_domain_id"].isin(classified_ids["domain_id"]).astype("int8"))
    incoming = incoming.assign(
        neighbour_classified=incoming["source_domain_id"].isin(classified_ids["domain_id"]).astype("int8"))
    out_stats = outgoing.groupby("source_domain_id").agg(
        out_degree=("target_domain_id", "size"),
        out_classified_fraction=("neighbour_classified", "mean")).reset_index().rename(
            columns={"source_domain_id": "domain_id"})
    in_stats = incoming.groupby("target_domain_id").agg(
        in_degree=("source_domain_id", "size"),
        in_classified_fraction=("neighbour_classified", "mean")).reset_index().rename(
            columns={"target_domain_id": "domain_id"})
    rows = rows.merge(out_stats, on="domain_id", how="left").merge(
        in_stats, on="domain_id", how="left")
    rows = rows.assign(
        out_degree=rows["out_degree"].fillna(0),
        in_degree=rows["in_degree"].fillna(0))
    sampled_labels = edges[edges["domain_id"].isin(sample["domain_id"])]
    study = rows.skb.apply(PopulationStudy(), y=sampled_labels)
    size_tables = []
    for name, mode, a, b in RULES:
        if mode not in ["all", "upper", "lower", "band"]:
            continue
        selected = counts[mask_for(counts, mode, a, b)]
        table = selected[["domain_id"]].count().to_frame(name="retained_domains")
        table = table.assign(rule=name).reset_index(drop=True)
        size_tables.append(table)
    sizes = size_tables[0].skb.concat(size_tables[1:], axis=0).reset_index(drop=True)
    overlap = target.merge(counts, on="domain_id", how="inner")
    return {
        "adversarial_comparisons": report_section(
            study, "adversarial").merge(sizes, on="rule", how="left"),
        "learning_curve": report_section(study, "curve"),
        "population_diagnostics": report_section(study, "population_diagnostics"),
        "recommendation": report_section(study, "recommendation"),
        "exact_tracker_count_rule_sizes": sizes,
        "unique_labelled_domain_count": counts.shape,
        "full_target_overlap": overlap.shape,
        "sampled_population_shape": rows.shape,
        "outgoing_graph_slice_shape": outgoing.shape,
        "incoming_graph_slice_shape": incoming.shape,
        "target_observables": rows[rows["is_target"] == 1].head(12),
    }
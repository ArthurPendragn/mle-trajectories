import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import skrub

TRACKING_GRAPH_TRAIN_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet"
DOMAINS_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet"
LINK_GRAPH_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/link-graph.parquet"
TRACKERS_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/trackers.tsv"
URL_CLASSIFICATION_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/url-classification.csv"
FREEDOM_OF_THE_PRESS_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/freedom-of-the-press.csv"
TARGET_PATH = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/target.tsv"


class TableSummaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, table_name=""):
        self.table_name = table_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_rows = len(X)
        n_cols = len(X.columns)
        null_count = int(X.isna().sum().sum())
        cols_summary = ", ".join(f"{c}:{X[c].dtype}" for c in X.columns)
        sample_preview = str(X.head(2).to_dict(orient="records"))
        if len(sample_preview) > 300:
            sample_preview = sample_preview[:300] + "..."
        return pd.DataFrame([{
            "table_name": self.table_name,
            "num_rows": n_rows,
            "num_columns": n_cols,
            "total_nulls": null_count,
            "column_types": cols_summary,
            "sample_records": sample_preview,
        }])


class DomainObservablesAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_target = X["is_target"].values == 1
        domains_str = X["domain"].astype(str).fillna("")
        length = domains_str.str.len()
        dots = domains_str.str.count(r"\.")
        tlds = domains_str.str.split(".").str[-1].str.lower()

        in_deg = X["in_degree"].fillna(0)
        out_deg = X["out_degree"].fillna(0)
        total_deg = in_deg + out_deg

        target_ids = set(X.loc[is_target, "domain_id"])
        train_ids = set(X.loc[~is_target, "domain_id"])
        overlap_count = len(target_ids & train_ids)

        tc = X.loc[~is_target, "tracker_count"]

        def tld_summary(s):
            top = s.value_counts(normalize=True).head(5)
            return ", ".join(f".{k}: {v*100:.1f}%" for k, v in top.items())

        rows = [
            {"metric": "total_domain_count", "target_val": str(len(target_ids)), "train_val": str(len(train_ids)), "notes": "Unique domains in each dataset"},
            {"metric": "target_train_overlap", "target_val": str(overlap_count), "train_val": str(overlap_count), "notes": "Target domains present in train"},
            {"metric": "domain_length_mean", "target_val": f"{length[is_target].mean():.2f}", "train_val": f"{length[~is_target].mean():.2f}", "notes": "Mean hostname length"},
            {"metric": "domain_length_median", "target_val": f"{length[is_target].median():.1f}", "train_val": f"{length[~is_target].median():.1f}", "notes": "Median hostname length"},
            {"metric": "domain_length_p90", "target_val": f"{length[is_target].quantile(0.90):.1f}", "train_val": f"{length[~is_target].quantile(0.90):.1f}", "notes": "90th percentile length"},
            {"metric": "dots_count_mean", "target_val": f"{dots[is_target].mean():.2f}", "train_val": f"{dots[~is_target].mean():.2f}", "notes": "Mean number of dots in domain"},
            {"metric": "single_dot_domain_pct", "target_val": f"{(dots[is_target] == 1).mean() * 100:.1f}%", "train_val": f"{(dots[~is_target] == 1).mean() * 100:.1f}%", "notes": "Percentage second-level domains (e.g. foo.com)"},
            {"metric": "multi_dot_domain_pct", "target_val": f"{(dots[is_target] > 1).mean() * 100:.1f}%", "train_val": f"{(dots[~is_target] > 1).mean() * 100:.1f}%", "notes": "Percentage subdomains (e.g. sub.foo.com)"},
            {"metric": "link_graph_any_link_pct", "target_val": f"{(total_deg[is_target] > 0).mean() * 100:.1f}%", "train_val": f"{(total_deg[~is_target] > 0).mean() * 100:.1f}%", "notes": "Percentage of domains present in link graph"},
            {"metric": "link_graph_in_link_pct", "target_val": f"{(in_deg[is_target] > 0).mean() * 100:.1f}%", "train_val": f"{(in_deg[~is_target] > 0).mean() * 100:.1f}%", "notes": "Percentage with in_degree > 0"},
            {"metric": "link_graph_out_link_pct", "target_val": f"{(out_deg[is_target] > 0).mean() * 100:.1f}%", "train_val": f"{(out_deg[~is_target] > 0).mean() * 100:.1f}%", "notes": "Percentage with out_degree > 0"},
            {"metric": "in_degree_mean", "target_val": f"{in_deg[is_target].mean():.2f}", "train_val": f"{in_deg[~is_target].mean():.2f}", "notes": "Mean incoming links"},
            {"metric": "in_degree_median", "target_val": f"{in_deg[is_target].median():.1f}", "train_val": f"{in_deg[~is_target].median():.1f}", "notes": "Median incoming links"},
            {"metric": "in_degree_p90", "target_val": f"{in_deg[is_target].quantile(0.90):.1f}", "train_val": f"{in_deg[~is_target].quantile(0.90):.1f}", "notes": "90th percentile incoming links"},
            {"metric": "out_degree_mean", "target_val": f"{out_deg[is_target].mean():.2f}", "train_val": f"{out_deg[~is_target].mean():.2f}", "notes": "Mean outgoing links"},
            {"metric": "out_degree_median", "target_val": f"{out_deg[is_target].median():.1f}", "train_val": f"{out_deg[~is_target].median():.1f}", "notes": "Median outgoing links"},
            {"metric": "out_degree_p90", "target_val": f"{out_deg[is_target].quantile(0.90):.1f}", "train_val": f"{out_deg[~is_target].quantile(0.90):.1f}", "notes": "90th percentile outgoing links"},
            {"metric": "top_5_tlds", "target_val": tld_summary(tlds[is_target]), "train_val": tld_summary(tlds[~is_target]), "notes": "Most frequent TLDs"},
            {"metric": "train_tracker_count_mean", "target_val": "N/A (unobserved)", "train_val": f"{tc.mean():.2f}", "notes": "Mean trackers per domain in train"},
            {"metric": "train_tracker_count_median", "target_val": "N/A (unobserved)", "train_val": f"{tc.median():.1f}", "notes": "Median trackers per domain in train"},
            {"metric": "train_tracker_count_p25_p75", "target_val": "N/A (unobserved)", "train_val": f"{tc.quantile(0.25):.0f} - {tc.quantile(0.75):.0f}", "notes": "Interquartile range of trackers"},
            {"metric": "train_tracker_count_p90", "target_val": "N/A (unobserved)", "train_val": f"{tc.quantile(0.90):.0f}", "notes": "90th percentile trackers per domain"},
            {"metric": "train_tracker_count_max", "target_val": "N/A (unobserved)", "train_val": f"{tc.max():.0f}", "notes": "Max trackers observed on a domain"},
        ]
        return pd.DataFrame(rows)


class AdversarialPopulationCheck(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        domains_str = X["domain"].astype(str).fillna("")
        length = domains_str.str.len().values.astype(np.float32)
        dots = domains_str.str.count(r"\.").values.astype(np.float32)
        digits = domains_str.str.count(r"[0-9]").values.astype(np.float32)
        hyphen = domains_str.str.contains("-", regex=False).astype(np.float32).values

        in_deg_val = X["in_degree"].fillna(0).values.astype(np.float32)
        out_deg_val = X["out_degree"].fillna(0).values.astype(np.float32)
        total_deg_val = in_deg_val + out_deg_val

        in_deg_log = np.log1p(in_deg_val)
        out_deg_log = np.log1p(out_deg_val)
        total_deg_log = np.log1p(total_deg_val)

        has_in = (in_deg_val > 0).astype(np.float32)
        has_out = (out_deg_val > 0).astype(np.float32)
        has_link = (total_deg_val > 0).astype(np.float32)

        tlds = domains_str.str.split(".").str[-1].str.lower()
        tld_freq_map = tlds.value_counts(normalize=True).to_dict()
        tld_freq = tlds.map(tld_freq_map).fillna(0.0).values.astype(np.float32)

        top_tlds = list(tlds.value_counts().head(10).index)
        tld_ohe = [
            (tlds == t).astype(np.float32).values
            for t in top_tlds
        ]

        feature_cols = [
            length, dots, digits, hyphen,
            in_deg_log, out_deg_log, total_deg_log,
            has_in, has_out, has_link, tld_freq,
        ] + tld_ohe

        features = np.column_stack(feature_cols)

        is_target = X["is_target"].values == 1
        target_idx = np.where(is_target)[0]
        train_idx = np.where(~is_target)[0]

        tc_train = X.loc[~is_target, "tracker_count"].values
        hl_train = has_link[train_idx]
        in_train = in_deg_val[train_idx]

        rules = [
            ("all_train", "All candidate training domains (tracker_count >= 1)",
             np.ones(len(train_idx), dtype=bool)),
            ("tc_ge_2", "tracker_count >= 2", tc_train >= 2),
            ("tc_ge_3", "tracker_count >= 3", tc_train >= 3),
            ("tc_ge_5", "tracker_count >= 5", tc_train >= 5),
            ("tc_ge_10", "tracker_count >= 10", tc_train >= 10),
            ("tc_le_20", "tracker_count <= 20", tc_train <= 20),
            ("tc_le_15", "tracker_count <= 15", tc_train <= 15),
            ("tc_le_10", "tracker_count <= 10", tc_train <= 10),
            ("tc_le_5", "tracker_count <= 5", tc_train <= 5),
            ("tc_1_to_20", "1 <= tracker_count <= 20", (tc_train >= 1) & (tc_train <= 20)),
            ("tc_2_to_20", "2 <= tracker_count <= 20", (tc_train >= 2) & (tc_train <= 20)),
            ("tc_2_to_15", "2 <= tracker_count <= 15", (tc_train >= 2) & (tc_train <= 15)),
            ("tc_2_to_10", "2 <= tracker_count <= 10", (tc_train >= 2) & (tc_train <= 10)),
            ("tc_3_to_20", "3 <= tracker_count <= 20", (tc_train >= 3) & (tc_train <= 20)),
            ("has_link", "has in_degree > 0 or out_degree > 0", hl_train > 0),
            ("in_link", "has in_degree > 0", in_train > 0),
            ("link_and_tc_ge_2", "has_link & tracker_count >= 2", (hl_train > 0) & (tc_train >= 2)),
            ("link_and_tc_2_to_20", "has_link & 2 <= tracker_count <= 20", (hl_train > 0) & (tc_train >= 2) & (tc_train <= 20)),
        ]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        records = []

        for rule_name, desc, train_mask in rules:
            selected_train_idx = train_idx[train_mask]
            n_train_sel = len(selected_train_idx)
            n_target_sel = len(target_idx)

            if n_train_sel < 100:
                continue

            max_per_class = 20000
            if n_train_sel > max_per_class:
                rng = np.random.default_rng(42)
                eval_train_idx = rng.choice(selected_train_idx, size=max_per_class, replace=False)
            else:
                eval_train_idx = selected_train_idx

            if n_target_sel > max_per_class:
                rng = np.random.default_rng(42)
                eval_target_idx = rng.choice(target_idx, size=max_per_class, replace=False)
            else:
                eval_target_idx = target_idx

            comb_idx = np.concatenate([eval_train_idx, eval_target_idx])
            comb_y = np.concatenate([np.zeros(len(eval_train_idx), dtype=int), np.ones(len(eval_target_idx), dtype=int)])
            comb_X = features[comb_idx]

            fold_aucs = []
            for tr_fold, te_fold in skf.split(comb_X, comb_y):
                clf = HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=25, random_state=42)
                clf.fit(comb_X[tr_fold], comb_y[tr_fold])
                probs = clf.predict_proba(comb_X[te_fold])[:, 1]
                fold_aucs.append(float(roc_auc_score(comb_y[te_fold], probs)))

            records.append({
                "rule_name": rule_name,
                "description": desc,
                "n_train_total": n_train_sel,
                "n_target_total": n_target_sel,
                "train_retention_pct": round(n_train_sel / len(train_idx) * 100, 1),
                "adv_auc_mean": round(float(np.mean(fold_aucs)), 4),
                "adv_auc_std": round(float(np.std(fold_aucs)), 4),
                "adv_auc_min": round(float(np.min(fold_aucs)), 4),
                "adv_auc_max": round(float(np.max(fold_aucs)), 4),
            })

        df_out = pd.DataFrame(records).sort_values("adv_auc_mean", ascending=True).reset_index(drop=True)
        return df_out


class TrackerDistributionAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_unique_domains = X["domain_id"].nunique()

        tracker_counts = X.groupby(["tracker_id", "company", "category"], as_index=False)["domain_id"].count().rename(columns={"domain_id": "edge_count"})
        tracker_counts["domain_coverage_pct"] = (tracker_counts["edge_count"] / n_unique_domains * 100).round(2)
        top_trackers = tracker_counts.sort_values(by="edge_count", ascending=False).head(20)

        records = []
        for _, r in top_trackers.iterrows():
            records.append({
                "tracker_id": int(r["tracker_id"]),
                "company": str(r["company"]),
                "category": str(r["category"]),
                "edge_count": int(r["edge_count"]),
                "domain_coverage_pct": float(r["domain_coverage_pct"]),
            })
        return pd.DataFrame(records)


class UrlClassificationAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_rows = len(X)
        n_unique_urls = X["url"].nunique() if "url" in X.columns else 0
        n_unique_cats = X["category"].nunique() if "category" in X.columns else 0
        cat_counts = X["category"].value_counts().head(10).to_dict() if "category" in X.columns else {}
        sample_urls = list(X["url"].head(3)) if "url" in X.columns else []

        return pd.DataFrame([{
            "total_rows": n_rows,
            "unique_urls": n_unique_urls,
            "unique_categories": n_unique_cats,
            "top_categories": str(cat_counts),
            "sample_urls": str(sample_urls),
        }])


class LearningCurveAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        domain_df = X.groupby("domain_id").agg({
            "domain": "first",
            "tracker_id": lambda s: set(s),
            "out_degree": "first",
            "in_degree": "first",
        }).reset_index()

        total_avail_domains = len(domain_df)
        sizes_to_test = [1000, 3000, 10000, 25000, 50000]
        sizes_to_test = [s for s in sizes_to_test if s <= total_avail_domains]
        if not sizes_to_test or sizes_to_test[-1] < total_avail_domains:
            sizes_to_test.append(total_avail_domains)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_domain_strings = domain_df["domain"].astype(str).fillna("unknown").values
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=256)
        all_tfidf = vectorizer.fit_transform(all_domain_strings).toarray().astype(np.float32)

        in_deg = np.log1p(domain_df["in_degree"].fillna(0).values.astype(np.float32)[:, None])
        out_deg = np.log1p(domain_df["out_degree"].fillna(0).values.astype(np.float32)[:, None])
        all_X_feats = np.hstack([all_tfidf, in_deg, out_deg])
        n_features = all_X_feats.shape[1]

        n_trackers = 355
        all_Y = np.zeros((total_avail_domains, n_trackers), dtype=np.float32)
        true_sets = domain_df["tracker_id"].values
        for i, t_set in enumerate(true_sets):
            valid_t = [t for t in t_set if 0 <= t < n_trackers]
            if valid_t:
                all_Y[i, valid_t] = 1.0

        results = []
        n_splits = 3

        for pool_size in sizes_to_test:
            sub_indices = np.arange(pool_size)
            sub_X = all_X_feats[sub_indices]
            sub_Y = all_Y[sub_indices]
            sub_true = [true_sets[idx] for idx in sub_indices]

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_recalls = []
            t0 = time.time()

            for train_idx, val_idx in kf.split(sub_X):
                X_tr = torch.from_numpy(sub_X[train_idx])
                Y_tr = torch.from_numpy(sub_Y[train_idx])
                X_va = torch.from_numpy(sub_X[val_idx])

                dataset = TensorDataset(X_tr, Y_tr)
                loader = DataLoader(dataset, batch_size=256, shuffle=True)

                net = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_trackers)
                ).to(device)

                optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
                criterion = nn.BCEWithLogitsLoss()

                net.train()
                for _ in range(6):
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad()
                        out = net(batch_x)
                        loss = criterion(out, batch_y)
                        loss.backward()
                        optimizer.step()

                net.eval()
                with torch.no_grad():
                    X_va_dev = X_va.to(device)
                    logits = net(X_va_dev)
                    _, top10_indices = torch.topk(logits, k=10, dim=1)
                    top10_np = top10_indices.cpu().numpy()

                val_recalls = []
                for j, row_preds in enumerate(top10_np):
                    true_t = sub_true[val_idx[j]]
                    if len(true_t) > 0:
                        rec = len(set(row_preds) & true_t) / len(true_t)
                    else:
                        rec = 0.0
                    val_recalls.append(rec)
                fold_recalls.append(float(np.mean(val_recalls)))

            elapsed = time.time() - t0
            time_per_fold = elapsed / n_splits
            mean_recall = float(np.mean(fold_recalls))
            std_recall = float(np.std(fold_recalls))
            proj_5fold_pipeline_s = round(time_per_fold * 5, 2)
            est_experiments = int(30000 / max(proj_5fold_pipeline_s, 1))

            results.append({
                "pool_size": pool_size,
                "recall_at_10_mean": round(mean_recall, 4),
                "recall_at_10_std": round(std_recall, 4),
                "time_per_fold_s": round(time_per_fold, 2),
                "total_eval_time_s": round(elapsed, 2),
                "proj_5fold_pipeline_s": proj_5fold_pipeline_s,
                "est_experiments_in_budget": est_experiments,
                "model_description": "PyTorch 2-layer MLP (TF-IDF char n-grams + degree -> 128 -> 355 logits, BCEWithLogitsLoss)",
            })

        return pd.DataFrame(results)


def build():
    tracking_raw = skrub.as_data_op(TRACKING_GRAPH_TRAIN_PATH).skb.apply_func(pd.read_parquet)
    domains_raw = skrub.as_data_op(DOMAINS_PATH).skb.apply_func(pd.read_parquet)
    link_raw = skrub.as_data_op(LINK_GRAPH_PATH).skb.apply_func(pd.read_parquet)
    trackers_raw = skrub.as_data_op(TRACKERS_PATH).skb.apply_func(pd.read_csv, sep="\t")
    url_raw = skrub.as_data_op(URL_CLASSIFICATION_PATH).skb.apply_func(pd.read_csv)
    freedom_raw = skrub.as_data_op(FREEDOM_OF_THE_PRESS_PATH).skb.apply_func(pd.read_csv)
    target_raw = skrub.as_data_op(TARGET_PATH).skb.apply_func(pd.read_csv, sep="\t")

    sum_tracking = tracking_raw.skb.apply(TableSummaryTransformer(table_name="tracking_graph_train"))
    sum_target = target_raw.skb.apply(TableSummaryTransformer(table_name="target"))
    sum_domains = domains_raw.skb.apply(TableSummaryTransformer(table_name="domains"))
    sum_link = link_raw.skb.apply(TableSummaryTransformer(table_name="link_graph"))
    sum_trackers = trackers_raw.skb.apply(TableSummaryTransformer(table_name="trackers"))
    sum_url = url_raw.skb.apply(TableSummaryTransformer(table_name="url_classification"))
    sum_freedom = freedom_raw.skb.apply(TableSummaryTransformer(table_name="freedom_of_the_press"))
    dataset_overview_op = sum_tracking.skb.concat(
        [sum_target, sum_domains, sum_link, sum_trackers, sum_url, sum_freedom], axis=0
    ).reset_index(drop=True)

    out_deg = link_raw[["source_domain_id", "target_domain_id"]].groupby(
        "source_domain_id", as_index=False
    ).count().rename(columns={"source_domain_id": "domain_id", "target_domain_id": "out_degree"})

    in_deg = link_raw[["target_domain_id", "source_domain_id"]].groupby(
        "target_domain_id", as_index=False
    ).count().rename(columns={"target_domain_id": "domain_id", "source_domain_id": "in_degree"})

    train_counts = tracking_raw[["domain_id", "tracker_id"]].groupby(
        "domain_id", as_index=False
    ).count().rename(columns={"tracker_id": "tracker_count"})

    target_rows = target_raw[["domain_id"]].assign(is_target=1, tracker_count=-1)[["domain_id", "is_target", "tracker_count"]]
    train_rows = train_counts[["domain_id", "tracker_count"]].assign(is_target=0)[["domain_id", "is_target", "tracker_count"]]

    all_domains = target_rows.skb.concat([train_rows], axis=0).reset_index(drop=True)
    all_domains = all_domains.merge(domains_raw[["domain_id", "domain"]], on="domain_id", how="left")
    all_domains = all_domains.merge(out_deg, on="domain_id", how="left")
    all_domains = all_domains.merge(in_deg, on="domain_id", how="left")

    domain_observables_op = all_domains.skb.apply(DomainObservablesAnalyzer())
    adversarial_op = all_domains.skb.apply(AdversarialPopulationCheck())

    tracking_trackers = tracking_raw.merge(trackers_raw, on="tracker_id", how="left")
    tracker_dist_op = tracking_trackers.skb.apply(TrackerDistributionAnalyzer())

    url_summary_op = url_raw.skb.apply(UrlClassificationAnalyzer())

    tracking_with_domains = tracking_raw.merge(domains_raw[["domain_id", "domain"]], on="domain_id", how="left")
    tracking_with_domains = tracking_with_domains.merge(out_deg, on="domain_id", how="left")
    tracking_with_domains = tracking_with_domains.merge(in_deg, on="domain_id", how="left")
    learning_curve_op = tracking_with_domains.skb.apply(LearningCurveAnalyzer())

    return {
        "dataset_overview": dataset_overview_op,
        "domain_observables": domain_observables_op,
        "adversarial_selection_rules": adversarial_op,
        "tracker_distributions": tracker_dist_op,
        "url_classification_summary": url_summary_op,
        "learning_curve": learning_curve_op,
    }
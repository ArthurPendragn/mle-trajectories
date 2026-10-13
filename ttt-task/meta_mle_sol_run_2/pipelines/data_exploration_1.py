import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import hstack, csr_matrix

ROOT = "gs://mle-trajectories-data/ttt-task/input/"
NUMERIC = ["host_length", "host_dots", "host_hyphens", "out_degree", "in_degree"]
CATEGORICAL = ["tld", "category"]


class PopulationStudy(TransformerMixin, BaseEstimator):
    def __init__(self, adversarial_size=6000):
        self.adversarial_size = adversarial_size

    def fit(self, X, y=None):
        start = time.perf_counter()
        data = X.copy()
        train = data[data["population"] == "labelled"].reset_index(drop=True)
        target = data[data["population"] == "target"].reset_index(drop=True)
        counts = train["tracker_count"].to_numpy()
        reports = []
        rules = {"all": np.ones(len(train), dtype=bool)}
        for threshold in [1, 2, 3, 5, 10, 20]:
            rules["at_least_%d" % threshold] = counts >= threshold
            rules["at_most_%d" % threshold] = counts <= threshold
        for lo, hi in [(1, 5), (2, 10), (5, 20), (2, 20)]:
            rules["band_%d_%d" % (lo, hi)] = (counts >= lo) & (counts <= hi)
        rules["has_links"] = (train["out_degree"] + train["in_degree"]).to_numpy() > 0
        rules["has_category"] = train["category"].ne("__missing__").to_numpy()
        for name, mask in rules.items():
            candidate = train.loc[mask]
            n = min(self.adversarial_size, len(candidate), len(target))
            if n < 50:
                continue
            a = candidate.sample(n=n, random_state=73)
            b = target.sample(n=n, random_state=73)
            obs = pd.concat([a, b], ignore_index=True)
            labels = np.r_[np.zeros(n), np.ones(n)]
            aucs = []
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=19).split(obs, labels):
                encoder = ColumnTransformer([
                    ("num", SimpleImputer(strategy="median"), NUMERIC),
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)
                ])
                ftr = encoder.fit_transform(obs.iloc[tr])
                fte = encoder.transform(obs.iloc[te])
                model = HistGradientBoostingClassifier(
                    max_iter=100, max_leaf_nodes=15, l2_regularization=5,
                    learning_rate=0.08, random_state=19
                )
                model.fit(ftr, labels[tr])
                aucs.append(roc_auc_score(labels[te], model.predict_proba(fte)[:, 1]))
            reports.append({
                "study": "population", "rule": name, "rows": len(candidate),
                "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "auc_folds": str(aucs), "mean_trackers": candidate["tracker_count"].mean(),
                "median_trackers": candidate["tracker_count"].median(),
                "p90_trackers": candidate["tracker_count"].quantile(0.9),
                "fraction_singleton": candidate["tracker_count"].eq(1).mean(),
                "comparison_rows_per_population": n
            })
        hash_model = HashingVectorizer(
            analyzer="char", ngram_range=(3, 5), n_features=8192,
            alternate_sign=False, norm="l2"
        )
        rng = np.random.RandomState(41)
        order = rng.permutation(len(train))
        sizes = sorted(set(min(n, len(train)) for n in [2000, 8000, 24000]))
        last_time = None
        last_size = None
        for size in sizes:
            if size < 100:
                continue
            if last_time is not None:
                projected = last_time * (size / last_size) ** 1.5
                if projected > 1800:
                    reports.append({"study": "learning_curve", "rule": "unmeasured_time_ceiling",
                                    "rows": size, "projected_seconds": projected})
                    break
            pool = train.iloc[order[:size]].reset_index(drop=True)
            began = time.perf_counter()
            recalls = []
            label_matrix = np.zeros((len(pool), 355), dtype=np.float32)
            for i, values in enumerate(pool["tracker_ids"]):
                label_matrix[i, np.asarray(values, dtype=int)] = 1
            for tr, te in KFold(2, shuffle=True, random_state=31).split(pool):
                prep = ColumnTransformer([
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ("num", make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler()), NUMERIC)
                ], sparse_threshold=1.0)
                a = prep.fit_transform(pool.iloc[tr])
                b = prep.transform(pool.iloc[te])
                a = hstack([csr_matrix(a), hash_model.transform(pool.iloc[tr]["domain"])], format="csr")
                b = hstack([csr_matrix(b), hash_model.transform(pool.iloc[te]["domain"])], format="csr")
                model = Ridge(alpha=20.0, solver="lsqr", max_iter=30, tol=0.005)
                model.fit(a, label_matrix[tr])
                scores = model.predict(b)
                top = np.argpartition(scores, -10, axis=1)[:, -10:]
                hits = np.take_along_axis(label_matrix[te], top, axis=1).sum(axis=1)
                recalls.append(float(np.mean(hits / label_matrix[te].sum(axis=1))))
            elapsed = time.perf_counter() - began
            reports.append({
                "study": "learning_curve", "rule": "hashed_hostname_metadata_ridge",
                "rows": size, "recall_at_10": np.mean(recalls),
                "recall_std": np.std(recalls), "recall_folds": str(recalls),
                "seconds_all_folds": elapsed,
                "features": "hostname char 3-5 grams, tld, category, link degrees, hostname shape"
            })
            last_time, last_size = elapsed, size
        reports.append({"study": "summary", "rule": "overall",
                        "labelled_rows": len(train), "target_rows": len(target),
                        "seconds": time.perf_counter() - start})
        self.report_ = pd.DataFrame(reports)
        return self

    def transform(self, X):
        return self.report_.copy()


def build():
    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(pd.read_parquet)
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id", "tracking_domain_id"])
    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"])
    target = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(pd.read_csv, sep="\t")
    categories = skrub.as_data_op(ROOT + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"])
    trackers = skrub.as_data_op(ROOT + "trackers.tsv").skb.apply_func(pd.read_csv, sep="\t")
    press = skrub.as_data_op(ROOT + "freedom-of-the-press.csv").skb.apply_func(pd.read_csv)

    labels = edges.groupby("domain_id").agg(
        tracker_ids=("tracker_id", list),
        tracker_count=("tracker_id", "nunique")
    ).reset_index()
    out_degree = links.groupby("source_domain_id").size().rename("out_degree").reset_index().rename(
        columns={"source_domain_id": "domain_id"})
    in_degree = links.groupby("target_domain_id").size().rename("in_degree").reset_index().rename(
        columns={"target_domain_id": "domain_id"})

    categories = categories.assign(domain=categories["url"].astype("string").str.lower()
                                   .str.replace(r"^https?://", "", regex=True)
                                   .str.split("/").str[0]
                                   .str.replace(r"^www\.", "", regex=True))
    categories = categories[["domain", "category"]].drop_duplicates("domain")
    hosts = domains.assign(domain=domains["domain"].astype("string").str.lower()
                           .str.replace(r"^www\.", "", regex=True))
    hosts = hosts.assign(
        tld=hosts["domain"].str.split(".").str[-1],
        host_length=hosts["domain"].str.len(),
        host_dots=hosts["domain"].str.count(r"\."),
        host_hyphens=hosts["domain"].str.count("-")
    )
    hosts = hosts.merge(categories, on="domain", how="left")
    hosts = hosts.merge(out_degree, on="domain_id", how="left").merge(in_degree, on="domain_id", how="left")
    hosts = hosts.fillna({"out_degree": 0, "in_degree": 0, "category": "__missing__", "domain": "", "tld": ""})
    labelled = labels.merge(hosts, on="domain_id", how="left").assign(population="labelled")
    predicted = target.merge(hosts, on="domain_id", how="left").assign(population="target")
    study_rows = labelled.skb.concat([predicted], axis=0).reset_index(drop=True)
    report = study_rows.skb.apply(PopulationStudy()).skb.set_name("population_and_learning_curve")
    overlap = labels[["domain_id"]].merge(target, on="domain_id", how="inner")
    return {
        "study": report,
        "domain_schema": domains.dtypes,
        "tracking_schema": edges.dtypes,
        "link_schema": links.dtypes,
        "domain_count": domains.shape[0],
        "tracking_edge_count": edges.shape[0],
        "link_edge_count": links.shape[0],
        "labelled_domain_count": labels.shape[0],
        "target_count": target.shape[0],
        "target_label_overlap": overlap.shape[0],
        "tracker_counts": labels["tracker_count"].describe(),
        "tracker_frequency": edges.groupby("tracker_id").size(),
        "target_observables": predicted[NUMERIC].describe(),
        "labelled_observables": labelled[NUMERIC].describe(),
        "target_tlds": predicted["tld"].value_counts().head(30),
        "labelled_tlds": labelled["tld"].value_counts().head(30),
        "trackers_preview": trackers.head(10),
        "press_preview": press.head(10),
        "category_count": categories.shape[0]
    }
import time
import numpy as np
import pandas as pd
import skrub
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier

ROOT = "gs://mle-trajectories-data/ttt-task/input/"

class EvidenceStudy(TransformerMixin, BaseEstimator):
    def __init__(self, sizes=(10000, 40000, 120000)):
        self.sizes = sizes

    def fit(self, X, y=None):
        self.report_ = self._study(X)
        return self

    def transform(self, X):
        return self.report_

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    def _study(self, data):
        torch.set_num_threads(16)
        rng = np.random.default_rng(42)
        population = data["population"]
        targets = data["targets"]
        candidates = data["candidates"]
        labels = data["labels"]
        columns = ["tld", "host_length", "host_dots", "host_hyphens",
                   "out_degree", "in_degree", "freedom_of_the_press"]
        records = []
        rules = {"all": np.ones(len(population), dtype=bool)}
        n = population["n_trackers"].to_numpy()
        for k in [1, 2, 3, 5, 10, 20]:
            rules["at_least_" + str(k)] = n >= k
            rules["at_most_" + str(k)] = n <= k
        for lo, hi in [(2, 3), (2, 5), (2, 10), (3, 5), (1, 3)]:
            rules[f"band_{lo}_{hi}"] = (n >= lo) & (n <= hi)
        rules["has_outgoing_links"] = population["out_degree"].to_numpy() > 0
        rules["has_incoming_links"] = population["in_degree"].to_numpy() > 0
        rules["both_link_directions"] = (
            rules["has_outgoing_links"] & rules["has_incoming_links"]
        )
        target_sample = targets.sample(n=min(6000, len(targets)), random_state=42)
        for name, mask in rules.items():
            idx = np.flatnonzero(mask)
            chosen = rng.choice(idx, size=min(6000, len(idx)), replace=False)
            a = population.iloc[chosen][columns].copy()
            b = target_sample[columns].iloc[:len(a)].copy()
            frame = pd.concat([a, b], ignore_index=True)
            frame["tld"] = frame["tld"].fillna("unknown").astype(str)
            for col in columns[1:]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            z = np.r_[np.zeros(len(a)), np.ones(len(b))]
            scores = []
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(frame, z):
                model = CatBoostClassifier(iterations=120, depth=5, learning_rate=0.08,
                                           verbose=False, thread_count=16,
                                           random_seed=42, allow_writing_files=False)
                model.fit(frame.iloc[tr], z[tr], cat_features=["tld"])
                scores.append(roc_auc_score(z[te], model.predict_proba(frame.iloc[te])[:, 1]))
            records.append(dict(study="population", rule=name, rows=len(idx),
                                auc=np.mean(scores), auc_std=np.std(scores),
                                fold_scores=str(scores)))
        del rules

        candidates = candidates.reset_index(drop=True)
        lookup = labels.set_index("domain_id")["tracker_id"]
        yy = np.zeros((len(candidates), 355), dtype=np.float32)
        for i, domain in enumerate(candidates["domain_id"]):
            ids = lookup.loc[domain]
            yy[i, np.asarray(ids, dtype=int)] = 1
        eligible = np.flatnonzero(candidates["n_trackers"].to_numpy() >= 2)
        test_ids = rng.choice(eligible, size=min(6000, len(eligible) // 4), replace=False)
        test_set = set(test_ids.tolist())
        train_all = np.asarray([i for i in rng.permutation(len(candidates))
                                if i not in test_set])
        train_two = train_all[candidates.iloc[train_all]["n_trackers"].to_numpy() >= 2]
        test_folds = np.array_split(test_ids, 3)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def_features = columns[1:]
        for size in self.sizes:
            for rule, pool in [("all", train_all), ("at_least_2", train_two)]:
                if len(pool) < size:
                    continue
                scores, elapsed, feature_times = [], [], []
                for fold, test in enumerate(test_folds):
                    start = time.perf_counter()
                    train = pool[:size]
                    a = candidates.iloc[train]
                    b = candidates.iloc[test]
                    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                            min_frequency=20, dtype=np.float32)
                    cat_a = encoder.fit_transform(a[["tld"]].fillna("unknown").astype(str))
                    cat_b = encoder.transform(b[["tld"]].fillna("unknown").astype(str))
                    hasher = HashingVectorizer(analyzer="char", ngram_range=(3, 5),
                                              n_features=512, dtype=np.float32,
                                              alternate_sign=False)
                    text_a = hasher.transform(a["domain"].fillna("").astype(str)).toarray()
                    text_b = hasher.transform(b["domain"].fillna("").astype(str)).toarray()
                    num_a = np.log1p(a[def_features].fillna(0).to_numpy(dtype=np.float32).clip(0))
                    num_b = np.log1p(b[def_features].fillna(0).to_numpy(dtype=np.float32).clip(0))
                    mean, std = num_a.mean(0), num_a.std(0) + 0.01
                    fa = np.concatenate([text_a, cat_a, (num_a - mean) / std], axis=1)
                    fb = np.concatenate([text_b, cat_b, (num_b - mean) / std], axis=1)
                    feature_times.append(time.perf_counter() - start)
                    torch.manual_seed(42 + fold)
                    net = torch.nn.Sequential(
                        torch.nn.Linear(fa.shape[1], 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(256, 355)
                    ).to(device)
                    ta = torch.as_tensor(fa, device=device)
                    ty = torch.as_tensor(yy[train], device=device)
                    weights = 1 / ty.sum(1).clamp(min=1)
                    prior = np.clip(yy[train].mean(0), 1e-5, 1 - 1e-5)
                    with torch.no_grad():
                        net[-1].bias.copy_(torch.as_tensor(np.log(prior / (1 - prior)), device=device))
                    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.001)
                    for epoch in range(12):
                        net.train()
                        order = torch.randperm(len(train), device=device)
                        for batch in order.split(1024):
                            opt.zero_grad()
                            logits = net(ta[batch])
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                logits, ty[batch], reduction="none"
                            )
                            loss = (loss * weights[batch, None]).mean()
                            loss.backward()
                            opt.step()
                    net.eval()
                    with torch.no_grad():
                        outputs = net(torch.as_tensor(fb, device=device)).cpu().numpy()
                    top = np.argpartition(outputs, -10, axis=1)[:, -10:]
                    hits = np.take_along_axis(yy[test], top, axis=1).sum(1)
                    scores.append(float(np.mean(hits / yy[test].sum(1))))
                    elapsed.append(time.perf_counter() - start)
                    del net, ta, ty
                records.append(dict(
                    study="learning_curve", rule=rule, rows=size,
                    model="hostname_hash_TLD_numeric_256_MLP_weighted_BCE",
                    recall_at_10=np.mean(scores), recall_std=np.std(scores),
                    fold_scores=str(scores), seconds_all_folds=sum(elapsed),
                    feature_seconds_all_folds=sum(feature_times),
                    timing_scope="fold-specific encoding, training and scoring; shared source reads excluded"
                ))
        curve = [r for r in records if r["study"] == "learning_curve"]
        if curve:
            largest = max(curve, key=lambda r: r["rows"])
            # Reserve at least twelve experiments, including a conservative allowance
            # for recorded-source graph construction on each experiment.
            limit = int(largest["rows"] * (80000 / 12) /
                        (largest["seconds_all_folds"] + 600))
            records.append(dict(study="sizing", rule="extrapolated_ceiling",
                                rows=limit, measured_max=largest["rows"],
                                source_graph_allowance_seconds=600,
                                note="Ceiling, not a target; choose measured plateau size if gains flatten."))
        records.append(dict(
            study="graph_scope", rule="leakage",
            note="Degree features need both link columns only. Later neighbor-label features should first filter links touching modelled or target ids, then join labels only from provably disjoint domains or fold-training marked y. Exclude self-return walks. Fixed test domains here are disjoint from every learning-curve training pool."
        ))
        return pd.DataFrame(records)

def build():
    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(pd.read_parquet)
    edges = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"])
    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"])
    target = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(pd.read_csv, sep="\t")
    press = skrub.as_data_op(ROOT + "freedom-of-the-press.csv").skb.apply_func(pd.read_csv, sep="\t")
    counts = edges.groupby("domain_id").size().rename("n_trackers").reset_index()
    outgoing = links.groupby("source_domain_id").size().rename("out_degree").reset_index()
    outgoing = outgoing.rename(columns={"source_domain_id": "domain_id"})
    incoming = links.groupby("target_domain_id").size().rename("in_degree").reset_index()
    incoming = incoming.rename(columns={"target_domain_id": "domain_id"})
    host = domains.assign(
        tld=domains["domain"].str.rsplit(".", n=1).str[-1],
        host_length=domains["domain"].str.len(),
        host_dots=domains["domain"].str.count(r"\."),
        host_hyphens=domains["domain"].str.count("-")
    )
    population = counts.merge(host, on="domain_id", how="left")
    population = population.merge(outgoing, on="domain_id", how="left")
    population = population.merge(incoming, on="domain_id", how="left")
    population = population.merge(press[["tld", "freedom_of_the_press"]], on="tld", how="left")
    targets = target.merge(host, on="domain_id", how="left")
    targets = targets.merge(outgoing, on="domain_id", how="left")
    targets = targets.merge(incoming, on="domain_id", how="left")
    targets = targets.merge(press[["tld", "freedom_of_the_press"]], on="tld", how="left")
    candidates = population.sample(n=400000, random_state=721).reset_index(drop=True)
    chosen_edges = edges[edges["domain_id"].isin(candidates["domain_id"])]
    labels = chosen_edges.groupby("domain_id")["tracker_id"].agg(list).reset_index()
    payload = skrub.as_data_op(dict(population=population, targets=targets,
                                   candidates=candidates, labels=labels))
    report = payload.skb.apply(EvidenceStudy(), no_wrap=True)
    return {"evidence": report, "press_preview": press.head(10)}
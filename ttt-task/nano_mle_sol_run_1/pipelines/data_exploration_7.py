import time
import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
N_POOL = 1206000
N_RESERVED = 6000
N_TRACKERS = 355
CATEGORIES = [
    "Adult", "Arts", "Business", "Computers", "Games", "Health",
    "Home", "Kids", "News", "Recreation", "Reference", "Science",
    "Shopping", "Society", "Sports"
]
BASE_COLUMNS = [
    "hostname_length", "hostname_dots", "hostname_digits",
    "hostname_hyphens", "log_out_degree", "log_in_degree",
    "direct_covered"
]


class LightweightSemanticNetwork(nn.Module):
    def __init__(self, numeric_width, tld_count):
        super().__init__()
        self.embedding = nn.Embedding(tld_count, 16)
        self.layers = nn.Sequential(
            nn.Linear(numeric_width + 16, 128),
            nn.GELU(),
            nn.Linear(128, N_TRACKERS)
        )

    def forward(self, numeric, tld):
        return self.layers(torch.cat([numeric, self.embedding(tld)], dim=1))


class SemanticComparison(BaseEstimator):
    """Exploratory learned-model comparison, not a scored candidate."""

    def __init__(self, semantic_columns, epochs=4, allowance=1200):
        self.semantic_columns = semantic_columns
        self.epochs = epochs
        self.allowance = allowance

    def fit(self, X, y=None):
        entered = time.perf_counter()
        construction_seconds = max(
            0.0,
            (pd.Timestamp.now(tz="UTC") - X["construction_started"].iloc[0]).total_seconds()
        )
        torch.set_num_threads(16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        truth = np.asarray(y, dtype=np.float32)
        training = np.arange(N_RESERVED, N_POOL)
        records = []
        scores = {}
        times = {}
        memory = {}
        first_seconds = None
        projected_seconds = None
        completed = False

        # Baseline fold zero is the timing gate. No augmented fit occurs before it.
        configurations = [
            ("baseline", 0), ("baseline", 1), ("baseline", 2),
            ("semantic", 0), ("semantic", 1), ("semantic", 2)
        ]
        for step, (variant, fold) in enumerate(configurations):
            if step == 1:
                projected_seconds = (
                    6.0 * construction_seconds + 6.0 * first_seconds
                )
                available = self.allowance - construction_seconds - first_seconds
                projected_remaining = (
                    5.0 * construction_seconds + 5.0 * first_seconds
                )
                if projected_remaining * 1.2 > available:
                    break
            start = time.perf_counter()
            torch.manual_seed(42)
            if device == "cuda":
                torch.cuda.manual_seed_all(42)
                torch.cuda.reset_peak_memory_stats()
            columns = list(BASE_COLUMNS)
            if variant == "semantic":
                columns += list(self.semantic_columns)

            train_frame = X.iloc[training]
            validation = np.arange(fold * 2000, (fold + 1) * 2000)
            valid_frame = X.iloc[validation]
            scaler = StandardScaler()
            numeric_train = np.nan_to_num(
                train_frame[columns].to_numpy(dtype=np.float32)
            )
            numeric_train = scaler.fit_transform(numeric_train).astype(np.float32)
            np.clip(numeric_train, -20, 20, out=numeric_train)
            numeric_valid = np.nan_to_num(
                valid_frame[columns].to_numpy(dtype=np.float32)
            )
            numeric_valid = scaler.transform(numeric_valid).astype(np.float32)
            np.clip(numeric_valid, -20, 20, out=numeric_valid)
            vocabulary = {
                value: i + 1 for i, value in enumerate(
                    sorted(train_frame["tld"].fillna("unknown").unique())
                )
            }
            tld_train = train_frame["tld"].fillna("unknown").map(
                vocabulary
            ).fillna(0).to_numpy(dtype=np.int64)
            tld_valid = valid_frame["tld"].fillna("unknown").map(
                vocabulary
            ).fillna(0).to_numpy(dtype=np.int64)
            labels = truth[training]
            weights = 1.0 / np.maximum(labels.sum(axis=1), 1.0)
            model = LightweightSemanticNetwork(
                len(columns), len(vocabulary) + 1
            ).to(device)
            prior = (labels * weights[:, None]).sum(axis=0) / weights.sum()
            prior = np.clip(prior, 1e-5, 1 - 1e-5)
            with torch.no_grad():
                model.layers[-1].weight.zero_()
                model.layers[-1].bias.copy_(
                    torch.as_tensor(np.log(prior / (1 - prior)), device=device)
                )
            dataset = TensorDataset(
                torch.from_numpy(numeric_train),
                torch.from_numpy(tld_train),
                torch.from_numpy(labels),
                torch.from_numpy(weights)
            )
            loader = DataLoader(
                dataset, batch_size=8192, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(42)
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            for epoch in range(self.epochs):
                model.train()
                loss_sum = 0.0
                seen = 0
                for numeric_batch, tld_batch, label_batch, weight_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(numeric_batch.to(device), tld_batch.to(device))
                    loss = (
                        nn.functional.binary_cross_entropy_with_logits(
                            logits, label_batch.to(device), reduction="none"
                        ).sum(dim=1) * weight_batch.to(device)
                    ).mean()
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(numeric_batch)
                    seen += len(numeric_batch)
                print(
                    "semantic_exploration", variant, fold, epoch + 1,
                    "loss", loss_sum / seen, flush=True
                )
            model.eval()
            predictions = []
            with torch.no_grad():
                for offset in range(0, len(validation), 8192):
                    predictions.append(
                        model(
                            torch.from_numpy(numeric_valid[offset:offset + 8192]).to(device),
                            torch.from_numpy(tld_valid[offset:offset + 8192]).to(device)
                        ).cpu().numpy()
                    )
            prediction = np.concatenate(predictions)
            guesses = np.argsort(-prediction, axis=1, kind="stable")[:, :10]
            valid_truth = truth[validation]
            recalls = np.take_along_axis(valid_truth, guesses, axis=1).sum(
                axis=1
            ) / valid_truth.sum(axis=1)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            score = float(recalls.mean())
            scores[(variant, fold)] = score
            times[(variant, fold)] = elapsed
            array_bytes = (
                numeric_train.nbytes + numeric_valid.nbytes +
                tld_train.nbytes + tld_valid.nbytes + labels.nbytes +
                weights.nbytes
            )
            gpu_bytes = (
                torch.cuda.max_memory_allocated() if device == "cuda" else 0
            )
            memory[(variant, fold)] = (array_bytes, gpu_bytes)
            records.append({
                "section": "fold", "variant": variant, "fold": fold,
                "recall_at_10": score,
                "feature_fit_train_score_seconds": elapsed,
                "conservative_complete_fold_seconds": construction_seconds + elapsed,
                "prepared_array_gib": array_bytes / 2**30,
                "gpu_peak_allocated_gib": gpu_bytes / 2**30
            })
            print(records[-1], flush=True)
            if step == 0:
                first_seconds = elapsed
            del model, optimizer, loader, dataset
            del numeric_train, numeric_valid, labels, weights, train_frame, valid_frame

        completed = len(scores) == 6
        if completed:
            for variant in ["baseline", "semantic"]:
                values = [scores[(variant, fold)] for fold in range(3)]
                variable_seconds = sum(times[(variant, fold)] for fold in range(3))
                records.append({
                    "section": "summary", "variant": variant,
                    "recall_at_10": float(np.mean(values)),
                    "fold_std": float(np.std(values)),
                    "feature_fit_train_score_seconds": variable_seconds,
                    "conservative_complete_three_fold_seconds":
                        3 * construction_seconds + variable_seconds
                })
            for fold in range(3):
                records.append({
                    "section": "paired_difference", "fold": fold,
                    "semantic_minus_baseline":
                        scores[("semantic", fold)] - scores[("baseline", fold)]
                })
        records.append({
            "section": "feasibility",
            "status": "comparison_complete" if completed else "stopped_at_timing_gate",
            "construction_seconds": construction_seconds,
            "first_baseline_fold_seconds": first_seconds,
            "projected_six_fold_complete_seconds": projected_seconds,
            "actual_shared_construction_comparison_seconds":
                construction_seconds + time.perf_counter() - entered,
            "allowance_seconds": self.allowance,
            "safety_factor": 1.2,
            "device": device
        })
        self.report_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.report_.copy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def category_block(edges, relevant, membership, focal, neighbour, prefix):
    selected = edges[edges[focal].isin(relevant["domain_id"])]
    selected = selected[selected[focal] != selected[neighbour]]
    selected = selected.drop_duplicates([focal, neighbour])
    degree = selected.groupby(focal).size().rename("degree").reset_index()
    labelled_edges = selected.merge(
        membership.rename(columns={"domain_id": neighbour}),
        on=neighbour, how="inner", sort=False
    )
    classified = labelled_edges[[focal, neighbour]].drop_duplicates()
    classified = classified.groupby(focal).size().rename(
        "classified_neighbours"
    ).reset_index()
    mass = labelled_edges.pivot_table(
        index=focal, columns="category", values="membership_weight",
        aggfunc="sum", fill_value=0
    ).reindex(columns=CATEGORIES, fill_value=0)
    mass = mass.div(mass.sum(axis=1), axis=0).fillna(0).reset_index()
    names = {category: prefix + "_category_" + category for category in CATEGORIES}
    mass = mass.rename(columns=names)
    table = relevant[["domain_id", "population"]].merge(
        degree.rename(columns={focal: "domain_id"}),
        on="domain_id", how="left", sort=False
    ).merge(
        classified.rename(columns={focal: "domain_id"}),
        on="domain_id", how="left", sort=False
    ).merge(
        mass.rename(columns={focal: "domain_id"}),
        on="domain_id", how="left", sort=False
    ).fillna(0)
    profile_columns = list(names.values())
    profile = table[profile_columns]
    entropy = -(profile * profile.clip(lower=1e-12).skb.apply_func(np.log)).sum(axis=1)
    table = table.assign(**{
        prefix + "_covered": (table["classified_neighbours"] > 0).astype("float32"),
        prefix + "_log_classified": table["classified_neighbours"].skb.apply_func(np.log1p),
        prefix + "_fraction": table["classified_neighbours"] / table["degree"].clip(lower=1),
        prefix + "_entropy": entropy,
        "log_" + prefix + "_degree": table["degree"].skb.apply_func(np.log1p)
    })
    feature_columns = profile_columns + [
        prefix + "_covered", prefix + "_log_classified",
        prefix + "_fraction", prefix + "_entropy"
    ]
    summary = table.groupby("population").agg(
        domains=("domain_id", "size"),
        covered=(prefix + "_covered", "sum"),
        coverage=(prefix + "_covered", "mean"),
        mean_classified_neighbours=("classified_neighbours", "mean"),
        mean_fraction=(prefix + "_fraction", "mean"),
        mean_entropy=(prefix + "_entropy", "mean")
    ).reset_index()
    return (
        table[["domain_id", "log_" + prefix + "_degree"] + feature_columns],
        feature_columns, summary
    )


def build():
    # The timestamp is evaluated as the first branch of the returned feature table.
    clock = skrub.as_data_op("now").skb.apply_func(pd.Timestamp, tz="UTC")
    labels = skrub.as_data_op(BASE + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    ).drop_duplicates(["domain_id", "tracker_id"])
    counts = labels.groupby("domain_id")["tracker_id"].nunique().rename(
        "known_tracker_count"
    ).reset_index()
    eligible = counts[counts["known_tracker_count"] >= 2]
    pool = eligible[["domain_id"]].sort_values("domain_id").sample(
        n=N_POOL, replace=False, random_state=42
    ).reset_index(drop=True)
    pool = pool.assign(construction_started=clock)
    targets = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    )
    relevant = pool[["domain_id"]].skb.concat([targets], axis=0).reset_index(drop=True)
    relevant = relevant.reset_index().rename(columns={"index": "profile_row"})
    relevant = relevant.assign(population="training")
    relevant = relevant.assign(
        population=relevant["population"].mask(
            relevant["profile_row"] >= N_POOL, "prediction"
        )
    )
    for fold in range(3):
        mask = (
            (relevant["profile_row"] >= fold * 2000) &
            (relevant["profile_row"] < (fold + 1) * 2000)
        )
        relevant = relevant.assign(
            population=relevant["population"].mask(mask, "validation_" + str(fold))
        )
    domains = skrub.as_data_op(BASE + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"]
    )
    classification = skrub.as_data_op(BASE + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"]
    )
    host = classification["url"].fillna("").str.lower().str.replace(
        "^https?://", "", regex=True
    ).str.split("/").str[0].str.replace("^www\\.", "", regex=True)
    classification = classification.assign(host=host)[["host", "category"]]
    classification = classification.dropna(subset=["category"]).drop_duplicates()
    category_counts = classification.groupby("host")["category"].nunique().rename(
        "category_count"
    ).reset_index()
    classification = classification.merge(category_counts, on="host", how="inner")
    classification = classification.assign(
        membership_weight=1.0 / classification["category_count"]
    )
    domain_host = domains.assign(
        host=domains["domain"].fillna("").str.lower().str.replace(
            "^www\\.", "", regex=True
        )
    )
    classified_domains = domain_host[domain_host["host"].isin(classification["host"])]
    membership = classified_domains[["domain_id", "host"]].merge(
        classification, on="host", how="inner", sort=False
    )[["domain_id", "category", "membership_weight"]]
    membership = membership.drop_duplicates(["domain_id", "category"])

    links = skrub.as_data_op(BASE + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
    )
    touching = links[
        links["source_domain_id"].isin(relevant["domain_id"]) |
        links["target_domain_id"].isin(relevant["domain_id"])
    ]
    outgoing, out_columns, out_summary = category_block(
        touching, relevant, membership, "source_domain_id",
        "target_domain_id", "out"
    )
    incoming, in_columns, in_summary = category_block(
        touching, relevant, membership, "target_domain_id",
        "source_domain_id", "in"
    )
    direct = membership[membership["domain_id"].isin(relevant["domain_id"])]
    direct_profiles = direct.pivot_table(
        index="domain_id", columns="category", values="membership_weight",
        aggfunc="sum", fill_value=0
    ).reindex(columns=CATEGORIES, fill_value=0).reset_index()
    direct_names = {c: "direct_category_" + c for c in CATEGORIES}
    direct_profiles = direct_profiles.rename(columns=direct_names)
    direct_coverage = direct[["domain_id"]].drop_duplicates().assign(direct_covered=1)
    direct_table = relevant[["domain_id", "population"]].merge(
        direct_coverage, on="domain_id", how="left", sort=False
    ).fillna({"direct_covered": 0})
    direct_summary = direct_table.groupby("population").agg(
        domains=("domain_id", "size"),
        covered=("direct_covered", "sum"),
        coverage=("direct_covered", "mean")
    ).reset_index()

    lookup = domain_host[domain_host["domain_id"].isin(relevant["domain_id"])]
    lookup = lookup.drop_duplicates("domain_id")
    hostname = lookup["host"]
    lookup = lookup.assign(
        tld=hostname.str.split(".").str[-1],
        hostname_length=hostname.str.len().astype("float32"),
        hostname_dots=hostname.str.count("\\.").astype("float32"),
        hostname_digits=hostname.str.count("[0-9]").astype("float32"),
        hostname_hyphens=hostname.str.count("-").astype("float32")
    ).drop(columns=["domain", "host"])
    X = pool.merge(lookup, on="domain_id", how="left", sort=False)
    X = X.merge(outgoing, on="domain_id", how="left", sort=False)
    X = X.merge(incoming, on="domain_id", how="left", sort=False)
    X = X.merge(direct_coverage, on="domain_id", how="left", sort=False)
    X = X.merge(direct_profiles, on="domain_id", how="left", sort=False)
    semantic_columns = out_columns + in_columns + list(direct_names.values())
    numeric_columns = BASE_COLUMNS + semantic_columns
    X = X.fillna(dict.fromkeys(numeric_columns, 0))
    pool_labels = labels.merge(pool[["domain_id"]], on="domain_id", how="inner")
    truth = pool_labels.assign(present=1).pivot_table(
        index="domain_id", columns="tracker_id", values="present",
        aggfunc="max", fill_value=0
    ).reindex(
        index=pool["domain_id"], columns=list(range(N_TRACKERS)), fill_value=0
    ).fillna(0).astype("uint8").reset_index(drop=True)
    report = X.skb.apply(SemanticComparison(semantic_columns), y=truth)
    return {
        "learned_semantic_comparison": report,
        "outgoing_coverage": out_summary,
        "incoming_coverage": in_summary,
        "direct_coverage": direct_summary,
        "protocol": skrub.as_data_op({
            "model": "CUDA numeric/TLD MLP, 128 hidden units, four epochs, inverse-cardinality BCE",
            "folds": "Exact locked sample/order: common rows 6000:1206000 train; consecutive reserved 2000-row validation subsets.",
            "baseline": "Hostname structure, TLD, hyperlink degrees and direct classification availability; no relational tracker profiles.",
            "augmentation": "Fractional direct/outgoing/incoming category profiles, directional classified-neighbour counts, fractions and entropy.",
            "label_safety": "No auxiliary tracker labels are used. Supervised labels enter each exploratory model only through its common training subset.",
            "timing": "Construction timestamp starts the pool branch; includes subsequent graph evaluation. Conservative gate charges construction independently for all six fits, with 20% safety margin. Harness wall time is authoritative for actual complete execution.",
            "memory": "Prepared-array bytes and CUDA peak allocation are reported; neither measures process peak RSS, which remains unfinished.",
            "parent_complementarity": "Unresolved: no authorized complete parent predictions were supplied. No artifacts are read and the parent is not refitted.",
            "interpretation": "This tests semantic signal in a lightweight learned baseline, not improvement over the selected relational parent. Fold spreads are descriptive, not independent-replicate uncertainty."
        })
    }
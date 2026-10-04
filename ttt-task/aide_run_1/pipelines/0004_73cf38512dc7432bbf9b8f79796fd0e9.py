import os
import numpy as np
import pandas as pd
import polars as pl


def extract_tld(host: str) -> str:
    """Extract top-level domain or common two-part ccTLD from hostname."""
    if not isinstance(host, str) or not host:
        return "unknown"
    parts = host.strip().lower().split(".")
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in {
            "co",
            "com",
            "org",
            "net",
            "gov",
            "edu",
            "ac",
        }:
            return f"{parts[-2]}.{parts[-1]}"
        return parts[-1]
    return parts[0]


def main():
    print("Loading tracker metadata and targets...")
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    tracker_id_to_domain_id = dict(
        zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
    )
    domain_id_to_tracker_id = dict(
        zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
    )
    tracker_domain_ids_set = set(trackers_df["tracking_domain_id"])
    num_trackers = len(trackers_df)

    target_df = pd.read_csv("input/target.tsv", sep="\t")
    target_domain_ids = target_df["domain_id"].to_numpy().astype(np.int64)

    print("Loading training tracking graph...")
    train_tg = pl.read_parquet(
        "input/tracking_graph_train.parquet",
        columns=["domain_id", "tracking_domain_id", "tracker_id"],
    )

    all_train_domains = train_tg["domain_id"].unique().to_numpy()
    available_train_domains = np.setdiff1d(
        all_train_domains, target_domain_ids, assume_unique=True
    )

    # 10,000 domain hold-out validation set
    np.random.seed(42)
    val_domain_ids = np.random.choice(
        available_train_domains, size=10000, replace=False
    )
    val_domain_set = set(val_domain_ids)

    # Split tracking data
    val_tg = (
        train_tg.filter(pl.col("domain_id").is_in(val_domain_set))
        .select(["domain_id", "tracker_id"])
        .unique()
    )
    train_tg_split = (
        train_tg.filter(~pl.col("domain_id").is_in(val_domain_set))
        .select(["domain_id", "tracker_id"])
        .unique()
    )

    # Validation ground truth
    val_grouped = val_tg.group_by("domain_id").agg(pl.col("tracker_id"))
    val_truth_dict = dict(
        zip(
            val_grouped["domain_id"].to_list(),
            [set(x) for x in val_grouped["tracker_id"].to_list()],
        )
    )

    # Training tracker profiles
    train_grouped = train_tg_split.group_by("domain_id").agg(pl.col("tracker_id"))
    train_domain_tracker_dict = dict(
        zip(
            train_grouped["domain_id"].to_list(),
            train_grouped["tracker_id"].to_list(),
        )
    )

    query_domain_set = set(val_domain_ids) | set(target_domain_ids)
    query_domains_list = list(query_domain_set)
    query_id_to_idx = {qid: idx for idx, qid in enumerate(query_domains_list)}
    num_query = len(query_domains_list)

    print(
        f"Query domains: {num_query} (Validation: {len(val_domain_ids)}, Target: {len(target_domain_ids)})"
    )

    print("Loading domains lookup and extracting TLDs...")
    needed_domains = query_domain_set | set(train_domain_tracker_dict.keys())
    needed_df = pl.DataFrame(
        {"domain_id": list(needed_domains)}, schema={"domain_id": pl.Int64}
    )

    domains_df = pl.read_parquet(
        "input/domains.parquet", columns=["domain_id", "domain"]
    )
    domains_filtered = domains_df.join(needed_df, on="domain_id", how="inner")

    domain_id_to_host = dict(
        zip(
            domains_filtered["domain_id"].to_list(),
            domains_filtered["domain"].to_list(),
        )
    )

    # Build empirical Bayes TLD tracker distribution
    print("Computing empirical Bayes TLD priors...")
    tld_counts = {}
    tld_tracker_counts = {}
    global_tracker_counts = np.zeros(num_trackers, dtype=np.float32)

    for dom_id, trackers in train_domain_tracker_dict.items():
        host = domain_id_to_host.get(dom_id, "")
        tld = extract_tld(host)
        tld_counts[tld] = tld_counts.get(tld, 0) + 1
        if tld not in tld_tracker_counts:
            tld_tracker_counts[tld] = np.zeros(num_trackers, dtype=np.float32)
        for t in trackers:
            tld_tracker_counts[tld][t] += 1.0
            global_tracker_counts[t] += 1.0

    total_train_domains = len(train_domain_tracker_dict)
    global_prior = (global_tracker_counts + 1.0) / (total_train_domains + num_trackers)

    # Precompute smoothed TLD priors
    smoothing_m = 15.0
    tld_smoothed_prior = {}
    for tld, count in tld_counts.items():
        smoothed = (tld_tracker_counts[tld] + smoothing_m * global_prior) / (
            count + smoothing_m
        )
        tld_smoothed_prior[tld] = smoothed.astype(np.float32)

    prior_matrix = np.zeros((num_query, num_trackers), dtype=np.float32)
    for qid, qidx in query_id_to_idx.items():
        host = domain_id_to_host.get(qid, "")
        tld = extract_tld(host)
        prior_matrix[qidx] = tld_smoothed_prior.get(tld, global_prior)

    # Compute tracker co-occurrence transition matrix
    print("Computing tracker stack co-occurrence transition matrix...")
    cooc_matrix = np.zeros((num_trackers, num_trackers), dtype=np.float32)
    tracker_freq = np.zeros(num_trackers, dtype=np.float32)

    for trackers in train_domain_tracker_dict.values():
        t_arr = np.array(trackers, dtype=np.int32)
        tracker_freq[t_arr] += 1.0
        for t_a in t_arr:
            cooc_matrix[t_a, t_arr] += 1.0

    cooc_transition = cooc_matrix / (tracker_freq[:, None] + 10.0)
    np.fill_diagonal(cooc_transition, 0.0)

    # Scan and filter hyperlink graph
    print("Processing link graph...")
    lg_lazy = pl.scan_parquet(
        "input/link-graph.parquet",
        columns=["source_domain_id", "target_domain_id"],
    )

    matched_edges = lg_lazy.filter(
        pl.col("source_domain_id").is_in(query_domains_list)
        | pl.col("target_domain_id").is_in(query_domains_list)
    ).collect()

    print(f"Extracted {len(matched_edges)} active edges for query neighborhoods.")

    direct_scores = np.zeros((num_query, num_trackers), dtype=np.float32)
    graph_scores = np.zeros((num_query, num_trackers), dtype=np.float32)
    graph_total_weights = np.zeros(num_query, dtype=np.float32)

    # Direct links to tracker hostnames
    out_edges = matched_edges.filter(
        pl.col("source_domain_id").is_in(query_domains_list)
    )
    direct_tracker_edges = out_edges.filter(
        pl.col("target_domain_id").is_in(list(tracker_domain_ids_set))
    )

    src_direct = direct_tracker_edges["source_domain_id"].to_numpy()
    tgt_direct = direct_tracker_edges["target_domain_id"].to_numpy()
    for s, t in zip(src_direct, tgt_direct):
        u_idx = query_id_to_idx[s]
        tracker_idx = domain_id_to_tracker_id[t]
        direct_scores[u_idx, tracker_idx] = 5.0

    # Direction-asymmetric neighbor propagation
    out_to_train = out_edges.filter(
        pl.col("target_domain_id").is_in(list(train_domain_tracker_dict.keys()))
    )
    v_counts_out = out_to_train["target_domain_id"].value_counts()
    v_deg_map_out = dict(
        zip(
            v_counts_out["target_domain_id"].to_list(),
            v_counts_out["count"].to_list(),
        )
    )

    src_arr_out = out_to_train["source_domain_id"].to_numpy()
    tgt_arr_out = out_to_train["target_domain_id"].to_numpy()
    for u, v in zip(src_arr_out, tgt_arr_out):
        u_idx = query_id_to_idx[u]
        trackers = train_domain_tracker_dict[v]
        deg = v_deg_map_out.get(v, 1)
        w = 1.0 / (np.log(2.0 + deg) * np.sqrt(len(trackers)))
        graph_total_weights[u_idx] += w
        for t in trackers:
            graph_scores[u_idx, t] += w

    in_edges = matched_edges.filter(
        pl.col("target_domain_id").is_in(query_domains_list)
    )
    in_from_train = in_edges.filter(
        pl.col("source_domain_id").is_in(list(train_domain_tracker_dict.keys()))
    )
    v_counts_in = in_from_train["source_domain_id"].value_counts()
    v_deg_map_in = dict(
        zip(
            v_counts_in["source_domain_id"].to_list(),
            v_counts_in["count"].to_list(),
        )
    )

    src_arr_in = in_from_train["source_domain_id"].to_numpy()
    tgt_arr_in = in_from_train["target_domain_id"].to_numpy()
    for v, u in zip(src_arr_in, tgt_arr_in):
        u_idx = query_id_to_idx[u]
        trackers = train_domain_tracker_dict[v]
        deg = v_deg_map_in.get(v, 1)
        w = 0.4 / (np.log(2.0 + deg) * np.sqrt(len(trackers)))
        graph_total_weights[u_idx] += w
        for t in trackers:
            graph_scores[u_idx, t] += w

    # Degree normalization and co-occurrence diffusion
    norm_factor = np.sqrt(1.0 + graph_total_weights)[:, None]
    graph_scores_norm = graph_scores / norm_factor

    base_scores = direct_scores + graph_scores_norm
    cooc_scores = base_scores @ cooc_transition

    # Combine signals
    final_scores = (
        1.0 * prior_matrix + direct_scores + graph_scores_norm + 0.20 * cooc_scores
    )

    # Compute validation Recall@10
    print("Evaluating on validation set...")
    val_recalls = []
    for val_id in val_domain_ids:
        u_idx = query_id_to_idx[val_id]
        scores = final_scores[u_idx]
        top10_trackers = np.argpartition(scores, -10)[-10:]
        true_trackers = val_truth_dict.get(val_id, set())

        if len(true_trackers) > 0:
            hits = len(true_trackers.intersection(top10_trackers))
            recall = hits / len(true_trackers)
            val_recalls.append(recall)
        else:
            val_recalls.append(1.0)

    mean_val_recall = float(np.mean(val_recalls))
    print(f"Validation Recall@10: {mean_val_recall:.4f}")

    # Generate predictions for target domains
    print("Generating predictions for target domains...")
    sub_domain_ids = []
    sub_tracking_domain_ids = []

    for target_id in target_domain_ids:
        u_idx = query_id_to_idx[target_id]
        scores = final_scores[u_idx]
        top10_trackers = np.argpartition(scores, -10)[-10:]
        top10_trackers = top10_trackers[np.argsort(-scores[top10_trackers])]

        for t_idx in top10_trackers:
            tracking_domain = tracker_id_to_domain_id[t_idx]
            sub_domain_ids.append(target_id)
            sub_tracking_domain_ids.append(tracking_domain)

    sub_df = pd.DataFrame(
        {
            "domain_id": sub_domain_ids,
            "tracking_domain_id": sub_tracking_domain_ids,
        }
    )

    os.makedirs("working", exist_ok=True)
    sub_df.to_csv("working/submission.csv", sep="\t", index=False)
    print(
        f"Submission saved successfully to working/submission.csv with {len(sub_df)} rows."
    )


if __name__ == "__main__":
    main()

import os
import gc
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq


def extract_tld(hostname):
    if not isinstance(hostname, str) or "." not in hostname:
        return "unknown"
    parts = hostname.lower().split(".")
    if (
        len(parts) >= 2
        and parts[-2] in {"co", "com", "org", "net", "edu", "gov"}
        and len(parts[-1]) == 2
    ):
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def compute_recall_at_10(scores, val_domains, ground_truth_dict):
    top10_indices = np.argpartition(-scores, 10, axis=1)[:, :10]
    recalls = []
    for i, d in enumerate(val_domains):
        true_trackers = ground_truth_dict[d]
        if len(true_trackers) == 0:
            continue
        predicted = top10_indices[i]
        hits = len(true_trackers.intersection(predicted))
        recalls.append(hits / len(true_trackers))
    return float(np.mean(recalls))


def main():
    start_time = time.time()
    os.makedirs("./working", exist_ok=True)
    np.random.seed(42)

    print("Loading tracker and target metadata...")
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    tracker_id_to_tracking_domain_id = dict(
        zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
    )
    tracking_domain_id_to_tracker_id = dict(
        zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
    )
    num_trackers = len(trackers_df)
    tracker_domain_ids = trackers_df["tracking_domain_id"].to_numpy()
    tracker_domains_set = set(tracker_domain_ids)

    target_df = pd.read_csv("input/target.tsv", sep="\t")
    target_domain_ids = target_df["domain_id"].to_numpy()
    target_domains_set = set(target_domain_ids)

    print("Loading training tracking graph...")
    train_graph_df = pd.read_parquet(
        "input/tracking_graph_train.parquet", columns=["domain_id", "tracker_id"]
    )
    domain_trackers = (
        train_graph_df.groupby("domain_id")["tracker_id"]
        .apply(lambda x: np.unique(x).astype(np.int16))
        .to_dict()
    )
    del train_graph_df
    gc.collect()

    all_train_domains = np.array(list(domain_trackers.keys()))
    np.random.shuffle(all_train_domains)

    n_val = 10000
    val_domains = all_train_domains[:n_val]
    val_domain_set = set(val_domains)
    train_domains = all_train_domains[n_val:]
    train_domain_set = set(train_domains)
    val_ground_truth = {d: set(domain_trackers[d]) for d in val_domains}

    print(
        f"Dataset split: {len(train_domains)} train domains, {len(val_domains)} validation domains."
    )

    # Global tracker prior
    global_counts = np.zeros(num_trackers, dtype=np.float64)
    for d in train_domains:
        global_counts[domain_trackers[d]] += 1.0
    p_global = (global_counts / len(train_domains)).astype(np.float32)

    # Tracker co-occurrence matrix from training domains
    print("Computing tracker co-occurrence transition matrix...")
    cooc_rows = []
    cooc_cols = []
    for idx, d in enumerate(train_domains):
        for t in domain_trackers[d]:
            cooc_rows.append(idx)
            cooc_cols.append(t)
    M_train = sp.csr_matrix(
        (np.ones(len(cooc_rows), dtype=np.float32), (cooc_rows, cooc_cols)),
        shape=(len(train_domains), num_trackers),
    )
    del cooc_rows, cooc_cols
    gc.collect()

    C_raw = (M_train.T @ M_train).toarray()
    del M_train
    gc.collect()
    diag = np.diag(C_raw)
    C_norm = (C_raw / (diag[:, None] + 1.0)).astype(np.float32)
    np.fill_diagonal(C_norm, 1.0)

    # Domain TLD extraction and smoothed TLD prior
    print("Loading domain hostnames and calculating TLD priors...")
    needed_domains = train_domain_set.union(val_domain_set).union(target_domains_set)
    domains_df = pd.read_parquet(
        "input/domains.parquet", columns=["domain_id", "domain"]
    )
    domains_df = domains_df[domains_df["domain_id"].isin(needed_domains)]
    domain_tld_map = dict(
        zip(domains_df["domain_id"], domains_df["domain"].apply(extract_tld))
    )
    del domains_df
    gc.collect()

    tld_counts = {}
    tld_totals = {}
    for d in train_domains:
        tld = domain_tld_map.get(d, "unknown")
        if tld not in tld_counts:
            tld_counts[tld] = np.zeros(num_trackers, dtype=np.float32)
            tld_totals[tld] = 0
        tld_counts[tld][domain_trackers[d]] += 1.0
        tld_totals[tld] += 1

    def get_tld_prior(d_id):
        tld = domain_tld_map.get(d_id, "unknown")
        if tld in tld_counts:
            return (tld_counts[tld] + 20.0 * p_global) / (tld_totals[tld] + 20.0)
        return p_global

    # Load link-graph and construct graph adjacency
    print("Loading link-graph...")
    link_table = pq.read_table(
        "input/link-graph.parquet", columns=["source_domain_id", "target_domain_id"]
    )
    src = link_table["source_domain_id"].to_numpy()
    dst = link_table["target_domain_id"].to_numpy()
    del link_table
    gc.collect()

    print("Indexing graph nodes...")
    all_graph_nodes = np.unique(np.concatenate([src, dst]))
    node_to_idx = {nid: idx for idx, nid in enumerate(all_graph_nodes)}
    N = len(all_graph_nodes)

    src_idx = np.array([node_to_idx[s] for s in src], dtype=np.int32)
    dst_idx = np.array([node_to_idx[d] for d in dst], dtype=np.int32)
    del src, dst
    gc.collect()

    # Track direct links from query domains to tracker domain IDs
    query_domain_set = val_domain_set.union(target_domains_set)
    val_direct_links = {d: set() for d in val_domains}
    target_direct_links = {d: set() for d in target_domain_ids}

    # Identify edges pointing to tracker hostnames
    tracker_links_mask = np.isin(all_graph_nodes[dst_idx], tracker_domain_ids)
    direct_src_ids = all_graph_nodes[src_idx[tracker_links_mask]]
    direct_dst_ids = all_graph_nodes[dst_idx[tracker_links_mask]]

    for s_id, t_did in zip(direct_src_ids, direct_dst_ids):
        t_id = tracking_domain_id_to_tracker_id.get(t_did)
        if t_id is not None:
            if s_id in val_direct_links:
                val_direct_links[s_id].add(t_id)
            elif s_id in target_direct_links:
                target_direct_links[s_id].add(t_id)

    del direct_src_ids, direct_dst_ids, tracker_links_mask
    gc.collect()

    # Build normalized adjacency matrices
    print("Building sparse normalized adjacency matrices...")
    A = sp.csr_matrix(
        (np.ones(len(src_idx), dtype=np.float32), (src_idx, dst_idx)), shape=(N, N)
    )
    del src_idx, dst_idx
    gc.collect()

    deg_out = np.diff(A.indptr)
    inv_deg_out = np.where(deg_out > 0, 1.0 / deg_out, 0.0).astype(np.float32)
    A_norm_out = sp.diags(inv_deg_out, dtype=np.float32) @ A

    A_T = A.T.tocsr()
    del A
    gc.collect()
    deg_in = np.diff(A_T.indptr)
    inv_deg_in = np.where(deg_in > 0, 1.0 / deg_in, 0.0).astype(np.float32)
    A_norm_in = sp.diags(inv_deg_in, dtype=np.float32) @ A_T
    del A_T
    gc.collect()

    # Build initial sparse label matrix for training nodes (validation and target strictly zeroed)
    print("Constructing sparse tracker label matrix...")
    y_rows = []
    y_cols = []
    y_vals = []
    for d in train_domains:
        if d in node_to_idx:
            u = node_to_idx[d]
            for t in domain_trackers[d]:
                y_rows.append(u)
                y_cols.append(t)
                y_vals.append(1.0)

    # Seed tracker domains with their corresponding tracker ID
    for t_did, t_id in tracking_domain_id_to_tracker_id.items():
        if t_did in node_to_idx:
            u = node_to_idx[t_did]
            y_rows.append(u)
            y_cols.append(t_id)
            y_vals.append(3.0)

    Y_sp = sp.csr_matrix(
        (y_vals, (y_rows, y_cols)), shape=(N, num_trackers), dtype=np.float32
    )
    del y_rows, y_cols, y_vals
    gc.collect()

    # Propagate tracker signals along outgoing and incoming edges
    print("Propagating graph signals across network...")
    W_out = (A_norm_out @ Y_sp).toarray().astype(np.float32)
    del A_norm_out
    gc.collect()

    Z_in = (A_norm_in @ Y_sp).toarray().astype(np.float32)
    del A_norm_in, Y_sp
    gc.collect()

    # Extract feature matrices for validation set
    print("Extracting validation features...")
    val_out = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_in = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_direct = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_tld = np.zeros((len(val_domains), num_trackers), dtype=np.float32)

    for i, d in enumerate(val_domains):
        val_tld[i] = get_tld_prior(d)
        for t in val_direct_links[d]:
            val_direct[i, t] = 1.0
        if d in node_to_idx:
            u = node_to_idx[d]
            val_out[i] = W_out[u]
            val_in[i] = Z_in[u]

    val_cooc = ((val_out + 0.5 * val_in) @ C_norm).astype(np.float32)

    # Grid search for optimal ensemble weights on hold-out validation set
    print("Optimizing ensemble weights on validation set...")
    best_recall = 0.0
    best_weights = (1.5, 0.7, 0.4, 6.0, 1.0)

    for w_out in [1.0, 1.8, 2.5]:
        for w_in in [0.4, 0.8]:
            for w_cooc in [0.2, 0.5]:
                for w_dir in [5.0, 10.0]:
                    for w_tld in [0.8, 1.2]:
                        scores = (
                            w_out * val_out
                            + w_in * val_in
                            + w_cooc * val_cooc
                            + w_dir * val_direct
                            + w_tld * val_tld
                        )
                        recall = compute_recall_at_10(
                            scores, val_domains, val_ground_truth
                        )
                        if recall > best_recall:
                            best_recall = recall
                            best_weights = (w_out, w_in, w_cooc, w_dir, w_tld)

    print(f"Optimal weights (w_out, w_in, w_cooc, w_dir, w_tld): {best_weights}")
    print(f"Validation Recall@10: {best_recall:.4f}")

    # Generate predictions for target domains
    print("Generating predictions for target domains...")
    target_out = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_in = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_direct = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_tld = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)

    for i, d in enumerate(target_domain_ids):
        target_tld[i] = get_tld_prior(d)
        for t in target_direct_links[d]:
            target_direct[i, t] = 1.0
        if d in node_to_idx:
            u = node_to_idx[d]
            target_out[i] = W_out[u]
            target_in[i] = Z_in[u]

    del W_out, Z_in
    gc.collect()

    target_cooc = ((target_out + 0.5 * target_in) @ C_norm).astype(np.float32)

    w_out, w_in, w_cooc, w_dir, w_tld = best_weights
    target_scores = (
        w_out * target_out
        + w_in * target_in
        + w_cooc * target_cooc
        + w_dir * target_direct
        + w_tld * target_tld
    )

    # Rank top 10 trackers per domain
    top10_target = np.argpartition(-target_scores, 10, axis=1)[:, :10]
    sub_domain_list = []
    sub_tracker_list = []

    for i, d in enumerate(target_domain_ids):
        ranked = top10_target[i][np.argsort(-target_scores[i, top10_target[i]])]
        for t_idx in ranked:
            sub_domain_list.append(d)
            sub_tracker_list.append(tracker_id_to_tracking_domain_id[t_idx])

    submission_df = pd.DataFrame(
        {"domain_id": sub_domain_list, "tracking_domain_id": sub_tracker_list}
    )

    submission_file_csv = "./working/submission.csv"
    submission_file_tsv = "./working/submission.tsv"
    submission_df.to_csv(submission_file_csv, sep="\t", index=False)
    submission_df.to_csv(submission_file_tsv, sep="\t", index=False)

    print(f"Saved submission with {len(submission_df)} rows to {submission_file_csv}")
    print(f"Completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

from collections import defaultdict
import gc
import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp


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

    tracking_domain_id_to_tracker_ids = defaultdict(list)
    for _, row in trackers_df.iterrows():
        tracking_domain_id_to_tracker_ids[int(row["tracking_domain_id"])].append(
            int(row["tracker_id"])
        )

    num_trackers = len(trackers_df)
    tracker_domain_ids = np.array(
        list(tracking_domain_id_to_tracker_ids.keys()), dtype=np.int64
    )

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
    np.random.seed(42)
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

    # Tracker affinity and asymmetric transition matrices from training domains
    print("Computing cosine affinity and asymmetric conditional transition matrices...")
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

    C_raw = (M_train.T @ M_train).toarray().astype(np.float32)
    del M_train
    gc.collect()

    diag = np.diag(C_raw)

    # 1. Cosine affinity matrix (symmetric, captures stack uniqueness)
    denom = np.sqrt((diag[:, None] + 5.0) * (diag[None, :] + 5.0))
    C_affinity = (C_raw / denom).astype(np.float32)
    np.fill_diagonal(C_affinity, 0.0)

    # 2. Asymmetric conditional transition matrix P(j | i) = C_raw[i, j] / (diag[i] + 15)
    C_cond = (C_raw / (diag[:, None] + 15.0)).astype(np.float32)
    np.fill_diagonal(C_cond, 0.0)

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
        "input/link-graph.parquet",
        columns=["source_domain_id", "target_domain_id"],
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
    val_direct_links = {d: set() for d in val_domains}
    target_direct_links = {d: set() for d in target_domain_ids}

    tracker_links_mask = np.isin(all_graph_nodes[dst_idx], tracker_domain_ids)
    direct_src_ids = all_graph_nodes[src_idx[tracker_links_mask]]
    direct_dst_ids = all_graph_nodes[dst_idx[tracker_links_mask]]

    for s_id, t_did in zip(direct_src_ids, direct_dst_ids):
        t_ids = tracking_domain_id_to_tracker_ids.get(t_did)
        if t_ids is not None:
            if s_id in val_direct_links:
                for t_id in t_ids:
                    val_direct_links[s_id].add(t_id)
            elif s_id in target_direct_links:
                for t_id in t_ids:
                    target_direct_links[s_id].add(t_id)

    del direct_src_ids, direct_dst_ids, tracker_links_mask
    gc.collect()

    # Build normalized adjacency matrices
    print("Building sparse normalized adjacency matrices...")
    A = sp.csr_matrix(
        (np.ones(len(src_idx), dtype=np.float32), (src_idx, dst_idx)),
        shape=(N, N),
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

    # Build initial sparse label matrix for training nodes
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

    # Seed tracker domains with their corresponding tracker IDs
    for t_did, t_ids in tracking_domain_id_to_tracker_ids.items():
        if t_did in node_to_idx:
            u = node_to_idx[t_did]
            for t_id in t_ids:
                y_rows.append(u)
                y_cols.append(t_id)
                y_vals.append(3.0)

    Y_sp = sp.csr_matrix(
        (y_vals, (y_rows, y_cols)), shape=(N, num_trackers), dtype=np.float32
    )
    del y_rows, y_cols, y_vals
    gc.collect()

    # Propagate base signals across network
    print("Propagating base 1-hop graph signals...")
    W_out = (A_norm_out @ Y_sp).toarray().astype(np.float32)
    Z_in = (A_norm_in @ Y_sp).toarray().astype(np.float32)
    del Y_sp
    gc.collect()

    # Extract multi-relational graph features for validation set
    print("Extracting multi-relational cross-hop features for validation set...")
    val_out = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_in = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_co_out = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_out_2hop = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_co_in = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_in_2hop = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_direct = np.zeros((len(val_domains), num_trackers), dtype=np.float32)
    val_tld = np.zeros((len(val_domains), num_trackers), dtype=np.float32)

    val_in_graph = np.array([d in node_to_idx for d in val_domains], dtype=bool)
    val_nodes = np.array(
        [node_to_idx[d] for d in val_domains if d in node_to_idx],
        dtype=np.int32,
    )

    for i, d in enumerate(val_domains):
        val_tld[i] = get_tld_prior(d)
        for t in val_direct_links[d]:
            val_direct[i, t] = 1.0

    if len(val_nodes) > 0:
        val_out[val_in_graph] = W_out[val_nodes]
        val_in[val_in_graph] = Z_in[val_nodes]

        A_sub_out = A_norm_out[val_nodes, :]
        A_sub_in = A_norm_in[val_nodes, :]
        val_co_out[val_in_graph] = (A_sub_out @ Z_in).astype(np.float32)
        val_out_2hop[val_in_graph] = (A_sub_out @ W_out).astype(np.float32)
        val_co_in[val_in_graph] = (A_sub_in @ W_out).astype(np.float32)
        val_in_2hop[val_in_graph] = (A_sub_in @ Z_in).astype(np.float32)
        del A_sub_out, A_sub_in
        gc.collect()

    # Calibrated probability distribution for co-outlinks
    co_out_sum = val_co_out.sum(axis=1, keepdims=True)
    val_co_out_norm = np.where(
        co_out_sum > 0, val_co_out / (co_out_sum + 1e-5), 0.0
    ).astype(np.float32)

    # Direct tracker link conditional transition
    val_dir_trans = (val_direct @ C_cond).astype(np.float32)

    # Neighborhood co-occurrence transitions (cosine affinity and conditional probability)
    val_base_evidence = val_out + 0.5 * val_in + 0.4 * val_co_out + 4.0 * val_direct
    val_cooc_cos = (val_base_evidence @ C_affinity).astype(np.float32)
    val_cooc_cond = (val_base_evidence @ C_cond).astype(np.float32)

    # Feature ensemble dictionary for validation
    val_features = {
        "out": val_out,
        "in": val_in,
        "co_out": val_co_out,
        "co_out_n": val_co_out_norm,
        "out2": val_out_2hop,
        "co_in": val_co_in,
        "in2": val_in_2hop,
        "dir": val_direct,
        "dir_trans": val_dir_trans,
        "cooc_cos": val_cooc_cos,
        "cooc_cond": val_cooc_cond,
        "tld": val_tld,
    }

    current_weights = {
        "out": 1.6,
        "in": 0.6,
        "co_out": 0.8,
        "co_out_n": 0.5,
        "out2": 0.3,
        "co_in": 0.2,
        "in2": 0.1,
        "dir": 8.0,
        "dir_trans": 1.5,
        "cooc_cos": 0.5,
        "cooc_cond": 0.6,
        "tld": 1.0,
    }

    search_grids = {
        "out": [0.6, 1.0, 1.5, 2.0, 2.6],
        "in": [0.2, 0.4, 0.6, 0.9, 1.2],
        "co_out": [0.3, 0.6, 0.9, 1.3, 1.8],
        "co_out_n": [0.0, 0.3, 0.6, 1.0, 1.5],
        "out2": [0.0, 0.15, 0.3, 0.5],
        "co_in": [0.0, 0.15, 0.3, 0.5],
        "in2": [0.0, 0.1, 0.25, 0.4],
        "dir": [6.0, 8.0, 10.0, 12.0, 15.0],
        "dir_trans": [0.0, 0.8, 1.5, 2.2, 3.0],
        "cooc_cos": [0.0, 0.3, 0.5, 0.8, 1.2],
        "cooc_cond": [0.0, 0.3, 0.6, 1.0, 1.5],
        "tld": [0.6, 0.8, 1.0, 1.3, 1.6],
    }

    def compute_ensemble_scores(w_dict, f_dict):
        scores = np.zeros_like(f_dict["out"])
        for k, v in w_dict.items():
            if v > 0:
                scores += v * f_dict[k]
        return scores

    best_scores = compute_ensemble_scores(current_weights, val_features)
    best_recall = compute_recall_at_10(best_scores, val_domains, val_ground_truth)

    print(f"Initial validation Recall@10: {best_recall:.4f}")
    print("Optimizing ensemble weights via coordinate ascent...")
    for p in range(3):
        for param, candidates in search_grids.items():
            base_score = best_scores - current_weights[param] * val_features[param]
            best_cand = current_weights[param]
            for cand in candidates:
                cand_score = base_score + cand * val_features[param]
                recall = compute_recall_at_10(cand_score, val_domains, val_ground_truth)
                if recall > best_recall:
                    best_recall = recall
                    best_cand = cand
            current_weights[param] = best_cand
            best_scores = base_score + best_cand * val_features[param]

    print(f"Optimal weights: {current_weights}")
    print(f"Validation Recall@10: {best_recall:.4f}")

    # Extract features for target domains
    print("Extracting features for target domains...")
    target_out = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_in = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_co_out = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_out_2hop = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_co_in = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_in_2hop = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_direct = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)
    target_tld = np.zeros((len(target_domain_ids), num_trackers), dtype=np.float32)

    target_in_graph = np.array(
        [d in node_to_idx for d in target_domain_ids], dtype=bool
    )
    target_nodes = np.array(
        [node_to_idx[d] for d in target_domain_ids if d in node_to_idx],
        dtype=np.int32,
    )

    for i, d in enumerate(target_domain_ids):
        target_tld[i] = get_tld_prior(d)
        for t in target_direct_links[d]:
            target_direct[i, t] = 1.0

    if len(target_nodes) > 0:
        target_out[target_in_graph] = W_out[target_nodes]
        target_in[target_in_graph] = Z_in[target_nodes]

        A_target_out = A_norm_out[target_nodes, :]
        A_target_in = A_norm_in[target_nodes, :]
        target_co_out[target_in_graph] = (A_target_out @ Z_in).astype(np.float32)
        target_out_2hop[target_in_graph] = (A_target_out @ W_out).astype(np.float32)
        target_co_in[target_in_graph] = (A_target_in @ W_out).astype(np.float32)
        target_in_2hop[target_in_graph] = (A_target_in @ Z_in).astype(np.float32)
        del A_target_out, A_target_in
        gc.collect()

    del A_norm_out, A_norm_in, W_out, Z_in
    gc.collect()

    target_co_out_sum = target_co_out.sum(axis=1, keepdims=True)
    target_co_out_norm = np.where(
        target_co_out_sum > 0, target_co_out / (target_co_out_sum + 1e-5), 0.0
    ).astype(np.float32)

    target_dir_trans = (target_direct @ C_cond).astype(np.float32)
    target_base_evidence = (
        target_out + 0.5 * target_in + 0.4 * target_co_out + 4.0 * target_direct
    )
    target_cooc_cos = (target_base_evidence @ C_affinity).astype(np.float32)
    target_cooc_cond = (target_base_evidence @ C_cond).astype(np.float32)

    target_features = {
        "out": target_out,
        "in": target_in,
        "co_out": target_co_out,
        "co_out_n": target_co_out_norm,
        "out2": target_out_2hop,
        "co_in": target_co_in,
        "in2": target_in_2hop,
        "dir": target_direct,
        "dir_trans": target_dir_trans,
        "cooc_cos": target_cooc_cos,
        "cooc_cond": target_cooc_cond,
        "tld": target_tld,
    }

    target_scores = compute_ensemble_scores(current_weights, target_features)

    # Rank top 10 trackers per domain
    print("Formatting final predictions...")
    top10_target = np.argpartition(-target_scores, 10, axis=1)[:, :10]
    sub_domain_list = []
    sub_tracker_list = []

    for i, d in enumerate(target_domain_ids):
        top_indices = top10_target[i]
        sorted_order = np.argsort(-target_scores[i, top_indices])
        ranked = top_indices[sorted_order]
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

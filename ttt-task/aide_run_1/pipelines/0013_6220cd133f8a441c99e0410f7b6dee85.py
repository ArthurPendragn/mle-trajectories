import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def extract_tld(hostname):
    """Extract top-level domain including recognized two-part TLDs."""
    if not hostname or not isinstance(hostname, str):
        return "unknown"
    parts = hostname.lower().strip().split(".")
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in {
            "co",
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "ac",
            "ne",
            "or",
            "go",
        }:
            return f"{parts[-2]}.{parts[-1]}"
        return parts[-1]
    return parts[0]


def extract_host_from_url(url):
    """Extract clean domain hostname from URL string."""
    if not isinstance(url, str):
        return ""
    s = url.lower().strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0].split(":", 1)[0]
    if s.startswith("www."):
        s = s[4:]
    return s


def compute_recall_at_10(scores, Y_val, val_lengths):
    """Vectorized calculation of Recall@10 on validation ground truth."""
    top10_idx = np.argpartition(-scores, 10, axis=1)[:, :10]
    hits = np.take_along_axis(Y_val, top10_idx, axis=1).sum(axis=1)
    recalls = hits / val_lengths
    return float(np.mean(recalls))


def main():
    start_time = time.time()
    print("=" * 70)
    print("Starting TrackTheTrackers Solution Pipeline (PyArrow + SciPy)")
    print("=" * 70)

    # 1. Load trackers metadata
    print("\n[1/7] Loading trackers metadata...")
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    n_trackers = len(trackers_df)
    print(f"Total candidate trackers: {n_trackers}")

    tracker_id_to_domain_id = dict(
        zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
    )
    domain_id_to_tracker_id = dict(
        zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
    )
    tracker_domain_ids = set(trackers_df["tracking_domain_id"].values)

    # 2. Load target domains
    print("\n[2/7] Loading target domains...")
    target_df = pd.read_csv("input/target.tsv", sep="\t")
    target_domain_ids = target_df["domain_id"].to_numpy()
    target_domain_set = set(target_domain_ids)
    n_target = len(target_domain_ids)
    print(f"Total target domains to predict: {n_target}")

    # 3. Load tracking graph train and setup hold-out validation split
    print("\n[3/7] Loading training tracking graph and creating hold-out split...")
    train_table = pq.read_table(
        "input/tracking_graph_train.parquet",
        columns=["domain_id", "tracking_domain_id", "tracker_id"],
    )
    train_graph_df = train_table.to_pandas()
    del train_table
    gc.collect()

    all_tracked_domains = train_graph_df["domain_id"].unique()
    n_unique_domains = len(all_tracked_domains)
    print(f"Unique tracked domains in training data: {n_unique_domains:,}")

    # Hold-out validation set of 50,000 domains (matching test target set size)
    rng = np.random.RandomState(42)
    shuffled_domains = rng.permutation(all_tracked_domains)
    n_val = 50000
    val_domain_list = shuffled_domains[:n_val]
    val_domain_set = set(val_domain_list)
    val_domain_to_idx = {d: i for i, d in enumerate(val_domain_list)}

    train_domains_list = shuffled_domains[n_val:]
    train_domain_set = set(train_domains_list)
    train_d_to_idx = {d: i for i, d in enumerate(train_domains_list)}
    print(
        f"Train split domains: {len(train_domain_set):,}, Validation split domains: {len(val_domain_set):,}"
    )

    # Split training graph into train split and validation split
    val_mask = np.isin(train_graph_df["domain_id"].to_numpy(), val_domain_list)
    val_graph_df = train_graph_df[val_mask]
    train_graph_split_df = train_graph_df[~val_mask]

    # Construct validation ground-truth matrix Y_val (shape: 50000 x 355)
    val_d_col = val_graph_df["domain_id"].to_numpy()
    val_t_col = val_graph_df["tracker_id"].to_numpy()
    val_row_idx = np.array([val_domain_to_idx[d] for d in val_d_col], dtype=np.int32)
    val_col_idx = val_t_col.astype(np.int32)

    Y_val = np.zeros((n_val, n_trackers), dtype=bool)
    Y_val[val_row_idx, val_col_idx] = True
    Y_val_lengths = Y_val.sum(axis=1).astype(np.float32)

    # Sparse indicator matrix Y_train for training split domains (shape: N_train x 355)
    tr_d_col = train_graph_split_df["domain_id"].to_numpy()
    tr_t_col = train_graph_split_df["tracker_id"].to_numpy()
    tr_row_idx = np.array([train_d_to_idx[d] for d in tr_d_col], dtype=np.int32)
    tr_col_idx = tr_t_col.astype(np.int32)

    Y_train = sp.csr_matrix(
        (np.ones(len(tr_d_col), dtype=np.float32), (tr_row_idx, tr_col_idx)),
        shape=(len(train_domains_list), n_trackers),
    )

    # Global tracker prior from training split
    tracker_train_counts = np.array(Y_train.sum(axis=0)).flatten()
    p_global = (tracker_train_counts / len(train_domains_list)).astype(np.float32)

    # Tracker co-occurrence transition matrix (Collaborative Filtering)
    print("Computing tracker co-occurrence transition matrix...")
    C_cooc = (Y_train.T @ Y_train).toarray()
    diag = np.diag(C_cooc).astype(np.float32)
    C_norm_train = np.zeros_like(C_cooc, dtype=np.float32)
    for i in range(n_trackers):
        if diag[i] > 0:
            C_norm_train[i, :] = C_cooc[i, :] / (diag[i] + 15.0)
        C_norm_train[i, i] = 0.0
    del C_cooc
    gc.collect()

    # 4. Load domain names and extract TLD and category priors
    print("\n[4/7] Processing domain metadata, TLDs, and URL classifications...")
    domains_table = pq.read_table(
        "input/domains.parquet", columns=["domain_id", "domain"]
    )
    all_needed_domains = val_domain_set.union(target_domain_set).union(train_domain_set)

    domains_df = domains_table.to_pandas()
    del domains_table
    gc.collect()

    domains_sub = domains_df[domains_df["domain_id"].isin(all_needed_domains)]
    del domains_df
    gc.collect()
    print(f"Extracted {len(domains_sub):,} domain hostnames for analysis.")

    domain_ids_arr = domains_sub["domain_id"].to_numpy()
    domain_names_arr = domains_sub["domain"].to_numpy()

    domain_to_tld = {}
    domain_to_name = {}
    for d_id, name in zip(domain_ids_arr, domain_names_arr):
        domain_to_tld[d_id] = extract_tld(name)
        domain_to_name[d_id] = name if isinstance(name, str) else ""

    # Build TLD tracker distributions from train split
    train_tld_series = pd.Series(
        [domain_to_tld.get(d, "unknown") for d in tr_d_col], name="tld"
    )
    tld_tracker_df = pd.DataFrame({"tld": train_tld_series, "tracker_id": tr_t_col})
    tld_counts_df = (
        tld_tracker_df.groupby(["tld", "tracker_id"]).size().reset_index(name="count")
    )

    train_domain_tlds = pd.Series(
        [domain_to_tld.get(d, "unknown") for d in train_domains_list], name="tld"
    )
    tld_totals = train_domain_tlds.value_counts().to_dict()

    tld_priors = {}
    grouped_tld = tld_counts_df.groupby("tld")
    for tld_val, grp in grouped_tld:
        tot = tld_totals.get(tld_val, 1)
        cnts = np.zeros(n_trackers, dtype=np.float32)
        cnts[grp["tracker_id"].to_numpy()] = grp["count"].to_numpy()
        tld_priors[tld_val] = (cnts + 10.0 * p_global) / (tot + 10.0)

    # Process URL Classification prior
    print("Processing content category metadata...")
    url_df = pd.read_csv("input/url-classification.csv")
    url_df["host"] = [extract_host_from_url(u) for u in url_df["url"]]
    host_to_cat = dict(zip(url_df["host"], url_df["category"]))
    del url_df
    gc.collect()

    domain_to_cat = {}
    for d_id, name in domain_to_name.items():
        clean_name = name.lower().strip()
        if clean_name.startswith("www."):
            clean_name = clean_name[4:]
        if clean_name in host_to_cat:
            domain_to_cat[d_id] = host_to_cat[clean_name]

    cat_tr_records = []
    for d, t in zip(tr_d_col, tr_t_col):
        if d in domain_to_cat:
            cat_tr_records.append((domain_to_cat[d], t))

    if cat_tr_records:
        cat_df = pd.DataFrame(cat_tr_records, columns=["cat", "tracker_id"])
        cat_counts_df = (
            cat_df.groupby(["cat", "tracker_id"]).size().reset_index(name="count")
        )
        cat_domain_series = pd.Series(
            [domain_to_cat[d] for d in train_domains_list if d in domain_to_cat]
        )
        cat_totals = cat_domain_series.value_counts().to_dict()
        cat_priors = {}
        for cat_val, grp in cat_counts_df.groupby("cat"):
            tot = cat_totals.get(cat_val, 1)
            cnts = np.zeros(n_trackers, dtype=np.float32)
            cnts[grp["tracker_id"].to_numpy()] = grp["count"].to_numpy()
            cat_priors[cat_val] = (cnts + 10.0 * p_global) / (tot + 10.0)
    else:
        cat_priors = {}

    # Hostname text modeling with character n-grams (Naive Bayes)
    print("Fitting character n-gram text model on domain hostnames...")
    sample_train_size = min(150000, len(train_domains_list))
    sample_train_ids = rng.choice(
        train_domains_list, size=sample_train_size, replace=False
    )
    sample_names = [domain_to_name.get(d, "") for d in sample_train_ids]

    vec = CountVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=30, max_features=4000
    )
    X_sample = vec.fit_transform(sample_names)

    sample_d_to_idx = {d: i for i, d in enumerate(sample_train_ids)}
    s_edges_mask = np.isin(tr_d_col, sample_train_ids)
    s_d_col = tr_d_col[s_edges_mask]
    s_t_col = tr_t_col[s_edges_mask]
    s_row = np.array([sample_d_to_idx[d] for d in s_d_col], dtype=np.int32)
    s_col = s_t_col.astype(np.int32)
    Y_sample = sp.csr_matrix(
        (np.ones(len(s_d_col), dtype=np.float32), (s_row, s_col)),
        shape=(sample_train_size, n_trackers),
    )

    W_feat = (Y_sample.T @ X_sample).toarray() + 1.0  # shape: (355, 4000)
    log_W = np.log(W_feat / W_feat.sum(axis=1, keepdims=True))
    log_W_centered = (log_W - log_W.mean(axis=0, keepdims=True)).astype(np.float32)

    del X_sample, Y_sample, W_feat
    gc.collect()

    # 5. Link Graph Propagation
    print("\n[5/7] Processing link graph and propagating neighbor trackers...")
    link_table = pq.read_table(
        "input/link-graph.parquet", columns=["source_domain_id", "target_domain_id"]
    )
    src_all = link_table["source_domain_id"].to_numpy()
    dst_all = link_table["target_domain_id"].to_numpy()
    del link_table
    gc.collect()
    print(f"Total link graph edges loaded: {len(src_all):,}")

    max_domain_id = max(src_all.max(), dst_all.max())
    print(f"Maximum domain ID in link graph: {max_domain_id:,}")

    # Create fast boolean lookup arrays for membership
    is_eval = np.zeros(max_domain_id + 1, dtype=bool)
    is_eval[val_domain_list] = True
    is_eval[target_domain_ids] = True

    is_tracker = np.zeros(max_domain_id + 1, dtype=bool)
    for td in tracker_domain_ids:
        if td <= max_domain_id:
            is_tracker[td] = True

    is_train_split = np.zeros(max_domain_id + 1, dtype=bool)
    for td in train_domains_list:
        if td <= max_domain_id:
            is_train_split[td] = True

    is_all_train = np.zeros(max_domain_id + 1, dtype=bool)
    for td in all_tracked_domains:
        if td <= max_domain_id:
            is_all_train[td] = True

    # 5a. Direct tracker links from eval domains (validation + target)
    eval_src_mask = is_eval[src_all]
    sub_src = src_all[eval_src_mask]
    sub_dst = dst_all[eval_src_mask]

    direct_mask = is_tracker[sub_dst]
    dir_src = sub_src[direct_mask]
    dir_dst = sub_dst[direct_mask]

    M_direct_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    M_direct_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    target_d_to_idx = {d: i for i, d in enumerate(target_domain_ids)}

    for s, d in zip(dir_src, dir_dst):
        tr_id = domain_id_to_tracker_id.get(d)
        if tr_id is not None:
            if s in val_domain_to_idx:
                M_direct_val[val_domain_to_idx[s], tr_id] += 1.0
            elif s in target_d_to_idx:
                M_direct_tgt[target_d_to_idx[s], tr_id] += 1.0

    # 5b. Validation Out-edges to train split
    val_src_mask = np.zeros(max_domain_id + 1, dtype=bool)
    val_src_mask[val_domain_list] = True

    edge_val_out_mask = val_src_mask[sub_src] & is_train_split[sub_dst]
    v_out_src = sub_src[edge_val_out_mask]
    v_out_dst = sub_dst[edge_val_out_mask]

    v_out_r = np.array([val_domain_to_idx[d] for d in v_out_src], dtype=np.int32)
    v_out_c = np.array([train_d_to_idx[d] for d in v_out_dst], dtype=np.int32)

    G_out_val = sp.csr_matrix(
        (np.ones(len(v_out_r), dtype=np.float32), (v_out_r, v_out_c)),
        shape=(n_val, len(train_domains_list)),
    )
    M_out_val = (G_out_val @ Y_train).toarray()
    val_out_deg = np.array(G_out_val.sum(axis=1)).flatten()
    M_out_val_freq = M_out_val / (val_out_deg[:, None] + 2.0)
    M_out_val_cnt = np.log1p(M_out_val)

    del G_out_val, v_out_r, v_out_c, v_out_src, v_out_dst
    gc.collect()

    # 5c. Validation In-edges from train split
    eval_dst_mask = is_eval[dst_all]
    in_sub_src = src_all[eval_dst_mask]
    in_sub_dst = dst_all[eval_dst_mask]

    edge_val_in_mask = val_src_mask[in_sub_dst] & is_train_split[in_sub_src]
    v_in_src = in_sub_src[edge_val_in_mask]
    v_in_dst = in_sub_dst[edge_val_in_mask]

    v_in_r = np.array([val_domain_to_idx[d] for d in v_in_dst], dtype=np.int32)
    v_in_c = np.array([train_d_to_idx[d] for d in v_in_src], dtype=np.int32)

    G_in_val = sp.csr_matrix(
        (np.ones(len(v_in_r), dtype=np.float32), (v_in_r, v_in_c)),
        shape=(n_val, len(train_domains_list)),
    )
    M_in_val = (G_in_val @ Y_train).toarray()
    val_in_deg = np.array(G_in_val.sum(axis=1)).flatten()
    M_in_val_freq = M_in_val / (val_in_deg[:, None] + 2.0)
    M_in_val_cnt = np.log1p(M_in_val)

    del G_in_val, v_in_r, v_in_c, v_in_src, v_in_dst
    gc.collect()

    # 5d. Target Out-edges and In-edges to ALL training domains
    print("Computing graph propagation for target domains using full training set...")
    tgt_mask_arr = np.zeros(max_domain_id + 1, dtype=bool)
    tgt_mask_arr[target_domain_ids] = True

    # Full training matrix Y_full (shape: N_all_train x 355)
    all_train_d_to_idx = {d: i for i, d in enumerate(all_tracked_domains)}
    all_tr_d = train_graph_df["domain_id"].to_numpy()
    all_tr_t = train_graph_df["tracker_id"].to_numpy()
    all_row = np.array([all_train_d_to_idx[d] for d in all_tr_d], dtype=np.int32)
    all_col = all_tr_t.astype(np.int32)
    Y_full = sp.csr_matrix(
        (np.ones(len(all_tr_d), dtype=np.float32), (all_row, all_col)),
        shape=(n_unique_domains, n_trackers),
    )

    edge_tgt_out_mask = tgt_mask_arr[sub_src] & is_all_train[sub_dst]
    tgt_out_src = sub_src[edge_tgt_out_mask]
    tgt_out_dst = sub_dst[edge_tgt_out_mask]

    tgt_out_r = np.array([target_d_to_idx[d] for d in tgt_out_src], dtype=np.int32)
    tgt_out_c = np.array([all_train_d_to_idx[d] for d in tgt_out_dst], dtype=np.int32)

    G_out_tgt = sp.csr_matrix(
        (np.ones(len(tgt_out_r), dtype=np.float32), (tgt_out_r, tgt_out_c)),
        shape=(n_target, n_unique_domains),
    )
    M_out_tgt = (G_out_tgt @ Y_full).toarray()
    tgt_out_deg = np.array(G_out_tgt.sum(axis=1)).flatten()
    M_out_tgt_freq = M_out_tgt / (tgt_out_deg[:, None] + 2.0)
    M_out_tgt_cnt = np.log1p(M_out_tgt)

    del G_out_tgt, tgt_out_r, tgt_out_c, tgt_out_src, tgt_out_dst, sub_src, sub_dst
    gc.collect()

    edge_tgt_in_mask = tgt_mask_arr[in_sub_dst] & is_all_train[in_sub_src]
    tgt_in_src = in_sub_src[edge_tgt_in_mask]
    tgt_in_dst = in_sub_dst[edge_tgt_in_mask]

    tgt_in_r = np.array([target_d_to_idx[d] for d in tgt_in_dst], dtype=np.int32)
    tgt_in_c = np.array([all_train_d_to_idx[d] for d in tgt_in_src], dtype=np.int32)

    G_in_tgt = sp.csr_matrix(
        (np.ones(len(tgt_in_r), dtype=np.float32), (tgt_in_r, tgt_in_c)),
        shape=(n_target, n_unique_domains),
    )
    M_in_tgt = (G_in_tgt @ Y_full).toarray()
    tgt_in_deg = np.array(G_in_tgt.sum(axis=1)).flatten()
    M_in_tgt_freq = M_in_tgt / (tgt_in_deg[:, None] + 2.0)
    M_in_tgt_cnt = np.log1p(M_in_tgt)

    del G_in_tgt, tgt_in_r, tgt_in_c, tgt_in_src, tgt_in_dst, in_sub_src, in_sub_dst
    del src_all, dst_all
    gc.collect()

    # Recompute full co-occurrence matrix on full train_graph for target inference
    C_full = (Y_full.T @ Y_full).toarray()
    diag_full = np.diag(C_full).astype(np.float32)
    C_norm_full = np.zeros_like(C_full, dtype=np.float32)
    for i in range(n_trackers):
        if diag_full[i] > 0:
            C_norm_full[i, :] = C_full[i, :] / (diag_full[i] + 15.0)
        C_norm_full[i, i] = 0.0

    p_global_full = (np.array(Y_full.sum(axis=0)).flatten() / n_unique_domains).astype(
        np.float32
    )
    del C_full, Y_full
    gc.collect()

    # 5e. Assemble Metadata Feature Matrices
    print("Assembling metadata feature matrices...")
    M_direct_val = np.log1p(M_direct_val)
    M_direct_tgt = np.log1p(M_direct_tgt)

    M_tld_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    M_cat_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    M_global_val = np.tile(p_global, (n_val, 1))

    for i, d_id in enumerate(val_domain_list):
        tld_val = domain_to_tld.get(d_id, "unknown")
        M_tld_val[i, :] = tld_priors.get(tld_val, p_global)
        if d_id in domain_to_cat:
            cat_val = domain_to_cat[d_id]
            M_cat_val[i, :] = cat_priors.get(cat_val, p_global)
        else:
            M_cat_val[i, :] = p_global

    val_names = [domain_to_name.get(d, "") for d in val_domain_list]
    X_val_text = vec.transform(val_names)
    text_scores_val = (X_val_text @ log_W_centered.T).astype(np.float32)
    M_text_val = np.clip(text_scores_val, -15.0, 15.0)
    M_text_val = (M_text_val - M_text_val.mean(axis=1, keepdims=True)) / (
        M_text_val.std(axis=1, keepdims=True) + 1e-4
    )

    M_tld_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    M_cat_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    M_global_tgt = np.tile(p_global_full, (n_target, 1))

    for i, d_id in enumerate(target_domain_ids):
        tld_val = domain_to_tld.get(d_id, "unknown")
        M_tld_tgt[i, :] = tld_priors.get(tld_val, p_global_full)
        if d_id in domain_to_cat:
            cat_val = domain_to_cat[d_id]
            M_cat_tgt[i, :] = cat_priors.get(cat_val, p_global_full)
        else:
            M_cat_tgt[i, :] = p_global_full

    tgt_names = [domain_to_name.get(d, "") for d in target_domain_ids]
    X_tgt_text = vec.transform(tgt_names)
    text_scores_tgt = (X_tgt_text @ log_W_centered.T).astype(np.float32)
    M_text_tgt = np.clip(text_scores_tgt, -15.0, 15.0)
    M_text_tgt = (M_text_tgt - M_text_tgt.mean(axis=1, keepdims=True)) / (
        M_text_tgt.std(axis=1, keepdims=True) + 1e-4
    )

    # 6. Validation Evaluation and Coordinate Search Weight Tuning
    print("\n[6/7] Evaluating validation set and tuning component weights...")

    base_recall = compute_recall_at_10(M_global_val, Y_val, Y_val_lengths)
    print(f"Baseline Recall@10 (Global Popularity): {base_recall:.5f}")

    weights = {
        "global": 1.0,
        "tld": 2.5,
        "cat": 1.0,
        "text": 0.6,
        "out_cnt": 3.0,
        "out_freq": 6.0,
        "in_cnt": 1.5,
        "in_freq": 3.0,
        "direct": 25.0,
        "cf": 0.15,
    }

    # Pass 1: Sequential tuning
    # Tune TLD
    best_score = base_recall
    for w in [0.5, 1.0, 2.0, 2.5, 3.5, 5.0]:
        s = weights["global"] * M_global_val + w * M_tld_val
        rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
        if rec > best_score:
            best_score = rec
            weights["tld"] = w
    print(
        f"Recall@10 after adding TLD Prior: {best_score:.5f} (w_tld={weights['tld']})"
    )

    # Tune Category
    for w in [0.0, 0.3, 0.6, 1.0, 1.5]:
        s = (
            weights["global"] * M_global_val
            + weights["tld"] * M_tld_val
            + w * M_cat_val
        )
        rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
        if rec > best_score:
            best_score = rec
            weights["cat"] = w
    print(
        f"Recall@10 after adding Category Prior: {best_score:.5f} (w_cat={weights['cat']})"
    )

    # Tune Text Model
    for w in [0.0, 0.2, 0.5, 0.8, 1.2]:
        s = (
            weights["global"] * M_global_val
            + weights["tld"] * M_tld_val
            + weights["cat"] * M_cat_val
            + w * M_text_val
        )
        rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
        if rec > best_score:
            best_score = rec
            weights["text"] = w
    print(
        f"Recall@10 after adding Text Model: {best_score:.5f} (w_text={weights['text']})"
    )

    # Tune Out-neighbor count & frequency
    for w_cnt in [1.0, 2.5, 4.0, 6.0]:
        for w_frq in [2.0, 5.0, 8.0]:
            s = (
                weights["global"] * M_global_val
                + weights["tld"] * M_tld_val
                + weights["cat"] * M_cat_val
                + weights["text"] * M_text_val
                + w_cnt * M_out_val_cnt
                + w_frq * M_out_val_freq
            )
            rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
            if rec > best_score:
                best_score = rec
                weights["out_cnt"] = w_cnt
                weights["out_freq"] = w_frq
    print(
        f"Recall@10 after Out-neighbor Graph Signals: {best_score:.5f} (w_out_cnt={weights['out_cnt']}, w_out_freq={weights['out_freq']})"
    )

    # Tune In-neighbor count & frequency
    for w_cnt in [0.5, 1.5, 3.0]:
        for w_frq in [1.0, 3.0, 5.0]:
            s = (
                weights["global"] * M_global_val
                + weights["tld"] * M_tld_val
                + weights["cat"] * M_cat_val
                + weights["text"] * M_text_val
                + weights["out_cnt"] * M_out_val_cnt
                + weights["out_freq"] * M_out_val_freq
                + w_cnt * M_in_val_cnt
                + w_frq * M_in_val_freq
            )
            rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
            if rec > best_score:
                best_score = rec
                weights["in_cnt"] = w_cnt
                weights["in_freq"] = w_frq
    print(
        f"Recall@10 after In-neighbor Graph Signals: {best_score:.5f} (w_in_cnt={weights['in_cnt']}, w_in_freq={weights['in_freq']})"
    )

    # Tune Direct Tracker Links
    for w_dir in [10.0, 20.0, 30.0, 45.0]:
        s = (
            weights["global"] * M_global_val
            + weights["tld"] * M_tld_val
            + weights["cat"] * M_cat_val
            + weights["text"] * M_text_val
            + weights["out_cnt"] * M_out_val_cnt
            + weights["out_freq"] * M_out_val_freq
            + weights["in_cnt"] * M_in_val_cnt
            + weights["in_freq"] * M_in_val_freq
            + w_dir * M_direct_val
        )
        rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
        if rec > best_score:
            best_score = rec
            weights["direct"] = w_dir
    print(
        f"Recall@10 after Direct Tracker Links: {best_score:.5f} (w_direct={weights['direct']})"
    )

    # Tune Co-occurrence Collaborative Filtering Diffusion
    base_s = (
        weights["global"] * M_global_val
        + weights["tld"] * M_tld_val
        + weights["cat"] * M_cat_val
        + weights["text"] * M_text_val
        + weights["out_cnt"] * M_out_val_cnt
        + weights["out_freq"] * M_out_val_freq
        + weights["in_cnt"] * M_in_val_cnt
        + weights["in_freq"] * M_in_val_freq
        + weights["direct"] * M_direct_val
    )
    cf_diff = base_s @ C_norm_train
    for w_cf in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        s = base_s + w_cf * cf_diff
        rec = compute_recall_at_10(s, Y_val, Y_val_lengths)
        if rec > best_score:
            best_score = rec
            weights["cf"] = w_cf
    print(
        f"Recall@10 after Tracker Co-occurrence Diffusion: {best_score:.5f} (w_cf={weights['cf']})"
    )

    final_val_recall = best_score
    print("-" * 70)
    print(f"FINAL VALIDATION Recall@10: {final_val_recall:.5f}")
    print("-" * 70)

    # 7. Generate Predictions for Target Domains
    print("\n[7/7] Generating predictions for target domains...")
    S_target = (
        weights["global"] * M_global_tgt
        + weights["tld"] * M_tld_tgt
        + weights["cat"] * M_cat_tgt
        + weights["text"] * M_text_tgt
        + weights["out_cnt"] * M_out_tgt_cnt
        + weights["out_freq"] * M_out_tgt_freq
        + weights["in_cnt"] * M_in_tgt_cnt
        + weights["in_freq"] * M_in_tgt_freq
        + weights["direct"] * M_direct_tgt
    )
    if weights["cf"] > 0:
        S_target += weights["cf"] * (S_target @ C_norm_full)

    # Select top 10 trackers per target domain
    print("Formatting submission for 50,000 target domains...")
    top10_indices = np.argpartition(-S_target, 10, axis=1)[:, :10]

    sub_domain_list = []
    sub_tracker_domain_list = []

    for i, d_id in enumerate(target_domain_ids):
        domain_top10 = top10_indices[i]
        sorted_top10 = domain_top10[np.argsort(-S_target[i, domain_top10])]
        for tr_id in sorted_top10:
            sub_domain_list.append(d_id)
            sub_tracker_domain_list.append(tracker_id_to_domain_id[tr_id])

    submission_df = pd.DataFrame(
        {"domain_id": sub_domain_list, "tracking_domain_id": sub_tracker_domain_list}
    )

    os.makedirs("working", exist_ok=True)
    csv_path = "working/submission.csv"
    tsv_path = "working/submission.tsv"
    submission_df.to_csv(csv_path, sep="\t", index=False)
    submission_df.to_csv(tsv_path, sep="\t", index=False)

    print(f"Submission saved to: {csv_path} and {tsv_path}")
    print(
        f"Total submission rows: {len(submission_df):,} ({len(target_domain_ids):,} domains x 10 trackers)"
    )
    print("\nSubmission preview (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))

    total_time = time.time() - start_time
    print(
        f"\nPipeline finished successfully in {total_time:.1f}s ({total_time / 60:.2f} mins)."
    )
    print(f"Validation Metric (Recall@10): {final_val_recall:.5f}")


if __name__ == "__main__":
    main()

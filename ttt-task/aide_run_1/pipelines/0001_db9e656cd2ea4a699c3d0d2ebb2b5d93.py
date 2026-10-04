import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer


def main():
    start_time = time.time()
    print("=" * 70)
    print("Starting TrackTheTrackers Solution Pipeline")
    print("=" * 70)

    # 1. Load trackers metadata
    print("\n[1/7] Loading trackers metadata...")
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    n_trackers = len(trackers_df)
    print(f"Total candidate trackers: {n_trackers}")

    # Mappings between tracker_id (0-354) and tracking_domain_id
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
    train_graph = pl.read_parquet("input/tracking_graph_train.parquet")
    print(f"Total training edges: {len(train_graph):,}")

    unique_domains = train_graph["domain_id"].unique().to_numpy()
    n_unique_domains = len(unique_domains)
    print(f"Unique tracked domains in training data: {n_unique_domains:,}")

    # Hold-out validation set of 50,000 domains (matching target set size)
    rng = np.random.RandomState(42)
    shuffled_domains = rng.permutation(unique_domains)
    n_val = 50000
    val_domain_list = shuffled_domains[:n_val]
    val_domain_set = set(val_domain_list)
    val_domain_to_idx = {d: i for i, d in enumerate(val_domain_list)}

    # Training split domains
    train_domain_set = set(shuffled_domains[n_val:])
    print(
        f"Train split domains: {len(train_domain_set):,}, Validation split domains: {len(val_domain_set):,}"
    )

    # Filter train graph into train split and validation split
    train_graph_split = train_graph.filter(pl.col("domain_id").is_in(train_domain_set))
    val_graph = train_graph.filter(pl.col("domain_id").is_in(val_domain_set))

    # Construct validation ground-truth matrix Y_val (shape: 50000 x 355)
    val_d_col = val_graph["domain_id"].to_numpy()
    val_t_col = val_graph["tracker_id"].to_numpy()
    val_row_idx = np.array([val_domain_to_idx[d] for d in val_d_col], dtype=np.int32)
    val_col_idx = val_t_col.astype(np.int32)

    Y_val = np.zeros((n_val, n_trackers), dtype=bool)
    Y_val[val_row_idx, val_col_idx] = True
    Y_val_lengths = Y_val.sum(axis=1).astype(np.float32)

    # Global tracker prior from train split
    tracker_counts_df = train_graph_split.group_by("tracker_id").len()
    p_global = np.zeros(n_trackers, dtype=np.float32)
    for row in tracker_counts_df.iter_rows():
        p_global[row[0]] = row[1] / len(train_domain_set)

    # Tracker co-occurrence matrix (item-item collaborative filtering)
    print("Computing tracker co-occurrence transition matrix...")
    train_domains_list = list(train_domain_set)
    train_d_to_idx = {d: i for i, d in enumerate(train_domains_list)}
    tr_d_col = train_graph_split["domain_id"].to_numpy()
    tr_t_col = train_graph_split["tracker_id"].to_numpy()
    tr_row_idx = np.array([train_d_to_idx[d] for d in tr_d_col], dtype=np.int32)
    tr_col_idx = tr_t_col.astype(np.int32)

    import scipy.sparse as sp

    A_train = sp.csr_matrix(
        (np.ones(len(tr_d_col), dtype=np.float32), (tr_row_idx, tr_col_idx)),
        shape=(len(train_domains_list), n_trackers),
    )
    C_cooc = (A_train.T @ A_train).toarray()
    diag = np.diag(C_cooc).astype(np.float32)
    C_norm = np.zeros_like(C_cooc, dtype=np.float32)
    for i in range(n_trackers):
        if diag[i] > 0:
            C_norm[i, :] = C_cooc[i, :] / (diag[i] + 15.0)
        C_norm[i, i] = 0.0  # Zero self-loop

    del A_train, C_cooc
    gc.collect()

    # 4. Load domain names and extract TLD and category priors
    print("\n[4/7] Processing domain metadata, TLDs, and URL classifications...")
    domains_df = pl.read_parquet("input/domains.parquet")
    needed_domains = val_domain_set.union(target_domain_set)
    domains_sub = domains_df.filter(
        pl.col("domain_id").is_in(needed_domains.union(train_domain_set))
    )
    print(f"Extracted {len(domains_sub):,} domain hostnames for analysis.")

    # Helper function for TLD extraction
    def extract_tld(hostname):
        if not hostname or not isinstance(hostname, str):
            return "unknown"
        parts = hostname.lower().strip().split(".")
        if len(parts) >= 2:
            if len(parts) >= 3 and parts[-2] in (
                "co",
                "com",
                "org",
                "net",
                "edu",
                "gov",
                "ac",
            ):
                return f"{parts[-2]}.{parts[-1]}"
            return parts[-1]
        return parts[0]

    domain_ids_arr = domains_sub["domain_id"].to_numpy()
    domain_names_arr = domains_sub["domain"].to_numpy()

    domain_to_tld = {}
    domain_to_name = {}
    for d_id, name in zip(domain_ids_arr, domain_names_arr):
        domain_to_tld[d_id] = extract_tld(name)
        domain_to_name[d_id] = name if isinstance(name, str) else ""

    # TLD tracker distributions
    tld_df = pl.DataFrame(
        {"domain_id": list(domain_to_tld.keys()), "tld": list(domain_to_tld.values())}
    )
    train_tld = train_graph_split.join(tld_df, on="domain_id")
    tld_tr_counts = train_tld.group_by(["tld", "tracker_id"]).len().to_pandas()
    tld_domain_counts = (
        train_tld.select(["domain_id", "tld"])
        .unique()
        .group_by("tld")
        .len()
        .to_pandas()
    )
    tld_tot_dict = dict(zip(tld_domain_counts["tld"], tld_domain_counts["len"]))

    tld_tracker_dict = {}
    for _, row in tld_tr_counts.iterrows():
        tld_val = row["tld"]
        tr_id = int(row["tracker_id"])
        c = row["len"]
        if tld_val not in tld_tracker_dict:
            tld_tracker_dict[tld_val] = np.zeros(n_trackers, dtype=np.float32)
        tld_tracker_dict[tld_val][tr_id] = c

    tld_priors = {}
    for tld_val, counts in tld_tracker_dict.items():
        tot = tld_tot_dict.get(tld_val, 1)
        tld_priors[tld_val] = (counts + 10.0 * p_global) / (tot + 10.0)

    # URL Classification prior
    print("Processing content category metadata...")

    def extract_host_from_url(url):
        if not isinstance(url, str):
            return ""
        s = url.lower().strip()
        if "://" in s:
            s = s.split("://", 1)[1]
        s = s.split("/", 1)[0].split(":", 1)[0]
        if s.startswith("www."):
            s = s[4:]
        return s

    url_df = pd.read_csv("input/url-classification.csv")
    url_df["host"] = [extract_host_from_url(u) for u in url_df["url"]]
    host_to_cat = dict(zip(url_df["host"], url_df["category"]))

    domain_to_cat = {}
    for d_id, name in domain_to_name.items():
        clean_name = name.lower().strip()
        if clean_name.startswith("www."):
            clean_name = clean_name[4:]
        if clean_name in host_to_cat:
            domain_to_cat[d_id] = host_to_cat[clean_name]

    cat_df = pl.DataFrame(
        {"domain_id": list(domain_to_cat.keys()), "cat": list(domain_to_cat.values())}
    )
    train_cat = train_graph_split.join(cat_df, on="domain_id")
    cat_tr_counts = train_cat.group_by(["cat", "tracker_id"]).len().to_pandas()
    cat_domain_counts = (
        train_cat.select(["domain_id", "cat"])
        .unique()
        .group_by("cat")
        .len()
        .to_pandas()
    )
    cat_tot_dict = dict(zip(cat_domain_counts["cat"], cat_domain_counts["len"]))

    cat_tracker_dict = {}
    for _, row in cat_tr_counts.iterrows():
        cat_val = row["cat"]
        tr_id = int(row["tracker_id"])
        c = row["len"]
        if cat_val not in cat_tracker_dict:
            cat_tracker_dict[cat_val] = np.zeros(n_trackers, dtype=np.float32)
        cat_tracker_dict[cat_val][tr_id] = c

    cat_priors = {}
    for cat_val, counts in cat_tracker_dict.items():
        tot = cat_tot_dict.get(cat_val, 1)
        cat_priors[cat_val] = (counts + 10.0 * p_global) / (tot + 10.0)

    # Domain text modeling with character n-grams (Naive Bayes)
    print("Fitting character n-gram text model on domain hostnames...")
    sample_train_ids = rng.choice(
        train_domains_list, size=min(150000, len(train_domains_list)), replace=False
    )
    sample_names = [domain_to_name.get(d, "") for d in sample_train_ids]

    vec = CountVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=50, max_features=4000
    )
    X_sample = vec.fit_transform(sample_names)

    sample_d_to_idx = {d: i for i, d in enumerate(sample_train_ids)}
    s_edges = train_graph_split.filter(pl.col("domain_id").is_in(set(sample_train_ids)))
    s_d_col = s_edges["domain_id"].to_numpy()
    s_t_col = s_edges["tracker_id"].to_numpy()
    s_row = np.array([sample_d_to_idx[d] for d in s_d_col], dtype=np.int32)
    s_col = s_t_col.astype(np.int32)
    Y_sample = sp.csr_matrix(
        (np.ones(len(s_d_col), dtype=np.float32), (s_row, s_col)),
        shape=(len(sample_train_ids), n_trackers),
    )

    W_feat = (Y_sample.T @ X_sample).toarray() + 1.0  # shape: (355, 4000)
    log_W = np.log(W_feat / W_feat.sum(axis=1, keepdims=True))
    log_W_centered = (log_W - log_W.mean(axis=0, keepdims=True)).astype(np.float32)

    del X_sample, Y_sample, W_feat, domains_df
    gc.collect()

    # 5. Link Graph Propagation
    print("\n[5/7] Processing link graph and propagating neighbor trackers...")
    link_graph = pl.read_parquet("input/link-graph.parquet")
    print(f"Total link graph edges: {len(link_graph):,}")

    trackers_pl = pl.DataFrame(
        {
            "tracking_domain_id": trackers_df["tracking_domain_id"].to_numpy(),
            "tracker_id": trackers_df["tracker_id"].to_numpy(),
        }
    )

    # Direct links to tracker domains from eval domains (validation + target)
    eval_domains = val_domain_set.union(target_domain_set)
    direct_edges = link_graph.filter(
        pl.col("source_domain_id").is_in(eval_domains)
        & pl.col("target_domain_id").is_in(tracker_domain_ids)
    )
    direct_joined = direct_edges.join(
        trackers_pl, left_on="target_domain_id", right_on="tracking_domain_id"
    )
    direct_counts = direct_joined.group_by(["source_domain_id", "tracker_id"]).len()

    # Out-edges and In-edges for Validation set (pointing to train split)
    print("Computing graph neighbor signals for validation domains...")
    val_out_edges = link_graph.filter(
        pl.col("source_domain_id").is_in(val_domain_set)
        & pl.col("target_domain_id").is_in(train_domain_set)
    )
    val_out_joined = val_out_edges.join(
        train_graph_split.select(["domain_id", "tracker_id"]),
        left_on="target_domain_id",
        right_on="domain_id",
    )
    val_out_counts = val_out_joined.group_by(["source_domain_id", "tracker_id"]).len()

    val_in_edges = link_graph.filter(
        pl.col("target_domain_id").is_in(val_domain_set)
        & pl.col("source_domain_id").is_in(train_domain_set)
    )
    val_in_joined = val_in_edges.join(
        train_graph_split.select(["domain_id", "tracker_id"]),
        left_on="source_domain_id",
        right_on="domain_id",
    )
    val_in_counts = val_in_joined.group_by(["target_domain_id", "tracker_id"]).len()

    # Assemble Validation Feature Matrices
    print("Assembling validation feature matrices...")
    M_direct_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    val_direct = direct_counts.filter(pl.col("source_domain_id").is_in(val_domain_set))
    for row in val_direct.iter_rows():
        r = val_domain_to_idx[row[0]]
        c = row[1]
        M_direct_val[r, c] = float(row[2])

    M_out_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    for row in val_out_counts.iter_rows():
        r = val_domain_to_idx[row[0]]
        c = row[1]
        M_out_val[r, c] = float(row[2])

    M_in_val = np.zeros((n_val, n_trackers), dtype=np.float32)
    for row in val_in_counts.iter_rows():
        r = val_domain_to_idx[row[0]]
        c = row[1]
        M_in_val[r, c] = float(row[2])

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
    # Clip and normalize
    M_text_val = np.clip(text_scores_val, -15.0, 15.0)
    M_text_val = (M_text_val - M_text_val.mean(axis=1, keepdims=True)) / (
        M_text_val.std(axis=1, keepdims=True) + 1e-4
    )

    # Log1p transform for counts
    M_direct_val = np.log1p(M_direct_val)
    M_out_val = np.log1p(M_out_val)
    M_in_val = np.log1p(M_in_val)

    # 6. Validation Evaluation and Weight Tuning
    print("\n[6/7] Evaluating validation set and tuning component weights...")

    def compute_recall_at_10(scores):
        # Fully vectorized Recall@10 computation
        top10_idx = np.argpartition(-scores, 10, axis=1)[:, :10]
        hits = np.take_along_axis(Y_val, top10_idx, axis=1).sum(axis=1)
        recalls = hits / Y_val_lengths
        return float(recalls.mean())

    # Initial baseline using only global prior
    base_recall = compute_recall_at_10(M_global_val)
    print(f"Baseline Recall@10 (Global Popularity): {base_recall:.5f}")

    # Coordinate search for optimal weights
    best_weights = {
        "global": 1.0,
        "tld": 2.0,
        "cat": 1.0,
        "text": 0.5,
        "out": 5.0,
        "in": 2.5,
        "direct": 15.0,
        "cf": 0.15,
    }

    # Step 1: Tune TLD weight
    best_score = base_recall
    for w_tld in [0.5, 1.0, 2.0, 3.0, 4.0]:
        s = best_weights["global"] * M_global_val + w_tld * M_tld_val
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["tld"] = w_tld
    print(
        f"Recall@10 after adding TLD Prior: {best_score:.5f} (w_tld={best_weights['tld']})"
    )

    # Step 2: Tune Category weight
    for w_cat in [0.0, 0.3, 0.6, 1.0, 1.5]:
        s = (
            best_weights["global"] * M_global_val
            + best_weights["tld"] * M_tld_val
            + w_cat * M_cat_val
        )
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["cat"] = w_cat
    print(
        f"Recall@10 after adding Category Prior: {best_score:.5f} (w_cat={best_weights['cat']})"
    )

    # Step 3: Tune Text model weight
    for w_txt in [0.0, 0.2, 0.5, 0.8, 1.2]:
        s = (
            best_weights["global"] * M_global_val
            + best_weights["tld"] * M_tld_val
            + best_weights["cat"] * M_cat_val
            + w_txt * M_text_val
        )
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["text"] = w_txt
    print(
        f"Recall@10 after adding Hostname Text Model: {best_score:.5f} (w_text={best_weights['text']})"
    )

    # Step 4: Tune Out-neighbor graph weight
    for w_out in [1.0, 3.0, 5.0, 7.0, 10.0]:
        s = (
            best_weights["global"] * M_global_val
            + best_weights["tld"] * M_tld_val
            + best_weights["cat"] * M_cat_val
            + best_weights["text"] * M_text_val
            + w_out * M_out_val
        )
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["out"] = w_out
    print(
        f"Recall@10 after adding Out-neighbor Graph Propagation: {best_score:.5f} (w_out={best_weights['out']})"
    )

    # Step 5: Tune In-neighbor graph weight
    for w_in in [0.5, 1.5, 2.5, 4.0, 6.0]:
        s = (
            best_weights["global"] * M_global_val
            + best_weights["tld"] * M_tld_val
            + best_weights["cat"] * M_cat_val
            + best_weights["text"] * M_text_val
            + best_weights["out"] * M_out_val
            + w_in * M_in_val
        )
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["in"] = w_in
    print(
        f"Recall@10 after adding In-neighbor Graph Propagation: {best_score:.5f} (w_in={best_weights['in']})"
    )

    # Step 6: Tune Direct tracker link weight
    for w_dir in [5.0, 10.0, 15.0, 20.0, 30.0]:
        s = (
            best_weights["global"] * M_global_val
            + best_weights["tld"] * M_tld_val
            + best_weights["cat"] * M_cat_val
            + best_weights["text"] * M_text_val
            + best_weights["out"] * M_out_val
            + best_weights["in"] * M_in_val
            + w_dir * M_direct_val
        )
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["direct"] = w_dir
    print(
        f"Recall@10 after adding Direct Tracker Links: {best_score:.5f} (w_direct={best_weights['direct']})"
    )

    # Step 7: Tune Collaborative Filtering (Co-occurrence diffusion) parameter
    base_s = (
        best_weights["global"] * M_global_val
        + best_weights["tld"] * M_tld_val
        + best_weights["cat"] * M_cat_val
        + best_weights["text"] * M_text_val
        + best_weights["out"] * M_out_val
        + best_weights["in"] * M_in_val
        + best_weights["direct"] * M_direct_val
    )
    cf_diff = base_s @ C_norm
    for w_cf in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        s = base_s + w_cf * cf_diff
        rec = compute_recall_at_10(s)
        if rec > best_score:
            best_score = rec
            best_weights["cf"] = w_cf
    print(
        f"Recall@10 after Co-occurrence Collaborative Filtering: {best_score:.5f} (w_cf={best_weights['cf']})"
    )

    final_val_recall = best_score
    print("-" * 70)
    print(f"FINAL VALIDATION Recall@10: {final_val_recall:.5f}")
    print("-" * 70)

    # 7. Generate Predictions for Target Domains
    print("\n[7/7] Generating predictions for target domains...")
    # For target domains, use ALL training data (train + validation)
    all_train_domains = set(unique_domains)
    target_d_to_idx = {d: i for i, d in enumerate(target_domain_ids)}

    # Target out-edges and in-edges connecting to full training set
    target_out_edges = link_graph.filter(
        pl.col("source_domain_id").is_in(target_domain_set)
        & pl.col("target_domain_id").is_in(all_train_domains)
    )
    target_out_joined = target_out_edges.join(
        train_graph.select(["domain_id", "tracker_id"]),
        left_on="target_domain_id",
        right_on="domain_id",
    )
    target_out_counts = target_out_joined.group_by(
        ["source_domain_id", "tracker_id"]
    ).len()

    target_in_edges = link_graph.filter(
        pl.col("target_domain_id").is_in(target_domain_set)
        & pl.col("source_domain_id").is_in(all_train_domains)
    )
    target_in_joined = target_in_edges.join(
        train_graph.select(["domain_id", "tracker_id"]),
        left_on="source_domain_id",
        right_on="domain_id",
    )
    target_in_counts = target_in_joined.group_by(
        ["target_domain_id", "tracker_id"]
    ).len()

    # Recompute full co-occurrence matrix on full train_graph for target inference
    all_d_to_idx = {d: i for i, d in enumerate(unique_domains)}
    all_tr_d = train_graph["domain_id"].to_numpy()
    all_tr_t = train_graph["tracker_id"].to_numpy()
    all_row = np.array([all_d_to_idx[d] for d in all_tr_d], dtype=np.int32)
    all_col = all_tr_t.astype(np.int32)
    A_all = sp.csr_matrix(
        (np.ones(len(all_tr_d), dtype=np.float32), (all_row, all_col)),
        shape=(n_unique_domains, n_trackers),
    )
    C_all = (A_all.T @ A_all).toarray()
    diag_all = np.diag(C_all).astype(np.float32)
    C_norm_full = np.zeros_like(C_all, dtype=np.float32)
    for i in range(n_trackers):
        if diag_all[i] > 0:
            C_norm_full[i, :] = C_all[i, :] / (diag_all[i] + 15.0)
        C_norm_full[i, i] = 0.0

    del A_all, C_all, link_graph
    gc.collect()

    # Assemble Target Feature Matrices
    M_direct_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    tgt_direct = direct_counts.filter(
        pl.col("source_domain_id").is_in(target_domain_set)
    )
    for row in tgt_direct.iter_rows():
        r = target_d_to_idx[row[0]]
        c = row[1]
        M_direct_tgt[r, c] = float(row[2])

    M_out_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    for row in target_out_counts.iter_rows():
        r = target_d_to_idx[row[0]]
        c = row[1]
        M_out_tgt[r, c] = float(row[2])

    M_in_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    for row in target_in_counts.iter_rows():
        r = target_d_to_idx[row[0]]
        c = row[1]
        M_in_tgt[r, c] = float(row[2])

    M_tld_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    M_cat_tgt = np.zeros((n_target, n_trackers), dtype=np.float32)
    M_global_tgt = np.tile(p_global, (n_target, 1))

    for i, d_id in enumerate(target_domain_ids):
        tld_val = domain_to_tld.get(d_id, "unknown")
        M_tld_tgt[i, :] = tld_priors.get(tld_val, p_global)
        if d_id in domain_to_cat:
            cat_val = domain_to_cat[d_id]
            M_cat_tgt[i, :] = cat_priors.get(cat_val, p_global)
        else:
            M_cat_tgt[i, :] = p_global

    tgt_names = [domain_to_name.get(d, "") for d in target_domain_ids]
    X_tgt_text = vec.transform(tgt_names)
    text_scores_tgt = (X_tgt_text @ log_W_centered.T).astype(np.float32)
    M_text_tgt = np.clip(text_scores_tgt, -15.0, 15.0)
    M_text_tgt = (M_text_tgt - M_text_tgt.mean(axis=1, keepdims=True)) / (
        M_text_tgt.std(axis=1, keepdims=True) + 1e-4
    )

    M_direct_tgt = np.log1p(M_direct_tgt)
    M_out_tgt = np.log1p(M_out_tgt)
    M_in_tgt = np.log1p(M_in_tgt)

    # Compute final combined scores for target domains
    S_target = (
        best_weights["global"] * M_global_tgt
        + best_weights["tld"] * M_tld_tgt
        + best_weights["cat"] * M_cat_tgt
        + best_weights["text"] * M_text_tgt
        + best_weights["out"] * M_out_tgt
        + best_weights["in"] * M_in_tgt
        + best_weights["direct"] * M_direct_tgt
    )
    if best_weights["cf"] > 0:
        S_target += best_weights["cf"] * (S_target @ C_norm_full)

    # Select top 10 trackers per domain
    print("Selecting top 10 trackers per target domain and formatting submission...")
    top10_indices = np.argpartition(-S_target, 10, axis=1)[:, :10]

    sub_domain_list = []
    sub_tracker_domain_list = []

    for i, d_id in enumerate(target_domain_ids):
        domain_top10 = top10_indices[i]
        # Sort top 10 by descending score
        sorted_top10 = domain_top10[np.argsort(-S_target[i, domain_top10])]
        for tr_id in sorted_top10:
            sub_domain_list.append(d_id)
            sub_tracker_domain_list.append(tracker_id_to_domain_id[tr_id])

    submission_df = pd.DataFrame(
        {"domain_id": sub_domain_list, "tracking_domain_id": sub_tracker_domain_list}
    )

    # Save submission file in ./working/submission.csv (and .tsv)
    os.makedirs("working", exist_ok=True)
    csv_path = "working/submission.csv"
    tsv_path = "working/submission.tsv"
    submission_df.to_csv(csv_path, sep="\t", index=False)
    submission_df.to_csv(tsv_path, sep="\t", index=False)

    print(f"Submission saved to: {csv_path}")
    print(
        f"Total submission rows: {len(submission_df):,} (covering {len(target_domain_ids):,} domains x 10 trackers)"
    )
    print("\nSubmission preview (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))

    total_time = time.time() - start_time
    print(
        f"\nExecution completed successfully in {total_time:.1f} seconds ({total_time / 60:.2f} minutes)."
    )
    print(f"Hold-out Validation Metric (Recall@10): {final_val_recall:.5f}")


if __name__ == "__main__":
    main()

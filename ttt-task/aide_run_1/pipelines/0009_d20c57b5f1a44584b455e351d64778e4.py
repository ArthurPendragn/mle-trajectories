import gc
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Pre-defined multi-part ccTLDs for robust domain parsing
SECOND_LEVEL_TLDS = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "edu.au",
    "com.br",
    "org.br",
    "co.jp",
    "ne.jp",
    "or.jp",
    "go.jp",
    "co.kr",
    "com.tw",
    "com.tr",
    "com.ar",
    "com.mx",
    "co.in",
    "net.in",
    "org.in",
    "com.ru",
    "co.za",
    "com.cn",
    "net.cn",
    "org.cn",
    "com.sg",
    "com.hk",
    "com.pl",
    "com.ua",
    "com.eg",
    "com.pk",
    "com.ng",
    "co.nz",
    "org.nz",
    "com.co",
    "com.ve",
    "com.pe",
    "com.ph",
    "co.id",
    "com.my",
    "com.vn",
    "co.th",
    "com.sa",
}


def extract_tld(host: str) -> str:
    """Extract top-level domain or common two-part ccTLD from hostname."""
    if not host or not isinstance(host, str):
        return "unknown"
    host = host.strip().lower()
    if ":" in host:
        host = host.split(":")[0]
    parts = host.split(".")
    if len(parts) >= 3:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in SECOND_LEVEL_TLDS or parts[-2] in {
            "co",
            "com",
            "org",
            "net",
            "gov",
            "edu",
            "ac",
            "ne",
            "or",
            "go",
        }:
            return two_part
    if len(parts) >= 2:
        return parts[-1]
    return host


def extract_root_domain(host: str) -> str:
    """Extract eTLD+1 (registered root domain) from hostname."""
    if not host or not isinstance(host, str):
        return ""
    host = host.strip().lower()
    if ":" in host:
        host = host.split(":")[0]
    parts = host.split(".")
    if len(parts) <= 1:
        return host
    if len(parts) >= 3:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in SECOND_LEVEL_TLDS or parts[-2] in {
            "co",
            "com",
            "org",
            "net",
            "gov",
            "edu",
            "ac",
            "ne",
            "or",
            "go",
        }:
            return f"{parts[-3]}.{two_part}"
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return host


def url_to_root(u: str) -> str:
    """Extract registered root domain from a raw URL string."""
    if not u or not isinstance(u, str):
        return ""
    if "://" in u:
        u = u.split("://", 1)[1]
    host = u.split("/", 1)[0]
    return extract_root_domain(host)


def main():
    print("Loading tracker metadata and targets...")
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    tracker_id_to_domain_id = dict(
        zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
    )

    tracking_domain_to_tracker_ids = {}
    for tr_id, tr_dom in zip(
        trackers_df["tracker_id"], trackers_df["tracking_domain_id"]
    ):
        tracking_domain_to_tracker_ids.setdefault(tr_dom, []).append(tr_id)

    tracker_domain_ids_set = set(trackers_df["tracking_domain_id"])
    num_trackers = len(trackers_df)

    target_df = pd.read_csv("input/target.tsv", sep="\t")
    target_domain_ids = target_df["domain_id"].to_numpy().astype(np.int64)

    print("Loading tracking_graph_train.parquet...")
    train_tg_table = pq.read_table(
        "input/tracking_graph_train.parquet",
        columns=["domain_id", "tracker_id"],
    )
    train_tg_df = train_tg_table.to_pandas()
    del train_tg_table
    gc.collect()

    train_tg_df = train_tg_df.drop_duplicates(subset=["domain_id", "tracker_id"])

    all_train_domains = train_tg_df["domain_id"].unique()
    available_train_domains = np.setdiff1d(
        all_train_domains, target_domain_ids, assume_unique=True
    )

    # 10,000 domain hold-out validation set
    np.random.seed(42)
    val_size = min(10000, len(available_train_domains))
    val_domain_ids = np.random.choice(
        available_train_domains, size=val_size, replace=False
    )

    # Split tracking data into validation and train
    val_mask = np.isin(train_tg_df["domain_id"].to_numpy(), val_domain_ids)
    val_tg_df = train_tg_df[val_mask]
    train_tg_split_df = train_tg_df[~val_mask]
    del train_tg_df
    gc.collect()

    # Ground truth for validation
    val_grouped = val_tg_df.groupby("domain_id")["tracker_id"].agg(set).to_dict()
    del val_tg_df
    gc.collect()

    # Build train tracker profiles using fast NumPy indexing
    train_tg_split_df = train_tg_split_df.sort_values("domain_id")
    dom_arr = train_tg_split_df["domain_id"].to_numpy(dtype=np.int64)
    tr_arr = train_tg_split_df["tracker_id"].to_numpy(dtype=np.int32)
    del train_tg_split_df
    gc.collect()

    unique_doms, idx_starts = np.unique(dom_arr, return_index=True)
    idx_splits = np.split(tr_arr, idx_starts[1:])
    train_domain_tracker_dict = dict(zip(unique_doms, idx_splits))
    del dom_arr, tr_arr, unique_doms, idx_starts, idx_splits
    gc.collect()

    query_domain_set = set(val_domain_ids) | set(target_domain_ids)
    query_domains_list = list(query_domain_set)
    query_id_to_idx = {qid: idx for idx, qid in enumerate(query_domains_list)}
    query_domains_arr = np.array(query_domains_list, dtype=np.int64)
    num_query = len(query_domains_list)

    print(
        f"Query domains: {num_query} (Validation: {len(val_domain_ids)}, Target: {len(target_domain_ids)})"
    )

    # Load domains.parquet and filter for needed hostnames
    print("Loading domains.parquet...")
    needed_domains_set = query_domain_set | set(train_domain_tracker_dict.keys())
    domains_table = pq.read_table(
        "input/domains.parquet", columns=["domain_id", "domain"]
    )
    domains_df = domains_table.to_pandas()
    del domains_table
    gc.collect()

    domains_filtered = domains_df[domains_df["domain_id"].isin(needed_domains_set)]
    domain_id_to_host = dict(
        zip(
            domains_filtered["domain_id"].to_numpy(),
            domains_filtered["domain"].to_numpy(),
        )
    )
    del domains_df, domains_filtered
    gc.collect()

    # Load url-classification.csv for content category priors
    print("Loading url-classification.csv...")
    cat_smoothed_prior = {}
    if os.path.exists("input/url-classification.csv"):
        url_df = pd.read_csv(
            "input/url-classification.csv", usecols=["url", "category"]
        )
        url_roots = [url_to_root(u) for u in url_df["url"].dropna().to_list()]
        root_to_cat = dict(zip(url_roots, url_df["category"].dropna().to_list()))
        del url_df, url_roots
        gc.collect()

        cat_counts = {}
        cat_tracker_counts = {}
        for dom_id, trackers in train_domain_tracker_dict.items():
            host = domain_id_to_host.get(dom_id, "")
            root = extract_root_domain(host)
            cat = root_to_cat.get(root)
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if cat not in cat_tracker_counts:
                    cat_tracker_counts[cat] = np.zeros(num_trackers, dtype=np.float64)
                cat_tracker_counts[cat][trackers] += 1.0

    # Build empirical Bayes TLD tracker distribution
    print("Computing empirical Bayes TLD & Category priors...")
    tld_counts = {}
    tld_tracker_counts = {}
    global_tracker_counts = np.zeros(num_trackers, dtype=np.float64)

    for dom_id, trackers in train_domain_tracker_dict.items():
        host = domain_id_to_host.get(dom_id, "")
        tld = extract_tld(host)
        tld_counts[tld] = tld_counts.get(tld, 0) + 1
        if tld not in tld_tracker_counts:
            tld_tracker_counts[tld] = np.zeros(num_trackers, dtype=np.float64)
        tld_tracker_counts[tld][trackers] += 1.0
        global_tracker_counts[trackers] += 1.0

    total_train_domains = len(train_domain_tracker_dict)
    global_prior = (global_tracker_counts + 1.0) / (total_train_domains + num_trackers)

    smoothing_m = 15.0
    tld_smoothed_prior = {}
    for tld, count in tld_counts.items():
        smoothed = (tld_tracker_counts[tld] + smoothing_m * global_prior) / (
            count + smoothing_m
        )
        tld_smoothed_prior[tld] = smoothed.astype(np.float32)

    # Compute smoothed category priors
    if os.path.exists("input/url-classification.csv") and "cat_counts" in locals():
        for cat, count in cat_counts.items():
            smoothed_cat = (cat_tracker_counts[cat] + smoothing_m * global_prior) / (
                count + smoothing_m
            )
            cat_smoothed_prior[cat] = smoothed_cat.astype(np.float32)

    prior_matrix = np.zeros((num_query, num_trackers), dtype=np.float32)
    for qid, qidx in query_id_to_idx.items():
        host = domain_id_to_host.get(qid, "")
        tld = extract_tld(host)
        tld_p = tld_smoothed_prior.get(tld, global_prior.astype(np.float32))

        root = extract_root_domain(host)
        cat = root_to_cat.get(root) if "root_to_cat" in locals() else None
        if cat and cat in cat_smoothed_prior:
            prior_matrix[qidx] = 0.65 * tld_p + 0.35 * cat_smoothed_prior[cat]
        else:
            prior_matrix[qidx] = tld_p

    # Compute root-domain sibling matching boost
    print("Indexing sister domains across identical root domains...")
    root_domain_trackers = {}
    for dom_id, trackers in train_domain_tracker_dict.items():
        host = domain_id_to_host.get(dom_id, "")
        root = extract_root_domain(host)
        if root:
            if root not in root_domain_trackers:
                root_domain_trackers[root] = []
            root_domain_trackers[root].append(trackers)

    root_boost_scores = np.zeros((num_query, num_trackers), dtype=np.float32)
    for qid, qidx in query_id_to_idx.items():
        host = domain_id_to_host.get(qid, "")
        root = extract_root_domain(host)
        if root and root in root_domain_trackers:
            for tr_list in root_domain_trackers[root]:
                root_boost_scores[qidx, tr_list] += 8.0

    # Compute tracker co-occurrence transition matrix
    print("Computing tracker stack co-occurrence transition matrix...")
    cooc_matrix = np.zeros((num_trackers, num_trackers), dtype=np.float32)
    tracker_freq = np.zeros(num_trackers, dtype=np.float32)

    for trackers in train_domain_tracker_dict.values():
        if len(trackers) == 0:
            continue
        tracker_freq[trackers] += 1.0
        for t_a in trackers:
            cooc_matrix[t_a, trackers] += 1.0

    cooc_transition = cooc_matrix / (tracker_freq[:, None] + 10.0)
    np.fill_diagonal(cooc_transition, 0.0)

    # Process link-graph.parquet safely and efficiently
    print("Processing link-graph.parquet...")
    lg_table = pq.read_table(
        "input/link-graph.parquet",
        columns=["source_domain_id", "target_domain_id"],
    )
    src_all = lg_table.column("source_domain_id").to_numpy(zero_copy_only=False)
    tgt_all = lg_table.column("target_domain_id").to_numpy(zero_copy_only=False)
    del lg_table
    gc.collect()

    max_edge_id = max(int(src_all.max()), int(tgt_all.max()))
    min_edge_id = min(int(src_all.min()), int(tgt_all.min()))

    # Build robust is_query index clamped to max_edge_id to avoid IndexError
    if min_edge_id >= 0 and max_edge_id < 200_000_000:
        valid_query = query_domains_arr[
            (query_domains_arr >= 0) & (query_domains_arr <= max_edge_id)
        ]
        is_query = np.zeros(max_edge_id + 1, dtype=bool)
        is_query[valid_query] = True
        mask_src = is_query[src_all]
        mask_tgt = is_query[tgt_all]
        del is_query
    else:
        q_sorted = np.sort(query_domains_arr)
        mask_src = np.isin(src_all, q_sorted)
        mask_tgt = np.isin(tgt_all, q_sorted)

    matched_mask = mask_src | mask_tgt
    src_matched = src_all[matched_mask]
    tgt_matched = tgt_all[matched_mask]
    mask_src_matched = mask_src[matched_mask]
    mask_tgt_matched = mask_tgt[matched_mask]
    del src_all, tgt_all, matched_mask, mask_src, mask_tgt
    gc.collect()

    print(f"Extracted {len(src_matched)} active edges for query neighborhoods.")

    direct_scores = np.zeros((num_query, num_trackers), dtype=np.float32)
    graph_scores = np.zeros((num_query, num_trackers), dtype=np.float32)
    graph_total_weights = np.zeros(num_query, dtype=np.float32)

    # 1. Direct links to tracker hostnames
    tracker_dom_ids = np.array(list(tracker_domain_ids_set), dtype=np.int64)
    if min_edge_id >= 0 and max_edge_id < 200_000_000:
        valid_trackers = tracker_dom_ids[
            (tracker_dom_ids >= 0) & (tracker_dom_ids <= max_edge_id)
        ]
        is_tracker = np.zeros(max_edge_id + 1, dtype=bool)
        is_tracker[valid_trackers] = True
        is_tgt_tracker = is_tracker[tgt_matched]
        is_src_tracker = is_tracker[src_matched]
        del is_tracker
    else:
        is_tgt_tracker = np.isin(tgt_matched, tracker_dom_ids)
        is_src_tracker = np.isin(src_matched, tracker_dom_ids)

    direct_mask = mask_src_matched & is_tgt_tracker
    if np.any(direct_mask):
        direct_src = src_matched[direct_mask]
        direct_tgt = tgt_matched[direct_mask]
        for s, t in zip(direct_src, direct_tgt):
            u_idx = query_id_to_idx[s]
            for tr_id in tracking_domain_to_tracker_ids.get(t, []):
                direct_scores[u_idx, tr_id] = max(direct_scores[u_idx, tr_id], 6.0)

    src_direct_mask = mask_tgt_matched & is_src_tracker
    if np.any(src_direct_mask):
        st_src = src_matched[src_direct_mask]
        st_tgt = tgt_matched[src_direct_mask]
        for s, t in zip(st_src, st_tgt):
            u_idx = query_id_to_idx[t]
            for tr_id in tracking_domain_to_tracker_ids.get(s, []):
                direct_scores[u_idx, tr_id] = max(direct_scores[u_idx, tr_id], 4.0)

    # 2. Outbound links: query domain -> train domain
    train_keys_arr = np.fromiter(train_domain_tracker_dict.keys(), dtype=np.int64)
    if min_edge_id >= 0 and max_edge_id < 200_000_000:
        valid_train = train_keys_arr[
            (train_keys_arr >= 0) & (train_keys_arr <= max_edge_id)
        ]
        is_train = np.zeros(max_edge_id + 1, dtype=bool)
        is_train[valid_train] = True
        out_to_train_mask = mask_src_matched & is_train[tgt_matched]
        in_from_train_mask = mask_tgt_matched & is_train[src_matched]
        del is_train
    else:
        train_keys_sorted = np.sort(train_keys_arr)
        out_to_train_mask = mask_src_matched & np.isin(tgt_matched, train_keys_sorted)
        in_from_train_mask = mask_tgt_matched & np.isin(src_matched, train_keys_sorted)

    out_src = src_matched[out_to_train_mask]
    out_tgt = tgt_matched[out_to_train_mask]
    if len(out_tgt) > 0:
        out_df = pd.DataFrame({"src": out_src, "tgt": out_tgt}).drop_duplicates()
        out_src = out_df["src"].to_numpy()
        out_tgt = out_df["tgt"].to_numpy()
        del out_df

        tgt_unique, tgt_counts = np.unique(out_tgt, return_counts=True)
        v_weights_out = {}
        for v, deg in zip(tgt_unique, tgt_counts):
            tr = train_domain_tracker_dict[v]
            if len(tr) > 0:
                v_weights_out[v] = 1.0 / (np.log(2.0 + deg) * np.sqrt(len(tr)))

        u_indices_out = np.array([query_id_to_idx[u] for u in out_src], dtype=np.int32)
        for i in range(len(out_src)):
            v = out_tgt[i]
            w = v_weights_out.get(v, 0.0)
            if w > 0:
                u_idx = u_indices_out[i]
                graph_total_weights[u_idx] += w
                graph_scores[u_idx, train_domain_tracker_dict[v]] += w

    # 3. Inbound links: train domain -> query domain
    in_src = src_matched[in_from_train_mask]
    in_tgt = tgt_matched[in_from_train_mask]
    if len(in_src) > 0:
        in_df = pd.DataFrame({"src": in_src, "tgt": in_tgt}).drop_duplicates()
        in_src = in_df["src"].to_numpy()
        in_tgt = in_df["tgt"].to_numpy()
        del in_df

        src_unique, src_counts = np.unique(in_src, return_counts=True)
        v_weights_in = {}
        for v, deg in zip(src_unique, src_counts):
            tr = train_domain_tracker_dict[v]
            if len(tr) > 0:
                v_weights_in[v] = 0.4 / (np.log(2.0 + deg) * np.sqrt(len(tr)))

        u_indices_in = np.array([query_id_to_idx[u] for u in in_tgt], dtype=np.int32)
        for i in range(len(in_src)):
            v = in_src[i]
            w = v_weights_in.get(v, 0.0)
            if w > 0:
                u_idx = u_indices_in[i]
                graph_total_weights[u_idx] += w
                graph_scores[u_idx, train_domain_tracker_dict[v]] += w

    del (
        src_matched,
        tgt_matched,
        out_src,
        out_tgt,
        in_src,
        in_tgt,
        mask_src_matched,
        mask_tgt_matched,
    )
    gc.collect()

    # Degree normalization and co-occurrence diffusion
    norm_factor = np.sqrt(1.0 + graph_total_weights)[:, None]
    graph_scores_norm = graph_scores / norm_factor

    detected_signals = root_boost_scores + direct_scores + graph_scores_norm
    cooc_scores = detected_signals @ cooc_transition

    # Combine signals
    final_scores = (
        1.5 * prior_matrix
        + root_boost_scores
        + direct_scores
        + 1.2 * graph_scores_norm
        + 0.25 * cooc_scores
    )

    # Compute validation Recall@10
    print("Evaluating on validation set...")
    val_indices = np.array(
        [query_id_to_idx[vid] for vid in val_domain_ids], dtype=np.int64
    )
    val_scores = final_scores[val_indices]
    val_top10 = np.argpartition(val_scores, -10, axis=1)[:, -10:]

    val_recalls = []
    for i, val_id in enumerate(val_domain_ids):
        top10_set = set(val_top10[i])
        true_set = val_grouped.get(val_id, set())
        if len(true_set) > 0:
            recall = len(true_set.intersection(top10_set)) / len(true_set)
            val_recalls.append(recall)
        else:
            val_recalls.append(1.0)

    mean_val_recall = float(np.mean(val_recalls))
    print(f"Validation Recall@10: {mean_val_recall:.4f}")

    # Generate predictions for target domains
    print("Generating predictions for target domains...")
    target_indices = np.array(
        [query_id_to_idx[tid] for tid in target_domain_ids], dtype=np.int64
    )
    target_scores = final_scores[target_indices]

    # Efficient top-20 partition to select top 10 unique tracking domain IDs per domain
    top_candidates = np.argpartition(-target_scores, 20, axis=1)[:, :20]
    row_indices = np.arange(len(target_domain_ids))[:, None]
    candidate_scores = target_scores[row_indices, top_candidates]
    sort_order = np.argsort(-candidate_scores, axis=1)
    sorted_candidates = top_candidates[row_indices, sort_order]

    out_targets = []
    out_tracking_domains = []

    for i, tid in enumerate(target_domain_ids):
        candidates = sorted_candidates[i]
        chosen = []
        seen = set()
        for tr in candidates:
            td = tracker_id_to_domain_id[tr]
            if td not in seen:
                seen.add(td)
                chosen.append(td)
                if len(chosen) == 10:
                    break
        if len(chosen) < 10:
            all_sorted = np.argsort(-target_scores[i])
            for tr in all_sorted:
                td = tracker_id_to_domain_id[tr]
                if td not in seen:
                    seen.add(td)
                    chosen.append(td)
                    if len(chosen) == 10:
                        break
        for td in chosen:
            out_targets.append(tid)
            out_tracking_domains.append(td)

    sub_df = pd.DataFrame(
        {
            "domain_id": out_targets,
            "tracking_domain_id": out_tracking_domains,
        }
    )

    os.makedirs("working", exist_ok=True)
    sub_df.to_csv("working/submission.csv", sep="\t", index=False)
    print(
        f"Submission successfully saved to working/submission.csv with {len(sub_df)} rows."
    )


if __name__ == "__main__":
    main()

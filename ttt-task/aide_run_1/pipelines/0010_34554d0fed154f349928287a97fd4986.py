import gc
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

# ---------------------------------------------------------
# Configuration & Seed
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
NUM_TRACKERS = 355
VAL_SIZE = 20000

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

print("--- Step 1: Loading tracker metadata and targets ---")
trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
print(f"Loaded {len(trackers_df)} trackers.")

tracker_id_to_tdid = np.zeros(NUM_TRACKERS, dtype=np.int64)
tdid_to_tracker_id = {}
for _, row in trackers_df.iterrows():
    t_id = int(row["tracker_id"])
    td_id = int(row["tracking_domain_id"])
    tracker_id_to_tdid[t_id] = td_id
    tdid_to_tracker_id[td_id] = t_id

tracker_domain_ids = set(trackers_df["tracking_domain_id"].astype(int).tolist())

target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
target_domain_ids = target_df["domain_id"].to_numpy(dtype=np.int64)
print(f"Loaded {len(target_domain_ids)} target domains.")

print("--- Step 2: Loading training tracking graph and splitting validation ---")
track_table = pq.read_table(
    os.path.join(INPUT_DIR, "tracking_graph_train.parquet"),
    columns=["domain_id", "tracker_id"],
)
train_edges_domain_all = track_table["domain_id"].to_numpy().astype(np.int64)
train_edges_tracker_all = track_table["tracker_id"].to_numpy().astype(np.int32)
del track_table

# Deduplicate training pairs if any
pair_code = (train_edges_domain_all << 10) | train_edges_tracker_all
_, unique_indices = np.unique(pair_code, return_index=True)
train_edges_domain_all = train_edges_domain_all[unique_indices]
train_edges_tracker_all = train_edges_tracker_all[unique_indices]
del pair_code, unique_indices

all_train_domains = np.unique(train_edges_domain_all)
print(f"Total unique domains in training tracking graph: {len(all_train_domains):,}")

# Hold-out validation split
np.random.seed(SEED)
val_indices = np.random.choice(len(all_train_domains), size=VAL_SIZE, replace=False)
val_domain_ids = all_train_domains[val_indices]
val_domain_set = set(val_domain_ids)

train_domain_mask = ~np.isin(all_train_domains, val_domain_ids)
effective_train_domains = all_train_domains[train_domain_mask]
effective_train_set = set(effective_train_domains)
print(
    f"Train split: {len(effective_train_domains):,}, Validation split: {len(val_domain_ids):,}"
)

# Partition training edges into train and val
val_edge_mask = np.isin(train_edges_domain_all, val_domain_ids)
val_edges_domain = train_edges_domain_all[val_edge_mask]
val_edges_tracker = train_edges_tracker_all[val_edge_mask]

train_edge_mask = ~val_edge_mask
train_edges_domain = train_edges_domain_all[train_edge_mask]
train_edges_tracker = train_edges_tracker_all[train_edge_mask]
del train_edges_domain_all, train_edges_tracker_all, val_edge_mask, train_edge_mask

# Ground truth dictionary for validation Recall@10
val_truth_map = {d: set() for d in val_domain_ids}
for d, tid in zip(val_edges_domain, val_edges_tracker):
    val_truth_map[d].add(tid)

# Global tracker prior
global_counts = np.bincount(train_edges_tracker, minlength=NUM_TRACKERS)[
    :NUM_TRACKERS
].astype(np.float32)
global_prior = (global_counts + 1.0) / (len(effective_train_domains) + NUM_TRACKERS)

# Build CSR representation for effective training domains
N_eff = len(effective_train_domains)
eff_domain_to_idx = {d: i for i, d in enumerate(effective_train_domains)}
train_edge_eff_idx = np.array(
    [eff_domain_to_idx[d] for d in train_edges_domain], dtype=np.int32
)

Y_train = csr_matrix(
    (
        np.ones(len(train_edges_domain), dtype=np.float32),
        (train_edge_eff_idx, train_edges_tracker),
    ),
    shape=(N_eff, NUM_TRACKERS),
)

# Tracker co-occurrence matrix C_co (355 x 355)
C_co = (Y_train.T @ Y_train).toarray()
diag_counts = np.diag(C_co).copy()
ALPHA_CO = 50.0
P_co = (
    (C_co + ALPHA_CO * global_prior[None, :]) / (diag_counts[:, None] + ALPHA_CO)
).astype(np.float32)
print("Computed global prior and tracker co-occurrence matrix.")

print("--- Step 3: Processing domains metadata, TLDs, and categories ---")
dom_table = pq.read_table(
    os.path.join(INPUT_DIR, "domains.parquet"), columns=["domain_id", "domain"]
)
dom_ids_all = dom_table["domain_id"].to_numpy().astype(np.int64)
dom_names_all = dom_table["domain"].to_numpy()
del dom_table

# Filter to needed domains (train, val, target)
needed_domain_set = set(all_train_domains).union(set(target_domain_ids))
max_dom_id_val = dom_ids_all.max()

if max_dom_id_val <= 100_000_000:
    is_needed = np.zeros(max_dom_id_val + 1, dtype=bool)
    is_needed[list(needed_domain_set)] = True
    needed_mask = is_needed[dom_ids_all]
    del is_needed
else:
    needed_mask = np.isin(dom_ids_all, np.fromiter(needed_domain_set, dtype=np.int64))

needed_dom_ids = dom_ids_all[needed_mask]
needed_dom_names = dom_names_all[needed_mask]
del dom_ids_all, dom_names_all, needed_mask
gc.collect()


def extract_tld(name):
    if not isinstance(name, str) or not name:
        return "unknown"
    parts = name.lower().split(".")
    n = len(parts)
    if n < 2:
        return "unknown"
    last = parts[-1]
    if len(last) == 2 and n >= 3:
        second_last = parts[-2]
        if second_last in {
            "co",
            "com",
            "org",
            "edu",
            "gov",
            "net",
            "ac",
            "ne",
            "or",
            "go",
            "gen",
            "res",
        }:
            return f"{second_last}.{last}"
    return last


domain_to_tld = {
    did: extract_tld(name) for did, name in zip(needed_dom_ids, needed_dom_names)
}

# URL classification mapping
domain_to_cat = {}
url_class_path = os.path.join(INPUT_DIR, "url-classification.csv")
if os.path.exists(url_class_path):
    print("Loading URL classification...")
    url_df = pd.read_csv(url_class_path, usecols=["url", "category"], dtype=str)

    def clean_url_host(u):
        if not isinstance(u, str):
            return ""
        if "://" in u:
            u = u.split("://", 1)[1]
        u = u.split("/")[0].split(":")[0].split("?")[0].lower()
        if u.startswith("www."):
            u = u[4:]
        return u

    url_df["host"] = url_df["url"].apply(clean_url_host)
    url_df = url_df[url_df["host"] != ""].drop_duplicates(subset=["host"])
    host_to_cat = dict(zip(url_df["host"], url_df["category"]))
    del url_df

    for did, name in zip(needed_dom_ids, needed_dom_names):
        if not isinstance(name, str):
            continue
        clean_name = name.lower()
        if clean_name.startswith("www."):
            clean_name = clean_name[4:]
        cat = host_to_cat.get(clean_name, None)
        if cat is not None:
            domain_to_cat[did] = cat
    del host_to_cat
    print(f"Assigned categories to {len(domain_to_cat):,} domains.")

# Compute smoothed empirical TLD priors
tld_domain_counts = {}
for d in effective_train_domains:
    tld = domain_to_tld.get(d, "unknown")
    tld_domain_counts[tld] = tld_domain_counts.get(tld, 0) + 1

tld_counts = {}
for d, tid in zip(train_edges_domain, train_edges_tracker):
    tld = domain_to_tld.get(d, "unknown")
    if tld not in tld_counts:
        tld_counts[tld] = np.zeros(NUM_TRACKERS, dtype=np.float32)
    tld_counts[tld][tid] += 1.0

ALPHA_TLD = 15.0
tld_priors = {}
for tld, total_d in tld_domain_counts.items():
    cnts = tld_counts.get(tld, np.zeros(NUM_TRACKERS, dtype=np.float32))
    tld_priors[tld] = (
        (cnts + ALPHA_TLD * global_prior) / (total_d + ALPHA_TLD)
    ).astype(np.float32)

# Compute smoothed empirical Category priors
cat_domain_counts = {}
for d in effective_train_domains:
    cat = domain_to_cat.get(d, None)
    if cat is not None:
        cat_domain_counts[cat] = cat_domain_counts.get(cat, 0) + 1

cat_counts = {}
for d, tid in zip(train_edges_domain, train_edges_tracker):
    cat = domain_to_cat.get(d, None)
    if cat is not None:
        if cat not in cat_counts:
            cat_counts[cat] = np.zeros(NUM_TRACKERS, dtype=np.float32)
        cat_counts[cat][tid] += 1.0

ALPHA_CAT = 15.0
cat_priors = {}
for cat, total_d in cat_domain_counts.items():
    cnts = cat_counts.get(cat, np.zeros(NUM_TRACKERS, dtype=np.float32))
    cat_priors[cat] = (
        (cnts + ALPHA_CAT * global_prior) / (total_d + ALPHA_CAT)
    ).astype(np.float32)
print("Empirical TLD and Category priors computed.")

print("--- Step 4: Processing link graph for target and validation domains ---")
eval_domain_ids = np.concatenate([val_domain_ids, target_domain_ids])
N_query = len(eval_domain_ids)
query_id_to_idx = {did: i for i, did in enumerate(eval_domain_ids)}

link_table = pq.read_table(
    os.path.join(INPUT_DIR, "link-graph.parquet"),
    columns=["source_domain_id", "target_domain_id"],
)
src_all = link_table["source_domain_id"].to_numpy().astype(np.int64)
dst_all = link_table["target_domain_id"].to_numpy().astype(np.int64)
del link_table

max_id_graph = max(
    src_all.max(),
    dst_all.max(),
    eval_domain_ids.max(),
    max(tracker_domain_ids),
)

if max_id_graph <= 150_000_000:
    is_query = np.zeros(max_id_graph + 1, dtype=bool)
    is_query[eval_domain_ids] = True

    is_train = np.zeros(max_id_graph + 1, dtype=bool)
    is_train[effective_train_domains] = True

    is_tracker = np.zeros(max_id_graph + 1, dtype=bool)
    is_tracker[list(tracker_domain_ids)] = True

    mask_direct = is_query[src_all] & is_tracker[dst_all]
    direct_src = src_all[mask_direct]
    direct_dst = dst_all[mask_direct]

    mask_out = is_query[src_all] & is_train[dst_all]
    out_src = src_all[mask_out]
    out_dst = dst_all[mask_out]

    mask_in = is_query[dst_all] & is_train[src_all]
    in_src = src_all[mask_in]
    in_dst = dst_all[mask_in]

    del is_query, is_train, is_tracker, mask_direct, mask_out, mask_in
else:
    eval_set = set(eval_domain_ids)
    mask_direct = np.isin(src_all, eval_domain_ids) & np.isin(
        dst_all, list(tracker_domain_ids)
    )
    direct_src = src_all[mask_direct]
    direct_dst = dst_all[mask_direct]

    mask_out = np.isin(src_all, eval_domain_ids) & np.isin(
        dst_all, effective_train_domains
    )
    out_src = src_all[mask_out]
    out_dst = dst_all[mask_out]

    mask_in = np.isin(dst_all, eval_domain_ids) & np.isin(
        src_all, effective_train_domains
    )
    in_src = src_all[mask_in]
    in_dst = dst_all[mask_in]

del src_all, dst_all
gc.collect()

print("Constructing sparse graph features...")
# Direct tracker indicator matrix
V_direct = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
for s, d in zip(direct_src, direct_dst):
    q_idx = query_id_to_idx[s]
    t_id = tdid_to_tracker_id.get(d, None)
    if t_id is not None:
        V_direct[q_idx, t_id] = 1.0

# Out-neighbor aggregation
out_q_idx = np.array([query_id_to_idx[s] for s in out_src], dtype=np.int32)
out_train_idx = np.array([eff_domain_to_idx[d] for d in out_dst], dtype=np.int32)
A_out = csr_matrix(
    (np.ones(len(out_src), dtype=np.float32), (out_q_idx, out_train_idx)),
    shape=(N_query, N_eff),
)
A_out.data.fill(1.0)
V_out_counts = (A_out @ Y_train).toarray()
deg_out = np.array(A_out.sum(axis=1)).flatten()
BETA_OUT = 3.0
V_out = (V_out_counts / (deg_out[:, None] + BETA_OUT)).astype(np.float32)

# In-neighbor aggregation
in_q_idx = np.array([query_id_to_idx[d] for d in in_dst], dtype=np.int32)
in_train_idx = np.array([eff_domain_to_idx[s] for s in in_src], dtype=np.int32)
A_in = csr_matrix(
    (np.ones(len(in_src), dtype=np.float32), (in_q_idx, in_train_idx)),
    shape=(N_query, N_eff),
)
A_in.data.fill(1.0)
V_in_counts = (A_in @ Y_train).toarray()
deg_in = np.array(A_in.sum(axis=1)).flatten()
BETA_IN = 5.0
V_in = (V_in_counts / (deg_in[:, None] + BETA_IN)).astype(np.float32)

# Co-occurrence features
V_cooccur = (V_direct @ P_co).astype(np.float32)
V_neighbor_cooccur = ((V_out + 0.5 * V_in) @ P_co).astype(np.float32)

# TLD, Category, and Global prior features
V_tld = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
V_cat = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
V_glob = np.tile(global_prior, (N_query, 1)).astype(np.float32)

for i, did in enumerate(eval_domain_ids):
    tld = domain_to_tld.get(did, "unknown")
    if tld in tld_priors:
        V_tld[i] = tld_priors[tld]
    else:
        V_tld[i] = global_prior

    cat = domain_to_cat.get(did, None)
    if cat is not None and cat in cat_priors:
        V_cat[i] = cat_priors[cat]
    else:
        V_cat[i] = V_tld[i]

del A_out, A_in, V_out_counts, V_in_counts, Y_train
gc.collect()

print("--- Step 5: Feature splitting and validation optimization ---")
N_val = len(val_domain_ids)

F_val = [
    V_direct[:N_val],
    V_cooccur[:N_val],
    V_out[:N_val],
    V_in[:N_val],
    V_neighbor_cooccur[:N_val],
    V_cat[:N_val],
    V_tld[:N_val],
    V_glob[:N_val],
]

F_tgt = [
    V_direct[N_val:],
    V_cooccur[N_val:],
    V_out[N_val:],
    V_in[N_val:],
    V_neighbor_cooccur[N_val:],
    V_cat[N_val:],
    V_tld[N_val:],
    V_glob[N_val:],
]


def evaluate_weights(w):
    scores = np.zeros((N_val, NUM_TRACKERS), dtype=np.float32)
    for k in range(len(w)):
        scores += w[k] * F_val[k]

    top10 = np.argpartition(-scores, 10, axis=1)[:, :10]
    total_recall = 0.0
    for i in range(N_val):
        true_set = val_truth_map[val_domain_ids[i]]
        if not true_set:
            continue
        p_set = top10[i]
        hits = 0
        for tid in p_set:
            if tid in true_set:
                hits += 1
        total_recall += hits / len(true_set)
    return total_recall / N_val


weights = np.array([20.0, 3.0, 7.0, 3.5, 1.5, 1.0, 1.5, 0.5], dtype=np.float32)
current_recall = evaluate_weights(weights)
print(f"Initial Validation Recall@10: {current_recall:.5f}")

# Direct coordinate descent on Recall@10
for pass_num in range(2):
    for param_idx in range(len(weights)):
        best_val = weights[param_idx]
        best_score = current_recall
        step = 0.5 if param_idx >= 5 else 1.0
        candidates = [
            max(0.01, best_val + delta * step) for delta in [-3, -2, -1, 1, 2, 3]
        ]
        for cand in candidates:
            test_w = weights.copy()
            test_w[param_idx] = cand
            score = evaluate_weights(test_w)
            if score > best_score:
                best_score = score
                best_val = cand
        weights[param_idx] = best_val
        current_recall = best_score

print(f"Optimized Weights: {weights.round(3)}")
print(f"Validation Recall@10: {current_recall:.5f}")

print("--- Step 6: Generating predictions for target domains ---")
N_target = len(target_domain_ids)
target_scores = np.zeros((N_target, NUM_TRACKERS), dtype=np.float32)
for k in range(len(weights)):
    target_scores += weights[k] * F_tgt[k]

top10_part = np.argpartition(-target_scores, 10, axis=1)[:, :10]
row_idx = np.arange(N_target)[:, None]
top10_sorted_within = np.argsort(-target_scores[row_idx, top10_part], axis=1)
top10_tracker_ids = top10_part[row_idx, top10_sorted_within]

# Map tracker_id (0-354) to tracking_domain_id
predicted_tdids = tracker_id_to_tdid[top10_tracker_ids]

flat_domain_ids = np.repeat(target_domain_ids, 10)
flat_tracking_domain_ids = predicted_tdids.flatten()

sub_df = pd.DataFrame(
    {"domain_id": flat_domain_ids, "tracking_domain_id": flat_tracking_domain_ids}
)

print(f"Submission shape: {sub_df.shape}")
assert sub_df["domain_id"].nunique() == len(
    target_domain_ids
), "Mismatch in unique domains!"
assert len(sub_df) == len(target_domain_ids) * 10, "Mismatch in row count!"

sub_csv_path = os.path.join(WORKING_DIR, "submission.csv")
sub_tsv_path = os.path.join(WORKING_DIR, "submission.tsv")

sub_df.to_csv(sub_csv_path, sep="\t", index=False)
sub_df.to_csv(sub_tsv_path, sep="\t", index=False)

print(f"Successfully saved test predictions to {sub_csv_path}")
print(f"Final Validation Recall@10: {current_recall:.5f}")

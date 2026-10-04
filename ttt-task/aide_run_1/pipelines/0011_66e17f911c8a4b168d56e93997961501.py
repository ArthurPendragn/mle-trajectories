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

# Ground truth map for validation Recall@10
val_truth_map = {d: set() for d in val_domain_ids}
for d, tid in zip(val_edges_domain, val_edges_tracker):
    val_truth_map[d].add(tid)
val_truth_list = [val_truth_map[d] for d in val_domain_ids]

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

print("--- Step 3: Processing domains metadata, TLDs, SLDs, and categories ---")
dom_table = pq.read_table(
    os.path.join(INPUT_DIR, "domains.parquet"), columns=["domain_id", "domain"]
)
dom_ids_all = dom_table["domain_id"].to_numpy().astype(np.int64)
dom_names_all = dom_table["domain"].to_numpy()
del dom_table

# Multi-part TLD suffixes for accurate Registrable Domain (SLD) parsing
TWO_PART_TLDS = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "me.uk",
    "ltd.uk",
    "com.au",
    "net.au",
    "org.au",
    "edu.au",
    "gov.au",
    "co.jp",
    "ne.jp",
    "or.jp",
    "ac.jp",
    "go.jp",
    "co.nz",
    "net.nz",
    "org.nz",
    "govt.nz",
    "co.za",
    "org.za",
    "web.za",
    "com.br",
    "org.br",
    "net.br",
    "gov.br",
    "com.mx",
    "org.mx",
    "gob.mx",
    "edu.mx",
    "com.tr",
    "org.tr",
    "edu.tr",
    "gov.tr",
    "com.ar",
    "org.ar",
    "gob.ar",
    "co.kr",
    "or.kr",
    "pe.kr",
    "re.kr",
    "co.in",
    "net.in",
    "org.in",
    "gen.in",
    "firm.in",
    "com.sg",
    "net.sg",
    "org.sg",
    "gov.sg",
    "com.tw",
    "org.tw",
    "club.tw",
    "com.hk",
    "org.hk",
    "net.hk",
    "edu.hk",
    "com.cn",
    "edu.cn",
    "gov.cn",
    "net.cn",
    "org.cn",
    "com.ru",
    "net.ru",
    "org.ru",
    "pp.ru",
    "co.il",
    "org.il",
    "net.il",
    "ac.il",
    "com.pl",
    "org.pl",
    "net.pl",
    "com.es",
    "nom.es",
    "org.es",
    "com.pt",
    "org.pt",
    "co.th",
    "in.th",
    "ac.th",
    "com.ua",
    "net.ua",
    "org.ua",
    "co.id",
    "or.id",
    "net.id",
}


def parse_host(name):
    if not isinstance(name, str) or not name:
        return "unknown", "unknown"
    name = name.lower().strip()
    if name.startswith("www."):
        name = name[4:]
    parts = name.split(".")
    n = len(parts)
    if n < 2:
        return name, "unknown"
    if n >= 3:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in TWO_PART_TLDS:
            return f"{parts[-3]}.{two_part}", two_part
    return f"{parts[-2]}.{parts[-1]}", parts[-1]


eval_domain_ids = np.concatenate([val_domain_ids, target_domain_ids])
N_query = len(eval_domain_ids)

# Filter domains table to only needed domains (train, val, target)
needed_domain_set = set(all_train_domains).union(set(target_domain_ids))
max_dom_id_table = dom_ids_all.max()
needed_mask = np.isin(dom_ids_all, np.fromiter(needed_domain_set, dtype=np.int64))
needed_dom_ids = dom_ids_all[needed_mask]
needed_dom_names = dom_names_all[needed_mask]

# Build lookup for needed domains
domain_to_tld = {}
domain_to_sld = {}
for did, name in zip(needed_dom_ids, needed_dom_names):
    sld, tld = parse_host(name)
    domain_to_sld[did] = sld
    domain_to_tld[did] = tld

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
        u = u.split("/")[0].split(":")[0].split("?")[0].lower().strip()
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
        cname = name.lower().strip()
        if cname.startswith("www."):
            cname = cname[4:]
        cat = host_to_cat.get(cname, None)
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

ALPHA_TLD = 20.0
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

ALPHA_CAT = 20.0
cat_priors = {}
for cat, total_d in cat_domain_counts.items():
    cnts = cat_counts.get(cat, np.zeros(NUM_TRACKERS, dtype=np.float32))
    cat_priors[cat] = (
        (cnts + ALPHA_CAT * global_prior) / (total_d + ALPHA_CAT)
    ).astype(np.float32)

# Compute Registrable Domain (SLD) tracker profiles for query SLDs
query_slds = {domain_to_sld.get(did, "unknown") for did in eval_domain_ids}
sld_counts = {}
sld_dom_counts = {}

# Map domain to its trackers in training
for d, tid in zip(train_edges_domain, train_edges_tracker):
    sld = domain_to_sld.get(d, "unknown")
    if sld in query_slds and sld != "unknown":
        if sld not in sld_counts:
            sld_counts[sld] = np.zeros(NUM_TRACKERS, dtype=np.float32)
            sld_dom_counts[sld] = set()
        sld_counts[sld][tid] += 1.0
        sld_dom_counts[sld].add(d)

V_sld = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
for i, did in enumerate(eval_domain_ids):
    sld = domain_to_sld.get(did, "unknown")
    if sld in sld_counts:
        num_doms = len(sld_dom_counts[sld])
        V_sld[i] = (sld_counts[sld] / num_doms).astype(np.float32)

del sld_counts, sld_dom_counts
print("Empirical TLD, Category, and SLD priors computed.")

print("--- Step 4: Loading and processing link graph ---")
link_table = pq.read_table(
    os.path.join(INPUT_DIR, "link-graph.parquet"),
    columns=["source_domain_id", "target_domain_id"],
)
src_all = link_table["source_domain_id"].to_numpy().astype(np.int64)
dst_all = link_table["target_domain_id"].to_numpy().astype(np.int64)
del link_table

# SAFE GLOBAL MAX DOMAIN ID - avoids all index out of bounds
MAX_DOMAIN_ID = int(
    max(
        dom_ids_all.max(),
        src_all.max(),
        dst_all.max(),
        all_train_domains.max(),
        target_domain_ids.max(),
        max(tracker_domain_ids),
    )
    + 1000
)
del dom_ids_all, dom_names_all

# Direct fast array lookups
is_query = np.zeros(MAX_DOMAIN_ID, dtype=bool)
is_query[eval_domain_ids] = True

is_train = np.zeros(MAX_DOMAIN_ID, dtype=bool)
is_train[effective_train_domains] = True

is_tracker = np.zeros(MAX_DOMAIN_ID, dtype=bool)
is_tracker[list(tracker_domain_ids)] = True

query_id_to_idx_arr = np.full(MAX_DOMAIN_ID, -1, dtype=np.int32)
query_id_to_idx_arr[eval_domain_ids] = np.arange(N_query, dtype=np.int32)

eff_dom_to_idx_arr = np.full(MAX_DOMAIN_ID, -1, dtype=np.int32)
eff_dom_to_idx_arr[effective_train_domains] = np.arange(N_eff, dtype=np.int32)

tracker_dom_to_tid_arr = np.full(MAX_DOMAIN_ID, -1, dtype=np.int16)
for tid, tdid in enumerate(tracker_id_to_tdid):
    if tdid < MAX_DOMAIN_ID:
        tracker_dom_to_tid_arr[tdid] = tid

# Direct links: Query -> Tracker
mask_direct = is_query[src_all] & is_tracker[dst_all]
d_src = src_all[mask_direct]
d_dst = dst_all[mask_direct]
V_direct = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
q_idx_dir = query_id_to_idx_arr[d_src]
t_idx_dir = tracker_dom_to_tid_arr[d_dst]
valid_dir = (q_idx_dir >= 0) & (t_idx_dir >= 0)
V_direct[q_idx_dir[valid_dir], t_idx_dir[valid_dir]] = 1.0

# Direct links: Tracker -> Query
mask_rev = is_tracker[src_all] & is_query[dst_all]
r_src = src_all[mask_rev]
r_dst = dst_all[mask_rev]
V_direct_rev = np.zeros((N_query, NUM_TRACKERS), dtype=np.float32)
qr_idx = query_id_to_idx_arr[r_dst]
tr_idx = tracker_dom_to_tid_arr[r_src]
valid_rev = (qr_idx >= 0) & (tr_idx >= 0)
V_direct_rev[qr_idx[valid_rev], tr_idx[valid_rev]] = 1.0

# Out-neighbor aggregation (Query -> Train)
mask_out = is_query[src_all] & is_train[dst_all]
out_q = query_id_to_idx_arr[src_all[mask_out]]
out_train = eff_dom_to_idx_arr[dst_all[mask_out]]

A_out = csr_matrix(
    (np.ones(len(out_q), dtype=np.float32), (out_q, out_train)),
    shape=(N_query, N_eff),
)
A_out.sum_duplicates()
A_out.data.fill(1.0)
V_out_counts = (A_out @ Y_train).toarray()
deg_out = np.array(A_out.sum(axis=1)).flatten()
BETA_OUT = 3.0
V_out = (V_out_counts / (deg_out[:, None] + BETA_OUT)).astype(np.float32)

# In-neighbor aggregation (Train -> Query)
mask_in = is_query[dst_all] & is_train[src_all]
in_q = query_id_to_idx_arr[dst_all[mask_in]]
in_train = eff_dom_to_idx_arr[src_all[mask_in]]

A_in = csr_matrix(
    (np.ones(len(in_q), dtype=np.float32), (in_q, in_train)),
    shape=(N_query, N_eff),
)
A_in.sum_duplicates()
A_in.data.fill(1.0)
V_in_counts = (A_in @ Y_train).toarray()
deg_in = np.array(A_in.sum(axis=1)).flatten()
BETA_IN = 5.0
V_in = (V_in_counts / (deg_in[:, None] + BETA_IN)).astype(np.float32)

del (
    src_all,
    dst_all,
    mask_direct,
    mask_rev,
    mask_out,
    mask_in,
    A_out,
    A_in,
    V_out_counts,
    V_in_counts,
    Y_train,
)
gc.collect()

# Co-occurrence features
V_cooccur_direct = (V_direct @ P_co).astype(np.float32)
V_cooccur_neighbor = ((V_out + 0.5 * V_in) @ P_co).astype(np.float32)
V_cooccur_sld = (V_sld @ P_co).astype(np.float32)

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

print("--- Step 5: Feature grouping and validation optimization ---")
features_all = [
    V_direct,
    V_direct_rev,
    V_sld,
    V_sld_cooccur,
    V_out,
    V_in,
    V_cooccur_neighbor,
    V_cooccur_direct,
    V_cat,
    V_tld,
    V_glob,
]

N_val = len(val_domain_ids)
F_val = [f[:N_val] for f in features_all]
F_tgt = [f[N_val:] for f in features_all]


def evaluate_weights(w):
    scores = np.zeros((N_val, NUM_TRACKERS), dtype=np.float32)
    for k in range(len(w)):
        scores += w[k] * F_val[k]

    top10 = np.argpartition(-scores, 10, axis=1)[:, :10]
    total_recall = 0.0
    for i in range(N_val):
        t_set = val_truth_list[i]
        if not t_set:
            continue
        p = top10[i]
        hits = 0
        for tid in p:
            if tid in t_set:
                hits += 1
        total_recall += hits / len(t_set)
    return total_recall / N_val


weights = np.array(
    [
        35.0,  # V_direct
        8.0,  # V_direct_rev
        25.0,  # V_sld
        5.0,  # V_sld_cooccur
        10.0,  # V_out
        5.0,  # V_in
        3.0,  # V_cooccur_neighbor
        4.0,  # V_cooccur_direct
        2.0,  # V_cat
        1.5,  # V_tld
        0.8,  # V_glob
    ],
    dtype=np.float32,
)

initial_recall = evaluate_weights(weights)
print(f"Initial Validation Recall@10: {initial_recall:.5f}")

best_recall = initial_recall
for pass_num in range(2):
    for param_idx in range(len(weights)):
        best_val = weights[param_idx]
        current_val = weights[param_idx]
        for mult in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            cand = current_val * mult
            test_w = weights.copy()
            test_w[param_idx] = cand
            score = evaluate_weights(test_w)
            if score > best_recall:
                best_recall = score
                best_val = cand
        weights[param_idx] = best_val

print(f"Optimized Weights: {weights.round(3)}")
print(f"Final Validation Recall@10: {best_recall:.5f}")

print("--- Step 6: Generating predictions for target domains ---")
N_target = len(target_domain_ids)
target_scores = np.zeros((N_target, NUM_TRACKERS), dtype=np.float32)
for k in range(len(weights)):
    target_scores += weights[k] * F_tgt[k]

top10_part = np.argpartition(-target_scores, 10, axis=1)[:, :10]
row_idx = np.arange(N_target)[:, None]
top10_sorted_within = np.argsort(-target_scores[row_idx, top10_part], axis=1)
top10_tracker_ids = top10_part[row_idx, top10_sorted_within]

# Map compact tracker_id (0-354) to tracking_domain_id
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

print(f"Saved submission to {sub_csv_path} and {sub_tsv_path}")
print(f"Validation Recall@10: {best_recall:.5f}")

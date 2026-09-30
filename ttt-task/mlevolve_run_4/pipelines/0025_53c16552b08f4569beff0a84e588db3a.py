import collections
import gc
import json
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Filter PyTorch lr_scheduler deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# Ensure required output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. LOAD METADATA AND DATA SPLITS
# =============================================================================

# Load candidate trackers metadata
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
num_trackers = len(trackers_df)

tracker_id_to_tracking_domain = dict(
    zip(
        trackers_df["tracker_id"].astype(int),
        trackers_df["tracking_domain_id"].astype(int),
    )
)
tracking_domain_to_tracker_id = dict(
    zip(
        trackers_df["tracking_domain_id"].astype(int),
        trackers_df["tracker_id"].astype(int),
    )
)
tracker_domains_set = set(trackers_df["tracking_domain_id"].astype(int))

# Load target test domains
target_df = pd.read_csv("input/target.tsv", sep="\t")
test_domain_ids = target_df["domain_id"].astype(int).values
test_ids_set = set(test_domain_ids)

# Load known tracker presence on training domains
train_graph_df = pd.read_parquet("input/tracking_graph_train.parquet")
train_graph_df["domain_id"] = train_graph_df["domain_id"].astype(int)
train_graph_df["tracker_id"] = train_graph_df["tracker_id"].astype(int)

# Identify unique available training domains (strictly disjoint from test)
all_train_domains = np.setdiff1d(train_graph_df["domain_id"].unique(), test_domain_ids)
shuffled_train_domains = np.random.permutation(all_train_domains)

# Hold-out validation split: strictly 25,000 domains
N_VAL = 25000
val_domain_ids = shuffled_train_domains[:N_VAL]
all_remaining_train = shuffled_train_domains[N_VAL:]

# Scale training set up to 350,000 domains to avoid memory paging while ensuring full coverage
MAX_TRAIN = 350000
if len(all_remaining_train) > MAX_TRAIN:
    train_domain_ids = all_remaining_train[:MAX_TRAIN]
else:
    train_domain_ids = all_remaining_train

val_ids_set = set(val_domain_ids)
train_ids_set = set(train_domain_ids)
all_needed_ids = train_ids_set | val_ids_set | test_ids_set

# Construct sparse binary ground-truth target matrices
train_edges = train_graph_df[train_graph_df["domain_id"].isin(train_ids_set)]
val_edges = train_graph_df[train_graph_df["domain_id"].isin(val_ids_set)]

train_domain_to_idx = {did: idx for idx, did in enumerate(train_domain_ids)}
val_domain_to_idx = {did: idx for idx, did in enumerate(val_domain_ids)}

train_rows = train_edges["domain_id"].map(train_domain_to_idx).values
train_cols = train_edges["tracker_id"].values
train_data = np.ones(len(train_edges), dtype=np.uint8)
Y_train = sp.csr_matrix(
    (train_data, (train_rows, train_cols)),
    shape=(len(train_domain_ids), num_trackers),
    dtype=np.uint8,
)

val_rows = val_edges["domain_id"].map(val_domain_to_idx).values
val_cols = val_edges["tracker_id"].values
val_data = np.ones(len(val_edges), dtype=np.uint8)
Y_val = sp.csr_matrix(
    (val_data, (val_rows, val_cols)),
    shape=(len(val_domain_ids), num_trackers),
    dtype=np.uint8,
)

# Compute global tracker popularity from training set only (prevent data leakage)
train_tracker_counts = np.asarray(Y_train.sum(axis=0)).flatten()
popular_tracker_indices = np.argsort(-train_tracker_counts)
popular_tracking_domain_ids = [
    tracker_id_to_tracking_domain[idx] for idx in popular_tracker_indices
]

# Compute log-odds prior probabilities for head bias initialization
tracker_priors = np.clip(
    train_tracker_counts / max(len(train_domain_ids), 1), 1e-5, 1.0 - 1e-5
)
prior_logits = np.log(tracker_priors / (1.0 - tracker_priors))

# Compute empirical conditional tracker co-occurrence matrix C where C[j, k] = P(tracker_k | tracker_j)
cooccur_counts = np.asarray((Y_train.T @ Y_train).toarray(), dtype=np.float32)
diag_counts = np.diag(cooccur_counts).copy()
diag_counts[diag_counts == 0] = 1.0
tracker_cooccur_matrix = cooccur_counts / diag_counts[:, None]
np.fill_diagonal(tracker_cooccur_matrix, 0.0)

del train_edges, val_edges, train_graph_df
gc.collect()

# =============================================================================
# 2. FEATURE EXTRACTION & GRAPH STREAMING
# =============================================================================

# Fast streaming of link-graph.parquet: direct tracker links, active degrees, and bidirectional neighbor tracker aggregation
tracker_out_degree = collections.defaultdict(int)
domain_in_degree = collections.defaultdict(int)
domain_out_degree = collections.defaultdict(int)
all_link_src = []
all_link_trk = []

max_trk_domain_id = max(tracker_domains_set) + 1
trk_id_lookup_arr = np.full(max_trk_domain_id, -1, dtype=np.int16)
for tid, tdid in tracker_id_to_tracking_domain.items():
    if tdid < max_trk_domain_id:
        trk_id_lookup_arr[tdid] = int(tid)

max_domain_id = max(int(all_train_domains.max()), int(test_domain_ids.max())) + 1
is_active_domain = np.zeros(max_domain_id, dtype=bool)
is_active_domain[train_domain_ids] = True
is_active_domain[val_domain_ids] = True
is_active_domain[test_domain_ids] = True

# Index lookups for bidirectional neighbor tracker aggregation
all_active_domain_ids = np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
num_active_domains = len(all_active_domain_ids)
num_train_domains = len(train_domain_ids)

active_did_to_idx = np.full(max_domain_id, -1, dtype=np.int32)
active_did_to_idx[all_active_domain_ids] = np.arange(num_active_domains, dtype=np.int32)

train_did_to_idx = np.full(max_domain_id, -1, dtype=np.int32)
train_did_to_idx[train_domain_ids] = np.arange(num_train_domains, dtype=np.int32)

out_edge_act = []
out_edge_trn = []
in_edge_act = []
in_edge_trn = []

link_file = pq.ParquetFile("input/link-graph.parquet")
for batch in link_file.iter_batches(
    batch_size=8000000, columns=["source_domain_id", "target_domain_id"]
):
    src_arr = batch["source_domain_id"].to_numpy()
    tgt_arr = batch["target_domain_id"].to_numpy()

    # Track direct links to tracker domains
    valid_trk = (tgt_arr >= 0) & (tgt_arr < max_trk_domain_id)
    if valid_trk.any():
        t_safe = np.where(valid_trk, tgt_arr, 0)
        t_sub = trk_id_lookup_arr[t_safe]
        ok = valid_trk & (t_sub >= 0)
        if ok.any():
            all_link_src.append(src_arr[ok])
            all_link_trk.append(t_sub[ok])

    # Accumulate global link-graph in/out degrees for active domains
    valid_s = (src_arr >= 0) & (src_arr < max_domain_id)
    valid_t = (tgt_arr >= 0) & (tgt_arr < max_domain_id)
    s_safe = np.where(valid_s, src_arr, 0)
    t_safe = np.where(valid_t, tgt_arr, 0)

    src_eval = valid_s & is_active_domain[s_safe]
    if src_eval.any():
        u_s, c_s = np.unique(src_arr[src_eval], return_counts=True)
        for d, c in zip(u_s, c_s):
            domain_out_degree[int(d)] += int(c)

    tgt_eval = valid_t & is_active_domain[t_safe]
    if tgt_eval.any():
        u_t, c_t = np.unique(tgt_arr[tgt_eval], return_counts=True)
        for d, c in zip(u_t, c_t):
            domain_in_degree[int(d)] += int(c)

    # Accumulate bidirectional neighbor links between active domains and labeled training domains
    act_s = np.where(valid_s, active_did_to_idx[s_safe], -1)
    trn_t = np.where(valid_t, train_did_to_idx[t_safe], -1)
    mask_out = (act_s >= 0) & (trn_t >= 0)
    if mask_out.any():
        out_edge_act.append(act_s[mask_out])
        out_edge_trn.append(trn_t[mask_out])

    act_t = np.where(valid_t, active_did_to_idx[t_safe], -1)
    trn_s = np.where(valid_s, train_did_to_idx[s_safe], -1)
    mask_in = (act_t >= 0) & (trn_s >= 0)
    if mask_in.any():
        in_edge_act.append(act_t[mask_in])
        in_edge_trn.append(trn_s[mask_in])

del link_file, trk_id_lookup_arr, is_active_domain, active_did_to_idx, train_did_to_idx
gc.collect()

if len(all_link_src) > 0:
    all_link_src_arr = np.concatenate(all_link_src)
    all_link_trk_arr = np.concatenate(all_link_trk)
else:
    all_link_src_arr = np.array([], dtype=np.int64)
    all_link_trk_arr = np.array([], dtype=np.int16)

del all_link_src, all_link_trk
gc.collect()

unique_src, src_counts = np.unique(all_link_src_arr, return_counts=True)
tracker_out_degree = dict(zip(unique_src, src_counts))

# Build unique neighbor sparse adjacency matrices
if len(out_edge_act) > 0:
    out_act_cat = np.concatenate(out_edge_act)
    out_trn_cat = np.concatenate(out_edge_trn)
    A_out = sp.csr_matrix(
        (np.ones(len(out_act_cat), dtype=np.float32), (out_act_cat, out_trn_cat)),
        shape=(num_active_domains, num_train_domains),
    )
    A_out.data = np.ones_like(A_out.data)
else:
    A_out = sp.csr_matrix((num_active_domains, num_train_domains), dtype=np.float32)

del out_edge_act, out_edge_trn
gc.collect()

if len(in_edge_act) > 0:
    in_act_cat = np.concatenate(in_edge_act)
    in_trn_cat = np.concatenate(in_edge_trn)
    A_in = sp.csr_matrix(
        (np.ones(len(in_act_cat), dtype=np.float32), (in_act_cat, in_trn_cat)),
        shape=(num_active_domains, num_train_domains),
    )
    A_in.data = np.ones_like(A_in.data)
else:
    A_in = sp.csr_matrix((num_active_domains, num_train_domains), dtype=np.float32)

del in_edge_act, in_edge_trn
gc.collect()

# Multiply sparse adjacency with labeled tracker targets for fast neighbor aggregation
C_out = (A_out @ Y_train).toarray().astype(np.float32)
deg_out = np.asarray(A_out.sum(axis=1)).flatten().astype(np.float32)

C_in = (A_in @ Y_train).toarray().astype(np.float32)
deg_in = np.asarray(A_in.sum(axis=1)).flatten().astype(np.float32)

# Strict leave-one-out masking on training samples (subtract self-label if self-loops exist)
diag_out = np.asarray(A_out.diagonal()[:num_train_domains]).flatten()
has_self_out = diag_out > 0
if has_self_out.any():
    C_out[:num_train_domains][has_self_out] -= Y_train[has_self_out].toarray()
    deg_out[:num_train_domains][has_self_out] -= diag_out[has_self_out]

diag_in = np.asarray(A_in.diagonal()[:num_train_domains]).flatten()
has_self_in = diag_in > 0
if has_self_in.any():
    C_in[:num_train_domains][has_self_in] -= Y_train[has_self_in].toarray()
    deg_in[:num_train_domains][has_self_in] -= diag_in[has_self_in]

np.clip(C_out, 0.0, None, out=C_out)
np.clip(deg_out, 0.0, None, out=deg_out)
np.clip(C_in, 0.0, None, out=C_in)
np.clip(deg_in, 0.0, None, out=deg_in)

del A_out, A_in
gc.collect()

# Empirical Bayesian shrinkage toward global tracker priors (alpha=10.0)
alpha_diff = 10.0
global_trk_prior = tracker_priors[None, :].astype(np.float32)

out_priors_all = (C_out + alpha_diff * global_trk_prior) / (deg_out[:, None] + alpha_diff)
in_priors_all = (C_in + alpha_diff * global_trk_prior) / (deg_in[:, None] + alpha_diff)
np.clip(out_priors_all, 0.0, 1.0, out=out_priors_all)
np.clip(in_priors_all, 0.0, 1.0, out=in_priors_all)

del C_out, C_in
gc.collect()

# Partition priors into train, val, and test arrays of shape (N, 355)
n_trn = len(train_domain_ids)
n_v = len(val_domain_ids)

out_priors_train = out_priors_all[:n_trn]
out_priors_val = out_priors_all[n_trn : n_trn + n_v]
out_priors_test = out_priors_all[n_trn + n_v :]

in_priors_train = in_priors_all[:n_trn]
in_priors_val = in_priors_all[n_trn : n_trn + n_v]
in_priors_test = in_priors_all[n_trn + n_v :]

# Precompute link-graph diffusion stats lookup by domain_id
diff_stats_lookup = {}
in_p_sums = in_priors_all.sum(axis=1)
in_p_maxs = in_priors_all.max(axis=1)
out_p_sums = out_priors_all.sum(axis=1)
out_p_maxs = out_priors_all.max(axis=1)

for idx, did in enumerate(all_active_domain_ids):
    diff_stats_lookup[int(did)] = (
        float(deg_in[idx]),
        float(deg_out[idx]),
        1.0 if deg_in[idx] > 0 else 0.0,
        float(in_p_sums[idx]),
        float(in_p_maxs[idx]),
        1.0 if deg_out[idx] > 0 else 0.0,
        float(out_p_sums[idx]),
        float(out_p_maxs[idx]),
    )

del deg_out, deg_in, in_p_sums, in_p_maxs, out_p_sums, out_p_maxs, all_active_domain_ids
gc.collect()

# Build company and category aggregations from trackers.tsv metadata
tracker_companies = sorted(
    trackers_df["company"].fillna("Unknown").astype(str).unique()
)
tracker_categories = sorted(
    trackers_df["category"].fillna("Unknown").astype(str).unique()
)
company_to_idx = {c: i for i, c in enumerate(tracker_companies)}
tracker_cat_to_idx = {c: i for i, c in enumerate(tracker_categories)}

tracker_to_company_matrix = np.zeros(
    (num_trackers, len(tracker_companies)), dtype=np.float32
)
tracker_to_category_matrix = np.zeros(
    (num_trackers, len(tracker_categories)), dtype=np.float32
)

for _, row in trackers_df.iterrows():
    t_id = int(row["tracker_id"])
    comp = str(row["company"]) if pd.notna(row["company"]) else "Unknown"
    cat = str(row["category"]) if pd.notna(row["category"]) else "Unknown"
    if 0 <= t_id < num_trackers:
        tracker_to_company_matrix[t_id, company_to_idx[comp]] = 1.0
        tracker_to_category_matrix[t_id, tracker_cat_to_idx[cat]] = 1.0


def extract_direct_link_matrix(domain_ids):
    n_samples = len(domain_ids)
    direct_mat = np.zeros((n_samples, num_trackers), dtype=np.float32)
    did_to_row = {did: idx for idx, did in enumerate(domain_ids)}
    if len(all_link_src_arr) > 0:
        mask = np.fromiter(
            (s in did_to_row for s in all_link_src_arr), dtype=bool, count=len(all_link_src_arr)
        )
        if mask.any():
            matched_src = all_link_src_arr[mask]
            matched_trk = all_link_trk_arr[mask]
            rows = [did_to_row[s] for s in matched_src]
            direct_mat[rows, matched_trk] = 1.0
    return direct_mat

# Expanded regional second-level domain (SLD) patterns for high-precision apex parsing
TWO_PART_SECOND_LEVELS = {
    "co", "com", "org", "net", "edu", "gov", "gob", "ac",
    "ne", "or", "go", "gen", "nom", "mil", "asn", "biz", "info", "me",
    "ltd", "asso", "gv", "spb", "msk", "plc", "sch", "res", "soc",
    "tm", "bel", "pp", "in", "presse", "priv", "fed", "state", "city",
}


def get_tld(hostname):
    if not isinstance(hostname, str) or "." not in hostname:
        return "other"
    parts = hostname.lower().strip().split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in TWO_PART_SECOND_LEVELS:
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def get_apex_domain(hostname):
    if not isinstance(hostname, str) or "." not in hostname:
        return hostname.lower().strip() if isinstance(hostname, str) else ""
    parts = hostname.lower().strip().split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in TWO_PART_SECOND_LEVELS:
        return ".".join(parts[-3:])
    elif len(parts) >= 2:
        return ".".join(parts[-2:])
    return ".".join(parts)


def get_brand_root(hostname):
    """Extracts the core SLD brand root token across all generic and regional ccTLDs."""
    if not isinstance(hostname, str) or "." not in hostname:
        return hostname.lower().strip() if isinstance(hostname, str) else ""
    parts = hostname.lower().strip().split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in TWO_PART_SECOND_LEVELS:
        return parts[-3]
    elif len(parts) >= 2:
        return parts[-2]
    return parts[0]


# Stream domain hostnames and build comprehensive apex and brand root mappings across domains
max_domain_id = max(int(all_train_domains.max()), int(test_domain_ids.max())) + 1
domain_to_apex_id = np.full(max_domain_id, -1, dtype=np.int32)
domain_to_brand_id = np.full(max_domain_id, -1, dtype=np.int32)
is_needed_domain = np.zeros(max_domain_id, dtype=bool)
is_needed_domain[all_train_domains] = True
is_needed_domain[test_domain_ids] = True

is_model_needed = np.zeros(max_domain_id, dtype=bool)
is_model_needed[list(all_needed_ids)] = True

domain_lookup = {}
apex_to_id = {}
apex_list = []
brand_to_id = {}
brand_list = []

domains_file = pq.ParquetFile("input/domains.parquet")
for batch in domains_file.iter_batches(
    batch_size=2000000, columns=["domain_id", "domain"]
):
    d_ids = batch["domain_id"].to_numpy()
    valid = (d_ids >= 0) & (d_ids < max_domain_id)
    needed_mask = valid & is_needed_domain[np.where(valid, d_ids, 0)]
    if needed_mask.any():
        matched_ids = d_ids[needed_mask]
        matched_domains = batch["domain"].filter(pa.array(needed_mask)).to_pylist()
        for did, dom in zip(matched_ids, matched_domains):
            did_int = int(did)
            if is_model_needed[did_int]:
                domain_lookup[did_int] = str(dom)
            apex = get_apex_domain(dom)
            aid = apex_to_id.get(apex)
            if aid is None:
                aid = len(apex_list)
                apex_to_id[apex] = aid
                apex_list.append(apex)
            domain_to_apex_id[did_int] = aid

            brand = get_brand_root(dom)
            bid = brand_to_id.get(brand)
            if bid is None:
                bid = len(brand_list)
                brand_to_id[brand] = bid
                brand_list.append(brand)
            domain_to_brand_id[did_int] = bid

del domains_file, is_needed_domain, is_model_needed
gc.collect()

# Stream full tracking_graph_train.parquet to aggregate apex and brand tracker counts across non-val domains
num_apexes = len(apex_list)
num_brands = len(brand_list)
apex_tracker_sums = np.zeros((num_apexes, num_trackers), dtype=np.float32)
brand_tracker_sums = np.zeros((num_brands, num_trackers), dtype=np.float32)

# Count unique non-validation domains per apex and brand root from all_remaining_train
non_val_aids = domain_to_apex_id[all_remaining_train]
valid_aids = non_val_aids[non_val_aids >= 0]
apex_domain_counts = np.bincount(valid_aids, minlength=num_apexes).astype(np.int32)

non_val_bids = domain_to_brand_id[all_remaining_train]
valid_bids = non_val_bids[non_val_bids >= 0]
brand_domain_counts = np.bincount(valid_bids, minlength=num_brands).astype(np.int32)

is_val_arr = np.zeros(max_domain_id, dtype=bool)
is_val_arr[val_domain_ids] = True

track_graph_file = pq.ParquetFile("input/tracking_graph_train.parquet")
for batch in track_graph_file.iter_batches(
    batch_size=4000000, columns=["domain_id", "tracker_id"]
):
    b_dids = batch["domain_id"].to_numpy()
    b_trks = batch["tracker_id"].to_numpy()
    valid = (b_dids >= 0) & (b_dids < max_domain_id)
    valid_non_val = valid & (~is_val_arr[np.where(valid, b_dids, 0)])
    if valid_non_val.any():
        f_dids = b_dids[valid_non_val]
        f_trks = b_trks[valid_non_val]

        f_aids = domain_to_apex_id[f_dids]
        ok_a = (f_aids >= 0) & (f_trks >= 0) & (f_trks < num_trackers)
        if ok_a.any():
            flat_indices_a = f_aids[ok_a].astype(np.int64) * num_trackers + f_trks[ok_a].astype(np.int64)
            np.add.at(apex_tracker_sums.ravel(), flat_indices_a, 1.0)

        f_bids = domain_to_brand_id[f_dids]
        ok_b = (f_bids >= 0) & (f_trks >= 0) & (f_trks < num_trackers)
        if ok_b.any():
            flat_indices_b = f_bids[ok_b].astype(np.int64) * num_trackers + f_trks[ok_b].astype(np.int64)
            np.add.at(brand_tracker_sums.ravel(), flat_indices_b, 1.0)

del track_graph_file, is_val_arr
gc.collect()

# Construct leave-one-out empirical apex tracker presence profiles strictly on training domains
train_aids = domain_to_apex_id[train_domain_ids]
train_k = np.where(train_aids >= 0, apex_domain_counts[np.maximum(train_aids, 0)], 0)
multi_mask = train_k > 1

apex_priors_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.float32)
if multi_mask.any():
    multi_aids = train_aids[multi_mask]
    multi_denom = (train_k[multi_mask] - 1.0)[:, None].astype(np.float32)
    apex_priors_train[multi_mask] = apex_tracker_sums[multi_aids] / multi_denom
    sub_edge_mask = multi_mask[train_rows]
    if sub_edge_mask.any():
        sub_rows = train_rows[sub_edge_mask]
        sub_cols = train_cols[sub_edge_mask]
        sub_cnts = (train_k[sub_rows] - 1.0).astype(np.float32)
        np.subtract.at(apex_priors_train, (sub_rows, sub_cols), 1.0 / sub_cnts)
    np.clip(apex_priors_train, 0.0, 1.0, out=apex_priors_train)

# Construct leave-one-out brand root priors with Empirical Bayesian shrinkage
train_bids = domain_to_brand_id[train_domain_ids]
train_kb = np.where(train_bids >= 0, brand_domain_counts[np.maximum(train_bids, 0)], 0)
multi_mask_b = train_kb > 1
alpha_brand = 1.0

brand_priors_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.float32)
if multi_mask_b.any():
    multi_bids = train_bids[multi_mask_b]
    multi_denom_b = (train_kb[multi_mask_b] - 1.0 + alpha_brand)[:, None].astype(np.float32)
    brand_priors_train[multi_mask_b] = (
        brand_tracker_sums[multi_bids] + alpha_brand * tracker_priors[None, :]
    ) / multi_denom_b
    sub_edge_mask_b = multi_mask_b[train_rows]
    if sub_edge_mask_b.any():
        sub_rows_b = train_rows[sub_edge_mask_b]
        sub_cols_b = train_cols[sub_edge_mask_b]
        sub_cnts_b = (train_kb[sub_rows_b] - 1.0 + alpha_brand).astype(np.float32)
        np.subtract.at(brand_priors_train, (sub_rows_b, sub_cols_b), 1.0 / sub_cnts_b)
    np.clip(brand_priors_train, 0.0, 1.0, out=brand_priors_train)

# Smooth single-count domains directly to global empirical Bayesian priors to eliminate train-eval shift
single_mask_b = (train_kb == 1)
if single_mask_b.any():
    brand_priors_train[single_mask_b] = tracker_priors[None, :]

del train_rows, train_cols, val_rows, val_cols

# Direct empirical apex means for evaluation (validation and test)
nonzero_counts = np.maximum(apex_domain_counts, 1).astype(np.float32)[:, None]
apex_means = (apex_tracker_sums / nonzero_counts).astype(np.float32)


def compute_eval_apex_priors(domain_ids):
    priors = np.zeros((len(domain_ids), num_trackers), dtype=np.float32)
    aids = domain_to_apex_id[domain_ids]
    has_prior = (aids >= 0) & (apex_domain_counts[np.maximum(aids, 0)] > 0)
    if has_prior.any():
        matched_aids = aids[has_prior]
        priors[has_prior] = apex_means[matched_aids]
    return priors


apex_priors_val = compute_eval_apex_priors(val_domain_ids)
apex_priors_test = compute_eval_apex_priors(test_domain_ids)

# Direct smoothed empirical brand root means for evaluation (validation and test)
nonzero_brand_counts = (brand_domain_counts + alpha_brand).astype(np.float32)[:, None]
brand_means = (
    (brand_tracker_sums + alpha_brand * tracker_priors[None, :]) / nonzero_brand_counts
).astype(np.float32)


def compute_eval_brand_priors(domain_ids):
    priors = np.zeros((len(domain_ids), num_trackers), dtype=np.float32)
    bids = domain_to_brand_id[domain_ids]
    has_prior = (bids >= 0) & (brand_domain_counts[np.maximum(bids, 0)] > 0)
    if has_prior.any():
        matched_bids = bids[has_prior]
        priors[has_prior] = brand_means[matched_bids]
    return priors


brand_priors_val = compute_eval_brand_priors(val_domain_ids)
brand_priors_test = compute_eval_brand_priors(test_domain_ids)

# Precompute prior intensity summaries (presence, sum, max) across train, val, test
prior_stats_lookup = {}
for dids, ap_arr, bp_arr in [
    (train_domain_ids, apex_priors_train, brand_priors_train),
    (val_domain_ids, apex_priors_val, brand_priors_val),
    (test_domain_ids, apex_priors_test, brand_priors_test),
]:
    ap_sums = ap_arr.sum(axis=1)
    ap_maxs = ap_arr.max(axis=1)
    bp_sums = bp_arr.sum(axis=1)
    bp_maxs = bp_arr.max(axis=1)
    for i, did in enumerate(dids):
        prior_stats_lookup[did] = (
            1.0 if ap_sums[i] > 0 else 0.0,
            float(ap_sums[i]),
            float(ap_maxs[i]),
            1.0 if bp_sums[i] > 0 else 0.0,
            float(bp_sums[i]),
            float(bp_maxs[i]),
        )

del apex_tracker_sums, brand_tracker_sums, apex_means, brand_means
del non_val_aids, valid_aids, non_val_bids, valid_bids
gc.collect()

# Compute smoothed empirical TLD-tracker prior distributions strictly from train
train_tlds = [get_tld(domain_lookup.get(did, "")) for did in train_domain_ids]
unique_tlds, tld_inverse = np.unique(train_tlds, return_inverse=True)
num_unique_tlds = len(unique_tlds)

T_train = sp.csr_matrix(
    (
        np.ones(len(train_domain_ids), dtype=np.float32),
        (tld_inverse, np.arange(len(train_domain_ids))),
    ),
    shape=(num_unique_tlds, len(train_domain_ids)),
)
tld_sums = (T_train @ Y_train.astype(np.float32)).toarray()
tld_counts_arr = np.bincount(tld_inverse, minlength=num_unique_tlds).astype(np.float32)[:, None]

m_smooth = 10.0
tld_priors_mat = (tld_sums + m_smooth * tracker_priors[None, :]) / (tld_counts_arr + m_smooth)
tld_prior_lookup = {
    tld: tld_priors_mat[i].astype(np.float32)
    for i, tld in enumerate(unique_tlds)
}
default_tld_prior = tracker_priors.astype(np.float32)


def compute_tld_priors(domain_ids):
    n_samples = len(domain_ids)
    priors = np.empty((n_samples, num_trackers), dtype=np.float32)
    for i, did in enumerate(domain_ids):
        tld = get_tld(domain_lookup.get(did, ""))
        priors[i] = tld_prior_lookup.get(tld, default_tld_prior)
    return priors


del T_train, tld_sums, tld_priors_mat
gc.collect()

# Process URL classification categories
url_df = pd.read_csv("input/url-classification.csv", usecols=["url", "category"])


def extract_netloc(url_str):
    if not isinstance(url_str, str):
        return ""
    pos = url_str.find("://")
    s = url_str[pos + 3 :] if pos != -1 else url_str
    slash = s.find("/")
    if slash != -1:
        s = s[:slash]
    colon = s.find(":")
    if colon != -1:
        s = s[:colon]
    return s.lower().strip()


url_df["netloc"] = [extract_netloc(u) for u in url_df["url"]]
url_df = url_df[url_df["netloc"] != ""]

# Filter URL classification to active hostnames and apex domains
active_hostnames = {h.lower().strip() for h in domain_lookup.values()}
active_apexes = {get_apex_domain(h) for h in domain_lookup.values()}
url_df["apex"] = [get_apex_domain(n) for n in url_df["netloc"]]
url_df = url_df[url_df["netloc"].isin(active_hostnames) | url_df["apex"].isin(active_apexes)]

categories = sorted(url_df["category"].dropna().unique())
cat_counts_df = url_df[url_df["netloc"].isin(active_hostnames)].groupby(["netloc", "category"]).size().unstack(fill_value=0)
cat_counts_df = cat_counts_df.reindex(columns=categories, fill_value=0)
cat_totals = cat_counts_df.sum(axis=1)
cat_probs_df = cat_counts_df.div(cat_totals.replace(0, 1), axis=0)

cat_lookup = {}
for netloc, row in cat_probs_df.iterrows():
    cat_lookup[netloc] = row.values.astype(np.float32)

apex_cat_counts_df = url_df.groupby(["apex", "category"]).size().unstack(fill_value=0)
apex_cat_counts_df = apex_cat_counts_df.reindex(columns=categories, fill_value=0)
apex_cat_totals = apex_cat_counts_df.sum(axis=1)
apex_cat_probs_df = apex_cat_counts_df.div(apex_cat_totals.replace(0, 1), axis=0)

apex_cat_lookup = {}
for apex_dom, row in apex_cat_probs_df.iterrows():
    apex_cat_lookup[apex_dom] = row.values.astype(np.float32)

del url_df, cat_counts_df, cat_probs_df, apex_cat_counts_df, apex_cat_probs_df
gc.collect()

# Process Freedom of the Press table
try:
    fotp_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
except Exception:
    fotp_df = pd.read_csv("input/freedom-of-the-press.csv")
if (
    "freedom_of_the_press" not in fotp_df.columns
    and len(fotp_df.columns) == 1
    and "\t" in fotp_df.columns[0]
):
    fotp_df = fotp_df[fotp_df.columns[0]].str.split("\t", expand=True)
    fotp_df.columns = ["tld", "country", "freedom_of_the_press"]

fotp_df["tld"] = fotp_df["tld"].astype(str).str.strip().str.lower()
fotp_df["freedom_of_the_press"] = pd.to_numeric(
    fotp_df["freedom_of_the_press"], errors="coerce"
)
tld_to_press_freedom = dict(zip(fotp_df["tld"], fotp_df["freedom_of_the_press"]))

# Fit TLD frequency encoding strictly on training domains (expanded to top 150 TLDs)
train_tlds = [get_tld(domain_lookup.get(did, "")) for did in train_domain_ids]
tld_counts = pd.Series(train_tlds).value_counts()
top_tlds = list(tld_counts.head(150).index)
tld_to_code = {tld: idx for idx, tld in enumerate(top_tlds)}

# Expand lexical representations to 1,024 TF-IDF features fitted across all training hostnames
train_hostnames = [domain_lookup.get(did, "") for did in train_domain_ids]
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), max_features=768, min_df=5
)
word_vec = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)[a-zA-Z0-9]+",
    ngram_range=(1, 2),
    max_features=256,
    min_df=5,
)
tfidf_vectorizer = FeatureUnion([("char", char_vec), ("word", word_vec)])
tfidf_vectorizer.fit(train_hostnames)

vowel_set = set("aeiou")
keywords = [
    "shop", "blog", "news", "store", "media", "tv", "game", "play", "tech",
    "app", "video", "forum", "porn", "sex", "adult", "finance", "money",
    "bank", "crypto", "coin", "travel", "hotel", "flight", "bet", "casino",
    "poker", "sport", "live", "music", "radio", "health", "med", "edu",
    "book", "job", "auto"
]


def build_feature_matrix(domain_ids, direct_mat):
    hostnames = [domain_lookup.get(did, "") for did in domain_ids]
    n_samples = len(domain_ids)

    lengths = np.empty(n_samples, dtype=np.float32)
    dots = np.empty(n_samples, dtype=np.float32)
    hyphens = np.empty(n_samples, dtype=np.float32)
    digits = np.empty(n_samples, dtype=np.float32)
    vowels = np.empty(n_samples, dtype=np.float32)
    starts_www = np.empty(n_samples, dtype=np.float32)
    sld_lengths = np.empty(n_samples, dtype=np.float32)
    subdomain_depths = np.empty(n_samples, dtype=np.float32)
    entropies = np.empty(n_samples, dtype=np.float32)
    tlds = []

    for i, h in enumerate(hostnames):
        hl = h.lower().strip()
        lh = len(hl)
        lengths[i] = lh
        dots[i] = hl.count(".")
        hyphens[i] = hl.count("-")
        d_cnt = sum(c.isdigit() for c in hl)
        v_cnt = sum(c in vowel_set for c in hl)
        digits[i] = d_cnt
        vowels[i] = v_cnt
        starts_www[i] = 1.0 if hl.startswith("www.") else 0.0

        parts = hl.split(".")
        sld_lengths[i] = len(parts[-2]) if len(parts) >= 2 else lh
        subdomain_depths[i] = max(0, len(parts) - 2)
        tlds.append(get_tld(hl))

        # Shannon character entropy metric
        if lh <= 1:
            entropies[i] = 0.0
        else:
            counts = collections.Counter(hl)
            ent = 0.0
            for cnt in counts.values():
                p = cnt / lh
                ent -= p * np.log2(p)
            entropies[i] = ent

    digit_ratios = digits / np.maximum(lengths, 1.0)
    vowel_ratios = vowels / np.maximum(lengths, 1.0)

    keyword_flags = np.zeros((n_samples, len(keywords)), dtype=np.float32)
    for k_idx, kw in enumerate(keywords):
        keyword_flags[:, k_idx] = [1.0 if kw in h.lower() else 0.0 for h in hostnames]

    tld_onehot = np.zeros((n_samples, len(top_tlds)), dtype=np.float32)
    for i, t in enumerate(tlds):
        if t in tld_to_code:
            tld_onehot[i, tld_to_code[t]] = 1.0

    is_cc_tld = np.array(
        [1.0 if (len(t) == 2 or ("." in t and len(t.split(".")[-1]) == 2)) else 0.0 for t in tlds],
        dtype=np.float32,
    )
    press_freedoms = np.array(
        [tld_to_press_freedom.get(t, tld_to_press_freedom.get(t.split(".")[-1], 40.0)) for t in tlds],
        dtype=np.float32,
    )
    has_press_freedom = np.array(
        [1.0 if (t in tld_to_press_freedom or t.split(".")[-1] in tld_to_press_freedom) else 0.0 for t in tlds],
        dtype=np.float32,
    )

    num_cats = len(categories)
    cat_features = np.zeros((n_samples, num_cats), dtype=np.float32)
    cat_known = np.zeros(n_samples, dtype=np.float32)
    for i, h in enumerate(hostnames):
        norm_h = h.lower().strip()
        h_no_www = norm_h[4:] if norm_h.startswith("www.") else norm_h
        apex_h = get_apex_domain(norm_h)
        if norm_h in cat_lookup:
            cat_features[i] = cat_lookup[norm_h]
            cat_known[i] = 1.0
        elif h_no_www in cat_lookup:
            cat_features[i] = cat_lookup[h_no_www]
            cat_known[i] = 1.0
        elif apex_h in cat_lookup:
            cat_features[i] = cat_lookup[apex_h]
            cat_known[i] = 1.0
        elif apex_h in apex_cat_lookup:
            cat_features[i] = apex_cat_lookup[apex_h]
            cat_known[i] = 1.0

    log_in_degree = np.array(
        [np.log1p(domain_in_degree.get(did, 0)) for did in domain_ids], dtype=np.float32
    )
    log_out_degree = np.array(
        [np.log1p(domain_out_degree.get(did, 0)) for did in domain_ids], dtype=np.float32
    )
    log_tracker_out = np.array(
        [np.log1p(tracker_out_degree.get(did, 0)) for did in domain_ids], dtype=np.float32
    )
    num_direct_links = direct_mat.sum(axis=1)[:, None]

    # Link-graph direct tracker link ratio relative to total domain degree
    total_degree = np.array(
        [domain_in_degree.get(did, 0) + domain_out_degree.get(did, 0) for did in domain_ids],
        dtype=np.float32,
    )
    direct_link_ratio = (direct_mat.sum(axis=1) / np.maximum(total_degree, 1.0)).astype(np.float32)[:, None]

    # Apex and brand domain support statistics and prior intensity summaries
    log_apex_counts = np.empty((n_samples, 1), dtype=np.float32)
    log_brand_counts = np.empty((n_samples, 1), dtype=np.float32)
    has_apex_prior = np.empty((n_samples, 1), dtype=np.float32)
    apex_prior_sum = np.empty((n_samples, 1), dtype=np.float32)
    apex_prior_max = np.empty((n_samples, 1), dtype=np.float32)
    has_brand_prior = np.empty((n_samples, 1), dtype=np.float32)
    brand_prior_sum = np.empty((n_samples, 1), dtype=np.float32)
    brand_prior_max = np.empty((n_samples, 1), dtype=np.float32)

    for i, did in enumerate(domain_ids):
        aid = domain_to_apex_id[did] if did < len(domain_to_apex_id) else -1
        log_apex_counts[i, 0] = np.log1p(apex_domain_counts[aid]) if aid >= 0 else 0.0

        bid = domain_to_brand_id[did] if did < len(domain_to_brand_id) else -1
        log_brand_counts[i, 0] = np.log1p(brand_domain_counts[bid]) if bid >= 0 else 0.0

        stats = prior_stats_lookup.get(did)
        if stats is not None:
            has_apex_prior[i, 0] = stats[0]
            apex_prior_sum[i, 0] = stats[1]
            apex_prior_max[i, 0] = stats[2]
            has_brand_prior[i, 0] = stats[3]
            brand_prior_sum[i, 0] = stats[4]
            brand_prior_max[i, 0] = stats[5]
        else:
            has_apex_prior[i, 0] = 0.0
            apex_prior_sum[i, 0] = 0.0
            apex_prior_max[i, 0] = 0.0
            has_brand_prior[i, 0] = 0.0
            brand_prior_sum[i, 0] = 0.0
            brand_prior_max[i, 0] = 0.0

    company_link_counts = direct_mat @ tracker_to_company_matrix
    category_link_counts = direct_mat @ tracker_to_category_matrix

    tfidf_mat = tfidf_vectorizer.transform(hostnames).toarray().astype(np.float32)

    # Link graph diffusion statistics
    log_in_neighbor_deg = np.empty(n_samples, dtype=np.float32)
    log_out_neighbor_deg = np.empty(n_samples, dtype=np.float32)
    has_in_diff = np.empty(n_samples, dtype=np.float32)
    in_diff_sum = np.empty(n_samples, dtype=np.float32)
    in_diff_max = np.empty(n_samples, dtype=np.float32)
    has_out_diff = np.empty(n_samples, dtype=np.float32)
    out_diff_sum = np.empty(n_samples, dtype=np.float32)
    out_diff_max = np.empty(n_samples, dtype=np.float32)

    for i, did in enumerate(domain_ids):
        d_stats = diff_stats_lookup.get(did)
        if d_stats is not None:
            log_in_neighbor_deg[i] = np.log1p(d_stats[0])
            log_out_neighbor_deg[i] = np.log1p(d_stats[1])
            has_in_diff[i] = d_stats[2]
            in_diff_sum[i] = d_stats[3]
            in_diff_max[i] = d_stats[4]
            has_out_diff[i] = d_stats[5]
            out_diff_sum[i] = d_stats[6]
            out_diff_max[i] = d_stats[7]
        else:
            log_in_neighbor_deg[i] = 0.0
            log_out_neighbor_deg[i] = 0.0
            has_in_diff[i] = 0.0
            in_diff_sum[i] = 0.0
            in_diff_max[i] = 0.0
            has_out_diff[i] = 0.0
            out_diff_sum[i] = 0.0
            out_diff_max[i] = 0.0

    # Domain support features positioned at the beginning for PriorGatingNet conditioning
    support_blocks = [
        log_in_degree[:, None],
        log_out_degree[:, None],
        log_tracker_out[:, None],
        num_direct_links,
        direct_link_ratio,
        log_in_neighbor_deg[:, None],
        log_out_neighbor_deg[:, None],
        has_in_diff[:, None],
        in_diff_sum[:, None],
        in_diff_max[:, None],
        has_out_diff[:, None],
        out_diff_sum[:, None],
        out_diff_max[:, None],
        log_apex_counts,
        log_brand_counts,
        has_apex_prior,
        apex_prior_sum,
        apex_prior_max,
        has_brand_prior,
        brand_prior_sum,
        brand_prior_max,
    ]
    other_cont_blocks = [
        lengths[:, None],
        dots[:, None],
        hyphens[:, None],
        digits[:, None],
        digit_ratios[:, None],
        vowels[:, None],
        vowel_ratios[:, None],
        sld_lengths[:, None],
        subdomain_depths[:, None],
        entropies[:, None],
        press_freedoms[:, None],
        company_link_counts,
        category_link_counts,
    ]
    continuous_blocks = support_blocks + other_cont_blocks
    uncentered_blocks = [
        starts_www[:, None],
        keyword_flags,
        tld_onehot,
        is_cc_tld[:, None],
        has_press_freedom[:, None],
        cat_features,
        cat_known[:, None],
        tfidf_mat,
    ]
    X_cont = np.hstack(continuous_blocks).astype(np.float32)
    X_uncentered = np.hstack(uncentered_blocks).astype(np.float32)
    return np.hstack([X_cont, X_uncentered]).astype(np.float32)


# Extract direct link matrices for train, val, test
direct_links_train = extract_direct_link_matrix(train_domain_ids)
direct_links_val = extract_direct_link_matrix(val_domain_ids)
direct_links_test = extract_direct_link_matrix(test_domain_ids)

# Extract full 355-dimensional empirical Bayesian-smoothed TLD prior vectors
tld_priors_train = compute_tld_priors(train_domain_ids)
tld_priors_val = compute_tld_priors(val_domain_ids)
tld_priors_test = compute_tld_priors(test_domain_ids)

# Number of continuous numerical features to selectively standardize (21 support + 11 lexical/geo + companies + categories)
num_cont_cols = 32 + len(tracker_companies) + len(tracker_categories)

# Build feature matrices
X_train_raw = build_feature_matrix(train_domain_ids, direct_links_train)
X_val_raw = build_feature_matrix(val_domain_ids, direct_links_val)
X_test_raw = build_feature_matrix(test_domain_ids, direct_links_test)

# Fit StandardScaler strictly on continuous numerical features of X_train (Zero Data Leakage)
scaler = StandardScaler()
X_train_cont = scaler.fit_transform(X_train_raw[:, :num_cont_cols]).astype(np.float32)
X_train = np.hstack([X_train_cont, X_train_raw[:, num_cont_cols:]]).astype(np.float32)

X_val_cont = scaler.transform(X_val_raw[:, :num_cont_cols]).astype(np.float32)
X_val = np.hstack([X_val_cont, X_val_raw[:, num_cont_cols:]]).astype(np.float32)

X_test_cont = scaler.transform(X_test_raw[:, :num_cont_cols]).astype(np.float32)
X_test = np.hstack([X_test_cont, X_test_raw[:, num_cont_cols:]]).astype(np.float32)

feature_dim = X_train.shape[1]

del X_train_raw, X_val_raw, X_test_raw, X_train_cont, X_val_cont, X_test_cont
gc.collect()

# Persist processed target matrices
sp.save_npz("./working/train_targets.npz", Y_train)
sp.save_npz("./working/val_targets.npz", Y_val)

metadata = {
    "num_trackers": num_trackers,
    "feature_dim": feature_dim,
    "top_10_popular_tracking_domain_ids": popular_tracking_domain_ids[:10],
    "top_10_popular_tracker_indices": [int(x) for x in popular_tracker_indices[:10]],
    "tracker_id_to_tracking_domain": {
        int(k): int(v) for k, v in tracker_id_to_tracking_domain.items()
    },
    "tracking_domain_to_tracker_id": {
        int(k): int(v) for k, v in tracking_domain_to_tracker_id.items()
    },
}

with open("./working/tracker_metadata.json", "w") as f:
    json.dump(metadata, f)

# =============================================================================
# 3. MODEL ARCHITECTURE DESIGN: ASYMMETRIC FOCAL LOSS & DYNAMIC PRIOR GATING
# =============================================================================


class AsymmetricLoss(nn.Module):
    """Asymmetric Focal Multi-Label Loss (ASL) for extreme class imbalance without gradient saturation."""

    def __init__(self, gamma_neg: float = 2.0, pos_weight: float = 2.5, eps: float = 1e-7):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        log_pos = F.logsigmoid(logits)
        log_neg = F.logsigmoid(-logits)
        p = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        loss_pos = -self.pos_weight * targets * log_pos
        loss_neg = -(1.0 - targets) * (p ** self.gamma_neg) * log_neg

        return (loss_pos + loss_neg).mean()


class PriorGatingNet(nn.Module):
    """Sample-conditioned dynamic multi-prior gating network."""

    def __init__(self, in_features: int = 21, hidden_dim: int = 64, num_priors: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, num_priors),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, support_feats: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(support_feats), dim=-1)


class TabularSEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation feature gating mechanism."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class TabularResBlock(nn.Module):
    """Tabular residual MLP block with LayerNorm and SE gating."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.30):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.se = TabularSEBlock(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(self.linear1(out))
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        out = self.se(out)
        return residual + out


class TrackerResNet(nn.Module):
    """Model 1: Deep Tabular Residual Network with unclipped linear head and sample-conditioned dynamic prior gating."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 355,
        hidden_dim: int = 384,
        num_blocks: int = 3,
        dropout_rate: float = 0.30,
        init_bias: np.ndarray = None,
        cooccur_matrix: np.ndarray = None,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head_dim = hidden_dim * num_blocks
        self.final_norm = nn.LayerNorm(self.head_dim)
        self.head = nn.Linear(self.head_dim, num_classes)

        if init_bias is not None:
            self.head.bias.data.copy_(torch.tensor(init_bias, dtype=torch.float32))

        # Dynamic sample-conditioned prior gating net
        self.support_dim = 21
        self.gating_net = PriorGatingNet(in_features=self.support_dim, hidden_dim=64, num_priors=5)

        # Prior scale parameters
        self.direct_scale = nn.Parameter(torch.full((num_classes,), 5.0))
        self.apex_scale = nn.Parameter(torch.full((num_classes,), 4.0))
        self.brand_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.in_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.out_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.tld_scale = nn.Parameter(torch.full((num_classes,), 2.0))

    def forward(
        self,
        x: torch.Tensor,
        direct_links: torch.Tensor = None,
        apex_priors: torch.Tensor = None,
        brand_priors: torch.Tensor = None,
        tld_priors: torch.Tensor = None,
        in_priors: torch.Tensor = None,
        out_priors: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.stem(x)
        block_outs = []
        for block in self.res_blocks:
            h = block(h)
            block_outs.append(h)
        h_concat = torch.cat(block_outs, dim=-1)
        h_norm = self.final_norm(h_concat)
        logits = self.head(h_norm)

        # Dynamic prior gating
        support_feats = x[:, : self.support_dim]
        gates = self.gating_net(support_feats)

        priors_blend = torch.zeros_like(logits)
        if apex_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 0:1] * apex_priors * F.softplus(self.apex_scale)
        if brand_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 1:2] * brand_priors * F.softplus(self.brand_scale)
        if in_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 2:3] * in_priors * F.softplus(self.in_scale)
        if out_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 3:4] * out_priors * F.softplus(self.out_scale)
        if tld_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 4:5] * tld_priors * F.softplus(self.tld_scale)

        if direct_links is not None:
            logits = logits + direct_links * F.softplus(self.direct_scale)

        logits = logits + priors_blend
        return logits


class DenseBlock(nn.Module):
    """Dense highway block with LayerNorm, Linear, SiLU, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.30):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.act(self.linear(out))
        out = self.dropout(out)
        return out


class TrackerDenseNet(nn.Module):
    """Model 2: Dense Highway Network with unclipped linear head and sample-conditioned dynamic prior gating."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 355,
        stem_dim: int = 256,
        layer_dims: list = None,
        bottleneck_dim: int = 384,
        dropout_rate: float = 0.30,
        init_bias: np.ndarray = None,
        cooccur_matrix: np.ndarray = None,
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [192, 192, 192]

        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_dim),
            nn.LayerNorm(stem_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        curr_dim = stem_dim
        self.dense_layers = nn.ModuleList()
        for l_dim in layer_dims:
            self.dense_layers.append(
                DenseBlock(curr_dim, l_dim, dropout_rate=dropout_rate)
            )
            curr_dim += l_dim

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(curr_dim),
            nn.Linear(curr_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.head = nn.Linear(bottleneck_dim, num_classes)
        if init_bias is not None:
            self.head.bias.data.copy_(torch.tensor(init_bias, dtype=torch.float32))

        # Dynamic sample-conditioned prior gating net
        self.support_dim = 21
        self.gating_net = PriorGatingNet(in_features=self.support_dim, hidden_dim=64, num_priors=5)

        # Prior scale parameters
        self.direct_scale = nn.Parameter(torch.full((num_classes,), 5.0))
        self.apex_scale = nn.Parameter(torch.full((num_classes,), 4.0))
        self.brand_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.in_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.out_scale = nn.Parameter(torch.full((num_classes,), 3.0))
        self.tld_scale = nn.Parameter(torch.full((num_classes,), 2.0))

    def forward(
        self,
        x: torch.Tensor,
        direct_links: torch.Tensor = None,
        apex_priors: torch.Tensor = None,
        brand_priors: torch.Tensor = None,
        tld_priors: torch.Tensor = None,
        in_priors: torch.Tensor = None,
        out_priors: torch.Tensor = None,
    ) -> torch.Tensor:
        feats = [self.stem(x)]
        for layer in self.dense_layers:
            in_cat = torch.cat(feats, dim=-1)
            out_layer = layer(in_cat)
            feats.append(out_layer)

        all_feats = torch.cat(feats, dim=-1)
        bottleneck_out = self.bottleneck(all_feats)
        logits = self.head(bottleneck_out)

        # Dynamic prior gating
        support_feats = x[:, : self.support_dim]
        gates = self.gating_net(support_feats)

        priors_blend = torch.zeros_like(logits)
        if apex_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 0:1] * apex_priors * F.softplus(self.apex_scale)
        if brand_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 1:2] * brand_priors * F.softplus(self.brand_scale)
        if in_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 2:3] * in_priors * F.softplus(self.in_scale)
        if out_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 3:4] * out_priors * F.softplus(self.out_scale)
        if tld_priors is not None:
            priors_blend = priors_blend + 5.0 * gates[:, 4:5] * tld_priors * F.softplus(self.tld_scale)

        if direct_links is not None:
            logits = logits + direct_links * F.softplus(self.direct_scale)

        logits = logits + priors_blend
        return logits


criterion = AsymmetricLoss(gamma_neg=2.0, pos_weight=2.5).to(device)

# =============================================================================
# 4. TRAINING, VALIDATION & INFERENCE PIPELINE
# =============================================================================

class ModelEMA:
    """Maintains Exponential Moving Average (EMA) of model parameters for evaluation stability."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
        self.backup = {}

    def get_state_dict(self, model: nn.Module):
        state = model.state_dict()
        for name, tensor in self.shadow.items():
            if name in state:
                state[name] = tensor.clone()
        return state


Y_train_dense = Y_train.toarray().astype(np.float32)
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(direct_links_train),
    torch.from_numpy(apex_priors_train),
    torch.from_numpy(brand_priors_train),
    torch.from_numpy(tld_priors_train),
    torch.from_numpy(in_priors_train),
    torch.from_numpy(out_priors_train),
    torch.from_numpy(Y_train_dense),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4096,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


def compute_recall_at_10(
    eval_model,
    eval_features,
    eval_direct_links,
    eval_apex_priors,
    eval_targets_csr,
    batch_size=4096,
    eval_brand_priors=None,
    eval_tld_priors=None,
    eval_in_priors=None,
    eval_out_priors=None,
):
    """Computes exact official Recall@10 across all evaluation domains."""
    eval_model.eval()
    n_samples = eval_features.shape[0]
    top10_list = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.from_numpy(eval_features[start_idx:end_idx]).to(device)
            batch_links = torch.from_numpy(
                eval_direct_links[start_idx:end_idx]
            ).to(device)
            batch_apex = torch.from_numpy(
                eval_apex_priors[start_idx:end_idx]
            ).to(device)
            batch_brand = (
                torch.from_numpy(eval_brand_priors[start_idx:end_idx]).to(device)
                if eval_brand_priors is not None
                else None
            )
            batch_tld = (
                torch.from_numpy(eval_tld_priors[start_idx:end_idx]).to(device)
                if eval_tld_priors is not None
                else None
            )
            batch_in = (
                torch.from_numpy(eval_in_priors[start_idx:end_idx]).to(device)
                if eval_in_priors is not None
                else None
            )
            batch_out = (
                torch.from_numpy(eval_out_priors[start_idx:end_idx]).to(device)
                if eval_out_priors is not None
                else None
            )
            logits = eval_model(
                batch_x,
                batch_links,
                batch_apex,
                batch_brand,
                batch_tld,
                batch_in,
                batch_out,
            )
            top10_batch = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
            top10_list.append(top10_batch)

    top10_preds = np.vstack(top10_list)

    # Vectorized sparse Recall@10 evaluation
    row_indices = np.repeat(np.arange(n_samples), 10)
    col_indices = top10_preds.flatten()
    data = np.ones(len(col_indices), dtype=np.uint8)
    P_sparse = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=eval_targets_csr.shape,
        dtype=np.uint8,
    )

    hits = np.asarray(eval_targets_csr.multiply(P_sparse).sum(axis=1)).flatten()
    totals = np.asarray(eval_targets_csr.sum(axis=1)).flatten()
    recalls = np.where(totals > 0, hits / totals, 0.0)
    return float(np.mean(recalls)), top10_preds


def predict_probabilities(
    eval_model,
    eval_features,
    eval_direct_links,
    eval_apex_priors,
    batch_size=4096,
    eval_brand_priors=None,
    eval_tld_priors=None,
    eval_in_priors=None,
    eval_out_priors=None,
):
    """Predicts calibrated tracker probabilities via sigmoid activation."""
    eval_model.eval()
    n_samples = eval_features.shape[0]
    probs_list = []
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.from_numpy(eval_features[start_idx:end_idx]).to(device)
            batch_links = torch.from_numpy(
                eval_direct_links[start_idx:end_idx]
            ).to(device)
            batch_apex = torch.from_numpy(
                eval_apex_priors[start_idx:end_idx]
            ).to(device)
            batch_brand = (
                torch.from_numpy(eval_brand_priors[start_idx:end_idx]).to(device)
                if eval_brand_priors is not None
                else None
            )
            batch_tld = (
                torch.from_numpy(eval_tld_priors[start_idx:end_idx]).to(device)
                if eval_tld_priors is not None
                else None
            )
            batch_in = (
                torch.from_numpy(eval_in_priors[start_idx:end_idx]).to(device)
                if eval_in_priors is not None
                else None
            )
            batch_out = (
                torch.from_numpy(eval_out_priors[start_idx:end_idx]).to(device)
                if eval_out_priors is not None
                else None
            )
            logits = eval_model(
                batch_x,
                batch_links,
                batch_apex,
                batch_brand,
                batch_tld,
                batch_in,
                batch_out,
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list)


def evaluate_probs_recall_at_10(probs, targets_csr):
    """Evaluates Recall@10 directly from continuous probability distributions."""
    probs_t = torch.from_numpy(probs)
    top10_preds = torch.topk(probs_t, k=10, dim=1).indices.numpy()
    n_samples = probs.shape[0]
    row_indices = np.repeat(np.arange(n_samples), 10)
    col_indices = top10_preds.flatten()
    data = np.ones(len(col_indices), dtype=np.uint8)
    P_sparse = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=targets_csr.shape,
        dtype=np.uint8,
    )
    hits = np.asarray(targets_csr.multiply(P_sparse).sum(axis=1)).flatten()
    totals = np.asarray(targets_csr.sum(axis=1)).flatten()
    recalls = np.where(totals > 0, hits / totals, 0.0)
    return float(np.mean(recalls)), top10_preds


def train_single_model(
    model,
    model_name: str,
    best_path: str,
    epochs: int = 10,
):
    """Trains a model with 1-epoch linear warmup, 9-epoch cosine annealing, EMA tracking, and validation Recall@10 checkpointing."""
    bypass_params = [
        model.direct_scale,
        model.apex_scale,
        model.brand_scale,
        model.in_scale,
        model.out_scale,
        model.tld_scale,
    ] + list(model.gating_net.parameters())
    bypass_param_ids = {id(p) for p in bypass_params}
    base_params = [p for p in model.parameters() if id(p) not in bypass_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": 1e-3, "weight_decay": 1e-4},
            {"params": bypass_params, "lr": 1e-4, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=1
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - 1, 1), eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[1]
    )

    ema = ModelEMA(model, decay=0.999)
    best_val_recall = -1.0
    best_epoch = 0

    print(f"\n=======================================================")
    print(f"Training Architecture: {model_name} ({epochs} epochs)")
    print(f"=======================================================")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for (
            batch_x,
            batch_links,
            batch_apex,
            batch_brand,
            batch_tld,
            batch_in,
            batch_out,
            batch_y,
        ) in train_loader:
            batch_x = batch_x.to(device)
            batch_links = batch_links.to(device)
            batch_apex = batch_apex.to(device)
            batch_brand = batch_brand.to(device)
            batch_tld = batch_tld.to(device)
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(
                batch_x,
                batch_links,
                batch_apex,
                batch_brand,
                batch_tld,
                batch_in,
                batch_out,
            )
            loss = criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            total_train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Evaluate validation Recall@10 on smoothed EMA weights
        ema.apply_shadow(model)
        val_recall, _ = compute_recall_at_10(
            model,
            X_val,
            direct_links_val,
            apex_priors_val,
            Y_val,
            batch_size=4096,
            eval_brand_priors=brand_priors_val,
            eval_tld_priors=tld_priors_val,
            eval_in_priors=in_priors_val,
            eval_out_priors=out_priors_val,
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
            torch.save(ema.get_state_dict(model), best_path)

        ema.restore(model)

        print(
            f"[{model_name}] Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {avg_train_loss:.4f} - Val Recall@10 (EMA): {val_recall:.6f}"
        )

    print(
        f"[{model_name}] Checkpoint Loaded: Best Val Recall@10 = {best_val_recall:.6f} at Epoch {best_epoch}"
    )
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return model, best_val_recall


# 1. Instantiate and Train Model 1: TrackerResNet
model1 = TrackerResNet(
    input_dim=feature_dim,
    num_classes=num_trackers,
    hidden_dim=384,
    num_blocks=3,
    dropout_rate=0.30,
    init_bias=prior_logits,
    cooccur_matrix=tracker_cooccur_matrix,
).to(device)

model1, m1_score = train_single_model(
    model1,
    model_name="TrackerResNet",
    best_path="./working/tracker_resnet_best.pt",
    epochs=10,
)

# 2. Instantiate and Train Model 2: TrackerDenseNet
model2 = TrackerDenseNet(
    input_dim=feature_dim,
    num_classes=num_trackers,
    stem_dim=256,
    layer_dims=[192, 192, 192],
    bottleneck_dim=384,
    dropout_rate=0.30,
    init_bias=prior_logits,
    cooccur_matrix=tracker_cooccur_matrix,
).to(device)

model2, m2_score = train_single_model(
    model2,
    model_name="TrackerDenseNet",
    best_path="./working/tracker_densenet_best.pt",
    epochs=10,
)

# 3. Validation Ensembling & Optimal Weight Blending Search
print("\n--- Computing Validation Predictions for Ensemble Optimization ---")
P1_val = predict_probabilities(
    model1,
    X_val,
    direct_links_val,
    apex_priors_val,
    batch_size=4096,
    eval_brand_priors=brand_priors_val,
    eval_tld_priors=tld_priors_val,
    eval_in_priors=in_priors_val,
    eval_out_priors=out_priors_val,
)
P2_val = predict_probabilities(
    model2,
    X_val,
    direct_links_val,
    apex_priors_val,
    batch_size=4096,
    eval_brand_priors=brand_priors_val,
    eval_tld_priors=tld_priors_val,
    eval_in_priors=in_priors_val,
    eval_out_priors=out_priors_val,
)

best_blend_w = 0.5
best_ensemble_score = -1.0

blend_weights = np.linspace(0.0, 1.0, 21)
for w in blend_weights:
    P_blend = w * P1_val + (1.0 - w) * P2_val
    score, _ = evaluate_probs_recall_at_10(P_blend, Y_val)
    if score > best_ensemble_score:
        best_ensemble_score = score
        best_blend_w = float(w)

print(
    f"Ensemble Grid Search: Optimal Weight w={best_blend_w:.2f} (Model1) + {1.0-best_blend_w:.2f} (Model2)"
)
print(f"Model 1 Val Score: {m1_score:.6f}")
print(f"Model 2 Val Score: {m2_score:.6f}")
print(f"Ensemble Validation Score: {best_ensemble_score:.6f}")

# 4. Test Set Inference with Calibrated Ensemble Distribution
print("\n--- Generating Test Set Ensemble Predictions ---")
P1_test = predict_probabilities(
    model1,
    X_test,
    direct_links_test,
    apex_priors_test,
    batch_size=4096,
    eval_brand_priors=brand_priors_test,
    eval_tld_priors=tld_priors_test,
    eval_in_priors=in_priors_test,
    eval_out_priors=out_priors_test,
)
P2_test = predict_probabilities(
    model2,
    X_test,
    direct_links_test,
    apex_priors_test,
    batch_size=4096,
    eval_brand_priors=brand_priors_test,
    eval_tld_priors=tld_priors_test,
    eval_in_priors=in_priors_test,
    eval_out_priors=out_priors_test,
)

P_test = best_blend_w * P1_test + (1.0 - best_blend_w) * P2_test
test_top10_preds = (
    torch.topk(torch.from_numpy(P_test), k=10, dim=1).indices.numpy()
)

# Map predicted compact indices to tracking_domain_id
n_test = X_test.shape[0]
flat_domain_ids = np.repeat(test_domain_ids, 10)
flat_tracking_domain_ids = np.zeros(len(flat_domain_ids), dtype=np.int64)

for i in range(n_test):
    for j in range(10):
        t_idx = int(test_top10_preds[i, j])
        flat_tracking_domain_ids[i * 10 + j] = tracker_id_to_tracking_domain.get(
            t_idx, popular_tracking_domain_ids[j]
        )

submission_df = pd.DataFrame(
    {"domain_id": flat_domain_ids, "tracking_domain_id": flat_tracking_domain_ids}
)

# Output submission files in required TSV format
submission_df.to_csv("./submission/submission.csv", sep="\t", index=False)
submission_df.to_csv("./submission/submission.tsv", sep="\t", index=False)

print(f"Final Validation Score: {best_ensemble_score:.6f}")
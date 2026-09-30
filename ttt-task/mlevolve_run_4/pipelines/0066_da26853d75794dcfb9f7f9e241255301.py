import gc
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Set deterministic random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

global_start_time = time.time()
MAX_TOTAL_RUNTIME = 45 * 60  # 45-minute safety threshold

# =========================================================================
# STEP 1: DATA LOADING, PREPROCESSING & FEATURE ENGINEERING
# =========================================================================

print("--- Step 1: Loading Tracker Metadata and Target Domains ---")
trackers_df = pd.read_csv("./input/trackers.tsv", sep="\t")
num_trackers = len(trackers_df)
print(f"Loaded {num_trackers} tracker definitions.")

# Parse tracker taxonomy metadata (company, functional category, operating country, parent brand)
unique_categories = sorted(
    [c for c in trackers_df["category"].dropna().unique() if c != "#"]
)
cat_to_col = {c: i for i, c in enumerate(unique_categories)}

company_counts = (
    trackers_df[~trackers_df["company"].isin(["#", ""]) & trackers_df["company"].notnull()]["company"]
    .value_counts()
)
top_companies = sorted(company_counts[company_counts >= 2].index.tolist())
comp_to_col = {c: i for i, c in enumerate(top_companies)}

unique_countries = sorted(
    [c for c in trackers_df["country"].dropna().unique() if c != "#"]
)
country_to_col = {c: i for i, c in enumerate(unique_countries)}

brand_counts = (
    trackers_df[~trackers_df["brand"].isin(["#", ""]) & trackers_df["brand"].notnull()]["brand"]
    .value_counts()
)
top_brands = sorted(brand_counts[brand_counts >= 2].index.tolist())
brand_to_col = {b: i for i, b in enumerate(top_brands)}

print(
    f"Constructed tracker taxonomy mappings: {len(unique_categories)} categories, {len(top_companies)} corporate suites, {len(unique_countries)} countries, {len(top_brands)} brands."
)

M_cat = np.zeros((num_trackers, len(unique_categories)), dtype=np.float32)
M_comp = np.zeros((num_trackers, len(top_companies)), dtype=np.float32)
M_country = np.zeros((num_trackers, len(unique_countries)), dtype=np.float32)
M_brand = np.zeros((num_trackers, len(top_brands)), dtype=np.float32)

for _, row in trackers_df.iterrows():
    t_id = int(row["tracker_id"])
    c_cat = row.get("category")
    c_comp = row.get("company")
    c_country = row.get("country")
    c_brand = row.get("brand")
    if pd.notna(c_cat) and c_cat in cat_to_col:
        M_cat[t_id, cat_to_col[c_cat]] = 1.0
    if pd.notna(c_comp) and c_comp in comp_to_col:
        M_comp[t_id, comp_to_col[c_comp]] = 1.0
    if pd.notna(c_country) and c_country in country_to_col:
        M_country[t_id, country_to_col[c_country]] = 1.0
    if pd.notna(c_brand) and c_brand in brand_to_col:
        M_brand[t_id, brand_to_col[c_brand]] = 1.0

# Build lookup from tracking_domain_id to tracker_id (0..354)
tracker_domain_ids = trackers_df["tracking_domain_id"].values.astype(np.int64)
tracker_ids = trackers_df["tracker_id"].values.astype(np.int32)
sorted_t_indices = np.argsort(tracker_domain_ids)
sorted_tracker_domain_ids = tracker_domain_ids[sorted_t_indices]
sorted_tracker_ids = tracker_ids[sorted_t_indices]

# Tracker ID to Tracking Domain ID reverse lookup array
tracker_id_to_domain_id = np.zeros(num_trackers, dtype=np.int64)
for _, row in trackers_df.iterrows():
    tracker_id_to_domain_id[int(row["tracker_id"])] = int(row["tracking_domain_id"])

# Load target test domains
target_df = pd.read_csv("./input/target.tsv", sep="\t")
test_domain_ids = target_df["domain_id"].values.astype(np.int64)
num_test = len(test_domain_ids)
print(f"Loaded {num_test} target test domains.")

print("--- Step 2: Constructing Strict Train / Validation Splits ---")
train_graph_table = pq.read_table(
    "./input/tracking_graph_train.parquet", columns=["domain_id", "tracker_id"]
)
train_graph_df = train_graph_table.to_pandas()
del train_graph_table
gc.collect()

test_domain_set = set(test_domain_ids)
train_graph_filtered = train_graph_df[~train_graph_df["domain_id"].isin(test_domain_set)]
total_non_test_domains = train_graph_filtered["domain_id"].nunique()
tracker_pop = train_graph_filtered["tracker_id"].value_counts()
global_tracker_prior = np.zeros(num_trackers, dtype=np.float32)
for t_id, cnt in tracker_pop.items():
    global_tracker_prior[int(t_id)] = cnt / float(total_non_test_domains)

domain_counts = train_graph_filtered["domain_id"].value_counts()
multi_tracker_domains = domain_counts[domain_counts >= 2].index.values.astype(np.int64)
single_tracker_domains = domain_counts[domain_counts == 1].index.values.astype(np.int64)

rng = np.random.RandomState(42)
rng.shuffle(multi_tracker_domains)
rng.shuffle(single_tracker_domains)

# Select a representative, stratified pool of 325,000 domains (300k train, 25k val)
n_target_pool = 325000
n_multi = min(len(multi_tracker_domains), int(n_target_pool * 0.65))
n_single = n_target_pool - n_multi

selected_domains = np.concatenate([
    multi_tracker_domains[:n_multi],
    single_tracker_domains[:n_single]
])
rng.shuffle(selected_domains)

val_domain_ids = selected_domains[:25000]
train_domain_ids = selected_domains[25000:]

num_train = len(train_domain_ids)
num_val = len(val_domain_ids)
print(f"Dataset split: Train={num_train}, Val={num_val}, Test={num_test} domains.")

# Construct Ground Truth Multi-Label Matrices Y_train and Y_val
print("Building multi-label target matrices...")
Y_train = np.zeros((num_train, num_trackers), dtype=np.uint8)
Y_val = np.zeros((num_val, num_trackers), dtype=np.uint8)

train_map = {did: idx for idx, did in enumerate(train_domain_ids)}
val_map = {did: idx for idx, did in enumerate(val_domain_ids)}

selected_set = set(selected_domains)
sub_df = train_graph_filtered[train_graph_filtered["domain_id"].isin(selected_set)]
sub_doms = sub_df["domain_id"].values
sub_trks = sub_df["tracker_id"].values

is_train = np.isin(sub_doms, train_domain_ids)
t_doms = sub_doms[is_train]
t_trks = sub_trks[is_train]
train_row_idx = np.fromiter((train_map[d] for d in t_doms), dtype=np.int32, count=len(t_doms))
Y_train[train_row_idx, t_trks] = 1

is_val = np.isin(sub_doms, val_domain_ids)
v_doms = sub_doms[is_val]
v_trks = sub_trks[is_val]
val_row_idx = np.fromiter((val_map[d] for d in v_doms), dtype=np.int32, count=len(v_doms))
Y_val[val_row_idx, v_trks] = 1

# Index complete leak-free labeled tracker pool across all non-test, non-validation domains
print("Indexing complete leak-free labeled pool from tracking_graph_train.parquet...")
val_domain_set = set(val_domain_ids)
pool_mask = ~train_graph_filtered["domain_id"].isin(val_domain_set)
pool_df = train_graph_filtered[pool_mask]
pool_domains = pool_df["domain_id"].values.astype(np.int64)
pool_trk_col = pool_df["tracker_id"].values.astype(np.int32)

pool_unique_domains, pool_inv_indices = np.unique(pool_domains, return_inverse=True)
N_pool = len(pool_unique_domains)
pool_trackers = np.zeros((N_pool, num_trackers), dtype=np.uint8)
pool_trackers[pool_inv_indices, pool_trk_col] = 1

# Compute Adamic-Adar / Resource Allocation peer link-degree discount factor w_p = 1.0 / log(2.0 + peer_out_degree)
print("Pre-computing peer out-degrees across leak-free pool from link-graph.parquet...")
pool_out_degrees = np.zeros(N_pool, dtype=np.float32)
link_pq_pre = pq.ParquetFile("./input/link-graph.parquet")
for batch in link_pq_pre.iter_batches(columns=["source_domain_id"], batch_size=10000000):
    src = batch.column("source_domain_id").to_numpy(zero_copy_only=False)
    src_p_pos = np.searchsorted(pool_unique_domains, src)
    valid_src_p = src_p_pos < N_pool
    src_pool_matched = np.zeros(len(src), dtype=bool)
    src_pool_matched[valid_src_p] = (
        pool_unique_domains[src_p_pos[valid_src_p]] == src[valid_src_p]
    )
    if np.any(src_pool_matched):
        np.add.at(pool_out_degrees, src_p_pos[src_pool_matched], 1.0)
del link_pq_pre

pool_weights = (1.0 / np.log(2.0 + pool_out_degrees)).astype(np.float32)
print(f"Indexed complete leak-free labeled pool: {N_pool} domains across {len(pool_df)} tracking links.")
print(
    f"Adamic-Adar peer weights computed: mean={pool_weights.mean():.4f}, min={pool_weights.min():.4f}, max={pool_weights.max():.4f}"
)

del (
    train_graph_df,
    train_graph_filtered,
    pool_df,
    pool_domains,
    pool_trk_col,
    pool_inv_indices,
    pool_out_degrees,
    sub_df,
    sub_doms,
    sub_trks,
    is_train,
    is_val,
    t_doms,
    t_trks,
    v_doms,
    v_trks,
    train_row_idx,
    val_row_idx,
    selected_set,
)
gc.collect()

print(
    f"Targets constructed: Y_train positive rate: {Y_train.mean():.4f}, Y_val positive rate: {Y_val.mean():.4f}"
)

print("Computing empirical NPMI graph diffusion matrix strictly from Y_train...")
N_train = float(num_train)
Y_train_f = Y_train.astype(np.float32)
C_matrix = (Y_train_f.T @ Y_train_f).astype(np.float64)
c_diag = np.diag(C_matrix)
p_ij = C_matrix / N_train
p_i = c_diag / N_train

with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((p_ij + 1e-12) / (p_i[:, None] * p_i[None, :] + 1e-12))
    denom = -np.log(p_ij + 1e-12)
    npmi = pmi / denom
    npmi[C_matrix == 0] = -1.0
    np.fill_diagonal(npmi, 1.0)

# Zero out values below 0.15, ensure symmetry and add self-loops
A_npmi = np.where(npmi >= 0.15, npmi, 0.0)
A_npmi = 0.5 * (A_npmi + A_npmi.T)
np.fill_diagonal(A_npmi, 1.0)

# Symmetric Laplacian normalization D^(-1/2) * A * D^(-1/2)
deg = np.sum(A_npmi, axis=1)
deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
A_npmi_norm = A_npmi * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]
A_npmi_tensor = torch.tensor(A_npmi_norm, dtype=torch.float32)

# Precompute 2nd-order diffusion matrix A^2 for polynomial graph diffusion
A_npmi_2_norm = A_npmi_norm @ A_npmi_norm
A_npmi_2_tensor = torch.tensor(A_npmi_2_norm, dtype=torch.float32)

# Prepare tracker taxonomy feature tensor for Dual-Tower alignment head
M_taxonomy = np.hstack([M_cat, M_comp, M_country, M_brand]).astype(np.float32)
tracker_taxonomy_tensor = torch.tensor(M_taxonomy, dtype=torch.float32)
print(
    f"Precomputed NPMI graph diffusion matrices: 1st-order {A_npmi_tensor.shape}, 2nd-order {A_npmi_2_tensor.shape}."
)

print("--- Step 3: Unifying Relevant Domains and Mapping Hostnames ---")
all_domain_ids = np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
num_total = len(all_domain_ids)
sorted_all_indices = np.argsort(all_domain_ids)
sorted_all_domain_ids = all_domain_ids[sorted_all_indices]

domain_to_hostname = {}
domains_pq = pq.ParquetFile("./input/domains.parquet")
all_domain_set = set(all_domain_ids)

for batch in domains_pq.iter_batches(
    columns=["domain_id", "domain"], batch_size=2500000
):
    b_df = batch.to_pandas()
    b_matched = b_df[b_df["domain_id"].isin(all_domain_set)]
    if len(b_matched) > 0:
        for did, host in zip(b_matched["domain_id"], b_matched["domain"]):
            domain_to_hostname[did] = str(host)

print(
    f"Mapped {len(domain_to_hostname)} / {num_total} domain hostnames from domains.parquet."
)

multi_tlds = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.jp",
    "ne.jp",
    "com.br",
    "com.cn",
    "net.cn",
    "gov.cn",
    "edu.cn",
    "co.in",
    "net.in",
    "org.in",
    "com.ru",
    "net.ru",
    "org.ru",
    "co.nz",
    "com.tw",
    "com.mx",
    "co.za",
    "com.ar",
    "com.tr",
    "com.pl",
}


def parse_hostname(h_str):
    if not h_str or h_str == "nan":
        return "unknown", "unknown", 0, 0, 0, 0, 0
    h = h_str.strip().lower()
    if h.startswith("www."):
        h = h[4:]
    parts = h.split(".")
    length = len(h)
    sub_count = max(0, len(parts) - 2)

    if len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in multi_tlds:
        tld = f"{parts[-2]}.{parts[-1]}"
        sld = parts[-3] if len(parts) >= 3 else ""
    elif len(parts) >= 1:
        tld = parts[-1]
        sld = parts[-2] if len(parts) >= 2 else parts[-1]
    else:
        tld = "unknown"
        sld = ""

    digits = sum(c.isdigit() for c in h)
    hyphens = h.count("-")
    vowels = sum(c in "aeiou" for c in h)
    return tld, sld, sub_count, length, digits, hyphens, vowels


parsed_tld = []
parsed_sld = []
parsed_sub_count = np.zeros(num_total, dtype=np.float32)
parsed_len = np.zeros(num_total, dtype=np.float32)
parsed_digits = np.zeros(num_total, dtype=np.float32)
parsed_hyphens = np.zeros(num_total, dtype=np.float32)
parsed_vowels = np.zeros(num_total, dtype=np.float32)
hostname_list = []

for idx, did in enumerate(all_domain_ids):
    h = domain_to_hostname.get(did, "")
    hostname_list.append(h)
    tld, sld, subs, l, d, hyp, v = parse_hostname(h)
    parsed_tld.append(tld)
    parsed_sld.append(sld)
    parsed_sub_count[idx] = subs
    parsed_len[idx] = l
    parsed_digits[idx] = d
    parsed_hyphens[idx] = hyp
    parsed_vowels[idx] = v

print("--- Step 4: Joining External Metadata (Press Freedom & Content Categories) ---")
df_fop = pd.read_csv("./input/freedom-of-the-press.csv", sep="\t")
tld_to_fop = {}
for _, row in df_fop.iterrows():
    tld_clean = str(row["tld"]).strip().lower()
    try:
        score = float(row["freedom_of_the_press"])
        tld_to_fop[tld_clean] = score
    except Exception:
        pass
median_fop = np.median(list(tld_to_fop.values())) if tld_to_fop else 30.0

fop_scores = np.full(num_total, median_fop, dtype=np.float32)
fop_matched = np.zeros(num_total, dtype=np.float32)

for idx, tld in enumerate(parsed_tld):
    direct_match = tld_to_fop.get(tld)
    if direct_match is not None:
        fop_scores[idx] = direct_match
        fop_matched[idx] = 1.0
    else:
        last_tld = tld.split(".")[-1]
        if last_tld in tld_to_fop:
            fop_scores[idx] = tld_to_fop[last_tld]
            fop_matched[idx] = 1.0

print("Parsing url-classification.csv and training universal topic transfer classifier...")
url_df = pd.read_csv("./input/url-classification.csv", usecols=["url", "category"])
url_df = url_df.dropna(subset=["url", "category"])
cat_list = sorted(url_df["category"].unique().tolist())
cat_to_idx = {cat: i for i, cat in enumerate(cat_list)}
num_cats = len(cat_list)
print(f"Extracted {num_cats} categories across {len(url_df)} labeled URLs.")

# 1. Exact string matching lookup
host_to_cat = {}
raw_urls = url_df["url"].astype(str).values
url_cats = url_df["category"].values

for u, c in zip(raw_urls, url_cats):
    if "://" in u:
        u = u.split("://", 1)[1]
    h = u.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].lower()
    if h.startswith("www."):
        h = h[4:]
    if h and h not in host_to_cat:
        host_to_cat[h] = cat_to_idx.get(c, -1)

url_exact_features = np.zeros((num_total, num_cats + 1), dtype=np.float32)
for idx, did in enumerate(all_domain_ids):
    h = domain_to_hostname.get(did, "").lower()
    if h.startswith("www."):
        h = h[4:]
    c_idx = host_to_cat.get(h, -1)
    if c_idx >= 0:
        url_exact_features[idx, c_idx] = 1.0
        url_exact_features[idx, -1] = 1.0

del host_to_cat
print(f"Exact URL categories matched for {url_exact_features[:, -1].sum():.0f} domains.")

# 2. Universal topic transfer classifier on character 3-5 grams
print("Fitting fast character n-gram SGDClassifier on labeled URLs...")
url_hashing = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    n_features=2**17,
    alternate_sign=False,
    norm="l2",
)
X_url_tokens = url_hashing.transform(raw_urls)
y_url_labels = np.array([cat_to_idx[c] for c in url_cats], dtype=np.int32)
del raw_urls, url_cats, url_df
gc.collect()

sgd_clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
    max_iter=12,
    tol=1e-3,
    random_state=42,
    n_jobs=-1,
)
sgd_clf.fit(X_url_tokens, y_url_labels)
del X_url_tokens, y_url_labels
gc.collect()

print("Inferring continuous posterior topic distributions across all 325,000 hostnames...")
X_host_tokens = url_hashing.transform(hostname_list)
url_topic_probs = sgd_clf.predict_proba(X_host_tokens).astype(np.float32)
del X_host_tokens, sgd_clf, url_hashing
gc.collect()

# 3. 32 Functional Role Keyword Intent Flags
print("Extracting 32 functional role keyword intent flags...")
functional_keywords = [
    "shop", "store", "news", "press", "blog", "dev", "adult", "casino",
    "forum", "mail", "wiki", "video", "tv", "music", "game", "play",
    "live", "download", "soft", "tech", "app", "info", "travel", "food",
    "health", "bank", "pay", "crypto", "trade", "job", "school", "edu",
]
keyword_intent_flags = np.zeros((num_total, len(functional_keywords)), dtype=np.float32)
for k_idx, kw in enumerate(functional_keywords):
    keyword_intent_flags[:, k_idx] = [1.0 if kw in h else 0.0 for h in hostname_list]

print(
    "--- Step 5: Streaming link-graph.parquet for Graph Features, Direct Links, and Peer Homophily ---"
)
inv_sorted_indices = np.empty_like(sorted_all_indices)
inv_sorted_indices[sorted_all_indices] = np.arange(num_total)

train_sorted_positions = inv_sorted_indices[:num_train]
is_train_sorted = np.zeros(num_total, dtype=bool)
is_train_sorted[train_sorted_positions] = True

train_trackers_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
train_trackers_sorted[train_sorted_positions] = Y_train.astype(np.float32)

in_degrees_sorted = np.zeros(num_total, dtype=np.float32)
out_degrees_sorted = np.zeros(num_total, dtype=np.float32)
tracker_direct_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
in_neighbor_tracker_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
out_neighbor_tracker_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
in_neighbor_degree_sorted = np.zeros(num_total, dtype=np.float32)
out_neighbor_degree_sorted = np.zeros(num_total, dtype=np.float32)

link_pq = pq.ParquetFile("./input/link-graph.parquet")
batch_count = 0

for batch in link_pq.iter_batches(
    columns=["source_domain_id", "target_domain_id"], batch_size=5000000
):
    src = batch.column("source_domain_id").to_numpy(zero_copy_only=False)
    dst = batch.column("target_domain_id").to_numpy(zero_copy_only=False)

    src_pos = np.searchsorted(sorted_all_domain_ids, src)
    valid_src = src_pos < num_total
    src_matched = np.zeros(len(src), dtype=bool)
    src_matched[valid_src] = sorted_all_domain_ids[src_pos[valid_src]] == src[valid_src]
    np.add.at(out_degrees_sorted, src_pos[src_matched], 1.0)

    dst_pos = np.searchsorted(sorted_all_domain_ids, dst)
    valid_dst = dst_pos < num_total
    dst_matched = np.zeros(len(dst), dtype=bool)
    dst_matched[valid_dst] = sorted_all_domain_ids[dst_pos[valid_dst]] == dst[valid_dst]
    np.add.at(in_degrees_sorted, dst_pos[dst_matched], 1.0)

    t_pos = np.searchsorted(sorted_tracker_domain_ids, dst)
    valid_t = t_pos < len(sorted_tracker_domain_ids)
    t_matched = np.zeros(len(dst), dtype=bool)
    t_matched[valid_t] = sorted_tracker_domain_ids[t_pos[valid_t]] == dst[valid_t]

    tracker_link_mask = src_matched & t_matched
    if np.any(tracker_link_mask):
        match_src_indices = src_pos[tracker_link_mask]
        match_trk_indices = sorted_tracker_ids[t_pos[tracker_link_mask]]
        np.add.at(tracker_direct_sorted, (match_src_indices, match_trk_indices), 1.0)

    # Peer-to-peer collaborative neighbor tracker homophily across complete leak-free labeled pool
    non_self = (src != dst)

    src_p_pos = np.searchsorted(pool_unique_domains, src)
    valid_src_p = src_p_pos < N_pool
    src_pool_matched = np.zeros(len(src), dtype=bool)
    src_pool_matched[valid_src_p] = (
        pool_unique_domains[src_p_pos[valid_src_p]] == src[valid_src_p]
    )

    dst_p_pos = np.searchsorted(pool_unique_domains, dst)
    valid_dst_p = dst_p_pos < N_pool
    dst_pool_matched = np.zeros(len(dst), dtype=bool)
    dst_pool_matched[valid_dst_p] = (
        pool_unique_domains[dst_p_pos[valid_dst_p]] == dst[valid_dst_p]
    )

    # 1. Inbound peer citation evidence: src in leak-free pool, dst in split set, weighted by 1 / log(1 + deg)
    in_mask = src_pool_matched & dst_matched & non_self
    if np.any(in_mask):
        s_p_idx = src_p_pos[in_mask]
        d_idx = dst_pos[in_mask]
        w_in = pool_weights[s_p_idx]
        np.add.at(
            in_neighbor_tracker_sorted,
            d_idx,
            pool_trackers[s_p_idx] * w_in[:, None],
        )
        np.add.at(in_neighbor_degree_sorted, d_idx, w_in)

    # 2. Outbound peer navigation evidence: dst in leak-free pool, src in split set, weighted by 1 / log(1 + deg)
    out_mask = dst_pool_matched & src_matched & non_self
    if np.any(out_mask):
        s_idx = src_pos[out_mask]
        d_p_idx = dst_p_pos[out_mask]
        w_out = pool_weights[d_p_idx]
        np.add.at(
            out_neighbor_tracker_sorted,
            s_idx,
            pool_trackers[d_p_idx] * w_out[:, None],
        )
        np.add.at(out_neighbor_degree_sorted, s_idx, w_out)

    batch_count += 1

print(f"Link graph streaming complete across {batch_count} batches.")

in_degrees = in_degrees_sorted[inv_sorted_indices]
out_degrees = out_degrees_sorted[inv_sorted_indices]
tracker_direct_raw = tracker_direct_sorted[inv_sorted_indices]
tracker_direct_all = np.log1p(tracker_direct_raw)

# Binary direct-link indicator flags differentiating qualitative presence from hyperlinking noise
tracker_direct_binary = (tracker_direct_raw > 0).astype(np.float32)

# Calculate direct tracker out-degree ratios and intensity
total_tracker_out_links = tracker_direct_raw.sum(axis=1)
tracker_intensity_ratio = (total_tracker_out_links / (out_degrees + 1.0)).astype(np.float32)
log1p_tracker_out_total = np.log1p(total_tracker_out_links).astype(np.float32)

in_neighbor_raw = in_neighbor_tracker_sorted[inv_sorted_indices]
out_neighbor_raw = out_neighbor_tracker_sorted[inv_sorted_indices]
in_neighbor_degrees = in_neighbor_degree_sorted[inv_sorted_indices]
out_neighbor_degrees = out_neighbor_degree_sorted[inv_sorted_indices]

in_degree_norm = np.maximum(in_neighbor_degrees[:, None], 1.0)
out_degree_norm = np.maximum(out_neighbor_degrees[:, None], 1.0)

tracker_in_neighbor_all = np.log1p(in_neighbor_raw / in_degree_norm).astype(np.float32)
tracker_out_neighbor_all = np.log1p(out_neighbor_raw / out_degree_norm).astype(np.float32)
np.nan_to_num(tracker_in_neighbor_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(tracker_out_neighbor_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# Shape: (num_total, num_trackers, 2)
tracker_neighbor_all = np.stack(
    [tracker_in_neighbor_all, tracker_out_neighbor_all], axis=-1
)

log1p_in_neighbor_degree = np.log1p(in_neighbor_degrees).reshape(-1, 1).astype(np.float32)
log1p_out_neighbor_degree = np.log1p(out_neighbor_degrees).reshape(-1, 1).astype(np.float32)

# Aggregate direct, in-neighbor, and out-neighbor signals across country, category, company, and brand levels
direct_cat_agg = np.log1p(tracker_direct_all @ M_cat)
direct_comp_agg = np.log1p(tracker_direct_all @ M_comp)
direct_country_agg = np.log1p(tracker_direct_all @ M_country)
direct_brand_agg = np.log1p(tracker_direct_all @ M_brand)

in_neighbor_cat_agg = np.log1p(tracker_in_neighbor_all @ M_cat)
in_neighbor_comp_agg = np.log1p(tracker_in_neighbor_all @ M_comp)
in_neighbor_country_agg = np.log1p(tracker_in_neighbor_all @ M_country)
in_neighbor_brand_agg = np.log1p(tracker_in_neighbor_all @ M_brand)

out_neighbor_cat_agg = np.log1p(tracker_out_neighbor_all @ M_cat)
out_neighbor_comp_agg = np.log1p(tracker_out_neighbor_all @ M_comp)
out_neighbor_country_agg = np.log1p(tracker_out_neighbor_all @ M_country)
out_neighbor_brand_agg = np.log1p(tracker_out_neighbor_all @ M_brand)

del (
    in_degrees_sorted,
    out_degrees_sorted,
    tracker_direct_sorted,
    tracker_direct_raw,
    in_neighbor_tracker_sorted,
    out_neighbor_tracker_sorted,
    in_neighbor_degree_sorted,
    out_neighbor_degree_sorted,
    train_trackers_sorted,
    is_train_sorted,
    in_neighbor_raw,
    out_neighbor_raw,
    in_neighbor_degrees,
    out_neighbor_degrees,
    in_degree_norm,
    out_degree_norm,
    pool_trackers,
    pool_unique_domains,
    pool_weights,
)
gc.collect()

log1p_in = np.log1p(in_degrees)
log1p_out = np.log1p(out_degrees)
degree_ratio = (log1p_in + 1.0) / (log1p_out + 1.0)
log1p_total = np.log1p(in_degrees + out_degrees)
is_leaf = (out_degrees == 0).astype(np.float32)
is_sink = (in_degrees == 0).astype(np.float32)

print(
    "--- Step 6: Computing Lexical N-Gram SVD Features (Fitted Strictly on Train) ---"
)
train_hostnames = [hostname_list[i] for i in range(num_train)]

tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    min_df=5,
    max_features=10000,
    sublinear_tf=True,
)
svd = TruncatedSVD(n_components=128, random_state=42)

print("Fitting TF-IDF and SVD (128 components) on training hostnames strictly...")
train_tfidf = tfidf.fit_transform(train_hostnames)
svd.fit(train_tfidf)
del train_tfidf
gc.collect()

all_tfidf = tfidf.transform(hostname_list)
all_svd_features = svd.transform(all_tfidf).astype(np.float32)
del all_tfidf, tfidf, svd
gc.collect()

print(
    "--- Step 7: Bayesian Empirical TLD Tracker Priors (Fitted Strictly on Train) ---"
)
# Retain global_tracker_prior derived across the complete 18.66M-domain crawl
train_tlds = parsed_tld[:num_train]
tld_counts_train = {}
tld_tracker_sums_train = {}

for i in range(num_train):
    t = train_tlds[i]
    tld_counts_train[t] = tld_counts_train.get(t, 0) + 1
    if t not in tld_tracker_sums_train:
        tld_tracker_sums_train[t] = Y_train[i].astype(np.float32).copy()
    else:
        tld_tracker_sums_train[t] += Y_train[i]

alpha = 25.0
tld_bayesian_priors = {}
for t, n in tld_counts_train.items():
    smoothed = (tld_tracker_sums_train[t] + alpha * global_tracker_prior) / (n + alpha)
    tld_bayesian_priors[t] = smoothed.astype(np.float32)

tld_priors_all = np.zeros((num_total, num_trackers), dtype=np.float32)
for idx, tld in enumerate(parsed_tld):
    tld_priors_all[idx] = tld_bayesian_priors.get(tld, global_tracker_prior)

top_tlds = [
    t
    for t, _ in sorted(tld_counts_train.items(), key=lambda x: x[1], reverse=True)[:24]
]
top_tld_to_idx = {t: i for i, t in enumerate(top_tlds)}
tld_one_hot = np.zeros((num_total, len(top_tlds) + 1), dtype=np.float32)
tld_freq_feature = np.zeros((num_total, 1), dtype=np.float32)

for idx, tld in enumerate(parsed_tld):
    count = tld_counts_train.get(tld, 0)
    tld_freq_feature[idx, 0] = np.log1p(count)
    if tld in top_tld_to_idx:
        tld_one_hot[idx, top_tld_to_idx[tld]] = 1.0
    else:
        tld_one_hot[idx, -1] = 1.0

print("--- Step 8: Assembling Dense Feature Matrix and Standardizing ---")
digit_ratio = parsed_digits / np.maximum(parsed_len, 1.0)
vowel_ratio = parsed_vowels / np.maximum(parsed_len, 1.0)

dense_feature_blocks = [
    in_degrees.reshape(-1, 1),
    out_degrees.reshape(-1, 1),
    log1p_in.reshape(-1, 1),
    log1p_out.reshape(-1, 1),
    degree_ratio.reshape(-1, 1),
    log1p_total.reshape(-1, 1),
    is_leaf.reshape(-1, 1),
    is_sink.reshape(-1, 1),
    tracker_intensity_ratio.reshape(-1, 1),
    log1p_tracker_out_total.reshape(-1, 1),
    log1p_in_neighbor_degree,
    log1p_out_neighbor_degree,
    parsed_sub_count.reshape(-1, 1),
    parsed_len.reshape(-1, 1),
    parsed_digits.reshape(-1, 1),
    digit_ratio.reshape(-1, 1),
    parsed_hyphens.reshape(-1, 1),
    parsed_vowels.reshape(-1, 1),
    vowel_ratio.reshape(-1, 1),
    fop_scores.reshape(-1, 1),
    fop_matched.reshape(-1, 1),
    url_topic_probs,
    url_exact_features,
    keyword_intent_flags,
    tld_one_hot,
    tld_freq_feature,
    all_svd_features,
    direct_cat_agg,
    direct_comp_agg,
    direct_country_agg,
    direct_brand_agg,
    in_neighbor_cat_agg,
    in_neighbor_comp_agg,
    in_neighbor_country_agg,
    in_neighbor_brand_agg,
    out_neighbor_cat_agg,
    out_neighbor_comp_agg,
    out_neighbor_country_agg,
    out_neighbor_brand_agg,
    tracker_direct_binary,
]

X_dense_all = np.hstack(dense_feature_blocks).astype(np.float32)
np.nan_to_num(X_dense_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_dense_train = X_dense_all[:num_train]
X_dense_val = X_dense_all[num_train : num_train + num_val]
X_dense_test = X_dense_all[num_train + num_val :]

tracker_direct_train = tracker_direct_all[:num_train]
tracker_direct_val = tracker_direct_all[num_train : num_train + num_val]
tracker_direct_test = tracker_direct_all[num_train + num_val :]

tracker_neighbor_train = tracker_neighbor_all[:num_train]
tracker_neighbor_val = tracker_neighbor_all[num_train : num_train + num_val]
tracker_neighbor_test = tracker_neighbor_all[num_train + num_val :]

tld_prior_train = tld_priors_all[:num_train]
tld_prior_val = tld_priors_all[num_train : num_train + num_val]
tld_prior_test = tld_priors_all[num_train + num_val :]

print("Fitting StandardScaler strictly on training split...")
scaler = StandardScaler()
X_dense_train = scaler.fit_transform(X_dense_train).astype(np.float32)
X_dense_val = scaler.transform(X_dense_val).astype(np.float32)
X_dense_test = scaler.transform(X_dense_test).astype(np.float32)

dense_feature_dim = X_dense_train.shape[1]
print(f"Features prepared: dense_feature_dim={dense_feature_dim}")

# =========================================================================
# STEP 2: MODEL ARCHITECTURE & RANKING OBJECTIVE
# =========================================================================


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResNetSEBlock(nn.Module):
    """Residual block with LayerNorm, GELU, Squeeze-and-Excitation, and Dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = SqueezeExcitation(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(self.act(self.linear1(self.norm1(x))))
        out = self.linear2(self.norm2(out))
        out = self.se(out)
        out = self.dropout(out)
        return residual + out


class LowRankCrossTrackerLinear(nn.Module):
    """Combines a per-tracker diagonal weight vector with a rank-32 factored

    bilinear matrix (x * w_diag + x @ U @ V.T + bias) for cross-tracker evidence propagation.
    """

    def __init__(
        self, num_trackers: int = 355, rank: int = 32, init_diag: float = 2.5
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.rank = rank
        self.w_diag = nn.Parameter(torch.ones(num_trackers) * init_diag)
        self.U = nn.Parameter(torch.empty(num_trackers, rank))
        self.V = nn.Parameter(torch.empty(num_trackers, rank))
        self.bias = nn.Parameter(torch.zeros(num_trackers))

        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.normal_(self.V, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diag_part = x * self.w_diag
        low_rank_part = torch.matmul(torch.matmul(x, self.U), self.V.t())
        return diag_part + low_rank_part + self.bias


class PolynomialNPMIDiffusion(nn.Module):
    """2nd-order polynomial empirical NPMI graph diffusion layer:
    z_out = z + gamma_1 * (z @ A_npmi) + gamma_2 * (z @ A_npmi_2)
    with learned per-tracker parameter vectors gamma_1 and gamma_2.
    """

    def __init__(
        self,
        A_npmi: torch.Tensor,
        A_npmi_2: torch.Tensor = None,
        num_trackers: int = 355,
    ):
        super().__init__()
        if A_npmi is None:
            A_npmi = torch.eye(num_trackers)
        self.register_buffer("A_npmi", A_npmi)

        if A_npmi_2 is None:
            A_npmi_2 = torch.matmul(A_npmi, A_npmi)
        self.register_buffer("A_npmi_2", A_npmi_2)

        num_trk = A_npmi.shape[0]
        self.gamma_1 = nn.Parameter(torch.zeros(num_trk))
        self.gamma_2 = nn.Parameter(torch.zeros(num_trk))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_A1 = torch.matmul(z, self.A_npmi)
        z_A2 = torch.matmul(z, self.A_npmi_2)
        return z + self.gamma_1 * z_A1 + self.gamma_2 * z_A2


ResidualNPMIDiffusion = PolynomialNPMIDiffusion


class TrackerRelationalSelfAttentionNet(nn.Module):
    """Model 1: Domain-Conditioned Tracker Relational Self-Attention (TRSA) network

    with Cross-Stage Partial Dense-Residual backbone, 4-head relational self-attention
    over 355 domain-conditioned tracker tokens, 2nd-order polynomial NPMI graph diffusion,
    and multi-channel directional low-rank cross-tracker bypasses.
    """

    def __init__(
        self,
        dense_dim: int,
        A_npmi: torch.Tensor = None,
        A_npmi_2: torch.Tensor = None,
        tracker_taxonomy: torch.Tensor = None,
        num_trackers: int = 355,
        hidden_dim: int = 384,
        embed_dim: int = 256,
        token_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.token_dim = token_dim

        # 1. Tracker Context Projection into Stem Alongside x_dense
        self.tracker_context = nn.Linear(num_trackers * 4, 64)

        # 2. Backbone with Cross-Stage Partial Dense-Residual connections
        self.stem = nn.Sequential(
            nn.Linear(dense_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage1 = ResNetSEBlock(hidden_dim, dropout=dropout)
        self.trans1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage2 = ResNetSEBlock(hidden_dim, dropout=dropout)
        self.stage_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Latent linear head from backbone representation
        self.latent_head = nn.Linear(hidden_dim, num_trackers)

        # 3. Domain-Conditioned Tracker Relational Self-Attention (TRSA)
        if tracker_taxonomy is None:
            tracker_taxonomy = torch.zeros(num_trackers, 1)
        self.register_buffer("tracker_taxonomy", tracker_taxonomy)
        tax_dim = tracker_taxonomy.shape[1]

        self.tracker_tax_proj = nn.Sequential(
            nn.Linear(tax_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.tracker_base_embed = nn.Parameter(torch.empty(num_trackers, token_dim))
        nn.init.normal_(self.tracker_base_embed, mean=0.0, std=0.02)

        # Project 4 tracker-specific signals (direct, in-neighbor, out-neighbor, prior_logits)
        self.tracker_signal_proj = nn.Linear(4, token_dim)

        # Project domain dense backbone context to condition tracker tokens
        self.domain_ctx_proj = nn.Sequential(
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # 4-head Multi-Head Self-Attention over the 355 tracker tokens
        self.norm_attn1 = nn.LayerNorm(token_dim)
        self.trsa_mha = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm_attn2 = nn.LayerNorm(token_dim)
        self.trsa_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.trsa_head = nn.Linear(token_dim, 1)

        # 4. 2nd-Order Polynomial NPMI Graph Diffusion Layer
        self.diffusion = PolynomialNPMIDiffusion(
            A_npmi, A_npmi_2, num_trackers=num_trackers
        )

        # 5. Multi-Stream Low-Rank Cross-Tracker Bypasses
        self.direct_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )
        self.in_neighbor_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )
        self.out_neighbor_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )

        self.prior_weight = nn.Parameter(torch.ones(num_trackers) * 3.5)
        self.prior_bias = nn.Parameter(torch.zeros(num_trackers))

        # 6. Context-Conditioned Dynamic Bypass Stream Gating Generator
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_trackers * 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        x_neighbor: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack directional neighbor channels
        if x_neighbor.dim() == 3:
            x_in = x_neighbor[:, :, 0]
            x_out = x_neighbor[:, :, 1]
        elif x_neighbor.dim() == 2 and x_neighbor.shape[1] == self.num_trackers * 2:
            x_in = x_neighbor[:, : self.num_trackers]
            x_out = x_neighbor[:, self.num_trackers :]
        else:
            x_in = x_neighbor
            x_out = x_neighbor

        prior_clamped = torch.clamp(x_prior, 1e-4, 1.0 - 1e-4)
        prior_logits = torch.log(prior_clamped / (1.0 - prior_clamped))

        # 1. Encode tracker graph signals into backbone stem alongside x_dense
        tracker_signals = torch.cat([x_direct, x_in, x_out, prior_logits], dim=-1)
        tracker_ctx = self.tracker_context(tracker_signals)
        h_stem = self.stem(torch.cat([x_dense, tracker_ctx], dim=-1))

        # 2. Cross-Stage Partial Dense-Residual Forward Flow
        h1 = self.stage1(h_stem)
        h1_trans = self.trans1(torch.cat([h_stem, h1], dim=-1))
        h2 = self.stage2(h1_trans)
        h_fused = self.stage_fuse(torch.cat([h_stem, h1, h2], dim=-1))
        latent_scores = self.latent_head(h_fused)

        # 3. Domain-Conditioned Tracker Relational Self-Attention (TRSA)
        tracker_feat = torch.stack([x_direct, x_in, x_out, prior_logits], dim=-1)
        signal_tokens = self.tracker_signal_proj(tracker_feat)  # (B, 355, 64)
        base_tokens = self.tracker_tax_proj(self.tracker_taxonomy) + self.tracker_base_embed  # (355, 64)
        domain_ctx = self.domain_ctx_proj(h_fused).unsqueeze(1)  # (B, 1, 64)

        tokens = base_tokens.unsqueeze(0) + signal_tokens + domain_ctx  # (B, 355, 64)
        norm_tokens = self.norm_attn1(tokens)
        attn_out, _ = self.trsa_mha(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.trsa_ffn(self.norm_attn2(tokens))
        trsa_scores = self.trsa_head(tokens).squeeze(-1)  # (B, 355)

        # 4. Multi-Channel Directional Low-Rank Bypasses
        direct_scores = self.direct_bypass(x_direct)
        in_scores = self.in_neighbor_bypass(x_in)
        out_scores = self.out_neighbor_bypass(x_out)
        prior_scores = prior_logits * self.prior_weight + self.prior_bias

        # 5. Dynamic Stream Gating
        gates = self.gate_net(h_fused)
        gate_direct = gates[:, : self.num_trackers]
        gate_in = gates[:, self.num_trackers : self.num_trackers * 2]
        gate_out = gates[:, self.num_trackers * 2 : self.num_trackers * 3]
        gate_prior = gates[:, self.num_trackers * 3 :]

        fused_logits = (
            latent_scores
            + trsa_scores
            + gate_direct * direct_scores
            + gate_in * in_scores
            + gate_out * out_scores
            + gate_prior * prior_scores
        )
        logits = self.diffusion(fused_logits)
        return logits


# Preserve interface compatibility with baseline GatedTrackerResNet class name
GatedTrackerResNet = TrackerRelationalSelfAttentionNet


class DenseTrackerHighwayNet(nn.Module):
    """Model 2: Dense Highway Network with cascading feature reuse, 2nd-order polynomial

    NPMI Graph Diffusion, and Multi-Channel Directional Low-Rank Cross-Tracker Bypasses.
    """

    def __init__(
        self,
        dense_dim: int,
        A_npmi: torch.Tensor = None,
        A_npmi_2: torch.Tensor = None,
        num_trackers: int = 355,
        hidden_dim: int = 320,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_trackers = num_trackers

        # Tracker Context Projection (dim=64) into Stem Alongside x_dense
        self.tracker_context = nn.Linear(num_trackers * 4, 64)

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(dense_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Cascading Dense Linear layers with SiLU
        self.dense1_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dense1_norm = nn.LayerNorm(hidden_dim)
        self.dense1_act = nn.SiLU()
        self.dense1_drop = nn.Dropout(dropout)

        self.dense2_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dense2_norm = nn.LayerNorm(hidden_dim)
        self.dense2_act = nn.SiLU()
        self.dense2_drop = nn.Dropout(dropout)

        self.dense3_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dense3_norm = nn.LayerNorm(hidden_dim)
        self.dense3_act = nn.SiLU()
        self.dense3_drop = nn.Dropout(dropout)

        # Multi-layer Highway Aggregation
        total_dim = hidden_dim * 4
        self.highway_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.out_head = nn.Linear(hidden_dim, num_trackers)

        # 2nd-Order Polynomial NPMI Graph Diffusion Layer
        self.diffusion = PolynomialNPMIDiffusion(
            A_npmi, A_npmi_2, num_trackers=num_trackers
        )

        # Direct, In-Neighbor, Out-Neighbor collaborative Low-Rank Bypasses and Prior bypass
        self.direct_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )
        self.in_neighbor_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )
        self.out_neighbor_bypass = LowRankCrossTrackerLinear(
            num_trackers=num_trackers, rank=32, init_diag=2.5
        )

        self.bypass_prior_w = nn.Parameter(torch.ones(num_trackers) * 3.5)
        self.bypass_prior_b = nn.Parameter(torch.zeros(num_trackers))

        # Dynamic highway bypass gates for the 4 bypass streams
        self.bypass_gate = nn.Sequential(
            nn.Linear(hidden_dim, num_trackers * 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        x_neighbor: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> torch.Tensor:
        # Unpack directional neighbor channels
        if x_neighbor.dim() == 3:
            x_in = x_neighbor[:, :, 0]
            x_out = x_neighbor[:, :, 1]
        elif x_neighbor.dim() == 2 and x_neighbor.shape[1] == self.num_trackers * 2:
            x_in = x_neighbor[:, : self.num_trackers]
            x_out = x_neighbor[:, self.num_trackers :]
        else:
            x_in = x_neighbor
            x_out = x_neighbor

        prior_clamped = torch.clamp(x_prior, 1e-4, 1.0 - 1e-4)
        prior_logits = torch.log(prior_clamped / (1.0 - prior_clamped))

        # Encode tracker graph signals into backbone stem alongside x_dense
        tracker_signals = torch.cat([x_direct, x_in, x_out, prior_logits], dim=-1)
        tracker_ctx = self.tracker_context(tracker_signals)
        h0 = self.stem(torch.cat([x_dense, tracker_ctx], dim=-1))

        h1 = self.dense1_drop(self.dense1_act(self.dense1_norm(self.dense1_linear(h0))))
        h2 = self.dense2_drop(
            self.dense2_act(
                self.dense2_norm(self.dense2_linear(torch.cat([h0, h1], dim=-1)))
            )
        )
        h3 = self.dense3_drop(
            self.dense3_act(
                self.dense3_norm(self.dense3_linear(torch.cat([h0, h1, h2], dim=-1)))
            )
        )

        h_fused = self.highway_proj(torch.cat([h0, h1, h2, h3], dim=-1))
        latent_logits = self.out_head(h_fused)

        direct_scores = self.direct_bypass(x_direct)
        in_scores = self.in_neighbor_bypass(x_in)
        out_scores = self.out_neighbor_bypass(x_out)
        prior_scores = prior_logits * self.bypass_prior_w + self.bypass_prior_b

        gates = self.bypass_gate(h_fused)
        gate_direct = gates[:, : self.num_trackers]
        gate_in = gates[:, self.num_trackers : self.num_trackers * 2]
        gate_out = gates[:, self.num_trackers * 2 : self.num_trackers * 3]
        gate_prior = gates[:, self.num_trackers * 3 :]

        fused_logits = (
            latent_logits
            + gate_direct * direct_scores
            + gate_in * in_scores
            + gate_out * out_scores
            + gate_prior * prior_scores
        )
        logits = self.diffusion(fused_logits)
        return logits


class AsymmetricRecall10Loss(nn.Module):
    """Combines Asymmetric Focal Loss with Top-10 Boundary Margin Ranking Loss

    calibrated strictly against the 10th hardest negative candidate logit.
    """

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        rank_weight: float = 2.0,
        margin: float = 1.0,
        top_k: int = 10,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.rank_weight = rank_weight
        self.margin = margin
        self.top_k = top_k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        # 1. Asymmetric Focal Multi-Label Component
        pos_loss = targets * torch.log(probs.clamp(min=1e-6))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - probs) ** self.gamma_pos)

        neg_probs = (probs - self.clip).clamp(min=0.0)
        neg_loss = (
            (1.0 - targets)
            * (neg_probs**self.gamma_neg)
            * torch.log((1.0 - neg_probs).clamp(min=1e-6))
        )

        asl_loss = -(pos_loss + neg_loss).sum(dim=-1).mean()

        # 2. Top-10 Boundary Margin Ranking Component calibrated strictly against the 10th hardest negative
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e4))
        s_10 = torch.topk(neg_logits, k=self.top_k, dim=-1)[0][:, -1:]

        diff = s_10 - logits + self.margin
        violations = F.softplus(diff)

        pos_violations = violations * targets
        num_pos = targets.sum(dim=-1).clamp(min=1.0)
        rank_loss = (pos_violations.sum(dim=-1) / num_pos).mean()

        total_loss = asl_loss + self.rank_weight * rank_loss
        return total_loss


def compute_numpy_recall_at_10(scores: np.ndarray, targets: np.ndarray) -> float:
    """Computes exact Recall@10 matching official competition evaluation."""
    top10_indices = np.argsort(-scores, axis=1)[:, :10]
    num_samples = len(targets)
    recalls = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        true_trackers = np.where(targets[i] == 1)[0]
        if len(true_trackers) == 0:
            recalls[i] = 1.0
        else:
            hits = np.isin(top10_indices[i], true_trackers).sum()
            recalls[i] = hits / len(true_trackers)

    return float(np.mean(recalls))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

criterion = AsymmetricRecall10Loss(
    gamma_neg=2.0,
    gamma_pos=0.0,
    clip=0.05,
    rank_weight=2.0,
    margin=1.0,
    top_k=10,
).to(device)

# =========================================================================
# STEP 3: TRAINING, VALIDATION & INFERENCE PIPELINE
# =========================================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_dense_train),
    torch.from_numpy(tracker_direct_train),
    torch.from_numpy(tracker_neighbor_train),
    torch.from_numpy(tld_prior_train),
    torch.from_numpy(Y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_dense_val),
    torch.from_numpy(tracker_direct_val),
    torch.from_numpy(tracker_neighbor_val),
    torch.from_numpy(tld_prior_val),
    torch.from_numpy(Y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_dense_test),
    torch.from_numpy(tracker_direct_test),
    torch.from_numpy(tracker_neighbor_test),
    torch.from_numpy(tld_prior_test),
)

batch_size = 2048
eval_batch_size = 4096

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)


def evaluate(model, loader, criterion, device):
    """Evaluates validation loss and exact official Recall@10 metric."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for b_dense, b_direct, b_neighbor, b_prior, b_y in loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_neighbor = b_neighbor.to(device, non_blocking=True)
            b_prior = b_prior.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            logits = model(b_dense, b_direct, b_neighbor, b_prior)
            loss = criterion(logits, b_y)
            total_loss += loss.item() * len(b_dense)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(b_y.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(all_targets)
    val_recall = compute_numpy_recall_at_10(all_logits, all_targets)
    return avg_loss, val_recall


def predict_probabilities(model, loader, device):
    """Generates sigmoid prediction probabilities."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            b_dense = batch[0].to(device, non_blocking=True)
            b_direct = batch[1].to(device, non_blocking=True)
            b_neighbor = batch[2].to(device, non_blocking=True)
            b_prior = batch[3].to(device, non_blocking=True)
            logits = model(b_dense, b_direct, b_neighbor, b_prior)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: str = "./working/best_model.pt",
) -> nn.Module:
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=1
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 1), eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[1],
    )
    best_val_recall = -1.0

    print(f"\n--- Training {model_name} for {epochs} epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        if time.time() - global_start_time > MAX_TOTAL_RUNTIME:
            print(f"Time guard triggered ({time.time() - global_start_time:.1f}s). Early stopping.")
            break

        model.train()
        running_loss = 0.0
        num_samples = 0

        for b_dense, b_direct, b_neighbor, b_prior, b_y in train_loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_neighbor = b_neighbor.to(device, non_blocking=True)
            b_prior = b_prior.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(b_dense, b_direct, b_neighbor, b_prior)
            loss = criterion(logits, b_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * len(b_dense)
            num_samples += len(b_dense)

        scheduler.step()
        epoch_train_loss = running_loss / num_samples
        val_loss, val_recall = evaluate(model, val_loader, criterion, device)

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs:02d} (lr: {current_lr:.6f}) | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_val_recall:.5f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    return model


epochs = 12

# 1. Train Model 1 (GatedTrackerResNet with Dual-Tower Alignment and Polynomial NPMI Diffusion)
model1_path = "./working/best_gated_tracker_resnet.pt"
model1 = GatedTrackerResNet(
    dense_dim=dense_feature_dim,
    A_npmi=A_npmi_tensor,
    A_npmi_2=A_npmi_2_tensor,
    tracker_taxonomy=tracker_taxonomy_tensor,
    num_trackers=num_trackers,
    hidden_dim=384,
    embed_dim=256,
    dropout=0.2,
).to(device)

model1 = train_single_model(
    model=model1,
    model_name="Model 1 (GatedTrackerResNet)",
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    device=device,
    epochs=epochs,
    lr=1e-3,
    weight_decay=1e-4,
    save_path=model1_path,
)

# 2. Train Model 2 (DenseTrackerHighwayNet with Polynomial NPMI Diffusion)
model2_path = "./working/best_dense_highway_net.pt"
model2 = DenseTrackerHighwayNet(
    dense_dim=dense_feature_dim,
    A_npmi=A_npmi_tensor,
    A_npmi_2=A_npmi_2_tensor,
    num_trackers=num_trackers,
    hidden_dim=320,
    dropout=0.2,
).to(device)

model2 = train_single_model(
    model=model2,
    model_name="Model 2 (DenseTrackerHighwayNet)",
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    device=device,
    epochs=epochs,
    lr=1e-3,
    weight_decay=1e-4,
    save_path=model2_path,
)

# =========================================================================
# STEP 4: VALIDATION BLEND OPTIMIZATION & ENSEMBLED INFERENCE
# =========================================================================

print("\n--- Validation Probability Blend Optimization ---")
val_probs1 = predict_probabilities(model1, val_loader, device)
val_probs2 = predict_probabilities(model2, val_loader, device)

score1 = compute_numpy_recall_at_10(val_probs1, Y_val)
score2 = compute_numpy_recall_at_10(val_probs2, Y_val)
print(f"Model 1 Standalone Val Recall@10: {score1:.5f}")
print(f"Model 2 Standalone Val Recall@10: {score2:.5f}")

best_alpha = 0.5
best_ensemble_score = max(score1, score2)
alpha_candidates = np.linspace(0.1, 0.9, 17)

for alpha in alpha_candidates:
    blend_probs = alpha * val_probs1 + (1.0 - alpha) * val_probs2
    blend_score = compute_numpy_recall_at_10(blend_probs, Y_val)
    print(f"Ensemble weight alpha={alpha:.2f} -> Val Recall@10: {blend_score:.5f}")
    if blend_score > best_ensemble_score:
        best_ensemble_score = blend_score
        best_alpha = alpha

print(f"Optimal Ensemble Weight (alpha Model 1): {best_alpha:.2f}")
print(f"Optimized Ensembled Val Recall@10: {best_ensemble_score:.5f}")

# Generate Ensembled Predictions on Test Set
print("\n--- Generating Ensembled Test Predictions ---")
test_probs1 = predict_probabilities(model1, test_loader, device)
test_probs2 = predict_probabilities(model2, test_loader, device)

final_test_probs = best_alpha * test_probs1 + (1.0 - best_alpha) * test_probs2
top10_test_indices = np.argsort(-final_test_probs, axis=1)[:, :10]

test_domain_ids_repeated = np.repeat(test_domain_ids, 10)
pred_tracking_domain_ids = tracker_id_to_domain_id[top10_test_indices.ravel()]

submission_df = pd.DataFrame(
    {
        "domain_id": test_domain_ids_repeated,
        "tracking_domain_id": pred_tracking_domain_ids,
    }
)

assert len(submission_df) == len(test_domain_ids) * 10
assert submission_df["domain_id"].nunique() == len(test_domain_ids)
assert not submission_df.isnull().any().any()

submission_csv_path = "./submission/submission.csv"
submission_tsv_path = "./submission/submission.tsv"

submission_df.to_csv(submission_csv_path, sep="\t", index=False)
submission_df.to_csv(submission_tsv_path, sep="\t", index=False)

print(f"Final Validation Score: {best_ensemble_score}")

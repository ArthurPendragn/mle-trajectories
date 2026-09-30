import gc
import json
import math
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Set deterministic random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =========================================================================
# STEP 1: DATA LOADING, PREPROCESSING & FEATURE ENGINEERING
# =========================================================================

print("--- Step 1: Loading Tracker Metadata and Target Domains ---")
trackers_df = pd.read_csv("./input/trackers.tsv", sep="\t")
num_trackers = len(trackers_df)
print(f"Loaded {num_trackers} tracker definitions.")

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

# Construct structured tracker metadata matrix encoding company, country, category, and brand
sorted_trackers_by_id = trackers_df.sort_values("tracker_id").reset_index(drop=True)
meta_cols = ["company", "country", "category", "brand"]
tracker_meta_df = pd.get_dummies(sorted_trackers_by_id[meta_cols].fillna("unknown"), drop_first=False)
tracker_meta_raw = tracker_meta_df.values.astype(np.float32)
tracker_meta_norms = np.linalg.norm(tracker_meta_raw, axis=1, keepdims=True)
tracker_meta_normalized = tracker_meta_raw / np.maximum(tracker_meta_norms, 1e-6)
print(f"Constructed tracker metadata matrix: shape {tracker_meta_normalized.shape}")

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

unique_train_domains = train_graph_df["domain_id"].unique().astype(np.int64)
# Exclude any test domain from training pool (leak-free guarantee)
test_domain_set = set(test_domain_ids)
train_pool = np.array(
    [d for d in unique_train_domains if d not in test_domain_set],
    dtype=np.int64,
)

# Deterministic reproducible shuffle
rng = np.random.RandomState(42)
rng.shuffle(train_pool)

if len(train_pool) >= 275000:
    train_domain_ids = train_pool[:250000]
    val_domain_ids = train_pool[250000:275000]
else:
    split_point = int(len(train_pool) * 0.9)
    train_domain_ids = train_pool[:split_point]
    val_domain_ids = train_pool[split_point:]

num_train = len(train_domain_ids)
num_val = len(val_domain_ids)
print(f"Dataset split: Train={num_train}, Val={num_val}, Test={num_test} domains.")

# Construct Ground Truth Multi-Label Matrices Y_train and Y_val
print("Building multi-label target matrices...")
Y_train = np.zeros((num_train, num_trackers), dtype=np.uint8)
Y_val = np.zeros((num_val, num_trackers), dtype=np.uint8)

train_map = {did: idx for idx, did in enumerate(train_domain_ids)}
val_map = {did: idx for idx, did in enumerate(val_domain_ids)}

graph_domains = train_graph_df["domain_id"].values
graph_trackers = train_graph_df["tracker_id"].values

train_mask = np.isin(graph_domains, train_domain_ids)
t_doms = graph_domains[train_mask]
t_trks = graph_trackers[train_mask]
train_row_idx = np.fromiter(
    (train_map[d] for d in t_doms), dtype=np.int32, count=len(t_doms)
)
Y_train[train_row_idx, t_trks] = 1

val_mask = np.isin(graph_domains, val_domain_ids)
v_doms = graph_domains[val_mask]
v_trks = graph_trackers[val_mask]
val_row_idx = np.fromiter(
    (val_map[d] for d in v_doms), dtype=np.int32, count=len(v_doms)
)
Y_val[val_row_idx, v_trks] = 1

del (
    train_graph_df,
    train_mask,
    val_mask,
    t_doms,
    t_trks,
    v_doms,
    v_trks,
    train_row_idx,
    val_row_idx,
)
gc.collect()

print(
    f"Targets constructed: Y_train positive rate: {Y_train.mean():.4f}, Y_val positive rate: {Y_val.mean():.4f}"
)

# Precompute empirical tracker co-occurrence transition matrix P(T_j | T_i) strictly from Y_train
print("Precomputing empirical tracker co-occurrence transition matrix strictly from Y_train...")
tracker_train_counts = Y_train.sum(axis=0).astype(np.float32)
cooccur_counts = (Y_train.T @ Y_train).astype(np.float32)
p_cooccur_matrix = cooccur_counts / np.maximum(tracker_train_counts[:, None], 1.0)
np.fill_diagonal(p_cooccur_matrix, 0.0)

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
        return "unknown", "unknown", "unknown", 0, 0, 0, 0
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

    apex = f"{sld}.{tld}" if sld else tld
    digits = sum(c.isdigit() for c in h)
    hyphens = h.count("-")
    vowels = sum(c in "aeiou" for c in h)
    return tld, sld, apex, sub_count, length, digits, hyphens, vowels


def compute_shannon_entropy(s):
    if not s:
        return 0.0
    char_counts = {}
    for c in s:
        char_counts[c] = char_counts.get(c, 0) + 1
    total = len(s)
    ent = 0.0
    for cnt in char_counts.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return float(ent)


role_tokens = [
    "shop",
    "store",
    "blog",
    "news",
    "forum",
    "app",
    "cdn",
    "media",
    "press",
    "mail",
    "dev",
    "tech",
    "video",
    "tv",
    "game",
    "org",
]

parsed_tld = []
parsed_sld = []
parsed_apex = []
parsed_sub_count = np.zeros(num_total, dtype=np.float32)
parsed_len = np.zeros(num_total, dtype=np.float32)
parsed_sld_len = np.zeros(num_total, dtype=np.float32)
parsed_digits = np.zeros(num_total, dtype=np.float32)
parsed_hyphens = np.zeros(num_total, dtype=np.float32)
parsed_vowels = np.zeros(num_total, dtype=np.float32)
parsed_consonants = np.zeros(num_total, dtype=np.float32)
parsed_entropy = np.zeros(num_total, dtype=np.float32)
num_dots = np.zeros(num_total, dtype=np.float32)
is_root_domain = np.zeros(num_total, dtype=np.float32)
sld_hyphens = np.zeros(num_total, dtype=np.float32)
role_features = np.zeros((num_total, len(role_tokens)), dtype=np.float32)
hostname_list = []

for idx, did in enumerate(all_domain_ids):
    h = domain_to_hostname.get(did, "")
    hostname_list.append(h)
    tld, sld, apex, subs, l, d, hyp, v = parse_hostname(h)
    parsed_tld.append(tld)
    parsed_sld.append(sld)
    parsed_apex.append(apex)
    parsed_sub_count[idx] = subs
    parsed_len[idx] = l
    parsed_sld_len[idx] = len(sld)
    parsed_digits[idx] = d
    parsed_hyphens[idx] = hyp
    parsed_vowels[idx] = v
    parsed_consonants[idx] = sum(c.isalpha() and c not in "aeiou" for c in h)
    parsed_entropy[idx] = compute_shannon_entropy(h)
    num_dots[idx] = h.count(".")
    is_root_domain[idx] = 1.0 if subs == 0 else 0.0
    sld_hyphens[idx] = sld.count("-")
    for r_idx, r_tok in enumerate(role_tokens):
        if r_tok in h:
            role_features[idx, r_idx] = 1.0

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

print("Parsing url-classification.csv...")
url_df = pd.read_csv("./input/url-classification.csv", usecols=["url", "category"])
cat_list = sorted(url_df["category"].dropna().unique().tolist())
cat_to_idx = {cat: i for i, cat in enumerate(cat_list)}
num_cats = len(cat_list)

host_to_cat = {}
for u, c in zip(url_df["url"], url_df["category"]):
    if not isinstance(u, str) or not isinstance(c, str):
        continue
    if "://" in u:
        u = u.split("://", 1)[1]
    h = u.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].lower()
    if h.startswith("www."):
        h = h[4:]
    if h and h not in host_to_cat:
        host_to_cat[h] = cat_to_idx.get(c, -1)

del url_df
gc.collect()

url_cat_features = np.zeros((num_total, num_cats + 1), dtype=np.float32)
for idx, did in enumerate(all_domain_ids):
    h = domain_to_hostname.get(did, "").lower()
    if h.startswith("www."):
        h = h[4:]
    c_idx = host_to_cat.get(h, -1)
    if c_idx >= 0:
        url_cat_features[idx, c_idx] = 1.0
        url_cat_features[idx, -1] = 1.0

del host_to_cat
gc.collect()
print(f"URL Categories matched for {url_cat_features[:, -1].sum():.0f} domains.")

print(
    "--- Step 5: Streaming link-graph.parquet for Graph Features, Direct Tracker Links & Tracker Homophily ---"
)
# Pre-sort train_domain_ids for binary search during streaming
train_sort_idx = np.argsort(train_domain_ids)
sorted_train_domain_ids = train_domain_ids[train_sort_idx]

in_degrees_sorted = np.zeros(num_total, dtype=np.float32)
out_degrees_sorted = np.zeros(num_total, dtype=np.float32)
tracker_direct_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
in_tracker_sum_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
out_tracker_sum_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)

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
        tracker_direct_sorted[match_src_indices, match_trk_indices] += 1.0

    # In-neighbor tracker homophily: edge src -> dst, src in train_domain_ids, dst in all_domains, src != dst
    src_tr_pos = np.searchsorted(sorted_train_domain_ids, src)
    valid_src_tr = src_tr_pos < num_train
    src_tr_matched = np.zeros(len(src), dtype=bool)
    src_tr_matched[valid_src_tr] = (
        sorted_train_domain_ids[src_tr_pos[valid_src_tr]] == src[valid_src_tr]
    )

    in_homo_mask = src_tr_matched & dst_matched & (src != dst)
    if np.any(in_homo_mask):
        m_dst_all = dst_pos[in_homo_mask]
        m_src_tr = train_sort_idx[src_tr_pos[in_homo_mask]]
        np.add.at(
            in_tracker_sum_sorted,
            m_dst_all,
            Y_train[m_src_tr].astype(np.float32),
        )

    # Out-neighbor tracker homophily: edge src -> dst, dst in train_domain_ids, src in all_domains, src != dst
    dst_tr_pos = np.searchsorted(sorted_train_domain_ids, dst)
    valid_dst_tr = dst_tr_pos < num_train
    dst_tr_matched = np.zeros(len(dst), dtype=bool)
    dst_tr_matched[valid_dst_tr] = (
        sorted_train_domain_ids[dst_tr_pos[valid_dst_tr]] == dst[valid_dst_tr]
    )

    out_homo_mask = dst_tr_matched & src_matched & (src != dst)
    if np.any(out_homo_mask):
        m_src_all = src_pos[out_homo_mask]
        m_dst_tr = train_sort_idx[dst_tr_pos[out_homo_mask]]
        np.add.at(
            out_tracker_sum_sorted,
            m_src_all,
            Y_train[m_dst_tr].astype(np.float32),
        )

    batch_count += 1

print(f"Link graph streaming complete across {batch_count} batches.")

inv_sorted_indices = np.empty_like(sorted_all_indices)
inv_sorted_indices[sorted_all_indices] = np.arange(num_total)

in_degrees = in_degrees_sorted[inv_sorted_indices]
out_degrees = out_degrees_sorted[inv_sorted_indices]

# Degree-normalized neighbor tracker homophily vectors
in_tracker_sum = in_tracker_sum_sorted[inv_sorted_indices]
out_tracker_sum = out_tracker_sum_sorted[inv_sorted_indices]

in_priors_all = in_tracker_sum / np.maximum(in_degrees[:, None], 1.0)
out_priors_all = out_tracker_sum / np.maximum(out_degrees[:, None], 1.0)

del in_tracker_sum_sorted, out_tracker_sum_sorted, in_tracker_sum, out_tracker_sum
gc.collect()

# Compute tracker out-link intensity metrics from stream
raw_tracker_direct = tracker_direct_sorted[inv_sorted_indices]
tracker_direct_all = np.log1p(raw_tracker_direct)
total_tracker_links = raw_tracker_direct.sum(axis=1)
log1p_tracker_links = np.log1p(total_tracker_links)
tracker_link_ratio = total_tracker_links / np.maximum(out_degrees, 1.0)
unique_tracker_counts = (raw_tracker_direct > 0).sum(axis=1).astype(np.float32)
unique_tracker_ratio = unique_tracker_counts / float(num_trackers)

del in_degrees_sorted, out_degrees_sorted, tracker_direct_sorted, raw_tracker_direct
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
    max_features=8000,
    sublinear_tf=True,
)
svd = TruncatedSVD(n_components=64, random_state=42)

print("Fitting TF-IDF and SVD on training hostnames strictly...")
train_tfidf = tfidf.fit_transform(train_hostnames)
svd.fit(train_tfidf)
del train_tfidf
gc.collect()

all_tfidf = tfidf.transform(hostname_list)
all_svd_features = svd.transform(all_tfidf).astype(np.float32)
del all_tfidf, tfidf, svd
gc.collect()

print(
    "--- Step 7: Bayesian Empirical Apex, SLD & TLD Tracker Priors (Fitted Strictly on Train) ---"
)
global_tracker_prior = Y_train.mean(axis=0).astype(np.float32)

train_tlds = parsed_tld[:num_train]
train_slds = parsed_sld[:num_train]
train_apexes = parsed_apex[:num_train]

tld_counts_train = {}
tld_tracker_sums_train = {}
sld_counts_train = {}
sld_tracker_sums_train = {}
apex_counts_train = {}
apex_tracker_sums_train = {}

for i in range(num_train):
    t = train_tlds[i]
    s = train_slds[i]
    a = train_apexes[i]
    y_i = Y_train[i].astype(np.float32)

    tld_counts_train[t] = tld_counts_train.get(t, 0) + 1
    if t not in tld_tracker_sums_train:
        tld_tracker_sums_train[t] = y_i.copy()
    else:
        tld_tracker_sums_train[t] += y_i

    sld_counts_train[s] = sld_counts_train.get(s, 0) + 1
    if s not in sld_tracker_sums_train:
        sld_tracker_sums_train[s] = y_i.copy()
    else:
        sld_tracker_sums_train[s] += y_i

    apex_counts_train[a] = apex_counts_train.get(a, 0) + 1
    if a not in apex_tracker_sums_train:
        apex_tracker_sums_train[a] = y_i.copy()
    else:
        apex_tracker_sums_train[a] += y_i

# 1. TLD Bayesian Priors
alpha_tld = 25.0
tld_bayesian_priors = {}
for t, n in tld_counts_train.items():
    smoothed = (tld_tracker_sums_train[t] + alpha_tld * global_tracker_prior) / (n + alpha_tld)
    tld_bayesian_priors[t] = smoothed.astype(np.float32)

tld_priors_all = np.zeros((num_total, num_trackers), dtype=np.float32)
for idx, tld in enumerate(parsed_tld):
    tld_priors_all[idx] = tld_bayesian_priors.get(tld, global_tracker_prior)

# 2. SLD Bayesian Priors (smoothed with global tracker prior)
beta_sld = 10.0
sld_bayesian_priors = {}
for s, n in sld_counts_train.items():
    smoothed = (sld_tracker_sums_train[s] + beta_sld * global_tracker_prior) / (n + beta_sld)
    sld_bayesian_priors[s] = smoothed.astype(np.float32)

# 3. Apex Domain Bayesian Priors (smoothed with SLD prior)
alpha_apex = 2.0
apex_bayesian_priors = {}
for a, n in apex_counts_train.items():
    s = a.split(".", 1)[0] if "." in a else a
    parent_sld_prior = sld_bayesian_priors.get(s, global_tracker_prior)
    smoothed = (apex_tracker_sums_train[a] + alpha_apex * parent_sld_prior) / (n + alpha_apex)
    apex_bayesian_priors[a] = smoothed.astype(np.float32)

apex_priors_all = np.zeros((num_total, num_trackers), dtype=np.float32)

# Leave-one-out Bayesian Apex Priors for training domains to eliminate target leakage
apex_priors_train = np.zeros((num_train, num_trackers), dtype=np.float32)
for i in range(num_train):
    a = train_apexes[i]
    s = train_slds[i]
    y_i = Y_train[i].astype(np.float32)
    n_a = apex_counts_train[a] - 1
    if n_a > 0:
        sum_a = apex_tracker_sums_train[a] - y_i
        n_s = sld_counts_train[s] - 1
        parent_prior = (
            (sld_tracker_sums_train[s] - y_i + beta_sld * global_tracker_prior) / (n_s + beta_sld)
            if n_s > 0
            else global_tracker_prior
        )
        apex_priors_train[i] = (sum_a + alpha_apex * parent_prior) / (n_a + alpha_apex)
    else:
        n_s = sld_counts_train[s] - 1
        if n_s > 0:
            apex_priors_train[i] = (
                sld_tracker_sums_train[s] - y_i + beta_sld * global_tracker_prior
            ) / (n_s + beta_sld)
        else:
            apex_priors_train[i] = global_tracker_prior

# Val and Test domain apex priors
for idx in range(num_train, num_total):
    a = parsed_apex[idx]
    s = parsed_sld[idx]
    if a in apex_bayesian_priors:
        apex_priors_all[idx] = apex_bayesian_priors[a]
    elif s in sld_bayesian_priors:
        apex_priors_all[idx] = sld_bayesian_priors[s]
    else:
        apex_priors_all[idx] = global_tracker_prior

apex_priors_val = apex_priors_all[num_train : num_train + num_val]
apex_priors_test = apex_priors_all[num_train + num_val :]

# Slices for neighbor tracker priors
in_priors_train = in_priors_all[:num_train]
in_priors_val = in_priors_all[num_train : num_train + num_val]
in_priors_test = in_priors_all[num_train + num_val :]

out_priors_train = out_priors_all[:num_train]
out_priors_val = out_priors_all[num_train : num_train + num_val]
out_priors_test = out_priors_all[num_train + num_val :]

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
consonant_ratio = parsed_consonants / np.maximum(parsed_vowels, 1.0)

# Continuous features to standardize
continuous_feature_blocks = [
    in_degrees.reshape(-1, 1),
    out_degrees.reshape(-1, 1),
    log1p_in.reshape(-1, 1),
    log1p_out.reshape(-1, 1),
    degree_ratio.reshape(-1, 1),
    log1p_total.reshape(-1, 1),
    log1p_tracker_links.reshape(-1, 1),
    tracker_link_ratio.reshape(-1, 1),
    unique_tracker_counts.reshape(-1, 1),
    unique_tracker_ratio.reshape(-1, 1),
    parsed_sub_count.reshape(-1, 1),
    parsed_len.reshape(-1, 1),
    parsed_sld_len.reshape(-1, 1),
    parsed_digits.reshape(-1, 1),
    digit_ratio.reshape(-1, 1),
    parsed_hyphens.reshape(-1, 1),
    parsed_vowels.reshape(-1, 1),
    vowel_ratio.reshape(-1, 1),
    parsed_consonants.reshape(-1, 1),
    consonant_ratio.reshape(-1, 1),
    parsed_entropy.reshape(-1, 1),
    num_dots.reshape(-1, 1),
    sld_hyphens.reshape(-1, 1),
    fop_scores.reshape(-1, 1),
    tld_freq_feature,
]

# Sparse / binary / uncentered feature blocks
uncentered_feature_blocks = [
    is_leaf.reshape(-1, 1),
    is_sink.reshape(-1, 1),
    is_root_domain.reshape(-1, 1),
    fop_matched.reshape(-1, 1),
    role_features,
    url_cat_features,
    tld_one_hot,
    all_svd_features,
]

X_cont_all = np.hstack(continuous_feature_blocks).astype(np.float32)
np.nan_to_num(X_cont_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_uncentered_all = np.hstack(uncentered_feature_blocks).astype(np.float32)
np.nan_to_num(X_uncentered_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_cont_train = X_cont_all[:num_train]
X_cont_val = X_cont_all[num_train : num_train + num_val]
X_cont_test = X_cont_all[num_train + num_val :]

print("Fitting StandardScaler strictly on continuous training features...")
scaler = StandardScaler()
X_cont_train = scaler.fit_transform(X_cont_train).astype(np.float32)
X_cont_val = scaler.transform(X_cont_val).astype(np.float32)
X_cont_test = scaler.transform(X_cont_test).astype(np.float32)

X_dense_train = np.hstack([X_cont_train, X_uncentered_all[:num_train]]).astype(np.float32)
X_dense_val = np.hstack([X_cont_val, X_uncentered_all[num_train : num_train + num_val]]).astype(np.float32)
X_dense_test = np.hstack([X_cont_test, X_uncentered_all[num_train + num_val :]]).astype(np.float32)

tracker_direct_train = tracker_direct_all[:num_train]
tracker_direct_val = tracker_direct_all[num_train : num_train + num_val]
tracker_direct_test = tracker_direct_all[num_train + num_val :]

tld_prior_train = tld_priors_all[:num_train]
tld_prior_val = tld_priors_all[num_train : num_train + num_val]
tld_prior_test = tld_priors_all[num_train + num_val :]

dense_feature_dim = X_dense_train.shape[1]
print(f"Features prepared: dense_feature_dim={dense_feature_dim}")

# =========================================================================
# STEP 2: MODEL ARCHITECTURE & RANKING OBJECTIVE
# =========================================================================


class ResidualDenseBlock(nn.Module):
    """Pre-activation residual block with LayerNorm, GELU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(self.act(self.linear1(self.norm1(x))))
        out = self.dropout(self.linear2(self.norm2(out)))
        return residual + out


class GatedTrackerFusionNet(nn.Module):
    """Graph-Neighborhood Gated Fusion Network equipped with dedicated channel-aligned

    logit bypasses for neighborhood, apex, and direct priors, projected tracker taxonomy
    representations, and empirical syndicate co-occurrence synergy.
    """

    def __init__(
        self,
        dense_dim: int,
        tracker_meta: np.ndarray = None,
        p_cooccur: np.ndarray = None,
        num_trackers: int = 355,
        hidden_dim: int = 384,
        embed_dim: int = 256,
        num_factors: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim
        self.num_factors = num_factors
        self.factor_dim = embed_dim // num_factors

        # 1. Latent Domain Feature Backbone
        self.stem = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualDenseBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualDenseBlock(hidden_dim, dropout=dropout)
        self.domain_proj = nn.Linear(hidden_dim, embed_dim)

        # 2. Tracker Taxonomy Projection & Learnable Query Embeddings
        if tracker_meta is not None:
            if isinstance(tracker_meta, np.ndarray):
                tracker_meta = torch.from_numpy(tracker_meta).float()
            self.register_buffer("tracker_meta", tracker_meta)
            tracker_meta_dim = tracker_meta.shape[1]
        else:
            self.register_buffer("tracker_meta", torch.eye(num_trackers))
            tracker_meta_dim = num_trackers

        self.tracker_meta_proj = nn.Sequential(
            nn.Linear(tracker_meta_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.tracker_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )
        self.tracker_bias = nn.Parameter(torch.zeros(num_trackers))
        self.factor_weights = nn.Parameter(torch.ones(num_factors))

        # 3. Channel-Aligned Dedicated Positive-Scaled Logit Bypasses
        self.direct_weight = nn.Parameter(torch.ones(num_trackers) * 2.0)
        self.in_prior_weight = nn.Parameter(torch.ones(num_trackers) * 2.5)
        self.out_prior_weight = nn.Parameter(torch.ones(num_trackers) * 2.5)
        self.apex_prior_weight = nn.Parameter(torch.ones(num_trackers) * 3.0)
        self.tld_prior_weight = nn.Parameter(torch.ones(num_trackers) * 1.5)

        # 4. Residual Context Projection
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_trackers),
        )

        # 5. Tracker Syndicate Co-Occurrence Synergy Layer
        if p_cooccur is not None:
            if isinstance(p_cooccur, np.ndarray):
                p_cooccur = torch.from_numpy(p_cooccur).float()
            self.register_buffer("p_cooccur", p_cooccur)
        else:
            self.register_buffer("p_cooccur", torch.zeros(num_trackers, num_trackers))

        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.synergy_layer = nn.Linear(num_trackers, num_trackers, bias=False)
        nn.init.zeros_(self.synergy_layer.weight)
        self.synergy_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        in_priors: torch.Tensor,
        out_priors: torch.Tensor,
        apex_priors: torch.Tensor,
        tld_priors: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.stem(x_dense)
        h = self.res1(h)
        h = self.res2(h)
        domain_emb = self.domain_proj(h)

        # Combine learnable tracker queries with projected metadata
        meta_emb = self.tracker_meta_proj(self.tracker_meta)
        full_tracker_emb = self.tracker_embeddings + meta_emb

        # Multi-factor bilinear domain-tracker matching
        domain_sub = domain_emb.view(-1, self.num_factors, self.factor_dim)
        tracker_sub = full_tracker_emb.view(
            self.num_trackers, self.num_factors, self.factor_dim
        )
        factor_scores = (
            torch.einsum("bfd,nfd->bnf", domain_sub, tracker_sub)
            / math.sqrt(self.factor_dim)
        )
        latent_scores = (
            torch.matmul(factor_scores, self.factor_weights) + self.tracker_bias
        )

        # Channel-aligned positive-scaled logit bypasses
        fused_scores = (
            latent_scores
            + self.context_proj(h)
            + F.softplus(self.direct_weight) * x_direct
            + F.softplus(self.in_prior_weight) * in_priors
            + F.softplus(self.out_prior_weight) * out_priors
            + F.softplus(self.apex_prior_weight) * apex_priors
        )
        if tld_priors is not None:
            fused_scores = fused_scores + F.softplus(self.tld_prior_weight) * tld_priors

        # Probability-space empirical co-occurrence injection & synergy layer
        probs = torch.sigmoid(fused_scores)
        cooccur_prior = torch.matmul(probs, self.p_cooccur)
        synergy_refined = (
            fused_scores
            + self.synergy_dropout(self.synergy_layer(probs))
            + self.alpha * cooccur_prior
        )
        return synergy_refined


class SmoothListwiseTopKLoss(nn.Module):
    """Smooth Listwise Top-K Hard-Negative Ranking Loss directly maximizing Recall@10

    by identifying top-10 hardest negative tracker logits per domain, computing
    temperature-scaled softplus pairwise violations against ground-truth positive
    trackers, normalized by positive count, and regularized with multi-label BCE.
    """

    def __init__(
        self,
        top_k: int = 10,
        margin: float = 1.0,
        temperature: float = 1.0,
        rank_weight: float = 2.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.margin = margin
        self.temperature = temperature
        self.rank_weight = rank_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        # Multi-label BCE regularization component
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Identify top-k negative tracker logits per domain
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e4))
        topk_neg_logits, _ = torch.topk(neg_logits, k=self.top_k, dim=-1)

        # Pairwise violation: topk_neg_logits [B, K] vs positive logits [B, C]
        diff = (
            topk_neg_logits.unsqueeze(1) - logits.unsqueeze(-1) + self.margin
        ) / self.temperature
        violations = self.temperature * F.softplus(diff)

        # Filter to only ground-truth positive trackers
        pos_mask = targets.unsqueeze(-1)
        masked_violations = (violations * pos_mask).sum(dim=-1) / float(self.top_k)

        # Normalize by positive count per domain
        num_pos = targets.sum(dim=-1).clamp(min=1.0)
        rank_loss = (masked_violations.sum(dim=-1) / num_pos).mean()

        total_loss = self.bce_weight * bce_loss + self.rank_weight * rank_loss
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
print(f"Instantiating model on compute device: {device}")

model = GatedTrackerFusionNet(
    dense_dim=dense_feature_dim,
    tracker_meta=tracker_meta_normalized,
    p_cooccur=p_cooccur_matrix,
    num_trackers=num_trackers,
    hidden_dim=384,
    embed_dim=256,
    num_factors=4,
    dropout=0.2,
).to(device)

criterion = SmoothListwiseTopKLoss(
    top_k=10,
    margin=1.0,
    temperature=1.0,
    rank_weight=2.0,
    bce_weight=1.0,
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)


class ModelEMA:
    """Exponential Moving Average of model parameters with constant decay."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.backup = {}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


ema = ModelEMA(model, decay=0.999)

# =========================================================================
# STEP 3: TRAINING, VALIDATION & INFERENCE PIPELINE
# =========================================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_dense_train),
    torch.from_numpy(tracker_direct_train),
    torch.from_numpy(in_priors_train),
    torch.from_numpy(out_priors_train),
    torch.from_numpy(apex_priors_train),
    torch.from_numpy(tld_prior_train),
    torch.from_numpy(Y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_dense_val),
    torch.from_numpy(tracker_direct_val),
    torch.from_numpy(in_priors_val),
    torch.from_numpy(out_priors_val),
    torch.from_numpy(apex_priors_val),
    torch.from_numpy(tld_prior_val),
    torch.from_numpy(Y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_dense_test),
    torch.from_numpy(tracker_direct_test),
    torch.from_numpy(in_priors_test),
    torch.from_numpy(out_priors_test),
    torch.from_numpy(apex_priors_test),
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

epochs = 20
warmup_epochs = 2
decay_epochs = epochs - warmup_epochs
min_lr_ratio = 1e-5 / 1e-3


def lr_lambda(current_epoch: int) -> float:
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(warmup_epochs)
    progress = float(current_epoch - warmup_epochs) / float(max(1, decay_epochs))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
        1.0 + math.cos(math.pi * progress)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
best_val_recall = -1.0
best_model_path = "./working/best_gated_tracker_fusion_net.pt"


def evaluate(model, loader, device):
    """Evaluates validation loss and exact official Recall@10 metric."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for b_dense, b_direct, b_in, b_out, b_apex, b_tld, b_y in loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_in = b_in.to(device, non_blocking=True)
            b_out = b_out.to(device, non_blocking=True)
            b_apex = b_apex.to(device, non_blocking=True)
            b_tld = b_tld.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            logits = model(b_dense, b_direct, b_in, b_out, b_apex, b_tld)
            loss = criterion(logits, b_y)
            total_loss += loss.item() * len(b_dense)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(b_y.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(all_targets)
    val_recall = compute_numpy_recall_at_10(all_logits, all_targets)
    return avg_loss, val_recall


print(f"Starting training for {epochs} epochs on {device}...")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for b_dense, b_direct, b_in, b_out, b_apex, b_tld, b_y in train_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_in = b_in.to(device, non_blocking=True)
        b_out = b_out.to(device, non_blocking=True)
        b_apex = b_apex.to(device, non_blocking=True)
        b_tld = b_tld.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        # Apply Neighborhood Dropout on graph priors during training
        b_in = F.dropout(b_in, p=0.2, training=True)
        b_out = F.dropout(b_out, p=0.2, training=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(b_dense, b_direct, b_in, b_out, b_apex, b_tld)
        loss = criterion(logits, b_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        ema.update(model)

        running_loss += loss.item() * len(b_dense)
        num_samples += len(b_dense)

    scheduler.step()
    epoch_train_loss = running_loss / num_samples

    ema.apply_shadow(model)
    val_loss, val_recall = evaluate(model, val_loader, device)

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), best_model_path)

    ema.restore(model)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_val_recall:.5f}"
    )

# Restore best checkpoint weights
model.load_state_dict(
    torch.load(best_model_path, map_location=device, weights_only=True)
)
model.eval()
final_val_loss, final_val_score = evaluate(model, val_loader, device)

# Run full inference on target test domains
test_logits_list = []
with torch.no_grad():
    for b_dense, b_direct, b_in, b_out, b_apex, b_tld in test_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_in = b_in.to(device, non_blocking=True)
        b_out = b_out.to(device, non_blocking=True)
        b_apex = b_apex.to(device, non_blocking=True)
        b_tld = b_tld.to(device, non_blocking=True)

        logits = model(b_dense, b_direct, b_in, b_out, b_apex, b_tld)
        test_logits_list.append(logits.cpu().numpy())

test_logits = np.concatenate(test_logits_list, axis=0)
top10_test_indices = np.argsort(-test_logits, axis=1)[:, :10]

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

submission_df.to_csv(submission_csv_path, sep=",", index=False)
submission_df.to_csv(submission_tsv_path, sep="\t", index=False)

print(f"Final Validation Score: {final_val_score}")

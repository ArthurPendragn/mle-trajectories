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
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix
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

# Construct normalized one-hot tracker taxonomy metadata matrix
trackers_sorted_by_id = trackers_df.sort_values("tracker_id").reset_index(drop=True)
meta_company = pd.get_dummies(trackers_sorted_by_id["company"].fillna("Unknown"), prefix="comp")
meta_country = pd.get_dummies(trackers_sorted_by_id["country"].fillna("Unknown"), prefix="cntry")
meta_category = pd.get_dummies(trackers_sorted_by_id["category"].fillna("Unknown"), prefix="cat")
meta_brand = pd.get_dummies(trackers_sorted_by_id["brand"].fillna("Unknown"), prefix="brand")
tracker_meta_df = pd.concat([meta_company, meta_country, meta_category, meta_brand], axis=1)
tracker_meta_raw = tracker_meta_df.values.astype(np.float32)
meta_norms = np.linalg.norm(tracker_meta_raw, axis=1, keepdims=True)
meta_norms[meta_norms == 0] = 1.0
tracker_metadata = torch.tensor(tracker_meta_raw / meta_norms, dtype=torch.float32)
print(f"Constructed tracker taxonomy metadata tensor of shape: {tracker_metadata.shape}")

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

print(
    f"Targets constructed: Y_train positive rate: {Y_train.mean():.4f}, Y_val positive rate: {Y_val.mean():.4f}"
)

# Build Non-Validation Full-Graph Ground Truth Representation (Zero Validation Leakage)
print("Indexing non-validation training graph domains for full-graph relational retrieval...")
val_domain_set = set(val_domain_ids)
test_domain_set = set(test_domain_ids)
non_val_mask = (~np.isin(graph_domains, val_domain_ids)) & (~np.isin(graph_domains, test_domain_ids))
non_val_doms = graph_domains[non_val_mask]
non_val_trks = graph_trackers[non_val_mask]

sorted_known_domains, non_val_inverse = np.unique(non_val_doms, return_inverse=True)
known_csr = csr_matrix(
    (np.ones(len(non_val_doms), dtype=np.float32), (non_val_inverse, non_val_trks)),
    shape=(len(sorted_known_domains), num_trackers),
)
known_domain_map = {did: idx for idx, did in enumerate(sorted_known_domains)}
print(f"Indexed {len(sorted_known_domains)} non-validation domains from tracking_graph_train.")

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
    non_val_mask,
    non_val_doms,
    non_val_trks,
    non_val_inverse,
)
gc.collect()

# Compute empirical tracker-to-tracker Jaccard co-occurrence affinity strictly from Y_train
print("Computing empirical tracker co-occurrence matrix and prior log-odds...")
C = Y_train.T.astype(np.float32) @ Y_train.astype(np.float32)
diag = np.diag(C)
denom = diag[:, None] + diag[None, :] - C + 1e-7
J = C / denom
np.fill_diagonal(J, 0.0)
row_sums = J.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
cooccur_affinity = (J / row_sums).astype(np.float32)

global_tracker_prior = Y_train.mean(axis=0).astype(np.float32)
p_clip = np.clip(global_tracker_prior, 1e-4, 1.0 - 1e-4)
prior_log_odds = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
print(f"Computed cooccur_affinity {cooccur_affinity.shape} and prior_log_odds {prior_log_odds.shape}.")

print("--- Step 3: Unifying Relevant Domains and Mapping Hostnames & Apex Knowledge ---")
all_domain_ids = np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
num_total = len(all_domain_ids)
sorted_all_indices = np.argsort(all_domain_ids)
sorted_all_domain_ids = all_domain_ids[sorted_all_indices]
all_domain_set = set(all_domain_ids)
needed_domain_set = all_domain_set | set(sorted_known_domains)

domain_to_hostname = {}
domain_to_apex = {}
apex_tracker_counts = {}
apex_total_counts = {}

multi_tlds = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "com.br", "com.cn", "net.cn", "gov.cn", "edu.cn",
    "co.in", "net.in", "org.in", "com.ru", "net.ru", "org.ru", "co.nz",
    "com.tw", "com.mx", "co.za", "com.ar", "com.tr", "com.pl",
}


def parse_apex(h_str):
    if not h_str or h_str == "nan":
        return "unknown"
    h = h_str.strip().lower()
    if h.startswith("www."):
        h = h[4:]
    parts = h.split(".")
    if len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in multi_tlds:
        sld = parts[-3] if len(parts) >= 3 else ""
        tld = f"{parts[-2]}.{parts[-1]}"
        return f"{sld}.{tld}" if sld else tld
    elif len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    elif len(parts) == 1:
        return parts[0]
    return "unknown"


domains_pq = pq.ParquetFile("./input/domains.parquet")
for batch in domains_pq.iter_batches(
    columns=["domain_id", "domain"], batch_size=2500000
):
    b_df = batch.to_pandas()
    b_matched = b_df[b_df["domain_id"].isin(needed_domain_set)]
    if len(b_matched) > 0:
        for did, host in zip(b_matched["domain_id"], b_matched["domain"]):
            host_str = str(host)
            apex = parse_apex(host_str)
            if did in all_domain_set:
                domain_to_hostname[did] = host_str
                domain_to_apex[did] = apex
            if did in known_domain_map:
                k_idx = known_domain_map[did]
                p_start = known_csr.indptr[k_idx]
                p_end = known_csr.indptr[k_idx + 1]
                t_inds = known_csr.indices[p_start:p_end]
                if len(t_inds) > 0:
                    if apex not in apex_tracker_counts:
                        apex_tracker_counts[apex] = {}
                    cur_counts = apex_tracker_counts[apex]
                    for trk in t_inds:
                        cur_counts[trk] = cur_counts.get(trk, 0) + 1
                    apex_total_counts[apex] = apex_total_counts.get(apex, 0) + len(t_inds)

del needed_domain_set, known_domain_map
gc.collect()

print(
    f"Mapped {len(domain_to_hostname)} / {num_total} target hostnames and aggregated {len(apex_tracker_counts)} apex knowledge profiles."
)

# Construct Apex Tracker Prior Matrices and Match Confidence Indicators
apex_priors_all = np.zeros((num_total, num_trackers), dtype=np.float32)
apex_match_flag = np.zeros(num_total, dtype=np.float32)
apex_counts_feat = np.zeros(num_total, dtype=np.float32)

for idx, did in enumerate(all_domain_ids):
    apex = domain_to_apex.get(did, "unknown")
    if apex in apex_tracker_counts:
        apex_match_flag[idx] = 1.0
        tot = float(apex_total_counts[apex])
        apex_counts_feat[idx] = tot
        row = (1.0 * global_tracker_prior).copy()
        for trk, count in apex_tracker_counts[apex].items():
            row[trk] += count
        apex_priors_all[idx] = row / (tot + 1.0)
    else:
        apex_match_flag[idx] = 0.0
        apex_counts_feat[idx] = 0.0
        apex_priors_all[idx] = global_tracker_prior

print(f"Apex match rate across target domains: {apex_match_flag.mean():.4f}")
del apex_tracker_counts, apex_total_counts, domain_to_apex
gc.collect()

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

print("Training SGDClassifier on character 3-5 grams of url-classification.csv...")
url_df = pd.read_csv("./input/url-classification.csv", usecols=["url", "category"]).dropna()
cat_list = sorted(url_df["category"].unique().tolist())
num_cats = len(cat_list)
print(f"Found {num_cats} URL categories: {cat_list}")

if len(url_df) > 300000:
    url_train_df = url_df.sample(n=300000, random_state=42)
else:
    url_train_df = url_df

url_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=5,
    max_features=25000,
    sublinear_tf=True,
)
X_url_tfidf = url_vectorizer.fit_transform(url_train_df["url"].values)
y_url_labels = url_train_df["category"].values

sgd_topic_clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=25,
    random_state=42,
    n_jobs=-1,
)
sgd_topic_clf.fit(X_url_tfidf, y_url_labels)
del X_url_tfidf, y_url_labels, url_train_df, url_df
gc.collect()

print("Inferring continuous topic posteriors for all 325,000 hostnames...")
X_hostnames_tfidf = url_vectorizer.transform(hostname_list)
url_topic_posteriors = sgd_topic_clf.predict_proba(X_hostnames_tfidf).astype(np.float32)
del X_hostnames_tfidf, url_vectorizer, sgd_topic_clf
gc.collect()
print(f"Inferred {url_topic_posteriors.shape[1]} continuous topic posteriors for {len(url_topic_posteriors)} domains.")

print(
    "--- Step 5: Streaming link-graph.parquet for Graph Features, Direct Links & Directional Neighbors ---"
)
in_degrees_sorted = np.zeros(num_total, dtype=np.float32)
out_degrees_sorted = np.zeros(num_total, dtype=np.float32)
tracker_direct_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
neighbor_out_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)
neighbor_in_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)

num_known = len(sorted_known_domains)
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

    # 1. Direct Tracker Matches
    t_pos = np.searchsorted(sorted_tracker_domain_ids, dst)
    valid_t = t_pos < len(sorted_tracker_domain_ids)
    t_matched = np.zeros(len(dst), dtype=bool)
    t_matched[valid_t] = sorted_tracker_domain_ids[t_pos[valid_t]] == dst[valid_t]

    tracker_link_mask = src_matched & t_matched
    if np.any(tracker_link_mask):
        np.add.at(
            tracker_direct_sorted,
            (src_pos[tracker_link_mask], sorted_tracker_ids[t_pos[tracker_link_mask]]),
            1.0,
        )

    # 2. Out-neighbor Tracker Distributions
    cand_out_mask = src_matched & (~t_matched)
    if np.any(cand_out_mask):
        c_src_pos = src_pos[cand_out_mask]
        c_dst = dst[cand_out_mask]
        k_pos = np.searchsorted(sorted_known_domains, c_dst)
        valid_k = k_pos < num_known
        k_matched = np.zeros(len(c_dst), dtype=bool)
        k_matched[valid_k] = sorted_known_domains[k_pos[valid_k]] == c_dst[valid_k]
        if np.any(k_matched):
            m_src = c_src_pos[k_matched]
            m_k = k_pos[k_matched]
            adj = csr_matrix(
                (np.ones(len(m_src), dtype=np.float32), (m_src, m_k)),
                shape=(num_total, num_known),
            )
            neighbor_out_sorted += (adj @ known_csr).toarray()

    # 3. In-neighbor Tracker Distributions
    if np.any(dst_matched):
        c_dst_pos = dst_pos[dst_matched]
        c_src = src[dst_matched]
        k_in_pos = np.searchsorted(sorted_known_domains, c_src)
        valid_k_in = k_in_pos < num_known
        k_in_matched = np.zeros(len(c_src), dtype=bool)
        k_in_matched[valid_k_in] = sorted_known_domains[k_in_pos[valid_k_in]] == c_src[valid_k_in]
        if np.any(k_in_matched):
            m_dst = c_dst_pos[k_in_matched]
            m_k_in = k_in_pos[k_in_matched]
            adj_in = csr_matrix(
                (np.ones(len(m_dst), dtype=np.float32), (m_dst, m_k_in)),
                shape=(num_total, num_known),
            )
            neighbor_in_sorted += (adj_in @ known_csr).toarray()

    batch_count += 1

print(f"Link graph streaming complete across {batch_count} batches.")

neighbor_raw_sorted = neighbor_out_sorted + 0.5 * neighbor_in_sorted
nbr_sums = neighbor_raw_sorted.sum(axis=1, keepdims=True)
neighbor_priors_sorted = np.where(
    nbr_sums > 0,
    neighbor_raw_sorted / (nbr_sums + 1e-5),
    global_tracker_prior,
).astype(np.float32)

log1p_neighbor_out_sorted = np.log1p(neighbor_out_sorted.sum(axis=1, keepdims=True))
log1p_neighbor_in_sorted = np.log1p(neighbor_in_sorted.sum(axis=1, keepdims=True))

inv_sorted_indices = np.empty_like(sorted_all_indices)
inv_sorted_indices[sorted_all_indices] = np.arange(num_total)

in_degrees = in_degrees_sorted[inv_sorted_indices]
out_degrees = out_degrees_sorted[inv_sorted_indices]
tracker_direct_raw = tracker_direct_sorted[inv_sorted_indices]
tracker_direct_all = np.log1p(tracker_direct_raw).astype(np.float32)

neighbor_priors_all = neighbor_priors_sorted[inv_sorted_indices]
log1p_neighbor_out = log1p_neighbor_out_sorted[inv_sorted_indices]
log1p_neighbor_in = log1p_neighbor_in_sorted[inv_sorted_indices]

# Calculate link-graph tracker intensity ratios and total tracker hyperlink intensity
raw_tracker_sum = tracker_direct_raw.sum(axis=1, keepdims=True)
tracker_intensity_ratio = raw_tracker_sum / (out_degrees.reshape(-1, 1) + 1.0)
log1p_tracker_sum = np.log1p(raw_tracker_sum)

del (
    in_degrees_sorted,
    out_degrees_sorted,
    tracker_direct_sorted,
    tracker_direct_raw,
    neighbor_out_sorted,
    neighbor_in_sorted,
    neighbor_raw_sorted,
    neighbor_priors_sorted,
    log1p_neighbor_out_sorted,
    log1p_neighbor_in_sorted,
    known_csr,
    sorted_known_domains,
)
gc.collect()

log1p_in = np.log1p(in_degrees)
log1p_out = np.log1p(out_degrees)
degree_ratio = (log1p_in + 1.0) / (log1p_out + 1.0)
log1p_total = np.log1p(in_degrees + out_degrees)
is_leaf = (out_degrees == 0).astype(np.float32)
is_sink = (in_degrees == 0).astype(np.float32)

print("Extracting 32 functional intent keyword indicators from hostnames...")
INTENT_KEYWORDS = [
    "shop", "store", "blog", "news", "app", "tech", "mail", "api",
    "dev", "cloud", "portal", "forum", "media", "video", "music", "game",
    "play", "live", "tv", "info", "online", "web", "pay", "bank",
    "travel", "hotel", "auto", "car", "health", "med", "edu", "gov",
]
intent_keyword_features = np.zeros((num_total, len(INTENT_KEYWORDS)), dtype=np.float32)
for j, kw in enumerate(INTENT_KEYWORDS):
    intent_keyword_features[:, j] = [1.0 if kw in h else 0.0 for h in hostname_list]

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
global_tracker_prior = Y_train.mean(axis=0).astype(np.float32)

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
    tracker_intensity_ratio,
    log1p_tracker_sum,
    log1p_neighbor_out,
    log1p_neighbor_in,
    apex_match_flag.reshape(-1, 1),
    np.log1p(apex_counts_feat).reshape(-1, 1),
    parsed_sub_count.reshape(-1, 1),
    parsed_len.reshape(-1, 1),
    parsed_digits.reshape(-1, 1),
    digit_ratio.reshape(-1, 1),
    parsed_hyphens.reshape(-1, 1),
    parsed_vowels.reshape(-1, 1),
    vowel_ratio.reshape(-1, 1),
    intent_keyword_features,
    fop_scores.reshape(-1, 1),
    fop_matched.reshape(-1, 1),
    url_topic_posteriors,
    tld_one_hot,
    tld_freq_feature,
    all_svd_features,
]

X_dense_all = np.hstack(dense_feature_blocks).astype(np.float32)
np.nan_to_num(X_dense_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_dense_train = X_dense_all[:num_train]
X_dense_val = X_dense_all[num_train : num_train + num_val]
X_dense_test = X_dense_all[num_train + num_val :]

tracker_direct_train = tracker_direct_all[:num_train]
tracker_direct_val = tracker_direct_all[num_train : num_train + num_val]
tracker_direct_test = tracker_direct_all[num_train + num_val :]

apex_priors_train = apex_priors_all[:num_train]
apex_priors_val = apex_priors_all[num_train : num_train + num_val]
apex_priors_test = apex_priors_all[num_train + num_val :]

neighbor_priors_train = neighbor_priors_all[:num_train]
neighbor_priors_val = neighbor_priors_all[num_train : num_train + num_val]
neighbor_priors_test = neighbor_priors_all[num_train + num_val :]

print("Fitting StandardScaler strictly on training split...")
scaler = StandardScaler()
X_dense_train = scaler.fit_transform(X_dense_train).astype(np.float32)
X_dense_val = scaler.transform(X_dense_val).astype(np.float32)
X_dense_test = scaler.transform(X_dense_test).astype(np.float32)

dense_feature_dim = X_dense_train.shape[1]
print(f"Features prepared: dense_feature_dim={dense_feature_dim}")

# =========================================================================
# STEP 2: DUAL MODEL ARCHITECTURES & CALIBRATED LOSS
# =========================================================================


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel gating mechanism."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, max(dim // reduction, 16), bias=False)
        self.fc2 = nn.Linear(max(dim // reduction, 16), dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.fc2(self.act(self.fc1(x))))
        return x * w


class PreActSEResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm, GELU, and SE attention."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.se = SqueezeExcitation(dim, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.linear1(self.act1(self.norm1(x))))
        out = self.dropout2(self.linear2(self.act2(self.norm2(out))))
        out = self.se(out)
        return residual + out


class TrackerResNet(nn.Module):
    """Model 1: Residual Gated Network with SE channel attention and un-attenuated direct skip routing."""

    def __init__(
        self,
        dense_dim: int,
        num_trackers: int = 355,
        hidden_dim: int = 384,
        dropout: float = 0.2,
        cooccur_affinity: torch.Tensor = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers

        self.stem_dense = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stem_direct = nn.Linear(num_trackers, hidden_dim)
        self.stem_apex = nn.Linear(num_trackers, hidden_dim)
        self.stem_neighbor = nn.Linear(num_trackers, hidden_dim)

        self.block1 = PreActSEResBlock(hidden_dim, dropout=dropout)
        self.block2 = PreActSEResBlock(hidden_dim, dropout=dropout)
        self.block3 = PreActSEResBlock(hidden_dim, dropout=dropout)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_trackers)

        # Un-attenuated direct link bypass initialized to +5.0
        self.direct_bypass = nn.Parameter(torch.full((num_trackers,), 5.0))
        # Apex and neighbor relational priority scalers
        self.apex_scaler = nn.Parameter(torch.full((num_trackers,), 3.2))
        self.neighbor_scaler = nn.Parameter(torch.full((num_trackers,), 1.8))

        if cooccur_affinity is not None:
            self.register_buffer("cooccur_affinity", cooccur_affinity.float())
        else:
            self.register_buffer("cooccur_affinity", torch.zeros(num_trackers, num_trackers))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        x_apex: torch.Tensor,
        x_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        h = (
            self.stem_dense(x_dense)
            + self.stem_direct(x_direct)
            + self.stem_apex(x_apex)
            + self.stem_neighbor(x_neighbor)
        )
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.out_norm(h)
        latent_logits = self.head(h)

        direct_logits = x_direct * self.direct_bypass

        apex_clamped = x_apex.clamp(1e-4, 1.0 - 1e-4)
        apex_log_odds = torch.log(apex_clamped / (1.0 - apex_clamped))
        apex_logits = apex_log_odds * self.apex_scaler

        nbr_clamped = x_neighbor.clamp(1e-4, 1.0 - 1e-4)
        nbr_log_odds = torch.log(nbr_clamped / (1.0 - nbr_clamped))
        nbr_logits = nbr_log_odds * self.neighbor_scaler

        logits = latent_logits + direct_logits + apex_logits + nbr_logits
        refined = logits + self.alpha * torch.matmul(logits, self.cooccur_affinity)
        return refined


class TrackerDenseNet(nn.Module):
    """Model 2: Dense Highway Network with cascading feature reuse and cross-tracker projections."""

    def __init__(
        self,
        dense_dim: int,
        num_trackers: int = 355,
        hidden_dim: int = 256,
        step_dim: int = 128,
        dropout: float = 0.2,
        cooccur_affinity: torch.Tensor = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers

        self.in_proj = nn.Linear(dense_dim + 3 * num_trackers, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, step_dim)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim + step_dim)
        self.fc2 = nn.Linear(hidden_dim + step_dim, step_dim)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(hidden_dim + 2 * step_dim)
        self.fc3 = nn.Linear(hidden_dim + 2 * step_dim, step_dim)
        self.drop3 = nn.Dropout(dropout)

        total_dense_dim = hidden_dim + 3 * step_dim
        self.norm_final = nn.LayerNorm(total_dense_dim)
        self.head = nn.Linear(total_dense_dim, num_trackers)

        self.cross_tracker_proj = nn.Sequential(
            nn.Linear(num_trackers, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_trackers),
        )
        nn.init.zeros_(self.cross_tracker_proj[-1].weight)
        nn.init.zeros_(self.cross_tracker_proj[-1].bias)

        self.direct_scale = nn.Parameter(torch.full((num_trackers,), 4.2))
        self.apex_scale = nn.Parameter(torch.full((num_trackers,), 2.8))

        if cooccur_affinity is not None:
            self.register_buffer("cooccur_affinity", cooccur_affinity.float())
        else:
            self.register_buffer("cooccur_affinity", torch.zeros(num_trackers, num_trackers))
        self.alpha = nn.Parameter(torch.tensor(0.08))

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        x_apex: torch.Tensor,
        x_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        x_cat = torch.cat([x_dense, x_direct, x_apex, x_neighbor], dim=-1)
        h0 = F.silu(self.in_proj(x_cat))

        h1 = self.drop1(F.silu(self.fc1(self.norm1(h0))))
        c1 = torch.cat([h0, h1], dim=-1)

        h2 = self.drop2(F.silu(self.fc2(self.norm2(c1))))
        c2 = torch.cat([h0, h1, h2], dim=-1)

        h3 = self.drop3(F.silu(self.fc3(self.norm3(c2))))
        c3 = torch.cat([h0, h1, h2, h3], dim=-1)

        latent_logits = self.head(self.norm_final(c3))

        cross_features = x_direct + 0.5 * x_apex + 0.5 * x_neighbor
        cross_logits = self.cross_tracker_proj(cross_features)

        apex_clamped = x_apex.clamp(1e-4, 1.0 - 1e-4)
        apex_log_odds = torch.log(apex_clamped / (1.0 - apex_clamped))

        logits = (
            latent_logits
            + x_direct * self.direct_scale
            + apex_log_odds * self.apex_scale
            + cross_logits
        )
        refined = logits + self.alpha * torch.matmul(logits, self.cooccur_affinity)
        return refined


class AsymmetricFocalLoss(nn.Module):
    """Calibrated Asymmetric Focal Loss with positive upweighting and gradient saturation prevention."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        pos_weight: float = 2.5,
        clip: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.pos_weight = pos_weight
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        pos_loss = targets * torch.log(probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - probs) ** self.gamma_pos)
        pos_loss = self.pos_weight * pos_loss

        if self.clip > 0.0:
            neg_probs = (probs - self.clip).clamp(min=0.0)
        else:
            neg_probs = probs
        neg_loss = (
            (1.0 - targets)
            * (neg_probs ** self.gamma_neg)
            * torch.log((1.0 - neg_probs).clamp(min=self.eps))
        )

        loss = -(pos_loss + neg_loss).sum(dim=-1).mean()
        return loss


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
print(f"Compute device: {device}")

# =========================================================================
# STEP 3: SEQUENTIAL MULTI-MODEL TRAINING & PROBABILITY BLENDING
# =========================================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_dense_train),
    torch.from_numpy(tracker_direct_train),
    torch.from_numpy(apex_priors_train),
    torch.from_numpy(neighbor_priors_train),
    torch.from_numpy(Y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_dense_val),
    torch.from_numpy(tracker_direct_val),
    torch.from_numpy(apex_priors_val),
    torch.from_numpy(neighbor_priors_val),
    torch.from_numpy(Y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_dense_test),
    torch.from_numpy(tracker_direct_test),
    torch.from_numpy(apex_priors_test),
    torch.from_numpy(neighbor_priors_test),
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

criterion = AsymmetricFocalLoss(
    gamma_neg=2.0,
    gamma_pos=0.0,
    pos_weight=2.5,
    clip=0.0,
).to(device)


def evaluate_model(model, loader, criterion, device):
    """Evaluates validation loss and official Recall@10 metric."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for b_dense, b_direct, b_apex, b_nbr, b_y in loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_apex = b_apex.to(device, non_blocking=True)
            b_nbr = b_nbr.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            logits = model(b_dense, b_direct, b_apex, b_nbr)
            loss = criterion(logits, b_y)
            total_loss += loss.item() * len(b_dense)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(b_y.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(all_targets)
    val_recall = compute_numpy_recall_at_10(all_logits, all_targets)
    return avg_loss, val_recall, all_logits


def predict_probabilities(model, loader, device):
    """Generates continuous probability predictions over all 355 trackers."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for b_dense, b_direct, b_apex, b_nbr in loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_apex = b_apex.to(device, non_blocking=True)
            b_nbr = b_nbr.to(device, non_blocking=True)

            logits = model(b_dense, b_direct, b_apex, b_nbr)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# --- 1. Train Model 1: TrackerResNet ---
print("\n--- Training Model 1: TrackerResNet ---")
cooccur_tensor = torch.from_numpy(cooccur_affinity).to(device)

model1 = TrackerResNet(
    dense_dim=dense_feature_dim,
    num_trackers=num_trackers,
    hidden_dim=384,
    dropout=0.2,
    cooccur_affinity=cooccur_tensor,
).to(device)

epochs_m1 = 11
optimizer1 = AdamW(model1.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler1 = CosineAnnealingLR(optimizer1, T_max=epochs_m1, eta_min=1e-5)
best_m1_recall = -1.0
best_m1_path = "./working/best_tracker_resnet.pt"

for epoch in range(1, epochs_m1 + 1):
    model1.train()
    running_loss = 0.0
    num_samples = 0

    for b_dense, b_direct, b_apex, b_nbr, b_y in train_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_apex = b_apex.to(device, non_blocking=True)
        b_nbr = b_nbr.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        optimizer1.zero_grad(set_to_none=True)
        logits = model1(b_dense, b_direct, b_apex, b_nbr)
        loss = criterion(logits, b_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
        optimizer1.step()

        running_loss += loss.item() * len(b_dense)
        num_samples += len(b_dense)

    scheduler1.step()
    epoch_loss = running_loss / num_samples
    val_loss, val_recall, _ = evaluate_model(model1, val_loader, criterion, device)

    if val_recall > best_m1_recall:
        best_m1_recall = val_recall
        torch.save(model1.state_dict(), best_m1_path)

    print(
        f"M1 Epoch {epoch:02d}/{epochs_m1:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_m1_recall:.5f}"
    )

model1.load_state_dict(torch.load(best_m1_path, map_location=device, weights_only=True))
_, val_recall_m1, val_logits_m1 = evaluate_model(model1, val_loader, criterion, device)
val_probs_m1 = 1.0 / (1.0 + np.exp(-val_logits_m1))
print(f"Model 1 (TrackerResNet) Best Hold-out Recall@10: {val_recall_m1:.5f}")

# --- 2. Train Model 2: TrackerDenseNet ---
print("\n--- Training Model 2: TrackerDenseNet ---")
model2 = TrackerDenseNet(
    dense_dim=dense_feature_dim,
    num_trackers=num_trackers,
    hidden_dim=256,
    step_dim=128,
    dropout=0.2,
    cooccur_affinity=cooccur_tensor,
).to(device)

epochs_m2 = 11
optimizer2 = AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler2 = CosineAnnealingLR(optimizer2, T_max=epochs_m2, eta_min=1e-5)
best_m2_recall = -1.0
best_m2_path = "./working/best_tracker_densenet.pt"

for epoch in range(1, epochs_m2 + 1):
    model2.train()
    running_loss = 0.0
    num_samples = 0

    for b_dense, b_direct, b_apex, b_nbr, b_y in train_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_apex = b_apex.to(device, non_blocking=True)
        b_nbr = b_nbr.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        optimizer2.zero_grad(set_to_none=True)
        logits = model2(b_dense, b_direct, b_apex, b_nbr)
        loss = criterion(logits, b_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
        optimizer2.step()

        running_loss += loss.item() * len(b_dense)
        num_samples += len(b_dense)

    scheduler2.step()
    epoch_loss = running_loss / num_samples
    val_loss, val_recall, _ = evaluate_model(model2, val_loader, criterion, device)

    if val_recall > best_m2_recall:
        best_m2_recall = val_recall
        torch.save(model2.state_dict(), best_m2_path)

    print(
        f"M2 Epoch {epoch:02d}/{epochs_m2:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_m2_recall:.5f}"
    )

model2.load_state_dict(torch.load(best_m2_path, map_location=device, weights_only=True))
_, val_recall_m2, val_logits_m2 = evaluate_model(model2, val_loader, criterion, device)
val_probs_m2 = 1.0 / (1.0 + np.exp(-val_logits_m2))
print(f"Model 2 (TrackerDenseNet) Best Hold-out Recall@10: {val_recall_m2:.5f}")

# --- 3. Grid Search Optimal Probability Blend Weight ---
print("\n--- Performing Validation Blend Grid Search ---")
best_blend_w = 0.5
best_ensemble_recall = -1.0

for w in np.linspace(0.0, 1.0, 21):
    p_blend = w * val_probs_m1 + (1.0 - w) * val_probs_m2
    score = compute_numpy_recall_at_10(p_blend, Y_val)
    if score > best_ensemble_recall:
        best_ensemble_recall = score
        best_blend_w = float(w)

print(
    f"Optimal ensemble weight w (Model 1): {best_blend_w:.2f}, w (Model 2): {1.0 - best_blend_w:.2f} | Ensembled Recall@10: {best_ensemble_recall:.5f}"
)
final_val_score = best_ensemble_recall

# --- 4. Ensembled Inference on Test Set ---
print("\n--- Generating Ensembled Predictions on Test Domains ---")
probs1_test = predict_probabilities(model1, test_loader, device)
probs2_test = predict_probabilities(model2, test_loader, device)
p_test_ensemble = best_blend_w * probs1_test + (1.0 - best_blend_w) * probs2_test

top10_test_indices = np.argsort(-p_test_ensemble, axis=1)[:, :10]

test_domain_ids_repeated = np.repeat(test_domain_ids, 10)
pred_tracking_domain_ids = tracker_id_to_domain_id[top10_test_indices.ravel()]

submission_df = pd.DataFrame(
    {
        "domain_id": test_domain_ids_repeated,
        "tracking_domain_id": pred_tracking_domain_ids,
    }
)

assert len(submission_df) == 500000
assert submission_df["domain_id"].nunique() == len(test_domain_ids)
assert not submission_df.isnull().any().any()

submission_csv_path = "./submission/submission.csv"
submission_tsv_path = "./submission/submission.tsv"

submission_df.to_csv(submission_csv_path, sep=",", index=False)
submission_df.to_csv(submission_tsv_path, sep="\t", index=False)
print(f"Submission saved successfully to {submission_csv_path} and {submission_tsv_path} ({len(submission_df)} rows).")

print(f"Final Validation Score: {final_val_score}")

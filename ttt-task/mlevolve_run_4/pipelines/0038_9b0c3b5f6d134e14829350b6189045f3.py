from collections import Counter
import json
import math
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducibility and required directories
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Load Metadata and Target Domains
# =============================================================================
trackers_df = pd.read_csv("./input/trackers.tsv", sep="\t")
target_df = pd.read_csv("./input/target.tsv", sep="\t")

test_domain_ids = target_df["domain_id"].values.astype(np.int64)
target_set = set(test_domain_ids)

num_trackers = 355
tracker_domain_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)

# Extract structured tracker taxonomy metadata matrix & category index map
tracker_meta_df = (
    trackers_df.drop_duplicates(subset=["tracker_id"])
    .sort_values("tracker_id")
    .set_index("tracker_id")
    .reindex(range(num_trackers))
)
tracker_meta_df["company"] = tracker_meta_df["company"].fillna("Unknown").astype(str)
tracker_meta_df["country"] = tracker_meta_df["country"].fillna("Unknown").astype(str)
tracker_meta_df["category"] = tracker_meta_df["category"].fillna("Unknown").astype(str)

unique_tracker_cats = sorted(tracker_meta_df["category"].unique())
num_tracker_cats = len(unique_tracker_cats)
tracker_cat_to_idx = {c: i for i, c in enumerate(unique_tracker_cats)}

# One-hot mapping from tracker_id to tracker category: (355, num_tracker_cats)
tracker_cat_matrix = np.zeros((num_trackers, num_tracker_cats), dtype=np.float32)
for tid, cat in enumerate(tracker_meta_df["category"]):
    tracker_cat_matrix[tid, tracker_cat_to_idx[cat]] = 1.0

comp_dummies = pd.get_dummies(tracker_meta_df["company"], prefix="comp", dtype=np.float32)
cntr_dummies = pd.get_dummies(tracker_meta_df["country"], prefix="cntr", dtype=np.float32)
cat_dummies = pd.get_dummies(tracker_meta_df["category"], prefix="cat", dtype=np.float32)

tracker_meta_matrix = np.hstack(
    [cat_dummies.values, comp_dummies.values, cntr_dummies.values]
).astype(np.float32)
dim_meta = tracker_meta_matrix.shape[1]

# =============================================================================
# 2. Define Leak-Free Train / Validation / Test Splits
# =============================================================================
tracking_train_pfile = pq.ParquetFile("./input/tracking_graph_train.parquet")
train_tracking_table = tracking_train_pfile.read(columns=["domain_id", "tracker_id"])
train_df_all = train_tracking_table.to_pandas()

unique_candidate_domains = train_df_all["domain_id"].unique()
valid_candidates = np.array(
    [d for d in unique_candidate_domains if d not in target_set], dtype=np.int64
)

rng = np.random.RandomState(42)
permuted_indices = rng.permutation(len(valid_candidates))
N_TRAIN = 600_000
N_VAL = 25_000

# Preserve the exact 25k validation set while expanding training domains to 600,000
val_domain_ids = valid_candidates[permuted_indices[450_000 : 450_000 + N_VAL]]
train_domain_ids = np.concatenate(
    [
        valid_candidates[permuted_indices[:450_000]],
        valid_candidates[permuted_indices[450_000 + N_VAL : N_TRAIN + N_VAL]],
    ]
)

train_set = set(train_domain_ids)
val_set = set(val_domain_ids)

assert (
    len(train_set.intersection(val_set)) == 0
), "Train and validation domains overlap!"
assert len(train_set.intersection(target_set)) == 0, "Train and test domains overlap!"
assert (
    len(val_set.intersection(target_set)) == 0
), "Validation and test domains overlap!"

# Multi-hot binary label matrices for train and validation
y_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.float32)
y_val = np.zeros((len(val_domain_ids), num_trackers), dtype=np.float32)

train_id_to_row = {dom_id: i for i, dom_id in enumerate(train_domain_ids)}
val_id_to_row = {dom_id: i for i, dom_id in enumerate(val_domain_ids)}

train_events = train_df_all[train_df_all["domain_id"].isin(train_set)]
train_rows = train_events["domain_id"].map(train_id_to_row).values
train_cols = train_events["tracker_id"].values
y_train[train_rows, train_cols] = 1.0

val_events = train_df_all[train_df_all["domain_id"].isin(val_set)]
val_rows = val_events["domain_id"].map(val_id_to_row).values
val_cols = val_events["tracker_id"].values
y_val[val_rows, val_cols] = 1.0

# Functional tracker-category distribution for auxiliary multi-task loss
y_train_cat = np.clip(np.matmul(y_train, tracker_cat_matrix), 0.0, 1.0).astype(np.float32)
y_val_cat = np.clip(np.matmul(y_val, tracker_cat_matrix), 0.0, 1.0).astype(np.float32)

# Global tracker prior from training split
global_prior = y_train.mean(axis=0).astype(np.float32)

# Build expanded leak-free label bank encompassing all ~1.325M candidate domains outside val and test
bank_mask = np.ones(len(valid_candidates), dtype=bool)
bank_mask[permuted_indices[450_000 : 450_000 + N_VAL]] = False
bank_domain_ids = valid_candidates[bank_mask]
bank_set = set(bank_domain_ids)

assert len(bank_set.intersection(val_set)) == 0, "Bank contains validation domains!"
assert len(bank_set.intersection(target_set)) == 0, "Bank contains test domains!"

bank_domain_ids_sorted = np.sort(bank_domain_ids)
N_bank = len(bank_domain_ids_sorted)
bank_id_to_idx = {dom_id: i for i, dom_id in enumerate(bank_domain_ids_sorted)}

bank_events = train_df_all[train_df_all["domain_id"].isin(bank_set)]
bank_rows = bank_events["domain_id"].map(bank_id_to_idx).values
bank_cols = bank_events["tracker_id"].values
bank_data = np.ones(len(bank_rows), dtype=np.float32)
Y_bank = csr_matrix(
    (bank_data, (bank_rows, bank_cols)),
    shape=(N_bank, num_trackers),
    dtype=np.float32,
)
Y_bank.data = np.ones_like(Y_bank.data)

del train_df_all, train_tracking_table, train_events, val_events, bank_events, bank_rows, bank_cols, bank_data

# Empirical tracker co-occurrence matrix from y_train
cooccur_counts = np.matmul(y_train.T, y_train)
diag = np.sqrt(np.diag(cooccur_counts))
denom = np.outer(diag, diag)
denom[denom == 0] = 1.0
cooccur_matrix = (cooccur_counts / denom).astype(np.float32)

# Sparsified PPMI (Positive Pointwise Mutual Information) Top-12 Neighbor Graph
domain_count = float(len(y_train))
tracker_counts = np.diag(cooccur_counts)
outer_counts = np.maximum(np.outer(tracker_counts, tracker_counts), 1.0)
pmi_matrix = np.log(np.maximum(cooccur_counts * domain_count / outer_counts, 1e-12))
ppmi_matrix = np.maximum(pmi_matrix, 0.0).astype(np.float32)

K_NEIGHBORS = 12
ppmi_no_diag = ppmi_matrix.copy()
np.fill_diagonal(ppmi_no_diag, 0.0)

neighbor_indices = np.zeros((num_trackers, K_NEIGHBORS + 1), dtype=np.int64)
neighbor_ppmi_weights = np.zeros((num_trackers, K_NEIGHBORS + 1), dtype=np.float32)

for tr_i in range(num_trackers):
    top_k = np.argsort(ppmi_no_diag[tr_i])[-K_NEIGHBORS:]
    neighbor_indices[tr_i, 0] = tr_i  # Self-loop
    neighbor_indices[tr_i, 1:] = top_k
    vals = ppmi_no_diag[tr_i, top_k]
    max_v = np.max(vals) if np.max(vals) > 0 else 1.0
    neighbor_ppmi_weights[tr_i, 0] = 1.0  # Self-loop weight
    neighbor_ppmi_weights[tr_i, 1:] = vals / max_v

all_eval_domain_ids = np.unique(
    np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
)
eval_domain_ids_sorted = np.sort(all_eval_domain_ids)
N_eval = len(eval_domain_ids_sorted)
eval_id_to_idx = {dom_id: i for i, dom_id in enumerate(eval_domain_ids_sorted)}
all_needed_sorted = eval_domain_ids_sorted
N_ALL = N_eval
needed_id_to_idx = eval_id_to_idx

# =============================================================================
# 3. Stream Link Graph for Topology & Direct 355-Tracker Bridge Features
# =============================================================================
in_degree = np.zeros(N_eval, dtype=np.int32)
out_degree = np.zeros(N_eval, dtype=np.int32)
bank_in_degree = np.zeros(N_bank, dtype=np.int32)
bank_out_degree = np.zeros(N_bank, dtype=np.int32)
direct_tracker_matrix = np.zeros((N_eval, num_trackers), dtype=np.int16)
direct_tracker_counts = np.zeros(N_eval, dtype=np.int32)

tracker_domain_ids = np.array(list(tracker_domain_to_tracker_id.keys()), dtype=np.int64)
sort_order = np.argsort(tracker_domain_ids)
tracker_domain_ids_sorted = tracker_domain_ids[sort_order]
tracker_id_sorted = np.array(
    [tracker_domain_to_tracker_id[td] for td in tracker_domain_ids_sorted],
    dtype=np.int16,
)

out_src_list = []
out_dst_list = []
in_src_list = []
in_dst_list = []

link_pfile = pq.ParquetFile("./input/link-graph.parquet")
for batch in link_pfile.iter_batches(
    batch_size=2_000_000, columns=["source_domain_id", "target_domain_id"]
):
    src = np.asarray(batch["source_domain_id"])
    dst = np.asarray(batch["target_domain_id"])

    idx_eval_src = np.searchsorted(eval_domain_ids_sorted, src)
    valid_eval_src = (idx_eval_src < N_eval) & (
        eval_domain_ids_sorted[np.clip(idx_eval_src, 0, N_eval - 1)] == src
    )

    idx_eval_dst = np.searchsorted(eval_domain_ids_sorted, dst)
    valid_eval_dst = (idx_eval_dst < N_eval) & (
        eval_domain_ids_sorted[np.clip(idx_eval_dst, 0, N_eval - 1)] == dst
    )

    if np.any(valid_eval_src):
        src_matched = idx_eval_src[valid_eval_src]
        counts_src = np.bincount(src_matched, minlength=N_eval)
        out_degree += counts_src.astype(np.int32)

    if np.any(valid_eval_dst):
        dst_matched = idx_eval_dst[valid_eval_dst]
        counts_dst = np.bincount(dst_matched, minlength=N_eval)
        in_degree += counts_dst.astype(np.int32)

    # Direct tracker hyperlinks from evaluation domains
    if np.any(valid_eval_src):
        dst_for_valid_src = dst[valid_eval_src]
        src_for_valid = idx_eval_src[valid_eval_src]

        t_idx = np.searchsorted(tracker_domain_ids_sorted, dst_for_valid_src)
        is_tracker = (t_idx < len(tracker_domain_ids_sorted)) & (
            tracker_domain_ids_sorted[
                np.clip(t_idx, 0, len(tracker_domain_ids_sorted) - 1)
            ]
            == dst_for_valid_src
        )

        if np.any(is_tracker):
            matched_src_idx = src_for_valid[is_tracker]
            matched_t_idx = t_idx[is_tracker]
            tr_ids = tracker_id_sorted[matched_t_idx]

            np.add.at(direct_tracker_counts, matched_src_idx, 1)
            np.add.at(direct_tracker_matrix, (matched_src_idx, tr_ids), 1)

    # Eval -> Bank outgoing edges
    idx_bank_dst = np.searchsorted(bank_domain_ids_sorted, dst)
    valid_bank_dst = (idx_bank_dst < N_bank) & (
        bank_domain_ids_sorted[np.clip(idx_bank_dst, 0, N_bank - 1)] == dst
    )
    if np.any(valid_bank_dst):
        counts_b_dst = np.bincount(idx_bank_dst[valid_bank_dst], minlength=N_bank)
        bank_in_degree += counts_b_dst.astype(np.int32)

    mask_out = valid_eval_src & valid_bank_dst
    if np.any(mask_out):
        e_src = idx_eval_src[mask_out]
        b_dst = idx_bank_dst[mask_out]
        non_self = eval_domain_ids_sorted[e_src] != bank_domain_ids_sorted[b_dst]
        if np.any(non_self):
            out_src_list.append(e_src[non_self])
            out_dst_list.append(b_dst[non_self])

    # Bank -> Eval incoming edges
    idx_bank_src = np.searchsorted(bank_domain_ids_sorted, src)
    valid_bank_src = (idx_bank_src < N_bank) & (
        bank_domain_ids_sorted[np.clip(idx_bank_src, 0, N_bank - 1)] == src
    )
    if np.any(valid_bank_src):
        counts_b_src = np.bincount(idx_bank_src[valid_bank_src], minlength=N_bank)
        bank_out_degree += counts_b_src.astype(np.int32)

    mask_in = valid_eval_dst & valid_bank_src
    if np.any(mask_in):
        b_src = idx_bank_src[mask_in]
        e_dst = idx_eval_dst[mask_in]
        non_self = bank_domain_ids_sorted[b_src] != eval_domain_ids_sorted[e_dst]
        if np.any(non_self):
            in_src_list.append(e_dst[non_self])
            in_dst_list.append(b_src[non_self])

# Project outgoing links to bank domains (Adamic-Adar / IDW Dirichlet smoothed probabilities, log-lift, log1p counts)
M_dirichlet = 10.0
bank_in_idw = (1.0 / np.maximum(np.log1p(bank_in_degree.astype(np.float32)), np.log(2.0))).astype(np.float32)
if len(out_src_list) > 0:
    b_out_rows = np.concatenate(out_src_list)
    b_out_cols = np.concatenate(out_dst_list)
    del out_src_list, out_dst_list
    B_out = csr_matrix(
        (np.ones(len(b_out_rows), dtype=np.float32), (b_out_rows, b_out_cols)),
        shape=(N_eval, N_bank),
    )
    B_out.data = bank_in_idw[B_out.indices]
    del b_out_rows, b_out_cols
    votes_out = B_out.dot(Y_bank).toarray().astype(np.float32)
    deg_out_bank = np.asarray(B_out.sum(axis=1)).flatten()
    del B_out
    out_neighbor_smooth = (votes_out + M_dirichlet * global_prior) / (deg_out_bank[:, None] + M_dirichlet)
    out_neighbor_lift = np.log((out_neighbor_smooth + 1e-4) / (global_prior + 1e-4))
    out_neighbor_log_cnt = np.log1p(votes_out)
    del votes_out
else:
    deg_out_bank = np.zeros(N_eval, dtype=np.float32)
    out_neighbor_smooth = np.tile(global_prior, (N_eval, 1)).astype(np.float32)
    out_neighbor_lift = np.zeros((N_eval, num_trackers), dtype=np.float32)
    out_neighbor_log_cnt = np.zeros((N_eval, num_trackers), dtype=np.float32)

# Project incoming links from bank domains (Adamic-Adar / IDW Dirichlet smoothed probabilities, log-lift, log1p counts)
bank_out_idw = (1.0 / np.maximum(np.log1p(bank_out_degree.astype(np.float32)), np.log(2.0))).astype(np.float32)
if len(in_src_list) > 0:
    b_in_rows = np.concatenate(in_src_list)
    b_in_cols = np.concatenate(in_dst_list)
    del in_src_list, in_dst_list
    B_in = csr_matrix(
        (np.ones(len(b_in_rows), dtype=np.float32), (b_in_rows, b_in_cols)),
        shape=(N_eval, N_bank),
    )
    B_in.data = bank_out_idw[B_in.indices]
    del b_in_rows, b_in_cols
    votes_in = B_in.dot(Y_bank).toarray().astype(np.float32)
    deg_in_bank = np.asarray(B_in.sum(axis=1)).flatten()
    del B_in
    in_neighbor_smooth = (votes_in + M_dirichlet * global_prior) / (deg_in_bank[:, None] + M_dirichlet)
    in_neighbor_lift = np.log((in_neighbor_smooth + 1e-4) / (global_prior + 1e-4))
    in_neighbor_log_cnt = np.log1p(votes_in)
    del votes_in
else:
    deg_in_bank = np.zeros(N_eval, dtype=np.float32)
    in_neighbor_smooth = np.tile(global_prior, (N_eval, 1)).astype(np.float32)
    in_neighbor_lift = np.zeros((N_eval, num_trackers), dtype=np.float32)
    in_neighbor_log_cnt = np.zeros((N_eval, num_trackers), dtype=np.float32)

del bank_in_degree, bank_out_degree, bank_in_idw, bank_out_idw

del Y_bank

# =============================================================================
# 4. Stream Hostnames from domains.parquet
# =============================================================================
domain_names = np.array([""] * N_ALL, dtype=object)
found_mask = np.zeros(N_ALL, dtype=bool)
domains_pfile = pq.ParquetFile("./input/domains.parquet")

for batch in domains_pfile.iter_batches(
    batch_size=2_000_000, columns=["domain_id", "domain"]
):
    b_ids = np.asarray(batch["domain_id"])
    idx = np.searchsorted(all_needed_sorted, b_ids)
    valid = (idx < N_ALL) & (all_needed_sorted[np.clip(idx, 0, N_ALL - 1)] == b_ids)

    if np.any(valid):
        b_doms = np.asarray(batch["domain"])
        matched_idx = idx[valid]
        domain_names[matched_idx] = b_doms[valid]
        found_mask[matched_idx] = True
        if np.all(found_mask):
            break

# =============================================================================
# 5. External Context: Freedom of the Press & URL Classification
# =============================================================================
try:
    press_df = pd.read_csv(
        "./input/freedom-of-the-press.csv", sep=None, engine="python"
    )
    if len(press_df.columns) == 1:
        press_df = pd.read_csv("./input/freedom-of-the-press.csv", sep="\t")
except Exception:
    press_df = pd.read_csv("./input/freedom-of-the-press.csv", sep="\t")

press_df.columns = [c.strip() for c in press_df.columns]
tld_col = [c for c in press_df.columns if "tld" in c.lower()][0]
score_col = [
    c for c in press_df.columns if "freedom" in c.lower() or "score" in c.lower()
][0]
press_map = dict(
    zip(
        press_df[tld_col].astype(str).str.lower().str.strip(),
        press_df[score_col].astype(float),
    )
)
median_press = float(np.median(list(press_map.values())))

url_df = pd.read_csv("./input/url-classification.csv", usecols=["url", "category"])
unique_categories = sorted(url_df["category"].dropna().unique().tolist())
cat_to_idx = {c: i for i, c in enumerate(unique_categories)}


def extract_hostname(u):
    if u.startswith("http://"):
        u = u[7:]
    elif u.startswith("https://"):
        u = u[8:]
    slash_pos = u.find("/")
    if slash_pos != -1:
        u = u[:slash_pos]
    colon_pos = u.find(":")
    if colon_pos != -1:
        u = u[:colon_pos]
    if u.startswith("www."):
        u = u[4:]
    return u.lower()


domain_to_category = {}
for u, c in zip(url_df["url"].astype(str), url_df["category"].astype(str)):
    if c in cat_to_idx:
        h = extract_hostname(u)
        if h and h not in domain_to_category:
            domain_to_category[h] = cat_to_idx[c]

del url_df


KNOWN_MULTI_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk", "com.au", "net.au",
    "org.au", "edu.au", "gov.au", "co.nz", "org.nz", "net.nz", "govt.nz",
    "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp", "com.br", "org.br", "net.br",
    "gov.br", "edu.br", "com.mx", "org.mx", "gob.mx", "edu.mx", "co.in", "net.in",
    "org.in", "gen.in", "co.za", "org.za", "net.za", "gov.za", "com.tr", "org.tr",
    "edu.tr", "gov.tr", "com.ru", "net.ru", "org.ru", "spb.ru", "msk.ru", "co.il",
    "org.il", "co.kr", "com.tw", "com.sg", "com.ar", "com.pl", "com.ua", "co.id",
    "com.my", "com.ph", "co.th", "com.vn", "com.ng", "com.pk", "com.co",
}
SECOND_LEVEL_PREFIXES = {
    "co", "com", "org", "net", "edu", "gov", "ac", "go", "or", "ne", "gob", "govt", "gen"
}


def get_tlds(d):
    if not isinstance(d, str) or "." not in d:
        return "", ""
    parts = d.lower().split(".")
    base_tld = parts[-1]
    if len(parts) >= 3:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in KNOWN_MULTI_TLDS or (len(parts[-1]) == 2 and parts[-2] in SECOND_LEVEL_PREFIXES):
            return two_part, base_tld
    return base_tld, base_tld


FUNCTIONAL_KEYWORDS = [
    ("is_shop", ("shop", "store", "buy", "cart", "market")),
    ("is_news", ("news", "press", "media", "daily", "times")),
    ("is_blog", ("blog", "wp-", "wordpress")),
    ("is_forum", ("forum", "board", "community", "discuss")),
    ("is_gov_edu", ("gov", "edu", "school", "univ")),
    ("is_tech", ("tech", "dev", "app", "cloud", "api")),
    ("is_media", ("video", "tv", "stream", "movie", "radio")),
    ("is_game", ("game", "play", "casino", "bet")),
]


def extract_sld(d):
    if not isinstance(d, str) or "." not in d:
        return d if isinstance(d, str) else ""
    parts = d.lower().split(".")
    if len(parts) >= 3:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in KNOWN_MULTI_TLDS or (len(parts[-1]) == 2 and parts[-2] in SECOND_LEVEL_PREFIXES):
            return parts[-3] if len(parts) >= 3 else parts[0]
    return parts[-2] if len(parts) >= 2 else parts[0]


def compute_sld_morphology(sld):
    if not sld:
        return [0.0, 0.0, 0.0, 0.0]
    s_len = float(len(sld))
    vowels = sum(c in "aeiouy" for c in sld)
    consonants = sum(c.isalpha() and c not in "aeiouy" for c in sld)
    ratio = float(consonants) / float(max(1, vowels))
    has_digit = 1.0 if any(c.isdigit() for c in sld) else 0.0
    counts = {}
    for c in sld:
        counts[c] = counts.get(c, 0) + 1
    entropy = -sum((cnt / s_len) * math.log2(cnt / s_len) for cnt in counts.values())
    return [s_len, ratio, entropy, has_digit]


def compute_lexical_features(domain_str):
    if not isinstance(domain_str, str) or len(domain_str) == 0:
        return [0.0] * 22
    d_lower = domain_str.lower()
    length = len(d_lower)
    dots = d_lower.count(".")
    hyphens = d_lower.count("-")
    digits = sum(c.isdigit() for c in d_lower)
    vowels = sum(c in "aeiouy" for c in d_lower)
    counts = {}
    for c in d_lower:
        counts[c] = counts.get(c, 0) + 1
    entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())

    base_feats = [
        float(length),
        float(dots),
        float(hyphens),
        float(digits),
        float(digits / max(1, length)),
        float(vowels / max(1, length)),
        float(entropy),
        1.0 if dots > 1 else 0.0,
        1.0 if d_lower.startswith("xn--") else 0.0,
        1.0 if any(c.isdigit() for c in d_lower) else 0.0,
    ]

    kw_feats = [
        1.0 if any(kw in d_lower for kw in kw_list) else 0.0
        for _, kw_list in FUNCTIONAL_KEYWORDS
    ]

    sld = extract_sld(d_lower)
    sld_feats = compute_sld_morphology(sld)

    return base_feats + kw_feats + sld_feats


# =============================================================================
# 6. Extract Multi-Modal Features
# =============================================================================
def count_subdomains(domain_str):
    if not isinstance(domain_str, str) or "." not in domain_str:
        return 0.0
    parts = domain_str.lower().split(".")
    two_part, _ = get_tlds(domain_str)
    tld_parts = len(two_part.split(".")) if two_part else 1
    return float(max(0, len(parts) - 1 - tld_parts))


def build_split_raw_features(dom_ids):
    indices = np.array([needed_id_to_idx[d] for d in dom_ids], dtype=np.int32)
    names = domain_names[indices]

    lexical_list = [compute_lexical_features(name) for name in names]
    lexical_arr = np.array(lexical_list, dtype=np.float32)

    deg_in = in_degree[indices].astype(np.float32)
    deg_out = out_degree[indices].astype(np.float32)
    log_in = np.log1p(deg_in)
    log_out = np.log1p(deg_out)
    deg_ratio = (log_in + 1.0) / (log_out + 1.0)

    dir_tr_counts = np.log1p(direct_tracker_counts[indices].astype(np.float32))

    tld_pairs = [get_tlds(name) for name in names]
    multi_tlds = [t[0] for t in tld_pairs]
    base_tlds = [t[1] for t in tld_pairs]

    press_scores = np.array(
        [press_map.get(m, press_map.get(b, median_press)) for m, b in zip(multi_tlds, base_tlds)],
        dtype=np.float32,
    )
    press_known = np.array(
        [1.0 if (m in press_map or b in press_map) else 0.0 for m, b in zip(multi_tlds, base_tlds)],
        dtype=np.float32,
    )

    # Bank connectivity features and subdomain count
    d_out_b = deg_out_bank[indices].astype(np.float32)
    d_in_b = deg_in_bank[indices].astype(np.float32)
    log_out_bank = np.log1p(d_out_b)
    log_in_bank = np.log1p(d_in_b)
    has_out_bank = (d_out_b > 0).astype(np.float32)
    has_in_bank = (d_in_b > 0).astype(np.float32)
    num_subdomains = np.array([count_subdomains(name) for name in names], dtype=np.float32)

    scalars = np.column_stack(
        [
            lexical_arr,
            log_in,
            log_out,
            deg_ratio,
            dir_tr_counts,
            press_scores,
            press_known,
            log_out_bank,
            log_in_bank,
            has_out_bank,
            has_in_bank,
            num_subdomains,
        ]
    )

    raw_direct = direct_tracker_matrix[indices].astype(np.float32)
    direct_tr_links = np.log1p(raw_direct)
    direct_sums = raw_direct.sum(axis=1, keepdims=True)
    direct_norm = raw_direct / np.maximum(direct_sums, 1.0)
    direct_diffused = np.matmul(direct_norm, cooccur_matrix) * (
        direct_sums > 0
    ).astype(np.float32)

    out_smooth = out_neighbor_smooth[indices]
    out_lift = out_neighbor_lift[indices]
    out_cnt = out_neighbor_log_cnt[indices]
    in_smooth = in_neighbor_smooth[indices]
    in_lift = in_neighbor_lift[indices]
    in_cnt = in_neighbor_log_cnt[indices]
    bidirectional_smooth = np.sqrt(np.maximum(out_smooth * in_smooth, 0.0)).astype(
        np.float32
    )

    cat_feats = np.zeros((len(dom_ids), len(unique_categories) + 1), dtype=np.float32)
    for i, name in enumerate(names):
        cat_idx = domain_to_category.get(name, len(unique_categories))
        cat_feats[i, cat_idx] = 1.0

    return (
        names,
        multi_tlds,
        scalars,
        direct_tr_links,
        direct_diffused,
        out_smooth,
        out_lift,
        out_cnt,
        in_smooth,
        in_lift,
        in_cnt,
        bidirectional_smooth,
        cat_feats,
    )


(
    train_names,
    train_tlds,
    train_scalars,
    train_direct,
    train_direct_diffused,
    train_out_smooth,
    train_out_lift,
    train_out_cnt,
    train_in_smooth,
    train_in_lift,
    train_in_cnt,
    train_bidir_smooth,
    train_cats,
) = build_split_raw_features(train_domain_ids)

(
    val_names,
    val_tlds,
    val_scalars,
    val_direct,
    val_direct_diffused,
    val_out_smooth,
    val_out_lift,
    val_out_cnt,
    val_in_smooth,
    val_in_lift,
    val_in_cnt,
    val_bidir_smooth,
    val_cats,
) = build_split_raw_features(val_domain_ids)

(
    test_names,
    test_tlds,
    test_scalars,
    test_direct,
    test_direct_diffused,
    test_out_smooth,
    test_out_lift,
    test_out_cnt,
    test_in_smooth,
    test_in_lift,
    test_in_cnt,
    test_bidir_smooth,
    test_cats,
) = build_split_raw_features(test_domain_ids)

# Standardize tabular scalars strictly on train
scaler = StandardScaler()
train_scalars_scaled = scaler.fit_transform(train_scalars).astype(np.float32)
val_scalars_scaled = scaler.transform(val_scalars).astype(np.float32)
test_scalars_scaled = scaler.transform(test_scalars).astype(np.float32)

# =============================================================================
# 7. TLD Categorical Encoding and Bayesian-Smoothed Tracker Prior (Train-Fit)
# =============================================================================
top_tlds = [t for t, _ in Counter(train_tlds).most_common(128)]
tld_to_idx = {t: i for i, t in enumerate(top_tlds)}
tld_train_counts = Counter(train_tlds)
tld_freq_map = {t: np.log1p(cnt) for t, cnt in tld_train_counts.items()}


def encode_tlds(tld_list):
    arr = np.zeros((len(tld_list), len(top_tlds) + 2), dtype=np.float32)
    for i, t in enumerate(tld_list):
        if t in tld_to_idx:
            arr[i, tld_to_idx[t]] = 1.0
        else:
            arr[i, len(top_tlds)] = 1.0
        arr[i, -1] = tld_freq_map.get(t, 0.0)
    return arr


train_tld_encoded = encode_tlds(train_tlds)
val_tld_encoded = encode_tlds(val_tlds)
test_tld_encoded = encode_tlds(test_tlds)

tld_tracker_sums = {}
for i, t in enumerate(train_tlds):
    if t not in tld_tracker_sums:
        tld_tracker_sums[t] = np.zeros(num_trackers, dtype=np.float32)
    tld_tracker_sums[t] += y_train[i]

tld_smoothed_prior = {}
m_smoothing = 20.0
for t, s_arr in tld_tracker_sums.items():
    n_cnt = float(tld_train_counts[t])
    tld_smoothed_prior[t] = (s_arr + m_smoothing * global_prior) / (n_cnt + m_smoothing)


def get_tld_priors(tld_list):
    arr = np.zeros((len(tld_list), num_trackers), dtype=np.float32)
    for i, t in enumerate(tld_list):
        arr[i] = tld_smoothed_prior.get(t, global_prior)
    return arr


train_priors = get_tld_priors(train_tlds)
val_priors = get_tld_priors(val_tlds)
test_priors = get_tld_priors(test_tlds)

# =============================================================================
# 8. Subword Character N-Gram TF-IDF Features (Train-Fit)
# =============================================================================
clean_train_names = [
    n if isinstance(n, str) and len(n) > 0 else "unknown" for n in train_names
]
clean_val_names = [
    n if isinstance(n, str) and len(n) > 0 else "unknown" for n in val_names
]
clean_test_names = [
    n if isinstance(n, str) and len(n) > 0 else "unknown" for n in test_names
]

dim_word = 384
dim_char = 512
dim_tfidf = dim_word + dim_char

tfidf_word = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)[a-zA-Z0-9]+",
    min_df=5,
    max_features=dim_word,
    sublinear_tf=True,
)
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=10,
    max_features=dim_char,
    sublinear_tf=True,
)

tfidf_word.fit(clean_train_names)
tfidf_char.fit(clean_train_names)

X_tfidf_train = np.hstack(
    [
        tfidf_word.transform(clean_train_names).astype(np.float32).toarray(),
        tfidf_char.transform(clean_train_names).astype(np.float32).toarray(),
    ]
)
X_tfidf_val = np.hstack(
    [
        tfidf_word.transform(clean_val_names).astype(np.float32).toarray(),
        tfidf_char.transform(clean_val_names).astype(np.float32).toarray(),
    ]
)
X_tfidf_test = np.hstack(
    [
        tfidf_word.transform(clean_test_names).astype(np.float32).toarray(),
        tfidf_char.transform(clean_test_names).astype(np.float32).toarray(),
    ]
)

# =============================================================================
# 9. Assemble Multi-Channel Tracker Evidence Tensors and Domain Features
# =============================================================================
# 10-channel tracker evidence: [direct, direct_diffused, out_smooth, out_lift, out_cnt, in_smooth, in_lift, in_cnt, bidirectional_smooth, priors]
evidence_train = np.stack(
    [
        train_direct,
        train_direct_diffused,
        train_out_smooth,
        train_out_lift,
        train_out_cnt,
        train_in_smooth,
        train_in_lift,
        train_in_cnt,
        train_bidir_smooth,
        train_priors,
    ],
    axis=-1,
).astype(np.float32)

evidence_val = np.stack(
    [
        val_direct,
        val_direct_diffused,
        val_out_smooth,
        val_out_lift,
        val_out_cnt,
        val_in_smooth,
        val_in_lift,
        val_in_cnt,
        val_bidir_smooth,
        val_priors,
    ],
    axis=-1,
).astype(np.float32)

evidence_test = np.stack(
    [
        test_direct,
        test_direct_diffused,
        test_out_smooth,
        test_out_lift,
        test_out_cnt,
        test_in_smooth,
        test_in_lift,
        test_in_cnt,
        test_bidir_smooth,
        test_priors,
    ],
    axis=-1,
).astype(np.float32)

X_train = np.hstack(
    [
        train_scalars_scaled,
        train_tld_encoded,
        train_cats,
        X_tfidf_train,
    ]
).astype(np.float32)

X_val = np.hstack(
    [
        val_scalars_scaled,
        val_tld_encoded,
        val_cats,
        X_tfidf_val,
    ]
).astype(np.float32)

X_test = np.hstack(
    [
        test_scalars_scaled,
        test_tld_encoded,
        test_cats,
        X_tfidf_test,
    ]
).astype(np.float32)

assert (
    X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
), "Feature dimension mismatch!"
assert not np.isnan(X_train).any(), "NaN found in X_train!"
assert not np.isnan(X_val).any(), "NaN found in X_val!"
assert not np.isnan(X_test).any(), "NaN found in X_test!"

num_features = X_train.shape[1]
dim_struct = train_scalars_scaled.shape[1] + train_tld_encoded.shape[1] + train_cats.shape[1]
num_evidence_channels = 10


# =============================================================================
# 10. Neural Architecture: Unbottlenecked Tracker Evidence Network
# =============================================================================
class SwishGLU(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        out = val * F.silu(gate)
        return self.dropout(self.norm(out))


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TrackerGraphAttention(nn.Module):

    def __init__(self, embed_dim=96, num_heads=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, H, neighbor_idx, neighbor_ppmi=None):
        B, N, D = H.shape
        Q = self.q_proj(H).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(H).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(H).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        K_nbr = K[:, :, neighbor_idx]  # (B, H, N, K_nbr, head_dim)
        V_nbr = V[:, :, neighbor_idx]  # (B, H, N, K_nbr, head_dim)

        scores = (Q.unsqueeze(-2) * K_nbr).sum(dim=-1) * self.scale
        if neighbor_ppmi is not None:
            scores = scores + self.bias_scale * neighbor_ppmi.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * V_nbr).sum(dim=-2)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return self.norm(H + self.dropout(out))


class UnbottleneckedTrackerEvidenceNet(nn.Module):

    def __init__(
        self,
        num_features=1075,
        num_trackers=355,
        dim_struct=179,
        dim_tfidf=896,
        num_evidence_channels=10,
        dim_meta=None,
        num_tracker_cats=None,
        embed_dim=96,
        hidden_dim=384,
        num_heads=4,
        dropout=0.2,
        cooccur_matrix=None,
        tracker_meta_matrix=None,
        neighbor_indices=None,
        neighbor_ppmi_weights=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_trackers = num_trackers
        self.dim_struct = dim_struct
        self.dim_tfidf = dim_tfidf
        self.num_evidence_channels = num_evidence_channels
        self.embed_dim = embed_dim
        self.num_tracker_cats = num_tracker_cats

        self.struct_proj = nn.Sequential(
            nn.Linear(dim_struct, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
        )

        self.tfidf_proj = nn.Sequential(
            nn.Linear(dim_tfidf, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
        )

        self.fusion_stem = SwishGLU(192 + 192, hidden_dim, dropout=dropout)
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)

        # Domain-Conditioned Dynamic Gating with Bank Connectivity Scalars (5 scalars: log_out_bank, log_in_bank, has_out, has_in, num_subdomains)
        self.bank_scalar_slice = slice(28, 33)
        dim_bank_scalars = 5
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim + dim_bank_scalars, 64),
            nn.SiLU(),
            nn.Linear(64, num_evidence_channels),
        )

        # Unbottlenecked Pointwise Evidence MLP
        self.evidence_mlp = nn.Sequential(
            nn.Linear(num_evidence_channels, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.per_tracker_weights = nn.Parameter(
            torch.ones(num_trackers, num_evidence_channels) * 0.5
        )
        self.evidence_bias = nn.Parameter(torch.zeros(num_trackers))

        # Stream 1: Direct Full-Rank Residual Head
        self.direct_head = nn.Linear(hidden_dim, num_trackers)

        # Stream 2: Dynamic Domain-Conditioned Tracker Graph Attention (DC-TGAT)
        self.domain_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.evidence_gcn_proj = nn.Linear(num_evidence_channels, embed_dim)

        if tracker_meta_matrix is not None:
            self.register_buffer(
                "tracker_meta", torch.tensor(tracker_meta_matrix, dtype=torch.float32)
            )
            self.meta_proj = nn.Sequential(
                nn.Linear(tracker_meta_matrix.shape[1], embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.tracker_meta = None
            self.meta_proj = None

        self.tracker_base_embed = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * 0.02
        )

        if neighbor_indices is not None:
            self.register_buffer(
                "neighbor_idx", torch.tensor(neighbor_indices, dtype=torch.long)
            )
        else:
            dummy_idx = torch.arange(num_trackers).unsqueeze(1).repeat(1, 13)
            self.register_buffer("neighbor_idx", dummy_idx)

        if neighbor_ppmi_weights is not None:
            self.register_buffer(
                "neighbor_ppmi", torch.tensor(neighbor_ppmi_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "neighbor_ppmi", torch.ones(num_trackers, 13, dtype=torch.float32)
            )

        self.gat1 = TrackerGraphAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.gat2 = TrackerGraphAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # Tracker logit projection
        self.gcn_head = nn.Linear(embed_dim, 1)

        # Auxiliary tracker-category prediction head
        if num_tracker_cats is not None:
            self.aux_cat_head = nn.Linear(hidden_dim, num_tracker_cats)
        else:
            self.aux_cat_head = None

    def forward(self, x, tracker_evidence=None, return_aux=None):
        if return_aux is None:
            return_aux = self.training

        s = self.dim_struct
        x_struct = x[:, :s]
        x_tfidf = x[:, s : s + self.dim_tfidf]

        h_struct = self.struct_proj(x_struct)
        h_tfidf = self.tfidf_proj(x_tfidf)
        h_fused = self.fusion_stem(torch.cat([h_struct, h_tfidf], dim=-1))
        h_latent = self.res2(self.res1(h_fused))

        # Stream 1: Direct Full-Rank Projection
        logits_full_rank = self.direct_head(h_latent)

        # Stochastic Evidence Channel Dropout (probability 0.15 during training)
        if tracker_evidence is not None:
            if self.training:
                drop_mask = (
                    torch.rand(
                        tracker_evidence.size(0),
                        1,
                        self.num_evidence_channels,
                        device=tracker_evidence.device,
                    )
                    >= 0.15
                ).float() / 0.85
                reg_evidence = tracker_evidence * drop_mask
            else:
                reg_evidence = tracker_evidence
            ev_gcn = self.evidence_gcn_proj(reg_evidence)
        else:
            reg_evidence = None
            ev_gcn = 0.0

        # Stream 2: Dynamic Domain-Conditioned Tracker Graph Attention Network (DC-TGAT)
        if self.meta_proj is not None and self.tracker_meta is not None:
            t_base = self.meta_proj(self.tracker_meta) + self.tracker_base_embed
        else:
            t_base = self.tracker_base_embed

        h_domain = self.domain_proj(h_latent).unsqueeze(1)
        H = t_base.unsqueeze(0) + h_domain + ev_gcn

        H = self.gat1(H, self.neighbor_idx, self.neighbor_ppmi)
        H = self.gat2(H, self.neighbor_idx, self.neighbor_ppmi)

        logits_gcn = self.gcn_head(H).squeeze(-1)

        # Stream 3: Unbottlenecked Dynamically Gated Tracker Evidence Conditioned on Bank Scalars
        if reg_evidence is not None:
            bank_scalars = x[:, self.bank_scalar_slice]
            gate_input = torch.cat([h_latent, bank_scalars], dim=-1)
            gate_weights = torch.sigmoid(self.channel_gate(gate_input)) * 2.0
            gated_evidence = reg_evidence * gate_weights.unsqueeze(1)
            shared_ev = self.evidence_mlp(gated_evidence).squeeze(-1)
            tracker_ev = (
                gated_evidence * self.per_tracker_weights.unsqueeze(0)
            ).sum(dim=-1)
            logits_evidence = shared_ev + tracker_ev + self.evidence_bias
        else:
            logits_evidence = 0.0

        logits = logits_full_rank + logits_gcn + logits_evidence

        if return_aux and self.aux_cat_head is not None:
            aux_cat_logits = self.aux_cat_head(h_latent)
            return logits, aux_cat_logits
        return logits


TrackerCoOccurNet = UnbottleneckedTrackerEvidenceNet


# =============================================================================
# 11. Objective Function: Adaptive Top-10 Exclusion Boundary Margin Loss
# =============================================================================
class AdaptiveTop10BoundaryMarginLoss(nn.Module):

    def __init__(
        self,
        margin=0.6,
        temp_list=1.0,
        temp_margin=1.0,
        alpha_listnet=0.5,
        alpha_margin=1.0,
        alpha_aux=0.2,
    ):
        super().__init__()
        self.margin = margin
        self.temp_list = temp_list
        self.temp_margin = temp_margin
        self.alpha_listnet = alpha_listnet
        self.alpha_margin = alpha_margin
        self.alpha_aux = alpha_aux

    def forward(self, logits, targets, aux_cat_logits=None, aux_cat_targets=None):
        batch_size, num_tr = logits.shape
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        pos_counts = torch.clamp(targets.sum(dim=-1, keepdim=True), min=1.0)
        has_positives = pos_mask.any(dim=-1)

        # 1. Globally Calibrated Probability Cross-Entropy / ListNet Loss
        target_dist = targets / pos_counts
        log_preds = F.log_softmax(logits / self.temp_list, dim=-1)
        domain_listnet = -(target_dist * log_preds).sum(dim=-1)
        loss_listnet = (
            domain_listnet[has_positives].mean()
            if has_positives.any()
            else torch.tensor(0.0, device=logits.device)
        )

        # 2. Exact Adaptive Top-10 Exclusion Boundary Margin Loss
        # Positive trackers falling below rank 10 exclusion threshold k* = clamp(10 - |P_i| + 1, 1, 10) are penalized
        if not has_positives.any():
            loss_margin = torch.tensor(0.0, device=logits.device)
        else:
            neg_logits = torch.where(
                neg_mask, logits, torch.tensor(-1e9, device=logits.device)
            )
            top_neg_logits, _ = torch.topk(neg_logits, k=10, dim=-1)  # (B, 10)

            # k* (1-indexed) = clamp(10 - |P_i| + 1, 1, 10) -> 0-indexed column is clamp(10 - |P_i|, 0, 9)
            num_pos = targets.sum(dim=-1, keepdim=True).long()
            k_star_idx = torch.clamp(10 - num_pos, min=0, max=9)  # (B, 1)
            s_neg_boundary = torch.gather(top_neg_logits, dim=1, index=k_star_idx)  # (B, 1)

            # Violation: softplus((s_neg_boundary - s_pos + margin) / temp)
            diff = (s_neg_boundary - logits + self.margin) / self.temp_margin  # (B, num_tr)
            violations = F.softplus(diff)

            # Domain margin loss averaged over positive trackers
            domain_margin = (
                (violations * pos_mask.float()).sum(dim=-1, keepdim=True)
                / pos_counts
            ).squeeze(-1)
            loss_margin = domain_margin[has_positives].mean()

        total_loss = self.alpha_listnet * loss_listnet + self.alpha_margin * loss_margin

        # 3. Auxiliary Tracker Category Regularization
        if aux_cat_logits is not None and aux_cat_targets is not None:
            aux_loss = F.binary_cross_entropy_with_logits(
                aux_cat_logits, aux_cat_targets
            )
            total_loss = total_loss + self.alpha_aux * aux_loss

        return total_loss


GloballyCalibratedListwiseLoss = AdaptiveTop10BoundaryMarginLoss
SmoothListwiseTopKRankingLoss = AdaptiveTop10BoundaryMarginLoss
Top10BoundaryMarginLoss = AdaptiveTop10BoundaryMarginLoss
AdaptiveTop10MarginLoss = AdaptiveTop10BoundaryMarginLoss


# =============================================================================
# 12. Training Pipeline & Exact Metric Evaluation
# =============================================================================
model = UnbottleneckedTrackerEvidenceNet(
    num_features=num_features,
    num_trackers=num_trackers,
    dim_struct=dim_struct,
    dim_tfidf=dim_tfidf,
    num_evidence_channels=num_evidence_channels,
    dim_meta=dim_meta,
    num_tracker_cats=num_tracker_cats,
    embed_dim=96,
    hidden_dim=384,
    num_heads=4,
    dropout=0.2,
    cooccur_matrix=cooccur_matrix,
    tracker_meta_matrix=tracker_meta_matrix,
    neighbor_indices=neighbor_indices,
    neighbor_ppmi_weights=neighbor_ppmi_weights,
).to(device)

criterion = AdaptiveTop10BoundaryMarginLoss(
    margin=0.6,
    temp_list=1.0,
    temp_margin=1.0,
    alpha_listnet=0.5,
    alpha_margin=1.0,
    alpha_aux=0.2,
).to(device)

evidence_params = [
    *model.evidence_mlp.parameters(),
    model.per_tracker_weights,
    model.evidence_bias,
    *model.channel_gate.parameters(),
]
evidence_ids = {id(p) for p in evidence_params}
base_params = [p for p in model.parameters() if id(p) not in evidence_ids]

optimizer = AdamW(
    [
        {"params": base_params, "lr": 1e-3, "weight_decay": 2e-4},
        {"params": evidence_params, "lr": 5e-4, "weight_decay": 1e-5},
    ],
    betas=(0.9, 0.99),
    eps=1e-8,
)

num_epochs = 10
warmup_epochs = 2
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epochs,
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=8,
    eta_min=1e-5,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

batch_size = 2048
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(evidence_train),
    torch.from_numpy(y_train),
    torch.from_numpy(y_train_cat),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)


def evaluate_recall_at_10(
    eval_model, X_eval, y_eval, eval_batch_size=2048, evidence_eval=None
):
    eval_model.eval()
    all_recalls = []
    num_samples = len(X_eval)

    with torch.no_grad():
        for i in range(0, num_samples, eval_batch_size):
            batch_x = torch.from_numpy(X_eval[i : i + eval_batch_size]).to(device)
            batch_y = torch.from_numpy(y_eval[i : i + eval_batch_size]).to(device)

            if evidence_eval is not None:
                batch_ev = torch.from_numpy(
                    evidence_eval[i : i + eval_batch_size]
                ).to(device)
                logits = eval_model(batch_x, batch_ev)
            else:
                logits = eval_model(batch_x)

            top10_indices = torch.topk(logits, k=10, dim=-1).indices

            hits = torch.gather(batch_y, dim=1, index=top10_indices).sum(dim=1)
            num_positives = torch.clamp(batch_y.sum(dim=1), min=1.0)
            recall = hits / num_positives
            all_recalls.append(recall.cpu().numpy())

    return float(np.mean(np.concatenate(all_recalls)))


best_val_recall = -1.0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0

    for batch_x, batch_ev, batch_y, batch_y_cat in train_loader:
        batch_x = batch_x.to(device)
        batch_ev = batch_ev.to(device)
        batch_y = batch_y.to(device)
        batch_y_cat = batch_y_cat.to(device)

        optimizer.zero_grad()
        logits, aux_logits = model(batch_x, batch_ev, return_aux=True)
        loss = criterion(logits, batch_y, aux_logits, batch_y_cat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

    scheduler.step()
    epoch_loss = running_loss / max(1, batch_count)
    val_recall = evaluate_recall_at_10(model, X_val, y_val, evidence_eval=evidence_val)

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_val_recall:.5f}"
    )

if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

final_val_score = evaluate_recall_at_10(
    model, X_val, y_val, evidence_eval=evidence_val
)

# =============================================================================
# 13. Test Inference & Submission Generation
# =============================================================================
model.eval()
test_batch_size = 2048
test_predictions = []

with torch.no_grad():
    for i in range(0, len(X_test), test_batch_size):
        batch_x = torch.from_numpy(X_test[i : i + test_batch_size]).to(device)
        batch_ev = torch.from_numpy(
            evidence_test[i : i + test_batch_size]
        ).to(device)
        logits = model(batch_x, batch_ev)
        top10 = torch.topk(logits, k=10, dim=-1).indices.cpu().numpy()
        test_predictions.append(top10)

test_top10_tracker_ids = np.concatenate(test_predictions, axis=0)

tracker_id_to_tracking_domain = np.zeros(num_trackers, dtype=np.int64)
for _, row in trackers_df.iterrows():
    t_id = int(row["tracker_id"])
    t_dom_id = int(row["tracking_domain_id"])
    if 0 <= t_id < num_trackers:
        tracker_id_to_tracking_domain[t_id] = t_dom_id

test_tracking_domain_ids = tracker_id_to_tracking_domain[test_top10_tracker_ids]

submission_domain_ids = np.repeat(test_domain_ids, 10)
submission_tracking_ids = test_tracking_domain_ids.reshape(-1)

submission_df = pd.DataFrame(
    {
        "domain_id": submission_domain_ids,
        "tracking_domain_id": submission_tracking_ids,
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, sep="\t", index=False)
submission_df.to_csv("./submission/submission.tsv", sep="\t", index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert (
    len(submission_df) == len(test_domain_ids) * 10
), "Row count mismatch in submission!"
assert list(submission_df.columns) == [
    "domain_id",
    "tracking_domain_id",
], "Incorrect column names in submission!"
assert not submission_df.isnull().any().any(), "Found null values in predictions!"

print(f"Final Validation Score: {final_val_score}")

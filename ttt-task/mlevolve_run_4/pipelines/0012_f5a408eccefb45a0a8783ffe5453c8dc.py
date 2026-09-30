import collections
import gc
import json
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Fast streaming of link-graph.parquet for 355 direct tracker links using C++ PyArrow filtering & vectorized arrays
tracker_out_degree = collections.defaultdict(int)
all_link_src = []
all_link_trk = []

max_trk_domain_id = max(tracker_domains_set) + 1
trk_id_lookup_arr = np.full(max_trk_domain_id, -1, dtype=np.int16)
for tid, tdid in tracker_id_to_tracking_domain.items():
    if tdid < max_trk_domain_id:
        trk_id_lookup_arr[tdid] = int(tid)

pa_tracker_domains = pa.array(list(tracker_domains_set), type=pa.int64())
link_file = pq.ParquetFile("input/link-graph.parquet")
for batch in link_file.iter_batches(
    batch_size=8000000, columns=["source_domain_id", "target_domain_id"]
):
    tgt = batch["target_domain_id"]
    tracker_mask = pc.is_in(tgt, value_set=pa_tracker_domains)
    if pc.any(tracker_mask).as_py():
        sub_batch = batch.filter(tracker_mask)
        src_arr = sub_batch["source_domain_id"].to_numpy()
        tgt_arr = sub_batch["target_domain_id"].to_numpy()
        valid_tgt = (tgt_arr >= 0) & (tgt_arr < max_trk_domain_id)
        s_filt = src_arr[valid_tgt]
        t_filt = trk_id_lookup_arr[tgt_arr[valid_tgt]]
        ok = t_filt >= 0
        all_link_src.append(s_filt[ok])
        all_link_trk.append(t_filt[ok])

del link_file, pa_tracker_domains, trk_id_lookup_arr
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

# Stream domain hostnames lookup efficiently for needed domains only
domain_lookup = {}
domains_file = pq.ParquetFile("input/domains.parquet")
for batch in domains_file.iter_batches(
    batch_size=2000000, columns=["domain_id", "domain"]
):
    d_ids = batch["domain_id"].to_numpy()
    needed_mask = np.fromiter(
        (did in all_needed_ids for did in d_ids), dtype=bool, count=len(d_ids)
    )
    if needed_mask.any():
        matched_ids = d_ids[needed_mask]
        matched_domains = batch["domain"].filter(pa.array(needed_mask)).to_pylist()
        for did, dom in zip(matched_ids, matched_domains):
            domain_lookup[int(did)] = str(dom)

del domains_file
gc.collect()

TWO_PART_SECOND_LEVELS = {
    "co", "com", "org", "net", "edu", "gov", "gob", "ac",
    "ne", "or", "go", "gen", "nom", "mil", "asn", "biz", "info", "me",
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


# Construct leave-one-out empirical tracker presence profiles strictly on training domains
train_apexes = [get_apex_domain(domain_lookup.get(did, "")) for did in train_domain_ids]
unique_train_apexes, train_apex_inverse = np.unique(train_apexes, return_inverse=True)
num_train_apexes = len(unique_train_apexes)

apex_counts = np.bincount(train_apex_inverse, minlength=num_train_apexes)
train_domain_cnts = apex_counts[train_apex_inverse]

A_train = sp.csr_matrix(
    (
        np.ones(len(train_domain_ids), dtype=np.float32),
        (train_apex_inverse, np.arange(len(train_domain_ids))),
    ),
    shape=(num_train_apexes, len(train_domain_ids)),
)
apex_sums_dense = (A_train @ Y_train.astype(np.float32)).toarray()

multi_mask = train_domain_cnts > 1
apex_priors_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.float32)

if multi_mask.any():
    multi_cnts = train_domain_cnts[multi_mask].astype(np.float32)[:, None]
    apex_priors_train[multi_mask] = apex_sums_dense[train_apex_inverse[multi_mask]] / (multi_cnts - 1.0)
    sub_edge_mask = train_domain_cnts[train_rows] > 1
    if sub_edge_mask.any():
        sub_rows = train_rows[sub_edge_mask]
        sub_cols = train_cols[sub_edge_mask]
        sub_cnts = (train_domain_cnts[sub_rows] - 1.0).astype(np.float32)
        np.subtract.at(apex_priors_train, (sub_rows, sub_cols), 1.0 / sub_cnts)
    np.clip(apex_priors_train, 0.0, 1.0, out=apex_priors_train)
del train_rows, train_cols, val_rows, val_cols

apex_to_idx = {apex: idx for idx, apex in enumerate(unique_train_apexes)}
nonzero_counts = np.maximum(apex_counts, 1).astype(np.float32)[:, None]
apex_means = (apex_sums_dense / nonzero_counts).astype(np.float32)


def compute_eval_apex_priors(domain_ids):
    priors = np.zeros((len(domain_ids), num_trackers), dtype=np.float32)
    for i, did in enumerate(domain_ids):
        h = domain_lookup.get(did, "")
        apex = get_apex_domain(h)
        idx = apex_to_idx.get(apex)
        if idx is not None:
            priors[i] = apex_means[idx]
    return priors


apex_priors_val = compute_eval_apex_priors(val_domain_ids)
apex_priors_test = compute_eval_apex_priors(test_domain_ids)

del A_train, apex_sums_dense, apex_means
gc.collect()

# Compute smoothed empirical TLD-tracker prior distributions and compress with SVD strictly from train
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

svd_tld = TruncatedSVD(n_components=16, random_state=42)
tld_embeddings = svd_tld.fit_transform(tld_priors_mat).astype(np.float32)
default_tld_emb = svd_tld.transform(tracker_priors[None, :].astype(np.float32))[0]
tld_emb_lookup = {tld: tld_embeddings[i] for i, tld in enumerate(unique_tlds)}

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
active_all = active_hostnames | active_apexes
url_df = url_df[url_df["netloc"].isin(active_all)]

categories = sorted(url_df["category"].dropna().unique())
cat_counts_df = url_df.groupby(["netloc", "category"]).size().unstack(fill_value=0)
cat_totals = cat_counts_df.sum(axis=1)
cat_probs_df = cat_counts_df.div(cat_totals, axis=0)

cat_lookup = {}
for netloc, row in cat_probs_df.iterrows():
    cat_lookup[netloc] = row.values.astype(np.float32)

del url_df, cat_counts_df, cat_probs_df
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

# Fit TLD frequency encoding strictly on training domains
train_tlds = [get_tld(domain_lookup.get(did, "")) for did in train_domain_ids]
tld_counts = pd.Series(train_tlds).value_counts()
top_tlds = list(tld_counts.head(50).index)
tld_to_code = {tld: idx for idx, tld in enumerate(top_tlds)}

# Fit fast 64-feature character n-gram TF-IDF on a representative hostname sample
train_hostnames = [domain_lookup.get(did, "") for did in train_domain_ids]
sample_size = min(100000, len(train_hostnames))
sampled_hostnames = list(np.random.choice(train_hostnames, size=sample_size, replace=False))
tfidf_vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), max_features=64, min_df=5
)
tfidf_vectorizer.fit(sampled_hostnames)

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
    tlds = []

    for i, h in enumerate(hostnames):
        hl = h.lower().strip()
        lengths[i] = len(hl)
        dots[i] = hl.count(".")
        hyphens[i] = hl.count("-")
        d_cnt = sum(c.isdigit() for c in hl)
        v_cnt = sum(c in vowel_set for c in hl)
        digits[i] = d_cnt
        vowels[i] = v_cnt
        starts_www[i] = 1.0 if hl.startswith("www.") else 0.0

        parts = hl.split(".")
        sld_lengths[i] = len(parts[-2]) if len(parts) >= 2 else len(hl)
        subdomain_depths[i] = max(0, len(parts) - 2)
        tlds.append(get_tld(hl))

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

    tld_tracker_feats = np.zeros((n_samples, 16), dtype=np.float32)
    for i, t in enumerate(tlds):
        tld_tracker_feats[i] = tld_emb_lookup.get(t, default_tld_emb)

    log_tracker_out = np.array(
        [np.log1p(tracker_out_degree.get(did, 0)) for did in domain_ids], dtype=np.float32
    )
    num_direct_links = direct_mat.sum(axis=1)[:, None]

    company_link_counts = direct_mat @ tracker_to_company_matrix
    category_link_counts = direct_mat @ tracker_to_category_matrix

    tfidf_mat = tfidf_vectorizer.transform(hostnames).toarray().astype(np.float32)

    feature_blocks = [
        lengths[:, None],
        dots[:, None],
        hyphens[:, None],
        digits[:, None],
        digit_ratios[:, None],
        vowels[:, None],
        vowel_ratios[:, None],
        starts_www[:, None],
        sld_lengths[:, None],
        subdomain_depths[:, None],
        keyword_flags,
        tld_onehot,
        is_cc_tld[:, None],
        press_freedoms[:, None],
        has_press_freedom[:, None],
        tld_tracker_feats,
        cat_features,
        cat_known[:, None],
        log_tracker_out[:, None],
        num_direct_links,
        company_link_counts,
        category_link_counts,
        tfidf_mat,
    ]
    return np.hstack(feature_blocks).astype(np.float32)


# Extract direct link matrices for train, val, test
direct_links_train = extract_direct_link_matrix(train_domain_ids)
direct_links_val = extract_direct_link_matrix(val_domain_ids)
direct_links_test = extract_direct_link_matrix(test_domain_ids)

# Build feature matrices with compact dimensionality
X_train_raw = build_feature_matrix(train_domain_ids, direct_links_train)
X_val_raw = build_feature_matrix(val_domain_ids, direct_links_val)
X_test_raw = build_feature_matrix(test_domain_ids, direct_links_test)

# Fit StandardScaler strictly on X_train (Zero Data Leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)

feature_dim = X_train.shape[1]

del X_train_raw, X_val_raw, X_test_raw
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
# 3. MODEL ARCHITECTURE DESIGN
# =============================================================================


class AsymmetricLoss(nn.Module):
    """Calibrated Asymmetric Loss with negative margin clipping and asymmetric focusing."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        targets = targets.float()

        if self.gamma_pos > 0.0:
            loss_pos = (
                targets
                * torch.pow(1.0 - p, self.gamma_pos)
                * torch.log(p.clamp(min=self.eps))
            )
        else:
            loss_pos = targets * torch.log(p.clamp(min=self.eps))

        p_neg = (p - self.clip).clamp(min=0.0)
        loss_neg = (
            (1.0 - targets)
            * torch.pow(p_neg, self.gamma_neg)
            * torch.log((1.0 - p_neg).clamp(min=self.eps))
        )

        loss = -loss_pos - loss_neg
        return loss.sum(dim=-1).mean()


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

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
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
    """Deep Tabular Residual Network with direct tracker & apex bypass and gated empirical co-occurrence refinement."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 355,
        hidden_dim: int = 384,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
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

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        if init_bias is not None:
            self.head.bias.data.copy_(torch.tensor(init_bias, dtype=torch.float32))

        # Direct tracker bypass initialized with strong positive diagonal weights
        self.direct_bypass = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.zeros_(self.direct_bypass.weight)
        with torch.no_grad():
            self.direct_bypass.weight.fill_diagonal_(5.0)

        # Dedicated Apex Direct Bypass layer initialized with diagonal weights of 4.0
        self.apex_bypass = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.zeros_(self.apex_bypass.weight)
        with torch.no_grad():
            self.apex_bypass.weight.fill_diagonal_(4.0)

        # Normalized empirical conditional co-occurrence matrix buffer
        if cooccur_matrix is not None:
            c_tensor = torch.tensor(cooccur_matrix, dtype=torch.float32)
        else:
            c_tensor = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        self.register_buffer("cooccur_matrix", c_tensor)

        # Positive learnable scale and sigmoid residual gating layer for co-occurrence propagation
        self.cooccur_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.cooccur_gate = nn.Linear(num_classes, num_classes)
        nn.init.zeros_(self.cooccur_gate.weight)
        nn.init.constant_(self.cooccur_gate.bias, -1.0)

    def forward(
        self,
        x: torch.Tensor,
        direct_links: torch.Tensor = None,
        apex_priors: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.stem(x)
        for block in self.res_blocks:
            h = block(h)
        h = self.final_norm(h)
        logits0 = self.head(h)
        if direct_links is not None:
            logits0 = logits0 + self.direct_bypass(direct_links)
        if apex_priors is not None:
            logits0 = logits0 + self.apex_bypass(apex_priors)

        p = torch.sigmoid(logits0)
        scale = F.softplus(self.cooccur_scale)
        propagated = (p @ self.cooccur_matrix) * scale
        gate = torch.sigmoid(self.cooccur_gate(logits0))
        refinement = gate * propagated
        final_logits = logits0 + refinement
        return final_logits


model = TrackerResNet(
    input_dim=feature_dim,
    num_classes=num_trackers,
    hidden_dim=384,
    num_blocks=3,
    dropout_rate=0.25,
    init_bias=prior_logits,
    cooccur_matrix=tracker_cooccur_matrix,
).to(device)

criterion = AsymmetricLoss(
    gamma_neg=4.0, gamma_pos=0.0, clip=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

EPOCHS = 12
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=1
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(EPOCHS - 1, 1), eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[1]
)

# Sanity check forward-backward pass
dummy_input = torch.randn(8, feature_dim, device=device)
dummy_links = torch.zeros(8, num_trackers, device=device)
dummy_apex = torch.zeros(8, num_trackers, device=device)
dummy_target = torch.randint(0, 2, (8, num_trackers), device=device).float()
model.train()
optimizer.zero_grad()
dummy_logits = model(dummy_input, dummy_links, dummy_apex)
dummy_loss = criterion(dummy_logits, dummy_target)
dummy_loss.backward()
optimizer.zero_grad()

# =============================================================================
# 4. TRAINING, VALIDATION & INFERENCE PIPELINE
# =============================================================================

Y_train_dense = Y_train.toarray().astype(np.float32)
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(direct_links_train),
    torch.from_numpy(apex_priors_train),
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
):
    """Computes exact official Recall@10 across all evaluation domains."""
    eval_model.eval()
    n_samples = eval_features.shape[0]
    top10_list = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.from_numpy(eval_features[start_idx:end_idx]).to(
                device
            )
            batch_links = torch.from_numpy(
                eval_direct_links[start_idx:end_idx]
            ).to(device)
            batch_apex = torch.from_numpy(
                eval_apex_priors[start_idx:end_idx]
            ).to(device)
            logits = eval_model(batch_x, batch_links, batch_apex)
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


best_val_recall = -1.0
best_epoch = 0
best_model_path = "./working/tracker_resnet_best.pt"
patience = 4
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_x, batch_links, batch_apex, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_links = batch_links.to(device)
        batch_apex = batch_apex.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, batch_links, batch_apex)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_train_loss / max(num_batches, 1)

    val_recall, _ = compute_recall_at_10(
        model, X_val, direct_links_val, apex_priors_val, Y_val, batch_size=4096
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS:02d} - Train Loss: {avg_train_loss:.4f} - Val Recall@10: {val_recall:.6f}"
    )

    if patience_counter >= patience:
        break

# Load best checkpoint for final evaluation and test inference
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

final_val_score, _ = compute_recall_at_10(
    model, X_val, direct_links_val, apex_priors_val, Y_val, batch_size=4096
)

# Test inference on all 50,000 target domains
n_test = X_test.shape[0]
test_top10_list = []

with torch.no_grad():
    for start_idx in range(0, n_test, 4096):
        end_idx = min(start_idx + 4096, n_test)
        batch_x = torch.from_numpy(X_test[start_idx:end_idx]).to(device)
        batch_links = torch.from_numpy(direct_links_test[start_idx:end_idx]).to(
            device
        )
        batch_apex = torch.from_numpy(apex_priors_test[start_idx:end_idx]).to(
            device
        )
        test_logits = model(batch_x, batch_links, batch_apex)
        test_top10_batch = (
            torch.topk(test_logits, k=10, dim=1).indices.cpu().numpy()
        )
        test_top10_list.append(test_top10_batch)

test_top10_preds = np.vstack(test_top10_list)

# Map predicted compact indices to tracking_domain_id
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

print(f"Final Validation Score: {final_val_score:.6f}")
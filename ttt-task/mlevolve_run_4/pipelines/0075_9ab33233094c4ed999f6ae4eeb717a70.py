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
tracker_meta_df = pd.concat([meta_company, meta_country, meta_category], axis=1)
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
    "--- Step 5: Streaming link-graph.parquet for Graph Features and Direct Tracker Links ---"
)
in_degrees_sorted = np.zeros(num_total, dtype=np.float32)
out_degrees_sorted = np.zeros(num_total, dtype=np.float32)
tracker_direct_sorted = np.zeros((num_total, num_trackers), dtype=np.float32)

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

    batch_count += 1

print(f"Link graph streaming complete across {batch_count} batches.")

inv_sorted_indices = np.empty_like(sorted_all_indices)
inv_sorted_indices[sorted_all_indices] = np.arange(num_total)

in_degrees = in_degrees_sorted[inv_sorted_indices]
out_degrees = out_degrees_sorted[inv_sorted_indices]
tracker_direct_raw = tracker_direct_sorted[inv_sorted_indices]
tracker_direct_all = np.log1p(tracker_direct_raw)

# Calculate link-graph tracker intensity ratios and total tracker hyperlink intensity
raw_tracker_sum = tracker_direct_raw.sum(axis=1, keepdims=True)
tracker_intensity_ratio = raw_tracker_sum / (out_degrees.reshape(-1, 1) + 1.0)
log1p_tracker_sum = np.log1p(raw_tracker_sum)

del in_degrees_sorted, out_degrees_sorted, tracker_direct_sorted, tracker_direct_raw
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
    """Deep Multi-Modal Tracker Architecture with Multi-Head Sub-Space Bilinear Matching,
    taxonomy metadata-anchored tracker query embeddings, and Bounded LayerNorm Bottleneck Synergy.
    """

    def __init__(
        self,
        dense_dim: int,
        tracker_metadata: torch.Tensor,
        num_trackers: int = 355,
        hidden_dim: int = 384,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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

        # 2. Metadata-Anchored Latent Tracker Query Embeddings: E_tracker = W_meta T_meta + Delta E
        self.register_buffer("tracker_metadata", tracker_metadata)
        meta_dim = tracker_metadata.shape[1]
        self.meta_proj = nn.Linear(meta_dim, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.meta_proj.weight, a=math.sqrt(5))

        self.delta_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * 0.02
        )
        self.tracker_bias = nn.Parameter(torch.zeros(num_trackers))

        # Learnable per-tracker multi-head attention weights
        self.head_weights = nn.Parameter(torch.zeros(num_trackers, num_heads))

        # 3. Direct Link and Prior Calibration Weights
        self.direct_weight = nn.Parameter(torch.ones(num_trackers) * 2.5)
        self.direct_bias = nn.Parameter(torch.zeros(num_trackers))

        self.prior_weight = nn.Parameter(torch.ones(num_trackers) * 3.5)
        self.prior_bias = nn.Parameter(torch.zeros(num_trackers))

        # 4. Context-Conditioned Gating Generator
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_trackers * 3),
        )

        # 5. Bounded LayerNorm Bottleneck Synergy Module (355 -> 64 -> 355)
        self.synergy_mlp = nn.Sequential(
            nn.LayerNorm(num_trackers),
            nn.Linear(num_trackers, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_trackers),
        )
        nn.init.zeros_(self.synergy_mlp[-1].weight)
        nn.init.zeros_(self.synergy_mlp[-1].bias)

    def forward(
        self,
        x_dense: torch.Tensor,
        x_direct: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> torch.Tensor:
        h = self.stem(x_dense)
        h = self.res1(h)
        h = self.res2(h)
        domain_emb = self.domain_proj(h)

        # Multi-Head Sub-Space Bilinear Matching
        domain_heads = domain_emb.view(-1, self.num_heads, self.head_dim)
        tracker_emb = self.meta_proj(self.tracker_metadata) + self.delta_embeddings
        tracker_heads = tracker_emb.view(self.num_trackers, self.num_heads, self.head_dim)

        head_scores = torch.einsum("bhd, thd -> bth", domain_heads, tracker_heads) / math.sqrt(self.head_dim)
        head_attn = F.softmax(self.head_weights, dim=-1)
        latent_scores = (head_scores * head_attn.unsqueeze(0)).sum(dim=-1) + self.tracker_bias

        direct_scores = x_direct * self.direct_weight + self.direct_bias

        prior_clamped = torch.clamp(x_prior, 1e-4, 1.0 - 1e-4)
        prior_logits = torch.log(prior_clamped / (1.0 - prior_clamped))
        prior_scores = prior_logits * self.prior_weight + self.prior_bias

        raw_gates = self.gate_net(h).view(-1, self.num_trackers, 3)
        gates = F.softmax(raw_gates, dim=-1)

        fused_scores = (
            gates[:, :, 0] * latent_scores
            + gates[:, :, 1] * direct_scores
            + gates[:, :, 2] * prior_scores
        )

        synergy_refined = fused_scores + 0.1 * self.synergy_mlp(fused_scores)
        return synergy_refined


class AsymmetricRecall10Loss(nn.Module):
    """Combines Asymmetric Focal Loss with a vectorized Top-K Hard-Negative

    Margin Ranking Loss directly optimizing the Recall@10 decision boundary.
    """

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        rank_weight: float = 2.0,
        margin: float = 1.0,
        top_k: int = 15,
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

        # 2. Vectorized Top-K Hard Negative Margin Ranking Component
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e4))
        topk_neg_logits, _ = torch.topk(neg_logits, k=self.top_k, dim=-1)

        diff = topk_neg_logits.unsqueeze(1) - logits.unsqueeze(-1) + self.margin
        violations = F.softplus(diff)

        pos_mask = targets.unsqueeze(-1)
        masked_violations = (violations * pos_mask).sum(dim=-1) / float(self.top_k)

        num_pos = targets.sum(dim=-1).clamp(min=1.0)
        rank_loss = (masked_violations.sum(dim=-1) / num_pos).mean()

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
print(f"Instantiating model on compute device: {device}")

model = GatedTrackerFusionNet(
    dense_dim=dense_feature_dim,
    tracker_metadata=tracker_metadata.to(device),
    num_trackers=num_trackers,
    hidden_dim=384,
    embed_dim=256,
    num_heads=4,
    dropout=0.2,
).to(device)

criterion = AsymmetricRecall10Loss(
    gamma_neg=2.0,
    gamma_pos=0.0,
    clip=0.05,
    rank_weight=2.0,
    margin=1.0,
    top_k=15,
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

# =========================================================================
# STEP 3: TRAINING, VALIDATION & INFERENCE PIPELINE
# =========================================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_dense_train),
    torch.from_numpy(tracker_direct_train),
    torch.from_numpy(tld_prior_train),
    torch.from_numpy(Y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_dense_val),
    torch.from_numpy(tracker_direct_val),
    torch.from_numpy(tld_prior_val),
    torch.from_numpy(Y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_dense_test),
    torch.from_numpy(tracker_direct_test),
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

epochs = 16
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
best_val_recall = -1.0
best_model_path = "./working/best_gated_tracker_fusion_net.pt"


def evaluate(model, loader, device):
    """Evaluates validation loss and exact official Recall@10 metric."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for b_dense, b_direct, b_prior, b_y in loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_direct = b_direct.to(device, non_blocking=True)
            b_prior = b_prior.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            logits = model(b_dense, b_direct, b_prior)
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

    for b_dense, b_direct, b_prior, b_y in train_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_prior = b_prior.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(b_dense, b_direct, b_prior)
        loss = criterion(logits, b_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * len(b_dense)
        num_samples += len(b_dense)

    scheduler.step()
    epoch_train_loss = running_loss / num_samples
    val_loss, val_recall = evaluate(model, val_loader, device)

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), best_model_path)

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
    for b_dense, b_direct, b_prior in test_loader:
        b_dense = b_dense.to(device, non_blocking=True)
        b_direct = b_direct.to(device, non_blocking=True)
        b_prior = b_prior.to(device, non_blocking=True)

        logits = model(b_dense, b_direct, b_prior)
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

submission_df.to_csv(submission_csv_path, sep="\t", index=False)
submission_df.to_csv(submission_tsv_path, sep="\t", index=False)

print(f"Final Validation Score: {final_val_score}")

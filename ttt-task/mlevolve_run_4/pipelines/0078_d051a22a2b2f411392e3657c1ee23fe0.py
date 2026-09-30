import collections
import gc
import json
import os
import numpy as np
import pandas as pd
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

# Hold-out validation split: 25,000 domains; training split: up to 250,000 domains
N_VAL = min(25000, int(len(shuffled_train_domains) * 0.1))
N_TRAIN = min(250000, len(shuffled_train_domains) - N_VAL)

val_domain_ids = shuffled_train_domains[:N_VAL]
train_domain_ids = shuffled_train_domains[N_VAL : N_VAL + N_TRAIN]

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

# Precompute empirical row-normalized tracker co-occurrence transition matrix strictly from Y_train
cooccur_counts = (Y_train.T @ Y_train).toarray().astype(np.float32)
np.fill_diagonal(cooccur_counts, 0.0)
row_sums = cooccur_counts.sum(axis=1, keepdims=True)
P_cooccur = cooccur_counts / (row_sums + 1e-6)
np.save("./working/P_cooccur.npy", P_cooccur)

del train_edges, val_edges, train_graph_df
gc.collect()

# =============================================================================
# 2. FEATURE EXTRACTION & GRAPH STREAMING
# =============================================================================

# Stream link-graph.parquet to compute degree centrality and direct links to all 355 candidate trackers
in_degree = collections.defaultdict(int)
out_degree = collections.defaultdict(int)
tracker_out_degree = collections.defaultdict(int)
direct_tracker_links = collections.defaultdict(set)

link_file = pq.ParquetFile("input/link-graph.parquet")
for batch in link_file.iter_batches(
    batch_size=2000000, columns=["source_domain_id", "target_domain_id"]
):
    src_arr = batch["source_domain_id"].to_numpy()
    tgt_arr = batch["target_domain_id"].to_numpy()

    src_needed_mask = pd.Series(src_arr).isin(all_needed_ids).to_numpy()
    tgt_needed_mask = pd.Series(tgt_arr).isin(all_needed_ids).to_numpy()

    if src_needed_mask.any():
        u_src, c_src = np.unique(src_arr[src_needed_mask], return_counts=True)
        for did, cnt in zip(u_src, c_src):
            out_degree[did] += cnt

    if tgt_needed_mask.any():
        u_tgt, c_tgt = np.unique(tgt_arr[tgt_needed_mask], return_counts=True)
        for did, cnt in zip(u_tgt, c_tgt):
            in_degree[did] += cnt

    tgt_is_tracker = pd.Series(tgt_arr).isin(tracker_domains_set).to_numpy()
    tracker_edges_mask = src_needed_mask & tgt_is_tracker
    if tracker_edges_mask.any():
        src_match = src_arr[tracker_edges_mask]
        tgt_match = tgt_arr[tracker_edges_mask]
        for s, t in zip(src_match, tgt_match):
            tracker_out_degree[s] += 1
            direct_tracker_links[s].add(tracking_domain_to_tracker_id[t])

del link_file
gc.collect()

# Load domain hostnames lookup
domains_table = pq.read_table("input/domains.parquet", columns=["domain_id", "domain"])
domains_df = domains_table.to_pandas()
del domains_table
gc.collect()

domains_df = domains_df[domains_df["domain_id"].isin(all_needed_ids)]
domain_lookup = dict(
    zip(domains_df["domain_id"].astype(int), domains_df["domain"].astype(str))
)
del domains_df
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

# Filter URL classification to active candidate hostnames for speed and efficiency
active_hostnames = {h.lower().strip() for h in domain_lookup.values()}
url_df = url_df[url_df["netloc"].isin(active_hostnames)]

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


def get_tld(hostname):
    if "." not in hostname:
        return "other"
    return hostname.lower().strip().rsplit(".", 1)[-1]


# Fit TLD frequency encoding strictly on training domains
train_tlds = [get_tld(domain_lookup.get(did, "")) for did in train_domain_ids]
tld_counts = pd.Series(train_tlds).value_counts()
top_tlds = list(tld_counts.head(50).index)
tld_to_code = {tld: idx for idx, tld in enumerate(top_tlds)}

# Fit TF-IDF and SVD strictly on training hostnames
train_hostnames = [domain_lookup.get(did, "") for did in train_domain_ids]
tfidf_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 4), max_features=3000, min_df=5
)
train_tfidf = tfidf_vectorizer.fit_transform(train_hostnames)

svd_model = TruncatedSVD(n_components=24, random_state=42)
train_svd = svd_model.fit_transform(train_tfidf)

# Build company-level and category-level aggregation matrices using trackers.tsv metadata
tracker_companies = trackers_df["company"].fillna("#").astype(str).values
tracker_categories = trackers_df["category"].fillna("Other").astype(str).values

unique_companies = sorted(list(set(tracker_companies)))
unique_categories = sorted(list(set(tracker_categories)))

company_to_idx = {c: i for i, c in enumerate(unique_companies)}
category_to_idx = {c: i for i, c in enumerate(unique_categories)}

tracker_company_mat = np.zeros((num_trackers, len(unique_companies)), dtype=np.float32)
tracker_category_mat = np.zeros((num_trackers, len(unique_categories)), dtype=np.float32)

for tid, comp, cat in zip(
    trackers_df["tracker_id"].astype(int).values,
    tracker_companies,
    tracker_categories,
):
    tracker_company_mat[tid, company_to_idx[comp]] = 1.0
    tracker_category_mat[tid, category_to_idx[cat]] = 1.0


def build_direct_tracker_matrix(domain_ids):
    """Constructs 355-dimensional binary direct link array for given domains."""
    n = len(domain_ids)
    mat = np.zeros((n, num_trackers), dtype=np.float32)
    for i, did in enumerate(domain_ids):
        if did in direct_tracker_links:
            for tid in direct_tracker_links[did]:
                mat[i, tid] = 1.0
    return mat


tracker_direct_train = build_direct_tracker_matrix(train_domain_ids)
tracker_direct_val = build_direct_tracker_matrix(val_domain_ids)
tracker_direct_test = build_direct_tracker_matrix(test_domain_ids)

np.save("./working/tracker_direct_train.npy", tracker_direct_train)
np.save("./working/tracker_direct_val.npy", tracker_direct_val)
np.save("./working/tracker_direct_test.npy", tracker_direct_test)

vowel_set = set("aeiou")
keywords = [
    "shop",
    "blog",
    "news",
    "store",
    "media",
    "tv",
    "game",
    "play",
    "tech",
    "app",
    "video",
    "forum",
]


def build_feature_matrix(domain_ids, direct_mat, is_train=False, svd_features=None):
    hostnames = [domain_lookup.get(did, "") for did in domain_ids]
    n_samples = len(domain_ids)

    lengths = np.empty(n_samples, dtype=np.float32)
    dots = np.empty(n_samples, dtype=np.float32)
    hyphens = np.empty(n_samples, dtype=np.float32)
    digits = np.empty(n_samples, dtype=np.float32)
    vowels = np.empty(n_samples, dtype=np.float32)
    starts_www = np.empty(n_samples, dtype=np.float32)
    tlds = []

    for i, h in enumerate(hostnames):
        hl = h.lower().strip()
        lengths[i] = len(hl)
        dots[i] = hl.count(".")
        hyphens[i] = hl.count("-")
        d_cnt = 0
        v_cnt = 0
        for char in hl:
            if char.isdigit():
                d_cnt += 1
            elif char in vowel_set:
                v_cnt += 1
        digits[i] = d_cnt
        vowels[i] = v_cnt
        starts_www[i] = 1.0 if hl.startswith("www.") else 0.0
        dot_pos = hl.rfind(".")
        tlds.append(hl[dot_pos + 1 :] if dot_pos != -1 else "other")

    digit_ratios = digits / np.maximum(lengths, 1.0)
    vowel_ratios = vowels / np.maximum(lengths, 1.0)

    keyword_flags = np.zeros((n_samples, len(keywords)), dtype=np.float32)
    for k_idx, kw in enumerate(keywords):
        keyword_flags[:, k_idx] = [1.0 if kw in h.lower() else 0.0 for h in hostnames]

    tld_onehot = np.zeros((n_samples, len(top_tlds)), dtype=np.float32)
    for i, t in enumerate(tlds):
        if t in tld_to_code:
            tld_onehot[i, tld_to_code[t]] = 1.0

    is_cc_tld = np.array([1.0 if len(t) == 2 else 0.0 for t in tlds], dtype=np.float32)
    press_freedoms = np.array(
        [tld_to_press_freedom.get(t, 40.0) for t in tlds], dtype=np.float32
    )
    has_press_freedom = np.array(
        [1.0 if t in tld_to_press_freedom else 0.0 for t in tlds], dtype=np.float32
    )

    num_cats = len(categories)
    cat_features = np.zeros((n_samples, num_cats), dtype=np.float32)
    cat_known = np.zeros(n_samples, dtype=np.float32)
    for i, h in enumerate(hostnames):
        norm_h = h.lower().strip()
        h_no_www = norm_h[4:] if norm_h.startswith("www.") else norm_h
        if norm_h in cat_lookup:
            cat_features[i] = cat_lookup[norm_h]
            cat_known[i] = 1.0
        elif h_no_www in cat_lookup:
            cat_features[i] = cat_lookup[h_no_www]
            cat_known[i] = 1.0

    log_in_deg = np.array(
        [np.log1p(in_degree[did]) for did in domain_ids], dtype=np.float32
    )
    log_out_deg = np.array(
        [np.log1p(out_degree[did]) for did in domain_ids], dtype=np.float32
    )
    deg_ratio = (log_in_deg + 1.0) / (log_out_deg + 1.0)
    log_tracker_out = np.array(
        [np.log1p(tracker_out_degree[did]) for did in domain_ids], dtype=np.float32
    )

    # Aggregate company-level and category-level link evidence from direct links
    company_feats = np.log1p(direct_mat @ tracker_company_mat)
    category_feats = np.log1p(direct_mat @ tracker_category_mat)

    if svd_features is None:
        tfidf_mat = tfidf_vectorizer.transform(hostnames)
        svd_mat = svd_model.transform(tfidf_mat).astype(np.float32)
    else:
        svd_mat = svd_features.astype(np.float32)

    feature_blocks = [
        lengths[:, None],
        dots[:, None],
        hyphens[:, None],
        digits[:, None],
        digit_ratios[:, None],
        vowels[:, None],
        vowel_ratios[:, None],
        starts_www[:, None],
        keyword_flags,
        tld_onehot,
        is_cc_tld[:, None],
        press_freedoms[:, None],
        has_press_freedom[:, None],
        cat_features,
        cat_known[:, None],
        log_in_deg[:, None],
        log_out_deg[:, None],
        deg_ratio[:, None],
        log_tracker_out[:, None],
        direct_mat,
        company_feats,
        category_feats,
        svd_mat,
    ]
    return np.hstack(feature_blocks).astype(np.float32)


# Build feature matrices
X_train_raw = build_feature_matrix(
    train_domain_ids, tracker_direct_train, is_train=True, svd_features=train_svd
)
X_val_raw = build_feature_matrix(
    val_domain_ids, tracker_direct_val, is_train=False
)
X_test_raw = build_feature_matrix(
    test_domain_ids, tracker_direct_test, is_train=False
)

# Fit StandardScaler strictly on X_train (Zero Data Leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)

feature_dim = X_train.shape[1]

del X_train_raw, X_val_raw, X_test_raw
gc.collect()

# Persist processed artifacts and metadata
feature_cols = [f"f_{i}" for i in range(feature_dim)]
df_train_feats = pd.DataFrame(X_train, columns=feature_cols)
df_train_feats["domain_id"] = train_domain_ids
df_train_feats.to_parquet("./working/train_features.parquet", index=False)

df_val_feats = pd.DataFrame(X_val, columns=feature_cols)
df_val_feats["domain_id"] = val_domain_ids
df_val_feats.to_parquet("./working/val_features.parquet", index=False)

df_test_feats = pd.DataFrame(X_test, columns=feature_cols)
df_test_feats["domain_id"] = test_domain_ids
df_test_feats.to_parquet("./working/test_features.parquet", index=False)

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
    """Asymmetric Loss for multi-label classification with negative margin discounting."""

    def __init__(
        self,
        gamma_neg: float = 3.0,
        gamma_pos: float = 1.0,
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

        loss_pos = (
            targets
            * torch.pow(1.0 - p, self.gamma_pos)
            * torch.log(p.clamp(min=self.eps))
        )

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


class CompoundRankingLoss(nn.Module):
    """Compound loss combining AsymmetricLoss with Top-K Truncated Hard-Margin Ranking Loss."""

    def __init__(
        self,
        gamma_neg: float = 3.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        top_m: int = 25,
        margin: float = 0.5,
        temperature: float = 1.0,
        ranking_weight: float = 1.0,
    ):
        super().__init__()
        self.asym_loss = AsymmetricLoss(
            gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip
        )
        self.top_m = top_m
        self.margin = margin
        self.temperature = temperature
        self.ranking_weight = ranking_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.asym_loss(logits, targets)

        # Top-K Truncated Hard-Margin Ranking Loss
        targets_bool = targets > 0.5
        neg_mask = ~targets_bool

        has_pos = targets_bool.any(dim=-1)
        has_neg = neg_mask.any(dim=-1)
        valid = has_pos & has_neg

        if not valid.any():
            return loss_bce

        logits_valid = logits[valid]
        targets_valid = targets_bool[valid]

        # Extract top-M hardest negatives per domain
        neg_logits = torch.where(
            targets_valid,
            torch.tensor(-1e9, device=logits.device, dtype=logits.dtype),
            logits_valid,
        )
        k_val = min(self.top_m, neg_logits.shape[-1])
        top_neg_logits, _ = torch.topk(neg_logits, k=k_val, dim=-1)

        # Pairwise ranking margin between true positives and top hardest negatives
        diff = (
            top_neg_logits.unsqueeze(1)
            - logits_valid.unsqueeze(2)
            + self.margin
        ) / self.temperature
        pair_losses = self.temperature * F.softplus(diff)

        # Mask only true positives
        pair_losses = pair_losses * targets_valid.unsqueeze(2)
        pos_counts = targets_valid.sum(dim=-1, keepdim=True).clamp(min=1)

        # Average over top-K negatives and positives per domain
        ranking_loss = (pair_losses.sum(dim=2) / k_val).sum(dim=1) / pos_counts.squeeze(1)
        ranking_loss = ranking_loss.mean()

        return loss_bce + self.ranking_weight * ranking_loss


class TrackerResNet(nn.Module):
    """Deep Tabular Residual Network with direct-link bypass and GCD refinement."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 355,
        hidden_dim: int = 384,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
        init_bias: np.ndarray = None,
        P_cooccur: np.ndarray = None,
    ):
        super().__init__()
        self.num_classes = num_classes
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

        # Direct-link linear bypass initialized with positive diagonal weights
        self.direct_bypass = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.zeros_(self.direct_bypass.weight)
        with torch.no_grad():
            self.direct_bypass.weight.fill_diagonal_(2.0)

        # Graph Correlation Diffusion (GCD) Refinement Layer
        if P_cooccur is not None:
            self.register_buffer(
                "P_cooccur", torch.from_numpy(P_cooccur.astype(np.float32))
            )
        else:
            self.register_buffer("P_cooccur", torch.zeros(num_classes, num_classes))

        self.diff_proj = nn.Linear(num_classes, num_classes)
        nn.init.eye_(self.diff_proj.weight)
        nn.init.zeros_(self.diff_proj.bias)

        self.b_diff = nn.Parameter(torch.zeros(num_classes))
        self.gamma = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self, x_dense: torch.Tensor, x_direct: torch.Tensor = None
    ) -> torch.Tensor:
        if x_direct is None:
            x_direct = torch.zeros(
                (x_dense.size(0), self.num_classes),
                dtype=x_dense.dtype,
                device=x_dense.device,
            )
        h = self.stem(x_dense)
        for block in self.res_blocks:
            h = block(h)
        h = self.final_norm(h)
        logits_backbone = self.head(h)

        # Stem logits incorporate backbone tabular predictions and direct link bypass
        logits_stem = logits_backbone + self.direct_bypass(x_direct)

        # GCD refinement step: logits_final = logits_stem + gamma * (Linear(GELU(logits_stem)) @ P_cooccur + b_diff)
        diff_act = F.gelu(logits_stem)
        diff_proj = self.diff_proj(diff_act)
        diff_out = torch.matmul(diff_proj, self.P_cooccur) + self.b_diff
        logits_final = logits_stem + self.gamma * diff_out

        return logits_final


model = TrackerResNet(
    input_dim=feature_dim,
    num_classes=num_trackers,
    hidden_dim=384,
    num_blocks=3,
    dropout_rate=0.25,
    init_bias=prior_logits,
    P_cooccur=P_cooccur,
).to(device)

criterion = CompoundRankingLoss(
    gamma_neg=3.0,
    gamma_pos=1.0,
    clip=0.05,
    top_m=25,
    margin=0.5,
    temperature=1.0,
    ranking_weight=1.0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

EPOCHS = 14
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# Sanity check forward-backward pass
dummy_input = torch.randn(8, feature_dim, device=device)
dummy_direct = torch.zeros(8, num_trackers, device=device)
dummy_target = torch.randint(0, 2, (8, num_trackers), device=device).float()
model.train()
optimizer.zero_grad()
dummy_logits = model(dummy_input, dummy_direct)
dummy_loss = criterion(dummy_logits, dummy_target)
dummy_loss.backward()
optimizer.zero_grad()

# =============================================================================
# 4. TRAINING, VALIDATION & INFERENCE PIPELINE
# =============================================================================

Y_train_dense = Y_train.toarray().astype(np.float32)
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(tracker_direct_train),
    torch.from_numpy(Y_train_dense),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


def compute_recall_at_10(
    eval_model, eval_features, eval_targets_csr, batch_size=4096, eval_direct=None
):
    """Computes exact official Recall@10 across all evaluation domains."""
    eval_model.eval()
    n_samples = eval_features.shape[0]
    if eval_direct is None:
        eval_direct = tracker_direct_val
    top10_list = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.from_numpy(eval_features[start_idx:end_idx]).to(device)
            batch_direct = torch.from_numpy(eval_direct[start_idx:end_idx]).to(device)
            logits = eval_model(batch_x, batch_direct)
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


best_val_recall = 0.0
best_epoch = 0
best_model_path = "./working/tracker_resnet_best.pt"
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_x, batch_direct, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_direct = batch_direct.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, batch_direct)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_train_loss / max(num_batches, 1)

    val_recall, _ = compute_recall_at_10(
        model, X_val, Y_val, batch_size=4096, eval_direct=tracker_direct_val
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
    model, X_val, Y_val, batch_size=4096, eval_direct=tracker_direct_val
)

# Test inference on all 50,000 target domains
n_test = X_test.shape[0]
test_top10_list = []

with torch.no_grad():
    for start_idx in range(0, n_test, 4096):
        end_idx = min(start_idx + 4096, n_test)
        batch_x = torch.from_numpy(X_test[start_idx:end_idx]).to(device)
        batch_direct = torch.from_numpy(
            tracker_direct_test[start_idx:end_idx]
        ).to(device)
        test_logits = model(batch_x, batch_direct)
        test_top10_batch = torch.topk(test_logits, k=10, dim=1).indices.cpu().numpy()
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

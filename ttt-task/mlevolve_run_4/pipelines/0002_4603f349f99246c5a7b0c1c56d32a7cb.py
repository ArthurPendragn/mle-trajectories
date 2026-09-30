import copy
import gc
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# Deterministic Environment Setup
# =========================================================================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# Step 1: Data Processing & Feature Engineering
# =========================================================================

# 1. Load target and tracker metadata
target_df = pd.read_csv("input/target.tsv", sep="\t")
test_domains = target_df["domain_id"].to_numpy(dtype=np.int64)
N_test = len(test_domains)

trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
trackers_df.columns = [c.strip() for c in trackers_df.columns]
tracker_id_to_tdid = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
tdid_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)
all_tracker_ids = np.sort(trackers_df["tracker_id"].unique())
N_trackers = len(all_tracker_ids)

# 2. Load tracking graph and construct leak-free train/val splits
train_edges = pd.read_parquet("input/tracking_graph_train.parquet")
train_edges["domain_id"] = train_edges["domain_id"].astype(np.int64)
train_edges["tracker_id"] = train_edges["tracker_id"].astype(np.int32)

all_train_domains = train_edges["domain_id"].unique()
np.random.shuffle(all_train_domains)

N_val = min(25000, int(len(all_train_domains) * 0.1))
val_domains = all_train_domains[:N_val]
train_pool = all_train_domains[N_val:]

tracker_counts = train_edges[train_edges["domain_id"].isin(train_pool)][
    "tracker_id"
].value_counts()
rare_trackers = set(tracker_counts[tracker_counts < 1500].index)

rare_domains = train_edges[
    train_edges["domain_id"].isin(train_pool)
    & train_edges["tracker_id"].isin(rare_trackers)
]["domain_id"].unique()

target_train_size = min(250000, len(train_pool))
remaining_needed = max(0, target_train_size - len(rare_domains))
non_rare_candidates = np.setdiff1d(train_pool, rare_domains)

sampled_non_rare = np.random.choice(
    non_rare_candidates,
    size=min(remaining_needed, len(non_rare_candidates)),
    replace=False,
)
train_domains = np.concatenate([rare_domains, sampled_non_rare])
np.random.shuffle(train_domains)
N_train = len(train_domains)


def build_label_matrix(domain_subset, edges_df):
    labels = np.zeros((len(domain_subset), N_trackers), dtype=np.uint8)
    domain_to_row = {d: i for i, d in enumerate(domain_subset)}
    sub_df = edges_df[edges_df["domain_id"].isin(domain_to_row)]
    d_arr = sub_df["domain_id"].to_numpy()
    t_arr = sub_df["tracker_id"].to_numpy()
    rows = np.fromiter(
        (domain_to_row[d] for d in d_arr), dtype=np.int32, count=len(d_arr)
    )
    valid = (t_arr >= 0) & (t_arr < N_trackers)
    labels[rows[valid], t_arr[valid]] = 1
    return labels


train_labels_mat = build_label_matrix(train_domains, train_edges)
val_labels_mat = build_label_matrix(val_domains, train_edges)

tracker_train_counts = train_labels_mat.sum(axis=0)
tracker_priors = tracker_train_counts / float(N_train)
top_voted_trackers = np.argsort(-tracker_train_counts)[:35]

# Save labels and metadata
trackers_df["train_count"] = [
    tracker_train_counts[tid] for tid in trackers_df["tracker_id"]
]
trackers_df["prior_prob"] = [tracker_priors[tid] for tid in trackers_df["tracker_id"]]
trackers_df.sort_values(by="prior_prob", ascending=False).to_parquet(
    "./working/tracker_meta.parquet", index=False
)

train_labels_df = pd.DataFrame(
    train_labels_mat, columns=[f"tracker_{i}" for i in range(N_trackers)]
)
train_labels_df.insert(0, "domain_id", train_domains)
train_labels_df.to_parquet("./working/train_labels.parquet", index=False)

val_labels_df = pd.DataFrame(
    val_labels_mat, columns=[f"tracker_{i}" for i in range(N_trackers)]
)
val_labels_df.insert(0, "domain_id", val_domains)
val_labels_df.to_parquet("./working/val_labels.parquet", index=False)

# 3. Global index mapping for high-speed graph operations
eval_domains = np.concatenate([train_domains, val_domains, test_domains])
N_eval = len(eval_domains)

max_id = max(int(eval_domains.max()), int(trackers_df["tracking_domain_id"].max()))
eval_id_map = np.full(max_id + 1, -1, dtype=np.int32)
eval_id_map[eval_domains] = np.arange(N_eval, dtype=np.int32)

# 4. Stream link graph: degrees, tracker direct links, & collaborative votes
in_degree = np.zeros(N_eval, dtype=np.int32)
out_degree = np.zeros(N_eval, dtype=np.int32)
out_tracker_links = np.zeros(N_eval, dtype=np.int32)
in_train_neighbors = np.zeros(N_eval, dtype=np.int32)
out_train_neighbors = np.zeros(N_eval, dtype=np.int32)

in_votes = np.zeros((35, N_eval), dtype=np.int32)
out_votes = np.zeros((35, N_eval), dtype=np.int32)

is_tracker_domain = np.zeros(max_id + 1, dtype=bool)
is_tracker_domain[trackers_df["tracking_domain_id"].values] = True

is_train_domain = np.zeros(max_id + 1, dtype=bool)
is_train_domain[train_domains] = True

train_has_tracker = np.zeros((35, max_id + 1), dtype=bool)
for rank_idx, tid in enumerate(top_voted_trackers):
    nodes_with_t = train_domains[train_labels_mat[:, tid] == 1]
    train_has_tracker[rank_idx, nodes_with_t] = True

link_graph_pf = pq.ParquetFile("input/link-graph.parquet")
for batch in link_graph_pf.iter_batches(
    batch_size=5000000, columns=["source_domain_id", "target_domain_id"]
):
    src = batch["source_domain_id"].to_numpy()
    dst = batch["target_domain_id"].to_numpy()

    valid_src = (src >= 0) & (src <= max_id)
    valid_dst = (dst >= 0) & (dst <= max_id)

    src_eval = np.full(len(src), -1, dtype=np.int32)
    src_eval[valid_src] = eval_id_map[src[valid_src]]

    dst_eval = np.full(len(dst), -1, dtype=np.int32)
    dst_eval[valid_dst] = eval_id_map[dst[valid_dst]]

    src_valid_idx = np.where(src_eval >= 0)[0]
    dst_valid_idx = np.where(dst_eval >= 0)[0]

    if len(src_valid_idx) > 0:
        out_degree += np.bincount(src_eval[src_valid_idx], minlength=N_eval)
    if len(dst_valid_idx) > 0:
        in_degree += np.bincount(dst_eval[dst_valid_idx], minlength=N_eval)

    if len(src_valid_idx) > 0:
        sub_dst = np.clip(dst[src_valid_idx], 0, max_id)
        sub_valid_dst = valid_dst[src_valid_idx]
        sub_src_eval = src_eval[src_valid_idx]

        tracker_mask = sub_valid_dst & is_tracker_domain[sub_dst]
        if np.any(tracker_mask):
            out_tracker_links += np.bincount(
                sub_src_eval[tracker_mask], minlength=N_eval
            )

        dst_is_tr = sub_valid_dst & is_train_domain[sub_dst]
        if np.any(dst_is_tr):
            out_train_neighbors += np.bincount(
                sub_src_eval[dst_is_tr], minlength=N_eval
            )

        for r in range(35):
            d_has_k = sub_valid_dst & train_has_tracker[r, sub_dst]
            if np.any(d_has_k):
                out_votes[r] += np.bincount(sub_src_eval[d_has_k], minlength=N_eval)

    if len(dst_valid_idx) > 0:
        sub_src = np.clip(src[dst_valid_idx], 0, max_id)
        sub_valid_src = valid_src[dst_valid_idx]
        sub_dst_eval = dst_eval[dst_valid_idx]

        src_is_tr = sub_valid_src & is_train_domain[sub_src]
        if np.any(src_is_tr):
            in_train_neighbors += np.bincount(sub_dst_eval[src_is_tr], minlength=N_eval)

        for r in range(35):
            s_has_k = sub_valid_src & train_has_tracker[r, sub_src]
            if np.any(s_has_k):
                in_votes[r] += np.bincount(sub_dst_eval[s_has_k], minlength=N_eval)

# 5. Extract hostnames and lexical domain features
domains_pf = pq.ParquetFile("input/domains.parquet")
eval_hostnames = [""] * N_eval

for batch in domains_pf.iter_batches(
    batch_size=3000000, columns=["domain_id", "domain"]
):
    b_ids = batch["domain_id"].to_numpy()
    valid_mask = (b_ids >= 0) & (b_ids <= max_id)
    if not np.any(valid_mask):
        continue
    valid_indices = np.where(valid_mask)[0]
    b_eval = eval_id_map[b_ids[valid_mask]]
    matched = np.where(b_eval >= 0)[0]
    if len(matched) > 0:
        raw_indices = valid_indices[matched]
        matched_eval_ids = b_eval[matched]
        py_names = batch.column("domain").take(raw_indices).to_pylist()
        for e_id, name in zip(matched_eval_ids, py_names):
            if name:
                eval_hostnames[e_id] = name


def compute_entropy(s):
    if len(s) <= 1:
        return 0.0
    cnts = Counter(s)
    tot = float(len(s))
    return -sum((c / tot) * math.log2(c / tot) for c in cnts.values())


domain_lens = np.array([len(s) for s in eval_hostnames], dtype=np.float32)
dot_counts = np.array([s.count(".") for s in eval_hostnames], dtype=np.float32)
has_subdomain = (dot_counts > 1).astype(np.float32)
hyphen_counts = np.array([s.count("-") for s in eval_hostnames], dtype=np.float32)
digit_counts = np.array(
    [sum(c.isdigit() for c in s) for s in eval_hostnames], dtype=np.float32
)
digit_ratio = digit_counts / np.maximum(1.0, domain_lens)
vowel_counts = np.array(
    [sum(c in "aeiou" for c in s.lower()) for s in eval_hostnames],
    dtype=np.float32,
)
alpha_counts = np.array(
    [sum(c.isalpha() for c in s) for s in eval_hostnames], dtype=np.float32
)
vowel_ratio = vowel_counts / np.maximum(1.0, alpha_counts)
entropies = np.array([compute_entropy(s) for s in eval_hostnames], dtype=np.float32)

tlds = [s.split(".")[-1].lower() if "." in s else "" for s in eval_hostnames]
common_gtlds = {"com", "net", "org", "info", "biz"}
is_common_gtld = np.array(
    [1.0 if t in common_gtlds else 0.0 for t in tlds], dtype=np.float32
)
is_cctld = np.array([1.0 if len(t) == 2 else 0.0 for t in tlds], dtype=np.float32)
is_edu_gov = np.array(
    [
        1.0 if any(k in s for k in [".edu", ".gov", ".ac."]) else 0.0
        for s in eval_hostnames
    ],
    dtype=np.float32,
)

train_tlds = [tlds[i] for i in range(N_train)]
tld_freq_lookup = {t: c / float(N_train) for t, c in Counter(train_tlds).items()}
tld_freq_feat = np.array([tld_freq_lookup.get(t, 0.0) for t in tlds], dtype=np.float32)

tfidf = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), min_df=50, max_features=1000
)
train_texts = [
    eval_hostnames[i] if eval_hostnames[i] else "unknown" for i in range(N_train)
]
tfidf.fit(train_texts)

svd = TruncatedSVD(n_components=16, random_state=42)
svd.fit(tfidf.transform(train_texts))

all_texts = [s if s else "unknown" for s in eval_hostnames]
svd_features = svd.transform(tfidf.transform(all_texts)).astype(np.float32)

# 6. Freedom of the Press Features
fop_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
fop_df.columns = [c.strip() for c in fop_df.columns]
tld_to_fop = dict(
    zip(
        fop_df["tld"].astype(str).str.lower(),
        fop_df["freedom_of_the_press"].astype(float),
    )
)

raw_fop = np.array([tld_to_fop.get(t, np.nan) for t in tlds], dtype=np.float32)
train_fop_median = float(np.nanmedian(raw_fop[:N_train]))
if np.isnan(train_fop_median):
    train_fop_median = 50.0
press_freedom_score = np.where(np.isnan(raw_fop), train_fop_median, raw_fop)
has_press_freedom = (~np.isnan(raw_fop)).astype(np.float32)

# 7. URL Classification Mapping
url_df = pd.read_csv("input/url-classification.csv", usecols=["url", "category"])


def extract_host(u):
    if not isinstance(u, str):
        return ""
    idx = u.find("://")
    if idx != -1:
        u = u[idx + 3 :]
    idx2 = u.find("/")
    if idx2 != -1:
        u = u[:idx2]
    idx3 = u.find(":")
    if idx3 != -1:
        u = u[:idx3]
    u = u.lower().strip()
    return u[4:] if u.startswith("www.") else u


url_df["host"] = [extract_host(x) for x in url_df["url"]]
cat_df = url_df.groupby(["host", "category"]).size().unstack(fill_value=0)
cat_columns = list(cat_df.columns)
cat_probs = cat_df.div(cat_df.sum(axis=1), axis=0)
cat_dict = {
    host: cat_probs.loc[host].to_numpy(dtype=np.float32) for host in cat_probs.index
}

url_cat_features = np.zeros((N_eval, len(cat_columns)), dtype=np.float32)
has_url_category = np.zeros(N_eval, dtype=np.float32)

for i, h in enumerate(eval_hostnames):
    cleaned_h = extract_host(h)
    if cleaned_h in cat_dict:
        url_cat_features[i] = cat_dict[cleaned_h]
        has_url_category[i] = 1.0

# 8. Assemble Full Feature Matrix
total_train_neighbors = in_train_neighbors + out_train_neighbors
total_degree = in_degree + out_degree

feature_dict = {
    "in_degree": in_degree.astype(np.float32),
    "out_degree": out_degree.astype(np.float32),
    "total_degree": total_degree.astype(np.float32),
    "log_in_degree": np.log1p(in_degree).astype(np.float32),
    "log_out_degree": np.log1p(out_degree).astype(np.float32),
    "log_total_degree": np.log1p(total_degree).astype(np.float32),
    "degree_ratio": ((in_degree + 1.0) / (out_degree + 1.0)).astype(np.float32),
    "is_isolated": (total_degree == 0).astype(np.float32),
    "out_tracker_links": out_tracker_links.astype(np.float32),
    "has_tracker_outlink": (out_tracker_links > 0).astype(np.float32),
    "in_train_neighbors": in_train_neighbors.astype(np.float32),
    "out_train_neighbors": out_train_neighbors.astype(np.float32),
    "total_train_neighbors": total_train_neighbors.astype(np.float32),
    "has_train_neighbors": (total_train_neighbors > 0).astype(np.float32),
    "domain_len": domain_lens,
    "dot_counts": dot_counts,
    "has_subdomain": has_subdomain,
    "hyphen_counts": hyphen_counts,
    "digit_counts": digit_counts,
    "digit_ratio": digit_ratio,
    "vowel_ratio": vowel_ratio,
    "entropy": entropies,
    "is_common_gtld": is_common_gtld,
    "is_cctld": is_cctld,
    "is_edu_gov": is_edu_gov,
    "tld_freq": tld_freq_feat,
    "press_freedom_score": press_freedom_score,
    "has_press_freedom": has_press_freedom,
    "has_url_category": has_url_category,
}

for c_idx, c_name in enumerate(cat_columns):
    feature_dict[f"url_cat_{c_name}"] = url_cat_features[:, c_idx]

for s_idx in range(16):
    feature_dict[f"domain_svd_{s_idx}"] = svd_features[:, s_idx]

denom_in = np.maximum(1.0, in_train_neighbors.astype(np.float32))
denom_out = np.maximum(1.0, out_train_neighbors.astype(np.float32))
for r_idx in range(35):
    feature_dict[f"in_vote_t{r_idx}"] = in_votes[r_idx].astype(np.float32)
    feature_dict[f"out_vote_t{r_idx}"] = out_votes[r_idx].astype(np.float32)
    feature_dict[f"in_norm_vote_t{r_idx}"] = (
        in_votes[r_idx].astype(np.float32) / denom_in
    )
    feature_dict[f"out_norm_vote_t{r_idx}"] = (
        out_votes[r_idx].astype(np.float32) / denom_out
    )

full_features_df = pd.DataFrame(feature_dict)
feature_names = list(full_features_df.columns)

assert not full_features_df.isna().any().any(), "Detected NaN values in feature matrix"
assert not np.isinf(
    full_features_df.to_numpy()
).any(), "Detected Inf values in feature matrix"

train_feat_df = full_features_df.iloc[:N_train].copy()
train_feat_df.insert(0, "domain_id", train_domains)

val_feat_df = full_features_df.iloc[N_train : N_train + N_val].copy()
val_feat_df.insert(0, "domain_id", val_domains)

test_feat_df = full_features_df.iloc[N_train + N_val :].copy()
test_feat_df.insert(0, "domain_id", test_domains)

assert (
    test_feat_df["domain_id"].values == target_df["domain_id"].values
).all(), "Test domain ordering mismatch!"

train_feat_df.to_parquet("./working/train_features.parquet", index=False)
val_feat_df.to_parquet("./working/val_features.parquet", index=False)
test_feat_df.to_parquet("./working/test_features.parquet", index=False)

metadata = {
    "num_features": len(feature_names),
    "feature_names": feature_names,
    "num_train": N_train,
    "num_val": N_val,
    "num_test": N_test,
    "num_trackers": N_trackers,
    "metric": "Recall@10",
}
with open("./working/feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# =========================================================================
# Step 2: Model Design
# =========================================================================


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.bn1(self.fc1(x))))
        out = self.drop2(self.bn2(self.fc2(out)))
        return F.gelu(out + residual)


class TrackerAttentionRankNet(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_trackers: int = 355,
        hidden_dim: int = 512,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        prior_probs: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        self.input_bn = nn.BatchNorm1d(in_features)
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout)

        self.res_block1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout=dropout)

        self.domain_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim)
        )

        self.tracker_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * 0.02
        )
        self.tracker_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.tracker_norm = nn.LayerNorm(embed_dim)

        self.direct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_trackers),
        )

        self.gate = nn.Parameter(torch.tensor([0.0]))

        if prior_probs is not None and len(prior_probs) == num_trackers:
            p = np.clip(np.asarray(prior_probs, dtype=np.float32), 1e-4, 1.0 - 1e-4)
            init_bias = np.log(p / (1.0 - p))
            self.prior_bias = nn.Parameter(
                torch.tensor(init_bias, dtype=torch.float32), requires_grad=True
            )
        else:
            self.prior_bias = nn.Parameter(
                torch.zeros(num_trackers, dtype=torch.float32),
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_drop(self.input_act(self.input_proj(self.input_bn(x))))
        h = self.res_block1(h)
        h = self.res_block2(h)

        z_domain = self.domain_proj(h)

        trackers = self.tracker_embeddings.unsqueeze(0)
        attn_out, _ = self.tracker_attn(trackers, trackers, trackers)
        refined_trackers = self.tracker_norm(
            self.tracker_embeddings + attn_out.squeeze(0)
        )

        match_logits = torch.matmul(z_domain, refined_trackers.t()) / math.sqrt(
            self.embed_dim
        )
        direct_logits = self.direct_head(h)

        alpha = torch.sigmoid(self.gate)
        logits = (
            (1.0 - alpha) * direct_logits
            + alpha * match_logits
            + self.prior_bias.unsqueeze(0)
        )
        return logits


class AsymmetricTop10RankingLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_neg: float = 0.05,
        margin: float = 1.0,
        rank_weight: float = 0.5,
        top_k: int = 10,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_neg = clip_neg
        self.margin = margin
        self.rank_weight = rank_weight
        self.top_k = top_k
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        pos_loss = targets * torch.log(probs.clamp(min=self.eps))
        if self.gamma_pos > 0.0:
            pos_loss = pos_loss * ((1.0 - probs) ** self.gamma_pos)

        p_neg = probs
        if self.clip_neg > 0.0:
            p_neg = (p_neg - self.clip_neg).clamp(min=0.0)
        neg_loss = (
            (1.0 - targets)
            * (p_neg**self.gamma_neg)
            * torch.log((1.0 - p_neg).clamp(min=self.eps))
        )
        asl_loss = -(pos_loss + neg_loss).sum(dim=-1).mean()

        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e9))
        topk_negs, _ = torch.topk(neg_logits, k=self.top_k, dim=-1)
        kth_neg = topk_negs[:, self.top_k - 1 : self.top_k]

        violation = F.relu(self.margin + kth_neg - logits)
        num_pos = targets.sum(dim=-1).clamp(min=1.0)
        rank_loss = ((violation * targets).sum(dim=-1) / num_pos).mean()

        return asl_loss + self.rank_weight * rank_loss


def build_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> AdamW:
    no_decay = [
        "bias",
        "LayerNorm.weight",
        "BatchNorm1d.weight",
        "input_bn.weight",
    ]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer, num_epochs: int = 12, eta_min: float = 1e-5
) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)


# =========================================================================
# Step 3: Training & Evaluation Pipeline
# =========================================================================

# Prepare Feature & Label Arrays
feature_cols = [c for c in train_feat_df.columns if c != "domain_id"]
X_train = train_feat_df[feature_cols].to_numpy(dtype=np.float32)
Y_train = train_labels_mat.astype(np.float32)

X_val = val_feat_df[feature_cols].to_numpy(dtype=np.float32)
Y_val = val_labels_mat.astype(np.float32)

X_test = test_feat_df[feature_cols].to_numpy(dtype=np.float32)

num_features = X_train.shape[1]

# Model and Optimizer Instantiation
model = TrackerAttentionRankNet(
    in_features=num_features,
    num_trackers=N_trackers,
    hidden_dim=512,
    embed_dim=128,
    num_heads=4,
    dropout=0.2,
    prior_probs=tracker_priors,
).to(device)

criterion = AsymmetricTop10RankingLoss(
    gamma_pos=0.0,
    gamma_neg=2.0,
    clip_neg=0.05,
    margin=1.0,
    rank_weight=0.5,
    top_k=10,
)

num_epochs = 12
optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
scheduler = build_scheduler(optimizer, num_epochs=num_epochs, eta_min=1e-5)

# DataLoader Construction
batch_size = 512
eval_batch_size = 1024
pin_mem = torch.cuda.is_available()

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem
)
val_loader = DataLoader(
    val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=pin_mem
)
test_loader = DataLoader(
    test_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=pin_mem
)


def compute_recall_at_10(logits: np.ndarray, targets: np.ndarray) -> float:
    top10_indices = np.argsort(-logits, axis=1)[:, :10]
    hits = np.take_along_axis(targets, top10_indices, axis=1).sum(axis=1)
    num_positives = targets.sum(axis=1)
    recalls = np.where(num_positives > 0, hits / num_positives, 0.0)
    return float(np.mean(recalls))


def evaluate(net, dataloader, dev, crit=None):
    net.eval()
    all_logits = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            out = net(bx)
            if crit is not None:
                loss = crit(out, by)
                total_loss += loss.item()
                batches += 1
            all_logits.append(out.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, batches) if crit is not None else 0.0
    return all_logits, avg_loss


best_val_recall = -1.0
best_weights = None

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_train_loss += loss.item()
        train_batches += 1

    scheduler.step()
    train_loss = total_train_loss / max(1, train_batches)

    val_logits, val_loss = evaluate(model, val_loader, device, crit=criterion)
    val_recall = compute_recall_at_10(val_logits, Y_val)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.5f} | LR: {current_lr:.6f}"
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, "./working/best_tracker_model.pt")

# Final Validation Assessment
if best_weights is not None:
    model.load_state_dict(best_weights)

final_val_logits, _ = evaluate(model, val_loader, device)
final_val_recall = compute_recall_at_10(final_val_logits, Y_val)

# Full Test Set Model Inference
model.eval()
test_logits_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device, non_blocking=True)
        out = model(bx)
        test_logits_list.append(out.cpu().numpy())

test_logits = np.concatenate(test_logits_list, axis=0)
test_top10 = np.argsort(-test_logits, axis=1)[:, :10]

num_classes = test_logits.shape[1]
tracker_id_to_tdid_arr = np.zeros(num_classes, dtype=np.int64)
for tid, tdid in tracker_id_to_tdid.items():
    if 0 <= tid < num_classes:
        tracker_id_to_tdid_arr[tid] = tdid

test_pred_tdids = tracker_id_to_tdid_arr[test_top10]

rep_domains = np.repeat(test_domains, 10)
flat_tdids = test_pred_tdids.flatten()

submission_df = pd.DataFrame(
    {"domain_id": rep_domains, "tracking_domain_id": flat_tdids}
)

sub_csv_path = "./submission/submission.csv"
sub_tsv_path = "./submission/submission.tsv"

submission_df.to_csv(sub_csv_path, sep="\t", index=False)
submission_df.to_csv(sub_tsv_path, sep="\t", index=False)

assert os.path.exists(sub_csv_path), "submission.csv was not generated!"
assert len(submission_df) == len(test_domains) * 10, "Row count mismatch!"
assert not submission_df.isna().any().any(), "Detected NaN values in submission!"

print(f"Final Validation Score: {final_val_recall}")

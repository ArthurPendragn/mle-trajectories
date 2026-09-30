import copy
import gc
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
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

# Compute 32-dimensional TruncatedSVD spectral embeddings from training label PPMI matrix
C = (train_labels_mat.T @ train_labels_mat).astype(np.float64)
diag = np.diag(C)
diag_safe = np.maximum(diag, 1.0)
denom = np.outer(diag_safe, diag_safe)
ratio = (C * N_train) / denom
ppmi_matrix = np.zeros_like(C, dtype=np.float32)
pos_mask = (C > 0) & (ratio > 1.0)
ppmi_matrix[pos_mask] = np.log(ratio[pos_mask]).astype(np.float32)

ppmi_svd = TruncatedSVD(n_components=32, random_state=42)
ppmi_embeddings = ppmi_svd.fit_transform(ppmi_matrix).astype(np.float32)
np.save("./working/ppmi_embeddings.npy", ppmi_embeddings)

# Augmented tracker metadata representation
cat_list = sorted(trackers_df["category"].dropna().unique())
cat_map = {c: i for i, c in enumerate(cat_list)}
country_list = trackers_df["country"].value_counts().head(10).index.tolist()
country_map = {c: i for i, c in enumerate(country_list)}

trackers_df_sorted = trackers_df.sort_values(by="tracker_id").reset_index(drop=True)
cat_onehot = np.zeros((N_trackers, len(cat_list)), dtype=np.float32)
country_onehot = np.zeros((N_trackers, len(country_list)), dtype=np.float32)
for idx, r in trackers_df_sorted.iterrows():
    if r["category"] in cat_map:
        cat_onehot[idx, cat_map[r["category"]]] = 1.0
    if r["country"] in country_map:
        country_onehot[idx, country_map[r["country"]]] = 1.0

log_train_cnts = (np.log1p(tracker_train_counts)[:, None] / 10.0).astype(np.float32)
priors_col = tracker_priors[:, None].astype(np.float32)

tracker_meta_matrix = np.concatenate(
    [cat_onehot, country_onehot, log_train_cnts, priors_col], axis=1
).astype(np.float32)

# Save labels and metadata
trackers_df["category_code"] = [cat_map.get(c, -1) for c in trackers_df["category"]]
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

# 4. Stream link graph: degrees, 355-tracker direct links, & bipartite projection lists
in_degree = np.zeros(N_eval, dtype=np.int32)
out_degree = np.zeros(N_eval, dtype=np.int32)
out_tracker_links = np.zeros(N_eval, dtype=np.int32)

tracker_domain_to_tid = np.full(max_id + 1, -1, dtype=np.int16)
for _, r in trackers_df.iterrows():
    tdid = int(r["tracking_domain_id"])
    tid = int(r["tracker_id"])
    if tdid <= max_id:
        tracker_domain_to_tid[tdid] = tid

tracker_flat_counts = np.zeros(N_eval * N_trackers, dtype=np.int32)

train_id_map = np.full(max_id + 1, -1, dtype=np.int32)
train_id_map[train_domains] = np.arange(N_train, dtype=np.int32)

out_src_list = []
out_dst_list = []
in_dst_list = []
in_src_list = []

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

        # Vectorized 355-tracker direct link extraction
        dst_tids = tracker_domain_to_tid[sub_dst]
        trk_mask = sub_valid_dst & (dst_tids >= 0)
        if np.any(trk_mask):
            trk_src = sub_src_eval[trk_mask]
            trk_tid = dst_tids[trk_mask]
            out_tracker_links += np.bincount(trk_src, minlength=N_eval)
            flat_idx = trk_src.astype(np.int64) * N_trackers + trk_tid.astype(np.int64)
            tracker_flat_counts += np.bincount(
                flat_idx, minlength=N_eval * N_trackers
            ).astype(np.int32)

        dst_tr = np.where(sub_valid_dst, train_id_map[sub_dst], -1)
        mask_out = dst_tr >= 0
        if np.any(mask_out):
            out_src_list.append(sub_src_eval[mask_out])
            out_dst_list.append(dst_tr[mask_out])

    if len(dst_valid_idx) > 0:
        sub_src = np.clip(src[dst_valid_idx], 0, max_id)
        sub_valid_src = valid_src[dst_valid_idx]
        sub_dst_eval = dst_eval[dst_valid_idx]

        src_tr = np.where(sub_valid_src, train_id_map[sub_src], -1)
        mask_in = src_tr >= 0
        if np.any(mask_in):
            in_dst_list.append(sub_dst_eval[mask_in])
            in_src_list.append(src_tr[mask_in])

direct_tracker_counts = tracker_flat_counts.reshape(N_eval, N_trackers)
direct_tracker_presence = (direct_tracker_counts > 0).astype(np.float32)
del tracker_flat_counts
gc.collect()

# Build directed sparse CSR projections and project across all 355 trackers
train_labels_f32 = train_labels_mat.astype(np.float32)

if len(out_src_list) > 0:
    out_src_arr = np.concatenate(out_src_list)
    out_dst_arr = np.concatenate(out_dst_list)
    del out_src_list, out_dst_list
    no_self_out = ~((out_src_arr < N_train) & (out_src_arr == out_dst_arr))
    out_src_arr = out_src_arr[no_self_out]
    out_dst_arr = out_dst_arr[no_self_out]
    A_out = csr_matrix(
        (np.ones(len(out_src_arr), dtype=np.float32), (out_src_arr, out_dst_arr)),
        shape=(N_eval, N_train),
    )
    del out_src_arr, out_dst_arr
    gc.collect()
else:
    A_out = csr_matrix((N_eval, N_train), dtype=np.float32)

out_train_neighbors = np.array(A_out.sum(axis=1), dtype=np.float32).flatten()
out_votes = A_out.dot(train_labels_f32)
del A_out
gc.collect()

if len(in_dst_list) > 0:
    in_dst_arr = np.concatenate(in_dst_list)
    in_src_arr = np.concatenate(in_src_list)
    del in_dst_list, in_src_list
    no_self_in = ~((in_dst_arr < N_train) & (in_dst_arr == in_src_arr))
    in_dst_arr = in_dst_arr[no_self_in]
    in_src_arr = in_src_arr[no_self_in]
    A_in = csr_matrix(
        (np.ones(len(in_dst_arr), dtype=np.float32), (in_dst_arr, in_src_arr)),
        shape=(N_eval, N_train),
    )
    del in_dst_arr, in_src_arr
    gc.collect()
else:
    A_in = csr_matrix((N_eval, N_train), dtype=np.float32)

in_train_neighbors = np.array(A_in.sum(axis=1), dtype=np.float32).flatten()
in_votes = A_in.dot(train_labels_f32)
del A_in, train_labels_f32
gc.collect()

denom_in = np.maximum(1.0, in_train_neighbors[:, None])
denom_out = np.maximum(1.0, out_train_neighbors[:, None])
in_norm_votes = (in_votes / denom_in).astype(np.float32)
out_norm_votes = (out_votes / denom_out).astype(np.float32)
log_votes = np.log1p(in_votes + out_votes).astype(np.float32)
del in_votes, out_votes
gc.collect()

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

for t in range(N_trackers):
    feature_dict[f"direct_tracker_{t}"] = direct_tracker_presence[:, t]
    feature_dict[f"in_norm_vote_t{t}"] = in_norm_votes[:, t]
    feature_dict[f"out_norm_vote_t{t}"] = out_norm_votes[:, t]
    feature_dict[f"log_vote_t{t}"] = log_votes[:, t]

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
        tracker_feat_indices: list = None,
        tracker_meta_matrix: np.ndarray = None,
        ppmi_embeddings: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        if tracker_feat_indices is not None:
            self.register_buffer(
                "tracker_indices",
                torch.tensor(tracker_feat_indices, dtype=torch.long),
            )
        else:
            self.register_buffer(
                "tracker_indices",
                torch.arange(in_features - 4 * num_trackers, in_features, dtype=torch.long),
            )

        self.input_bn = nn.BatchNorm1d(in_features)
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout)

        self.res_block1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout=dropout)

        self.domain_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim)
        )

        # Tracker metadata embeddings & self-attention refinement
        self.tracker_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * 0.02
        )
        if tracker_meta_matrix is not None:
            self.register_buffer(
                "tracker_meta",
                torch.from_numpy(tracker_meta_matrix).float(),
            )
            self.meta_proj = nn.Linear(tracker_meta_matrix.shape[1], embed_dim)
        else:
            self.meta_proj = None

        if ppmi_embeddings is not None:
            self.register_buffer(
                "ppmi_emb",
                torch.from_numpy(ppmi_embeddings).float(),
            )
            self.ppmi_proj = nn.Linear(ppmi_embeddings.shape[1], embed_dim)
        else:
            self.ppmi_proj = None

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

        # Multi-Channel Dedicated Tracker-Specific Graph Fusion Head (4 x 355 features)
        self.graph_stream = nn.Sequential(
            nn.Linear(4 * num_trackers, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_trackers),
        )
        self.graph_skip_weight = nn.Parameter(torch.ones(num_trackers) * 2.0)
        self.graph_gate = nn.Parameter(torch.tensor([0.0]))

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
        # Multi-Channel Tracker-Specific Graph Fusion Head
        tracker_graph_feats = x[:, self.tracker_indices]
        direct_links = tracker_graph_feats[:, : self.num_trackers]
        graph_logits = (
            self.graph_stream(tracker_graph_feats)
            + self.graph_skip_weight.unsqueeze(0) * direct_links
        )

        h = self.input_drop(self.input_act(self.input_proj(self.input_bn(x))))
        h = self.res_block1(h)
        h = self.res_block2(h)

        z_domain = self.domain_proj(h)

        base_tracker_emb = self.tracker_embeddings
        if self.meta_proj is not None:
            base_tracker_emb = base_tracker_emb + self.meta_proj(self.tracker_meta)
        if self.ppmi_proj is not None:
            base_tracker_emb = base_tracker_emb + self.ppmi_proj(self.ppmi_emb)

        trackers = base_tracker_emb.unsqueeze(0)
        attn_out, _ = self.tracker_attn(trackers, trackers, trackers)
        refined_trackers = self.tracker_norm(
            base_tracker_emb + attn_out.squeeze(0)
        )

        match_logits = torch.matmul(z_domain, refined_trackers.t()) / math.sqrt(
            self.embed_dim
        )
        direct_logits = self.direct_head(h)

        alpha = torch.sigmoid(self.gate)
        gamma = torch.sigmoid(self.graph_gate)
        logits = (
            (1.0 - alpha) * direct_logits
            + alpha * match_logits
            + gamma * graph_logits
            + self.prior_bias.unsqueeze(0)
        )
        return logits


class AdaptiveTop10RankingLoss(nn.Module):

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

        # Adaptive Top-10 Ranking Loss:
        # Dynamically set the negative rank threshold per domain to clamp(10 - P_i, min=0, max=9)
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e9))
        topk_negs, _ = torch.topk(neg_logits, k=self.top_k, dim=-1)

        num_pos = targets.sum(dim=-1).clamp(min=1.0)
        P_i = targets.sum(dim=-1).long()
        k_thresh = torch.clamp(self.top_k - P_i, min=0, max=self.top_k - 1)
        kth_neg = topk_negs.gather(1, k_thresh.unsqueeze(1))

        violation = F.relu(self.margin + kth_neg - logits)
        rank_loss = ((violation * targets).sum(dim=-1) / num_pos).mean()

        return asl_loss + self.rank_weight * rank_loss


AsymmetricTop10RankingLoss = AdaptiveTop10RankingLoss


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

direct_tracker_cols = [f"direct_tracker_{t}" for t in range(N_trackers)]
in_norm_vote_cols = [f"in_norm_vote_t{t}" for t in range(N_trackers)]
out_norm_vote_cols = [f"out_norm_vote_t{t}" for t in range(N_trackers)]
log_vote_cols = [f"log_vote_t{t}" for t in range(N_trackers)]

tracker_feat_cols = (
    direct_tracker_cols + in_norm_vote_cols + out_norm_vote_cols + log_vote_cols
)
tracker_feat_indices = [feature_cols.index(c) for c in tracker_feat_cols]

# Model and Optimizer Instantiation
model = TrackerAttentionRankNet(
    in_features=num_features,
    num_trackers=N_trackers,
    hidden_dim=512,
    embed_dim=128,
    num_heads=4,
    dropout=0.2,
    prior_probs=tracker_priors,
    tracker_feat_indices=tracker_feat_indices,
    tracker_meta_matrix=tracker_meta_matrix,
    ppmi_embeddings=ppmi_embeddings,
).to(device)

criterion = AdaptiveTop10RankingLoss(
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

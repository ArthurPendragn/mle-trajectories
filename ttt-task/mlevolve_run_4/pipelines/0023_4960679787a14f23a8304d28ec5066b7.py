from collections import Counter
import json
import math
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
tracker_companies = trackers_df["company"].fillna("Unknown").values
top_companies = [c for c, _ in Counter(tracker_companies).most_common(32)]
company_to_idx = {c: i for i, c in enumerate(top_companies)}
tracker_domain_to_company_idx = {
    row["tracking_domain_id"]: company_to_idx.get(row["company"], -1)
    for _, row in trackers_df.iterrows()
}

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
N_TRAIN = 250_000
N_VAL = 25_000

train_domain_ids = valid_candidates[permuted_indices[:N_TRAIN]]
val_domain_ids = valid_candidates[permuted_indices[N_TRAIN : N_TRAIN + N_VAL]]

train_set = set(train_domain_ids)
val_set = set(val_domain_ids)

assert (
    len(train_set.intersection(val_set)) == 0
), "Train and validation domains overlap!"
assert len(train_set.intersection(target_set)) == 0, "Train and test domains overlap!"
assert (
    len(val_set.intersection(target_set)) == 0
), "Validation and test domains overlap!"

# Multi-hot binary label matrices
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

del train_df_all, train_tracking_table, train_events, val_events

all_needed_domain_ids = np.unique(
    np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
)
all_needed_sorted = np.sort(all_needed_domain_ids)
N_ALL = len(all_needed_sorted)
needed_id_to_idx = {dom_id: i for i, dom_id in enumerate(all_needed_sorted)}

# =============================================================================
# 3. Stream Link Graph for Topology & Direct Tracker Bridge Features
# =============================================================================
in_degree = np.zeros(N_ALL, dtype=np.int32)
out_degree = np.zeros(N_ALL, dtype=np.int32)
direct_company_links = np.zeros((N_ALL, 32), dtype=np.int16)
direct_tracker_counts = np.zeros(N_ALL, dtype=np.int16)

tracker_domain_ids = np.array(list(tracker_domain_to_tracker_id.keys()), dtype=np.int64)
tracker_domain_ids_sorted = np.sort(tracker_domain_ids)
tracker_company_indices = np.array(
    [tracker_domain_to_company_idx[t_dom] for t_dom in tracker_domain_ids_sorted],
    dtype=np.int16,
)

link_pfile = pq.ParquetFile("./input/link-graph.parquet")
for batch in link_pfile.iter_batches(
    batch_size=2_000_000, columns=["source_domain_id", "target_domain_id"]
):
    src = np.asarray(batch["source_domain_id"])
    dst = np.asarray(batch["target_domain_id"])

    idx_src = np.searchsorted(all_needed_sorted, src)
    valid_src = (idx_src < N_ALL) & (
        all_needed_sorted[np.clip(idx_src, 0, N_ALL - 1)] == src
    )

    idx_dst = np.searchsorted(all_needed_sorted, dst)
    valid_dst = (idx_dst < N_ALL) & (
        all_needed_sorted[np.clip(idx_dst, 0, N_ALL - 1)] == dst
    )

    if np.any(valid_src):
        src_matched = idx_src[valid_src]
        counts_src = np.bincount(src_matched, minlength=N_ALL)
        out_degree += counts_src.astype(np.int32)

    if np.any(valid_dst):
        dst_matched = idx_dst[valid_dst]
        counts_dst = np.bincount(dst_matched, minlength=N_ALL)
        in_degree += counts_dst.astype(np.int32)

    if np.any(valid_src):
        dst_for_valid_src = dst[valid_src]
        src_for_valid = idx_src[valid_src]

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
            comp_idx = tracker_company_indices[matched_t_idx]

            np.add.at(direct_tracker_counts, matched_src_idx, 1)
            valid_comp = comp_idx >= 0
            if np.any(valid_comp):
                np.add.at(
                    direct_company_links,
                    (matched_src_idx[valid_comp], comp_idx[valid_comp]),
                    1,
                )

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


def get_tld(d):
    if not isinstance(d, str) or "." not in d:
        return ""
    return d.rsplit(".", 1)[-1].lower()


def compute_lexical_features(domain_str):
    if not isinstance(domain_str, str) or len(domain_str) == 0:
        return [0.0] * 10
    length = len(domain_str)
    dots = domain_str.count(".")
    hyphens = domain_str.count("-")
    digits = sum(c.isdigit() for c in domain_str)
    vowels = sum(c in "aeiouy" for c in domain_str)
    counts = {}
    for c in domain_str:
        counts[c] = counts.get(c, 0) + 1
    entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())
    return [
        float(length),
        float(dots),
        float(hyphens),
        float(digits),
        float(digits / max(1, length)),
        float(vowels / max(1, length)),
        float(entropy),
        1.0 if dots > 1 else 0.0,
        1.0 if domain_str.startswith("xn--") else 0.0,
        1.0 if any(c.isdigit() for c in domain_str) else 0.0,
    ]


# =============================================================================
# 6. Extract Multi-Modal Features
# =============================================================================
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

    tlds = [get_tld(name) for name in names]
    press_scores = np.array(
        [press_map.get(tld, median_press) for tld in tlds], dtype=np.float32
    )
    press_known = np.array(
        [1.0 if tld in press_map else 0.0 for tld in tlds], dtype=np.float32
    )

    scalars = np.column_stack(
        [
            lexical_arr,
            log_in,
            log_out,
            deg_ratio,
            dir_tr_counts,
            press_scores,
            press_known,
        ]
    )

    comp_links = np.log1p(direct_company_links[indices].astype(np.float32))

    cat_feats = np.zeros((len(dom_ids), len(unique_categories) + 1), dtype=np.float32)
    for i, name in enumerate(names):
        cat_idx = domain_to_category.get(name, len(unique_categories))
        cat_feats[i, cat_idx] = 1.0

    return names, tlds, scalars, comp_links, cat_feats


train_names, train_tlds, train_scalars, train_comp, train_cats = (
    build_split_raw_features(train_domain_ids)
)
val_names, val_tlds, val_scalars, val_comp, val_cats = build_split_raw_features(
    val_domain_ids
)
test_names, test_tlds, test_scalars, test_comp, test_cats = build_split_raw_features(
    test_domain_ids
)

# Standardize tabular scalars strictly on train
scaler = StandardScaler()
train_scalars_scaled = scaler.fit_transform(train_scalars).astype(np.float32)
val_scalars_scaled = scaler.transform(val_scalars).astype(np.float32)
test_scalars_scaled = scaler.transform(test_scalars).astype(np.float32)

# =============================================================================
# 7. TLD Categorical Encoding and Bayesian-Smoothed Tracker Prior (Train-Fit)
# =============================================================================
top_tlds = [t for t, _ in Counter(train_tlds).most_common(64)]
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

global_prior = y_train.mean(axis=0)
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

tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=10,
    max_features=256,
    sublinear_tf=True,
)
tfidf.fit(clean_train_names)
X_tfidf_train = tfidf.transform(clean_train_names).astype(np.float32).toarray()
X_tfidf_val = tfidf.transform(clean_val_names).astype(np.float32).toarray()
X_tfidf_test = tfidf.transform(clean_test_names).astype(np.float32).toarray()

# =============================================================================
# 9. Assemble Final Multi-Modal Feature Matrices
# =============================================================================
X_train = np.hstack(
    [
        train_scalars_scaled,
        train_tld_encoded,
        train_cats,
        train_comp,
        X_tfidf_train,
        train_priors,
    ]
).astype(np.float32)

X_val = np.hstack(
    [val_scalars_scaled, val_tld_encoded, val_cats, val_comp, X_tfidf_val, val_priors]
).astype(np.float32)

X_test = np.hstack(
    [
        test_scalars_scaled,
        test_tld_encoded,
        test_cats,
        test_comp,
        X_tfidf_test,
        test_priors,
    ]
).astype(np.float32)

assert (
    X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
), "Feature dimension mismatch!"
assert not np.isnan(X_train).any(), "NaN found in X_train!"
assert not np.isnan(X_val).any(), "NaN found in X_val!"
assert not np.isnan(X_test).any(), "NaN found in X_test!"

num_features = X_train.shape[1]
dim_tfidf = 256
dim_struct = num_features - dim_tfidf - num_trackers


# =============================================================================
# 10. Neural Architecture: TrackerCoOccurNet
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


class TrackerCoOccurNet(nn.Module):

    def __init__(
        self,
        num_features=741,
        num_trackers=355,
        dim_struct=130,
        dim_tfidf=256,
        hidden_dim=384,
        dropout=0.2,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_trackers = num_trackers
        self.dim_struct = dim_struct
        self.dim_tfidf = dim_tfidf

        self.struct_proj = nn.Sequential(
            nn.Linear(dim_struct, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
        )

        self.lexical_proj = nn.Sequential(
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

        self.tracker_prototypes = nn.Linear(hidden_dim, num_trackers)

        self.cooccur_matrix = nn.Parameter(
            torch.zeros(num_trackers, num_trackers) * 0.01
        )
        self.cooccur_gain = nn.Parameter(torch.tensor(0.1))

        self.prior_gate = nn.Sequential(
            nn.Linear(hidden_dim, num_trackers), nn.Sigmoid()
        )

    def forward(self, x):
        x_struct = x[:, : self.dim_struct]
        x_tfidf = x[:, self.dim_struct : self.dim_struct + self.dim_tfidf]
        x_prior = x[:, self.dim_struct + self.dim_tfidf :]

        h_struct = self.struct_proj(x_struct)
        h_lexical = self.lexical_proj(x_tfidf)

        h_fused = self.fusion_stem(torch.cat([h_struct, h_lexical], dim=-1))
        h_latent = self.res2(self.res1(h_fused))

        z_direct = self.tracker_prototypes(h_latent)

        cooccur_norm = torch.tanh(self.cooccur_matrix)
        z_refined = z_direct + self.cooccur_gain * torch.matmul(z_direct, cooccur_norm)

        p_clamped = torch.clamp(x_prior, 1e-4, 1.0 - 1e-4)
        logit_prior = torch.log(p_clamped) - torch.log(1.0 - p_clamped)

        alpha = self.prior_gate(h_latent)
        z_final = (1.0 - alpha) * z_refined + alpha * logit_prior

        return z_final


# =============================================================================
# 11. Objective Function: Vectorized Top10SoftMarginRankingLoss
# =============================================================================
class Top10SoftMarginRankingLoss(nn.Module):

    def __init__(
        self,
        top_k_neg=15,
        margin=0.6,
        temp=2.0,
        gamma_neg=2.0,
        alpha_focal=0.15,
        smooth_pos=0.05,
    ):
        super().__init__()
        self.top_k_neg = top_k_neg
        self.margin = margin
        self.temp = temp
        self.gamma_neg = gamma_neg
        self.alpha_focal = alpha_focal
        self.smooth_pos = smooth_pos

    def forward(self, logits, targets):
        batch_size, num_tr = logits.shape
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask

        probs = torch.sigmoid(logits)
        targets_smoothed = targets * (1.0 - self.smooth_pos) + 0.5 * (
            self.smooth_pos / num_tr
        )

        bce_pos = -targets_smoothed * torch.log(torch.clamp(probs, 1e-6, 1.0))
        focal_weight_neg = torch.pow(torch.clamp(probs, 0.0, 1.0), self.gamma_neg)
        bce_neg = (
            -(1.0 - targets_smoothed)
            * focal_weight_neg
            * torch.log(torch.clamp(1.0 - probs, 1e-6, 1.0))
        )
        focal_loss = (bce_pos + bce_neg).sum(dim=-1).mean()

        neg_logits = torch.where(
            neg_mask, logits, torch.tensor(-1e9, device=logits.device)
        )
        k_neg = min(self.top_k_neg, num_tr - 1)
        top_neg_logits, _ = torch.topk(neg_logits, k=k_neg, dim=-1)

        has_positives = pos_mask.any(dim=-1)
        if not has_positives.any():
            return focal_loss

        diff = top_neg_logits.unsqueeze(1) - logits.unsqueeze(2) + self.margin
        diff_clamped = torch.clamp(diff, max=12.0)
        soft_rank_penalty = (
            torch.log1p(torch.sum(torch.exp(self.temp * diff_clamped), dim=-1))
            / self.temp
        )

        pos_count = torch.clamp(pos_mask.float().sum(dim=-1), min=1.0)
        domain_rank_loss = (soft_rank_penalty * pos_mask.float()).sum(
            dim=-1
        ) / pos_count
        rank_loss = domain_rank_loss[has_positives].mean()

        total_loss = rank_loss + self.alpha_focal * focal_loss
        return total_loss


# =============================================================================
# 12. Training Pipeline & Exact Metric Evaluation
# =============================================================================
model = TrackerCoOccurNet(
    num_features=num_features,
    num_trackers=num_trackers,
    dim_struct=dim_struct,
    dim_tfidf=dim_tfidf,
    hidden_dim=384,
    dropout=0.2,
).to(device)

criterion = Top10SoftMarginRankingLoss(
    top_k_neg=15,
    margin=0.6,
    temp=2.0,
    gamma_neg=2.0,
    alpha_focal=0.15,
    smooth_pos=0.05,
)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.99),
    eps=1e-8,
)

num_epochs = 12
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

batch_size = 1024
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)


def evaluate_recall_at_10(eval_model, X_eval, y_eval, eval_batch_size=2048):
    eval_model.eval()
    all_recalls = []
    num_samples = len(X_eval)

    with torch.no_grad():
        for i in range(0, num_samples, eval_batch_size):
            batch_x = torch.from_numpy(X_eval[i : i + eval_batch_size]).to(device)
            batch_y = torch.from_numpy(y_eval[i : i + eval_batch_size]).to(device)

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

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

    scheduler.step()
    epoch_loss = running_loss / max(1, batch_count)
    val_recall = evaluate_recall_at_10(model, X_val, y_val)

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_val_recall:.5f}"
    )

if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

final_val_score = evaluate_recall_at_10(model, X_val, y_val)

# =============================================================================
# 13. Test Inference & Submission Generation
# =============================================================================
model.eval()
test_batch_size = 2048
test_predictions = []

with torch.no_grad():
    for i in range(0, len(X_test), test_batch_size):
        batch_x = torch.from_numpy(X_test[i : i + test_batch_size]).to(device)
        logits = model(batch_x)
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

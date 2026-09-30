from collections import Counter, defaultdict
import gc
import math
import os
import pickle
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def calc_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in counts.values())


def extract_host_from_url(url_str: str) -> str:
    """Extract clean hostname from URL string."""
    if not isinstance(url_str, str):
        return ""
    s = url_str.split("://", 1)[-1]
    host = s.split("/", 1)[0].split(":", 1)[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


class TrackerSyndicateNet(nn.Module):
    """Deep multi-modal architecture with tracker-aligned channel projection,

    relational syndicate graph message passing, tracker-specific dynamic FiLM modulation,
    and base-frequency logit calibration.
    """

    def __init__(
        self,
        num_trackers: int = 355,
        num_tracker_channels: int = 4,
        context_dim: int = 160,
        tracker_emb_dim: int = 24,
        context_hidden_dim: int = 256,
        syndicate_rank: int = 32,
        dropout_rate: float = 0.2,
        trackers_tsv_path: str = "input/trackers.tsv",
        prior_bias: torch.Tensor = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.num_tracker_channels = num_tracker_channels
        self.context_dim = context_dim
        self.tracker_emb_dim = tracker_emb_dim
        self.syndicate_rank = syndicate_rank

        # 1. Tracker-aligned Channel Feature Projection (applied per tracker)
        self.tracker_encoder = nn.Sequential(
            nn.Linear(num_tracker_channels, 32),
            nn.GELU(),
            nn.Linear(32, tracker_emb_dim),
            nn.LayerNorm(tracker_emb_dim),
        )

        # 2. Build corporate and categorical syndicate graph adjacency matrix A_synd (355x355)
        A_raw = np.eye(num_trackers, dtype=np.float32) * 2.0
        if os.path.exists(trackers_tsv_path):
            tdf = pd.read_csv(trackers_tsv_path, sep="\t")
            tid_to_company = {}
            tid_to_category = {}
            tid_to_brand = {}
            for _, row in tdf.iterrows():
                tid = int(row["tracker_id"])
                if tid < num_trackers:
                    tid_to_company[tid] = (
                        str(row["company"]).strip() if pd.notna(row["company"]) else ""
                    )
                    tid_to_category[tid] = (
                        str(row["category"]).strip() if pd.notna(row["category"]) else ""
                    )
                    tid_to_brand[tid] = (
                        str(row["brand"]).strip() if pd.notna(row["brand"]) else ""
                    )

            for i in range(num_trackers):
                comp_i = tid_to_company.get(i, "")
                cat_i = tid_to_category.get(i, "")
                br_i = tid_to_brand.get(i, "")
                for j in range(i + 1, num_trackers):
                    comp_j = tid_to_company.get(j, "")
                    cat_j = tid_to_category.get(j, "")
                    br_j = tid_to_brand.get(j, "")
                    w = 0.0
                    if comp_i and comp_j and comp_i != "#" and comp_i == comp_j:
                        w += 2.0
                    if br_i and br_j and br_i != "#" and br_i == br_j:
                        w += 1.5
                    if cat_i and cat_j and cat_i != "#" and cat_i == cat_j:
                        w += 0.5
                    if w > 0.0:
                        A_raw[i, j] = w
                        A_raw[j, i] = w

        # Row-normalized adjacency matrix: D^{-1} A
        row_sums = A_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A_norm = A_raw / row_sums
        self.register_buffer("A_synd", torch.from_numpy(A_norm).float())

        # 3. Latent Relational Graph Message Passing
        self.W_rel = nn.Linear(tracker_emb_dim, tracker_emb_dim, bias=False)
        self.rel_dropout = nn.Dropout(dropout_rate)
        self.rel_norm = nn.LayerNorm(tracker_emb_dim)

        # 4. Global Domain Context Encoder (lexical TF-IDF + centrality/metadata)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(context_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.75),
        )

        # 5. Tracker-specific Dynamic FiLM Modulation
        self.film_generator = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2 * tracker_emb_dim),
        )
        self.tracker_film_scale = nn.Parameter(
            torch.ones(1, num_trackers, tracker_emb_dim)
        )
        self.tracker_film_bias = nn.Parameter(
            torch.zeros(1, num_trackers, tracker_emb_dim)
        )

        # 6. Tracker Prediction Heads
        self.tracker_head = nn.Sequential(
            nn.Linear(tracker_emb_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.context_direct_head = nn.Linear(128, num_trackers)

        # 7. Learnable Low-Rank Tracker Syndicate Interaction Matrix
        self.syndicate_U = nn.Parameter(
            torch.randn(num_trackers, syndicate_rank) / math.sqrt(syndicate_rank)
        )
        self.syndicate_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # 8. Tracker Base-Frequency Prior Logit Bias
        if prior_bias is not None:
            self.prior_bias = nn.Parameter(prior_bias.float().clone().detach())
        else:
            self.prior_bias = nn.Parameter(torch.zeros(num_trackers, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self, "film_generator"):
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
        if hasattr(self, "tracker_head"):
            nn.init.normal_(self.tracker_head[-1].weight, std=0.01)
            if self.tracker_head[-1].bias is not None:
                nn.init.zeros_(self.tracker_head[-1].bias)
        if hasattr(self, "context_direct_head"):
            nn.init.normal_(self.context_direct_head.weight, std=0.01)
            if self.context_direct_head.bias is not None:
                nn.init.zeros_(self.context_direct_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        split_idx = self.num_trackers * self.num_tracker_channels

        # Split tracker-aligned channels vs domain context
        x_trackers_flat = x[:, :split_idx]  # [B, 4 * 355]
        x_context = x[:, split_idx:]  # [B, context_dim]

        # Reshape to [B, num_trackers, num_tracker_channels]
        x_trackers = x_trackers_flat.view(
            batch_size, self.num_tracker_channels, self.num_trackers
        ).permute(0, 2, 1)

        # Encode tracker-aligned signals -> H: [B, num_trackers, tracker_emb_dim]
        H = self.tracker_encoder(x_trackers)

        # Relational message passing over corporate syndicate graph:
        # H_rel = LayerNorm(H + Dropout(GELU(A_synd @ H @ W_rel)))
        H_aggr = torch.matmul(self.A_synd.unsqueeze(0), H)
        H_proj = self.W_rel(H_aggr)
        H_rel = self.rel_norm(H + self.rel_dropout(F.gelu(H_proj)))

        # Encode global domain context -> [B, 128]
        c_emb = self.context_encoder(x_context)

        # Dynamic FiLM scale and shift from domain context
        film = self.film_generator(c_emb)  # [B, 2 * tracker_emb_dim]
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # [B, 1, tracker_emb_dim]
        beta = beta.unsqueeze(1)  # [B, 1, tracker_emb_dim]

        # Tracker-specific dynamic FiLM modulation
        H_mod = (H_rel * self.tracker_film_scale + self.tracker_film_bias) * (1.0 + gamma) + beta

        # Tracker logits from modulated representation and direct context
        z_base = self.tracker_head(H_mod).squeeze(-1)  # [B, num_trackers]
        z_ctx = self.context_direct_head(c_emb)  # [B, num_trackers]
        z_comb = z_base + z_ctx  # [B, num_trackers]

        # Tracker Syndicate Diffusion via Low-Rank Covariance
        syndicate_matrix = torch.matmul(
            self.syndicate_U, self.syndicate_U.t()
        ) / math.sqrt(self.syndicate_rank)
        z_synd = torch.matmul(z_comb, syndicate_matrix)  # [B, num_trackers]

        logits = z_comb + self.syndicate_alpha * z_synd + self.prior_bias
        return logits


class CombinedRankingLoss(nn.Module):
    """Numerically stable composite loss combining weighted Binary Cross-Entropy

    for posterior calibration and ListNet permutation cross-entropy for
    direct within-domain Top-K ranking in O(B * K) complexity.
    """

    def __init__(self, pos_weight_val: float = 12.0, rank_weight: float = 2.0):
        super().__init__()
        self.pos_weight_val = pos_weight_val
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Pointwise BCE with positive weighting (calibrated posteriors)
        pos_weight = torch.tensor(
            self.pos_weight_val, device=logits.device, dtype=logits.dtype
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )

        # 2. ListNet permutation ranking loss over within-domain candidate trackers
        log_probs = F.log_softmax(logits, dim=-1)
        pos_sums = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
        target_dist = targets / pos_sums
        pos_mask = (targets.sum(dim=-1) > 0).float()

        # Cross-entropy between normalized ground-truth tracker distribution and predicted distribution
        sample_listnet = -(target_dist * log_probs).sum(dim=-1)
        listnet_loss = (sample_listnet * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

        return bce_loss + self.rank_weight * listnet_loss


def compute_recall_at_10(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact competition metric: Recall@10 averaged across all domains."""
    with torch.no_grad():
        top10_indices = torch.topk(logits, k=10, dim=1).indices
        hits = torch.gather(targets, dim=1, index=top10_indices).sum(dim=1)
        pos_counts = targets.sum(dim=1).clamp(min=1.0)
        sample_recall = hits / pos_counts
        return float(sample_recall.mean().item())


def main():
    global_start_time = time.time()

    # Set random seeds for deterministic execution
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Starting leak-free data processing and feature engineering pipeline...")

    # ---------------------------------------------------------
    # 1. Load Trackers & Metadata
    # ---------------------------------------------------------
    trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
    num_trackers = 355
    tracker_ids = trackers_df["tracker_id"].values.astype(np.int32)
    tracking_domain_ids = trackers_df["tracking_domain_id"].values.astype(np.int64)

    tracker_id_to_domain_id = np.zeros(num_trackers, dtype=np.int64)
    domain_id_to_tracker_id = {}
    for tid, did in zip(tracker_ids, tracking_domain_ids):
        tracker_id_to_domain_id[tid] = did
        domain_id_to_tracker_id[did] = tid

    # ---------------------------------------------------------
    # 2. Split Domains: Train, Holdout Validation, and Test
    # ---------------------------------------------------------
    tracking_train_df = pd.read_parquet("input/tracking_graph_train.parquet")
    all_train_domains = (
        tracking_train_df["domain_id"].drop_duplicates().values.astype(np.int64)
    )

    rng = np.random.RandomState(42)
    shuffled_domains = rng.permutation(all_train_domains)
    val_size = 25000
    val_domain_ids = shuffled_domains[:val_size]
    train_domain_ids = shuffled_domains[val_size:]

    test_target_df = pd.read_csv("input/target.tsv", sep="\t")
    test_domain_ids = test_target_df["domain_id"].values.astype(np.int64)

    print(
        f"Domain Split -> Train: {len(train_domain_ids)}, Val:"
        f" {len(val_domain_ids)}, Test: {len(test_domain_ids)}"
    )

    # ---------------------------------------------------------
    # 3. Construct Target Label Matrices (Train & Val)
    # ---------------------------------------------------------
    train_domain_to_row = {d: i for i, d in enumerate(train_domain_ids)}
    val_domain_to_row = {d: i for i, d in enumerate(val_domain_ids)}

    y_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.uint8)
    y_val = np.zeros((len(val_domain_ids), num_trackers), dtype=np.uint8)

    train_mask = np.isin(
        tracking_train_df["domain_id"],
        train_domain_ids,
        assume_unique=False,
    )
    val_mask = np.isin(
        tracking_train_df["domain_id"],
        val_domain_ids,
        assume_unique=False,
    )

    train_edges = tracking_train_df[train_mask]
    val_edges = tracking_train_df[val_mask]

    train_rows = train_edges["domain_id"].map(train_domain_to_row).to_numpy()
    train_cols = train_edges["tracker_id"].to_numpy()
    y_train[train_rows, train_cols] = 1

    val_rows = val_edges["domain_id"].map(val_domain_to_row).to_numpy()
    val_cols = val_edges["tracker_id"].to_numpy()
    y_val[val_rows, val_cols] = 1

    del tracking_train_df, train_edges, val_edges, train_rows, val_rows
    gc.collect()

    # ---------------------------------------------------------
    # 4. Domain Lookup & Graph Subgraph Indexing
    # ---------------------------------------------------------
    all_relevant_domain_ids = np.concatenate(
        [train_domain_ids, val_domain_ids, test_domain_ids]
    )
    num_rel = len(all_relevant_domain_ids)

    domains_df = pd.read_parquet(
        "input/domains.parquet", columns=["domain_id", "domain"]
    )
    max_domain_id = max(
        int(domains_df["domain_id"].max()),
        int(all_relevant_domain_ids.max()),
        int(tracking_domain_ids.max()),
    )

    id_to_idx = np.full(max_domain_id + 1, -1, dtype=np.int32)
    id_to_idx[all_relevant_domain_ids] = np.arange(num_rel, dtype=np.int32)

    tracker_domain_to_id = np.full(max_domain_id + 1, -1, dtype=np.int16)
    tracker_domain_to_id[tracking_domain_ids] = tracker_ids.astype(np.int16)

    set_relevant_ids = set(all_relevant_domain_ids)
    relevant_domains_df = domains_df[domains_df["domain_id"].isin(set_relevant_ids)]
    domain_dict = dict(
        zip(relevant_domains_df["domain_id"], relevant_domains_df["domain"])
    )
    del domains_df, relevant_domains_df
    gc.collect()

    # ---------------------------------------------------------
    # 5. Stream Link-Graph: Degrees, Tracker Links & Adjacency
    # ---------------------------------------------------------
    print("Streaming link-graph.parquet for topology & tracker signals...")
    global_out_degree = np.zeros(num_rel, dtype=np.int32)
    global_in_degree = np.zeros(num_rel, dtype=np.int32)

    rel_src_list = []
    rel_dst_list = []
    tr_src_list = []
    tr_id_list = []

    link_graph_pq = pq.ParquetFile("input/link-graph.parquet")
    for batch in link_graph_pq.iter_batches(
        batch_size=3000000, columns=["source_domain_id", "target_domain_id"]
    ):
        src = batch["source_domain_id"].to_numpy(zero_copy_only=False)
        dst = batch["target_domain_id"].to_numpy(zero_copy_only=False)

        bound_mask = (src <= max_domain_id) & (dst <= max_domain_id)
        if not np.all(bound_mask):
            src = src[bound_mask]
            dst = dst[bound_mask]

        src_idx = id_to_idx[src]
        dst_idx = id_to_idx[dst]

        src_valid = src_idx >= 0
        dst_valid = dst_idx >= 0

        if np.any(src_valid):
            np.add.at(global_out_degree, src_idx[src_valid], 1)
        if np.any(dst_valid):
            np.add.at(global_in_degree, dst_idx[dst_valid], 1)

        tr_ids = tracker_domain_to_id[dst]
        tr_mask = src_valid & (tr_ids >= 0)
        if np.any(tr_mask):
            tr_src_list.append(src_idx[tr_mask])
            tr_id_list.append(tr_ids[tr_mask])

        rel_mask = src_valid & dst_valid
        if np.any(rel_mask):
            rel_src_list.append(src_idx[rel_mask])
            rel_dst_list.append(dst_idx[rel_mask])

    print("Link graph streaming completed.")

    # ---------------------------------------------------------
    # 6. Direct Tracker Links Matrix
    # ---------------------------------------------------------
    if tr_src_list:
        all_tr_src = np.concatenate(tr_src_list)
        all_tr_id = np.concatenate(tr_id_list)
        direct_tracker_csr = csr_matrix(
            (np.ones(len(all_tr_src), dtype=np.float32), (all_tr_src, all_tr_id)),
            shape=(num_rel, num_trackers),
        )
        direct_tracker_links = np.minimum(direct_tracker_csr.toarray(), 1.0).astype(
            np.float32
        )
        tracker_link_counts = np.array(direct_tracker_csr.sum(axis=1)).flatten()
    else:
        direct_tracker_links = np.zeros((num_rel, num_trackers), dtype=np.float32)
        tracker_link_counts = np.zeros(num_rel, dtype=np.float32)

    del tr_src_list, tr_id_list
    gc.collect()

    # ---------------------------------------------------------
    # 7. Collaborative Neighborhood Tracker Diffusion (Zero-Leakage)
    # ---------------------------------------------------------
    print("Computing collaborative neighborhood tracker diffusion...")
    if rel_src_list:
        all_rel_src = np.concatenate(rel_src_list)
        all_rel_dst = np.concatenate(rel_dst_list)
        del rel_src_list, rel_dst_list
        gc.collect()

        non_self_mask = all_rel_src != all_rel_dst
        all_rel_src = all_rel_src[non_self_mask]
        all_rel_dst = all_rel_dst[non_self_mask]

        adj_matrix = csr_matrix(
            (np.ones(len(all_rel_src), dtype=np.float32), (all_rel_src, all_rel_dst)),
            shape=(num_rel, num_rel),
        )
        adj_matrix.sum_duplicates()
        adj_matrix.data = np.clip(adj_matrix.data, 0.0, 1.0)
    else:
        adj_matrix = csr_matrix((num_rel, num_rel), dtype=np.float32)

    # Populated strictly for training domains
    y_train_full = np.zeros((num_rel, num_trackers), dtype=np.float32)
    train_indices_in_rel = id_to_idx[train_domain_ids]
    y_train_full[train_indices_in_rel] = y_train.astype(np.float32)

    is_train_domain_mask = np.zeros((num_rel, 1), dtype=np.float32)
    is_train_domain_mask[train_indices_in_rel] = 1.0

    out_tracker_sum = adj_matrix.dot(y_train_full)
    out_train_counts = adj_matrix.dot(is_train_domain_mask)
    out_tracker_dist = (out_tracker_sum / np.maximum(out_train_counts, 1.0)).astype(
        np.float32
    )

    in_tracker_sum = adj_matrix.T.dot(y_train_full)
    in_train_counts = adj_matrix.T.dot(is_train_domain_mask)
    in_tracker_dist = (in_tracker_sum / np.maximum(in_train_counts, 1.0)).astype(
        np.float32
    )

    subgraph_out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    subgraph_in_degree = np.array(adj_matrix.sum(axis=0)).flatten()

    del adj_matrix, y_train_full, out_tracker_sum, in_tracker_sum
    gc.collect()

    # ---------------------------------------------------------
    # 8. TLD Parsing & Bayesian Smoothed Regional Priors
    # ---------------------------------------------------------
    print("Constructing Bayesian smoothed TLD regional priors...")
    domain_tlds = []
    domain_lens = np.zeros(num_rel, dtype=np.float32)
    subdomain_counts = np.zeros(num_rel, dtype=np.float32)
    hyphen_counts = np.zeros(num_rel, dtype=np.float32)
    digit_ratios = np.zeros(num_rel, dtype=np.float32)
    vowel_ratios = np.zeros(num_rel, dtype=np.float32)
    entropies = np.zeros(num_rel, dtype=np.float32)
    domain_names_list = []

    vowels = set("aeiou")
    for i, did in enumerate(all_relevant_domain_ids):
        name = domain_dict.get(did, "")
        domain_names_list.append(name)
        n_len = len(name)
        domain_lens[i] = float(n_len)
        if n_len > 0:
            tld = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            subdomain_counts[i] = float(name.count("."))
            hyphen_counts[i] = float(name.count("-"))
            digit_count = sum(c.isdigit() for c in name)
            digit_ratios[i] = float(digit_count / n_len)
            vowel_count = sum(c in vowels for c in name)
            vowel_ratios[i] = float(vowel_count / n_len)
            entropies[i] = calc_entropy(name)
        else:
            tld = ""
        domain_tlds.append(tld)

    train_tld_list = [domain_tlds[i] for i in train_indices_in_rel]
    tld_counts = Counter(train_tld_list)
    tld_tracker_sums = defaultdict(lambda: np.zeros(num_trackers, dtype=np.float32))
    for idx_in_train, rel_idx in enumerate(train_indices_in_rel):
        tld_tracker_sums[domain_tlds[rel_idx]] += y_train[idx_in_train]

    global_base_rate = (y_train.sum(axis=0).astype(np.float32) + 1.0) / (
        len(train_domain_ids) + num_trackers
    )
    alpha = 20.0

    tld_prior_cache = {}
    for tld, count in tld_counts.items():
        tld_prior_cache[tld] = (
            (tld_tracker_sums[tld] + alpha * global_base_rate) / (count + alpha)
        ).astype(np.float32)

    tld_tracker_prior = np.zeros((num_rel, num_trackers), dtype=np.float32)
    for i, tld in enumerate(domain_tlds):
        if tld in tld_prior_cache:
            tld_tracker_prior[i] = tld_prior_cache[tld]
        else:
            tld_tracker_prior[i] = global_base_rate

    # ---------------------------------------------------------
    # 9. Freedom of the Press Score Integration
    # ---------------------------------------------------------
    press_df = pd.read_csv(
        "input/freedom-of-the-press.csv", sep=r"\t|,", engine="python"
    )
    press_df.columns = [c.strip() for c in press_df.columns]
    tld_c = [c for c in press_df.columns if "tld" in c.lower()][0]
    score_c = [c for c in press_df.columns if "freedom" in c.lower()][0]

    press_dict = dict(
        zip(
            press_df[tld_c].str.strip().str.lower(),
            press_df[score_c].astype(np.float32),
        )
    )
    median_press = float(np.median(list(press_dict.values())))

    press_scores = np.zeros(num_rel, dtype=np.float32)
    has_press = np.zeros(num_rel, dtype=np.float32)
    for i, tld in enumerate(domain_tlds):
        if tld in press_dict:
            press_scores[i] = press_dict[tld]
            has_press[i] = 1.0
        else:
            press_scores[i] = median_press
            has_press[i] = 0.0

    # ---------------------------------------------------------
    # 10. URL Classification Category Integration
    # ---------------------------------------------------------
    url_cat_df = pd.read_csv(
        "input/url-classification.csv", usecols=["url", "category"]
    )
    url_cat_df["host"] = url_cat_df["url"].apply(extract_host_from_url)
    url_cat_df = url_cat_df[url_cat_df["host"].str.len() > 0]

    target_hosts = set(url_cat_df["host"].unique())
    host_to_rel_idx = {}
    for i, name in enumerate(domain_names_list):
        if name:
            lname = name.lower()
            if lname in target_hosts:
                host_to_rel_idx[lname] = i

    url_cat_df["rel_idx"] = url_cat_df["host"].map(host_to_rel_idx)
    matched_cats = url_cat_df.dropna(subset=["rel_idx"]).copy()
    matched_cats["rel_idx"] = matched_cats["rel_idx"].astype(np.int32)

    all_cats = sorted(url_cat_df["category"].dropna().unique())
    cat_to_col = {c: idx for idx, c in enumerate(all_cats)}
    num_cats = len(all_cats)

    url_category_features = np.zeros((num_rel, num_cats), dtype=np.float32)
    has_url_category = np.zeros(num_rel, dtype=np.float32)

    if len(matched_cats) > 0:
        for rel_idx, cat in zip(matched_cats["rel_idx"], matched_cats["category"]):
            if cat in cat_to_col:
                url_category_features[rel_idx, cat_to_col[cat]] += 1.0
                has_url_category[rel_idx] = 1.0

        row_sums = url_category_features.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        url_category_features = (url_category_features / row_sums).astype(np.float32)

    del url_cat_df, matched_cats
    gc.collect()

    # ---------------------------------------------------------
    # 11. Subword Character N-Gram TF-IDF (Fit ONLY on Train)
    # ---------------------------------------------------------
    print("Fitting lexical subword TF-IDF representations...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=128,
        sublinear_tf=True,
    )

    train_names = [domain_names_list[i] for i in train_indices_in_rel]
    tfidf_vectorizer.fit(train_names[:500000])

    val_indices_in_rel = id_to_idx[val_domain_ids]
    test_indices_in_rel = id_to_idx[test_domain_ids]

    val_names = [domain_names_list[i] for i in val_indices_in_rel]
    test_names = [domain_names_list[i] for i in test_indices_in_rel]

    tfidf_train = tfidf_vectorizer.transform(train_names).astype(np.float32).toarray()
    tfidf_val = tfidf_vectorizer.transform(val_names).astype(np.float32).toarray()
    tfidf_test = tfidf_vectorizer.transform(test_names).astype(np.float32).toarray()

    # ---------------------------------------------------------
    # 12. Dense Tabular Block Assembly & Standard Scaling
    # ---------------------------------------------------------
    tabular_raw = np.column_stack(
        [
            np.log1p(global_out_degree),
            np.log1p(global_in_degree),
            (global_in_degree + 1.0) / (global_out_degree + 1.0),
            np.log1p(subgraph_out_degree),
            np.log1p(subgraph_in_degree),
            np.log1p(tracker_link_counts),
            np.log1p(out_train_counts.flatten()),
            np.log1p(in_train_counts.flatten()),
            domain_lens,
            subdomain_counts,
            hyphen_counts,
            digit_ratios,
            vowel_ratios,
            entropies,
            press_scores,
            has_press,
            has_url_category,
            url_category_features,
        ]
    ).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(tabular_raw[train_indices_in_rel])

    tabular_train = scaler.transform(tabular_raw[train_indices_in_rel]).astype(
        np.float32
    )
    tabular_val = scaler.transform(tabular_raw[val_indices_in_rel]).astype(np.float32)
    tabular_test = scaler.transform(tabular_raw[test_indices_in_rel]).astype(np.float32)

    del tabular_raw
    gc.collect()

    # ---------------------------------------------------------
    # 13. Assemble Final Multi-Modal Feature Matrices
    # ---------------------------------------------------------
    print("Assembling final feature matrices...")
    X_train = np.column_stack(
        [
            direct_tracker_links[train_indices_in_rel],
            out_tracker_dist[train_indices_in_rel],
            in_tracker_dist[train_indices_in_rel],
            tld_tracker_prior[train_indices_in_rel],
            tfidf_train,
            tabular_train,
        ]
    )

    X_val = np.column_stack(
        [
            direct_tracker_links[val_indices_in_rel],
            out_tracker_dist[val_indices_in_rel],
            in_tracker_dist[val_indices_in_rel],
            tld_tracker_prior[val_indices_in_rel],
            tfidf_val,
            tabular_val,
        ]
    )

    X_test = np.column_stack(
        [
            direct_tracker_links[test_indices_in_rel],
            out_tracker_dist[test_indices_in_rel],
            in_tracker_dist[test_indices_in_rel],
            tld_tracker_prior[test_indices_in_rel],
            tfidf_test,
            tabular_test,
        ]
    )

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Free large intermediate arrays immediately to eliminate ~110 GB of redundant memory
    del direct_tracker_links, out_tracker_dist, in_tracker_dist, tld_tracker_prior
    del tfidf_train, tfidf_val, tfidf_test, tabular_train, tabular_val, tabular_test
    gc.collect()

    num_features = X_train.shape[1]
    num_tracker_channels = 4
    context_dim = num_features - (num_trackers * num_tracker_channels)

    print(
        f"Feature dimensions -> Train: {X_train.shape}, Val: {X_val.shape}, Test:"
        f" {X_test.shape}"
    )
    print(f"Context feature dimension: {context_dim}")

    # ---------------------------------------------------------
    # 14. Setup PyTorch DataLoaders & Architecture
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on compute device: {device}")

    # Fast vectorized in-memory batch loader (avoids multiprocessing IPC queue bottleneck)
    class FastBatchLoader:
        def __init__(self, X, y=None, batch_size=4096, shuffle=True):
            self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
            self.y = (
                torch.from_numpy(y)
                if (isinstance(y, np.ndarray) and y is not None)
                else y
            )
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.n_samples = len(self.X)

        def __iter__(self):
            if self.shuffle:
                indices = torch.randperm(self.n_samples)
            else:
                indices = torch.arange(self.n_samples)
            for i in range(0, self.n_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                bx = self.X[batch_idx]
                if self.y is not None:
                    by = self.y[batch_idx]
                    yield bx, by
                else:
                    yield bx, None

        def __len__(self):
            return (self.n_samples + self.batch_size - 1) // self.batch_size

    batch_size = 8192

    train_loader = FastBatchLoader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    val_loader = FastBatchLoader(
        X_val, y_val.astype(np.float32), batch_size=batch_size * 2, shuffle=False
    )
    test_loader = FastBatchLoader(
        X_test, None, batch_size=batch_size * 2, shuffle=False
    )

    # Compute empirical log-odds prior bias for tracker logit calibration
    p_clipped = np.clip(global_base_rate, 1e-6, 1.0 - 1e-6)
    prior_logits_np = np.log(p_clipped / (1.0 - p_clipped))
    prior_bias_tensor = torch.from_numpy(prior_logits_np.astype(np.float32))

    model = TrackerSyndicateNet(
        num_trackers=num_trackers,
        num_tracker_channels=num_tracker_channels,
        context_dim=context_dim,
        tracker_emb_dim=24,
        context_hidden_dim=256,
        syndicate_rank=32,
        dropout_rate=0.2,
        trackers_tsv_path="input/trackers.tsv",
        prior_bias=prior_bias_tensor,
    ).to(device)

    criterion = CombinedRankingLoss(pos_weight_val=12.0, rank_weight=2.0).to(device)

    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.10 * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        min_factor = 1e-5 / 2.0e-3
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------------------------------------------------------
    # 15. Training & Hold-out Metric Evaluation
    # ---------------------------------------------------------
    best_val_recall = -1.0
    best_epoch = 0
    checkpoint_path = "./working/best_tracker_model.pt"
    MAX_TOTAL_RUNTIME = 48 * 60  # Calibrated 48-min global timeout guard (hard limit 60m)

    for epoch in range(epochs):
        elapsed_global = time.time() - global_start_time
        if elapsed_global > MAX_TOTAL_RUNTIME:
            print(
                f"Global time guard triggered before epoch {epoch+1} ({elapsed_global:.1f}s elapsed). Stopping training gracefully."
            )
            break

        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        time_to_stop = False

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            if (batch_idx + 1) % 500 == 0:
                if (time.time() - global_start_time) > MAX_TOTAL_RUNTIME:
                    print(
                        f"Time guard triggered at batch {batch_idx + 1}/{len(train_loader)}. Finishing current epoch safely."
                    )
                    time_to_stop = True
                    break

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Validation Evaluation
        model.eval()
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(batch_x)
                val_logits_list.append(logits.cpu())
                val_targets_list.append(batch_y)

        all_val_logits = torch.cat(val_logits_list, dim=0)
        all_val_targets = torch.cat(val_targets_list, dim=0)
        val_recall = compute_recall_at_10(all_val_logits, all_val_targets)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {avg_train_loss:.4f} -"
            f" Val Recall@10: {val_recall:.6f} - LR: {current_lr:.6f}"
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_recall": val_recall,
                },
                checkpoint_path,
            )

        if time_to_stop:
            break

    print(
        f"Training completed. Optimal checkpoint at Epoch {best_epoch} with Val"
        f" Recall@10: {best_val_recall:.6f}"
    )

    # ---------------------------------------------------------
    # 16. Load Optimal Checkpoint & Final Validation Assessment
    # ---------------------------------------------------------
    del train_loader, X_train, y_train
    gc.collect()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_logits_list = []
    val_targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch_x)
            val_logits_list.append(logits.cpu())
            val_targets_list.append(batch_y)

    final_val_logits = torch.cat(val_logits_list, dim=0)
    final_val_targets = torch.cat(val_targets_list, dim=0)
    final_val_score = compute_recall_at_10(final_val_logits, final_val_targets)

    # ---------------------------------------------------------
    # 17. Test Inference & Submission Generation
    # ---------------------------------------------------------
    print("Performing inference on all test domains...")
    test_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch_x)
            test_logits_list.append(logits.cpu())

    all_test_logits = torch.cat(test_logits_list, dim=0)
    top10_test_tids = (
        torch.topk(all_test_logits, k=10, dim=1).indices.numpy().astype(np.int32)
    )

    top10_test_tracking_dids = tracker_id_to_domain_id[top10_test_tids]

    repeated_domain_ids = np.repeat(test_domain_ids, 10)
    flat_tracking_domain_ids = top10_test_tracking_dids.flatten()

    submission_df = pd.DataFrame(
        {
            "domain_id": repeated_domain_ids,
            "tracking_domain_id": flat_tracking_domain_ids,
        }
    )

    submission_file = "./submission/submission.csv"
    submission_tsv = "./submission/submission.tsv"

    submission_df.to_csv(submission_file, sep="\t", index=False)
    submission_df.to_csv(submission_tsv, sep="\t", index=False)

    print(f"Submission generated successfully with shape: {submission_df.shape}")
    print(f"Unique test domains submitted: {submission_df['domain_id'].nunique()}")

    print(f"Final Validation Score: {final_val_score:.6f}")


if __name__ == "__main__":
    main()

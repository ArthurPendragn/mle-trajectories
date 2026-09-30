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
    """Dual-Stream Residual Architecture with Dynamic Latent Syndicate Factorization Bottleneck,
    context-conditioned FiLM per-tracker channel gates, tracker identity & metadata embeddings,
    two-tower semantic interaction head, and domain-conditioned coalition dynamics.
    """

    def __init__(
        self,
        num_trackers: int = 355,
        num_tracker_channels: int = 5,
        context_dim: int = 416,
        tracker_emb_dim: int = 24,
        context_hidden_dim: int = 512,
        syndicate_rank: int = 64,
        dropout_rate: float = 0.10,
        trackers_tsv_path: str = "input/trackers.tsv",
        cond_cooccur_matrix: np.ndarray = None,
        ppmi_matrix: np.ndarray = None,
        cooccur_matrix: np.ndarray = None,
        prior_log_odds: np.ndarray = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.num_tracker_channels = num_tracker_channels
        self.context_dim = context_dim
        self.latent_dim = syndicate_rank
        self.syndicate_rank = syndicate_rank

        # 1. Load trackers metadata to construct identity & categorical embeddings
        trackers_df = pd.read_csv(trackers_tsv_path, sep="\t")
        trackers_df = trackers_df.sort_values("tracker_id").reset_index(drop=True)

        companies = trackers_df["company"].fillna("Unknown").astype(str).values
        categories = trackers_df["category"].fillna("Unknown").astype(str).values
        countries = trackers_df["country"].fillna("Unknown").astype(str).values
        brands = (
            trackers_df["brand"].fillna("Unknown").astype(str).values
            if "brand" in trackers_df.columns
            else companies
        )

        unique_comps, comp_idx = np.unique(companies, return_inverse=True)
        unique_cats, cat_idx = np.unique(categories, return_inverse=True)
        unique_ctrys, ctry_idx = np.unique(countries, return_inverse=True)

        id_dim = 16
        comp_dim = 8
        cat_dim = 4
        ctry_dim = 4
        static_dim = id_dim + comp_dim + cat_dim + ctry_dim  # 32

        self.tracker_id_emb = nn.Embedding(num_trackers, id_dim)
        self.comp_emb = nn.Embedding(len(unique_comps), comp_dim)
        self.cat_emb = nn.Embedding(len(unique_cats), cat_dim)
        self.ctry_emb = nn.Embedding(len(unique_ctrys), ctry_dim)

        self.register_buffer("comp_indices", torch.tensor(comp_idx, dtype=torch.long))
        self.register_buffer("cat_indices", torch.tensor(cat_idx, dtype=torch.long))
        self.register_buffer("ctry_indices", torch.tensor(ctry_idx, dtype=torch.long))
        self.register_buffer(
            "tracker_indices", torch.arange(num_trackers, dtype=torch.long)
        )

        # Construct Corporate Co-Membership Adjacency Prior
        corp_adj_np = np.zeros((num_trackers, num_trackers), dtype=np.float32)
        invalid_tags = {"#", "", "unknown", "none"}
        for i in range(num_trackers):
            for j in range(num_trackers):
                if i == j:
                    continue
                c_i, c_j = companies[i].strip().lower(), companies[j].strip().lower()
                b_i, b_j = brands[i].strip().lower(), brands[j].strip().lower()
                same_comp = (c_i == c_j) and (c_i not in invalid_tags)
                same_brand = (b_i == b_j) and (b_i not in invalid_tags)
                if same_comp or same_brand:
                    corp_adj_np[i, j] = 1.0

        row_sums = corp_adj_np.sum(axis=1, keepdims=True)
        corp_adj_norm = np.where(
            row_sums > 0, corp_adj_np / np.maximum(row_sums, 1.0), 0.0
        ).astype(np.float32)
        self.register_buffer(
            "corp_adj", torch.tensor(corp_adj_norm, dtype=torch.float32)
        )

        # Register Empirical Relational Topologies, 2-hop propagation, and Log-Odds Priors
        if cond_cooccur_matrix is None and cooccur_matrix is not None:
            cond_cooccur_matrix = cooccur_matrix
        if cond_cooccur_matrix is None:
            cond_cooccur_matrix = np.zeros((num_trackers, num_trackers), dtype=np.float32)
        if ppmi_matrix is None:
            ppmi_matrix = np.zeros((num_trackers, num_trackers), dtype=np.float32)
        if prior_log_odds is None:
            prior_log_odds = np.zeros(num_trackers, dtype=np.float32)

        p_cond_t = torch.tensor(cond_cooccur_matrix, dtype=torch.float32)
        p_ppmi_t = torch.tensor(ppmi_matrix, dtype=torch.float32)
        p_cond2_t = torch.matmul(p_cond_t, p_cond_t)
        eye_t = torch.eye(num_trackers)
        p_cond_t = p_cond_t * (1.0 - eye_t)
        p_ppmi_t = p_ppmi_t * (1.0 - eye_t)
        p_cond2_t = p_cond2_t * (1.0 - eye_t)

        self.register_buffer("P_cond", p_cond_t)
        self.register_buffer("P_ppmi", p_ppmi_t)
        self.register_buffer("P_cond2", p_cond2_t)
        self.register_buffer(
            "b_prior", torch.tensor(prior_log_odds, dtype=torch.float32)
        )

        # 11-term multi-channel polynomial head with informed positive prior initialization
        self.num_poly_terms = 11
        self.tracker_skip_weight = nn.Parameter(
            torch.empty(num_trackers, self.num_poly_terms)
        )
        self.tracker_skip_bias = nn.Parameter(torch.zeros(num_trackers))

        # Tracker Representation Tower: projects static metadata into key embeddings -> [num_trackers, 128]
        self.tracker_proj = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Domain Context Tower: 512-dim GELU MLP context encoder -> [B, 128]
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(context_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.75),
        )
        # 4-Head Multi-Aspect Semantic Dot-Product Attention Projections (head_dim=32, total=128)
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)

        # Context-conditioned FiLM per-tracker channel modulation generator
        self.context_to_tracker_film = nn.Linear(
            128, num_trackers * num_tracker_channels * 2
        )

        # Direct Context Logit Head
        self.direct_context_head = nn.Linear(128, num_trackers)

        # Dynamic Latent Syndicate Factorization Bottleneck (64-dim) with non-linear MLP interaction
        self.syn_down = nn.Linear(num_trackers, self.latent_dim)
        self.syn_ctx = nn.Linear(128, self.latent_dim)
        self.syn_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )
        self.syn_up = nn.Linear(self.latent_dim, num_trackers)

        # Learned Per-Tracker Relational Parameter Matrix across P_cond, PPMI, corp_adj, P_cond^2
        self.W_rel = nn.Parameter(torch.full((num_trackers, 4), 0.1, dtype=torch.float32))

        # Dynamic context-conditioned syndicate gating
        self.syndicate_gate = nn.Linear(128, num_trackers)

        self._init_weights()
        with torch.no_grad():
            self.direct_context_head.bias.copy_(self.b_prior)
            nn.init.zeros_(self.syndicate_gate.bias)
            nn.init.zeros_(self.context_to_tracker_film.weight)
            nn.init.zeros_(self.context_to_tracker_film.bias)
            nn.init.zeros_(self.syn_up.weight)
            nn.init.zeros_(self.syn_up.bias)
            # Informed positive prior initialization across 11 polynomial terms
            init_constants = torch.tensor(
                [0.5, 0.3, 0.2, 0.2, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1],
                dtype=torch.float32,
            )
            self.tracker_skip_weight.copy_(
                init_constants.unsqueeze(0).expand(num_trackers, self.num_poly_terms)
            )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        split_idx = self.num_trackers * self.num_tracker_channels

        # Split tracker-aligned channels vs domain context
        x_trackers_flat = x[:, :split_idx]  # [B, 5 * 355]
        x_context = x[:, split_idx:]  # [B, context_dim]

        # Reshape to [B, num_trackers, num_tracker_channels]
        x_trackers = x_trackers_flat.view(
            batch_size, self.num_tracker_channels, self.num_trackers
        ).permute(0, 2, 1)

        # Domain Context Tower -> [B, 128]
        h_ctx = self.context_encoder(x_context)

        # Compute binary indicator on raw direct links BEFORE FiLM modulation
        c_bin = (x_trackers[:, :, 0] > 0.0).float()

        # Dynamic context-conditioned FiLM per-tracker channel gates
        film = self.context_to_tracker_film(h_ctx).view(
            batch_size, self.num_trackers, self.num_tracker_channels, 2
        )
        film_scale = 1.0 + torch.tanh(film[..., 0])
        film_shift = film[..., 1]
        gated_trackers = film_scale * x_trackers + film_shift  # [B, num_trackers, 5]

        # Expand 5 input channels into 11 polynomial & cross-relational terms
        c0 = gated_trackers[:, :, 0]  # direct links
        c1 = gated_trackers[:, :, 1]  # 1-hop collaborative out-diffusion
        c2 = gated_trackers[:, :, 2]  # 2-hop community out-diffusion
        c3 = gated_trackers[:, :, 3]  # collaborative in-diffusion
        c4 = gated_trackers[:, :, 4]  # Bayesian smoothed TLD regional prior

        poly_terms = torch.stack(
            [
                c0,
                c1,
                c2,
                c3,
                c4,
                c_bin,
                c0 * c1,
                c0 * c2,
                c1 * c2,
                c1 * c3,
                (c1 + c2) * c4,
            ],
            dim=-1,
        )  # [B, num_trackers, 11]

        # Vectorized 11-term polynomial per-tracker channel response
        z_skip = (poly_terms * self.tracker_skip_weight).sum(dim=-1) + self.tracker_skip_bias  # [B, num_trackers]

        # Static tracker metadata representations -> [num_trackers, 128]
        static_embs = torch.cat(
            [
                self.tracker_id_emb(self.tracker_indices),
                self.comp_emb(self.comp_indices),
                self.cat_emb(self.cat_indices),
                self.ctry_emb(self.ctry_indices),
            ],
            dim=-1,
        )
        tracker_key_emb = self.tracker_proj(static_embs)  # [num_trackers, 128]

        # 4-Head Multi-Aspect Semantic Dot-Product Attention
        q_heads = self.q_proj(h_ctx).view(batch_size, 4, 32)
        k_heads = self.k_proj(tracker_key_emb).view(self.num_trackers, 4, 32)
        z_semantic = torch.einsum("bhd,thd->bt", q_heads, k_heads) / math.sqrt(32)  # [B, num_trackers]

        # Direct Context Logit Head
        z_ctx = self.direct_context_head(h_ctx)  # [B, num_trackers]

        # Base logits combining semantic interaction, direct context head, and channel response
        z_base = z_ctx + z_semantic + z_skip  # [B, num_trackers]

        # Dynamic Latent Syndicate Factorization Bottleneck (64-dim)
        p_base = torch.sigmoid(z_base)  # [B, num_trackers]
        u_syn = self.syn_down(p_base)   # [B, 64]
        u_ctx = self.syn_ctx(h_ctx)     # [B, 64]
        h_syn = u_syn + self.syn_mlp(torch.cat([u_syn, u_ctx], dim=-1))  # [B, 64]
        diff_syn = self.syn_up(h_syn)   # [B, num_trackers]

        # Relational Prior Graph Diffusion across 4 topologies with learned per-tracker weights
        d0 = torch.matmul(p_base, self.P_cond)
        d1 = torch.matmul(p_base, self.P_ppmi)
        d2 = torch.matmul(p_base, self.corp_adj)
        d3 = torch.matmul(p_base, self.P_cond2)
        diff_prior = (
            torch.stack([d0, d1, d2, d3], dim=-1) * self.W_rel
        ).sum(dim=-1)  # [B, num_trackers]

        # Dynamic context-conditioned syndicate gating
        gate = torch.sigmoid(self.syndicate_gate(h_ctx))  # [B, num_trackers]

        # Final logits
        logits = z_base + gate * (diff_syn + diff_prior)  # [B, num_trackers]
        return logits


class SmoothRecall10Loss(nn.Module):
    """Un-truncated Smooth Top-10 Negative Intruder Ranking Loss combined with focal binary cross-entropy.
    Grants un-truncated restorative ranking gradients to all ground-truth positives against negative intruders,
    eliminating intra-positive gradient cancellation while suppressing easy distractor negatives.
    """

    def __init__(
        self,
        tau: float = 0.5,
        beta: float = 1.0,
        bce_weight: float = 0.10,
        pos_weight: float = 5.0,
        gamma_focal: float = 1.5,
        **kwargs,
    ):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.bce_weight = bce_weight
        self.gamma_focal = gamma_focal
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        batch_size = logits.size(0)

        # Locate ground-truth positive tracker indices
        pos_indices = torch.nonzero(targets > 0.5, as_tuple=True)
        pos_b = pos_indices[0]
        pos_j = pos_indices[1]

        if pos_b.numel() > 0:
            pos_logits = logits[pos_b, pos_j]  # [N_pos]
            batch_logits = logits[pos_b]       # [N_pos, num_trackers]
            batch_targets = targets[pos_b]     # [N_pos, num_trackers]

            # Mask out ground-truth positives to strictly isolate negative distractor intruders
            neg_mask = (batch_targets < 0.5).float()  # [N_pos, num_trackers]

            # Peer positive mask excluding positive tracker j itself
            peer_pos_mask = (batch_targets > 0.5).float().clone()
            peer_pos_mask[torch.arange(pos_b.size(0), device=logits.device), pos_j] = 0.0

            # Active Margin-Enforced Top-10 Ranking: soft negative intruders with margin Delta=1.0
            diff = (batch_logits - pos_logits.unsqueeze(1) + 1.0) / self.tau
            neg_intruders = (torch.sigmoid(diff) * neg_mask).sum(dim=1)  # [N_pos]

            # Peer positives ahead of tracker j with detached peer logits
            diff_pos = (batch_logits.detach() - pos_logits.unsqueeze(1)) / self.tau
            pos_ahead_j = (torch.sigmoid(diff_pos) * peer_pos_mask).sum(dim=1)  # [N_pos]

            # Dynamic allowed negative intruder budget based on peer positives ahead
            tau_allowed = torch.clamp(9.0 - pos_ahead_j, min=0.0)

            # Un-truncated ranking penalty: grants restorative gradients to all true positives without artificial cutoff
            rank_penalty = F.softplus((neg_intruders - tau_allowed) / self.beta) * self.beta

            # Normalize penalty per domain
            pos_counts = targets.sum(dim=1).clamp(min=1.0)
            sample_rank_loss = torch.zeros(
                batch_size, device=logits.device, dtype=logits.dtype
            ).scatter_add(0, pos_b, rank_penalty / pos_counts[pos_b])

            rank_loss = sample_rank_loss.mean()
        else:
            rank_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        # Focal Binary Cross-Entropy Loss to suppress easy distractor negatives
        p = torch.sigmoid(logits)
        pos_w = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        pos_loss = -pos_w * ((1.0 - p) ** self.gamma_focal) * targets * torch.log(p.clamp(min=1e-7))
        neg_loss = -(p ** self.gamma_focal) * (1.0 - targets) * torch.log((1.0 - p).clamp(min=1e-7))
        focal_bce = (pos_loss + neg_loss).mean()

        return rank_loss + self.bce_weight * focal_bce


class ModelEMA:
    """Maintains Exponential Moving Average (EMA) of model parameters for validation evaluation
    and test checkpoint selection.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> dict:
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])


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
        direct_tracker_links = np.log1p(direct_tracker_csr.toarray()).astype(
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
    print("Computing collaborative neighborhood tracker diffusion with inverse hub discounting...")
    if rel_src_list:
        all_rel_src = np.concatenate(rel_src_list)
        all_rel_dst = np.concatenate(rel_dst_list)
        del rel_src_list, rel_dst_list
        gc.collect()

        non_self_mask = all_rel_src != all_rel_dst
        all_rel_src = all_rel_src[non_self_mask]
        all_rel_dst = all_rel_dst[non_self_mask]

        adj_bin = csr_matrix(
            (np.ones(len(all_rel_src), dtype=np.float32), (all_rel_src, all_rel_dst)),
            shape=(num_rel, num_rel),
        )
        adj_bin.sum_duplicates()
        adj_bin.data = np.clip(adj_bin.data, 0.0, 1.0)

        # Inverse in-degree hub discounting for out-diffusion: edge (u, v) weighted by 1 / sqrt(log(2 + in_deg(v)))
        w_ij = (1.0 / np.sqrt(np.log(2.0 + global_in_degree[adj_bin.indices]))).astype(np.float32)
        adj_matrix = csr_matrix(
            (w_ij, adj_bin.indices, adj_bin.indptr), shape=(num_rel, num_rel)
        )

        # Inverse out-degree hub discounting for reverse diffusion: edge (v, u) weighted by 1 / sqrt(log(2 + out_deg(u)))
        adj_bin_rev = adj_bin.T.tocsr()
        del adj_bin
        w_rev_ji = (1.0 / np.sqrt(np.log(2.0 + global_out_degree[adj_bin_rev.indices]))).astype(np.float32)
        adj_matrix_rev = csr_matrix(
            (w_rev_ji, adj_bin_rev.indices, adj_bin_rev.indptr), shape=(num_rel, num_rel)
        )
        del adj_bin_rev
    else:
        adj_matrix = csr_matrix((num_rel, num_rel), dtype=np.float32)
        adj_matrix_rev = csr_matrix((num_rel, num_rel), dtype=np.float32)

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

    in_tracker_sum = adj_matrix_rev.dot(y_train_full)
    in_train_counts = adj_matrix_rev.dot(is_train_domain_mask)
    in_tracker_dist = (in_tracker_sum / np.maximum(in_train_counts, 1.0)).astype(
        np.float32
    )

    subgraph_out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    subgraph_in_degree = np.array(adj_matrix_rev.sum(axis=1)).flatten()

    # Dedicated 2-hop collaborative community out-diffusion channel
    out_2hop_sum = adj_matrix.dot(out_tracker_dist)
    out_2hop_dist = (
        out_2hop_sum / np.maximum(subgraph_out_degree[:, None], 1.0)
    ).astype(np.float32)
    del out_2hop_sum

    # Fallback for in-diffusion
    in_zero_mask = in_train_counts.flatten() == 0
    if np.any(in_zero_mask):
        in_2hop_sum = adj_matrix_rev.dot(in_tracker_dist)
        in_2hop_fallback = (
            in_2hop_sum / np.maximum(subgraph_in_degree[:, None], 1.0)
        ).astype(np.float32)
        del in_2hop_sum
        in_tracker_dist[in_zero_mask] = in_2hop_fallback[in_zero_mask]
        del in_2hop_fallback

    del adj_matrix, adj_matrix_rev, y_train_full, out_tracker_sum, in_tracker_sum
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
        ngram_range=(3, 5),
        max_features=384,
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
    num_tracker_channels = 5
    tracker_feature_dim = num_trackers * num_tracker_channels
    context_feature_dim = tfidf_train.shape[1] + tabular_train.shape[1]
    num_features = tracker_feature_dim + context_feature_dim

    X_train = np.empty((len(train_indices_in_rel), num_features), dtype=np.float32)
    X_val = np.empty((len(val_indices_in_rel), num_features), dtype=np.float32)
    X_test = np.empty((len(test_indices_in_rel), num_features), dtype=np.float32)

    col = 0
    # Channel 0: direct_tracker_links
    X_train[:, col : col + num_trackers] = direct_tracker_links[train_indices_in_rel]
    X_val[:, col : col + num_trackers] = direct_tracker_links[val_indices_in_rel]
    X_test[:, col : col + num_trackers] = direct_tracker_links[test_indices_in_rel]
    col += num_trackers
    del direct_tracker_links
    gc.collect()

    # Channel 1: out_tracker_dist (1-hop collaborative diffusion)
    X_train[:, col : col + num_trackers] = out_tracker_dist[train_indices_in_rel]
    X_val[:, col : col + num_trackers] = out_tracker_dist[val_indices_in_rel]
    X_test[:, col : col + num_trackers] = out_tracker_dist[test_indices_in_rel]
    col += num_trackers
    del out_tracker_dist
    gc.collect()

    # Channel 2: out_2hop_dist (2-hop community out-diffusion)
    X_train[:, col : col + num_trackers] = out_2hop_dist[train_indices_in_rel]
    X_val[:, col : col + num_trackers] = out_2hop_dist[val_indices_in_rel]
    X_test[:, col : col + num_trackers] = out_2hop_dist[test_indices_in_rel]
    col += num_trackers
    del out_2hop_dist
    gc.collect()

    # Channel 3: in_tracker_dist (collaborative in-diffusion)
    X_train[:, col : col + num_trackers] = in_tracker_dist[train_indices_in_rel]
    X_val[:, col : col + num_trackers] = in_tracker_dist[val_indices_in_rel]
    X_test[:, col : col + num_trackers] = in_tracker_dist[test_indices_in_rel]
    col += num_trackers
    del in_tracker_dist
    gc.collect()

    # Channel 4: tld_tracker_prior (Bayesian smoothed TLD regional prior)
    X_train[:, col : col + num_trackers] = tld_tracker_prior[train_indices_in_rel]
    X_val[:, col : col + num_trackers] = tld_tracker_prior[val_indices_in_rel]
    X_test[:, col : col + num_trackers] = tld_tracker_prior[test_indices_in_rel]
    col += num_trackers
    del tld_tracker_prior
    gc.collect()

    # Context block: Subword char TF-IDF (384 features)
    n_tfidf = tfidf_train.shape[1]
    X_train[:, col : col + n_tfidf] = tfidf_train
    X_val[:, col : col + n_tfidf] = tfidf_val
    X_test[:, col : col + n_tfidf] = tfidf_test
    col += n_tfidf
    del tfidf_train, tfidf_val, tfidf_test
    gc.collect()

    # Context block: Dense Tabular (32 features)
    n_tab = tabular_train.shape[1]
    X_train[:, col : col + n_tab] = tabular_train
    X_val[:, col : col + n_tab] = tabular_val
    X_test[:, col : col + n_tab] = tabular_test
    col += n_tab
    del tabular_train, tabular_val, tabular_test
    gc.collect()

    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    context_dim = num_features - (num_trackers * num_tracker_channels)

    print(
        f"Feature dimensions -> Train: {X_train.shape}, Val: {X_val.shape}, Test:"
        f" {X_test.shape}"
    )
    print(f"Context feature dimension: {context_dim}")

    # ---------------------------------------------------------
    # 13b. Compute Empirical Co-occurrence Topology & Prior Log-Odds
    # ---------------------------------------------------------
    print("Computing empirical conditional co-occurrence and PPMI priors from y_train...")
    C_cooccur = np.zeros((num_trackers, num_trackers), dtype=np.float64)
    chunk_size = 500000
    for i_chunk in range(0, len(y_train), chunk_size):
        sub_y = y_train[i_chunk : i_chunk + chunk_size].astype(np.float32)
        C_cooccur += sub_y.T @ sub_y

    N_train = float(len(y_train))
    diag_C = np.diag(C_cooccur).copy()
    eps = 1e-7

    # Empirical Conditional Co-occurrence P(T_j | T_i) = C_ij / N_i
    C_offdiag = C_cooccur.copy()
    np.fill_diagonal(C_offdiag, 0.0)
    P_cond = (C_offdiag / (diag_C[:, None] + eps)).astype(np.float32)

    # Row-normalized Positive Pointwise Mutual Information (PPMI)
    outer_N = np.outer(diag_C, diag_C)
    valid_mask = (C_offdiag > 0) & (outer_N > 0)
    ppmi = np.zeros((num_trackers, num_trackers), dtype=np.float32)
    pmi_val = np.log((C_offdiag[valid_mask] * N_train) / (outer_N[valid_mask] + eps))
    ppmi[valid_mask] = np.maximum(0.0, pmi_val)
    np.fill_diagonal(ppmi, 0.0)
    ppmi_row_sums = ppmi.sum(axis=1, keepdims=True)
    P_ppmi = np.where(ppmi_row_sums > 0, ppmi / np.maximum(ppmi_row_sums, eps), 0.0).astype(np.float32)

    p_marginal = diag_C / N_train
    eps_prior = 1e-5
    prior_log_odds = np.log((p_marginal + eps_prior) / (1.0 - p_marginal + eps_prior)).astype(np.float32)
    del C_cooccur

    # ---------------------------------------------------------
    # 14. Setup PyTorch DataLoaders & Architecture
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on compute device: {device}")

    # Chunk-buffered fast batch loader to optimize CPU cache locality
    class FastBatchLoader:
        def __init__(self, X, y=None, batch_size=4096, shuffle=True, drop_last=False):
            self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
            self.y = (
                torch.from_numpy(y)
                if (isinstance(y, np.ndarray) and y is not None)
                else y
            )
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.n_samples = len(self.X)

        def __iter__(self):
            if self.shuffle:
                chunk_size = 65536
                num_chunks = (self.n_samples + chunk_size - 1) // chunk_size
                chunk_order = torch.randperm(num_chunks)
                indices = torch.empty(self.n_samples, dtype=torch.long)
                curr = 0
                for c in chunk_order:
                    start = c.item() * chunk_size
                    end = min(start + chunk_size, self.n_samples)
                    chunk_len = end - start
                    perm = torch.randperm(chunk_len)
                    indices[curr : curr + chunk_len] = torch.arange(start, end)[perm]
                    curr += chunk_len
            else:
                indices = torch.arange(self.n_samples)

            for i in range(0, self.n_samples, self.batch_size):
                if self.drop_last and (i + self.batch_size > self.n_samples):
                    break
                batch_idx = indices[i : i + self.batch_size]
                bx = self.X[batch_idx]
                if self.y is not None:
                    by = self.y[batch_idx]
                    yield bx, by
                else:
                    yield bx, None

        def __len__(self):
            if self.drop_last:
                return self.n_samples // self.batch_size
            return (self.n_samples + self.batch_size - 1) // self.batch_size

    batch_size = 16384

    train_loader = FastBatchLoader(
        X_train, y_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = FastBatchLoader(
        X_val, y_val.astype(np.float32), batch_size=batch_size * 2, shuffle=False, drop_last=False
    )
    test_loader = FastBatchLoader(
        X_test, None, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )

    model = TrackerSyndicateNet(
        num_trackers=num_trackers,
        num_tracker_channels=num_tracker_channels,
        context_dim=context_dim,
        tracker_emb_dim=24,
        context_hidden_dim=512,
        syndicate_rank=64,
        dropout_rate=0.10,
        cond_cooccur_matrix=P_cond,
        ppmi_matrix=P_ppmi,
        prior_log_odds=prior_log_odds,
    ).to(device)

    criterion = SmoothRecall10Loss(
        tau=0.5,
        beta=1.0,
        bce_weight=0.10,
        pos_weight=5.0,
        gamma_focal=1.5,
    ).to(device)

    epochs = 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=0.999)

    # ---------------------------------------------------------
    # 15. Training & Hold-out Metric Evaluation with EMA
    # ---------------------------------------------------------
    best_val_recall = -1.0
    best_epoch = "None"
    checkpoint_path = "./working/best_tracker_model.pt"
    MAX_TOTAL_RUNTIME = 50 * 60  # 50-min global timeout guard (hard limit 60m)

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
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_train_loss += loss.item()
            num_train_batches += 1

            if batch_idx % 250 == 0:
                if (time.time() - global_start_time) > MAX_TOTAL_RUNTIME:
                    print(
                        f"Time guard triggered at batch {batch_idx}/{len(train_loader)}. Finishing current epoch safely."
                    )
                    time_to_stop = True
                    break

        scheduler.step()
        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Validation Evaluation for standard online model
        model.eval()
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(batch_x)
                val_logits_list.append(logits.float().cpu())
                val_targets_list.append(batch_y.float())

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
            best_epoch = f"Online-E{epoch+1}"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_recall": val_recall,
                },
                checkpoint_path,
            )

        # Validation Evaluation for EMA model
        backup = ema.apply_shadow(model)
        model.eval()
        ema_val_logits_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(batch_x)
                ema_val_logits_list.append(logits.float().cpu())

        ema_val_logits = torch.cat(ema_val_logits_list, dim=0)
        ema_val_recall = compute_recall_at_10(ema_val_logits, all_val_targets)
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} [EMA decay=0.999] - "
            f"Val Recall@10: {ema_val_recall:.6f}"
        )

        if ema_val_recall > best_val_recall:
            best_val_recall = ema_val_recall
            best_epoch = f"EMA-E{epoch+1}"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_recall": ema_val_recall,
                },
                checkpoint_path,
            )

        ema.restore(model, backup)

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
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
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
            val_logits_list.append(logits.float().cpu())
            val_targets_list.append(batch_y.float())

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
            test_logits_list.append(logits.float().cpu())

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

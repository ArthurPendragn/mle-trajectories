import copy
import json
import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Set deterministic random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def find_file(filename: str) -> str:
    """Locate file across standard directory structures."""
    search_dirs = ["./input", "./input/input", ".", "./data"]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"File {filename} not found in candidate paths: {search_dirs}"
    )


# =========================================================================
# 1. MODEL ARCHITECTURE: TABULAR GATED CROSS-ATTENTION NETWORK (T-GCAN)
# =========================================================================


class PeriodicFeatureTokenizer(nn.Module):
    """Periodic non-linear embedding tokenizer for continuous tabular features."""

    def __init__(self, num_features: int, d_token: int, n_frequencies: int = 16):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.empty(num_features, n_frequencies))
        self.weight = nn.Parameter(torch.empty(num_features, 2 * n_frequencies, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.frequencies, mean=0.0, std=1.0)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(2 * self.n_frequencies)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        x_expanded = x_num.unsqueeze(-1)  # (b, n, 1)
        angles = 2.0 * math.pi * x_expanded * self.frequencies.unsqueeze(0)  # (b, n, k)
        bases = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (b, n, 2k)
        tokens = torch.einsum("bnf,nfd->bnd", bases, self.weight) + self.bias.unsqueeze(0)
        return tokens


# Alias for backward compatibility
NumericalFeatureTokenizer = PeriodicFeatureTokenizer


class SwiGLU(nn.Module):
    """Gated Feed-Forward Unit with Swish activation for expressive non-linear gating."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TabularTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with Multi-Head Attention and SwiGLU FFN."""

    def __init__(
        self,
        d_token: int,
        n_heads: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_token)
        hidden_dim = int(d_token * ffn_mult)
        self.ffn = SwiGLU(d_token, hidden_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TabularGCAN(nn.Module):
    """Tabular Gated Cross-Attention Network (T-GCAN)

    Tokenizes discrete and continuous variables into a shared latent space,
    captures non-linear dependencies via multi-head attention, and generates
    binary classification logits via a context-pooled prediction head.
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        d_token: int = 48,
        n_blocks: int = 3,
        n_heads: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.15,
        token_dropout: float = 0.10,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.token_dropout = token_dropout

        # Categorical feature tokenizers
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(max(cardinality, 1) + 2, d_token)
                for cardinality in cat_cardinalities
            ]
        )

        # Continuous feature tokenizer
        if num_continuous > 0:
            self.num_tokenizer = PeriodicFeatureTokenizer(num_continuous, d_token)
        else:
            self.num_tokenizer = None

        # Class [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TabularTransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Attentive multi-token pooling layers
        self.pool_attn = nn.Linear(d_token, 1)
        self.pool_input_attn = nn.Linear(d_token, 1)

        # Wide-residual shortcut classification head
        self.head_norm_context = nn.LayerNorm(d_token)
        self.head_norm_input = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token * 2, d_token),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        token_list = [self.cls_token.expand(b, -1, -1)]

        # Collect categorical tokens
        if x_cat is not None and len(self.cat_embeddings) > 0:
            cat_tokens = [
                emb(x_cat[:, i]).unsqueeze(1)
                for i, emb in enumerate(self.cat_embeddings)
            ]
            token_list.extend(cat_tokens)

        # Collect numerical tokens
        if x_num is not None and self.num_tokenizer is not None:
            num_tokens = self.num_tokenizer(x_num)
            token_list.append(num_tokens)

        # Concatenate along token sequence dimension: (b, 1 + n_cat + n_num, d_token)
        init_tokens = torch.cat(token_list, dim=1)
        tokens = init_tokens

        # Feature Token Dropout: stochastic masking over feature tokens prior to transformer blocks
        if self.training and self.token_dropout > 0.0:
            cls_t = tokens[:, :1, :]
            feat_t = tokens[:, 1:, :]
            n_feats = feat_t.shape[1]
            mask = torch.bernoulli(
                torch.full((b, n_feats, 1), 1.0 - self.token_dropout, device=tokens.device)
            )
            feat_t = feat_t * mask / (1.0 - self.token_dropout)
            tokens = torch.cat([cls_t, feat_t], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        # Context representation from transformer stack
        attn_scores = self.pool_attn(tokens)  # (b, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_context = torch.sum(tokens * attn_weights, dim=1)

        # Wide-residual shortcut representation directly from initial tokens
        input_scores = self.pool_input_attn(init_tokens)
        input_weights = F.softmax(input_scores, dim=1)
        pooled_input = torch.sum(init_tokens * input_weights, dim=1)

        # Wide-residual fusion directly into classification head
        fused = torch.cat(
            [self.head_norm_context(pooled_context), self.head_norm_input(pooled_input)],
            dim=-1,
        )
        logits = self.head(fused).squeeze(-1)
        return logits


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """Label-Smoothed Binary Cross-Entropy with Logits."""

    def __init__(self, eps: float = 0.05, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1.0 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight
        )


def get_loss_criterion(eps: float = 0.05, pos_weight: Optional[float] = None) -> nn.Module:
    """Returns label-smoothed binary cross-entropy loss with logits."""
    weight_tensor = (
        torch.tensor([pos_weight], dtype=torch.float32)
        if pos_weight is not None
        else None
    )
    return LabelSmoothedBCEWithLogitsLoss(eps=eps, pos_weight=weight_tensor)


def get_optimizer(
    model: nn.Module, lr: float = 3.5e-4, weight_decay: float = 1e-4
) -> AdamW:
    """Configures AdamW excluding LayerNorm and bias parameters from weight decay."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=lr)


class TabularSpaceshipDataset(Dataset):
    """PyTorch tabular dataset wrapper for continuous and categorical tensors."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray = None):
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]


# =========================================================================
# 2. DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================


def process_features():
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train = len(train_df)
    y_train = train_df["Transported"].astype(int)
    test_df["Transported"] = np.nan

    # Combine for relational metadata and topology extraction
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Group decomposition
    df["GroupId"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["GroupPos"] = df["PassengerId"].apply(lambda x: int(x.split("_")[1]))
    group_sizes = df["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(group_sizes)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    # Name and surname analysis
    df["Name"] = df["Name"].fillna("Unknown Unknown")
    df["FirstName"] = df["Name"].apply(
        lambda x: x.split()[0] if len(x.split()) > 0 else "Unknown"
    )
    df["LastName"] = df["Name"].apply(
        lambda x: x.split()[-1] if len(x.split()) > 1 else "Unknown"
    )
    family_counts = df[df["LastName"] != "Unknown"]["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_counts).fillna(1).astype(int)
    df["IsSoloFamily"] = (df["FamilySize"] == 1).astype(int)

    # Cabin topology decomposition
    cabin_split = (
        df["Cabin"]
        .fillna("Missing/Missing/Missing")
        .apply(
            lambda x: (
                x.split("/") if "/" in str(x) else ["Missing", "Missing", "Missing"]
            )
        )
    )
    df["CabinDeck"] = cabin_split.apply(lambda x: x[0] if len(x) == 3 else "Missing")
    df["CabinNumRaw"] = cabin_split.apply(lambda x: x[1] if len(x) == 3 else "Missing")
    df["CabinSide"] = cabin_split.apply(lambda x: x[2] if len(x) == 3 else "Missing")

    df["CabinNum"] = pd.to_numeric(df["CabinNumRaw"], errors="coerce")
    deck_num_median = df.groupby("CabinDeck")["CabinNum"].transform("median").fillna(-1)
    df["CabinNum"] = df["CabinNum"].fillna(deck_num_median)
    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)
    df["CabinCluster50"] = (df["CabinNum"] // 50).fillna(-1).astype(int)
    df["CabinDeck_Side_Cluster"] = (
        df["CabinDeck"]
        + "_"
        + df["CabinSide"]
        + "_"
        + df["CabinCluster50"].astype(str)
    )

    # Domain-informed physical deduction
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    # CryoSleep spending invariant: CryoSleep == True cannot spend on amenities
    for amenity in amenities:
        df.loc[df["CryoSleep"] == True, amenity] = df.loc[
            df["CryoSleep"] == True, amenity
        ].fillna(0.0)

    # If spending > 0, CryoSleep cannot be True -> must be False
    total_raw_spend = df[amenities].sum(axis=1)
    df.loc[(df["CryoSleep"].isna()) & (total_raw_spend > 0), "CryoSleep"] = False

    # Minors under 13 cannot spend on amenities
    for amenity in amenities:
        df.loc[df["Age"] < 13, amenity] = df.loc[df["Age"] < 13, amenity].fillna(0.0)

    # VIP physical deck & planet invariants
    df.loc[(df["HomePlanet"] == "Earth") & (df["VIP"].isna()), "VIP"] = False
    df.loc[(df["CabinDeck"].isin(["G", "T"])) & (df["VIP"].isna()), "VIP"] = False

    # Intra-Group Relational Imputation for HomePlanet
    group_planet = df[df["HomePlanet"].notna()].groupby("GroupId")["HomePlanet"].first()
    df.loc[df["HomePlanet"].isna(), "HomePlanet"] = df.loc[
        df["HomePlanet"].isna(), "GroupId"
    ].map(group_planet)

    # Deck constraint imputation for HomePlanet
    df.loc[
        (df["HomePlanet"].isna()) & (df["CabinDeck"].isin(["A", "B", "C", "T"])),
        "HomePlanet",
    ] = "Europa"
    df.loc[(df["HomePlanet"].isna()) & (df["CabinDeck"] == "G"), "HomePlanet"] = "Earth"

    # Fill remaining amenities with 0.0
    for amenity in amenities:
        df[amenity] = df[amenity].fillna(0.0)

    # Retain NaN CryoSleep after physical deductions: Dedicated missingness indicator
    df["CryoSleep_is_na"] = df["CryoSleep"].isna().astype(int)

    # Explicit ternary categorical representation (0 = Awake/False, 1 = Asleep/True, 2 = Unknown/Missing)
    cryo_ternary = pd.Series(2, index=df.index, dtype=int)
    cryo_ternary[df["CryoSleep"] == False] = 0
    cryo_ternary[df["CryoSleep"] == True] = 1
    df["CryoSleep"] = cryo_ternary

    df["VIP"] = (df["VIP"] == True).astype(int)
    df["HomePlanet"] = df["HomePlanet"].fillna("Missing")
    df["Destination"] = df["Destination"].fillna("Missing")

    # Age Imputation: Median by Deck and HomePlanet
    age_medians = (
        df.groupby(["HomePlanet", "CabinDeck"])["Age"]
        .transform("median")
        .fillna(df["Age"].median())
    )
    df["Age"] = df["Age"].fillna(age_medians)

    # Advanced Expenditure & Ratio Features
    df["TotalSpend"] = df[amenities].sum(axis=1)
    df["HasSpent"] = (df["TotalSpend"] > 0).astype(int)
    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["AmenitiesSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["AmenitiesRatio"] = df["AmenitiesSpend"] / (df["TotalSpend"] + 1.0)

    for amenity in amenities:
        df[f"{amenity}_Log"] = np.log1p(df[amenity])
        df[f"{amenity}_Ratio"] = df[amenity] / (df["TotalSpend"] + 1.0)
    df["TotalSpend_Log"] = np.log1p(df["TotalSpend"])
    df["NumAmenitiesUsed"] = (df[amenities] > 0).sum(axis=1)

    # Group Expenditure Dynamics
    group_mean_spend = df.groupby("GroupId")["TotalSpend"].transform("mean")
    df["GroupMeanSpend"] = group_mean_spend
    df["SpendDiffFromGroup"] = df["TotalSpend"] - group_mean_spend

    # Demographics
    df["IsInfant"] = (df["Age"] <= 4).astype(int)
    df["IsChild"] = ((df["Age"] > 4) & (df["Age"] <= 12)).astype(int)
    df["IsTeen"] = ((df["Age"] > 12) & (df["Age"] <= 17)).astype(int)
    df["IsSenior"] = (df["Age"] >= 60).astype(int)

    # Spatial & Route Cross-Features
    df["Route"] = df["HomePlanet"] + "_to_" + df["Destination"]
    df["Deck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]

    discrete_cols = [
        "CryoSleep",
        "CryoSleep_is_na",
        "VIP",
        "HasSpent",
        "IsSolo",
        "IsSoloFamily",
        "IsInfant",
        "IsChild",
        "IsTeen",
        "IsSenior",
        "CabinRegion",
        "CabinCluster50",
    ]

    categorical_cols = [
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "Route",
        "Deck_Side",
        "CabinDeck_Side_Cluster",
    ] + discrete_cols

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes.astype(int)
        df[col] = df[col].clip(lower=0)

    drop_cols = ["Name", "FirstName", "LastName", "Cabin", "CabinNumRaw", "GroupId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Separate back into train and test
    train_processed = df.iloc[:n_train].copy()
    test_processed = df.iloc[n_train:].copy()

    train_processed["Transported"] = y_train.values
    test_processed = test_processed.drop(columns=["Transported"])

    # 5-fold Stratified Group K-Fold partitioned strictly by GroupId
    train_group_ids = train_df["PassengerId"].apply(lambda x: x.split("_")[0]).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_processed["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(
            train_processed, train_processed["Transported"], groups=train_group_ids
        )
    ):
        train_processed.loc[val_idx, "fold"] = fold_idx

    feature_cols = [
        c
        for c in train_processed.columns
        if c not in ["PassengerId", "Transported", "fold"]
    ]
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    os.makedirs("./working", exist_ok=True)
    metadata = {
        "id_col": "PassengerId",
        "target_col": "Transported",
        "fold_col": "fold",
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return train_processed, test_processed, categorical_cols, numerical_cols


# =========================================================================
# 3. TRAINING, EVALUATION & INFERENCE PIPELINE
# =========================================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df, cat_cols, num_cols = process_features()

    cat_cardinalities = [
        int(max(train_df[c].max(), test_df[c].max())) + 1 for c in cat_cols
    ]

    n_splits = 5
    oof_probs = np.zeros(len(train_df), dtype=np.float32)
    test_probs_accum = np.zeros(len(test_df), dtype=np.float32)
    y_all = train_df["Transported"].values.astype(int)

    batch_size = 128
    epochs = 28
    patience = 7

    for fold in range(n_splits):
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        # Leak-free scaling fitted strictly on training fold continuous features
        scaler = StandardScaler()
        x_train_num = scaler.fit_transform(train_df.loc[train_mask, num_cols].values)
        x_val_num = scaler.transform(train_df.loc[val_mask, num_cols].values)
        x_test_num = scaler.transform(test_df[num_cols].values)

        x_train_cat = train_df.loc[train_mask, cat_cols].values.astype(np.int64)
        x_val_cat = train_df.loc[val_mask, cat_cols].values.astype(np.int64)
        x_test_cat = test_df[cat_cols].values.astype(np.int64)

        y_train = train_df.loc[train_mask, "Transported"].values.astype(np.float32)
        y_val = train_df.loc[val_mask, "Transported"].values.astype(np.float32)

        train_dataset = TabularSpaceshipDataset(x_train_num, x_train_cat, y_train)
        val_dataset = TabularSpaceshipDataset(x_val_num, x_val_cat, y_val)
        test_dataset = TabularSpaceshipDataset(x_test_num, x_test_cat)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

        model = TabularGCAN(
            num_continuous=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            d_token=48,
            n_blocks=3,
            n_heads=4,
            dropout=0.15,
            token_dropout=0.10,
        ).to(device)

        criterion = get_loss_criterion(eps=0.05)
        optimizer = get_optimizer(model, lr=3.5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_acc = 0.0
        best_model_weights = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            train_samples = 0

            for b_x_num, b_x_cat, b_y in train_loader:
                b_x_num = b_x_num.to(device)
                b_x_cat = b_x_cat.to(device)
                b_y = b_y.to(device)

                optimizer.zero_grad()
                logits = model(b_x_num, b_x_cat)
                loss = criterion(logits, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                train_loss_total += loss.item() * len(b_y)
                train_samples += len(b_y)

            scheduler.step()
            train_loss = train_loss_total / train_samples

            # Validation forward pass
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for b_x_num, b_x_cat, b_y in val_loader:
                    b_x_num = b_x_num.to(device)
                    b_x_cat = b_x_cat.to(device)
                    b_y = b_y.to(device)

                    logits = model(b_x_num, b_x_cat)
                    loss = criterion(logits, b_y)
                    val_loss_total += loss.item() * len(b_y)

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    val_correct += (preds == b_y).sum().item()
                    val_samples += len(b_y)

            val_loss = val_loss_total / val_samples
            val_acc = val_correct / val_samples

            print(
                f"Fold {fold + 1} | Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Load best weights for fold predictions
        model.load_state_dict(best_model_weights)
        model.eval()

        fold_val_probs = []
        with torch.no_grad():
            for b_x_num, b_x_cat, _ in val_loader:
                b_x_num = b_x_num.to(device)
                b_x_cat = b_x_cat.to(device)
                logits = model(b_x_num, b_x_cat)
                fold_val_probs.extend(torch.sigmoid(logits).cpu().numpy())
        oof_probs[val_mask] = np.array(fold_val_probs)

        fold_test_probs = []
        with torch.no_grad():
            for b_x_num, b_x_cat in test_loader:
                b_x_num = b_x_num.to(device)
                b_x_cat = b_x_cat.to(device)
                logits = model(b_x_num, b_x_cat)
                fold_test_probs.extend(torch.sigmoid(logits).cpu().numpy())
        test_probs_accum += np.array(fold_test_probs) / n_splits

    # Standard default threshold evaluation for binary cross-entropy logits
    best_threshold = 0.5
    best_oof_acc = accuracy_score(y_all, (oof_probs >= best_threshold).astype(int))

    # Generate competition submission file
    os.makedirs("./submission", exist_ok=True)
    test_binary_predictions = test_probs_accum >= best_threshold
    test_bool_predictions = [bool(p) for p in test_binary_predictions]

    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_bool_predictions}
    )

    submission_out_path = "./submission/submission.csv"
    submission_df.to_csv(submission_out_path, index=False)

    # Final validation score output required by metric parser
    score = best_oof_acc
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
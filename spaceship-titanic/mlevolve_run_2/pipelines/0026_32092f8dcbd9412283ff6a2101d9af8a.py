import copy
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# 0. Reproducibility Configuration
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Path Resolution & Data Ingestion
# -----------------------------------------------------------------------------
input_dir = "./input"
train_path = os.path.join(input_dir, "train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(input_dir, "input", "train.csv")

test_path = os.path.join(input_dir, "test.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(input_dir, "input", "test.csv")

os.makedirs("./working", exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

test_ids = df_test["PassengerId"].copy()
y = df_train["Transported"].astype(int).values

df_train_raw = df_train.drop(columns=["Transported"]).copy()
df_test_raw = df_test.copy()

n_train = len(df_train_raw)
combined = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)

# -----------------------------------------------------------------------------
# 2. Domain-Specific Feature Engineering & Physical Constraint Resolution
# -----------------------------------------------------------------------------
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Parse PassengerId into Group and Position
combined["Group"] = combined["PassengerId"].apply(lambda x: x.split("_")[0])
combined["GroupNum"] = combined["PassengerId"].apply(lambda x: int(x.split("_")[1]))

# Manifest-level group size
group_sizes = combined["Group"].value_counts().to_dict()
combined["GroupSize"] = combined["Group"].map(group_sizes)
combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
combined["GroupPosRatio"] = (combined["GroupNum"] / combined["GroupSize"]).astype(float)

# Extract Surname for family group identification
combined["Surname"] = combined["Name"].apply(
    lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
)
surname_counts = combined["Surname"].value_counts().to_dict()
combined["FamilySize"] = combined["Surname"].map(surname_counts).fillna(1)

# Deduce HomePlanet from group members if missing
group_homeplanet = combined.groupby("Group")["HomePlanet"].first().dropna().to_dict()
combined["HomePlanet"] = combined["HomePlanet"].fillna(
    combined["Group"].map(group_homeplanet)
)

# Extract Cabin components: Deck / Num / Side
combined["CabinDeck"] = combined["Cabin"].apply(
    lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan
)
combined["CabinNum"] = combined["Cabin"].apply(
    lambda x: float(str(x).split("/")[1]) if pd.notna(x) else np.nan
)
combined["CabinSide"] = combined["Cabin"].apply(
    lambda x: str(x).split("/")[2] if pd.notna(x) else np.nan
)

# Deduce CabinDeck and CabinSide from group members if missing
group_deck = combined.groupby("Group")["CabinDeck"].first().dropna().to_dict()
combined["CabinDeck"] = combined["CabinDeck"].fillna(combined["Group"].map(group_deck))

group_side = combined.groupby("Group")["CabinSide"].first().dropna().to_dict()
combined["CabinSide"] = combined["CabinSide"].fillna(combined["Group"].map(group_side))

# Deduce HomePlanet from CabinDeck domain invariants (Decks A, B, C, T to Europa; Deck G to Earth)
combined.loc[
    combined["HomePlanet"].isna() & combined["CabinDeck"].isin(["A", "B", "C", "T"]),
    "HomePlanet",
] = "Europa"
combined.loc[
    combined["HomePlanet"].isna() & (combined["CabinDeck"] == "G"), "HomePlanet"
] = "Earth"

# Deterministic physical constraint resolution:
# 1) If in CryoSleep, luxury spending is strictly 0
for col in spend_cols:
    combined.loc[combined["CryoSleep"] == True, col] = combined.loc[
        combined["CryoSleep"] == True, col
    ].fillna(0.0)

# 2) If total spend > 0, CryoSleep must be False
known_spend_sum = combined[spend_cols].fillna(0).sum(axis=1)
combined.loc[(combined["CryoSleep"].isna()) & (known_spend_sum > 0), "CryoSleep"] = (
    False
)

# 3) Children (Age < 13) cannot spend on amenities
child_mask = combined["Age"] < 13
for col in spend_cols:
    combined.loc[child_mask, col] = combined.loc[child_mask, col].fillna(0.0)

# Impute remaining missing values for spend columns with 0.0 (baseline non-spenders)
for col in spend_cols:
    combined[col] = combined[col].fillna(0.0)

# Deduce CryoSleep from group consensus first before fallback
group_cryo = (
    combined.groupby("Group")["CryoSleep"]
    .agg(lambda s: s.mode().iloc[0] if not s.dropna().empty else np.nan)
    .dropna()
    .to_dict()
)
combined["CryoSleep"] = combined["CryoSleep"].fillna(combined["Group"].map(group_cryo))
combined["CryoSleep"] = (
    combined["CryoSleep"].infer_objects(copy=False).fillna(0).astype(bool).astype(int)
)
combined["VIP"] = (
    combined["VIP"].infer_objects(copy=False).fillna(False).astype(bool).astype(int)
)

# Expenditure aggregation and non-linear ratios
combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
combined["ZeroSpend"] = (combined["TotalSpend"] == 0).astype(int)
combined["LuxurySpend"] = combined["RoomService"] + combined["Spa"] + combined["VRDeck"]
combined["ZeroLuxurySpend"] = (combined["LuxurySpend"] == 0).astype(int)
combined["BasicSpend"] = combined["FoodCourt"] + combined["ShoppingMall"]
combined["AmenityCount"] = (combined[spend_cols] > 0).sum(axis=1)
combined["Luxury_to_Total"] = combined["LuxurySpend"] / (combined["TotalSpend"] + 1.0)

for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
    combined[f"Log_{col}"] = np.log1p(combined[col])

for col in spend_cols:
    combined[f"Pct_{col}"] = combined[col] / (combined["TotalSpend"] + 1.0)

# Cabin spatial regions and localized blocks
combined["CabinRegion"] = (combined["CabinNum"] // 300).fillna(-1)
combined["CabinBlock"] = (combined["CabinNum"] // 50).fillna(-1)
combined["CabinParity"] = (combined["CabinNum"] % 2).fillna(-1)

# Deck-relative normalized cabin positions
deck_max = combined.groupby("CabinDeck")["CabinNum"].transform("max")
combined["CabinDeck_RelPos"] = (combined["CabinNum"] / deck_max).fillna(0.0)

# Route interaction & CabinDeck_CabinSide interaction
combined["HomePlanet"] = combined["HomePlanet"].fillna("Missing")
combined["Destination"] = combined["Destination"].fillna("Missing")
combined["Route"] = combined["HomePlanet"] + "_" + combined["Destination"]
combined["CabinDeck_CabinSide"] = (
    combined["CabinDeck"].fillna("Missing").astype(str)
    + "_"
    + combined["CabinSide"].fillna("Missing").astype(str)
)
combined["CryoSleep_CabinSide"] = (
    combined["CryoSleep"].astype(str)
    + "_"
    + combined["CabinSide"].fillna("Missing").astype(str)
)
combined["HomePlanet_CabinDeck"] = (
    combined["HomePlanet"].fillna("Missing").astype(str)
    + "_"
    + combined["CabinDeck"].fillna("Missing").astype(str)
)

# Uncontaminated leave-one-out peer statistics
group_cryo_sum = combined.groupby("Group")["CryoSleep"].transform("sum")
group_spend_sum = combined.groupby("Group")["TotalSpend"].transform("sum")
has_peers = combined["GroupSize"] > 1

combined["Peer_CryoSleep_Rate"] = 0.0
combined.loc[has_peers, "Peer_CryoSleep_Rate"] = (
    (group_cryo_sum[has_peers] - combined.loc[has_peers, "CryoSleep"])
    / (combined.loc[has_peers, "GroupSize"] - 1)
)

combined["Peer_TotalSpend"] = 0.0
combined.loc[has_peers, "Peer_TotalSpend"] = (
    (group_spend_sum[has_peers] - combined.loc[has_peers, "TotalSpend"])
    / (combined.loc[has_peers, "GroupSize"] - 1)
)
combined["Log_Peer_TotalSpend"] = np.log1p(combined["Peer_TotalSpend"])

# Age lifecycle buckets
combined["AgeGroup"] = pd.cut(
    combined["Age"].fillna(-1),
    bins=[-2, 0, 12, 18, 30, 50, 70, 100],
    labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
).astype(str)

# -----------------------------------------------------------------------------
# 3. Leak-Free Cross-Validation, Imputation & Categorical Encoding
# -----------------------------------------------------------------------------
categorical_cols = [
    "HomePlanet",
    "Destination",
    "CabinDeck",
    "CabinSide",
    "CabinDeck_CabinSide",
    "CryoSleep_CabinSide",
    "HomePlanet_CabinDeck",
    "Route",
    "AgeGroup",
    "CryoSleep",
    "VIP",
    "ZeroSpend",
    "ZeroLuxurySpend",
    "IsAlone",
    "CabinBlock",
    "CabinParity",
]

for cat_col in categorical_cols:
    combined[cat_col] = combined[cat_col].fillna("Missing").astype(str)

drop_cols = ["PassengerId", "Cabin", "Name", "Group", "Surname"]
combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])

# Split back to train and test
train_features = combined.iloc[:n_train].copy().reset_index(drop=True)
test_features = combined.iloc[n_train:].copy().reset_index(drop=True)

# Stratified partition: Fold 0 reserved as holdout validation split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_features["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
    train_features.loc[val_idx, "fold"] = fold

numerical_cols = [
    c for c in train_features.columns if c not in categorical_cols and c not in ["fold"]
]

# Impute numerical features using medians fitted strictly on the train split (fold != 0)
base_train_mask = train_features["fold"] != 0
num_medians = train_features.loc[base_train_mask, numerical_cols].median().to_dict()

train_features[numerical_cols] = train_features[numerical_cols].fillna(num_medians)
test_features[numerical_cols] = test_features[numerical_cols].fillna(num_medians)

# Ordinal encode categorical features fitted strictly on the train split
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(train_features.loc[base_train_mask, categorical_cols])

train_features[categorical_cols] = encoder.transform(train_features[categorical_cols])
test_features[categorical_cols] = encoder.transform(test_features[categorical_cols])

# Add target column back
train_features["Transported"] = y

# Form train and holdout validation splits
val_mask = train_features["fold"] == 0
train_split = train_features[~val_mask].drop(columns=["fold"]).reset_index(drop=True)
val_split = train_features[val_mask].drop(columns=["fold"]).reset_index(drop=True)

test_features["PassengerId"] = test_ids.values

# Fit RobustScaler strictly on training partition for continuous numerical features
scaler = RobustScaler()
train_split[numerical_cols] = scaler.fit_transform(train_split[numerical_cols])
val_split[numerical_cols] = scaler.transform(val_split[numerical_cols])
test_features[numerical_cols] = scaler.transform(test_features[numerical_cols])

# Compute categorical cardinalities from the training partition
cat_cardinalities = [int(max(train_split[c].max(), 0)) + 1 for c in categorical_cols]


# -----------------------------------------------------------------------------
# 4. Architecture Definition: Deep & Cross Network v2 (DCN-v2)
# -----------------------------------------------------------------------------
class CrossLayer(nn.Module):
    """Low-rank explicit feature crossing layer: x_{l+1} = LayerNorm(x_0 * (W_up W_down x_l + b) + x_l)."""

    def __init__(self, in_features: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, in_features, bias=True)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x_0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        crossed = x_0 * self.up(self.down(x_l)) + x_l
        return self.norm(crossed)


class CrossNetwork(nn.Module):
    """Stack of recursive cross layers capturing degree-D interactions with rank decomposition."""

    def __init__(self, in_features: int, num_layers: int = 3, rank: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossLayer(in_features, rank=rank) for _ in range(num_layers)]
        )

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        x_l = x_0
        for layer in self.layers:
            x_l = layer(x_0, x_l)
        return x_l


class DeepBlock(nn.Module):
    """Feed-forward block with LayerNorm, Swish (SiLU), Dropout, and residual connection."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class PeriodicEmbedding(nn.Module):
    """Learnable Periodic Fourier sinusoidal embedding for continuous features."""

    def __init__(self, n_frequencies: int = 16, d_token: int = 16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.1)
        self.proj = nn.Linear(2 * n_frequencies, d_token)
        self.norm = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * np.pi * x * self.frequencies
        sin_cos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.norm(self.proj(sin_cos))


class DCNv2(nn.Module):
    """Deep & Cross Network v2: Parallel Cross & Deep streams with categorical embeddings."""

    def __init__(
        self,
        cat_cardinalities: list,
        num_continuous: int,
        embedding_dim_cap: int = 16,
        cross_layers: int = 2,
        deep_hidden_dims: list = [192, 96, 48],
        dropout_rate: float = 0.2,
        d_token: int = 16,
        token_mask_prob: float = 0.06,
    ):
        super().__init__()
        self.d_token = d_token
        self.token_mask_prob = token_mask_prob

        # Standardize categorical entity embeddings to uniform token dimension d_token
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 2, self.d_token) for card in cat_cardinalities]
        )

        # Continuous feature tokenizer: Periodic Fourier sinusoidal embeddings
        self.num_continuous = num_continuous
        self.cont_tokenizers = nn.ModuleList(
            [
                PeriodicEmbedding(n_frequencies=16, d_token=self.d_token)
                for _ in range(num_continuous)
            ]
        )

        # Pre-LayerNorm Multi-Head Self-Attention interaction layer with gated residual projection
        self.norm = nn.LayerNorm(self.d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_token, num_heads=2, dropout=0.1, batch_first=True
        )
        self.interaction_gate = nn.Linear(self.d_token, self.d_token)
        self.interaction_proj = nn.Linear(self.d_token, self.d_token)
        self.interaction_norm = nn.LayerNorm(self.d_token)

        # Total input dimension into Cross and Deep branches after flattening token sequence
        num_features = len(cat_cardinalities) + num_continuous
        input_dim = num_features * self.d_token

        # Gated Feature Selection Block (GLU with LayerNorm & residual connection)
        self.gate_linear = nn.Linear(input_dim, input_dim)
        self.gate_sigmoid = nn.Linear(input_dim, input_dim)
        self.gate_norm = nn.LayerNorm(input_dim)
        self.feature_dropout = nn.Dropout(0.1)

        # Parallel Stream 1: Explicit Low-Rank Cross Network
        self.cross_net = CrossNetwork(input_dim, num_layers=cross_layers, rank=32)

        # Parallel Stream 2: Deep MLP Network with residual connections
        deep_layers = []
        curr_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.append(DeepBlock(curr_dim, hidden_dim, dropout_rate))
            curr_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)

        # Direct linear projection for continuous residual skip connection
        self.cont_skip_dim = 16 if num_continuous > 0 else 0
        if num_continuous > 0:
            self.cont_skip = nn.Linear(num_continuous, self.cont_skip_dim)
        else:
            self.cont_skip = None

        # Combined Prediction Head
        head_in_dim = input_dim + deep_hidden_dims[-1] + self.cont_skip_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor = None,
        token_mask_prob: float = None,
        mixup_lambda: float = None,
        mixup_perm: torch.Tensor = None,
    ) -> torch.Tensor:
        tokens_list = []
        if len(self.embeddings) > 0 and x_cat is not None:
            for i, emb_layer in enumerate(self.embeddings):
                col_idx = torch.clamp(x_cat[:, i].long() + 1, min=0)
                tokens_list.append(emb_layer(col_idx))

        if x_cont is not None and self.num_continuous > 0:
            for j, cont_tok in enumerate(self.cont_tokenizers):
                tokens_list.append(cont_tok(x_cont[:, j : j + 1].float()))

        # Combined token sequence of shape (B, N_features, d_token)
        token_seq = torch.stack(tokens_list, dim=1)

        # Pre-LayerNorm Multi-Head Self-Attention interaction with gated residual projection
        norm_tokens = self.norm(token_seq)
        attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
        gate = torch.sigmoid(self.interaction_gate(attn_out))
        token_seq = self.interaction_norm(token_seq + gate * self.interaction_proj(attn_out))

        # Tabular Feature Token Masking across feature tokens during training
        eff_mask_prob = (
            token_mask_prob if token_mask_prob is not None else self.token_mask_prob
        )
        if self.training and eff_mask_prob > 0.0:
            mask = (
                torch.rand(
                    token_seq.size(0), token_seq.size(1), 1, device=token_seq.device
                )
                >= eff_mask_prob
            ).float()
            token_seq = token_seq * mask

        # Flatten token sequence into 1D representation vector
        x_0 = token_seq.flatten(start_dim=1)

        # Gated feature selection with residual connection and feature dropout
        x_gated = self.gate_norm(
            x_0 + self.gate_linear(x_0) * torch.sigmoid(self.gate_sigmoid(x_0))
        )
        x_gated = self.feature_dropout(x_gated)

        # Optional latent Manifold Mixup (preserved for interface compatibility)
        if mixup_lambda is not None and mixup_perm is not None:
            x_gated = mixup_lambda * x_gated + (1.0 - mixup_lambda) * x_gated[mixup_perm]

        x_cross = self.cross_net(x_gated)
        x_deep = self.deep_net(x_gated)

        fused_components = [x_cross, x_deep]
        if self.cont_skip is not None and x_cont is not None:
            c_skip = self.cont_skip(x_cont.float())
            if mixup_lambda is not None and mixup_perm is not None:
                c_skip = (
                    mixup_lambda * c_skip + (1.0 - mixup_lambda) * c_skip[mixup_perm]
                )
            fused_components.append(c_skip)

        x_fused = torch.cat(fused_components, dim=1)
        logits = self.head(x_fused)
        return logits


# -----------------------------------------------------------------------------
# 5. Model Initialization & Tensor DataLoaders
# -----------------------------------------------------------------------------
model = DCNv2(
    cat_cardinalities=cat_cardinalities,
    num_continuous=len(numerical_cols),
    embedding_dim_cap=16,
    cross_layers=2,
    deep_hidden_dims=[192, 96, 48],
    dropout_rate=0.2,
    d_token=16,
    token_mask_prob=0.06,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
)

warmup_epochs = 3
num_epochs = 50
min_lr_ratio = 1e-5 / 5e-4


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Exponential Moving Average (EMA) model tracking
ema_decay = 0.995
ema_model = copy.deepcopy(model).to(device)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad = False

X_train_cat = torch.tensor(train_split[categorical_cols].values, dtype=torch.long)
X_train_cont = torch.tensor(train_split[numerical_cols].values, dtype=torch.float32)
y_train = torch.tensor(
    train_split["Transported"].values, dtype=torch.float32
).unsqueeze(1)

X_val_cat = torch.tensor(val_split[categorical_cols].values, dtype=torch.long)
X_val_cont = torch.tensor(val_split[numerical_cols].values, dtype=torch.float32)
y_val = torch.tensor(val_split["Transported"].values, dtype=torch.float32).unsqueeze(1)

X_test_cat = torch.tensor(test_features[categorical_cols].values, dtype=torch.long)
X_test_cont = torch.tensor(test_features[numerical_cols].values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_cat, X_train_cont, y_train)
val_dataset = TensorDataset(X_val_cat, X_val_cont, y_val)
test_dataset = TensorDataset(X_test_cat, X_test_cont)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# -----------------------------------------------------------------------------
# 6. Training Loop with Tabular Swap-Noise, EMA Weight Tracking & Early Stopping
# -----------------------------------------------------------------------------
patience = 12
best_val_loss = float("inf")
best_weights = copy.deepcopy(ema_model.state_dict())
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for b_cat, b_cont, b_y in train_loader:
        b_cat = b_cat.to(device)
        b_cont = b_cont.to(device)
        b_y = b_y.to(device)

        batch_size = b_y.size(0)
        mixup_lambda = float(np.random.beta(0.2, 0.2))
        mixup_perm = torch.randperm(batch_size, device=device)

        optimizer.zero_grad()
        logits = model(
            b_cat,
            b_cont,
            token_mask_prob=0.06,
            mixup_lambda=mixup_lambda,
            mixup_perm=mixup_perm,
        )
        y_smooth = b_y * (1.0 - 0.05) + 0.5 * 0.05
        y_mix = mixup_lambda * y_smooth + (1.0 - mixup_lambda) * y_smooth[mixup_perm]
        loss = criterion(logits, y_mix)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update EMA shadow model weights
        with torch.no_grad():
            for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)
            for b_ema, b_model in zip(ema_model.buffers(), model.buffers()):
                b_ema.copy_(b_model)

        total_train_loss += loss.item() * len(b_y)

    train_loss = total_train_loss / len(train_dataset)

    # Validation evaluation on EMA weights
    ema_model.eval()
    total_val_loss = 0.0
    val_preds_list = []
    val_targets_list = []
    with torch.no_grad():
        for b_cat, b_cont, b_y in val_loader:
            b_cat = b_cat.to(device)
            b_cont = b_cont.to(device)
            b_y = b_y.to(device)

            logits = ema_model(b_cat, b_cont)
            loss = criterion(logits, b_y)
            total_val_loss += loss.item() * len(b_y)

            probs = torch.sigmoid(logits)
            val_preds_list.append((probs >= 0.5).int().cpu().numpy())
            val_targets_list.append(b_y.int().cpu().numpy())

    val_loss = total_val_loss / len(val_dataset)
    val_preds = np.vstack(val_preds_list).ravel()
    val_targets = np.vstack(val_targets_list).ravel()
    val_acc = accuracy_score(val_targets, val_preds)

    scheduler.step()

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(ema_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# -----------------------------------------------------------------------------
# 7. Model Restoration & Validation Decision Threshold Calibration
# -----------------------------------------------------------------------------
model.load_state_dict(best_weights)
model.eval()

final_val_probs_list = []
final_val_targets_list = []
with torch.no_grad():
    for b_cat, b_cont, b_y in val_loader:
        b_cat = b_cat.to(device)
        b_cont = b_cont.to(device)
        logits = model(b_cat, b_cont)
        probs = torch.sigmoid(logits)
        final_val_probs_list.append(probs.cpu().numpy())
        final_val_targets_list.append(b_y.int().cpu().numpy())

final_val_probs = np.vstack(final_val_probs_list).ravel()
final_val_targets = np.vstack(final_val_targets_list).ravel()

# Validation evaluation at standard 0.5 decision threshold
final_val_preds = (final_val_probs >= 0.5).astype(int)
final_val_acc = accuracy_score(final_val_targets, final_val_preds)

# -----------------------------------------------------------------------------
# 8. Out-of-Sample Test Inference & Submission Generation
# -----------------------------------------------------------------------------
test_preds_list = []
with torch.no_grad():
    for b_cat, b_cont in test_loader:
        b_cat = b_cat.to(device)
        b_cont = b_cont.to(device)
        logits = model(b_cat, b_cont)
        probs = torch.sigmoid(logits)
        test_preds_list.append((probs >= 0.5).bool().cpu().numpy())

test_preds = np.vstack(test_preds_list).ravel()

os.makedirs("./submission", exist_ok=True)
sub = pd.DataFrame(
    {"PassengerId": test_features["PassengerId"].values, "Transported": test_preds}
)
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_acc}")
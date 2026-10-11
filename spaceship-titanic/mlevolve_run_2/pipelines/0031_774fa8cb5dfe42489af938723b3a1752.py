import copy
import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Reproducibility & Setup
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_data_file(filename):
    for candidate in [
        os.path.join(".", "input", filename),
        os.path.join(".", "input", "input", filename),
        os.path.join("input", filename),
        os.path.join("input", "input", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {filename}")


train_path = find_data_file("train.csv")
test_path = find_data_file("test.csv")
sub_sample_path = find_data_file("sample_submission.csv")

raw_train = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_sample_path)

# ---------------------------------------------------------
# 2. Structural Parsing Across Static Voyage Manifest
# ---------------------------------------------------------
all_manifest = pd.concat([raw_train, raw_test], ignore_index=True)

all_manifest["GroupId"] = all_manifest["PassengerId"].apply(lambda x: x.split("_")[0])
all_manifest["GroupNum"] = all_manifest["PassengerId"].apply(
    lambda x: int(x.split("_")[1])
)
all_manifest["LastName"] = all_manifest["Name"].apply(
    lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
)

group_sizes = all_manifest["GroupId"].value_counts().to_dict()
family_sizes = all_manifest["LastName"].dropna().value_counts().to_dict()


def parse_structural_features(df):
    d = df.copy()
    d["GroupId"] = d["PassengerId"].apply(lambda x: x.split("_")[0])
    d["GroupNum"] = d["PassengerId"].apply(lambda x: int(x.split("_")[1]))
    d["GroupSize"] = d["GroupId"].map(group_sizes).fillna(1).astype(int)
    d["IsAlone"] = (d["GroupSize"] == 1).astype(int)
    d["GroupPosRatio"] = d["GroupNum"] / d["GroupSize"]

    d["LastName"] = d["Name"].apply(
        lambda x: str(x).split()[-1] if pd.notna(x) else np.nan
    )
    d["FamilySize"] = d["LastName"].map(family_sizes).fillna(1).astype(int)
    d["IsFamilyAlone"] = (d["FamilySize"] == 1).astype(int)

    # Cabin decomposition
    d["Cabin_Deck"] = d["Cabin"].apply(
        lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan
    )
    d["Cabin_Num"] = pd.to_numeric(
        d["Cabin"].apply(
            lambda x: (
                str(x).split("/")[1]
                if pd.notna(x) and len(str(x).split("/")) > 1
                else np.nan
            )
        ),
        errors="coerce",
    )
    d["Cabin_Side"] = d["Cabin"].apply(
        lambda x: (
            str(x).split("/")[2]
            if pd.notna(x) and len(str(x).split("/")) > 2
            else np.nan
        )
    )
    d["Cabin_Region"] = d["Cabin_Num"].apply(
        lambda x: int(x // 300) if pd.notna(x) else np.nan
    )
    return d


parsed_train = parse_structural_features(raw_train)
parsed_test = parse_structural_features(raw_test)

# ---------------------------------------------------------
# 3. Strict Stratified Train / Validation Split
# ---------------------------------------------------------
y_all = parsed_train["Transported"].astype(int)
train_df, val_df = train_test_split(
    parsed_train, test_size=0.20, random_state=SEED, stratify=y_all
)
test_df = parsed_test.copy()

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ---------------------------------------------------------
# 4. Leak-Free Relational Knowledge Mappings (From Train Only)
# ---------------------------------------------------------
train_known_planet = train_df.dropna(subset=["HomePlanet"])
group_planet_map = train_known_planet.groupby("GroupId")["HomePlanet"].first().to_dict()
family_planet_map = (
    train_known_planet.dropna(subset=["LastName"])
    .groupby("LastName")["HomePlanet"]
    .first()
    .to_dict()
)

train_known_deck = train_df.dropna(subset=["Cabin_Deck"])
group_deck_map = train_known_deck.groupby("GroupId")["Cabin_Deck"].first().to_dict()

train_known_side = train_df.dropna(subset=["Cabin_Side"])
group_side_map = train_known_side.groupby("GroupId")["Cabin_Side"].first().to_dict()

train_median_cabin_num = train_df["Cabin_Num"].median()
train_median_age = train_df["Age"].median()

deck_spend_map = (
    train_df.groupby("Cabin_Deck")[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ]
    .sum()
    .sum(axis=1)
    .to_dict()
)

# ---------------------------------------------------------
# 5. Domain Invariant Resolution & Feature Engineering
# ---------------------------------------------------------
amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def apply_domain_resolution_and_features(df):
    d = df.copy()

    # Physical Invariant 1: Suspended passengers have zero expenditure
    cryo_mask = d["CryoSleep"] == True
    for col in amenities:
        d.loc[cryo_mask & d[col].isna(), col] = 0.0

    # Physical Invariant 2: Active spenders cannot be in cryo-sleep
    partial_spend = d[amenities].fillna(0).sum(axis=1)
    has_spend = partial_spend > 0
    d.loc[has_spend & d["CryoSleep"].isna(), "CryoSleep"] = False

    # Physical Invariant 3: Children under 13 have zero luxury amenity expenditures
    child_mask = d["Age"] < 13
    for col in amenities:
        d.loc[child_mask & d[col].isna(), col] = 0.0

    # Physical Invariant 4: Earth passengers are never VIP
    earth_mask = d["HomePlanet"] == "Earth"
    d.loc[earth_mask & d["VIP"].isna(), "VIP"] = False

    # Physical Invariant 5: Minors are never VIP
    minor_mask = d["Age"] < 18
    d.loc[minor_mask & d["VIP"].isna(), "VIP"] = False

    # Physical Invariant 6: Specific cabin decks map to origin planets
    d.loc[
        d["Cabin_Deck"].isin(["A", "B", "C", "T"]) & d["HomePlanet"].isna(),
        "HomePlanet",
    ] = "Europa"
    d.loc[(d["Cabin_Deck"] == "G") & d["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Relational Knowledge Deduction via Group and Family mappings
    local_group_planet = (
        d.dropna(subset=["HomePlanet"])
        .groupby("GroupId")["HomePlanet"]
        .first()
        .to_dict()
    )
    d["HomePlanet"] = d["HomePlanet"].fillna(d["GroupId"].map(local_group_planet))
    d["HomePlanet"] = d["HomePlanet"].fillna(d["GroupId"].map(group_planet_map))
    d["HomePlanet"] = d["HomePlanet"].fillna(d["LastName"].map(family_planet_map))

    local_group_side = (
        d.dropna(subset=["Cabin_Side"])
        .groupby("GroupId")["Cabin_Side"]
        .first()
        .to_dict()
    )
    d["Cabin_Side"] = d["Cabin_Side"].fillna(d["GroupId"].map(local_group_side))
    d["Cabin_Side"] = d["Cabin_Side"].fillna(d["GroupId"].map(group_side_map))

    local_group_deck = (
        d.dropna(subset=["Cabin_Deck"])
        .groupby("GroupId")["Cabin_Deck"]
        .first()
        .to_dict()
    )
    d["Cabin_Deck"] = d["Cabin_Deck"].fillna(d["GroupId"].map(local_group_deck))
    d["Cabin_Deck"] = d["Cabin_Deck"].fillna(d["GroupId"].map(group_deck_map))

    for col in amenities:
        d[col] = d[col].fillna(0.0)

    d["TotalSpend"] = d[amenities].sum(axis=1)
    d["HasSpent"] = (d["TotalSpend"] > 0).astype(int)
    d["LuxurySpend"] = d["Spa"] + d["VRDeck"] + d["RoomService"]
    d["SustenanceSpend"] = d["FoodCourt"] + d["ShoppingMall"]
    d["LuxuryRatio"] = d["LuxurySpend"] / (d["TotalSpend"] + 1.0)
    d["SustenanceRatio"] = d["SustenanceSpend"] / (d["TotalSpend"] + 1.0)

    for col in amenities:
        d[f"{col}_Ratio"] = d[col] / (d["TotalSpend"] + 1.0)
        d[f"Log_{col}"] = np.log1p(d[col])

    d["Log_TotalSpend"] = np.log1p(d["TotalSpend"])
    d["Log_LuxurySpend"] = np.log1p(d["LuxurySpend"])
    d["Log_SustenanceSpend"] = np.log1p(d["SustenanceSpend"])
    d["NumAmenitiesUsed"] = sum((d[col] > 0).astype(int) for col in amenities)
    d["SpendMax"] = d[amenities].max(axis=1)
    d["SpendStd"] = d[amenities].std(axis=1)

    d["Age_isna"] = d["Age"].isna().astype(int)
    d["Age"] = d["Age"].fillna(train_median_age)
    d["IsChild"] = (d["Age"] < 13).astype(int)
    d["IsTeen"] = ((d["Age"] >= 13) & (d["Age"] < 18)).astype(int)
    d["IsAdult"] = ((d["Age"] >= 18) & (d["Age"] < 60)).astype(int)
    d["IsSenior"] = (d["Age"] >= 60).astype(int)

    age_bins = [-1, 12, 17, 25, 35, 50, 65, 120]
    d["AgeGroup"] = pd.cut(
        d["Age"], bins=age_bins, labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(int)

    d["Cabin_Num_isna"] = d["Cabin_Num"].isna().astype(int)
    d["Cabin_Num"] = d["Cabin_Num"].fillna(train_median_cabin_num)
    d["Cabin_Region"] = d["Cabin_Num"] // 300

    d["Deck_Side"] = d["Cabin_Deck"].astype(str) + "_" + d["Cabin_Side"].astype(str)
    d["Planet_Dest"] = d["HomePlanet"].astype(str) + "_" + d["Destination"].astype(str)
    d["Deck_Planet"] = d["Cabin_Deck"].astype(str) + "_" + d["HomePlanet"].astype(str)
    d["Cryo_Deck"] = d["CryoSleep"].astype(str) + "_" + d["Cabin_Deck"].astype(str)

    d["DeckSpendBenchmark"] = (
        d["Cabin_Deck"].map(deck_spend_map).fillna(0.0).astype(float)
    )

    categorical_cols_to_fill = [
        "HomePlanet",
        "Destination",
        "Cabin_Deck",
        "Cabin_Side",
    ]
    for col in categorical_cols_to_fill:
        d[col] = d[col].fillna("Unknown")

    d["CryoSleep"] = d["CryoSleep"].fillna(False).astype(int)
    d["VIP"] = d["VIP"].fillna(False).astype(int)

    return d


X_train_df = apply_domain_resolution_and_features(train_df)
X_val_df = apply_domain_resolution_and_features(val_df)
X_test_df = apply_domain_resolution_and_features(test_df)

y_train = train_df["Transported"].astype(int).values
y_val = val_df["Transported"].astype(int).values

cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "Deck_Side",
    "Planet_Dest",
    "Deck_Planet",
    "Cryo_Deck",
]

drop_cols = ["PassengerId", "Cabin", "Name", "GroupId", "LastName"]
if "Transported" in X_train_df.columns:
    drop_cols.append("Transported")

feature_cols = [c for c in X_train_df.columns if c not in drop_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

cat_cardinalities = []
for col in cat_cols:
    unique_cats = sorted(
        list(
            set(
                X_train_df[col].astype(str).tolist()
                + X_val_df[col].astype(str).tolist()
                + X_test_df[col].astype(str).tolist()
            )
        )
    )
    cat_to_idx = {cat_val: idx for idx, cat_val in enumerate(unique_cats)}
    X_train_df[col] = X_train_df[col].astype(str).map(cat_to_idx).fillna(0).astype(np.int64)
    X_val_df[col] = X_val_df[col].astype(str).map(cat_to_idx).fillna(0).astype(np.int64)
    X_test_df[col] = X_test_df[col].astype(str).map(cat_to_idx).fillna(0).astype(np.int64)
    cat_cardinalities.append(len(unique_cats))

for col in num_cols:
    X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce").fillna(0.0).astype(np.float32)
    X_val_df[col] = pd.to_numeric(X_val_df[col], errors="coerce").fillna(0.0).astype(np.float32)
    X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce").fillna(0.0).astype(np.float32)


# ---------------------------------------------------------
# 6. Bi-Interaction Sparse Mixture-of-Experts Architecture
# ---------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc_gate = nn.Linear(in_features, hidden_features)
        self.fc_val = nn.Linear(in_features, hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.fc_gate(x)) * self.fc_val(x)


class SwiGLUResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.swiglu(h)
        h = self.dropout(h)
        h = self.proj(h)
        h = self.dropout(h)
        return x + h


class SqueezeAndExcitation(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(8, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class BiInteractionMoENet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list,
        num_continuous: int,
        d_token: int = 48,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_mult: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.num_continuous = num_continuous
        self.d_token = d_token
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        self.n_freq = 16
        fourier_dim = 1 + 2 * self.n_freq
        self.fourier_freqs = nn.Parameter(
            torch.randn(num_continuous, self.n_freq) * 0.5
        )
        self.fourier_phases = nn.Parameter(
            torch.zeros(num_continuous, self.n_freq)
        )
        self.cont_proj_w = nn.Parameter(
            torch.empty(num_continuous, fourier_dim, d_token).normal_(std=0.02)
        )
        self.cont_proj_b = nn.Parameter(torch.zeros(num_continuous, d_token))
        self.token_norm = nn.LayerNorm(d_token)
        self.attn_norm = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_token, num_heads=4, dropout=dropout, batch_first=True
        )

        self.rep_dim = d_token * 2
        self.router = nn.Linear(self.rep_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                SwiGLUResidualBlock(
                    dim=self.rep_dim,
                    hidden_dim=self.rep_dim * expert_hidden_mult,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        self.head_dim = self.rep_dim * 2
        self.wide_highway = nn.Linear(num_continuous, self.head_dim)
        self.pre_head_norm = nn.LayerNorm(self.head_dim)
        self.se_block = SqueezeAndExcitation(self.head_dim, reduction=4)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout + 0.05),
            nn.Linear(self.head_dim, self.rep_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.rep_dim, 1),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        x_cont_unsq = x_cont.unsqueeze(-1)
        angles = 2.0 * np.pi * x_cont_unsq * self.fourier_freqs.unsqueeze(0) + self.fourier_phases.unsqueeze(0)
        fourier_feats = torch.cat(
            [x_cont_unsq, torch.sin(angles), torch.cos(angles)], dim=-1
        )
        cont_tokens = torch.einsum(
            "bcf,cfd->bcd", fourier_feats, self.cont_proj_w
        ) + self.cont_proj_b.unsqueeze(0)

        cat_tokens = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
        ]
        cat_tokens = torch.stack(cat_tokens, dim=1)

        tokens = torch.cat([cont_tokens, cat_tokens], dim=1)
        tokens = self.token_norm(tokens)

        if self.training:
            token_mask = (
                torch.rand(tokens.size(0), tokens.size(1), 1, device=tokens.device) >= 0.10
            ).float() / 0.90
            tokens = tokens * token_mask

        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.mha(attn_in, attn_in, attn_in)
        tokens = tokens + attn_out

        sum_tokens = tokens.sum(dim=1)
        sum_sq_tokens = (tokens**2).sum(dim=1)
        bilinear_pool = 0.5 * ((sum_tokens**2) - sum_sq_tokens)
        mean_tokens = tokens.mean(dim=1)

        h_rep = torch.cat([mean_tokens, bilinear_pool], dim=-1)

        gate_logits = self.router(h_rep)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        expert_stack = torch.stack([expert(h_rep) for expert in self.experts], dim=1)
        sparse_weights = torch.zeros_like(gate_probs).scatter(
            1, topk_indices, topk_weights
        )
        h_moe = torch.einsum("be,bed->bd", sparse_weights, expert_stack)

        router_mask = torch.zeros_like(gate_probs).scatter(1, topk_indices, 1.0).detach()
        mean_gate_prob = gate_probs.mean(dim=0)
        mean_router_mask = router_mask.mean(dim=0)
        aux_loss = 0.01 * self.num_experts * torch.sum(mean_gate_prob * mean_router_mask)

        h_concat = torch.cat([h_moe, bilinear_pool, mean_tokens], dim=-1) + self.wide_highway(x_cont)
        h_norm = self.pre_head_norm(h_concat)
        h_calibrated = self.se_block(h_norm)

        logits = self.classifier(h_calibrated)
        return logits, aux_loss


# ---------------------------------------------------------
# 7. Data Standardization & PyTorch Setup
# ---------------------------------------------------------
scaler = StandardScaler()
X_tr_cont = scaler.fit_transform(X_train_df[num_cols].values).astype(np.float32)
X_va_cont = scaler.transform(X_val_df[num_cols].values).astype(np.float32)
X_te_cont = scaler.transform(X_test_df[num_cols].values).astype(np.float32)

X_tr_cat = X_train_df[cat_cols].values.astype(np.int64)
X_va_cat = X_val_df[cat_cols].values.astype(np.int64)
X_te_cat = X_test_df[cat_cols].values.astype(np.int64)

BATCH_SIZE = 128
train_dataset = TensorDataset(
    torch.tensor(X_tr_cont, dtype=torch.float32),
    torch.tensor(X_tr_cat, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
)
val_dataset = TensorDataset(
    torch.tensor(X_va_cont, dtype=torch.float32),
    torch.tensor(X_va_cat, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
)
test_dataset = TensorDataset(
    torch.tensor(X_te_cont, dtype=torch.float32),
    torch.tensor(X_te_cat, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ---------------------------------------------------------
# 8. Training Loop with Early Stopping & Checkpointing
# ---------------------------------------------------------
model = BiInteractionMoENet(
    cat_cardinalities=cat_cardinalities,
    num_continuous=len(num_cols),
    d_token=48,
    num_experts=4,
    top_k=2,
    expert_hidden_mult=2,
    dropout=0.15,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    eps=1e-8,
)

EPOCHS = 60
PATIENCE = 15
WARMUP_EPOCHS = 5

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS],
)

best_val_acc = -1.0
best_model_weights = copy.deepcopy(model.state_dict())
checkpoint_path = "./working/best_bi_interaction_moe.pt"
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_cont_b, x_cat_b, y_batch in train_loader:
        x_cont_b = x_cont_b.to(DEVICE)
        x_cat_b = x_cat_b.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        y_smooth = y_batch * (1.0 - 0.05) + 0.5 * 0.05

        optimizer.zero_grad()
        logits, aux_loss = model(x_cont_b, x_cat_b)
        loss = criterion(logits, y_smooth) + aux_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_loss / max(1, num_batches)

    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x_cont_val_b, x_cat_val_b, y_val_b in val_loader:
            x_cont_val_b = x_cont_val_b.to(DEVICE)
            x_cat_val_b = x_cat_val_b.to(DEVICE)
            val_logits, _ = model(x_cont_val_b, x_cat_val_b)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds_list.append(val_probs)
            val_targets_list.append(y_val_b.numpy())

    val_preds = np.vstack(val_preds_list).ravel()
    val_targets = np.vstack(val_targets_list).ravel()
    epoch_val_acc = accuracy_score(val_targets, (val_preds >= 0.5).astype(int))

    print(
        f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Best Val Acc: {max(best_val_acc, epoch_val_acc):.4f}"
    )

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, checkpoint_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

# ---------------------------------------------------------
# 9. Final Validation Assessment & Test Inference
# ---------------------------------------------------------
model.load_state_dict(best_model_weights)
model.eval()

final_val_preds = []
final_val_targets = []

with torch.no_grad():
    for x_cont_val_b, x_cat_val_b, y_val_b in val_loader:
        x_cont_val_b = x_cont_val_b.to(DEVICE)
        x_cat_val_b = x_cat_val_b.to(DEVICE)
        val_logits, _ = model(x_cont_val_b, x_cat_val_b)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        final_val_preds.append(val_probs)
        final_val_targets.append(y_val_b.numpy())

final_val_probs = np.vstack(final_val_preds).ravel()
final_val_labels = np.vstack(final_val_targets).ravel()

best_threshold = 0.5
best_val_score = -1.0
threshold_grid = np.linspace(0.40, 0.60, 201)
for th in threshold_grid:
    score = accuracy_score(final_val_labels, (final_val_probs >= th).astype(int))
    if score > best_val_score:
        best_val_score = score
        best_threshold = th

final_val_score = best_val_score

test_preds_list = []
with torch.no_grad():
    for x_cont_test_b, x_cat_test_b in test_loader:
        x_cont_test_b = x_cont_test_b.to(DEVICE)
        x_cat_test_b = x_cat_test_b.to(DEVICE)
        test_logits, _ = model(x_cont_test_b, x_cat_test_b)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds_list.append(test_probs)

test_probs = np.vstack(test_preds_list).ravel()
test_predictions = (test_probs >= best_threshold).astype(bool)

submission_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
)

submission_file = "./submission/submission.csv"
submission_df.to_csv(submission_file, index=False)

assert os.path.exists(submission_file), "Submission file does not exist"
assert len(submission_df) == len(
    raw_test
), f"Expected {len(raw_test)} rows, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "PassengerId",
    "Transported",
], "Incorrect column headers"
assert submission_df["Transported"].dtype == bool, "Transported must be boolean"
assert (
    submission_df["Transported"].isna().sum() == 0
), "NaN values detected in predictions"

print(f"Final Validation Score: {final_val_score:.5f}")

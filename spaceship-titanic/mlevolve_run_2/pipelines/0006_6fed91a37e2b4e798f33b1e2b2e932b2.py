import copy
import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 2. Locate and load raw data robustly
def get_data_path(filename):
    for candidate in [
        f"./input/{filename}",
        f"./input/input/{filename}",
        f"./{filename}",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in ./input or ./input/input")


train_path = get_data_path("train.csv")
test_path = get_data_path("test.csv")

df_train_raw = pd.read_csv(train_path)
df_test_raw = pd.read_csv(test_path)

# 3. Extract manifest-level relational mappings (Group and Family counts)
all_pids = pd.concat(
    [df_train_raw["PassengerId"], df_test_raw["PassengerId"]], ignore_index=True
)
all_groups = all_pids.str.split("_").str[0]
group_counts = all_groups.value_counts().to_dict()

all_names = pd.concat(
    [df_train_raw["Name"].dropna(), df_test_raw["Name"].dropna()], ignore_index=True
)
all_surnames = all_names.str.split().str[-1]
surname_counts = all_surnames.value_counts().to_dict()

# 4. Manifest Relational Imputation & Domain Feature Extraction
df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

# Passenger ID structural features
df_all["Group"] = df_all["PassengerId"].str.split("_").str[0]
df_all["Group_Num"] = df_all["PassengerId"].str.split("_").str[1].astype(int)
df_all["Group_Size"] = df_all["Group"].map(group_counts).fillna(1).astype(int)
df_all["Is_Alone"] = (df_all["Group_Size"] == 1).astype(int)
df_all["Surname"] = df_all["Name"].str.split().str[-1]
df_all["Family_Size"] = df_all["Surname"].map(surname_counts).fillna(1).astype(int)

# Cabin decomposition
cabin_split = df_all["Cabin"].str.split("/", expand=True)
df_all["Cabin_Deck"] = cabin_split[0]
df_all["Cabin_Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
df_all["Cabin_Side"] = cabin_split[2]

# Amenity spending missingness indicator flags
amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in amenity_cols:
    df_all[f"{col}_Missing"] = df_all[col].isna().astype(float)

# Physical consistency: spending > 0 implies passenger cannot be in CryoSleep
has_spending = (df_all[amenity_cols] > 0).any(axis=1)
df_all.loc[has_spending, "CryoSleep"] = False

# Physical consistency: decks A/B/C/T belong to Europa, deck G belongs to Earth
df_all.loc[
    df_all["Cabin_Deck"].isin(["A", "B", "C", "T"]) & df_all["HomePlanet"].isna(),
    "HomePlanet",
] = "Europa"
df_all.loc[(df_all["Cabin_Deck"] == "G") & df_all["HomePlanet"].isna(), "HomePlanet"] = (
    "Earth"
)

# Relational propagation within identical travel groups (near-perfect fidelity)
group_hp = df_all.dropna(subset=["HomePlanet"]).groupby("Group")["HomePlanet"].first()
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(df_all["Group"].map(group_hp))

group_deck = (
    df_all.dropna(subset=["Cabin_Deck"]).groupby("Group")["Cabin_Deck"].first()
)
df_all["Cabin_Deck"] = df_all["Cabin_Deck"].fillna(df_all["Group"].map(group_deck))

group_side = (
    df_all.dropna(subset=["Cabin_Side"]).groupby("Group")["Cabin_Side"].first()
)
df_all["Cabin_Side"] = df_all["Cabin_Side"].fillna(df_all["Group"].map(group_side))

group_cnum = df_all.dropna(subset=["Cabin_Num"]).groupby("Group")["Cabin_Num"].first()
df_all["Cabin_Num"] = df_all["Cabin_Num"].fillna(df_all["Group"].map(group_cnum))

# Relational propagation within identical family surnames
surname_hp = (
    df_all.dropna(subset=["HomePlanet"])
    .groupby("Surname")["HomePlanet"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(df_all["Surname"].map(surname_hp))

surname_deck = (
    df_all.dropna(subset=["Cabin_Deck"])
    .groupby("Surname")["Cabin_Deck"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["Cabin_Deck"] = df_all["Cabin_Deck"].fillna(df_all["Surname"].map(surname_deck))

surname_side = (
    df_all.dropna(subset=["Cabin_Side"])
    .groupby("Surname")["Cabin_Side"]
    .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
)
df_all["Cabin_Side"] = df_all["Cabin_Side"].fillna(df_all["Surname"].map(surname_side))

# Re-apply physical deck-planet deduction after deck imputation
df_all.loc[
    df_all["Cabin_Deck"].isin(["A", "B", "C", "T"]) & df_all["HomePlanet"].isna(),
    "HomePlanet",
] = "Europa"
df_all.loc[(df_all["Cabin_Deck"] == "G") & df_all["HomePlanet"].isna(), "HomePlanet"] = (
    "Earth"
)

# Spatial deck and side interaction
df_all["Cabin_Deck_Side"] = df_all.apply(
    lambda r: (
        f"{r['Cabin_Deck']}_{r['Cabin_Side']}"
        if pd.notna(r["Cabin_Deck"]) and pd.notna(r["Cabin_Side"])
        else np.nan
    ),
    axis=1,
)

# Spatial cabin zone binning (longitudinal regions)
df_all["Cabin_Region"] = (df_all["Cabin_Num"] // 300).apply(
    lambda v: f"Region_{int(v)}" if pd.notna(v) else np.nan
)

# Domain rules: Passengers in CryoSleep or children under 13 spend nothing
for col in amenity_cols:
    df_all.loc[df_all["CryoSleep"] == True, col] = 0.0
    df_all.loc[df_all["Age"] < 13, col] = 0.0

df_all["TotalSpend"] = df_all[amenity_cols].sum(axis=1, min_count=1)
df_all["ZeroSpend"] = (df_all["TotalSpend"] == 0).astype(int)

for col in amenity_cols:
    df_all[f"{col}_Log"] = np.log1p(df_all[col].clip(lower=0))
    df_all[f"{col}_Ratio"] = (df_all[col] / (df_all["TotalSpend"] + 1.0)).fillna(0.0)

df_all["TotalSpend_Log"] = np.log1p(df_all["TotalSpend"].clip(lower=0))

# Age bins
df_all["Age_Group"] = pd.cut(
    df_all["Age"],
    bins=[-np.inf, 12, 18, 30, 50, np.inf],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
).astype(str)
df_all.loc[df_all["Age"].isna(), "Age_Group"] = np.nan

# Separate back into df_train_feat and df_test_feat
n_train = len(df_train_raw)
df_train_feat = df_all.iloc[:n_train].copy().reset_index(drop=True)
df_test_feat = df_all.iloc[n_train:].copy().reset_index(drop=True)

y = df_train_feat["Transported"].astype(int).values
test_ids = df_test_feat["PassengerId"].values

# 5. Strict Train/Validation Split Before Stateful Transformations
train_idx, val_idx = train_test_split(
    np.arange(len(df_train_feat)), test_size=0.2, random_state=42, stratify=y
)

train_split = df_train_feat.iloc[train_idx].copy().reset_index(drop=True)
val_split = df_train_feat.iloc[val_idx].copy().reset_index(drop=True)
test_split = df_test_feat.copy().reset_index(drop=True)

y_train = y[train_idx]
y_val = y[val_idx]

# Define Feature Sets
cat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Cabin_Deck",
    "Cabin_Side",
    "Cabin_Deck_Side",
    "Age_Group",
    "Cabin_Region",
]
num_cols = [
    "Age",
    "Group_Num",
    "Group_Size",
    "Is_Alone",
    "Family_Size",
    "Cabin_Num",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "ZeroSpend",
    "TotalSpend_Log",
    "RoomService_Log",
    "FoodCourt_Log",
    "ShoppingMall_Log",
    "Spa_Log",
    "VRDeck_Log",
    "RoomService_Ratio",
    "FoodCourt_Ratio",
    "ShoppingMall_Ratio",
    "Spa_Ratio",
    "VRDeck_Ratio",
    "RoomService_Missing",
    "FoodCourt_Missing",
    "ShoppingMall_Missing",
    "Spa_Missing",
    "VRDeck_Missing",
]

# Impute and Encode Categoricals strictly on train_split
for col in cat_cols:
    train_split[col] = (
        train_split[col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    )
    val_split[col] = (
        val_split[col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    )
    test_split[col] = (
        test_split[col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    )

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_split[cat_cols] = encoder.fit_transform(train_split[cat_cols])
val_split[cat_cols] = encoder.transform(val_split[cat_cols])
test_split[cat_cols] = encoder.transform(test_split[cat_cols])

# Impute Numerics using medians fit strictly on train_split
num_medians = train_split[num_cols].median()
train_split[num_cols] = train_split[num_cols].fillna(num_medians)
val_split[num_cols] = val_split[num_cols].fillna(num_medians)
test_split[num_cols] = test_split[num_cols].fillna(num_medians)

# Standardize Numerical features: Fit strictly on train_split
scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_split[num_cols])
val_num_scaled = scaler.transform(val_split[num_cols])
test_num_scaled = scaler.transform(test_split[num_cols])

# Reassemble feature matrices
X_train = np.hstack([train_split[cat_cols].values, train_num_scaled]).astype(np.float32)
X_val = np.hstack([val_split[cat_cols].values, val_num_scaled]).astype(np.float32)
X_test = np.hstack([test_split[cat_cols].values, test_num_scaled]).astype(np.float32)

num_cats = len(cat_cols)
num_nums = len(num_cols)

# Compute safe embedding vocabulary sizes for categorical tokens
cat_cardinalities = []
for i in range(num_cats):
    max_val = int(X_train[:, i].max())
    cardinality = max(max_val + 10, 30)
    cat_cardinalities.append(cardinality)


# 6. FT-Transformer Architecture Definition
class FTTransformer(nn.Module):
    def __init__(
        self,
        num_cats,
        num_nums,
        cat_cardinalities,
        d_token=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.20,
        n_frequencies=32,
        token_dropout=0.10,
    ):
        super(FTTransformer, self).__init__()
        self.num_cats = num_cats
        self.num_nums = num_nums
        self.d_token = d_token
        self.cat_cardinalities = cat_cardinalities
        self.n_frequencies = n_frequencies
        self.token_dropout = token_dropout

        # Categorical feature tokenizers
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )

        # Learnable Hybrid Periodic Fourier + Direct Linear Embeddings for continuous features
        if num_nums > 0:
            self.frequencies = nn.Parameter(torch.randn(num_nums, n_frequencies))
            self.num_proj_weight = nn.Parameter(
                torch.empty(num_nums, 2 * n_frequencies, d_token)
            )
            self.num_proj_bias = nn.Parameter(torch.empty(num_nums, d_token))
            fan_in = 2 * n_frequencies
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.num_proj_weight, -bound, bound)
            nn.init.zeros_(self.num_proj_bias)

            self.num_linear_weight = nn.Parameter(torch.empty(num_nums, d_token))
            lin_bound = 1.0 / math.sqrt(d_token)
            nn.init.uniform_(self.num_linear_weight, -lin_bound, lin_bound)
        else:
            self.frequencies = None
            self.num_proj_weight = None
            self.num_proj_bias = None
            self.num_linear_weight = None

        # Token-level feature gating layer
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.Sigmoid(),
        )

        # Learnable [CLS] classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Readout Classification Head: contextualized [CLS] + mean-pooled features
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_token),
            nn.Linear(2 * d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x, x_num=None):
        if x_num is None:
            cat_x = x[:, : self.num_cats]
            num_x = x[:, self.num_cats :]
        else:
            cat_x = x
            num_x = x_num

        batch_size = cat_x.size(0)

        # Tokenize categoricals (offset -1 to 0 for unknown category)
        cat_tokens = []
        cat_idx = (cat_x + 1.0).clamp(min=0).long()
        for i, emb in enumerate(self.cat_embeddings):
            safe_idx = cat_idx[:, i].clamp(0, self.cat_cardinalities[i] - 1)
            cat_tokens.append(emb(safe_idx))

        if len(cat_tokens) > 0:
            cat_tokens = torch.stack(cat_tokens, dim=1)
        else:
            cat_tokens = torch.empty(batch_size, 0, self.d_token, device=x.device)

        # Tokenize continuous features with hybrid Periodic Fourier and Linear Embeddings
        if self.num_nums > 0:
            angles = 2.0 * math.pi * num_x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            fourier_tokens = (
                torch.einsum("bnf,nfd->bnd", fourier, self.num_proj_weight)
                + self.num_proj_bias.unsqueeze(0)
            )
            linear_tokens = num_x.unsqueeze(-1) * self.num_linear_weight.unsqueeze(0)
            num_tokens = fourier_tokens + linear_tokens
        else:
            num_tokens = torch.empty(batch_size, 0, self.d_token, device=x.device)

        # Concatenate feature tokens
        feature_tokens = torch.cat([cat_tokens, num_tokens], dim=1)

        # Token-level gating before transformer attention
        feature_tokens = feature_tokens * self.feature_gate(feature_tokens)

        # Feature Token Dropout during training
        if self.training and self.token_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full(
                    (batch_size, feature_tokens.size(1), 1),
                    1.0 - self.token_dropout,
                    device=feature_tokens.device,
                )
            )
            feature_tokens = feature_tokens * mask / (1.0 - self.token_dropout)

        # Prepend [CLS] token
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_expanded, feature_tokens], dim=1)

        # Transformer contextual self-attention
        tokens = self.transformer(tokens)

        # Readout: Contextualized [CLS] concatenated with mean-pooled feature tokens
        cls_out = tokens[:, 0, :]
        mean_pooled = tokens[:, 1:, :].mean(dim=1)
        pooled_repr = torch.cat([cls_out, mean_pooled], dim=-1)
        logits = self.head(pooled_repr).view(-1)
        return logits


# 7. Model Instantiation & Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FTTransformer(
    num_cats=num_cats,
    num_nums=num_nums,
    cat_cardinalities=cat_cardinalities,
    d_token=64,
    n_heads=4,
    n_layers=3,
    d_ff=128,
    dropout=0.20,
    token_dropout=0.10,
).to(device)

# Exponential Moving Average (EMA) Tracker
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.state_dict().items()
        }
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].lerp_(param, 1.0 - self.decay)
                    else:
                        self.shadow[name].copy_(param)

    def apply_shadow(self, model):
        self.backup = {
            name: param.clone().detach()
            for name, param in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


ema = EMA(model, decay=0.995)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

warmup_epochs = 3
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=warmup_epochs
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50 - warmup_epochs, eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

# PyTorch DataLoaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
)
test_dataset = TensorDataset(torch.from_numpy(X_test).float())

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

# 8. Training Loop with Early Stopping
max_epochs = 50
patience = 12
best_val_score = -1.0
best_epoch = 0
best_model_weights = copy.deepcopy(model.state_dict())
patience_counter = 0

epsilon = 0.05

for epoch in range(max_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        y_smooth = batch_y * (1.0 - epsilon) + 0.5 * epsilon
        loss = criterion(logits, y_smooth)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update(model)

        running_train_loss += loss.item() * batch_x.size(0)

    scheduler.step()
    epoch_train_loss = running_train_loss / len(train_dataset)

    # Validation phase on EMA weights
    ema.apply_shadow(model)
    model.eval()
    running_val_loss = 0.0
    val_preds_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            running_val_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            val_preds_list.append(probs.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_dataset)
    val_probs = np.concatenate(val_preds_list, axis=0)
    val_preds = (val_probs >= 0.5).astype(int)
    epoch_val_acc = accuracy_score(y_val, val_preds)

    print(
        f"Epoch {epoch + 1:02d}/{max_epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
    )

    if epoch_val_acc > best_val_score:
        best_val_score = epoch_val_acc
        best_epoch = epoch + 1
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    ema.restore(model)

    if patience_counter >= patience:
        break

# 9. Restore Best Model and Compute Final Validation Metric
model.load_state_dict(best_model_weights)
model.eval()

val_eval_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        val_eval_preds.append(probs.cpu().numpy())

final_val_probs = np.concatenate(val_eval_preds, axis=0)
final_val_preds = (final_val_probs >= 0.5).astype(int)
final_val_acc = float(accuracy_score(y_val, final_val_preds))

# 10. Generate Test Predictions and Submission File
test_eval_preds = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        test_eval_preds.append(probs.cpu().numpy())

test_probs = np.concatenate(test_eval_preds, axis=0)
test_preds_bool = test_probs >= 0.5

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"PassengerId": test_ids, "Transported": test_preds_bool.astype(bool)}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

# Submission Integrity Assertions
assert len(submission_df) == len(
    test_split
), f"Expected {len(test_split)} rows, found {len(submission_df)}"
assert list(submission_df.columns) == [
    "PassengerId",
    "Transported",
], "Incorrect columns"
assert not submission_df["Transported"].isna().any(), "Found NaN in predictions"
assert submission_df["Transported"].dtype == bool, "Target must be boolean"

# 11. Final Validation Metric Output
print(f"Final Validation Score: {final_val_acc}")

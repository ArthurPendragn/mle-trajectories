import copy
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
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

# Impute CryoSleep and VIP
combined["CryoSleep"] = combined["CryoSleep"].fillna(False).astype(int)
combined["VIP"] = combined["VIP"].fillna(False).astype(int)

# Expenditure aggregation and non-linear ratios
combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
combined["ZeroSpend"] = (combined["TotalSpend"] == 0).astype(int)
combined["LuxurySpend"] = combined["RoomService"] + combined["Spa"] + combined["VRDeck"]
combined["BasicSpend"] = combined["FoodCourt"] + combined["ShoppingMall"]

for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
    combined[f"Log_{col}"] = np.log1p(combined[col])

for col in spend_cols:
    combined[f"Pct_{col}"] = combined[col] / (combined["TotalSpend"] + 1.0)

# Cabin spatial regions
combined["CabinRegion"] = (combined["CabinNum"] // 300).fillna(-1)

# Route interaction
combined["HomePlanet"] = combined["HomePlanet"].fillna("Missing")
combined["Destination"] = combined["Destination"].fillna("Missing")
combined["Route"] = combined["HomePlanet"] + "_" + combined["Destination"]

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
    "Route",
    "AgeGroup",
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

# Compute categorical cardinalities from the training partition
cat_cardinalities = [int(max(train_split[c].max(), 0)) + 1 for c in categorical_cols]


# -----------------------------------------------------------------------------
# 4. Architecture Definition: Deep & Cross Network v2 (DCN-v2)
# -----------------------------------------------------------------------------
class CrossLayer(nn.Module):
    """Explicit feature crossing layer: x_{l+1} = LayerNorm(x_0 * W x_l + b + x_l)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features, bias=True)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x_0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        crossed = x_0 * self.linear(x_l) + x_l
        return self.norm(crossed)


class CrossNetwork(nn.Module):
    """Stack of recursive cross layers capturing degree-D interactions."""

    def __init__(self, in_features: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossLayer(in_features) for _ in range(num_layers)]
        )

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        x_l = x_0
        for layer in self.layers:
            x_l = layer(x_0, x_l)
        return x_l


class DeepBlock(nn.Module):
    """Feed-forward block with LayerNorm, Swish (SiLU), and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DCNv2(nn.Module):
    """Deep & Cross Network v2: Parallel Cross & Deep streams with categorical embeddings."""

    def __init__(
        self,
        cat_cardinalities: list,
        num_continuous: int,
        embedding_dim_cap: int = 16,
        cross_layers: int = 3,
        deep_hidden_dims: list = [128, 64],
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Categorical Entity Embeddings (+1 offset reserved for unknown/missing)
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for card in cat_cardinalities:
            emb_dim = min(embedding_dim_cap, max(4, int(card**0.5) + 2))
            self.embeddings.append(nn.Embedding(card + 2, emb_dim))
            total_emb_dim += emb_dim

        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.cont_norm = nn.BatchNorm1d(num_continuous)
        else:
            self.cont_norm = nn.Identity()

        # Total input dimension into Cross and Deep branches
        input_dim = total_emb_dim + num_continuous

        # Parallel Stream 1: Explicit Cross Network
        self.cross_net = CrossNetwork(input_dim, num_layers=cross_layers)

        # Parallel Stream 2: Deep MLP Network
        deep_layers = []
        curr_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.append(DeepBlock(curr_dim, hidden_dim, dropout_rate))
            curr_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)

        # Combined Prediction Head
        head_in_dim = input_dim + deep_hidden_dims[-1]
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor = None) -> torch.Tensor:
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            col_idx = torch.clamp(x_cat[:, i].long() + 1, min=0)
            embedded.append(emb_layer(col_idx))
        x_emb = torch.cat(embedded, dim=1) if len(embedded) > 0 else None

        if x_cont is not None and self.num_continuous > 0:
            x_cont_norm = self.cont_norm(x_cont.float())
            x_0 = (
                torch.cat([x_emb, x_cont_norm], dim=1)
                if x_emb is not None
                else x_cont_norm
            )
        else:
            x_0 = x_emb

        x_cross = self.cross_net(x_0)
        x_deep = self.deep_net(x_0)

        x_fused = torch.cat([x_cross, x_deep], dim=1)
        logits = self.head(x_fused)
        return logits


# -----------------------------------------------------------------------------
# 5. Model Initialization & Tensor DataLoaders
# -----------------------------------------------------------------------------
model = DCNv2(
    cat_cardinalities=cat_cardinalities,
    num_continuous=len(numerical_cols),
    embedding_dim_cap=16,
    cross_layers=3,
    deep_hidden_dims=[128, 64],
    dropout_rate=0.2,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-5,
)

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
# 6. Training Loop with Early Stopping & Validation Checkpointing
# -----------------------------------------------------------------------------
num_epochs = 50
patience = 12
best_val_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for b_cat, b_cont, b_y in train_loader:
        b_cat = b_cat.to(device)
        b_cont = b_cont.to(device)
        b_y = b_y.to(device)

        optimizer.zero_grad()
        logits = model(b_cat, b_cont)
        loss = criterion(logits, b_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item() * len(b_y)

    train_loss = total_train_loss / len(train_dataset)

    model.eval()
    total_val_loss = 0.0
    val_preds_list = []
    val_targets_list = []
    with torch.no_grad():
        for b_cat, b_cont, b_y in val_loader:
            b_cat = b_cat.to(device)
            b_cont = b_cont.to(device)
            b_y = b_y.to(device)

            logits = model(b_cat, b_cont)
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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# -----------------------------------------------------------------------------
# 7. Model Restoration & Final Hold-out Evaluation
# -----------------------------------------------------------------------------
model.load_state_dict(best_weights)
model.eval()

final_val_preds = []
final_val_targets = []
with torch.no_grad():
    for b_cat, b_cont, b_y in val_loader:
        b_cat = b_cat.to(device)
        b_cont = b_cont.to(device)
        logits = model(b_cat, b_cont)
        probs = torch.sigmoid(logits)
        final_val_preds.append((probs >= 0.5).int().cpu().numpy())
        final_val_targets.append(b_y.int().cpu().numpy())

final_val_preds = np.vstack(final_val_preds).ravel()
final_val_targets = np.vstack(final_val_targets).ravel()
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

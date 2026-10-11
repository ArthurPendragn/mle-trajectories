import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ------------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ------------------------------------------------------------------------------
# 1. Data Ingestion & Domain-Specific Feature Engineering
# ------------------------------------------------------------------------------
def find_data_file(filename: str) -> str:
    """Locate data files across expected directory structures."""
    candidates = [
        os.path.join(".", "input", filename),
        os.path.join(".", "input", "input", filename),
        os.path.join(".", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Unable to find {filename}. Checked: {candidates}")


def apply_domain_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific deterministic rules to resolve missing values."""
    df = df.copy()

    # Rule 1: CryoSleep passengers spend nothing on amenities
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df.loc[df["CryoSleep"] == True, col] = 0.0

    # Rule 2: Passengers with positive amenity spending cannot be in CryoSleep
    total_known_spend = df[amenity_cols].fillna(0).sum(axis=1)
    df.loc[(df["CryoSleep"].isna()) & (total_known_spend > 0), "CryoSleep"] = False

    # Rule 3: Children under 13 have zero luxury expenditure
    for col in amenity_cols:
        df.loc[(df["Age"] < 13) & (df[col].isna()), col] = 0.0

    # Rule 4: Physical deck constraints map deterministically to origin planets
    if "CabinDeck" in df.columns:
        df.loc[
            df["HomePlanet"].isna() & df["CabinDeck"].isin(["A", "B", "C", "T"]),
            "HomePlanet",
        ] = "Europa"
        df.loc[df["HomePlanet"].isna() & (df["CabinDeck"] == "G"), "HomePlanet"] = (
            "Earth"
        )

    # Rule 5: Earth passengers are never VIP
    df.loc[df["VIP"].isna() & (df["HomePlanet"] == "Earth"), "VIP"] = False

    return df


def engineer_features(raw_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Extract physical, financial, and relational features."""
    df = raw_df.copy()

    # 1. Passenger ID decomposition
    split_id = df["PassengerId"].str.split("_", expand=True)
    df["GroupID"] = split_id[0]
    df["GroupNum"] = split_id[1].astype(int)

    # 2. Cabin decomposition
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]

    # 3. Name & Surname extraction
    name_split = df["Name"].str.split(" ", expand=True)
    df["FirstName"] = name_split[0]
    df["LastName"] = name_split[1]

    # Apply deterministic domain rules
    df = apply_domain_cleaning(df)

    # Co-traveler imputation within group (deterministic physical facts)
    group_home = (
        df.dropna(subset=["HomePlanet"]).groupby("GroupID")["HomePlanet"].first()
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(df["GroupID"].map(group_home))

    group_side = df.dropna(subset=["CabinSide"]).groupby("GroupID")["CabinSide"].first()
    df["CabinSide"] = df["CabinSide"].fillna(df["GroupID"].map(group_side))

    # Spatial interaction token combining CabinDeck and CabinSide
    df["Deck_Side"] = (
        df["CabinDeck"].fillna("Missing").astype(str)
        + "_"
        + df["CabinSide"].fillna("Missing").astype(str)
    )

    # 4. Relational size features
    group_sizes = df["GroupID"].value_counts()
    df["GroupSize"] = df["GroupID"].map(group_sizes).astype(int)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    family_sizes = df["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_sizes).fillna(1).astype(int)
    df["HasFamily"] = (df["FamilySize"] > 1).astype(int)

    # 5. Financial & Expenditure Engineering
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = df[col].fillna(0.0)

    df["TotalSpend"] = df[amenity_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)

    # Group-level context aggregations
    df["GroupMeanTotalSpend"] = df.groupby("GroupID")["TotalSpend"].transform("mean")
    df["CryoSleep_num"] = df["CryoSleep"].map({True: 1.0, False: 0.0})
    df["GroupCryoSleepRatio"] = (
        df.groupby("GroupID")["CryoSleep_num"].transform("mean").fillna(0.0)
    )
    df.drop(columns=["CryoSleep_num"], inplace=True)

    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["CommunalSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"] + 1.0)
    df["CommunalRatio"] = df["CommunalSpend"] / (df["TotalSpend"] + 1.0)

    for col in amenity_cols:
        df[f"{col}_Ratio"] = df[col] / (df["TotalSpend"] + 1.0)
        df[f"{col}_log"] = np.log1p(df[col])

    df["TotalSpend_log"] = np.log1p(df["TotalSpend"])
    df["LuxurySpend_log"] = np.log1p(df["LuxurySpend"])
    df["CommunalSpend_log"] = np.log1p(df["CommunalSpend"])

    # 6. Demographic & Spatial Position Features
    df["Age_is_na"] = raw_df["Age"].isna().astype(int)
    df["IsInfant"] = (df["Age"] <= 4).astype(int)
    df["IsChild"] = ((df["Age"] > 4) & (df["Age"] <= 12)).astype(int)
    df["IsTeen"] = ((df["Age"] > 12) & (df["Age"] <= 17)).astype(int)
    df["IsSenior"] = (df["Age"] >= 55).astype(int)

    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)

    return df


def prepare_datasets():
    """Load, engineer features, split, encode, and return datasets and metadata."""
    os.makedirs("./working", exist_ok=True)

    train_path = find_data_file("train.csv")
    test_path = find_data_file("test.csv")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df = engineer_features(train_raw, is_train=True)
    test_df = engineer_features(test_raw, is_train=False)

    train_df["Transported"] = train_df["Transported"].astype(int)

    # Imputation using train statistics ONLY
    age_median = train_df["Age"].median()
    cabin_num_median = train_df["CabinNum"].median()

    train_df["Age"] = train_df["Age"].fillna(age_median)
    test_df["Age"] = test_df["Age"].fillna(age_median)

    train_df["CabinNum"] = train_df["CabinNum"].fillna(cabin_num_median)
    test_df["CabinNum"] = test_df["CabinNum"].fillna(cabin_num_median)

    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "Deck_Side",
    ]
    for col in cat_columns:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)

    # Encode categoricals strictly fitting encoders on training data
    for col in cat_columns:
        unique_vals = sorted(train_df[col].unique().tolist())
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        train_df[col + "_Code"] = train_df[col].map(val_to_idx).fillna(-1).astype(int)
        test_df[col + "_Code"] = test_df[col].map(val_to_idx).fillna(-1).astype(int)

    # Frequency encoding fit on train
    deck_freq = train_df["CabinDeck"].value_counts(normalize=True).to_dict()
    dest_freq = train_df["Destination"].value_counts(normalize=True).to_dict()

    train_df["Deck_Freq"] = train_df["CabinDeck"].map(deck_freq).fillna(0.0)
    test_df["Deck_Freq"] = test_df["CabinDeck"].map(deck_freq).fillna(0.0)

    train_df["Dest_Freq"] = train_df["Destination"].map(dest_freq).fillna(0.0)
    test_df["Dest_Freq"] = test_df["Destination"].map(dest_freq).fillna(0.0)

    numeric_features = [
        "Age",
        "CabinNum",
        "GroupSize",
        "GroupNum",
        "FamilySize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
        "LuxurySpend",
        "CommunalSpend",
        "LuxuryRatio",
        "CommunalRatio",
        "RoomService_Ratio",
        "FoodCourt_Ratio",
        "ShoppingMall_Ratio",
        "Spa_Ratio",
        "VRDeck_Ratio",
        "RoomService_log",
        "FoodCourt_log",
        "ShoppingMall_log",
        "Spa_log",
        "VRDeck_log",
        "TotalSpend_log",
        "LuxurySpend_log",
        "CommunalSpend_log",
        "Deck_Freq",
        "Dest_Freq",
        "GroupMeanTotalSpend",
        "GroupCryoSleepRatio",
    ]

    binary_features = [
        "IsSolo",
        "HasFamily",
        "ZeroSpend",
        "Age_is_na",
        "IsInfant",
        "IsChild",
        "IsTeen",
        "IsSenior",
    ]

    encoded_cat_features = [col + "_Code" for col in cat_columns] + ["CabinRegion"]

    all_features = numeric_features + binary_features + encoded_cat_features

    # Stratified Group K-Fold split to prevent co-traveler leakage
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(train_df, train_df["Transported"], groups=train_df["GroupID"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    val_split = train_df[train_df["fold"] == 0].copy()
    train_split = train_df[train_df["fold"] != 0].copy()

    # Fit QuantileTransformer strictly on training numerical features and transform validation and test sets
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    train_split[numeric_features] = qt.fit_transform(train_split[numeric_features])
    val_split[numeric_features] = qt.transform(val_split[numeric_features])
    test_df[numeric_features] = qt.transform(test_df[numeric_features])

    # Save artifacts to disk
    train_split.to_csv("./working/train_fold0.csv", index=False)
    val_split.to_csv("./working/val_fold0.csv", index=False)
    train_full = pd.concat([train_split, val_split]).sort_index()
    train_full.to_csv("./working/train_full_cv.csv", index=False)
    test_df.to_csv("./working/test_processed.csv", index=False)

    metadata = {
        "features": all_features,
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_encoded_features": encoded_cat_features,
        "target": "Transported",
        "id_col": "PassengerId",
        "num_folds": 5,
    }

    with open("./working/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return train_split, val_split, test_df, metadata


# ------------------------------------------------------------------------------
# 2. Tabular Residual Network Architecture
# ------------------------------------------------------------------------------
class GatedContextualFusion(nn.Module):
    """Gated Contextual Fusion layer combining continuous and categorical representations."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_val = nn.Linear(in_features, out_features)
        self.linear_gate = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.linear_val(x)
        gate = torch.sigmoid(self.linear_gate(x))
        return val * gate


class SEBlock(nn.Module):
    """Squeeze-and-Excitation feature recalibration block for tabular representations."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PeriodicNumericalTokenizer(nn.Module):
    """Projects each numerical feature into a token_dim embedding using learnable Fourier features."""

    def __init__(self, num_features: int, token_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        self.n_frequencies = n_frequencies

        # Learnable frequencies per feature
        self.frequencies = nn.Parameter(torch.randn(num_features, n_frequencies) * 0.5)

        # Linear projection from [cos, sin] bases (2 * n_frequencies) to token_dim per feature
        self.weight = nn.Parameter(torch.empty(num_features, 2 * n_frequencies, token_dim))
        self.bias = nn.Parameter(torch.empty(num_features, token_dim))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = 2 * n_frequencies
        bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_features)
        # 2 * pi * x * frequencies -> shape: (B, num_features, n_frequencies)
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        # Concat cos and sin -> shape: (B, num_features, 2 * n_frequencies)
        periodic = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        # Linear projection per feature -> shape: (B, num_features, token_dim)
        return torch.einsum("bnf,nfd->bnd", periodic, self.weight) + self.bias.unsqueeze(0)


NumericalFeatureTokenizer = PeriodicNumericalTokenizer


class TabularResNetBlock(nn.Module):
    """Residual block with Pre-LayerNorm, Gated Linear Unit (GLU), Dropout, and DropPath."""

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 2,
        dropout: float = 0.15,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        inner_dim = hidden_dim * expansion_factor
        self.linear_val = nn.Linear(hidden_dim, inner_dim)
        self.linear_gate = nn.Linear(hidden_dim, inner_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        norm_x = self.norm(x)
        val = self.linear_val(norm_x)
        gate = torch.sigmoid(self.linear_gate(norm_x))
        out = val * gate
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + self.drop_path(out)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet with per-feature Tokenizer, MHSA, and DropPath."""

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list,
        embedding_dims: list = None,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
        token_dim: int = 32,
        token_dropout_rate: float = 0.10,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_categories = len(cat_cardinalities)
        self.token_dim = token_dim
        self.token_dropout_rate = token_dropout_rate

        total_features = num_continuous + len(cat_cardinalities)

        if num_continuous > 0:
            self.num_bn = nn.BatchNorm1d(num_continuous)
            self.num_tokenizer = PeriodicNumericalTokenizer(num_continuous, token_dim)
        else:
            self.num_bn = nn.Identity()
            self.num_tokenizer = None

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(card + 2, token_dim, padding_idx=0)
                for card in cat_cardinalities
            ]
        )

        # Wide linear highway shortcut
        self.wide_num = (
            nn.Linear(num_continuous, 1, bias=False) if num_continuous > 0 else None
        )
        self.wide_cat = (
            nn.Linear(len(cat_cardinalities) * token_dim, 1, bias=False)
            if len(cat_cardinalities) > 0
            else None
        )

        self.mhsa = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(token_dim)

        self.token_proj = nn.Linear(total_features * token_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.se_block = SEBlock(hidden_dim, reduction=4)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TabularResNetBlock(
                    hidden_dim,
                    expansion_factor=2,
                    dropout=dropout,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        tokens_list = []

        if self.num_tokenizer is not None and x_num is not None:
            norm_num = self.num_bn(x_num)
            num_tokens = self.num_tokenizer(norm_num)
            tokens_list.append(num_tokens)

        cat_tokens = None
        if len(self.embeddings) > 0 and x_cat is not None:
            cat_tokens_list = []
            for i, emb_layer in enumerate(self.embeddings):
                col_cat = torch.clamp(
                    x_cat[:, i] + 1, min=0, max=emb_layer.num_embeddings - 1
                )
                cat_tokens_list.append(emb_layer(col_cat).unsqueeze(1))
            cat_tokens = torch.cat(cat_tokens_list, dim=1)
            tokens_list.append(cat_tokens)

        if not tokens_list:
            raise ValueError("No input features provided to TabularResNet.")

        tokens = torch.cat(tokens_list, dim=1)

        # Feature Token Dropout during training
        if self.training and self.token_dropout_rate > 0.0:
            keep_prob = 1.0 - self.token_dropout_rate
            mask = torch.bernoulli(
                torch.full(
                    (tokens.size(0), tokens.size(1), 1),
                    keep_prob,
                    device=tokens.device,
                )
            )
            tokens = tokens * mask / keep_prob

        # Multi-Head Feature Self-Attention with residual connection and LayerNorm
        attn_out, _ = self.mhsa(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Project recalibrated token sequence into hidden_dim
        flat_tokens = tokens.flatten(start_dim=1)
        x = self.token_proj(flat_tokens)

        x = self.input_norm(x)
        x = self.se_block(x)
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.head(x)

        # Direct wide highway shortcut
        if self.wide_num is not None and x_num is not None:
            logits = logits + self.wide_num(x_num)
        if self.wide_cat is not None and cat_tokens is not None:
            logits = logits + self.wide_cat(cat_tokens.flatten(start_dim=1))

        return logits


# ------------------------------------------------------------------------------
# 3. Tabular Dataset & PyTorch DataLoader
# ------------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]


# ------------------------------------------------------------------------------
# 4. Main Training and Validation Pipeline
# ------------------------------------------------------------------------------
def main():
    train_split, val_split, test_df, metadata = prepare_datasets()
    full_train_df = pd.concat([train_split, val_split]).sort_index()

    continuous_cols = metadata.get("numeric_features", []) + metadata.get(
        "binary_features", []
    )
    categorical_cols = metadata.get("categorical_encoded_features", [])

    x_num_test = test_df[continuous_cols].values.astype(np.float32)
    x_cat_test = test_df[categorical_cols].values.astype(np.int64)

    # Determine categorical cardinalities from training set
    cat_cardinalities = []
    for col in categorical_cols:
        col_max = int(full_train_df[col].max())
        cat_cardinalities.append(max(10, col_max + 2))

    test_cat_tensor = x_cat_test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = metadata.get("num_folds", 5)

    oof_probs = np.zeros(len(full_train_df), dtype=np.float32)
    oof_targets = full_train_df["Transported"].values.astype(np.float32)
    test_fold_probs = np.zeros(len(test_df), dtype=np.float32)

    epochs = 35
    patience = 8

    for fold in range(num_folds):
        print(f"\n--- Training Fold {fold}/{num_folds} ---")
        val_mask = full_train_df["fold"] == fold
        train_mask = full_train_df["fold"] != fold

        fold_train_df = full_train_df[train_mask]
        fold_val_df = full_train_df[val_mask]
        val_indices = np.where(val_mask.to_numpy())[0]

        # Strictly fit QuantileTransformer on this fold's training split only
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        x_num_train = fold_train_df[continuous_cols].values.astype(np.float32)
        x_num_val = fold_val_df[continuous_cols].values.astype(np.float32)
        x_num_test_fold = test_df[continuous_cols].values.astype(np.float32)

        num_indices = [continuous_cols.index(col) for col in metadata["numeric_features"]]
        x_num_train[:, num_indices] = qt.fit_transform(x_num_train[:, num_indices])
        x_num_val[:, num_indices] = qt.transform(x_num_val[:, num_indices])
        x_num_test_fold[:, num_indices] = qt.transform(x_num_test_fold[:, num_indices])

        x_cat_train = fold_train_df[categorical_cols].values.astype(np.int64)
        y_train = fold_train_df["Transported"].values.astype(np.float32)

        x_cat_val = fold_val_df[categorical_cols].values.astype(np.int64)
        y_val = fold_val_df["Transported"].values.astype(np.float32)

        train_dataset = TabularDataset(x_num_train, x_cat_train, y_train)
        val_dataset = TabularDataset(x_num_val, x_cat_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + fold)

        model = TabularResNet(
            num_continuous=len(continuous_cols),
            cat_cardinalities=cat_cardinalities,
            hidden_dim=128,
            num_blocks=3,
            dropout=0.15,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        best_model_path = f"./working/best_tabular_resnet_fold{fold}.pt"

        label_smoothing = 0.05
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for x_num_b, x_cat_b, y_b in train_loader:
                x_num_b = x_num_b.to(device)
                x_cat_b = x_cat_b.to(device)
                y_b = y_b.to(device)

                # Label smoothing regularization
                y_smoothed = y_b * (1.0 - label_smoothing) + 0.5 * label_smoothing

                optimizer.zero_grad()
                logits = model(x_num_b, x_cat_b)
                loss = criterion(logits, y_smoothed)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_train_loss += loss.item() * len(y_b)

            scheduler.step()
            epoch_train_loss = running_train_loss / len(train_dataset)

            # Validation pass
            model.eval()
            running_val_loss = 0.0
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for x_num_b, x_cat_b, y_b in val_loader:
                    x_num_b = x_num_b.to(device)
                    x_cat_b = x_cat_b.to(device)
                    y_b = y_b.to(device)

                    logits = model(x_num_b, x_cat_b)
                    loss = criterion(logits, y_b)
                    running_val_loss += loss.item() * len(y_b)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append(probs)
                    val_targets_list.append(y_b.cpu().numpy())

            epoch_val_loss = running_val_loss / len(val_dataset)
            val_probs = np.vstack(val_preds_list).flatten()
            val_targets = np.vstack(val_targets_list).flatten()
            epoch_val_acc = accuracy_score(val_targets, (val_probs >= 0.5).astype(int))

            print(
                f"Fold {fold} Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
            )

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best checkpoint for this fold
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        fold_val_preds = []
        with torch.no_grad():
            for x_num_b, x_cat_b, _ in val_loader:
                x_num_b = x_num_b.to(device)
                x_cat_b = x_cat_b.to(device)
                logits = model(x_num_b, x_cat_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_val_preds.append(probs)

        fold_val_probs = np.vstack(fold_val_preds).flatten()
        oof_probs[val_indices] = fold_val_probs
        fold_acc = accuracy_score(y_val, (fold_val_probs >= 0.5).astype(int))
        print(f"Fold {fold} Best Val Acc: {fold_acc:.4f}")

        # Fold test predictions
        fold_test_dataset = TabularDataset(x_num_test_fold, test_cat_tensor)
        fold_test_loader = DataLoader(fold_test_dataset, batch_size=256, shuffle=False)
        fold_test_preds = []
        with torch.no_grad():
            for x_num_b, x_cat_b in fold_test_loader:
                x_num_b = x_num_b.to(device)
                x_cat_b = x_cat_b.to(device)
                logits = model(x_num_b, x_cat_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_test_preds.append(probs)

        fold_test_probs = np.vstack(fold_test_preds).flatten()
        test_fold_probs += fold_test_probs / num_folds

    # Global Out-Of-Fold Validation Score
    final_val_acc = accuracy_score(oof_targets, (oof_probs >= 0.5).astype(int))

    test_predictions = (test_fold_probs >= 0.5).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_acc}")


if __name__ == "__main__":
    main()
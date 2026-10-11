import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
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

    # Save artifacts to disk
    train_split.to_csv("./working/train_fold0.csv", index=False)
    val_split.to_csv("./working/val_fold0.csv", index=False)
    train_df.to_csv("./working/train_full_cv.csv", index=False)
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
class TabularResNetBlock(nn.Module):
    """Residual block with Pre-LayerNorm, GELU, and Dropout."""

    def __init__(
        self, hidden_dim: int, expansion_factor: int = 2, dropout: float = 0.15
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):
    """Deep Tabular Residual Network with Categorical Entity Embeddings."""

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list,
        embedding_dims: list = None,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_categories = len(cat_cardinalities)

        if num_continuous > 0:
            self.num_bn = nn.BatchNorm1d(num_continuous)
            self.num_proj = nn.Linear(num_continuous, hidden_dim)
        else:
            self.num_bn = nn.Identity()
            self.num_proj = None

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if embedding_dims is None:
            embedding_dims = [
                min(16, max(4, int(2 * (card**0.5)))) for card in cat_cardinalities
            ]

        for card, emb_dim in zip(cat_cardinalities, embedding_dims):
            self.embeddings.append(nn.Embedding(card + 2, emb_dim, padding_idx=0))
            total_emb_dim += emb_dim

        if total_emb_dim > 0:
            self.cat_proj = nn.Linear(total_emb_dim, hidden_dim)
        else:
            self.cat_proj = None

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TabularResNetBlock(hidden_dim, expansion_factor=2, dropout=dropout)
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
        features = []

        if self.num_proj is not None and x_num is not None:
            norm_num = self.num_bn(x_num)
            proj_num = self.num_proj(norm_num)
            features.append(proj_num)

        if self.cat_proj is not None and x_cat is not None and len(self.embeddings) > 0:
            emb_list = []
            for i, emb_layer in enumerate(self.embeddings):
                col_cat = torch.clamp(
                    x_cat[:, i] + 1, min=0, max=emb_layer.num_embeddings - 1
                )
                emb_list.append(emb_layer(col_cat))
            cat_concat = torch.cat(emb_list, dim=-1)
            proj_cat = self.cat_proj(cat_concat)
            features.append(proj_cat)

        if len(features) == 2:
            x = features[0] + features[1]
        elif len(features) == 1:
            x = features[0]
        else:
            raise ValueError("No input features provided to TabularResNet.")

        x = self.input_norm(x)
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.head(x)
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

    continuous_cols = metadata.get("numeric_features", []) + metadata.get(
        "binary_features", []
    )
    categorical_cols = metadata.get("categorical_encoded_features", [])

    x_num_train = train_split[continuous_cols].values.astype(np.float32)
    x_cat_train = train_split[categorical_cols].values.astype(np.int64)
    y_train = train_split["Transported"].values.astype(np.float32)

    x_num_val = val_split[continuous_cols].values.astype(np.float32)
    x_cat_val = val_split[categorical_cols].values.astype(np.int64)
    y_val = val_split["Transported"].values.astype(np.float32)

    x_num_test = test_df[continuous_cols].values.astype(np.float32)
    x_cat_test = test_df[categorical_cols].values.astype(np.int64)

    # Determine categorical cardinalities from training set
    cat_cardinalities = []
    for col in categorical_cols:
        col_max = int(train_split[col].max())
        cat_cardinalities.append(max(10, col_max + 2))

    train_dataset = TabularDataset(x_num_train, x_cat_train, y_train)
    val_dataset = TabularDataset(x_num_val, x_cat_val, y_val)
    test_dataset = TabularDataset(x_num_test, x_cat_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularResNet(
        num_continuous=len(continuous_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.15,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 35
    patience = 8
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_model_path = "./working/best_tabular_resnet.pt"

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_num_b, x_cat_b)
            loss = criterion(logits, y_b)
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
            f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
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

    # --------------------------------------------------------------------------
    # 5. Final Evaluation and Test Inference
    # --------------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_final_preds = []
    with torch.no_grad():
        for x_num_b, x_cat_b, _ in val_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            logits = model(x_num_b, x_cat_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_final_preds.append(probs)

    final_val_probs = np.vstack(val_final_preds).flatten()
    final_val_acc = accuracy_score(y_val, (final_val_probs >= 0.5).astype(int))

    test_preds_list = []
    with torch.no_grad():
        for x_num_b, x_cat_b in test_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            logits = model(x_num_b, x_cat_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds_list.append(probs)

    test_probs = np.vstack(test_preds_list).flatten()
    test_predictions = (test_probs >= 0.5).astype(bool)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_predictions}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_acc}")


if __name__ == "__main__":
    main()

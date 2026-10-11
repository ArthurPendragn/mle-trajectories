import copy
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Step 1: Data Processing & Feature Engineering
# ---------------------------------------------------------------------------
def locate_input_file(filename):
    """Search possible paths for the input file."""
    candidates = [
        os.path.join("./input", filename),
        os.path.join("./input/input", filename),
        os.path.join(".", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"File {filename} not found in expected locations.")


def extract_domain_features(df, global_manifest=None):
    """Extract deterministic structural, relational, and domain features prior to stateful imputation."""
    df = df.copy()

    # Build or parse global manifest to allow global demographic and relational propagation
    if global_manifest is None:
        try:
            tr_p = locate_input_file("train.csv")
            te_p = locate_input_file("test.csv")
            tr_df = pd.read_csv(tr_p)
            te_df = pd.read_csv(te_p)
            gm = pd.concat(
                [tr_df.drop(columns=["Transported"], errors="ignore"), te_df],
                ignore_index=True,
            )
        except Exception:
            gm = df.copy()
    else:
        gm = global_manifest.copy()

    # Ensure structural identity columns exist on global manifest
    if "GroupId" not in gm.columns:
        gm_p_parts = gm["PassengerId"].str.split("_", expand=True)
        gm["GroupId"] = gm_p_parts[0]
        gm["GroupNum"] = gm_p_parts[1].astype(int)
    if "CabinDeck" not in gm.columns:
        gm_c_parts = gm["Cabin"].str.split("/", expand=True)
        gm["CabinDeck"] = gm_c_parts[0]
        gm["CabinNum"] = pd.to_numeric(gm_c_parts[1], errors="coerce")
        gm["CabinSide"] = gm_c_parts[2]
    if "Surname" not in gm.columns:
        gm_names = gm["Name"].str.split(" ", expand=True)
        gm["Surname"] = gm_names[1].fillna("Unknown")

    # 1. Structural passenger decomposition on local df
    p_parts = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = p_parts[0]
    df["GroupNum"] = p_parts[1].astype(int)

    c_parts = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = c_parts[0]
    df["CabinNum"] = pd.to_numeric(c_parts[1], errors="coerce")
    df["CabinSide"] = c_parts[2]

    names = df["Name"].str.split(" ", expand=True)
    df["Surname"] = names[1].fillna("Unknown")

    # Global manifest demographic counts and group leader flag
    global_group_counts = gm["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(global_group_counts).fillna(1).astype(int)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["IsGroupLeader"] = (df["GroupNum"] == 1).astype(int)

    valid_surnames = gm[gm["Surname"] != "Unknown"]["Surname"]
    global_surname_counts = valid_surnames.value_counts()
    df["FamilySize"] = df["Surname"].map(global_surname_counts).fillna(1).astype(int)
    df.loc[df["Surname"] == "Unknown", "FamilySize"] = 1

    # 2. Relational co-traveler and deductive imputation
    # Deterministic physical deck rules: Decks A, B, C, T are Europa; Deck G is Earth
    gm.loc[gm["CabinDeck"].isin(["A", "B", "C", "T"]) & gm["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    gm.loc[(gm["CabinDeck"] == "G") & gm["HomePlanet"].isna(), "HomePlanet"] = "Earth"
    df.loc[df["CabinDeck"].isin(["A", "B", "C", "T"]) & df["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    df.loc[(df["CabinDeck"] == "G") & df["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Propagate HomePlanet via GroupId and Surname
    hp_valid = gm.dropna(subset=["HomePlanet"])
    hp_group = hp_valid.groupby("GroupId")["HomePlanet"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["GroupId"].map(hp_group))

    hp_sur_valid = hp_valid[hp_valid["Surname"] != "Unknown"]
    hp_surname = hp_sur_valid.groupby("Surname")["HomePlanet"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["Surname"].map(hp_surname))

    # Re-apply deck rules if Deck was known
    df.loc[df["CabinDeck"].isin(["A", "B", "C", "T"]) & df["HomePlanet"].isna(), "HomePlanet"] = "Europa"
    df.loc[(df["CabinDeck"] == "G") & df["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    # Propagate CabinDeck via GroupId and Surname
    deck_valid = gm.dropna(subset=["CabinDeck"])
    deck_valid = deck_valid[~deck_valid["CabinDeck"].isin(["Missing", ""])]
    deck_group = deck_valid.groupby("GroupId")["CabinDeck"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinDeck"] = df["CabinDeck"].fillna(df["GroupId"].map(deck_group))

    deck_sur_valid = deck_valid[deck_valid["Surname"] != "Unknown"]
    deck_surname = deck_sur_valid.groupby("Surname")["CabinDeck"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinDeck"] = df["CabinDeck"].fillna(df["Surname"].map(deck_surname))

    # Propagate CabinSide via GroupId and Surname
    side_valid = gm.dropna(subset=["CabinSide"])
    side_valid = side_valid[~side_valid["CabinSide"].isin(["Missing", ""])]
    side_group = side_valid.groupby("GroupId")["CabinSide"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinSide"] = df["CabinSide"].fillna(df["GroupId"].map(side_group))

    side_sur_valid = side_valid[side_valid["Surname"] != "Unknown"]
    side_surname = side_sur_valid.groupby("Surname")["CabinSide"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
    df["CabinSide"] = df["CabinSide"].fillna(df["Surname"].map(side_surname))

    # Propagate CabinNum via GroupId and Surname
    num_valid = gm.dropna(subset=["CabinNum"])
    num_group = num_valid.groupby("GroupId")["CabinNum"].median()
    df["CabinNum"] = df["CabinNum"].fillna(df["GroupId"].map(num_group))

    num_sur_valid = num_valid[num_valid["Surname"] != "Unknown"]
    num_surname = num_sur_valid.groupby("Surname")["CabinNum"].median()
    df["CabinNum"] = df["CabinNum"].fillna(df["Surname"].map(num_surname))

    df["CabinDeck"] = df["CabinDeck"].fillna("Missing")
    df["CabinSide"] = df["CabinSide"].fillna("Missing")

    # Spatial hull features
    df["CabinDeck_Side"] = df["CabinDeck"] + "_" + df["CabinSide"]
    df["CabinZone"] = df["CabinNum"] // 300

    # 3. Amenity expenditure, luxury vs communal spending, and constraints
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physics constraint: Confirmed CryoSleep passengers spend 0 on luxury amenities
    is_cryo = df["CryoSleep"] == True
    for col in amenity_cols:
        df.loc[is_cryo & df[col].isna(), col] = 0.0

    # Physics constraint: Children under 13 cannot spend on amenities
    is_child = df["Age"] < 13
    for col in amenity_cols:
        df.loc[is_child & df[col].isna(), col] = 0.0

    # Total, Luxury, and Communal spending aggregations
    df["TotalSpending"] = df[amenity_cols].sum(axis=1, skipna=True)
    df["LuxurySpending"] = (
        df["RoomService"].fillna(0.0)
        + df["Spa"].fillna(0.0)
        + df["VRDeck"].fillna(0.0)
    )
    df["CommunalSpending"] = (
        df["FoodCourt"].fillna(0.0)
        + df["ShoppingMall"].fillna(0.0)
    )

    # Explicit spend and zero-spend flags
    df["HasSpent"] = (df["TotalSpending"] > 0).astype(int)
    df["ZeroSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["ZeroLuxury"] = (df["LuxurySpending"] == 0).astype(int)
    df["ZeroCommunal"] = (df["CommunalSpending"] == 0).astype(int)

    # Inverse physics constraint: Anyone who spent money was NOT in CryoSleep
    spent_money = df["TotalSpending"] > 0
    df.loc[spent_money & df["CryoSleep"].isna(), "CryoSleep"] = False

    # Proportional and log transformations
    for col in amenity_cols:
        df[f"{col}_Ratio"] = df[col].fillna(0.0) / (df["TotalSpending"] + 1.0)
        df[f"Log_{col}"] = np.log1p(np.maximum(df[col].fillna(0.0), 0.0))

    df["Log_TotalSpending"] = np.log1p(np.maximum(df["TotalSpending"], 0.0))
    df["Log_LuxurySpending"] = np.log1p(np.maximum(df["LuxurySpending"], 0.0))
    df["Log_CommunalSpending"] = np.log1p(np.maximum(df["CommunalSpending"], 0.0))

    df["Luxury_Ratio"] = df["LuxurySpending"] / (df["TotalSpending"] + 1.0)
    df["Communal_Ratio"] = df["CommunalSpending"] / (df["TotalSpending"] + 1.0)

    # Number of amenities used
    df["AmenitiesUsedCount"] = (df[amenity_cols].fillna(0.0) > 0).sum(axis=1)

    # VIP deductive constraints
    df.loc[df["Age"] < 18, "VIP"] = False
    df.loc[df["HomePlanet"] == "Earth", "VIP"] = False
    df.loc[df["CabinDeck"].isin(["G", "T"]), "VIP"] = False

    return df


def prepare_datasets():
    os.makedirs("./working", exist_ok=True)

    train_path = locate_input_file("train.csv")
    test_path = locate_input_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df[["PassengerId"]].copy()

    # Extract target from train
    y = train_df["Transported"].astype(int).values

    # Unified global passenger manifest (excluding labels)
    full_manifest = pd.concat(
        [train_df.drop(columns=["Transported"], errors="ignore"), test_df],
        ignore_index=True,
    )

    # Stratified hold-out split (80% train, 20% validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(train_df, y))

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Apply deterministic and relational feature extraction with global manifest
    X_train_raw = extract_domain_features(train_split, global_manifest=full_manifest)
    X_val_raw = extract_domain_features(val_split, global_manifest=full_manifest)
    X_test_raw = extract_domain_features(test_df, global_manifest=full_manifest)

    # Categorical and numerical columns definition
    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "CabinDeck_Side",
    ]
    numeric_cols = [
        "Age",
        "CabinNum",
        "CabinZone",
        "GroupSize",
        "GroupNum",
        "IsAlone",
        "IsGroupLeader",
        "FamilySize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpending",
        "LuxurySpending",
        "CommunalSpending",
        "HasSpent",
        "ZeroSpending",
        "ZeroLuxury",
        "ZeroCommunal",
        "AmenitiesUsedCount",
        "Log_RoomService",
        "Log_FoodCourt",
        "Log_ShoppingMall",
        "Log_Spa",
        "Log_VRDeck",
        "Log_TotalSpending",
        "Log_LuxurySpending",
        "Log_CommunalSpending",
        "RoomService_Ratio",
        "FoodCourt_Ratio",
        "ShoppingMall_Ratio",
        "Spa_Ratio",
        "VRDeck_Ratio",
        "Luxury_Ratio",
        "Communal_Ratio",
    ]
    binary_cols = [
        "IsAlone",
        "IsGroupLeader",
        "HasSpent",
        "ZeroSpending",
        "ZeroLuxury",
        "ZeroCommunal",
    ]

    # STRICT ISOLATION: Learn imputers strictly from training split
    num_medians = {col: X_train_raw[col].median() for col in numeric_cols}
    for col in numeric_cols:
        if pd.isna(num_medians[col]):
            num_medians[col] = 0.0

    cat_modes = {}
    for col in categorical_cols:
        mode_val = X_train_raw[col].mode(dropna=True)
        cat_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"

    def apply_imputation(df):
        df_imputed = df.copy()
        for col in numeric_cols:
            df_imputed[col] = df_imputed[col].fillna(num_medians[col])
        for col in categorical_cols:
            df_imputed[col] = (
                df_imputed[col].astype(object).fillna(cat_modes[col]).astype(str)
            )
        return df_imputed

    X_train_imp = apply_imputation(X_train_raw)
    X_val_imp = apply_imputation(X_val_raw)
    X_test_imp = apply_imputation(X_test_raw)

    # One-hot encoding fitted strictly on train categories
    encoded_train_list = [X_train_imp[numeric_cols].reset_index(drop=True)]
    encoded_val_list = [X_val_imp[numeric_cols].reset_index(drop=True)]
    encoded_test_list = [X_test_imp[numeric_cols].reset_index(drop=True)]

    for col in categorical_cols:
        train_cats = sorted(X_train_imp[col].unique().tolist())
        for cat in train_cats:
            col_name = f"{col}_{cat}"
            encoded_train_list.append(
                pd.Series((X_train_imp[col] == cat).astype(float), name=col_name)
            )
            encoded_val_list.append(
                pd.Series((X_val_imp[col] == cat).astype(float), name=col_name)
            )
            encoded_test_list.append(
                pd.Series((X_test_imp[col] == cat).astype(float), name=col_name)
            )

    X_train_final = pd.concat(encoded_train_list, axis=1)
    X_val_final = pd.concat(encoded_val_list, axis=1)
    X_test_final = pd.concat(encoded_test_list, axis=1)

    # Standard scale continuous features (strictly fit on train)
    scaler = StandardScaler()
    continuous_features = [
        col for col in numeric_cols if col not in binary_cols
    ]
    X_train_final[continuous_features] = scaler.fit_transform(
        X_train_final[continuous_features]
    )
    X_val_final[continuous_features] = scaler.transform(
        X_val_final[continuous_features]
    )
    X_test_final[continuous_features] = scaler.transform(
        X_test_final[continuous_features]
    )

    y_train_df = pd.DataFrame({"Transported": y_train})
    y_val_df = pd.DataFrame({"Transported": y_val})

    # Save prepared data to ./working
    X_train_final.to_csv("./working/X_train.csv", index=False)
    y_train_df.to_csv("./working/y_train.csv", index=False)
    X_val_final.to_csv("./working/X_val.csv", index=False)
    y_val_df.to_csv("./working/y_val.csv", index=False)
    X_test_final.to_csv("./working/X_test.csv", index=False)
    test_ids.to_csv("./working/test_ids.csv", index=False)

    return (
        X_train_final,
        y_train_df,
        X_val_final,
        y_val_df,
        X_test_final,
        test_ids,
    )


# ---------------------------------------------------------------------------
# Step 2: TabNet Architecture
# ---------------------------------------------------------------------------
class GLUBlock(nn.Module):
    """Gated Linear Unit (GLU) layer with normalization, linear projection, and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_features * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.fc(x))
        out = self.dropout(out)
        val, gate = out.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class FeatureBlock(nn.Module):
    """Feature Transformer block with optional residual connection scaled by 1/sqrt(2)."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GLUBlock(in_features, out_features, dropout=dropout)
        self.use_residual = in_features == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.glu(x)
        if self.use_residual:
            out = (x + out) * 0.7071067811865475  # 1 / sqrt(2)
        return out


class AttentiveTransformer(nn.Module):
    """Learns sparse feature selection masks at each decision step."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(
        self, a_prev: torch.Tensor, prior: torch.Tensor, tau: float = 0.5
    ) -> torch.Tensor:
        x = self.bn(self.fc(a_prev))
        x = x * prior
        mask = torch.softmax(x / tau, dim=-1)
        return mask


class TabNetClassifier(nn.Module):
    """TabNet architecture for tabular classification."""

    def __init__(
        self,
        in_features: int,
        n_d: int = 24,
        n_a: int = 24,
        n_steps: int = 4,
        gamma: float = 1.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.hidden_dim = n_d + n_a

        self.input_bn = nn.BatchNorm1d(in_features)

        self.shared_block1 = FeatureBlock(in_features, self.hidden_dim, dropout=dropout)
        self.shared_block2 = FeatureBlock(
            self.hidden_dim, self.hidden_dim, dropout=dropout
        )

        self.step_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    FeatureBlock(self.hidden_dim, self.hidden_dim, dropout=dropout),
                    FeatureBlock(self.hidden_dim, self.hidden_dim, dropout=dropout),
                )
                for _ in range(n_steps)
            ]
        )

        self.attentive_transformers = nn.ModuleList(
            [AttentiveTransformer(n_a, in_features) for _ in range(n_steps)]
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_d, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_norm = self.input_bn(x)

        prior = torch.ones_like(x_norm)
        shared_init = self.shared_block2(self.shared_block1(x_norm))
        a_prev = shared_init[:, self.n_d :]

        decision_out = torch.zeros(batch_size, self.n_d, device=x.device, dtype=x.dtype)

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](a_prev, prior)
            prior = prior * (self.gamma - mask)

            x_masked = mask * x_norm
            feat = self.shared_block1(x_masked)
            feat = self.shared_block2(feat)
            feat = self.step_blocks[step](feat)

            d_step = F.relu(feat[:, : self.n_d])
            a_prev = feat[:, self.n_d :]

            decision_out = decision_out + d_step

        logits = self.head(decision_out)
        return logits


# ---------------------------------------------------------------------------
# Step 3: Training, Evaluation & Test Inference
# ---------------------------------------------------------------------------
def main():
    X_train_final, y_train_df, X_val_final, y_val_df, X_test_final, test_ids = (
        prepare_datasets()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train_final.shape[1]
    model = TabNetClassifier(
        in_features=input_dim,
        n_d=24,
        n_a=24,
        n_steps=4,
        gamma=1.3,
        dropout=0.15,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Tensor representations
    X_tr_tensor = torch.tensor(X_train_final.values, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train_df["Transported"].values, dtype=torch.float32)

    X_va_tensor = torch.tensor(X_val_final.values, dtype=torch.float32)
    y_va_tensor = torch.tensor(y_val_df["Transported"].values, dtype=torch.float32)

    X_te_tensor = torch.tensor(X_test_final.values, dtype=torch.float32)

    pin_mem = torch.cuda.is_available()
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_mem,
    )

    val_dataset = TensorDataset(X_va_tensor, y_va_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_mem,
    )

    test_dataset = TensorDataset(X_te_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_mem,
    )

    epochs = 60
    patience = 14
    patience_counter = 0
    best_val_acc = 0.0
    best_model_weights = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_curr = len(y_batch)
            total_train_loss += loss.item() * batch_size_curr
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_samples += batch_size_curr

        scheduler.step()

        train_loss = total_train_loss / train_samples
        train_acc = train_correct / train_samples

        # Validation step
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for x_val_b, y_val_b in val_loader:
                x_val_b = x_val_b.to(device)
                y_val_b = y_val_b.to(device)

                logits_val = model(x_val_b).squeeze(-1)
                loss_val = criterion(logits_val, y_val_b)

                batch_size_val = len(y_val_b)
                total_val_loss += loss_val.item() * batch_size_val
                preds_val = (torch.sigmoid(logits_val) >= 0.5).float()
                val_correct += (preds_val == y_val_b).sum().item()
                val_samples += batch_size_val

        val_loss = total_val_loss / val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Restore best model checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Validation evaluation using model inference
    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x_val_b, y_val_b in val_loader:
            x_val_b = x_val_b.to(device)
            logits_val = model(x_val_b).squeeze(-1)
            preds_val = (torch.sigmoid(logits_val) >= 0.5).cpu().numpy().astype(bool)
            val_preds_list.extend(preds_val)
            val_targets_list.extend(y_val_b.numpy().astype(bool))

    final_score = accuracy_score(val_targets_list, val_preds_list)

    # Test inference using model forward pass
    test_preds_list = []
    with torch.no_grad():
        for (x_test_b,) in test_loader:
            x_test_b = x_test_b.to(device)
            logits_test = model(x_test_b).squeeze(-1)
            preds_test = (torch.sigmoid(logits_test) >= 0.5).cpu().numpy().astype(bool)
            test_preds_list.extend(preds_test)

    # Generate submission file matching sample_submission format
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "PassengerId": test_ids["PassengerId"],
            "Transported": test_preds_list,
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================================================================
# 0. Reproducibility & Environment Setup
# =========================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Data Ingestion & Domain Feature Engineering
# =========================================================================
def locate_data_dir() -> str:
    candidates = ["./input/input", "./input", "../input", "."]
    for path in candidates:
        if os.path.exists(os.path.join(path, "train.csv")) and os.path.exists(
            os.path.join(path, "test.csv")
        ):
            return path
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in input directories."
    )


def engineer_features() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    data_dir = locate_data_dir()
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    n_train = len(train_df)
    target_col = "Transported"
    y_train = train_df[target_col].astype(bool).astype(int)
    test_df[target_col] = np.nan
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 0. Missingness Indicators (captured prior to imputation & constraint rules)
    full_df["CryoSleep_is_missing"] = full_df["CryoSleep"].isna().astype(int)
    full_df["Cabin_is_missing"] = full_df["Cabin"].isna().astype(int)
    full_df["Age_is_missing"] = full_df["Age"].isna().astype(int)
    full_df["VIP_is_missing"] = full_df["VIP"].isna().astype(int)

    # 1. Structural Identifiers & Group Decomposition
    full_df["PassengerGroup"] = full_df["PassengerId"].apply(lambda x: x.split("_")[0])
    full_df["PassengerNum"] = full_df["PassengerId"].apply(
        lambda x: int(x.split("_")[1])
    )

    group_size_map = full_df["PassengerGroup"].value_counts().to_dict()
    full_df["GroupSize"] = full_df["PassengerGroup"].map(group_size_map)
    full_df["IsSolo"] = (full_df["GroupSize"] == 1).astype(int)

    # 2. Cabin Spatial Decomposition & Shared Cabin Flag
    cabin_counts = full_df["Cabin"].dropna().value_counts()
    full_df["Group_Shared_Cabin_Flag"] = (
        full_df["Cabin"].map(cabin_counts).fillna(0) > 1
    ).astype(int)

    full_df["CabinDeck"] = full_df["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) else np.nan
    )
    full_df["CabinNum"] = full_df["Cabin"].apply(
        lambda x: float(x.split("/")[1]) if pd.notna(x) else np.nan
    )
    full_df["CabinSide"] = full_df["Cabin"].apply(
        lambda x: x.split("/")[2] if pd.notna(x) else np.nan
    )

    # 3. Family Name Extraction
    full_df["FamilyName"] = full_df["Name"].apply(
        lambda x: x.split()[-1] if pd.notna(x) else np.nan
    )

    # 4. Expenditure Amenities & Physical Constraints
    amenity_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    # Constraint rule 1: If CryoSleep is True, luxury spending is strictly 0
    cryo_true_mask = full_df["CryoSleep"] == True
    for col in amenity_cols:
        full_df.loc[cryo_true_mask & full_df[col].isna(), col] = 0.0

    # Constraint rule 2: Children under 13 cannot spend on amenities
    child_mask = full_df["Age"] <= 12
    for col in amenity_cols:
        full_df.loc[child_mask & full_df[col].isna(), col] = 0.0

    # Impute remaining missing amenity values with 0
    for col in amenity_cols:
        full_df[col] = full_df[col].fillna(0.0)

    # Expenditure Aggregations
    full_df["TotalSpent"] = full_df[amenity_cols].sum(axis=1)
    full_df["HasSpentTotal"] = (full_df["TotalSpent"] > 0).astype(int)

    # Constraint rule 3: If TotalSpent > 0, CryoSleep must be False
    full_df.loc[
        full_df["CryoSleep"].isna() & (full_df["TotalSpent"] > 0), "CryoSleep"
    ] = False

    # Impute remaining CryoSleep with mode (False)
    full_df["CryoSleep"] = full_df["CryoSleep"].infer_objects(copy=False).fillna(False).astype(int)
    full_df["Group_Cryo_Ratio"] = full_df.groupby("PassengerGroup")[
        "CryoSleep"
    ].transform("mean")

    # Constraint rule 4: Propagate HomePlanet within the same PassengerGroup
    group_home = (
        full_df.groupby("PassengerGroup")["HomePlanet"].first().dropna().to_dict()
    )
    full_df["HomePlanet"] = full_df["HomePlanet"].fillna(
        full_df["PassengerGroup"].map(group_home)
    )

    # Constraint rule 5: Propagate HomePlanet from FamilyName
    family_home = full_df.groupby("FamilyName")["HomePlanet"].first().dropna().to_dict()
    full_df["HomePlanet"] = full_df["HomePlanet"].fillna(
        full_df["FamilyName"].map(family_home)
    )

    home_planet_mode = full_df["HomePlanet"].mode()[0]
    full_df["HomePlanet"] = full_df["HomePlanet"].fillna(home_planet_mode)

    # Propagate Cabin Deck and Side within PassengerGroup
    group_deck = (
        full_df.groupby("PassengerGroup")["CabinDeck"].first().dropna().to_dict()
    )
    full_df["CabinDeck"] = (
        full_df["CabinDeck"]
        .fillna(full_df["PassengerGroup"].map(group_deck))
        .fillna("Missing")
    )

    group_side = (
        full_df.groupby("PassengerGroup")["CabinSide"].first().dropna().to_dict()
    )
    full_df["CabinSide"] = (
        full_df["CabinSide"]
        .fillna(full_df["PassengerGroup"].map(group_side))
        .fillna("Missing")
    )

    full_df["DeckSide"] = full_df["CabinDeck"] + "_" + full_df["CabinSide"]

    # Impute CabinNum with median per deck
    deck_num_medians = full_df.groupby("CabinDeck")["CabinNum"].transform("median")
    full_df["CabinNum"] = (
        full_df["CabinNum"]
        .fillna(deck_num_medians)
        .fillna(full_df["CabinNum"].median())
    )

    deck_num_max = full_df.groupby("CabinDeck")["CabinNum"].transform("max") + 1e-5
    full_df["CabinDeckNormalizedPos"] = full_df["CabinNum"] / deck_num_max
    full_df["CabinCorridorBin"] = (full_df["CabinNum"] // 300).astype(int)

    # 5. Advanced Consumption Features
    for col in amenity_cols:
        full_df[f"Log_{col}"] = np.log1p(full_df[col])
        full_df[f"{col}_Ratio"] = full_df[col] / (full_df["TotalSpent"] + 1.0)
        full_df[f"HasSpent_{col}"] = (full_df[col] > 0).astype(int)

    full_df["Log_TotalSpent"] = np.log1p(full_df["TotalSpent"])
    full_df["NumAmenitiesUsed"] = full_df[[f"HasSpent_{c}" for c in amenity_cols]].sum(
        axis=1
    )
    full_df["Zero_Amenity_Count"] = (
        (full_df[amenity_cols] == 0.0).sum(axis=1).astype(int)
    )

    full_df["LuxurySpent"] = full_df["RoomService"] + full_df["Spa"] + full_df["VRDeck"]
    full_df["SustenanceSpent"] = full_df["FoodCourt"] + full_df["ShoppingMall"]
    full_df["Log_LuxurySpent"] = np.log1p(full_df["LuxurySpent"])
    full_df["Log_SustenanceSpent"] = np.log1p(full_df["SustenanceSpent"])
    full_df["LuxuryRatio"] = full_df["LuxurySpent"] / (full_df["TotalSpent"] + 1.0)

    # Group Spending Aggregations
    full_df["GroupTotalSpent"] = full_df.groupby("PassengerGroup")[
        "TotalSpent"
    ].transform("sum")
    full_df["GroupMeanSpent"] = full_df.groupby("PassengerGroup")[
        "TotalSpent"
    ].transform("mean")
    full_df["SpentRatioToGroup"] = full_df["TotalSpent"] / (
        full_df["GroupMeanSpent"] + 1.0
    )

    # 6. Demographics & VIP
    age_median = full_df.groupby(["HomePlanet", "CryoSleep"])["Age"].transform("median")
    full_df["Age"] = full_df["Age"].fillna(age_median).fillna(full_df["Age"].median())

    full_df["IsChild"] = (full_df["Age"] <= 12).astype(int)
    full_df["IsYoungAdult"] = ((full_df["Age"] > 12) & (full_df["Age"] <= 25)).astype(
        int
    )
    full_df["IsSenior"] = (full_df["Age"] >= 60).astype(int)

    age_bins = [-1, 12, 18, 25, 35, 50, 65, 120]
    full_df["AgeGroup"] = pd.cut(full_df["Age"], bins=age_bins, labels=False).astype(
        int
    )

    full_df.loc[
        full_df["VIP"].isna()
        & ((full_df["HomePlanet"] == "Earth") | (full_df["Age"] < 18)),
        "VIP",
    ] = False
    full_df["VIP"] = full_df["VIP"].infer_objects(copy=False).fillna(False).astype(int)

    dest_mode = full_df["Destination"].mode()[0]
    full_df["Destination"] = full_df["Destination"].fillna(dest_mode)

    full_df["Planet_Dest"] = full_df["HomePlanet"] + "_" + full_df["Destination"]
    full_df["Planet_Deck"] = full_df["HomePlanet"] + "_" + full_df["CabinDeck"]

    # 7. Frequency Encodings
    for freq_col in [
        "PassengerGroup",
        "FamilyName",
        "CabinDeck",
        "CabinSide",
        "DeckSide",
        "Planet_Dest",
    ]:
        counts = full_df[freq_col].value_counts().to_dict()
        full_df[f"{freq_col}_Freq"] = (
            full_df[freq_col].map(counts).fillna(1).astype(int)
        )

    # 8. Categorical Label Encoding
    cat_columns = [
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "DeckSide",
        "Planet_Dest",
        "Planet_Deck",
    ]
    for col in cat_columns:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))

    # Separate Train and Test
    processed_train = full_df.iloc[:n_train].copy().reset_index(drop=True)
    processed_test = full_df.iloc[n_train:].copy().reset_index(drop=True)

    processed_train[target_col] = y_train.values
    processed_test.drop(columns=[target_col], inplace=True)

    # 9. Stratified Group K-Fold Cross-Validation Splitting
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    processed_train["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(
            processed_train,
            processed_train[target_col],
            groups=processed_train["PassengerGroup"],
        )
    ):
        processed_train.loc[val_idx, "fold"] = fold_idx

    ignore_cols = [
        "PassengerId",
        "Name",
        "Cabin",
        "FamilyName",
        "PassengerGroup",
        target_col,
        "fold",
    ]
    feature_cols = [c for c in processed_train.columns if c not in ignore_cols]

    # Persist artifacts for auditing
    processed_train.to_parquet(
        os.path.join(output_dir, "train_processed.parquet"), index=False
    )
    processed_test.to_parquet(
        os.path.join(output_dir, "test_processed.parquet"), index=False
    )

    metadata = {
        "n_train": n_train,
        "n_test": len(processed_test),
        "n_features": len(feature_cols),
        "target_col": target_col,
        "fold_col": "fold",
        "feature_cols": feature_cols,
        "cat_columns": cat_columns,
    }
    with open(os.path.join(output_dir, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return processed_train, processed_test, feature_cols, cat_columns


# =========================================================================
# 2. Architectural Components: FT-SwiGLU Tabular Transformer
# =========================================================================
class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DomainGLUProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_token: int = 64,
        hidden_features: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc_proj = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, d_token),
        )
        self.fc_gate = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, d_token),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        proj = self.fc_proj(x_norm)
        gate = self.fc_gate(x_norm)
        out = self.out_norm(proj * gate)
        return self.dropout(out)


def build_domain_feature_map(
    cat_cols: List[str], num_cols: List[str]
) -> Dict[str, Dict[str, List[int]]]:
    domain_spec = {
        "expenditure": {
            "cat_names": [],
            "num_names": [
                "RoomService",
                "FoodCourt",
                "ShoppingMall",
                "Spa",
                "VRDeck",
                "TotalSpent",
                "HasSpentTotal",
                "Log_RoomService",
                "RoomService_Ratio",
                "HasSpent_RoomService",
                "Log_FoodCourt",
                "FoodCourt_Ratio",
                "HasSpent_FoodCourt",
                "Log_ShoppingMall",
                "ShoppingMall_Ratio",
                "HasSpent_ShoppingMall",
                "Log_Spa",
                "Spa_Ratio",
                "HasSpent_Spa",
                "Log_VRDeck",
                "VRDeck_Ratio",
                "HasSpent_VRDeck",
                "Log_TotalSpent",
                "NumAmenitiesUsed",
                "Zero_Amenity_Count",
                "LuxurySpent",
                "SustenanceSpent",
                "Log_LuxurySpent",
                "Log_SustenanceSpent",
                "LuxuryRatio",
                "GroupTotalSpent",
                "GroupMeanSpent",
                "SpentRatioToGroup",
            ],
        },
        "spatial": {
            "cat_names": ["CabinDeck", "CabinSide", "DeckSide"],
            "num_names": [
                "CabinNum",
                "CabinDeckNormalizedPos",
                "CabinCorridorBin",
                "Group_Shared_Cabin_Flag",
                "CabinDeck_Freq",
                "CabinSide_Freq",
                "DeckSide_Freq",
            ],
        },
        "group": {
            "cat_names": [],
            "num_names": [
                "PassengerNum",
                "GroupSize",
                "IsSolo",
                "PassengerGroup_Freq",
                "FamilyName_Freq",
                "Group_Cryo_Ratio",
            ],
        },
        "demographics": {
            "cat_names": ["HomePlanet", "Destination", "Planet_Dest", "Planet_Deck"],
            "num_names": [
                "Age",
                "AgeGroup",
                "IsChild",
                "IsYoungAdult",
                "IsSenior",
                "VIP",
                "Planet_Dest_Freq",
            ],
        },
        "physical_missing": {
            "cat_names": [],
            "num_names": [
                "CryoSleep",
                "CryoSleep_is_missing",
                "Cabin_is_missing",
                "Age_is_missing",
                "VIP_is_missing",
            ],
        },
    }

    cat_to_idx = {name: i for i, name in enumerate(cat_cols)}
    num_to_idx = {name: i for i, name in enumerate(num_cols)}

    mapped_cat_indices = set()
    mapped_num_indices = set()
    domain_map = {}

    for d_name, d_cfg in domain_spec.items():
        cat_indices = [cat_to_idx[c] for c in d_cfg["cat_names"] if c in cat_to_idx]
        num_indices = [num_to_idx[n] for n in d_cfg["num_names"] if n in num_to_idx]
        mapped_cat_indices.update(cat_indices)
        mapped_num_indices.update(num_indices)
        domain_map[d_name] = {"cat_idx": cat_indices, "num_idx": num_indices}

    unmapped_cat = [i for i in range(len(cat_cols)) if i not in mapped_cat_indices]
    unmapped_num = [i for i in range(len(num_cols)) if i not in mapped_num_indices]
    if unmapped_cat:
        domain_map["demographics"]["cat_idx"].extend(unmapped_cat)
    if unmapped_num:
        domain_map["physical_missing"]["num_idx"].extend(unmapped_num)

    return domain_map


class HierarchicalDomainTokenizer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_features: int,
        domain_feature_map: Optional[Dict[str, Dict[str, List[int]]]] = None,
        d_token: int = 64,
        cat_emb_dim: int = 16,
        dropout: float = 0.1,
        mask_prob: float = 0.15,
    ):
        super().__init__()
        self.d_token = d_token
        self.mask_prob = mask_prob

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cardinality + 1, embedding_dim=cat_emb_dim)
                for cardinality in cat_cardinalities
            ]
        )

        if domain_feature_map is None:
            domain_feature_map = {
                "expenditure": {"cat_idx": [], "num_idx": list(range(0, num_features // 5))},
                "spatial": {
                    "cat_idx": list(range(0, min(3, len(cat_cardinalities)))),
                    "num_idx": list(range(num_features // 5, 2 * num_features // 5)),
                },
                "group": {"cat_idx": [], "num_idx": list(range(2 * num_features // 5, 3 * num_features // 5))},
                "demographics": {
                    "cat_idx": list(range(min(3, len(cat_cardinalities)), len(cat_cardinalities))),
                    "num_idx": list(range(3 * num_features // 5, 4 * num_features // 5)),
                },
                "physical_missing": {"cat_idx": [], "num_idx": list(range(4 * num_features // 5, num_features))},
            }

        self.domain_names = [
            "expenditure",
            "spatial",
            "group",
            "demographics",
            "physical_missing",
        ]
        self.domain_cat_indices = {}
        self.domain_projections = nn.ModuleDict()

        for d_name in self.domain_names:
            cat_idx = domain_feature_map[d_name]["cat_idx"]
            num_idx = domain_feature_map[d_name]["num_idx"]
            self.domain_cat_indices[d_name] = cat_idx

            self.register_buffer(
                f"{d_name}_num_idx",
                torch.tensor(num_idx, dtype=torch.long),
                persistent=False,
            )

            in_features = len(cat_idx) * cat_emb_dim + len(num_idx)
            self.domain_projections[d_name] = DomainGLUProjection(
                in_features=in_features,
                d_token=d_token,
                hidden_features=d_token * 2,
                dropout=dropout,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.mask_token, std=0.02)

        self.slot_embeddings = nn.Parameter(torch.zeros(1, 6, d_token))
        nn.init.normal_(self.slot_embeddings, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x_cat: Optional[torch.Tensor], x_num: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = (
            x_cat.size(0)
            if x_cat is not None
            else (x_num.size(0) if x_num is not None else 1)
        )

        cat_embs = {}
        if x_cat is not None and len(self.cat_embeddings) > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                c_idx = torch.clamp(x_cat[:, i], 0, emb_layer.num_embeddings - 1).long()
                cat_embs[i] = emb_layer(c_idx)

        domain_tokens_list = []
        for d_name in self.domain_names:
            cat_indices = self.domain_cat_indices[d_name]
            num_idx_buf = getattr(self, f"{d_name}_num_idx")
            proj_layer = self.domain_projections[d_name]

            parts = []
            if len(cat_indices) > 0 and x_cat is not None:
                d_cat = torch.cat([cat_embs[idx] for idx in cat_indices], dim=-1)
                parts.append(d_cat)

            if len(num_idx_buf) > 0 and x_num is not None:
                d_num = x_num.index_select(1, num_idx_buf)
                parts.append(d_num)

            domain_in = torch.cat(parts, dim=-1)
            domain_token = proj_layer(domain_in)
            domain_tokens_list.append(domain_token)

        domain_tokens = torch.stack(domain_tokens_list, dim=1)

        if self.training and self.mask_prob > 0.0:
            mask = (
                torch.rand(batch_size, 5, 1, device=domain_tokens.device)
                < self.mask_prob
            )
            domain_tokens = torch.where(
                mask, self.mask_token.expand_as(domain_tokens), domain_tokens
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, domain_tokens], dim=1)
        seq = seq + self.slot_embeddings
        return self.dropout(seq)


class TransformerSwiGLUBlock(nn.Module):
    def __init__(
        self,
        d_token: int = 64,
        n_heads: int = 4,
        ffn_hidden: int = 128,
        dropout: float = 0.1,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = SwiGLU(
            in_features=d_token,
            hidden_features=ffn_hidden,
            out_features=d_token,
            dropout=dropout,
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + self.drop_path1(self.dropout1(attn_out))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class FTSwiGLUTabularTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_features: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_multiplier: float = 2.0,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        domain_feature_map: Optional[Dict[str, Dict[str, List[int]]]] = None,
        mask_prob: float = 0.15,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_cats = len(cat_cardinalities)
        self.d_token = d_token

        self.tokenizer = HierarchicalDomainTokenizer(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            domain_feature_map=domain_feature_map,
            d_token=d_token,
            cat_emb_dim=16,
            dropout=dropout,
            mask_prob=mask_prob,
        )

        ffn_hidden = int(d_token * ffn_multiplier)
        self.layers = nn.ModuleList(
            [
                TransformerSwiGLUBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_token)

        if num_features > 0:
            self.highway_proj = nn.Sequential(
                nn.Linear(num_features, d_token),
                nn.SiLU(),
            )
            self.highway_gate = nn.Linear(num_features, d_token)
        else:
            self.highway_proj = None
            self.highway_gate = None

        head_in_features = d_token * 3 if num_features > 0 else d_token * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_in_features),
            nn.Linear(head_in_features, d_token),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_token, 1),
        )

    def forward(
        self, x_cat: Optional[torch.Tensor], x_num: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.tokenizer(x_cat, x_num)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        cls_rep = x[:, 0, :]
        mean_rep = x[:, 1:, :].mean(dim=1)

        if self.highway_proj is not None and x_num is not None:
            highway_rep = self.highway_proj(x_num) * torch.sigmoid(
                self.highway_gate(x_num)
            )
            combined = torch.cat([cls_rep, mean_rep, highway_rep], dim=-1)
        else:
            combined = torch.cat([cls_rep, mean_rep], dim=-1)

        logits = self.head(combined).squeeze(-1)
        return logits


def build_loss_function(
    label_smoothing: float = 0.01,
) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss()


def build_optimizer(
    model: nn.Module,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or "bias" in name
            or "norm" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    pct_start: float = 0.1,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.OneCycleLR:
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1.0 / min_lr_ratio,
    )


# =========================================================================
# 3. Tabular Dataset
# =========================================================================
class SpaceshipTitanicDataset(Dataset):
    def __init__(
        self,
        x_cat: np.ndarray,
        x_num: np.ndarray,
        y: Optional[np.ndarray] = None,
    ):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.x_cat)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.x_cat[idx], self.x_num[idx], self.y[idx]
        return self.x_cat[idx], self.x_num[idx]


# =========================================================================
# 4. End-to-End Cross-Validation Training & Inference
# =========================================================================
def main():
    # 1. Feature Engineering
    df_train, df_test, feat_cols, cat_cols = engineer_features()
    num_cols = [c for c in feat_cols if c not in cat_cols]
    target_col = "Transported"
    fold_col = "fold"

    cat_cardinalities = [
        int(max(df_train[col].max(), df_test[col].max()) + 1) for col in cat_cols
    ]
    domain_feature_map = build_domain_feature_map(cat_cols, num_cols)

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 25
    PATIENCE = 6
    LEARNING_RATE = 8e-4
    WEIGHT_DECAY = 1e-4
    N_FOLDS = 5

    oof_predictions = np.zeros(len(df_train), dtype=np.float32)
    test_fold_predictions = np.zeros((N_FOLDS, len(df_test)), dtype=np.float32)
    os.makedirs("./working/checkpoints", exist_ok=True)

    # 2. 5-Fold Stratified Group Cross-Validation
    for fold in range(N_FOLDS):
        train_mask = df_train[fold_col] != fold
        val_mask = df_train[fold_col] == fold

        df_fold_train = df_train[train_mask].reset_index(drop=True)
        df_fold_val = df_train[val_mask].reset_index(drop=True)
        val_indices = df_train.index[val_mask].to_numpy()

        # Strict fold-isolated feature scaling
        scaler = StandardScaler()
        x_train_num = scaler.fit_transform(df_fold_train[num_cols].values)
        x_val_num = scaler.transform(df_fold_val[num_cols].values)
        x_test_num = scaler.transform(df_test[num_cols].values)

        x_train_num = np.nan_to_num(x_train_num, nan=0.0)
        x_val_num = np.nan_to_num(x_val_num, nan=0.0)
        x_test_num = np.nan_to_num(x_test_num, nan=0.0)

        x_train_cat = df_fold_train[cat_cols].values
        x_val_cat = df_fold_val[cat_cols].values
        x_test_cat = df_test[cat_cols].values

        y_train = df_fold_train[target_col].astype(float).values
        y_val = df_fold_val[target_col].astype(float).values

        train_dataset = SpaceshipTitanicDataset(x_train_cat, x_train_num, y_train)
        val_dataset = SpaceshipTitanicDataset(x_val_cat, x_val_num, y_val)
        test_dataset = SpaceshipTitanicDataset(x_test_cat, x_test_num)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        model = FTSwiGLUTabularTransformer(
            cat_cardinalities=cat_cardinalities,
            num_features=len(num_cols),
            d_token=64,
            n_layers=3,
            n_heads=4,
            ffn_multiplier=2.0,
            dropout=0.1,
            head_dropout=0.2,
            domain_feature_map=domain_feature_map,
            mask_prob=0.15,
            drop_path_rate=0.10,
        ).to(device)

        criterion = build_loss_function(label_smoothing=0.01)
        optimizer = build_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * EPOCHS
        scheduler = build_scheduler(optimizer, total_steps=total_steps, pct_start=0.1)

        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = f"./working/checkpoints/ft_swiglu_fold_{fold}.pt"

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for batch_cat, batch_num, batch_y in train_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_cat, batch_num)
                loss = criterion(logits, batch_y)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch_cat, batch_num, batch_y in val_loader:
                    batch_cat = batch_cat.to(device)
                    batch_num = batch_num.to(device)
                    batch_y = batch_y.to(device)

                    logits = model(batch_cat, batch_num)
                    loss = criterion(logits, batch_y)
                    probs = torch.sigmoid(logits)

                    val_losses.append(loss.item())
                    val_preds_list.append(probs.cpu().numpy())
                    val_targets_list.append(batch_y.cpu().numpy())

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_val_preds = np.concatenate(val_preds_list)
            epoch_val_targets = np.concatenate(val_targets_list)
            epoch_val_acc = accuracy_score(
                epoch_val_targets, (epoch_val_preds >= 0.5).astype(int)
            )

            print(
                f"Fold {fold} | Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
            )

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # OOF pass
        val_probs = []
        with torch.no_grad():
            for batch_cat, batch_num, _ in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                logits = model(batch_cat, batch_num)
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
        oof_predictions[val_indices] = np.concatenate(val_probs)

        # Test pass
        test_probs = []
        with torch.no_grad():
            for batch_cat, batch_num in test_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                logits = model(batch_cat, batch_num)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.cpu().numpy())
        test_fold_predictions[fold] = np.concatenate(test_probs)

    # 3. Decision Threshold Optimization
    ground_truth = df_train[target_col].astype(int).values
    best_threshold = 0.5
    best_score = accuracy_score(
        ground_truth, (oof_predictions >= best_threshold).astype(int)
    )

    candidate_thresholds = np.linspace(0.42, 0.58, 33)
    for t in candidate_thresholds:
        score_t = accuracy_score(ground_truth, (oof_predictions >= t).astype(int))
        if score_t > best_score:
            best_score = score_t
            best_threshold = t

    # 4. Generate Official Submission
    ensemble_test_probs = np.mean(test_fold_predictions, axis=0)
    final_test_preds = (ensemble_test_probs >= best_threshold).astype(bool)

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame(
        {"PassengerId": df_test["PassengerId"], "Transported": final_test_preds}
    )
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"
    assert len(submission_df) == 4277, f"Expected 4277 rows, found {len(submission_df)}"
    assert list(submission_df.columns) == [
        "PassengerId",
        "Transported",
    ], f"Unexpected columns {submission_df.columns}"

    final_score = round(float(best_score), 4)
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()

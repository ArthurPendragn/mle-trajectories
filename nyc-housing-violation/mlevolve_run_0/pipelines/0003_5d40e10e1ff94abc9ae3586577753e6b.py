import copy
import json
import math
import os
import sys
import gcsfs
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# 1. Environment and Configuration Setup
# -------------------------------------------------------------------------
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
storage_options = {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else {}
fs = (
    gcsfs.GCSFileSystem(token=TOKEN_PATH)
    if os.path.exists(TOKEN_PATH)
    else gcsfs.GCSFileSystem()
)

LAKE_FULL = "mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full"
TEST_ENTITIES_PATH = (
    "gs://mle-nyc-lake/tasks/housing_violation_risk/v1/test_entities.parquet"
)

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 2. Key Standardization and Lake Ingestion Utilities
# -------------------------------------------------------------------------
def clean_bbl_series(s):
    """Converts a BBL series into clean 10-digit zero-padded string representation."""
    s_clean = (
        pd.to_numeric(s, errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.zfill(10)
    )
    return s_clean


def extract_violations_bbl(df_viol):
    """Implements exact official task definition for violation BBL:

    Row's BBL is 'bbl' when 10 digits; otherwise boroid (1 digit) +
    block (5 digits, zero-padded) + lot (4 digits, zero-padded).
    """
    if "bbl" in df_viol.columns:
        raw_bbl = (
            pd.to_numeric(df_viol["bbl"], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
        )
        is_10_digit = (raw_bbl.str.len() == 10) & (raw_bbl != "0000000000")
    else:
        is_10_digit = pd.Series(False, index=df_viol.index)
        raw_bbl = pd.Series("", index=df_viol.index)

    boro = (
        pd.to_numeric(df_viol["boroid"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
    )
    block = (
        pd.to_numeric(df_viol["block"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )
    lot = (
        pd.to_numeric(df_viol["lot"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
        .str.zfill(4)
    )
    fallback = boro + block + lot
    final_bbl = raw_bbl.where(is_10_digit, fallback)

    return final_bbl.str.zfill(10)


def list_lake_tables():
    try:
        return fs.ls(LAKE_FULL)
    except Exception as e:
        return []


LAKE_TABLES = list_lake_tables()


def load_pluto_cohort(release_name):
    """Finds and loads the PLUTO release (19v2, 20v7, 21v4, 22v3)."""
    candidates = []
    for tbl in LAKE_TABLES:
        tbl_name = tbl.split("/")[-1].lower()
        if "pluto" in tbl_name:
            candidates.append(f"{tbl}/release={release_name}")
            candidates.append(f"{tbl}/{release_name}")
            if release_name.lower() in tbl_name:
                candidates.append(tbl)

    for p in candidates:
        try:
            if fs.exists(p):
                df = pd.read_parquet(f"gs://{p}", storage_options=storage_options)
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            continue

    for tbl in LAKE_TABLES:
        if "pluto" in tbl.lower():
            try:
                df = pd.read_parquet(
                    f"gs://{tbl}",
                    filters=[("release", "==", release_name)],
                    storage_options=storage_options,
                )
                if len(df) > 0:
                    df.columns = [c.lower() for c in df.columns]
                    return df
            except Exception:
                pass

    direct_paths = [
        f"{LAKE_FULL}/pluto/release={release_name}",
        f"{LAKE_FULL}/dcp_pluto/release={release_name}",
        f"{LAKE_FULL}/pluto_{release_name}",
    ]
    for dp in direct_paths:
        try:
            if fs.exists(dp):
                df = pd.read_parquet(f"gs://{dp}", storage_options=storage_options)
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            pass

    raise FileNotFoundError(f"Unable to locate PLUTO release {release_name} in lake.")


def load_hpd_violations(start_year=2017):
    """Efficiently loads HPD violations from lake."""
    viol_tbl = None
    for tbl in LAKE_TABLES:
        tbl_name = tbl.split("/")[-1].lower()
        if "hpd_violations" in tbl_name or (
            "hpd" in tbl_name and "violation" in tbl_name
        ):
            viol_tbl = tbl
            break
    if viol_tbl is None:
        viol_tbl = f"{LAKE_FULL}/hpd_violations"

    needed_cols = [
        "bbl",
        "boroid",
        "block",
        "lot",
        "class",
        "inspectiondate",
        "currentstatus",
    ]

    try:
        subdirs = fs.ls(viol_tbl)
    except Exception:
        subdirs = []

    year_partitions = [
        s
        for s in subdirs
        if "year=" in s and any(str(yr) in s for yr in range(start_year, 2024))
    ]

    if year_partitions:
        dfs = []
        for p in sorted(year_partitions):
            try:
                part = pd.read_parquet(
                    f"gs://{p}",
                    storage_options=storage_options,
                    columns=[c for c in needed_cols if c != "year"],
                )
                part.columns = [c.lower() for c in part.columns]
                dfs.append(part)
            except Exception:
                try:
                    part = pd.read_parquet(f"gs://{p}", storage_options=storage_options)
                    part.columns = [c.lower() for c in part.columns]
                    dfs.append(part)
                except Exception:
                    pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    try:
        df_v = pd.read_parquet(
            f"gs://{viol_tbl}",
            storage_options=storage_options,
            columns=needed_cols,
        )
    except Exception:
        df_v = pd.read_parquet(f"gs://{viol_tbl}", storage_options=storage_options)
    df_v.columns = [c.lower() for c in df_v.columns]
    return df_v


# -------------------------------------------------------------------------
# 3. Leak-Free Feature Engineering
# -------------------------------------------------------------------------
def extract_features(entities_df, violations_df, cutoff_timestamp):
    """Computes point-in-time structural, temporal, velocity, and severity

    features for a given entity cohort strictly using data prior to
    cutoff_timestamp.
    """
    T = pd.Timestamp(cutoff_timestamp)

    df = entities_df.copy()
    if "bbl" not in df.columns:
        df["bbl"] = clean_bbl_series(df.index)
    else:
        df["bbl"] = clean_bbl_series(df["bbl"])

    pluto_num_defaults = {
        "unitsres": 3.0,
        "unitstotal": 3.0,
        "numfloors": 3.0,
        "bldgarea": 3000.0,
        "lotarea": 2500.0,
        "builtfar": 1.5,
        "residfar": 1.5,
        "yearbuilt": 1940.0,
        "assessland": 50000.0,
        "assesstot": 150000.0,
    }

    for col, default_val in pluto_num_defaults.items():
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(default_val)
                .clip(lower=0)
            )
        else:
            df[col] = default_val

    # Building domain interactions
    df["feat_building_age"] = (T.year - df["yearbuilt"]).clip(0, 200)
    df["feat_is_prewar"] = (df["yearbuilt"] < 1940).astype(np.float32)
    df["feat_units_per_floor"] = df["unitsres"] / np.maximum(df["numfloors"], 1.0)
    df["feat_area_per_unit"] = df["bldgarea"] / np.maximum(df["unitsres"], 1.0)
    df["feat_assessed_per_unit"] = df["assesstot"] / np.maximum(df["unitsres"], 1.0)
    df["feat_land_value_ratio"] = df["assessland"] / np.maximum(df["assesstot"], 1.0)
    df["feat_commercial_unit_ratio"] = np.maximum(
        df["unitstotal"] - df["unitsres"], 0.0
    ) / np.maximum(df["unitstotal"], 1.0)

    # Categorical encodings
    if "borough" in df.columns:
        df["feat_borough"] = (
            pd.to_numeric(df["borough"], errors="coerce").fillna(0).astype(np.float32)
        )
    elif "borocode" in df.columns:
        df["feat_borough"] = (
            pd.to_numeric(df["borocode"], errors="coerce").fillna(0).astype(np.float32)
        )
    else:
        df["feat_borough"] = (
            pd.to_numeric(df["bbl"].str[0], errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )

    if "bldgclass" in df.columns:
        major_class = (
            df["bldgclass"].fillna("").astype(str).str[:1].astype("category")
        )
        df["feat_bldgclass_code"] = major_class.cat.codes.astype(np.float32)
    else:
        df["feat_bldgclass_code"] = 0.0

    if "cd" in df.columns:
        df["feat_cd"] = (
            pd.to_numeric(df["cd"], errors="coerce").fillna(0).astype(np.float32)
        )
    else:
        df["feat_cd"] = 0.0

    # Historical violation aggregations strictly prior to T
    v_hist = violations_df[violations_df["inspectiondate"] < T]
    t_3yr = T - pd.Timedelta(days=1095)
    v_3yr = v_hist[v_hist["inspectiondate"] >= t_3yr].copy()

    v_c = v_3yr[v_3yr["class"] == "C"]
    v_b = v_3yr[v_3yr["class"] == "B"]
    v_a = v_3yr[v_3yr["class"] == "A"]

    c_30 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=30))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_30d")
    )
    c_90 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_90d")
    )
    c_180 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=180))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_180d")
    )
    c_365 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_365d")
    )
    c_730 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=730))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_730d")
    )
    c_1095 = v_c.groupby("clean_bbl").size().rename("feat_c_count_1095d")

    b_90 = (
        v_b[v_b["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_b_count_90d")
    )
    b_365 = (
        v_b[v_b["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_b_count_365d")
    )
    b_1095 = v_b.groupby("clean_bbl").size().rename("feat_b_count_1095d")

    a_365 = (
        v_a[v_a["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_a_count_365d")
    )
    a_1095 = v_a.groupby("clean_bbl").size().rename("feat_a_count_1095d")

    tot_30 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=30))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_30d")
    )
    tot_90 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_90d")
    )
    tot_365 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_365d")
    )
    tot_730 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=730))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_730d")
    )
    tot_1095 = v_3yr.groupby("clean_bbl").size().rename("feat_tot_viol_1095d")

    last_c_date = v_c.groupby("clean_bbl")["inspectiondate"].max()
    days_since_c = ((T - last_c_date).dt.total_seconds() / 86400.0).rename(
        "feat_days_since_last_c"
    )

    last_tot_date = v_3yr.groupby("clean_bbl")["inspectiondate"].max()
    days_since_any = ((T - last_tot_date).dt.total_seconds() / 86400.0).rename(
        "feat_days_since_last_any"
    )

    if "currentstatus" in v_3yr.columns:
        is_open = (
            v_3yr["currentstatus"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains("OPEN")
        )
        open_c = (
            v_3yr[is_open & (v_3yr["class"] == "C")]
            .groupby("clean_bbl")
            .size()
            .rename("feat_open_c_count")
        )
        open_tot = v_3yr[is_open].groupby("clean_bbl").size().rename("feat_open_total")
    else:
        open_c = pd.Series(0, index=[], name="feat_open_c_count")
        open_tot = pd.Series(0, index=[], name="feat_open_total")

    df = df.set_index("bbl")
    for s in [
        c_30,
        c_90,
        c_180,
        c_365,
        c_730,
        c_1095,
        b_90,
        b_365,
        b_1095,
        a_365,
        a_1095,
        tot_30,
        tot_90,
        tot_365,
        tot_730,
        tot_1095,
        open_c,
        open_tot,
    ]:
        df = df.join(s, how="left")
        df[s.name] = df[s.name].fillna(0.0).astype(np.float32)

    df = df.join(days_since_c, how="left")
    df["feat_days_since_last_c"] = (
        df["feat_days_since_last_c"].fillna(3650.0).astype(np.float32)
    )

    df = df.join(days_since_any, how="left")
    df["feat_days_since_last_any"] = (
        df["feat_days_since_last_any"].fillna(3650.0).astype(np.float32)
    )

    # Dynamic ratios and velocity interactions
    df["feat_has_prior_c_365d"] = (df["feat_c_count_365d"] > 0).astype(np.float32)
    df["feat_has_prior_c_1095d"] = (df["feat_c_count_1095d"] > 0).astype(np.float32)
    df["feat_has_prior_any_365d"] = (df["feat_tot_viol_365d"] > 0).astype(np.float32)

    df["feat_c_ratio_365d"] = df["feat_c_count_365d"] / (df["feat_tot_viol_365d"] + 1.0)
    df["feat_b_ratio_365d"] = df["feat_b_count_365d"] / (df["feat_tot_viol_365d"] + 1.0)

    df["feat_c_velocity_90_365"] = (df["feat_c_count_90d"] * 4.0) / (
        df["feat_c_count_365d"] + 1.0
    )
    df["feat_tot_velocity_90_365"] = (df["feat_tot_viol_90d"] * 4.0) / (
        df["feat_tot_viol_365d"] + 1.0
    )

    df["feat_severity_index_365d"] = (
        1.0 * df["feat_a_count_365d"]
        + 3.0 * df["feat_b_count_365d"]
        + 6.0 * df["feat_c_count_365d"]
    )
    df["feat_severity_index_90d"] = (
        3.0 * df["feat_b_count_90d"] + 6.0 * df["feat_c_count_90d"]
    )

    df["feat_viol_per_unit_365d"] = df["feat_tot_viol_365d"] / np.maximum(
        df["unitsres"], 1.0
    )
    df["feat_c_viol_per_unit_365d"] = df["feat_c_count_365d"] / np.maximum(
        df["unitsres"], 1.0
    )

    df = df.reset_index()

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    for col in feature_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    return df[["bbl"] + feature_cols]


# -------------------------------------------------------------------------
# 4. Neural Architecture and Loss Definitions
# -------------------------------------------------------------------------
class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss with probability margin clipping on negatives."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-7,
        pos_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        probs = torch.sigmoid(logits)

        loss_pos = (
            -self.pos_weight
            * targets
            * torch.pow(1.0 - probs, self.gamma_pos)
            * torch.log(probs.clamp(min=self.eps))
        )

        p_neg = (probs - self.clip).clamp(min=0.0)
        loss_neg = (
            -(1.0 - targets)
            * torch.pow(p_neg, self.gamma_neg)
            * torch.log((1.0 - p_neg).clamp(min=self.eps))
        )

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FeatureTokenizer(nn.Module):
    """Continuous feature projection module converting features into token embeddings."""

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.empty(num_features, embed_dim))
        self.bias = nn.Parameter(torch.empty(num_features, embed_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = 1
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return tokens


class TabularFeatureInteractionNet(nn.Module):
    """Deep Tabular Feature Interaction Architecture with Multi-Head Self-

    Attention and Direct Wide Bypass Residual Highway.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.input_norm = nn.LayerNorm(num_features)
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.wide_highway = nn.Sequential(
            nn.Linear(num_features, embed_dim), nn.SiLU(), nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_norm = self.input_norm(x)

        tokens = self.tokenizer(x_norm)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = self.transformer(tokens)
        cls_out = self.post_norm(tokens[:, 0, :])

        wide_out = self.wide_highway(x_norm)

        fused = torch.cat([cls_out, wide_out], dim=-1)
        logits = self.head(fused)
        return logits


def build_model(
    num_features: int,
    embed_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    return TabularFeatureInteractionNet(
        num_features=num_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_loss(
    gamma_neg: float = 2.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    pos_weight: float = 1.0,
) -> nn.Module:
    return AsymmetricFocalLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
        pos_weight=pos_weight,
        reduction="mean",
    )


def build_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "cls_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    pct_start: float = 0.1,
    min_lr_ratio: float = 1e-2,
):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1.0 / min_lr_ratio,
    )


# -------------------------------------------------------------------------
# 5. Cohort Generation and Feature Pipeline Execution
# -------------------------------------------------------------------------
print("Loading HPD violations dataset...")
df_violations = load_hpd_violations(start_year=2017)
df_violations["inspectiondate"] = pd.to_datetime(
    df_violations["inspectiondate"], errors="coerce"
)
df_violations = df_violations[df_violations["inspectiondate"].notna()].copy()
df_violations["clean_bbl"] = extract_violations_bbl(df_violations)
df_violations = df_violations[
    (df_violations["clean_bbl"].str.len() == 10)
    & (df_violations["clean_bbl"].str.isdigit())
].copy()

# Test Cohort (Cutoff: 2023-01-01)
df_test_raw = pd.read_parquet(TEST_ENTITIES_PATH, storage_options=storage_options)
df_test_raw.columns = [c.lower() for c in df_test_raw.columns]
df_test_raw["bbl"] = clean_bbl_series(df_test_raw["bbl"])

if len(df_test_raw.columns) <= 2:
    pluto_22v3 = load_pluto_cohort("22v3")
    pluto_22v3["bbl"] = clean_bbl_series(pluto_22v3["bbl"])
    pluto_22v3 = pluto_22v3[pluto_22v3["unitsres"] >= 3].drop_duplicates(subset=["bbl"])
    df_test_entities = df_test_raw[["bbl"]].merge(pluto_22v3, on="bbl", how="left")
else:
    df_test_entities = df_test_raw

test_features = extract_features(
    df_test_entities, df_violations, cutoff_timestamp="2023-01-01"
)

# Validation Cohort (Cutoff: 2022-01-01, Release: 21v4)
pluto_21v4 = load_pluto_cohort("21v4")
pluto_21v4["bbl"] = clean_bbl_series(pluto_21v4["bbl"])
df_val_entities = (
    pluto_21v4[pluto_21v4["unitsres"] >= 3]
    .drop_duplicates(subset=["bbl"])
    .reset_index(drop=True)
)

val_features = extract_features(
    df_val_entities, df_violations, cutoff_timestamp="2022-01-01"
)
val_label_mask = (
    (df_violations["class"] == "C")
    & (df_violations["inspectiondate"] >= "2022-01-01")
    & (df_violations["inspectiondate"] < "2023-01-01")
)
val_positive_bbls = set(df_violations.loc[val_label_mask, "clean_bbl"].unique())
val_features["target"] = val_features["bbl"].isin(val_positive_bbls).astype(np.int32)

# Training Cohort (Cutoff: 2021-01-01, Release: 20v7)
pluto_20v7 = load_pluto_cohort("20v7")
pluto_20v7["bbl"] = clean_bbl_series(pluto_20v7["bbl"])
df_train_entities = (
    pluto_20v7[pluto_20v7["unitsres"] >= 3]
    .drop_duplicates(subset=["bbl"])
    .reset_index(drop=True)
)

train_features = extract_features(
    df_train_entities, df_violations, cutoff_timestamp="2021-01-01"
)
train_label_mask = (
    (df_violations["class"] == "C")
    & (df_violations["inspectiondate"] >= "2021-01-01")
    & (df_violations["inspectiondate"] < "2022-01-01")
)
train_positive_bbls = set(df_violations.loc[train_label_mask, "clean_bbl"].unique())
train_features["target"] = (
    train_features["bbl"].isin(train_positive_bbls).astype(np.int32)
)

feature_columns = [
    c
    for c in train_features.columns
    if c.startswith("feat_") and c != "target" and c != "bbl"
]
num_features = len(feature_columns)

# -------------------------------------------------------------------------
# 6. Data Scaling and PyTorch DataLoaders
# -------------------------------------------------------------------------
scaler = StandardScaler()
X_train_raw = train_features[feature_columns].values.astype(np.float32)
X_val_raw = val_features[feature_columns].values.astype(np.float32)
X_test_raw = test_features[feature_columns].values.astype(np.float32)

X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

y_train = train_features["target"].values.astype(np.float32)
y_val = val_features["target"].values.astype(np.float32)

batch_size = 2048
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)

# -------------------------------------------------------------------------
# 7. Model Training and Validation Loop
# -------------------------------------------------------------------------
epochs = 12
patience = 4

model = build_model(
    num_features=num_features,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
).to(device)

criterion = build_loss(gamma_neg=2.0, gamma_pos=1.0, clip=0.05, pos_weight=1.0).to(
    device
)

total_steps = epochs * len(train_loader)
optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
scheduler = build_scheduler(
    optimizer, total_steps=total_steps, pct_start=0.1, min_lr_ratio=1e-2
)

best_val_ap = -1.0
best_model_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * len(yb)
        total_samples += len(yb)

    train_loss = running_loss / total_samples

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds_list.append(probs)

    val_preds = np.concatenate(val_preds_list)
    val_ap = average_precision_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val AP: {val_ap:.4f} - Val AUC: {val_auc:.4f}"
    )

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# -------------------------------------------------------------------------
# 8. Final Hold-Out Evaluation and Submission Generation
# -------------------------------------------------------------------------
model.load_state_dict(best_model_state)
model.eval()

final_val_preds_list = []
with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        final_val_preds_list.append(probs)

final_val_preds = np.concatenate(final_val_preds_list)
final_val_ap = average_precision_score(y_val, final_val_preds)

# Full test inference pass
test_preds_list = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds_list.append(probs)

test_preds = np.concatenate(test_preds_list)

submission_df = pd.DataFrame(
    {
        "bbl": test_features["bbl"].astype(str).str.zfill(10),
        "score": test_preds.astype(float),
    }
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file not found!"
assert len(submission_df) == len(
    test_features
), f"Expected {len(test_features)} rows, got {len(submission_df)}"
assert submission_df["bbl"].str.len().eq(10).all(), "All BBLs must be 10 digits"
assert submission_df["score"].notna().all(), "Scores must not contain NaN or nulls"

print(f"Final Validation Score: {final_val_ap}")

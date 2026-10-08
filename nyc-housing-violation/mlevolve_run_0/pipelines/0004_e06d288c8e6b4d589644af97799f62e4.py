import copy
import json
import os
import re
import gcsfs
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
storage_options = {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else None

GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
TEST_ENTITIES_PATH = f"{GCS_BASE}/test_entities.parquet"
LAKE_FULL = f"{GCS_BASE}/lake/full"


# ---------------------------------------------------------
# Helper Functions: BBL Cleaning & Table Loading
# ---------------------------------------------------------
def clean_bbl_series(df):
    """Vectorized, exact 10-digit NYC BBL string construction per competition specification."""
    if "bbl" in df.columns:
        s_num = pd.to_numeric(df["bbl"], errors="coerce").fillna(0).astype("int64")
        s = s_num.astype(str).str.zfill(10)
        valid = s.str.match(r"^[1-5]\d{9}$")
    else:
        valid = pd.Series(False, index=df.index)

    boro_col = next((c for c in ["boroid", "boro", "borough"] if c in df.columns), None)
    block_col = next((c for c in ["block"] if c in df.columns), None)
    lot_col = next((c for c in ["lot"] if c in df.columns), None)

    if boro_col is not None and block_col is not None and lot_col is not None:
        b = (
            pd.to_numeric(df[boro_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
            .str.strip()
            .str.zfill(1)
        )
        blk = (
            pd.to_numeric(df[block_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
            .str.strip()
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df[lot_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
            .str.strip()
            .str.zfill(4)
        )
        fallback = b + blk + lt
    else:
        fallback = pd.Series("", index=df.index)

    if "bbl" in df.columns:
        return s.where(valid, fallback)
    return fallback


# ---------------------------------------------------------
# Load Entities (Test, Val, Train)
# ---------------------------------------------------------
df_test = pd.read_parquet(TEST_ENTITIES_PATH, storage_options=storage_options)
df_test.columns = [c.lower() for c in df_test.columns]
df_test["bbl"] = clean_bbl_series(df_test)
df_test = df_test.drop_duplicates(subset=["bbl"]).reset_index(drop=True)

pluto_cols = [
    "bbl",
    "boroid",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "bldgarea",
    "lotarea",
    "numfloors",
    "numbldgs",
    "yearbuilt",
    "yearalter1",
    "builtfar",
    "residfar",
    "assesstot",
    "assessland",
    "bldgclass",
    "cd",
    "council",
    "zipcode",
]


def load_pluto_release(release_tag, fallback_df):
    """Dynamically discover and load PLUTO release parquet table, or fall back safely."""
    fs = gcsfs.GCSFileSystem(token=TOKEN_PATH if os.path.exists(TOKEN_PATH) else None)
    discovered_paths = []
    try:
        entries = fs.ls("mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full")
        for entry in entries:
            name = entry.split("/")[-1].lower()
            if release_tag.lower() in name and "pluto" in name:
                discovered_paths.append(f"gs://{entry}")
            elif "pluto" in name:
                try:
                    for sub in fs.ls(entry):
                        if release_tag.lower() in sub.split("/")[-1].lower():
                            discovered_paths.append(f"gs://{sub}")
                except Exception:
                    pass
    except Exception:
        pass

    candidate_paths = discovered_paths + [
        f"{LAKE_FULL}/pluto_{release_tag}",
        f"{LAKE_FULL}/pluto_{release_tag.lower()}",
        f"{LAKE_FULL}/dcp_pluto_{release_tag.lower()}",
        f"{LAKE_FULL}/pluto/release={release_tag}",
        f"{LAKE_FULL}/pluto/version={release_tag}",
        f"{LAKE_FULL}/pluto_{release_tag.replace('v', '_')}",
    ]
    for p in candidate_paths:
        try:
            df = pd.read_parquet(p, storage_options=storage_options)
            df.columns = [c.lower() for c in df.columns]
            df["bbl"] = clean_bbl_series(df)
            if "unitsres" in df.columns:
                df["unitsres"] = (
                    pd.to_numeric(df["unitsres"], errors="coerce")
                    .fillna(0)
                    .astype(np.float32)
                )
                df = df[df["unitsres"] >= 3].copy()
            df = df.drop_duplicates(subset=["bbl"]).reset_index(drop=True)
            if len(df) > 50000:
                return df
        except Exception:
            continue
    return fallback_df.copy()


df_pluto_22v3 = load_pluto_release("22v3", pd.DataFrame())
if len(df_pluto_22v3) > 0:
    cols_to_add = [
        c
        for c in pluto_cols
        if c in df_pluto_22v3.columns and c not in df_test.columns
    ]
    if cols_to_add:
        df_test = df_test.merge(
            df_pluto_22v3[["bbl"] + cols_to_add], on="bbl", how="left"
        )

df_val_entities = load_pluto_release("21v4", df_test)
df_train_entities = load_pluto_release("20v7", df_test)

# ---------------------------------------------------------
# Load HPD Violations (2016-2022)
# ---------------------------------------------------------
hpd_viol_path = f"{LAKE_FULL}/hpd_violations"
desired_viol_cols = [
    "class",
    "inspectiondate",
    "bbl",
    "boroid",
    "block",
    "lot",
    "violationstatus",
    "currentstatus",
]

try:
    df_viol = pd.read_parquet(
        hpd_viol_path,
        columns=desired_viol_cols,
        filters=[("year", ">=", 2016)],
        storage_options=storage_options,
    )
except Exception:
    try:
        df_viol = pd.read_parquet(
            hpd_viol_path,
            columns=desired_viol_cols,
            storage_options=storage_options,
        )
    except Exception:
        df_viol = pd.read_parquet(hpd_viol_path, storage_options=storage_options)

df_viol.columns = [c.lower() for c in df_viol.columns]
df_viol["clean_bbl"] = clean_bbl_series(df_viol)
df_viol["class_clean"] = df_viol["class"].astype(str).str.strip().str.upper()
df_viol["inspectiondate_dt"] = pd.to_datetime(
    df_viol["inspectiondate"], errors="coerce"
)
df_viol = df_viol.dropna(subset=["inspectiondate_dt"]).copy()


# ---------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------
def extract_point_in_time_features(df_entities, df_violations, cutoff_str):
    """Strictly point-in-time feature extraction for entities at a specified cutoff date."""
    t_cutoff = pd.Timestamp(cutoff_str)
    cutoff_year = t_cutoff.year

    t_6m = t_cutoff - pd.DateOffset(months=6)
    t_1y = t_cutoff - pd.DateOffset(years=1)
    t_2y = t_cutoff - pd.DateOffset(years=2)
    t_3y = t_cutoff - pd.DateOffset(years=3)
    t_5y = t_cutoff - pd.DateOffset(years=5)

    mask_hist = df_violations["inspectiondate_dt"] < t_cutoff
    df_h = df_violations[mask_hist].copy()

    is_c = df_h["class_clean"] == "C"
    is_b = df_h["class_clean"] == "B"
    is_a = df_h["class_clean"] == "A"
    is_i = df_h["class_clean"] == "I"

    status_col = next(
        (c for c in ["violationstatus", "currentstatus"] if c in df_h.columns),
        None,
    )
    if status_col:
        is_open = (
            df_h[status_col]
            .astype(str)
            .str.upper()
            .str.contains("OPEN|ACTIVE", regex=True)
        )
    else:
        is_open = pd.Series(False, index=df_h.index)

    dates = df_h["inspectiondate_dt"]
    is_6m = dates >= t_6m
    is_1y = dates >= t_1y
    is_2y = dates >= t_2y
    is_3y = dates >= t_3y
    is_5y = dates >= t_5y

    df_h["c_6m"] = (is_c & is_6m).astype(np.int32)
    df_h["c_1y"] = (is_c & is_1y).astype(np.int32)
    df_h["c_2y"] = (is_c & is_2y).astype(np.int32)
    df_h["c_3y"] = (is_c & is_3y).astype(np.int32)
    df_h["c_5y"] = (is_c & is_5y).astype(np.int32)
    df_h["c_all"] = is_c.astype(np.int32)
    df_h["c_open"] = (is_c & is_open).astype(np.int32)

    df_h["b_1y"] = (is_b & is_1y).astype(np.int32)
    df_h["b_2y"] = (is_b & is_2y).astype(np.int32)
    df_h["b_all"] = is_b.astype(np.int32)

    df_h["a_1y"] = (is_a & is_1y).astype(np.int32)
    df_h["a_all"] = is_a.astype(np.int32)
    df_h["i_all"] = is_i.astype(np.int32)

    df_h["tot_6m"] = is_6m.astype(np.int32)
    df_h["tot_1y"] = is_1y.astype(np.int32)
    df_h["tot_2y"] = is_2y.astype(np.int32)
    df_h["tot_3y"] = is_3y.astype(np.int32)
    df_h["tot_all"] = 1

    df_h["last_c_date"] = df_h["inspectiondate_dt"].where(is_c)
    df_h["last_any_date"] = df_h["inspectiondate_dt"]

    agg_funcs = {
        "c_6m": "sum",
        "c_1y": "sum",
        "c_2y": "sum",
        "c_3y": "sum",
        "c_5y": "sum",
        "c_all": "sum",
        "c_open": "sum",
        "b_1y": "sum",
        "b_2y": "sum",
        "b_all": "sum",
        "a_1y": "sum",
        "a_all": "sum",
        "i_all": "sum",
        "tot_6m": "sum",
        "tot_1y": "sum",
        "tot_2y": "sum",
        "tot_3y": "sum",
        "tot_all": "sum",
        "last_c_date": "max",
        "last_any_date": "max",
    }
    bbl_agg = df_h.groupby("clean_bbl").agg(agg_funcs).reset_index()

    res = df_entities[["bbl"]].copy()
    for col in df_entities.columns:
        if col not in res.columns and col in pluto_cols:
            res[col] = df_entities[col]

    res = res.merge(bbl_agg, left_on="bbl", right_on="clean_bbl", how="left")
    if "clean_bbl" in res.columns:
        res = res.drop(columns=["clean_bbl"])

    count_cols = [
        "c_6m",
        "c_1y",
        "c_2y",
        "c_3y",
        "c_5y",
        "c_all",
        "c_open",
        "b_1y",
        "b_2y",
        "b_all",
        "a_1y",
        "a_all",
        "i_all",
        "tot_6m",
        "tot_1y",
        "tot_2y",
        "tot_3y",
        "tot_all",
    ]
    res[count_cols] = res[count_cols].fillna(0).astype(np.float32)

    days_c = (t_cutoff - res["last_c_date"]).dt.days
    res["days_since_last_c"] = days_c.fillna(3650.0).clip(lower=0).astype(np.float32)
    days_any = (t_cutoff - res["last_any_date"]).dt.days
    res["days_since_last_any"] = (
        days_any.fillna(3650.0).clip(lower=0).astype(np.float32)
    )

    res["recency_decay_c"] = np.exp(-res["days_since_last_c"] / 365.0).astype(
        np.float32
    )
    res["recency_decay_any"] = np.exp(-res["days_since_last_any"] / 365.0).astype(
        np.float32
    )
    res = res.drop(columns=["last_c_date", "last_any_date"])

    res["c_ratio_1y"] = (res["c_1y"] / (res["tot_1y"] + 1e-4)).astype(np.float32)
    res["c_ratio_all"] = (res["c_all"] / (res["tot_all"] + 1e-4)).astype(np.float32)
    c_prev_1y = np.maximum(0, res["c_2y"] - res["c_1y"])
    res["c_trend_1y"] = (res["c_1y"] - c_prev_1y).astype(np.float32)
    res["c_accel"] = (res["c_1y"] / (c_prev_1y + 0.5)).astype(np.float32)
    res["open_c_ratio"] = (res["c_open"] / (res["c_all"] + 1e-4)).astype(np.float32)

    if "unitsres" in res.columns:
        units = (
            pd.to_numeric(res["unitsres"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)
        )
    else:
        units = pd.Series(1.0, index=res.index, dtype=np.float32)

    if "bldgarea" in res.columns:
        bldgarea = (
            pd.to_numeric(res["bldgarea"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        bldgarea = pd.Series(0.0, index=res.index, dtype=np.float32)

    if "numfloors" in res.columns:
        numfloors = (
            pd.to_numeric(res["numfloors"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)
        )
    else:
        numfloors = pd.Series(1.0, index=res.index, dtype=np.float32)

    res["c_per_unit_1y"] = (res["c_1y"] / units).astype(np.float32)
    res["c_per_unit_2y"] = (res["c_2y"] / units).astype(np.float32)
    res["c_per_unit_all"] = (res["c_all"] / units).astype(np.float32)
    res["tot_per_unit_1y"] = (res["tot_1y"] / units).astype(np.float32)
    res["tot_per_unit_all"] = (res["tot_all"] / units).astype(np.float32)
    res["c_per_1k_sqft_1y"] = (res["c_1y"] / (bldgarea / 1000.0 + 1.0)).astype(
        np.float32
    )
    res["c_per_floor_1y"] = (res["c_1y"] / numfloors).astype(np.float32)

    if "yearbuilt" in res.columns:
        yearbuilt = (
            pd.to_numeric(res["yearbuilt"], errors="coerce")
            .fillna(1950.0)
            .values
        )
    else:
        yearbuilt = np.full(len(res), 1950.0, dtype=np.float32)
    yearbuilt = np.where(
        (yearbuilt < 1800) | (yearbuilt > cutoff_year), 1950.0, yearbuilt
    )
    res["building_age"] = np.maximum(0, cutoff_year - yearbuilt).astype(
        np.float32
    )
    res["is_prewar"] = (yearbuilt < 1940).astype(np.float32)
    res["log_unitsres"] = np.log1p(units).astype(np.float32)
    res["log_bldgarea"] = np.log1p(bldgarea).astype(np.float32)
    res["area_per_unit"] = (bldgarea / units).astype(np.float32)
    res["units_per_floor"] = (units / numfloors).astype(np.float32)

    if "yearalter1" in res.columns:
        alter = (
            pd.to_numeric(res["yearalter1"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        res["has_alteration"] = (alter > 0).astype(np.float32)
        res["years_since_alter"] = np.where(
            alter > 0, np.maximum(0, cutoff_year - alter), 99.0
        ).astype(np.float32)
    else:
        res["has_alteration"] = np.zeros(len(res), dtype=np.float32)
        res["years_since_alter"] = np.full(len(res), 99.0, dtype=np.float32)

    if "assesstot" in res.columns:
        assessed = (
            pd.to_numeric(res["assesstot"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
        res["log_assesstot"] = np.log1p(assessed).astype(np.float32)
        res["assess_per_unit"] = (assessed / units).astype(np.float32)
    else:
        res["log_assesstot"] = np.zeros(len(res), dtype=np.float32)
        res["assess_per_unit"] = np.zeros(len(res), dtype=np.float32)

    if "builtfar" in res.columns and "residfar" in res.columns:
        builtfar = (
            pd.to_numeric(res["builtfar"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
        residfar = (
            pd.to_numeric(res["residfar"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
        res["far_utilization"] = (builtfar / (residfar + 0.01)).astype(
            np.float32
        )
    else:
        res["far_utilization"] = np.zeros(len(res), dtype=np.float32)

    if "bldgclass" in res.columns:
        bldgclass = res["bldgclass"].fillna("UNK").astype(str).str.strip()
        res["bldgclass_major"] = bldgclass.str[:1]
    else:
        res["bldgclass_major"] = "U"

    boro = res["bbl"].str[:1]
    res["borocode"] = (
        pd.to_numeric(boro, errors="coerce").fillna(0).astype(np.int32)
    )

    return res


def compute_target(df_entities, df_violations, cutoff_str):
    """Compute ground truth label: >= 1 Class C violation in [T, T + 12 months)."""
    t_start = pd.Timestamp(cutoff_str)
    t_end = t_start + pd.DateOffset(months=12)

    mask = (
        (df_violations["class_clean"] == "C")
        & (df_violations["inspectiondate_dt"] >= t_start)
        & (df_violations["inspectiondate_dt"] < t_end)
    )
    pos_bbls = set(df_violations.loc[mask, "clean_bbl"].unique())
    return df_entities["bbl"].isin(pos_bbls).astype(np.int32)


# Extract cohorts
X_test_df = extract_point_in_time_features(df_test, df_viol, "2023-01-01")
X_val_df = extract_point_in_time_features(df_val_entities, df_viol, "2022-01-01")
y_val = compute_target(df_val_entities, df_viol, "2022-01-01")
X_val_df["target"] = y_val

X_train_df = extract_point_in_time_features(df_train_entities, df_viol, "2021-01-01")
y_train = compute_target(df_train_entities, df_viol, "2021-01-01")
X_train_df["target"] = y_train

# Strict train-only target encoding
global_target_mean = float(y_train.mean())
prior_weight = 50.0

for cat_col in ["bldgclass_major", "borocode"]:
    if cat_col in X_train_df.columns:
        stats = (
            X_train_df.groupby(cat_col)["target"].agg(["count", "mean"]).reset_index()
        )
        stats["te"] = (
            stats["count"] * stats["mean"] + prior_weight * global_target_mean
        ) / (stats["count"] + prior_weight)
        te_map = dict(zip(stats[cat_col], stats["te"]))

        col_name = f"{cat_col}_te"
        X_train_df[col_name] = (
            X_train_df[cat_col].map(te_map).fillna(global_target_mean)
        )
        X_val_df[col_name] = X_val_df[cat_col].map(te_map).fillna(global_target_mean)
        X_test_df[col_name] = X_test_df[cat_col].map(te_map).fillna(global_target_mean)

drop_cols = [
    col
    for col in X_train_df.columns
    if X_train_df[col].dtype == "object" and col != "bbl"
]
X_train_df = X_train_df.drop(columns=drop_cols)
X_val_df = X_val_df.drop(columns=drop_cols)
X_test_df = X_test_df.drop(columns=drop_cols)

feature_cols = [c for c in X_train_df.columns if c not in ["bbl", "target", "unitsres"]]


# ---------------------------------------------------------
# Tabular Residual Network & Focal Loss Architecture
# ---------------------------------------------------------
class TabularResidualBlock(nn.Module):
    """Residual block with LayerNorm, SiLU activations, and Dropout for tabular features."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """Wide & Deep Tabular Residual Network combining linear and non-linear risk pathways."""

    def __init__(
        self,
        in_features,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.25,
        feature_names=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.input_norm = nn.LayerNorm(in_features)

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.deep_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.wide_linear = nn.Linear(in_features, 1, bias=True)
        self._init_weights(feature_names)

    def _init_weights(self, feature_names):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.deep_head[-1].weight, std=0.01)
        nn.init.constant_(self.deep_head[-1].bias, 0.0)

        nn.init.zeros_(self.wide_linear.weight)
        nn.init.constant_(self.wide_linear.bias, -1.5)

        if feature_names is not None:
            prior_weights = {
                "c_per_unit_1y": 1.0,
                "c_1y": 0.5,
                "c_trend_1y": 0.25,
                "recency_decay_c": 0.2,
                "open_c_ratio": 0.1,
            }
            with torch.no_grad():
                for name, weight in prior_weights.items():
                    if name in feature_names:
                        idx = feature_names.index(name)
                        self.wide_linear.weight[0, idx] = weight

    def forward(self, x):
        normed_x = self.input_norm(x)
        h = self.input_proj(normed_x)
        for block in self.blocks:
            h = block(h)
        deep_out = self.deep_head(h)
        wide_out = self.wide_linear(normed_x)
        return (deep_out + wide_out).squeeze(-1)


class FocalLossWithLogits(nn.Module):
    """Focal Loss with binary logits for prioritizing tail ranking precision."""

    def __init__(self, alpha=0.35, gamma=1.5, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------
# Training Preparation & Evaluation Function
# ---------------------------------------------------------
common_features = [
    c
    for c in feature_cols
    if c in X_train_df.columns and c in X_val_df.columns and c in X_test_df.columns
]

X_train_np = (
    X_train_df[common_features]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .values.astype(np.float32)
)
y_train_np = X_train_df["target"].values.astype(np.float32)

X_val_np = (
    X_val_df[common_features]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .values.astype(np.float32)
)
y_val_np = X_val_df["target"].values.astype(np.float32)

X_test_np = (
    X_test_df[common_features]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .values.astype(np.float32)
)

# 1. Train Gradient Boosted Decision Tree (LightGBM)
lgb_model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=7,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_np,
    y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
)
val_lgb_probs = lgb_model.predict_proba(X_val_np)[:, 1].astype(float)
test_lgb_probs = lgb_model.predict_proba(X_test_np)[:, 1].astype(float)
lgb_val_ap = float(average_precision_score(y_val_np, val_lgb_probs))
print(f"LightGBM Val AP: {lgb_val_ap:.6f}")

# 2. Train TabularResNet with Normalized Features & Focal Loss
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_np).astype(np.float32)
X_val_norm = scaler.transform(X_val_np).astype(np.float32)
X_test_norm = scaler.transform(X_test_np).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabularResNet(
    in_features=len(common_features),
    hidden_dim=256,
    num_blocks=3,
    dropout=0.25,
    feature_names=common_features,
).to(device)

criterion = FocalLossWithLogits(alpha=0.35, gamma=1.5, reduction="mean")
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
num_epochs = 8
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

batch_size = 2048
eval_batch_size = 16384
train_dataset = TensorDataset(
    torch.from_numpy(X_train_norm), torch.from_numpy(y_train_np)
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)


def evaluate_nn(eval_model, X_data, y_data):
    """Compute calibrated probability scores and AP on hold-out set."""
    eval_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_data), eval_batch_size):
            batch_x = torch.tensor(
                X_data[i : i + eval_batch_size],
                dtype=torch.float32,
                device=device,
            )
            probs = torch.sigmoid(eval_model(batch_x))
            preds.append(probs.detach().cpu().numpy())
    scores = np.concatenate(preds).astype(float)
    ap = float(average_precision_score(y_data, scores))
    return ap, scores


def predict_nn(eval_model, X_data):
    """Generate forward pass calibrated test predictions."""
    eval_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_data), eval_batch_size):
            batch_x = torch.tensor(
                X_data[i : i + eval_batch_size],
                dtype=torch.float32,
                device=device,
            )
            probs = torch.sigmoid(eval_model(batch_x))
            preds.append(probs.detach().cpu().numpy())
    return np.concatenate(preds).astype(float)


best_val_ap = -1.0
best_model_weights = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * len(batch_y)
        num_samples += len(batch_y)

    scheduler.step()
    epoch_loss = running_loss / max(1, num_samples)

    val_nn_ap, _ = evaluate_nn(model, X_val_norm, y_val_np)
    if val_nn_ap > best_val_ap:
        best_val_ap = val_nn_ap
        best_model_weights = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs:02d} - Train Loss: {epoch_loss:.5f} - Val AP: {val_nn_ap:.6f}"
    )

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

val_nn_ap, val_nn_probs = evaluate_nn(model, X_val_norm, y_val_np)
test_nn_probs = predict_nn(model, X_test_norm)

# ---------------------------------------------------------
# Dynamic Ensemble & Submission Generation
# ---------------------------------------------------------
best_ens_ap = -1.0
best_w = 1.0

for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    ens_val_probs = w * val_lgb_probs + (1.0 - w) * val_nn_probs
    score = float(average_precision_score(y_val_np, ens_val_probs))
    if score > best_ens_ap:
        best_ens_ap = score
        best_w = w

final_val_score = best_ens_ap
final_test_scores = best_w * test_lgb_probs + (1.0 - best_w) * test_nn_probs

sub_df = pd.DataFrame(
    {
        "bbl": df_test["bbl"].astype(str).str.strip().str.zfill(10),
        "score": final_test_scores,
    }
)
sub_df.to_csv("./submission/submission.csv", index=False)

assert os.path.exists("./submission/submission.csv"), "Submission missing!"
assert len(sub_df) == len(df_test), f"Length mismatch: {len(sub_df)} vs {len(df_test)}"
assert sub_df["score"].isna().sum() == 0, "NaNs in submission scores"
assert (sub_df["bbl"].str.len() == 10).all(), "Malformed BBL length"

print(f"Final Validation Score: {final_val_score:.6f}")

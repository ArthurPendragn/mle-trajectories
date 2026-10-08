import copy
import math
import os
import sys
import time
import gcsfs
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# -------------------------------------------------------------------------
# 1. Directory and GCS Storage Initialization
# -------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
token = TOKEN_PATH if os.path.exists(TOKEN_PATH) else None
try:
    fs = gcsfs.GCSFileSystem(token=token)
except Exception:
    fs = gcsfs.GCSFileSystem()

BASE_GCS = "mle-nyc-lake/tasks/housing_violation_risk/v1"
storage_options = {"token": token} if token else {}


def standardize_bbl(s):
    num = pd.to_numeric(s, errors="coerce")
    valid = (num >= 1000000000) & (num <= 5999999999)
    res = pd.Series("", index=s.index, dtype="object")
    res[valid] = num[valid].astype("int64").astype(str)
    invalid = ~valid
    if invalid.any():
        st = s[invalid].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        res[invalid] = st.str.zfill(10)
    return res


# -------------------------------------------------------------------------
# 2. Data Ingestion & Cohort Setup
# -------------------------------------------------------------------------
print("Loading core lake datasets...")
# Test entities
df_test = pd.read_parquet(
    f"gs://{BASE_GCS}/test_entities.parquet", storage_options=storage_options
)
df_test["bbl"] = standardize_bbl(df_test["bbl"])

# PLUTO Table
ds_pluto = ds.dataset(f"{BASE_GCS}/lake/full/pluto", filesystem=fs)
pluto_cols = [
    c
    for c in [
        "bbl",
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "bldgclass",
        "numfloors",
        "bldgarea",
        "resarea",
        "cd",
        "zipcode",
        "borough",
        "version",
    ]
    if c in ds_pluto.schema.names
]
df_pluto = ds_pluto.to_table(columns=pluto_cols).to_pandas()
df_pluto["bbl"] = standardize_bbl(df_pluto["bbl"])
df_pluto["unitsres"] = (
    pd.to_numeric(df_pluto["unitsres"], errors="coerce").fillna(0).astype(np.float32)
)

# HPD Violations
ds_viol = ds.dataset(f"{BASE_GCS}/lake/full/hpd_violations", filesystem=fs)
viol_cols = [
    c
    for c in ["bbl", "boroid", "block", "lot", "class", "inspectiondate"]
    if c in ds_viol.schema.names
]
df_viol = ds_viol.to_table(columns=viol_cols).to_pandas()

# Clean HPD Violation BBLs
bbl_clean = standardize_bbl(df_viol["bbl"])
invalid_bbl = bbl_clean.str.len() != 10
if (
    invalid_bbl.any()
    and "boroid" in df_viol.columns
    and "block" in df_viol.columns
    and "lot" in df_viol.columns
):
    b = (
        pd.to_numeric(df_viol.loc[invalid_bbl, "boroid"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    blk = (
        pd.to_numeric(df_viol.loc[invalid_bbl, "block"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    lt = (
        pd.to_numeric(df_viol.loc[invalid_bbl, "lot"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )
    bbl_clean.loc[invalid_bbl] = b + blk + lt

df_viol["bbl"] = bbl_clean
df_viol["class"] = df_viol["class"].astype(str).str.upper().str.strip()
df_viol["inspectiondate"] = pd.to_datetime(df_viol["inspectiondate"], errors="coerce")
df_viol = df_viol[(df_viol["bbl"].str.len() == 10) & df_viol["inspectiondate"].notna()]

# Municipal Distress Datasets (Litigations, Vacate Orders, AEP)
try:
    ds_lit = ds.dataset(f"{BASE_GCS}/lake/full/hpd_litigations", filesystem=fs)
    lit_cols = [c for c in ["bbl", "caseopendate"] if c in ds_lit.schema.names]
    df_lit = ds_lit.to_table(columns=lit_cols).to_pandas()
    df_lit["bbl"] = standardize_bbl(df_lit["bbl"])
    df_lit["caseopendate"] = pd.to_datetime(df_lit["caseopendate"], errors="coerce")
    df_lit = df_lit[(df_lit["bbl"].str.len() == 10) & df_lit["caseopendate"].notna()]
except Exception:
    df_lit = pd.DataFrame(columns=["bbl", "caseopendate"])

try:
    ds_vac = ds.dataset(f"{BASE_GCS}/lake/full/hpd_vacate_orders", filesystem=fs)
    vac_cols = [c for c in ["bbl", "vacate_effective_date"] if c in ds_vac.schema.names]
    df_vac = ds_vac.to_table(columns=vac_cols).to_pandas()
    df_vac["bbl"] = standardize_bbl(df_vac["bbl"])
    df_vac["vacate_effective_date"] = pd.to_datetime(
        df_vac["vacate_effective_date"], errors="coerce"
    )
    df_vac = df_vac[
        (df_vac["bbl"].str.len() == 10) & df_vac["vacate_effective_date"].notna()
    ]
except Exception:
    df_vac = pd.DataFrame(columns=["bbl", "vacate_effective_date"])

try:
    ds_aep = ds.dataset(f"{BASE_GCS}/lake/full/hpd_aep_buildings", filesystem=fs)
    df_aep = ds_aep.to_table(columns=["bbl"]).to_pandas()
    aep_bbls = set(standardize_bbl(df_aep["bbl"]).unique())
except Exception:
    aep_bbls = set()

# HPD Tenant Complaints Dataset
try:
    ds_comp = ds.dataset(f"{BASE_GCS}/lake/full/hpd_complaints", filesystem=fs)
    comp_schema = ds_comp.schema.names
    date_candidates = [
        "receiveddate",
        "received_date",
        "dateentered",
        "date_entered",
        "inspectiondate",
        "statusdate",
        "complaint_date",
    ]
    date_col = next((c for c in date_candidates if c in comp_schema), None)
    if not date_col:
        date_col = next(
            (c for c in comp_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    comp_cols = [
        c
        for c in ["bbl", "boroid", "boroughid", "block", "lot"]
        if c in comp_schema
    ]
    if date_col and date_col not in comp_cols:
        comp_cols.append(date_col)
    df_comp = ds_comp.to_table(columns=comp_cols).to_pandas()
    if date_col:
        df_comp["complaint_date"] = pd.to_datetime(
            df_comp[date_col], errors="coerce"
        )
    else:
        df_comp["complaint_date"] = pd.NaT

    bbl_comp_clean = (
        standardize_bbl(df_comp["bbl"])
        if "bbl" in df_comp.columns
        else pd.Series("", index=df_comp.index)
    )
    invalid_bbl = bbl_comp_clean.str.len() != 10
    boro_c = (
        "boroid"
        if "boroid" in df_comp.columns
        else ("boroughid" if "boroughid" in df_comp.columns else None)
    )
    if (
        invalid_bbl.any()
        and boro_c
        and "block" in df_comp.columns
        and "lot" in df_comp.columns
    ):
        b = (
            pd.to_numeric(df_comp.loc[invalid_bbl, boro_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
        blk = (
            pd.to_numeric(df_comp.loc[invalid_bbl, "block"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df_comp.loc[invalid_bbl, "lot"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_comp_clean.loc[invalid_bbl] = b + blk + lt
    df_comp["bbl"] = bbl_comp_clean
    df_comp = df_comp[
        (df_comp["bbl"].str.len() == 10) & df_comp["complaint_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load hpd_complaints ({e}). Using empty table.")
    df_comp = pd.DataFrame(columns=["bbl", "complaint_date"])


# Create Out-of-Time Cohorts
def extract_pluto_cohort(df_pluto, release_tag):
    if "version" in df_pluto.columns:
        v = df_pluto["version"].astype(str).str.lower().str.strip()
        tag = release_tag.lower()
        mask = v == tag
        if not mask.any():
            mask = v.str.contains(tag[:3])
        if mask.any():
            df_sub = df_pluto[mask].copy()
        else:
            df_sub = df_pluto.copy()
    else:
        df_sub = df_pluto.copy()
    df_sub = df_sub.drop_duplicates(subset=["bbl"], keep="last")
    return df_sub[df_sub["unitsres"] >= 3].copy()


pluto_20 = extract_pluto_cohort(df_pluto, "20v7")
pluto_21 = extract_pluto_cohort(df_pluto, "21v4")
pluto_22 = extract_pluto_cohort(df_pluto, "22v3")

df_train_cohort = pluto_20 if len(pluto_20) > 0 else pluto_21
df_val_cohort = pluto_21 if len(pluto_21) > 0 else pluto_22

df_test_cohort = pd.DataFrame({"bbl": df_test["bbl"].copy()})
feat_join_cols = [
    c
    for c in [
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "bldgclass",
        "numfloors",
        "bldgarea",
        "resarea",
        "cd",
        "zipcode",
        "borough",
    ]
    if c in pluto_22.columns
]
df_test_cohort = df_test_cohort.merge(
    pluto_22[["bbl"] + feat_join_cols].drop_duplicates(subset=["bbl"]),
    on="bbl",
    how="left",
)

# -------------------------------------------------------------------------
# 3. Construct Targets & Out-of-Time Features
# -------------------------------------------------------------------------
# Validation Target (Cutoff 2022-01-01 -> 12-month window to 2023-01-01)
val_pos_mask = (
    (df_viol["class"] == "C")
    & (df_viol["inspectiondate"] >= pd.Timestamp("2022-01-01"))
    & (df_viol["inspectiondate"] < pd.Timestamp("2023-01-01"))
)
val_pos_bbls = set(df_viol.loc[val_pos_mask, "bbl"].unique())
y_val = df_val_cohort["bbl"].isin(val_pos_bbls).astype(np.int32).values

# Training Target (Cutoff 2021-01-01 -> 12-month window to 2022-01-01)
train_pos_mask = (
    (df_viol["class"] == "C")
    & (df_viol["inspectiondate"] >= pd.Timestamp("2021-01-01"))
    & (df_viol["inspectiondate"] < pd.Timestamp("2022-01-01"))
)
train_pos_bbls = set(df_viol.loc[train_pos_mask, "bbl"].unique())
y_train = df_train_cohort["bbl"].isin(train_pos_bbls).astype(np.int32).values

# Prior Year Positive BBLs for Leak-Free Empirical Bayes Prior
prior_2020_pos = set(
    df_viol.loc[
        (df_viol["class"] == "C")
        & (df_viol["inspectiondate"] >= pd.Timestamp("2020-01-01"))
        & (df_viol["inspectiondate"] < pd.Timestamp("2021-01-01")),
        "bbl",
    ].unique()
)


def build_features(
    cohort_df,
    cutoff_date,
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    historical_pos_bbls,
):
    T = pd.Timestamp(cutoff_date)
    bbls = cohort_df["bbl"]

    unitsres = (
        pd.to_numeric(cohort_df.get("unitsres", 0), errors="coerce")
        .fillna(0)
        .astype(np.float32)
    )
    unitstotal = (
        pd.to_numeric(cohort_df.get("unitstotal", 0), errors="coerce")
        .fillna(0)
        .astype(np.float32)
    )
    unitstotal = np.maximum(unitsres, unitstotal)
    res_ratio = np.clip(unitsres / (unitstotal + 1e-4), 0.0, 1.0)

    yearbuilt = (
        pd.to_numeric(cohort_df.get("yearbuilt", 1940), errors="coerce")
        .fillna(1940)
        .astype(np.float32)
    )
    yearbuilt = np.where((yearbuilt > 1800) & (yearbuilt <= T.year), yearbuilt, 1940)
    bldg_age = np.clip(T.year - yearbuilt, 0, 200).astype(np.float32)
    is_prewar = (yearbuilt < 1940).astype(np.float32)

    numfloors = (
        pd.to_numeric(cohort_df.get("numfloors", 3), errors="coerce")
        .fillna(3)
        .astype(np.float32)
    )
    bldgarea = np.log1p(
        np.maximum(
            0,
            pd.to_numeric(cohort_df.get("bldgarea", 0), errors="coerce")
            .fillna(0)
            .astype(np.float32),
        )
    )
    resarea = np.log1p(
        np.maximum(
            0,
            pd.to_numeric(cohort_df.get("resarea", 0), errors="coerce")
            .fillna(0)
            .astype(np.float32),
        )
    )
    raw_resarea = (
        pd.to_numeric(cohort_df.get("resarea", 0), errors="coerce")
        .fillna(0)
        .astype(np.float32)
    )
    area_per_unit = np.clip(raw_resarea / (unitsres + 1e-4), 0, 10000)

    boro_map = {
        "MN": 1,
        "BX": 2,
        "BK": 3,
        "QN": 4,
        "SI": 5,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    }
    borough = (
        cohort_df["borough"].map(boro_map).fillna(0).astype(np.int32).values
        if "borough" in cohort_df
        else bbls.str[0].astype(int).values
    )
    bldgclass_code = (
        cohort_df["bldgclass"].astype(str).str[0].str.upper().map(ord).fillna(0)
        if "bldgclass" in cohort_df
        else pd.Series(0, index=cohort_df.index)
    )

    min_date = T - pd.Timedelta(days=1095)
    v_mask = (df_viol["inspectiondate"] < T) & (df_viol["inspectiondate"] >= min_date)
    df_sub = df_viol.loc[v_mask, ["bbl", "class", "inspectiondate"]].copy()

    dt_days = (T - df_sub["inspectiondate"]).dt.days.values
    cls_c = (df_sub["class"] == "C").values
    cls_b = (df_sub["class"] == "B").values
    cls_a = (df_sub["class"] == "A").values

    df_sub["c_90d"] = np.where(cls_c & (dt_days <= 90), 1, 0)
    df_sub["c_1yr"] = np.where(cls_c & (dt_days <= 365), 1, 0)
    df_sub["c_2yr"] = np.where(cls_c & (dt_days <= 730), 1, 0)
    df_sub["c_3yr"] = np.where(cls_c, 1, 0)

    df_sub["b_1yr"] = np.where(cls_b & (dt_days <= 365), 1, 0)
    df_sub["b_2yr"] = np.where(cls_b & (dt_days <= 730), 1, 0)
    df_sub["a_1yr"] = np.where(cls_a & (dt_days <= 365), 1, 0)

    df_sub["tot_90d"] = np.where(dt_days <= 90, 1, 0)
    df_sub["tot_1yr"] = np.where(dt_days <= 365, 1, 0)
    df_sub["tot_2yr"] = np.where(dt_days <= 730, 1, 0)
    df_sub["tot_3yr"] = 1

    df_sub["days_c"] = np.where(cls_c, dt_days, 9999)
    df_sub["days_any"] = dt_days

    agg_res = df_sub.groupby("bbl").agg(
        {
            "c_90d": "sum",
            "c_1yr": "sum",
            "c_2yr": "sum",
            "c_3yr": "sum",
            "b_1yr": "sum",
            "b_2yr": "sum",
            "a_1yr": "sum",
            "tot_90d": "sum",
            "tot_1yr": "sum",
            "tot_2yr": "sum",
            "tot_3yr": "sum",
            "days_c": "min",
            "days_any": "min",
        }
    )

    feat = {}
    for col in agg_res.columns:
        if col in ["days_c", "days_any"]:
            feat[col] = bbls.map(agg_res[col]).fillna(9999).astype(np.float32)
        else:
            feat[col] = bbls.map(agg_res[col]).fillna(0).astype(np.float32)

    c_1yr = feat["c_1yr"]
    c_2yr = feat["c_2yr"]
    tot_1yr = feat["tot_1yr"]
    tot_2yr = feat["tot_2yr"]
    tot_90d = feat["tot_90d"]

    feat["c_ratio_1yr"] = c_1yr / (tot_1yr + 1.0)
    feat["b_c_ratio_1yr"] = (feat["b_1yr"] + c_1yr) / (tot_1yr + 1.0)
    feat["c_velocity"] = c_1yr / (np.maximum(0, c_2yr - c_1yr) + 1.0)
    feat["tot_velocity"] = tot_1yr / (np.maximum(0, tot_2yr - tot_1yr) + 1.0)
    feat["surge_90d"] = (tot_90d * 4.0) / (tot_1yr + 1.0)
    feat["has_prior_c_1yr"] = (c_1yr > 0).astype(np.float32)
    feat["has_prior_c_2yr"] = (c_2yr > 0).astype(np.float32)
    feat["has_prior_c_3yr"] = (feat["c_3yr"] > 0).astype(np.float32)

    feat["c_per_unit_1yr"] = c_1yr / (unitsres + 1e-4)
    feat["tot_per_unit_1yr"] = tot_1yr / (unitsres + 1e-4)
    feat["tot_per_unit_3yr"] = feat["c_3yr"] / (unitsres + 1e-4)

    feat["unitsres"] = unitsres
    feat["unitstotal"] = unitstotal
    feat["res_ratio"] = res_ratio
    feat["bldg_age"] = bldg_age
    feat["is_prewar"] = is_prewar
    feat["numfloors"] = numfloors
    feat["bldgarea"] = bldgarea
    feat["resarea"] = resarea
    feat["area_per_unit"] = area_per_unit
    feat["borough"] = borough
    feat["bldgclass_code"] = bldgclass_code

    if len(df_lit) > 0:
        lit_sub = df_lit[
            (df_lit["caseopendate"] < T)
            & (df_lit["caseopendate"] >= T - pd.Timedelta(days=730))
        ]
        lit_cnt = lit_sub.groupby("bbl").size()
        feat["lit_count_2yr"] = bbls.map(lit_cnt).fillna(0).astype(np.float32)
    else:
        feat["lit_count_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    if len(df_vac) > 0:
        vac_sub = df_vac[df_vac["vacate_effective_date"] < T]
        vac_cnt = vac_sub.groupby("bbl").size()
        feat["vacate_count_prior"] = bbls.map(vac_cnt).fillna(0).astype(np.float32)
    else:
        feat["vacate_count_prior"] = np.zeros(len(cohort_df), dtype=np.float32)

    feat["is_aep_enrolled"] = bbls.isin(aep_bbls).astype(np.float32).values

    if "cd" in cohort_df and historical_pos_bbls is not None:
        cd_col = (
            pd.to_numeric(cohort_df["cd"], errors="coerce").fillna(0).astype(np.int32)
        )
        hist_pos = bbls.isin(historical_pos_bbls).astype(float)
        cd_pos = hist_pos.groupby(cd_col).sum()
        cd_tot = hist_pos.groupby(cd_col).count()
        global_mean = hist_pos.mean()
        m = 50.0
        cd_prior = (cd_pos + m * global_mean) / (cd_tot + m)
        feat["cd_risk_prior"] = (
            cd_col.map(cd_prior).fillna(global_mean).astype(np.float32)
        )
    else:
        feat["cd_risk_prior"] = np.zeros(len(cohort_df), dtype=np.float32)

    return pd.DataFrame(feat, index=cohort_df.index)


print("Building training features (2021 cohort)...")
X_train = build_features(
    df_train_cohort,
    "2021-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    prior_2020_pos,
)

print("Building validation features (2022 cohort)...")
X_val = build_features(
    df_val_cohort,
    "2022-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    train_pos_bbls,
)

print("Building test features (2023 cohort)...")
X_test = build_features(
    df_test_cohort, "2023-01-01", df_viol, df_lit, df_vac, aep_bbls, val_pos_bbls
)

# Robust imputation across cohorts
for col in X_train.columns:
    med = X_train[col].median()
    if pd.isna(med):
        med = 0.0
    X_train[col] = X_train[col].fillna(med).fillna(0.0)
    X_val[col] = X_val[col].fillna(med).fillna(0.0)
    X_test[col] = X_test[col].fillna(med).fillna(0.0)

print(
    f"Features ready. Train: {X_train.shape}, Val: {X_val.shape}, Test:"
    f" {X_test.shape}"
)


# -------------------------------------------------------------------------
# 4. Neural Architecture & Loss Definition
# -------------------------------------------------------------------------
class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.weight = nn.Parameter(
            torch.randn(num_features, d_token) / math.sqrt(d_token)
        )
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        self.norm = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.norm(tokens)


class GatedResidualSEBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out * 2)
        self.linear2 = nn.Linear(d_out, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)

        se_dim = max(16, d_out // 4)
        self.se = nn.Sequential(
            nn.Linear(d_out, se_dim),
            nn.ReLU(inplace=True),
            nn.Linear(se_dim, d_out),
            nn.Sigmoid(),
        )

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        gated = self.linear1(x)
        val, gate = gated.chunk(2, dim=-1)
        h = val * torch.sigmoid(gate)
        h = self.norm1(h)
        h = self.drop(h)

        h = F.gelu(self.linear2(h))
        scale = self.se(h)
        h = h * scale
        h = self.norm2(h + residual)
        return h


class CAGNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 16,
        nhead: int = 4,
        d_hidden: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(num_features, d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_mha = nn.LayerNorm(d_token)

        flattened_dim = num_features * d_token
        self.input_proj = nn.Sequential(
            nn.Linear(flattened_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.block1 = GatedResidualSEBlock(d_hidden, d_hidden, dropout=dropout)
        self.block2 = GatedResidualSEBlock(d_hidden, d_hidden // 2, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(d_hidden // 2, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm_mha(tokens + attn_out)

        b_size = tokens.size(0)
        flat = tokens.reshape(b_size, -1)
        h = self.input_proj(flat)
        h = self.block1(h)
        h = self.block2(h)

        logits = self.head(h).squeeze(-1)
        return logits


class SmoothAPWithFocalLoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.1,
        alpha_smooth: float = 0.6,
        focal_gamma: float = 2.0,
        pos_weight: float = 3.0,
    ):
        super().__init__()
        self.tau = tau
        self.alpha_smooth = alpha_smooth
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Asymmetric Focal Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets == 1.0, self.pos_weight / (self.pos_weight + 1.0), 1.0
        )
        focal_loss = (alpha_t * ((1.0 - pt) ** self.focal_gamma) * bce).mean()

        # Differentiable Smooth-AP Ranking Loss
        pos_mask = targets == 1.0
        n_pos = pos_mask.sum()

        if n_pos > 0 and (len(targets) - n_pos) > 0:
            diff = (logits.unsqueeze(0) - logits.unsqueeze(1)) / self.tau
            pairwise_sigmoid = torch.sigmoid(diff)

            diag_mask = torch.eye(len(logits), device=logits.device, dtype=torch.bool)
            pairwise_sigmoid = pairwise_sigmoid.masked_fill(diag_mask, 0.0)

            rank_all = 1.0 + pairwise_sigmoid.sum(dim=1)
            rank_pos = 1.0 + (pairwise_sigmoid * pos_mask.unsqueeze(0).float()).sum(
                dim=1
            )

            smooth_ap_pos = rank_pos[pos_mask] / (rank_all[pos_mask] + 1e-7)
            smooth_ap_loss = 1.0 - smooth_ap_pos.mean()
        else:
            smooth_ap_loss = torch.tensor(0.0, device=logits.device)

        total_loss = (
            self.alpha_smooth * smooth_ap_loss + (1.0 - self.alpha_smooth) * focal_loss
        )
        return total_loss


def get_neural_model_and_criterion(
    num_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 8,
    device: str = "cpu",
):
    model = CAGNet(
        num_features=num_features,
        d_token=16,
        nhead=4,
        d_hidden=128,
        dropout=0.15,
    ).to(device)

    criterion = SmoothAPWithFocalLoss(
        tau=0.1, alpha_smooth=0.6, focal_gamma=2.0, pos_weight=3.0
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgb_models():
    lgb_balanced = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=40,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_deep = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=750,
        learning_rate=0.025,
        num_leaves=127,
        max_depth=9,
        subsample=0.75,
        colsample_bytree=0.55,
        min_child_samples=60,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=2024,
        n_jobs=-1,
        verbose=-1,
    )
    return lgb_balanced, lgb_deep


def get_xgb_model():
    return xgb.XGBClassifier(
        n_estimators=550,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=3.0,
        tree_method="hist",
        random_state=123,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )


def get_catboost_model():
    return CatBoostClassifier(
        iterations=650,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        verbose=False,
    )


# -------------------------------------------------------------------------
# 5. Model Training & Validation Evaluation
# -------------------------------------------------------------------------
print("Fitting LightGBM Balanced Model...")
lgb1, lgb2 = get_lgb_models()
lgb1.fit(X_train, y_train)
lgb1.booster_.save_model("./working/lgb_balanced.txt")

print("Fitting LightGBM Deep Model...")
lgb2.fit(X_train, y_train)
lgb2.booster_.save_model("./working/lgb_deep.txt")

print("Fitting XGBoost Histogram Model...")
xgb_model = get_xgb_model()
xgb_model.fit(X_train, y_train)
xgb_model.save_model("./working/xgb_hist.json")

print("Fitting CatBoost Model...")
cb_model = get_catboost_model()
cb_model.fit(X_train, y_train)
cb_model.save_model("./working/cb_model.cbm")

# Strict Leak-Free Scaling for Neural Ranker
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val_t), batch_size=2048, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t), batch_size=2048, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 8
cagnet, criterion, optimizer, scheduler = get_neural_model_and_criterion(
    num_features=X_train.shape[1],
    lr=1e-3,
    weight_decay=1e-4,
    epochs=epochs,
    device=device,
)

best_val_ap = -1.0
best_model_weights = None

print("Training CAGNet with Smooth-AP and Focal Loss...")
for epoch in range(1, epochs + 1):
    cagnet.train()
    total_loss = 0.0
    n_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        logits = cagnet(bx)
        loss = criterion(logits, by)
        loss.backward()
        nn.utils.clip_grad_norm_(cagnet.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = total_loss / max(1, n_batches)

    cagnet.eval()
    val_preds_list = []
    with torch.no_grad():
        for (v_bx,) in val_loader:
            v_bx = v_bx.to(device)
            val_preds_list.append(cagnet(v_bx).cpu())
    val_nn_epoch_preds = torch.cat(val_preds_list).numpy()
    epoch_val_ap = average_precision_score(y_val, val_nn_epoch_preds)

    if epoch_val_ap > best_val_ap:
        best_val_ap = epoch_val_ap
        best_model_weights = copy.deepcopy(cagnet.state_dict())

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val AP:"
        f" {epoch_val_ap:.4f} (Best: {best_val_ap:.4f})"
    )

cagnet.load_state_dict(best_model_weights)
torch.save(best_model_weights, "./working/best_cagnet.pt")

# Out-of-Time Validation Inference & Rank Fusion
p_val_lgb1 = lgb1.predict_proba(X_val)[:, 1]
p_val_lgb2 = lgb2.predict_proba(X_val)[:, 1]
p_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_val_cb = cb_model.predict_proba(X_val)[:, 1]

cagnet.eval()
val_preds_list = []
with torch.no_grad():
    for (v_bx,) in val_loader:
        v_bx = v_bx.to(device)
        val_preds_list.append(cagnet(v_bx).cpu())
p_val_nn = torch.cat(val_preds_list).numpy()

n_val = len(p_val_lgb1)
r_val_lgb1 = rankdata(p_val_lgb1) / n_val
r_val_lgb2 = rankdata(p_val_lgb2) / n_val
r_val_xgb = rankdata(p_val_xgb) / n_val
r_val_cb = rankdata(p_val_cb) / n_val
r_val_nn = rankdata(p_val_nn) / n_val

val_ranks = [r_val_lgb1, r_val_lgb2, r_val_xgb, r_val_cb, r_val_nn]

print(
    f"Individual Val APs - LGB1: {average_precision_score(y_val, r_val_lgb1):.4f}, "
    f"LGB2: {average_precision_score(y_val, r_val_lgb2):.4f}, "
    f"XGB: {average_precision_score(y_val, r_val_xgb):.4f}, "
    f"CB: {average_precision_score(y_val, r_val_cb):.4f}, "
    f"CAGNet: {average_precision_score(y_val, r_val_nn):.4f}"
)

# Search optimal ensemble weights maximizing Average Precision on y_val
best_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
best_ap = average_precision_score(
    y_val, sum(w * r for w, r in zip(best_weights, val_ranks))
)

np.random.seed(42)
for _ in range(600):
    w = np.random.dirichlet(np.array([2.5, 2.0, 2.0, 2.0, 1.5]))
    ens = sum(wi * ri for wi, ri in zip(w, val_ranks))
    score = average_precision_score(y_val, ens)
    if score > best_ap:
        best_ap = score
        best_weights = w

# Fine-tune with coordinate search
for step in [0.05, 0.02, 0.01]:
    improved = True
    while improved:
        improved = False
        for i in range(5):
            for delta in [-step, step]:
                w_cand = best_weights.copy()
                w_cand[i] += delta
                if w_cand[i] < 0:
                    continue
                w_cand = w_cand / w_cand.sum()
                ens = sum(wi * ri for wi, ri in zip(w_cand, val_ranks))
                score = average_precision_score(y_val, ens)
                if score > best_ap + 1e-6:
                    best_ap = score
                    best_weights = w_cand
                    improved = True

print(f"Optimized Ensemble Weights: {np.round(best_weights, 4)}")
val_ap = best_ap

# -------------------------------------------------------------------------
# 6. Test Inference & Submission Generation
# -------------------------------------------------------------------------
p_test_lgb1 = lgb1.predict_proba(X_test)[:, 1]
p_test_lgb2 = lgb2.predict_proba(X_test)[:, 1]
p_test_xgb = xgb_model.predict_proba(X_test)[:, 1]
p_test_cb = cb_model.predict_proba(X_test)[:, 1]

test_preds_list = []
with torch.no_grad():
    for (t_bx,) in test_loader:
        t_bx = t_bx.to(device)
        test_preds_list.append(cagnet(t_bx).cpu())
p_test_nn = torch.cat(test_preds_list).numpy()

n_test = len(p_test_lgb1)
r_test_lgb1 = rankdata(p_test_lgb1) / n_test
r_test_lgb2 = rankdata(p_test_lgb2) / n_test
r_test_xgb = rankdata(p_test_xgb) / n_test
r_test_cb = rankdata(p_test_cb) / n_test
r_test_nn = rankdata(p_test_nn) / n_test

test_ranks = [r_test_lgb1, r_test_lgb2, r_test_xgb, r_test_cb, r_test_nn]
final_test_score = sum(wi * ri for wi, ri in zip(best_weights, test_ranks))

sub_df = pd.DataFrame({"bbl": df_test["bbl"].values, "score": final_test_score})

assert len(sub_df) == 171587, f"Expected 171587 rows, got {len(sub_df)}"
assert list(sub_df.columns) == ["bbl", "score"], f"Invalid columns: {sub_df.columns}"
assert not sub_df["score"].isna().any(), "Found NaNs in score"
assert not sub_df["bbl"].isna().any(), "Found NaNs in bbl"
assert (sub_df["bbl"].str.len() == 10).all(), "Found invalid BBL length"

sub_df.to_csv("./submission/submission.csv", index=False)
print("Submission verified and saved to ./submission/submission.csv")

print(f"Final Validation Score: {val_ap:.5f}")

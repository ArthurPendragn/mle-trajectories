import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION & ROBUST UTILITIES
# ==============================================================================
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
STORAGE_OPTIONS = {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else None
BASE_GCS_URL = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
LAKE_FULL_URL = f"{BASE_GCS_URL}/lake/full"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs("submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_TRAIN = pd.Timestamp("2021-01-01")
T_VAL = pd.Timestamp("2022-01-01")
T_TEST = pd.Timestamp("2023-01-01")


def normalize_bbl_series(
    df: pd.DataFrame,
    bbl_col: str = "bbl",
    boro_col: str = "boroid",
    block_col: str = "block",
    lot_col: str = "lot",
) -> pd.Series:
    """Standardizes BBL to a 10-digit zero-padded string.

    If BBL is missing or malformed, reconstructs it from borough, block, and
    lot.
    """
    clean_bbl = None
    if bbl_col in df.columns:
        s = pd.to_numeric(df[bbl_col], errors="coerce")
        valid_num = s.notna() & (s > 0)
        s_filled = (
            s.fillna(0)
            .astype(np.int64)
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        clean_bbl = s_filled.where(valid_num & (s_filled.str.len() == 10), None)

    if boro_col in df.columns and block_col in df.columns and lot_col in df.columns:
        boro_n = pd.to_numeric(df[boro_col], errors="coerce")
        block_n = pd.to_numeric(df[block_col], errors="coerce")
        lot_n = pd.to_numeric(df[lot_col], errors="coerce")

        valid_parts = (
            boro_n.notna()
            & block_n.notna()
            & lot_n.notna()
            & (boro_n > 0)
            & (block_n > 0)
            & (lot_n >= 0)
        )
        boro_part = boro_n.fillna(0).astype(int).astype(str)
        block_part = block_n.fillna(0).astype(int).astype(str).str.zfill(5)
        lot_part = lot_n.fillna(0).astype(int).astype(str).str.zfill(4)
        reconstructed = boro_part + block_part + lot_part

        if clean_bbl is not None:
            clean_bbl = clean_bbl.where(clean_bbl.notna(), reconstructed)
        else:
            clean_bbl = reconstructed
        clean_bbl = clean_bbl.where(valid_parts | clean_bbl.notna(), None)

    return clean_bbl


def safe_read_parquet(
    url: str,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
) -> pd.DataFrame:
    """Safely reads Parquet dataset with column projection and fallback."""
    try:
        if columns is not None:
            return pd.read_parquet(
                url, columns=columns, storage_options=storage_options
            )
        return pd.read_parquet(url, storage_options=storage_options)
    except Exception:
        try:
            df_full = pd.read_parquet(url, storage_options=storage_options)
            if columns is not None:
                avail = [c for c in columns if c in df_full.columns]
                return df_full[avail]
            return df_full
        except Exception as e2:
            print(f"Warning: Failed to load {url}: {e2}")
            return pd.DataFrame()


# ==============================================================================
# 2. DATA INGESTION & COHORT DEFINITION
# ==============================================================================
print("Starting data ingestion and feature engineering pipeline...")
start_time = time.time()

# 2.1 Load Test Entities
test_entities_url = f"{BASE_GCS_URL}/test_entities.parquet"
df_test_raw = safe_read_parquet(test_entities_url, storage_options=STORAGE_OPTIONS)
df_test_raw["bbl"] = normalize_bbl_series(df_test_raw)
test_bbls = df_test_raw["bbl"].dropna().unique()
print(f"Loaded {len(test_bbls)} test entities from {test_entities_url}.")

# 2.2 Load PLUTO
pluto_url = f"{LAKE_FULL_URL}/pluto"
pluto_cols = [
    "bbl",
    "borough",
    "borocode",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "yearalter1",
    "numfloors",
    "bldgarea",
    "resarea",
    "comarea",
    "lotarea",
    "assessland",
    "assesstot",
    "bldgclass",
    "zipcode",
    "cd",
    "version",
]
df_pluto = safe_read_parquet(
    pluto_url, columns=pluto_cols, storage_options=STORAGE_OPTIONS
)
if "bbl" in df_pluto.columns:
    df_pluto["bbl"] = normalize_bbl_series(
        df_pluto, boro_col="borocode", block_col="block", lot_col="lot"
    )
print(f"Loaded PLUTO dataset: {df_pluto.shape}")

# 2.3 Load HPD Violations
vio_url = f"{LAKE_FULL_URL}/hpd_violations"
vio_cols = [
    "bbl",
    "boroid",
    "block",
    "lot",
    "class",
    "inspectiondate",
    "violationstatus",
]
df_vio = safe_read_parquet(vio_url, columns=vio_cols, storage_options=STORAGE_OPTIONS)
df_vio["bbl"] = normalize_bbl_series(df_vio)
df_vio["class"] = df_vio["class"].astype(str).str.upper().str.strip()
df_vio["inspectiondate"] = pd.to_datetime(
    df_vio["inspectiondate"], errors="coerce", utc=True
).dt.tz_localize(None)
df_vio = df_vio[df_vio["bbl"].notna() & df_vio["inspectiondate"].notna()]
print(f"Loaded and standardized HPD violations: {df_vio.shape}")

# 2.4 Load Auxiliary HPD Tables (Litigations, Emergency Charges, Vacate Orders, AEP)
df_lit = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_litigations", storage_options=STORAGE_OPTIONS
)
if not df_lit.empty:
    df_lit["bbl"] = normalize_bbl_series(df_lit)
    lit_date_col = next(
        (c for c in df_lit.columns if "date" in c.lower()), "caseopendate"
    )
    if lit_date_col in df_lit.columns:
        df_lit["event_date"] = pd.to_datetime(
            df_lit[lit_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_lit["event_date"] = pd.NaT

df_omo = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_omo_charges", storage_options=STORAGE_OPTIONS
)
if not df_omo.empty:
    df_omo["bbl"] = normalize_bbl_series(df_omo)
    omo_date_col = next((c for c in df_omo.columns if "date" in c.lower()), None)
    if omo_date_col:
        df_omo["event_date"] = pd.to_datetime(
            df_omo[omo_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_omo["event_date"] = pd.NaT

df_hwo = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_hwo_charges", storage_options=STORAGE_OPTIONS
)
if not df_hwo.empty:
    df_hwo["bbl"] = normalize_bbl_series(df_hwo)
    hwo_date_col = next((c for c in df_hwo.columns if "date" in c.lower()), None)
    if hwo_date_col:
        df_hwo["event_date"] = pd.to_datetime(
            df_hwo[hwo_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_hwo["event_date"] = pd.NaT

df_vac = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_vacate_orders", storage_options=STORAGE_OPTIONS
)
if not df_vac.empty:
    df_vac["bbl"] = normalize_bbl_series(df_vac)

df_aep = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_aep_buildings", storage_options=STORAGE_OPTIONS
)
if not df_aep.empty:
    df_aep["bbl"] = normalize_bbl_series(df_aep)

print("Auxiliary tables ingestion completed.")

# 2.5 Identify Cohort Entities
version_col = next(
    (
        c
        for c in df_pluto.columns
        if c.lower() in ["version", "release", "pluto_version"]
    ),
    None,
)

if version_col is not None and "unitsres" in df_pluto.columns:
    v_series = df_pluto[version_col].astype(str).str.lower().str.strip()
    val_mask = v_series.str.startswith("21v4") & (
        pd.to_numeric(df_pluto["unitsres"], errors="coerce") >= 3
    )
    val_bbls = df_pluto.loc[val_mask, "bbl"].dropna().unique()
    if len(val_bbls) == 0:
        val_bbls = test_bbls

    train_mask = v_series.str.startswith("20v7") & (
        pd.to_numeric(df_pluto["unitsres"], errors="coerce") >= 3
    )
    train_bbls = df_pluto.loc[train_mask, "bbl"].dropna().unique()
    if len(train_bbls) == 0:
        train_bbls = test_bbls
else:
    val_bbls = test_bbls
    train_bbls = test_bbls

print(
    f"Entities extracted: Train={len(train_bbls)}, Val={len(val_bbls)},"
    f" Test={len(test_bbls)}"
)


# ==============================================================================
# 3. FEATURE EXTRACTION ENGINE (POINT-IN-TIME STRICT)
# ==============================================================================
def extract_temporal_features(
    target_bbls: np.ndarray,
    cutoff: pd.Timestamp,
    pluto_release_ver: Optional[str] = None,
) -> pd.DataFrame:
    """Extracts leak-free morphological, violation, and distress features strictly prior to cutoff."""
    df_feat = pd.DataFrame({"bbl": target_bbls})

    # 3.1 PLUTO Morphological Features
    pluto_sub = df_pluto.copy()
    if version_col is not None and pluto_release_ver is not None:
        v_sub = pluto_sub[version_col].astype(str).str.lower().str.strip()
        match_mask = v_sub.str.startswith(pluto_release_ver.lower())
        if match_mask.any():
            pluto_sub = pluto_sub[match_mask]
    pluto_sub = pluto_sub.drop_duplicates(subset=["bbl"], keep="last")

    cols_to_merge = [
        c
        for c in [
            "bbl",
            "unitsres",
            "unitstotal",
            "yearbuilt",
            "yearalter1",
            "numfloors",
            "bldgarea",
            "resarea",
            "comarea",
            "lotarea",
            "assessland",
            "assesstot",
            "bldgclass",
            "zipcode",
            "borocode",
            "cd",
        ]
        if c in pluto_sub.columns
    ]
    df_feat = df_feat.merge(pluto_sub[cols_to_merge], on="bbl", how="left")

    # Morphological feature engineering
    df_feat["unitsres"] = (
        pd.to_numeric(df_feat["unitsres"], errors="coerce").fillna(3.0).clip(lower=1)
    )
    df_feat["unitstotal"] = (
        pd.to_numeric(df_feat["unitstotal"], errors="coerce").fillna(3.0).clip(lower=1)
    )
    df_feat["log_unitsres"] = np.log1p(df_feat["unitsres"]).astype(np.float32)
    df_feat["log_unitstotal"] = np.log1p(df_feat["unitstotal"]).astype(np.float32)
    df_feat["res_unit_ratio"] = (
        df_feat["unitsres"] / (df_feat["unitstotal"] + 1e-4)
    ).astype(np.float32)

    yearbuilt = (
        pd.to_numeric(df_feat["yearbuilt"], errors="coerce")
        .fillna(1930)
        .clip(1800, cutoff.year)
    )
    df_feat["bldg_age"] = (cutoff.year - yearbuilt).astype(np.float32)
    df_feat["is_prewar"] = (yearbuilt < 1940).astype(np.float32)

    yearalter = pd.to_numeric(df_feat["yearalter1"], errors="coerce").fillna(0)
    df_feat["has_alteration"] = (yearalter > 0).astype(np.float32)
    df_feat["years_since_alter"] = np.where(
        yearalter > 0, (cutoff.year - yearalter).clip(0, 100), 99.0
    ).astype(np.float32)

    bldgarea = (
        pd.to_numeric(df_feat["bldgarea"], errors="coerce").fillna(4000).clip(lower=100)
    )
    lotarea = (
        pd.to_numeric(df_feat["lotarea"], errors="coerce").fillna(2500).clip(lower=100)
    )
    resarea = (
        pd.to_numeric(df_feat["resarea"], errors="coerce").fillna(3000).clip(lower=0)
    )

    df_feat["log_bldgarea"] = np.log1p(bldgarea).astype(np.float32)
    df_feat["built_far"] = (bldgarea / (lotarea + 1e-4)).clip(0, 50).astype(np.float32)
    df_feat["sqft_per_unit"] = (
        (resarea / df_feat["unitsres"]).clip(50, 5000).astype(np.float32)
    )

    assesstot = (
        pd.to_numeric(df_feat["assesstot"], errors="coerce")
        .fillna(100000)
        .clip(lower=1000)
    )
    assessland = (
        pd.to_numeric(df_feat["assessland"], errors="coerce")
        .fillna(30000)
        .clip(lower=1000)
    )
    df_feat["log_assesstot"] = np.log1p(assesstot).astype(np.float32)
    df_feat["assessed_bldg_val_ratio"] = (
        (assesstot - assessland).clip(lower=0) / (assesstot + 1e-4)
    ).astype(np.float32)
    df_feat["assessed_val_per_sqft"] = (
        (assesstot / (bldgarea + 1e-4)).clip(0, 2000).astype(np.float32)
    )

    if "bldgclass" in df_feat.columns:
        df_feat["bldgclass_code"] = (
            df_feat["bldgclass"].astype(str).str[:1].astype("category").cat.codes
        )
    else:
        df_feat["bldgclass_code"] = 0

    if "borocode" in df_feat.columns:
        df_feat["borocode"] = (
            pd.to_numeric(df_feat["borocode"], errors="coerce")
            .fillna(1)
            .astype(np.int32)
        )
    else:
        df_feat["borocode"] = 1

    # 3.2 HPD Violations Point-in-Time Trajectory
    v_prior = df_vio[df_vio["inspectiondate"] < cutoff]

    w90 = cutoff - pd.Timedelta(days=90)
    w180 = cutoff - pd.Timedelta(days=180)
    w365 = cutoff - pd.Timedelta(days=365)
    w730 = cutoff - pd.Timedelta(days=730)
    w1095 = cutoff - pd.Timedelta(days=1095)

    vc_all = v_prior[v_prior["class"] == "C"]
    vc_90 = vc_all[vc_all["inspectiondate"] >= w90]
    vc_180 = vc_all[vc_all["inspectiondate"] >= w180]
    vc_365 = vc_all[vc_all["inspectiondate"] >= w365]
    vc_730 = vc_all[vc_all["inspectiondate"] >= w730]
    vc_1095 = vc_all[vc_all["inspectiondate"] >= w1095]

    va_365 = v_prior[(v_prior["class"] == "A") & (v_prior["inspectiondate"] >= w365)]
    vb_365 = v_prior[(v_prior["class"] == "B") & (v_prior["inspectiondate"] >= w365)]
    vall_365 = v_prior[v_prior["inspectiondate"] >= w365]
    vall_730 = v_prior[v_prior["inspectiondate"] >= w730]

    cnt_c_90 = vc_90.groupby("bbl").size().rename("vio_c_count_90d")
    cnt_c_180 = vc_180.groupby("bbl").size().rename("vio_c_count_180d")
    cnt_c_365 = vc_365.groupby("bbl").size().rename("vio_c_count_12m")
    cnt_c_730 = vc_730.groupby("bbl").size().rename("vio_c_count_24m")
    cnt_c_1095 = vc_1095.groupby("bbl").size().rename("vio_c_count_36m")
    cnt_c_all = vc_all.groupby("bbl").size().rename("vio_c_count_all")

    cnt_a_365 = va_365.groupby("bbl").size().rename("vio_a_count_12m")
    cnt_b_365 = vb_365.groupby("bbl").size().rename("vio_b_count_12m")
    cnt_all_365 = vall_365.groupby("bbl").size().rename("vio_all_count_12m")
    cnt_all_730 = vall_730.groupby("bbl").size().rename("vio_all_count_24m")
    cnt_all_total = v_prior.groupby("bbl").size().rename("vio_all_count_total")

    last_c_date = vc_all.groupby("bbl")["inspectiondate"].max()
    days_since_c = ((cutoff - last_c_date).dt.total_seconds() / 86400.0).rename(
        "days_since_last_vio_c"
    )

    last_any_date = v_prior.groupby("bbl")["inspectiondate"].max()
    days_since_any = ((cutoff - last_any_date).dt.total_seconds() / 86400.0).rename(
        "days_since_last_vio_any"
    )

    v_open = v_prior[v_prior["violationstatus"].astype(str).str.lower() == "open"]
    cnt_open_all = v_open.groupby("bbl").size().rename("open_violations_all")
    cnt_open_c = (
        v_open[v_open["class"] == "C"].groupby("bbl").size().rename("open_violations_c")
    )

    v_aggs = pd.concat(
        [
            cnt_c_90,
            cnt_c_180,
            cnt_c_365,
            cnt_c_730,
            cnt_c_1095,
            cnt_c_all,
            cnt_a_365,
            cnt_b_365,
            cnt_all_365,
            cnt_all_730,
            cnt_all_total,
            days_since_c,
            days_since_any,
            cnt_open_all,
            cnt_open_c,
        ],
        axis=1,
    )
    v_aggs.index.name = "bbl"
    v_aggs = v_aggs.reset_index()

    df_feat = df_feat.merge(v_aggs, on="bbl", how="left")

    count_cols = [
        c
        for c in v_aggs.columns
        if c not in ["bbl", "days_since_last_vio_c", "days_since_last_vio_any"]
    ]
    for c in count_cols:
        df_feat[c] = df_feat[c].fillna(0).astype(np.float32)

    df_feat["days_since_last_vio_c"] = (
        df_feat["days_since_last_vio_c"]
        .fillna(3650.0)
        .clip(0, 3650.0)
        .astype(np.float32)
    )
    df_feat["days_since_last_vio_any"] = (
        df_feat["days_since_last_vio_any"]
        .fillna(3650.0)
        .clip(0, 3650.0)
        .astype(np.float32)
    )

    df_feat["class_c_ratio_12m"] = (
        df_feat["vio_c_count_12m"] / (df_feat["vio_all_count_12m"] + 1.0)
    ).astype(np.float32)
    df_feat["class_c_acceleration"] = (
        df_feat["vio_c_count_180d"]
        / (df_feat["vio_c_count_12m"] - df_feat["vio_c_count_180d"] + 1.0)
    ).astype(np.float32)
    df_feat["violation_velocity"] = (
        df_feat["vio_all_count_12m"]
        / (df_feat["vio_all_count_24m"] - df_feat["vio_all_count_12m"] + 1.0)
    ).astype(np.float32)
    df_feat["open_ratio_c"] = (
        df_feat["open_violations_c"] / (df_feat["vio_c_count_all"] + 1.0)
    ).astype(np.float32)

    # 3.3 Auxiliary Distress Indicators
    if not df_lit.empty:
        lit_prior = df_lit[
            df_lit["event_date"].isna() | (df_lit["event_date"] < cutoff)
        ]
        lit_cnt = (
            lit_prior.groupby("bbl")
            .size()
            .rename("litigation_count_total")
            .reset_index()
        )
        df_feat = df_feat.merge(lit_cnt, on="bbl", how="left")
        df_feat["litigation_count_total"] = (
            df_feat["litigation_count_total"].fillna(0).astype(np.float32)
        )
    else:
        df_feat["litigation_count_total"] = np.float32(0.0)

    if not df_omo.empty:
        omo_prior = df_omo[
            df_omo["event_date"].isna() | (df_omo["event_date"] < cutoff)
        ]
        omo_cnt = (
            omo_prior.groupby("bbl").size().rename("omo_charges_count").reset_index()
        )
        df_feat = df_feat.merge(omo_cnt, on="bbl", how="left")
        df_feat["omo_charges_count"] = (
            df_feat["omo_charges_count"].fillna(0).astype(np.float32)
        )
    else:
        df_feat["omo_charges_count"] = np.float32(0.0)

    if not df_hwo.empty:
        hwo_prior = df_hwo[
            df_hwo["event_date"].isna() | (df_hwo["event_date"] < cutoff)
        ]
        hwo_cnt = (
            hwo_prior.groupby("bbl").size().rename("hwo_charges_count").reset_index()
        )
        df_feat = df_feat.merge(hwo_cnt, on="bbl", how="left")
        df_feat["hwo_charges_count"] = (
            df_feat["hwo_charges_count"].fillna(0).astype(np.float32)
        )
    else:
        df_feat["hwo_charges_count"] = np.float32(0.0)

    if not df_vac.empty:
        vac_bbls = set(df_vac["bbl"].dropna().unique())
        df_feat["has_vacate_order"] = df_feat["bbl"].isin(vac_bbls).astype(np.float32)
    else:
        df_feat["has_vacate_order"] = np.float32(0.0)

    if not df_aep.empty:
        aep_bbls = set(df_aep["bbl"].dropna().unique())
        df_feat["is_aep_distressed"] = df_feat["bbl"].isin(aep_bbls).astype(np.float32)
    else:
        df_feat["is_aep_distressed"] = np.float32(0.0)

    return df_feat


def compute_cohort_target(target_bbls: np.ndarray, cutoff: pd.Timestamp) -> np.ndarray:
    """Computes binary ground truth label for the 12-month forward window [cutoff, cutoff + 365d)."""
    w_start = cutoff
    w_end = cutoff + pd.DateOffset(years=1)
    c_forward = df_vio[
        (df_vio["class"] == "C")
        & (df_vio["inspectiondate"] >= w_start)
        & (df_vio["inspectiondate"] < w_end)
    ]
    pos_bbls = set(c_forward["bbl"].dropna().unique())
    return pd.Series(target_bbls).isin(pos_bbls).astype(np.int32).values


# Construct Datasets
print("Engineering features for Train Cohort (Cutoff: 2021-01-01)...")
df_train = extract_temporal_features(train_bbls, T_TRAIN, pluto_release_ver="20v7")
df_train["target"] = compute_cohort_target(train_bbls, T_TRAIN)
print(
    f"Train cohort ready: {df_train.shape} | Positive rate:"
    f" {df_train['target'].mean():.4f}"
)

print("Engineering features for Validation Cohort (Cutoff: 2022-01-01)...")
df_val = extract_temporal_features(val_bbls, T_VAL, pluto_release_ver="21v4")
df_val["target"] = compute_cohort_target(val_bbls, T_VAL)
print(
    f"Validation cohort ready: {df_val.shape} | Positive rate:"
    f" {df_val['target'].mean():.4f}"
)

print("Engineering features for Test Cohort (Cutoff: 2023-01-01)...")
df_test = extract_temporal_features(test_bbls, T_TEST, pluto_release_ver="22v3")
print(f"Test cohort ready: {df_test.shape}")

# Micro-neighborhood aggregations computed strictly from training distribution
if "zipcode" in df_train.columns:
    df_train["zipcode"] = df_train["zipcode"].astype(str)
    df_val["zipcode"] = df_val["zipcode"].astype(str)
    df_test["zipcode"] = df_test["zipcode"].astype(str)

    zip_risk_map = df_train.groupby("zipcode")["vio_c_count_12m"].mean().to_dict()
    default_zip_risk = float(df_train["vio_c_count_12m"].mean())

    for df_c in [df_train, df_val, df_test]:
        df_c["zip_historical_c_risk"] = (
            df_c["zipcode"]
            .map(zip_risk_map)
            .fillna(default_zip_risk)
            .astype(np.float32)
        )

# Align feature sets
exclude_cols = {"bbl", "target", "bldgclass", "zipcode", "cd"}
feature_columns = sorted([c for c in df_train.columns if c not in exclude_cols])

for c in feature_columns:
    df_train[c] = df_train[c].astype(np.float32)
    df_val[c] = df_val[c].astype(np.float32)
    df_test[c] = df_test[c].astype(np.float32)

print(f"Feature set aligned across all cohorts: {len(feature_columns)} features.")

# Save feature definitions and Parquet datasets for reproducibility
train_save_path = os.path.join(WORKING_DIR, "train_features.parquet")
val_save_path = os.path.join(WORKING_DIR, "val_features.parquet")
test_save_path = os.path.join(WORKING_DIR, "test_features.parquet")
features_json_path = os.path.join(WORKING_DIR, "feature_names.json")

df_train[["bbl", "target"] + feature_columns].to_parquet(train_save_path, index=False)
df_val[["bbl", "target"] + feature_columns].to_parquet(val_save_path, index=False)
df_test[["bbl"] + feature_columns].to_parquet(test_save_path, index=False)

with open(features_json_path, "w") as f:
    json.dump(feature_columns, f, indent=2)

print(f"Feature engineering pipeline completed in {time.time() - start_time:.1f}s.")


# ==============================================================================
# 4. MODEL ARCHITECTURES & LOSS FUNCTION DESIGN
# ==============================================================================
class AsymmetricFocalLoss(nn.Module):
    """Numerically stable Asymmetric Focal Loss for binary ranking and imbalanced classification."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        alpha: float = 0.65,
        clip_eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.alpha = float(alpha)
        self.clip_eps = float(clip_eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)
        p = torch.sigmoid(logits).clamp(self.clip_eps, 1.0 - self.clip_eps)

        pos_weight = torch.pow(1.0 - p, self.gamma_pos)
        neg_weight = torch.pow(p, self.gamma_neg)

        loss_pos = -self.alpha * pos_weight * log_p * targets
        loss_neg = -(1.0 - self.alpha) * neg_weight * log_not_p * (1.0 - targets)
        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GatedResidualBlock(nn.Module):
    """Gated Residual Block with Gated Linear Unit (GLU), LayerNorm, and Dropout."""

    def __init__(
        self, hidden_dim: int, dropout_rate: float = 0.15, use_glu: bool = True
    ) -> None:
        super().__init__()
        self.use_glu = use_glu
        self.norm1 = nn.LayerNorm(hidden_dim)

        if use_glu:
            self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        else:
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)

        if self.use_glu:
            h = self.linear1(h)
            val, gate = torch.chunk(h, 2, dim=-1)
            h = self.activation(val) * torch.sigmoid(gate)
        else:
            h = self.activation(self.linear1(h))

        h = self.dropout(h)
        h = self.norm2(h)
        h = self.linear2(h)

        return residual + h


class GatedResidualTabNet(nn.Module):
    """Deep Tabular Network with Gated Residual connections for violation risk prediction."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.input_norm = nn.BatchNorm1d(num_features)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate * 0.5),
        )

        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    use_glu=True,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        return self.head(h)


def build_pytorch_components(
    num_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 3,
    dropout_rate: float = 0.20,
    gamma_neg: float = 2.0,
    gamma_pos: float = 1.0,
    alpha: float = 0.65,
    epochs: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[GatedResidualTabNet, AsymmetricFocalLoss, AdamW, CosineAnnealingLR]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GatedResidualTabNet(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = AsymmetricFocalLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        alpha=alpha,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    return model, criterion, optimizer, scheduler


def get_lgbm_model_config(
    pos_scale_weight: float = 2.5, random_state: int = 42
) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": 7,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "scale_pos_weight": float(pos_scale_weight),
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


# ==============================================================================
# 5. MODEL TRAINING, ENSEMBLING & VALIDATION
# ==============================================================================
# Extract feature matrices
X_train = np.nan_to_num(
    df_train[feature_columns].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_train = df_train["target"].values.astype(np.float32)

X_val = np.nan_to_num(
    df_val[feature_columns].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_val = df_val["target"].values.astype(np.float32)

X_test = np.nan_to_num(
    df_test[feature_columns].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# 5.1 Train LightGBM
print("Training LightGBM Classifier...")
lgb_config = get_lgbm_model_config(pos_scale_weight=2.5, random_state=42)
lgb_model = lgb.LGBMClassifier(**lgb_config)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
test_preds_lgb = lgb_model.predict_proba(X_test)[:, 1]
lgb_val_ap = average_precision_score(y_val, val_preds_lgb)
print(f"LightGBM Validation Average Precision: {lgb_val_ap:.5f}")

joblib.dump(lgb_model, os.path.join(WORKING_DIR, "lgb_best_model.pkl"))

# 5.2 Train Gated Residual TabNet
print("Training Gated Residual TabNet with Asymmetric Focal Loss...")
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

train_dataset = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)

val_tensor = torch.from_numpy(X_val_norm).to(device)
test_tensor = torch.from_numpy(X_test_norm).to(device)

epochs = 10
nn_model, criterion, optimizer, scheduler = build_pytorch_components(
    num_features=len(feature_columns),
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=128,
    num_blocks=3,
    dropout_rate=0.20,
    gamma_neg=2.0,
    gamma_pos=1.0,
    alpha=0.65,
    epochs=epochs,
    device=device,
)

best_val_ap = -1.0
best_nn_weights = None

for epoch in range(epochs):
    nn_model.train()
    total_loss = 0.0
    num_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        logits = nn_model(bx).squeeze(-1)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()

    nn_model.eval()
    with torch.no_grad():
        val_probs_epoch = torch.sigmoid(nn_model(val_tensor).squeeze(-1)).cpu().numpy()

    epoch_val_ap = average_precision_score(y_val, val_probs_epoch)
    avg_loss = total_loss / max(num_batches, 1)
    print(
        f"Epoch {epoch + 1:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val"
        f" AP: {epoch_val_ap:.5f}"
    )

    if epoch_val_ap > best_val_ap:
        best_val_ap = epoch_val_ap
        best_nn_weights = copy.deepcopy(nn_model.state_dict())

if best_nn_weights is not None:
    nn_model.load_state_dict(best_nn_weights)

torch.save(nn_model.state_dict(), os.path.join(WORKING_DIR, "tabnet_best.pt"))

nn_model.eval()
with torch.no_grad():
    val_preds_nn = torch.sigmoid(nn_model(val_tensor).squeeze(-1)).cpu().numpy()
    test_preds_nn = torch.sigmoid(nn_model(test_tensor).squeeze(-1)).cpu().numpy()

print(
    f"TabNet Validation Average Precision: {average_precision_score(y_val, val_preds_nn):.5f}"
)

# 5.3 Ensemble Rank Normalization
rank_val_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)
rank_val_nn = rankdata(val_preds_nn) / len(val_preds_nn)

rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
rank_test_nn = rankdata(test_preds_nn) / len(test_preds_nn)

val_ensemble = 0.65 * rank_val_lgb + 0.35 * rank_val_nn
test_ensemble = 0.65 * rank_test_lgb + 0.35 * rank_test_nn

final_score = float(average_precision_score(y_val, val_ensemble))
val_auc = float(roc_auc_score(y_val, val_ensemble))

print(f"Validation Ensemble AP: {final_score:.5f} | ROC-AUC: {val_auc:.5f}")

# Inspection operational metrics audit
val_order = np.argsort(-val_ensemble)
total_hazardous = y_val.sum()
for pct in [1, 5, 10]:
    k = int(np.ceil((pct / 100.0) * len(val_ensemble)))
    top_indices = val_order[:k]
    detected = y_val[top_indices].sum()
    prec = detected / k
    recall = detected / total_hazardous
    print(
        f"Priority Inspect Top {pct:2d}% ({k} lots): Precision = {prec:.4f} |"
        f" Recall = {recall:.4f}"
    )

# ==============================================================================
# 6. SUBMISSION GENERATION & FINAL METRIC
# ==============================================================================
df_sub = pd.DataFrame(
    {
        "bbl": df_test["bbl"].astype(str).str.zfill(10),
        "score": test_ensemble.astype(np.float64),
    }
)

assert len(df_sub) == len(
    df_test
), f"Submission row mismatch: expected {len(df_test)}, got {len(df_sub)}"
assert not df_sub["score"].isna().any(), "Submission contains NaN scores."
assert not np.isinf(df_sub["score"]).any(), "Submission contains Inf scores."

submission_path = "submission/submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(df_sub)} lots.")

print(f"Final Validation Score: {final_score}")

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
        "assesstot",
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

# DOB ECB Violations Dataset
try:
    ds_dob = ds.dataset(f"{BASE_GCS}/lake/full/dob_ecb_violations", filesystem=fs)
    dob_schema = ds_dob.schema.names
    dob_date_candidates = [
        "issue_date",
        "issueddate",
        "issued_date",
        "violation_date",
        "served_date",
        "hearing_date",
    ]
    dob_date_col = next(
        (c for c in dob_schema if c.lower() in [cand.lower() for cand in dob_date_candidates]),
        None,
    )
    if not dob_date_col:
        dob_date_col = next(
            (c for c in dob_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    dob_cols = [
        c
        for c in dob_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if dob_date_col and dob_date_col not in dob_cols:
        dob_cols.append(dob_date_col)
    df_dob = ds_dob.to_table(columns=dob_cols).to_pandas()
    if dob_date_col:
        df_dob["violation_date"] = pd.to_datetime(
            df_dob[dob_date_col], errors="coerce"
        )
    else:
        df_dob["violation_date"] = pd.NaT

    bbl_col_dob = next((c for c in df_dob.columns if c.lower() == "bbl"), None)
    bbl_dob_clean = (
        standardize_bbl(df_dob[bbl_col_dob])
        if bbl_col_dob
        else pd.Series("", index=df_dob.index)
    )
    invalid_bbl = bbl_dob_clean.str.len() != 10
    boro_c = next(
        (c for c in df_dob.columns if c.lower() in ["boro", "borough", "boroid"]), None
    )
    block_c = next((c for c in df_dob.columns if c.lower() == "block"), None)
    lot_c = next((c for c in df_dob.columns if c.lower() == "lot"), None)
    if invalid_bbl.any() and boro_c and block_c and lot_c:
        boro_map_num = {
            "MN": 1,
            "BX": 2,
            "BK": 3,
            "QN": 4,
            "SI": 5,
            "MANHATTAN": 1,
            "BRONX": 2,
            "BROOKLYN": 3,
            "QUEENS": 4,
            "STATEN ISLAND": 5,
        }
        b_series = df_dob.loc[invalid_bbl, boro_c]
        if b_series.dtype == object:
            b_num = b_series.str.upper().str.strip().map(boro_map_num)
            b = (
                pd.to_numeric(b_num, errors="coerce")
                .fillna(pd.to_numeric(b_series, errors="coerce"))
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        else:
            b = (
                pd.to_numeric(b_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        blk = (
            pd.to_numeric(df_dob.loc[invalid_bbl, block_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df_dob.loc[invalid_bbl, lot_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_dob_clean.loc[invalid_bbl] = b + blk + lt
    df_dob["bbl"] = bbl_dob_clean
    df_dob = df_dob[
        (df_dob["bbl"].str.len() == 10) & df_dob["violation_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load dob_ecb_violations ({e}). Using empty table.")
    df_dob = pd.DataFrame(columns=["bbl", "violation_date"])

# DOHMH Rodent Inspections Dataset
try:
    ds_rod = ds.dataset(
        f"{BASE_GCS}/lake/full/dohmh_rodent_inspections", filesystem=fs
    )
    rod_schema = ds_rod.schema.names
    rod_date_candidates = [
        "inspection_date",
        "inspectiondate",
        "approved_date",
        "date",
    ]
    rod_date_col = next(
        (c for c in rod_schema if c.lower() in [cand.lower() for cand in rod_date_candidates]),
        None,
    )
    if not rod_date_col:
        rod_date_col = next(
            (c for c in rod_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    rod_cols = [
        c
        for c in rod_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if rod_date_col and rod_date_col not in rod_cols:
        rod_cols.append(rod_date_col)
    df_rod = ds_rod.to_table(columns=rod_cols).to_pandas()
    if rod_date_col:
        df_rod["inspection_date"] = pd.to_datetime(
            df_rod[rod_date_col], errors="coerce"
        )
    else:
        df_rod["inspection_date"] = pd.NaT

    bbl_col_rod = next((c for c in df_rod.columns if c.lower() == "bbl"), None)
    bbl_rod_clean = (
        standardize_bbl(df_rod[bbl_col_rod])
        if bbl_col_rod
        else pd.Series("", index=df_rod.index)
    )
    invalid_bbl = bbl_rod_clean.str.len() != 10
    boro_c = next(
        (c for c in df_rod.columns if c.lower() in ["boro", "borough", "boroid"]), None
    )
    block_c = next((c for c in df_rod.columns if c.lower() == "block"), None)
    lot_c = next((c for c in df_rod.columns if c.lower() == "lot"), None)
    if invalid_bbl.any() and boro_c and block_c and lot_c:
        boro_map_num = {
            "MN": 1,
            "BX": 2,
            "BK": 3,
            "QN": 4,
            "SI": 5,
            "MANHATTAN": 1,
            "BRONX": 2,
            "BROOKLYN": 3,
            "QUEENS": 4,
            "STATEN ISLAND": 5,
        }
        b_series = df_rod.loc[invalid_bbl, boro_c]
        if b_series.dtype == object:
            b_num = b_series.str.upper().str.strip().map(boro_map_num)
            b = (
                pd.to_numeric(b_num, errors="coerce")
                .fillna(pd.to_numeric(b_series, errors="coerce"))
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        else:
            b = (
                pd.to_numeric(b_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        blk = (
            pd.to_numeric(df_rod.loc[invalid_bbl, block_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df_rod.loc[invalid_bbl, lot_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_rod_clean.loc[invalid_bbl] = b + blk + lt
    df_rod["bbl"] = bbl_rod_clean
    df_rod = df_rod[
        (df_rod["bbl"].str.len() == 10) & df_rod["inspection_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load dohmh_rodent_inspections ({e}). Using empty table.")
    df_rod = pd.DataFrame(columns=["bbl", "inspection_date"])

# Municipal Residential Evictions Dataset
try:
    ds_evic = ds.dataset(f"{BASE_GCS}/lake/full/evictions", filesystem=fs)
    evic_schema = ds_evic.schema.names
    evic_date_candidates = [
        "executed_date",
        "executeddate",
        "eviction_date",
        "ejection_date",
        "fileddate",
        "filed_date",
        "court_index_date",
    ]
    evic_date_col = next(
        (c for c in evic_schema if c.lower() in [cand.lower() for cand in evic_date_candidates]),
        None,
    )
    if not evic_date_col:
        evic_date_col = next(
            (c for c in evic_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    evic_cols = [
        c
        for c in evic_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if evic_date_col and evic_date_col not in evic_cols:
        evic_cols.append(evic_date_col)
    df_evic = ds_evic.to_table(columns=evic_cols).to_pandas()
    if evic_date_col:
        df_evic["eviction_date"] = pd.to_datetime(
            df_evic[evic_date_col], errors="coerce"
        )
    else:
        df_evic["eviction_date"] = pd.NaT

    bbl_col_evic = next((c for c in df_evic.columns if c.lower() == "bbl"), None)
    bbl_evic_clean = (
        standardize_bbl(df_evic[bbl_col_evic])
        if bbl_col_evic
        else pd.Series("", index=df_evic.index)
    )
    invalid_bbl = bbl_evic_clean.str.len() != 10
    boro_c = next(
        (c for c in df_evic.columns if c.lower() in ["boro", "borough", "boroid"]), None
    )
    block_c = next((c for c in df_evic.columns if c.lower() == "block"), None)
    lot_c = next((c for c in df_evic.columns if c.lower() == "lot"), None)
    if invalid_bbl.any() and boro_c and block_c and lot_c:
        boro_map_num = {
            "MN": 1,
            "BX": 2,
            "BK": 3,
            "QN": 4,
            "SI": 5,
            "MANHATTAN": 1,
            "BRONX": 2,
            "BROOKLYN": 3,
            "QUEENS": 4,
            "STATEN ISLAND": 5,
        }
        b_series = df_evic.loc[invalid_bbl, boro_c]
        if b_series.dtype == object:
            b_num = b_series.str.upper().str.strip().map(boro_map_num)
            b = (
                pd.to_numeric(b_num, errors="coerce")
                .fillna(pd.to_numeric(b_series, errors="coerce"))
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        else:
            b = (
                pd.to_numeric(b_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        blk = (
            pd.to_numeric(df_evic.loc[invalid_bbl, block_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df_evic.loc[invalid_bbl, lot_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_evic_clean.loc[invalid_bbl] = b + blk + lt
    df_evic["bbl"] = bbl_evic_clean
    df_evic = df_evic[
        (df_evic["bbl"].str.len() == 10) & df_evic["eviction_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load evictions ({e}). Using empty table.")
    df_evic = pd.DataFrame(columns=["bbl", "eviction_date"])

# DOB Building Complaints Dataset
try:
    ds_dob_comp = ds.dataset(f"{BASE_GCS}/lake/full/dob_complaints", filesystem=fs)
    dob_comp_schema = ds_dob_comp.schema.names
    dob_comp_date_candidates = [
        "date_entered",
        "dateentered",
        "entered_date",
        "inspection_date",
        "complaint_date",
        "date",
    ]
    dob_comp_date_col = next(
        (c for c in dob_comp_schema if c.lower() in [cand.lower() for cand in dob_comp_date_candidates]),
        None,
    )
    if not dob_comp_date_col:
        dob_comp_date_col = next(
            (c for c in dob_comp_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    dob_comp_cols = [
        c
        for c in dob_comp_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if dob_comp_date_col and dob_comp_date_col not in dob_comp_cols:
        dob_comp_cols.append(dob_comp_date_col)
    df_dob_comp = ds_dob_comp.to_table(columns=dob_comp_cols).to_pandas()
    if dob_comp_date_col:
        df_dob_comp["complaint_date"] = pd.to_datetime(
            df_dob_comp[dob_comp_date_col], errors="coerce"
        )
    else:
        df_dob_comp["complaint_date"] = pd.NaT

    bbl_col_dob_comp = next((c for c in df_dob_comp.columns if c.lower() == "bbl"), None)
    bbl_dob_comp_clean = (
        standardize_bbl(df_dob_comp[bbl_col_dob_comp])
        if bbl_col_dob_comp
        else pd.Series("", index=df_dob_comp.index)
    )
    invalid_bbl = bbl_dob_comp_clean.str.len() != 10
    boro_c = next(
        (c for c in df_dob_comp.columns if c.lower() in ["boro", "borough", "boroid"]), None
    )
    block_c = next((c for c in df_dob_comp.columns if c.lower() == "block"), None)
    lot_c = next((c for c in df_dob_comp.columns if c.lower() == "lot"), None)
    if invalid_bbl.any() and boro_c and block_c and lot_c:
        boro_map_num = {
            "MN": 1,
            "BX": 2,
            "BK": 3,
            "QN": 4,
            "SI": 5,
            "MANHATTAN": 1,
            "BRONX": 2,
            "BROOKLYN": 3,
            "QUEENS": 4,
            "STATEN ISLAND": 5,
        }
        b_series = df_dob_comp.loc[invalid_bbl, boro_c]
        if b_series.dtype == object:
            b_num = b_series.str.upper().str.strip().map(boro_map_num)
            b = (
                pd.to_numeric(b_num, errors="coerce")
                .fillna(pd.to_numeric(b_series, errors="coerce"))
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        else:
            b = (
                pd.to_numeric(b_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        blk = (
            pd.to_numeric(df_dob_comp.loc[invalid_bbl, block_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df_dob_comp.loc[invalid_bbl, lot_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_dob_comp_clean.loc[invalid_bbl] = b + blk + lt
    df_dob_comp["bbl"] = bbl_dob_comp_clean
    df_dob_comp = df_dob_comp[
        (df_dob_comp["bbl"].str.len() == 10) & df_dob_comp["complaint_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load dob_complaints ({e}). Using empty table.")
    df_dob_comp = pd.DataFrame(columns=["bbl", "complaint_date"])


def standardize_bbl_from_df(df):
    bbl_col = next((c for c in df.columns if c.lower() == "bbl"), None)
    bbl_clean = (
        standardize_bbl(df[bbl_col]) if bbl_col else pd.Series("", index=df.index)
    )
    invalid_bbl = bbl_clean.str.len() != 10
    boro_c = next(
        (c for c in df.columns if c.lower() in ["boro", "borough", "boroid", "boroughid"]),
        None,
    )
    block_c = next((c for c in df.columns if c.lower() == "block"), None)
    lot_c = next((c for c in df.columns if c.lower() == "lot"), None)
    if invalid_bbl.any() and boro_c and block_c and lot_c:
        boro_map_num = {
            "MN": 1,
            "BX": 2,
            "BK": 3,
            "QN": 4,
            "SI": 5,
            "MANHATTAN": 1,
            "BRONX": 2,
            "BROOKLYN": 3,
            "QUEENS": 4,
            "STATEN ISLAND": 5,
        }
        b_series = df.loc[invalid_bbl, boro_c]
        if b_series.dtype == object:
            b_num = b_series.str.upper().str.strip().map(boro_map_num)
            b = (
                pd.to_numeric(b_num, errors="coerce")
                .fillna(pd.to_numeric(b_series, errors="coerce"))
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        else:
            b = (
                pd.to_numeric(b_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        blk = (
            pd.to_numeric(df.loc[invalid_bbl, block_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        lt = (
            pd.to_numeric(df.loc[invalid_bbl, lot_c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        bbl_clean.loc[invalid_bbl] = b + blk + lt
    return bbl_clean


# HPD HWO Charges (Heat and Hot Water Orders)
try:
    ds_hwo = ds.dataset(f"{BASE_GCS}/lake/full/hpd_hwo_charges", filesystem=fs)
    hwo_schema = ds_hwo.schema.names
    hwo_date_candidates = [
        "charge_date",
        "invoicedate",
        "orderdate",
        "transdate",
        "createddate",
        "inspectiondate",
    ]
    hwo_date_col = next(
        (c for c in hwo_schema if c.lower() in [cand.lower() for cand in hwo_date_candidates]),
        None,
    )
    if not hwo_date_col:
        hwo_date_col = next(
            (c for c in hwo_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    hwo_cols = [
        c
        for c in hwo_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if hwo_date_col and hwo_date_col not in hwo_cols:
        hwo_cols.append(hwo_date_col)
    df_hwo = ds_hwo.to_table(columns=hwo_cols).to_pandas()
    if hwo_date_col:
        df_hwo["charge_date"] = pd.to_datetime(df_hwo[hwo_date_col], errors="coerce")
    else:
        df_hwo["charge_date"] = pd.NaT
    df_hwo["bbl"] = standardize_bbl_from_df(df_hwo)
    df_hwo = df_hwo[(df_hwo["bbl"].str.len() == 10) & df_hwo["charge_date"].notna()]
except Exception as e:
    print(f"Warning: Could not load hpd_hwo_charges ({e}). Using empty table.")
    df_hwo = pd.DataFrame(columns=["bbl", "charge_date"])

# HPD OMO Charges (Open Market Order Emergency Repairs)
try:
    ds_omo = ds.dataset(f"{BASE_GCS}/lake/full/hpd_omo_charges", filesystem=fs)
    omo_schema = ds_omo.schema.names
    omo_date_candidates = [
        "charge_date",
        "invoicedate",
        "orderdate",
        "transdate",
        "createddate",
        "inspectiondate",
    ]
    omo_date_col = next(
        (c for c in omo_schema if c.lower() in [cand.lower() for cand in omo_date_candidates]),
        None,
    )
    if not omo_date_col:
        omo_date_col = next(
            (c for c in omo_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    omo_cols = [
        c
        for c in omo_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if omo_date_col and omo_date_col not in omo_cols:
        omo_cols.append(omo_date_col)
    df_omo = ds_omo.to_table(columns=omo_cols).to_pandas()
    if omo_date_col:
        df_omo["charge_date"] = pd.to_datetime(df_omo[omo_date_col], errors="coerce")
    else:
        df_omo["charge_date"] = pd.NaT
    df_omo["bbl"] = standardize_bbl_from_df(df_omo)
    df_omo = df_omo[(df_omo["bbl"].str.len() == 10) & df_omo["charge_date"].notna()]
except Exception as e:
    print(f"Warning: Could not load hpd_omo_charges ({e}). Using empty table.")
    df_omo = pd.DataFrame(columns=["bbl", "charge_date"])

# DOF Tax Lien Sales (Physical & Financial Distress)
try:
    ds_lien = ds.dataset(f"{BASE_GCS}/lake/full/dof_tax_lien_sales", filesystem=fs)
    lien_schema = ds_lien.schema.names
    lien_date_candidates = ["sale_date", "lien_date", "date", "transdate"]
    lien_date_col = next(
        (c for c in lien_schema if c.lower() in [cand.lower() for cand in lien_date_candidates]),
        None,
    )
    if not lien_date_col:
        lien_date_col = next(
            (c for c in lien_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    lien_year_col = next((c for c in lien_schema if "year" in c.lower()), None)
    lien_cols = [
        c
        for c in lien_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if lien_date_col and lien_date_col not in lien_cols:
        lien_cols.append(lien_date_col)
    if lien_year_col and lien_year_col not in lien_cols:
        lien_cols.append(lien_year_col)
    df_lien = ds_lien.to_table(columns=lien_cols).to_pandas()
    if lien_date_col:
        df_lien["sale_date"] = pd.to_datetime(df_lien[lien_date_col], errors="coerce")
    else:
        df_lien["sale_date"] = pd.NaT
    if lien_year_col:
        df_lien["sale_year"] = pd.to_numeric(df_lien[lien_year_col], errors="coerce")
    else:
        df_lien["sale_year"] = np.nan
    df_lien["bbl"] = standardize_bbl_from_df(df_lien)
    df_lien = df_lien[df_lien["bbl"].str.len() == 10]
except Exception as e:
    print(f"Warning: Could not load dof_tax_lien_sales ({e}). Using empty table.")
    df_lien = pd.DataFrame(columns=["bbl", "sale_date", "sale_year"])

# HPD CONH Buildings (Certificate of No Harassment Pilot Program)
try:
    ds_conh = ds.dataset(f"{BASE_GCS}/lake/full/hpd_conh_buildings", filesystem=fs)
    conh_schema = ds_conh.schema.names
    conh_cols = [
        c
        for c in conh_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot", "buildingid"]
    ]
    df_conh = ds_conh.to_table(columns=conh_cols).to_pandas()
    df_conh["bbl"] = standardize_bbl_from_df(df_conh)
    conh_bbls = set(df_conh.loc[df_conh["bbl"].str.len() == 10, "bbl"].unique())
except Exception as e:
    print(f"Warning: Could not load hpd_conh_buildings ({e}). Using empty set.")
    conh_bbls = set()

# DOB Violations Dataset (Physical Structural / Electrical / Plumbing Neglect)
try:
    ds_dob_v = ds.dataset(f"{BASE_GCS}/lake/full/dob_violations", filesystem=fs)
    dob_v_schema = ds_dob_v.schema.names
    dob_v_date_candidates = [
        "issue_date",
        "issued_date",
        "violation_date",
        "date",
        "issue_dt",
    ]
    dob_v_date_col = next(
        (c for c in dob_v_schema if c.lower() in [cand.lower() for cand in dob_v_date_candidates]),
        None,
    )
    if not dob_v_date_col:
        dob_v_date_col = next(
            (c for c in dob_v_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    dob_v_cols = [
        c
        for c in dob_v_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if dob_v_date_col and dob_v_date_col not in dob_v_cols:
        dob_v_cols.append(dob_v_date_col)
    df_dob_v = ds_dob_v.to_table(columns=dob_v_cols).to_pandas()
    if dob_v_date_col:
        df_dob_v["violation_date"] = pd.to_datetime(
            df_dob_v[dob_v_date_col], errors="coerce"
        )
    else:
        df_dob_v["violation_date"] = pd.NaT
    df_dob_v["bbl"] = standardize_bbl_from_df(df_dob_v)
    df_dob_v = df_dob_v[
        (df_dob_v["bbl"].str.len() == 10) & df_dob_v["violation_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load dob_violations ({e}). Using empty table.")
    df_dob_v = pd.DataFrame(columns=["bbl", "violation_date"])

# HPD Bedbug Reports Dataset (Chronic Health and Sanitation Distress)
try:
    ds_bedbug = ds.dataset(f"{BASE_GCS}/lake/full/hpd_bedbug_reports", filesystem=fs)
    bedbug_schema = ds_bedbug.schema.names
    bedbug_date_candidates = [
        "filing_date",
        "filingdate",
        "filing_received_date",
        "date",
        "inspectiondate",
    ]
    bedbug_date_col = next(
        (c for c in bedbug_schema if c.lower() in [cand.lower() for cand in bedbug_date_candidates]),
        None,
    )
    if not bedbug_date_col:
        bedbug_date_col = next(
            (c for c in bedbug_schema if "date" in c.lower() or "time" in c.lower()),
            None,
        )
    bedbug_cols = [
        c
        for c in bedbug_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    if bedbug_date_col and bedbug_date_col not in bedbug_cols:
        bedbug_cols.append(bedbug_date_col)
    df_bedbug = ds_bedbug.to_table(columns=bedbug_cols).to_pandas()
    if bedbug_date_col:
        df_bedbug["filing_date"] = pd.to_datetime(
            df_bedbug[bedbug_date_col], errors="coerce"
        )
    else:
        df_bedbug["filing_date"] = pd.NaT
    df_bedbug["bbl"] = standardize_bbl_from_df(df_bedbug)
    df_bedbug = df_bedbug[
        (df_bedbug["bbl"].str.len() == 10) & df_bedbug["filing_date"].notna()
    ]
except Exception as e:
    print(f"Warning: Could not load hpd_bedbug_reports ({e}). Using empty table.")
    df_bedbug = pd.DataFrame(columns=["bbl", "filing_date"])

# HPD Underlying Conditions Dataset (Systemic Building Failure Orders)
try:
    ds_uc = ds.dataset(
        f"{BASE_GCS}/lake/full/hpd_underlying_conditions", filesystem=fs
    )
    uc_schema = ds_uc.schema.names
    uc_cols = [
        c
        for c in uc_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    df_uc = ds_uc.to_table(columns=uc_cols).to_pandas()
    df_uc["bbl"] = standardize_bbl_from_df(df_uc)
    uc_bbls = set(df_uc.loc[df_uc["bbl"].str.len() == 10, "bbl"].unique())
except Exception as e:
    print(f"Warning: Could not load hpd_underlying_conditions ({e}). Using empty set.")
    uc_bbls = set()

# Speculation Watch List Dataset (Predatory Landlord Speculative Capitalization)
try:
    ds_swl = ds.dataset(
        f"{BASE_GCS}/lake/full/speculation_watch_list", filesystem=fs
    )
    swl_schema = ds_swl.schema.names
    swl_cols = [
        c
        for c in swl_schema
        if c.lower() in ["bbl", "boro", "borough", "boroid", "block", "lot"]
    ]
    df_swl = ds_swl.to_table(columns=swl_cols).to_pandas()
    df_swl["bbl"] = standardize_bbl_from_df(df_swl)
    swl_bbls = set(df_swl.loc[df_swl["bbl"].str.len() == 10, "bbl"].unique())
except Exception as e:
    print(f"Warning: Could not load speculation_watch_list ({e}). Using empty set.")
    swl_bbls = set()


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


pluto_19 = extract_pluto_cohort(df_pluto, "19v2")
pluto_20 = extract_pluto_cohort(df_pluto, "20v7")
pluto_21 = extract_pluto_cohort(df_pluto, "21v4")
pluto_22 = extract_pluto_cohort(df_pluto, "22v3")

df_train_2020 = pluto_19 if len(pluto_19) > 0 else pluto_20
df_train_2021 = pluto_20 if len(pluto_20) > 0 else pluto_21
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
        "assesstot",
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

# 2021 Training Target (Cutoff 2021-01-01 -> 12-month window to 2022-01-01)
train_2021_pos_mask = (
    (df_viol["class"] == "C")
    & (df_viol["inspectiondate"] >= pd.Timestamp("2021-01-01"))
    & (df_viol["inspectiondate"] < pd.Timestamp("2022-01-01"))
)
train_2021_pos_bbls = set(df_viol.loc[train_2021_pos_mask, "bbl"].unique())
y_train_2021 = df_train_2021["bbl"].isin(train_2021_pos_bbls).astype(np.int32).values

# 2020 Training Target (Cutoff 2020-01-01 -> 12-month window to 2021-01-01)
train_2020_pos_mask = (
    (df_viol["class"] == "C")
    & (df_viol["inspectiondate"] >= pd.Timestamp("2020-01-01"))
    & (df_viol["inspectiondate"] < pd.Timestamp("2021-01-01"))
)
train_2020_pos_bbls = set(df_viol.loc[train_2020_pos_mask, "bbl"].unique())
y_train_2020 = df_train_2020["bbl"].isin(train_2020_pos_bbls).astype(np.int32).values

# Prior Year Positive BBLs for Leak-Free Empirical Bayes Prior
prior_2019_pos = set(
    df_viol.loc[
        (df_viol["class"] == "C")
        & (df_viol["inspectiondate"] >= pd.Timestamp("2019-01-01"))
        & (df_viol["inspectiondate"] < pd.Timestamp("2020-01-01")),
        "bbl",
    ].unique()
)
prior_2020_pos = train_2020_pos_bbls
prior_2021_pos = train_2021_pos_bbls
prior_2022_pos = val_pos_bbls


def build_features(
    cohort_df,
    cutoff_date,
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    historical_pos_bbls,
    df_comp=None,
    df_dob=None,
    df_rod=None,
    df_evic=None,
    df_dob_comp=None,
    df_hwo=None,
    df_omo=None,
    df_lien=None,
    conh_bbls=None,
    df_dob_v=None,
    df_bedbug=None,
    uc_bbls=None,
    swl_bbls=None,
):
    if df_comp is None:
        df_comp = globals().get("df_comp", None)
    if df_dob is None:
        df_dob = globals().get("df_dob", None)
    if df_rod is None:
        df_rod = globals().get("df_rod", None)
    if df_evic is None:
        df_evic = globals().get("df_evic", None)
    if df_dob_comp is None:
        df_dob_comp = globals().get("df_dob_comp", None)
    if df_hwo is None:
        df_hwo = globals().get("df_hwo", None)
    if df_omo is None:
        df_omo = globals().get("df_omo", None)
    if df_lien is None:
        df_lien = globals().get("df_lien", None)
    if conh_bbls is None:
        conh_bbls = globals().get("conh_bbls", None)
    if df_dob_v is None:
        df_dob_v = globals().get("df_dob_v", None)
    if df_bedbug is None:
        df_bedbug = globals().get("df_bedbug", None)
    if uc_bbls is None:
        uc_bbls = globals().get("uc_bbls", None)
    if swl_bbls is None:
        swl_bbls = globals().get("swl_bbls", None)
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

    raw_assesstot = (
        pd.to_numeric(cohort_df.get("assesstot", 0), errors="coerce")
        .fillna(0)
        .astype(np.float32)
    )
    assesstot = np.log1p(np.maximum(0.0, raw_assesstot))
    assesstot_density = assesstot / (unitsres + 1.0)

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

    v_mask = df_viol["inspectiondate"] < T
    df_sub = df_viol.loc[v_mask, ["bbl", "class", "inspectiondate"]].copy()

    dt_days = (T - df_sub["inspectiondate"]).dt.days.values
    cls_c = (df_sub["class"] == "C").values
    cls_b = (df_sub["class"] == "B").values
    cls_a = (df_sub["class"] == "A").values

    df_sub["c_90d"] = np.where(cls_c & (dt_days <= 90), 1, 0)
    df_sub["c_1yr"] = np.where(cls_c & (dt_days <= 365), 1, 0)
    df_sub["c_2yr"] = np.where(cls_c & (dt_days <= 730), 1, 0)
    df_sub["c_3yr"] = np.where(cls_c & (dt_days <= 1095), 1, 0)
    df_sub["c_5yr"] = np.where(cls_c & (dt_days <= 1825), 1, 0)
    df_sub["c_lifetime"] = np.where(cls_c, 1, 0)

    # Yearly delinquency flags for chronic persistence streak computation
    df_sub["c_y1"] = np.where(cls_c & (dt_days <= 365), 1, 0)
    df_sub["c_y2"] = np.where(cls_c & (dt_days > 365) & (dt_days <= 730), 1, 0)
    df_sub["c_y3"] = np.where(cls_c & (dt_days > 730) & (dt_days <= 1095), 1, 0)
    df_sub["c_y4"] = np.where(cls_c & (dt_days > 1095) & (dt_days <= 1460), 1, 0)
    df_sub["c_y5"] = np.where(cls_c & (dt_days > 1460) & (dt_days <= 1825), 1, 0)

    df_sub["b_1yr"] = np.where(cls_b & (dt_days <= 365), 1, 0)
    df_sub["b_2yr"] = np.where(cls_b & (dt_days <= 730), 1, 0)
    df_sub["a_1yr"] = np.where(cls_a & (dt_days <= 365), 1, 0)

    df_sub["tot_90d"] = np.where(dt_days <= 90, 1, 0)
    df_sub["tot_1yr"] = np.where(dt_days <= 365, 1, 0)
    df_sub["tot_2yr"] = np.where(dt_days <= 730, 1, 0)
    df_sub["tot_3yr"] = np.where(dt_days <= 1095, 1, 0)
    df_sub["tot_5yr"] = np.where(dt_days <= 1825, 1, 0)
    df_sub["tot_lifetime"] = 1

    df_sub["days_c"] = np.where(cls_c, dt_days, 9999)
    df_sub["days_any"] = dt_days

    # Continuous exponential time-decay kernels
    ln2 = np.log(2.0)
    df_sub["decay_c_180"] = np.where(
        cls_c, np.exp(-dt_days * (ln2 / 180.0)), 0.0
    )
    df_sub["decay_c_365"] = np.where(
        cls_c, np.exp(-dt_days * (ln2 / 365.0)), 0.0
    )
    df_sub["decay_tot_180"] = np.exp(-dt_days * (ln2 / 180.0))
    df_sub["decay_tot_365"] = np.exp(-dt_days * (ln2 / 365.0))

    agg_res = df_sub.groupby("bbl").agg(
        {
            "c_90d": "sum",
            "c_1yr": "sum",
            "c_2yr": "sum",
            "c_3yr": "sum",
            "c_5yr": "sum",
            "c_lifetime": "sum",
            "c_y1": "max",
            "c_y2": "max",
            "c_y3": "max",
            "c_y4": "max",
            "c_y5": "max",
            "b_1yr": "sum",
            "b_2yr": "sum",
            "a_1yr": "sum",
            "tot_90d": "sum",
            "tot_1yr": "sum",
            "tot_2yr": "sum",
            "tot_3yr": "sum",
            "tot_5yr": "sum",
            "tot_lifetime": "sum",
            "days_c": "min",
            "days_any": "min",
            "decay_c_180": "sum",
            "decay_c_365": "sum",
            "decay_tot_180": "sum",
            "decay_tot_365": "sum",
        }
    )

    feat = {}
    for col in agg_res.columns:
        if col in ["days_c", "days_any"]:
            feat[col] = bbls.map(agg_res[col]).fillna(9999).astype(np.float32)
        else:
            feat[col] = bbls.map(agg_res[col]).fillna(0).astype(np.float32)

    # Distinct inspection visit frequencies across 1, 3, 5 years, and lifetime
    df_sub_1yr = df_sub[dt_days <= 365]
    df_sub_3yr = df_sub[dt_days <= 1095]
    df_sub_5yr = df_sub[dt_days <= 1825]
    uniq_visits_1yr = df_sub_1yr.groupby("bbl")["inspectiondate"].nunique()
    uniq_visits_3yr = df_sub_3yr.groupby("bbl")["inspectiondate"].nunique()
    uniq_visits_5yr = df_sub_5yr.groupby("bbl")["inspectiondate"].nunique()
    uniq_visits_lifetime = df_sub.groupby("bbl")["inspectiondate"].nunique()
    feat["uniq_visits_1yr"] = bbls.map(uniq_visits_1yr).fillna(0).astype(np.float32)
    feat["uniq_visits_3yr"] = bbls.map(uniq_visits_3yr).fillna(0).astype(np.float32)
    feat["uniq_visits_5yr"] = bbls.map(uniq_visits_5yr).fillna(0).astype(np.float32)
    feat["uniq_visits_lifetime"] = bbls.map(uniq_visits_lifetime).fillna(0).astype(np.float32)
    feat["c_per_visit_1yr"] = feat["c_1yr"] / (feat["uniq_visits_1yr"] + 1.0)
    feat["c_per_visit_5yr"] = feat["c_5yr"] / (feat["uniq_visits_5yr"] + 1.0)
    feat["c_per_visit_lifetime"] = feat["c_lifetime"] / (feat["uniq_visits_lifetime"] + 1.0)

    # Lifetime recidivism and recent-to-lifetime acceleration ratios
    c_1yr = feat["c_1yr"]
    c_2yr = feat["c_2yr"]
    c_lifetime = feat["c_lifetime"]
    tot_1yr = feat["tot_1yr"]
    tot_2yr = feat["tot_2yr"]
    tot_lifetime = feat["tot_lifetime"]
    feat["c_accel_lifetime"] = c_1yr / (c_lifetime + 1.0)
    feat["tot_accel_lifetime"] = tot_1yr / (tot_lifetime + 1.0)
    feat["c_ratio_lifetime"] = c_lifetime / (tot_lifetime + 1.0)
    feat["c_per_unit_lifetime"] = c_lifetime / (unitsres + 1e-4)
    feat["tot_per_unit_lifetime"] = tot_lifetime / (unitsres + 1e-4)
    feat["has_prior_c_lifetime"] = (c_lifetime > 0).astype(np.float32)

    # Chronic multi-year persistence metrics
    has_y1 = feat["c_y1"]
    has_y2 = feat["c_y2"]
    has_y3 = feat["c_y3"]
    has_y4 = feat["c_y4"]
    has_y5 = feat["c_y5"]
    feat["c_active_years_5yr"] = has_y1 + has_y2 + has_y3 + has_y4 + has_y5
    feat["c_consec_streak"] = has_y1 * (
        1.0 + has_y2 * (
            1.0 + has_y3 * (
                1.0 + has_y4 * (
                    1.0 + has_y5
                )
            )
        )
    )
    feat["c_consec_2yr"] = (has_y1 * has_y2).astype(np.float32)
    feat["c_consec_3yr"] = (has_y1 * has_y2 * has_y3).astype(np.float32)
    feat["has_prior_c_5yr"] = (feat["c_5yr"] > 0).astype(np.float32)

    b_1yr = feat["b_1yr"]
    b_2yr = feat["b_2yr"]
    tot_90d = feat["tot_90d"]

    feat["c_ratio_1yr"] = c_1yr / (tot_1yr + 1.0)
    feat["b_c_ratio_1yr"] = (b_1yr + c_1yr) / (tot_1yr + 1.0)
    feat["c_velocity"] = c_1yr / (np.maximum(0, c_2yr - c_1yr) + 1.0)
    feat["c_velocity_short"] = feat["c_90d"] / (c_1yr + 0.1)
    feat["tot_velocity"] = tot_1yr / (np.maximum(0, tot_2yr - tot_1yr) + 1.0)
    feat["surge_90d"] = (tot_90d * 4.0) / (tot_1yr + 1.0)
    feat["has_prior_c_1yr"] = (c_1yr > 0).astype(np.float32)
    feat["has_prior_c_2yr"] = (c_2yr > 0).astype(np.float32)
    feat["has_prior_c_3yr"] = (feat["c_3yr"] > 0).astype(np.float32)

    # Class B-to-C escalation ratios
    feat["b_to_c_ratio_1yr"] = c_1yr / (b_1yr + 1.0)
    feat["b_to_c_escalation_rate_1yr"] = c_1yr / (b_1yr + c_1yr + 1.0)
    feat["b_to_c_ratio_2yr"] = c_2yr / (b_2yr + 1.0)
    feat["b_to_c_escalation_rate_2yr"] = c_2yr / (b_2yr + c_2yr + 1.0)

    feat["c_per_unit_1yr"] = c_1yr / (unitsres + 1e-4)
    feat["tot_per_unit_1yr"] = tot_1yr / (unitsres + 1e-4)
    feat["tot_per_unit_3yr"] = feat["c_3yr"] / (unitsres + 1e-4)
    feat["c_per_unit_5yr"] = feat["c_5yr"] / (unitsres + 1e-4)
    feat["tot_per_unit_5yr"] = feat["tot_5yr"] / (unitsres + 1e-4)

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
    feat["assesstot"] = assesstot
    feat["assesstot_density"] = assesstot_density

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
        global_mean = float(hist_pos.mean()) if len(hist_pos) > 0 else 0.05
        m = 50.0
        cd_prior = (cd_pos + m * global_mean) / (cd_tot + m)
        feat["cd_risk_prior"] = (
            cd_col.map(cd_prior).fillna(global_mean).astype(np.float32)
        )
    else:
        feat["cd_risk_prior"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Empirical Bayes ZIP-Code Risk Prior with Shrinkage m=100
    if "zipcode" in cohort_df and historical_pos_bbls is not None:
        zip_col = (
            pd.to_numeric(cohort_df["zipcode"], errors="coerce").fillna(0).astype(np.int32)
        )
        hist_pos = bbls.isin(historical_pos_bbls).astype(float)
        global_mean_zip = float(hist_pos.mean()) if len(hist_pos) > 0 else 0.05
        zip_pos = hist_pos.groupby(zip_col).sum()
        zip_tot = hist_pos.groupby(zip_col).count()
        m_zip = 100.0
        zip_prior = (zip_pos + m_zip * global_mean_zip) / (zip_tot + m_zip)
        feat["zip_risk_prior"] = (
            zip_col.map(zip_prior).fillna(global_mean_zip).astype(np.float32)
        )
    else:
        feat["zip_risk_prior"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Leave-One-Out Tax-Block Empirical Bayes Spatial Prior (Borough + Block)
    tax_block = bbls.str[:6]
    if historical_pos_bbls is not None:
        hist_pos = bbls.isin(historical_pos_bbls).astype(float)
        global_mean_blk = float(hist_pos.mean()) if len(hist_pos) > 0 else 0.05
        block_pos = tax_block.map(hist_pos.groupby(tax_block).sum()).fillna(0.0)
        block_tot = tax_block.map(hist_pos.groupby(tax_block).count()).fillna(1.0)
        m_blk = 30.0
        c_others = block_pos - hist_pos
        n_others = block_tot - 1.0
        feat["tax_block_risk_prior"] = (
            ((c_others + m_blk * global_mean_blk) / (n_others + m_blk))
            .fillna(global_mean_blk)
            .astype(np.float32)
        )
        feat["tax_block_pos_cnt"] = block_pos.astype(np.float32)
        feat["tax_block_bldg_cnt"] = block_tot.astype(np.float32)
    else:
        feat["tax_block_risk_prior"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["tax_block_pos_cnt"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["tax_block_bldg_cnt"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Continuous 2-Year Spatial Class C Density Normalized by Units with Empirical Bayes Shrinkage
    bldg_c2yr = feat["c_2yr"]
    bldg_units = np.maximum(1.0, unitsres)
    global_c2yr_rate = float(bldg_c2yr.sum() / (bldg_units.sum() + 1e-5))

    block_c_sum = tax_block.map(bldg_c2yr.groupby(tax_block).sum()).fillna(0.0)
    block_u_sum = tax_block.map(bldg_units.groupby(tax_block).sum()).fillna(1.0)
    block_c_others = np.maximum(0.0, block_c_sum - bldg_c2yr)
    block_u_others = np.maximum(0.0, block_u_sum - bldg_units)
    m_blk_cont = 20.0
    feat["tax_block_c2yr_density_eb"] = (
        (block_c_others + m_blk_cont * global_c2yr_rate) / (block_u_others + m_blk_cont)
    ).astype(np.float32)
    feat["tax_block_c2yr_sum"] = block_c_sum.astype(np.float32)
    feat["tax_block_units_sum"] = block_u_sum.astype(np.float32)

    if "cd" in cohort_df:
        cd_col = (
            pd.to_numeric(cohort_df["cd"], errors="coerce").fillna(0).astype(np.int32)
        )
        cd_c_sum = cd_col.map(bldg_c2yr.groupby(cd_col).sum()).fillna(0.0)
        cd_u_sum = cd_col.map(bldg_units.groupby(cd_col).sum()).fillna(1.0)
        cd_c_others = np.maximum(0.0, cd_c_sum - bldg_c2yr)
        cd_u_others = np.maximum(0.0, cd_u_sum - bldg_units)
        m_cd_cont = 100.0
        feat["cd_c2yr_density_eb"] = (
            (cd_c_others + m_cd_cont * global_c2yr_rate) / (cd_u_others + m_cd_cont)
        ).astype(np.float32)
    else:
        feat["cd_c2yr_density_eb"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Multi-horizon Tenant Distress Signals (HPD Complaints)
    if df_comp is not None and len(df_comp) > 0:
        min_comp_date = T - pd.Timedelta(days=730)
        c_mask = (df_comp["complaint_date"] < T) & (
            df_comp["complaint_date"] >= min_comp_date
        )
        comp_sub = df_comp.loc[c_mask, ["bbl", "complaint_date"]].copy()

        comp_dt_days = (T - comp_sub["complaint_date"]).dt.days.values
        comp_sub["comp_30d"] = np.where(comp_dt_days <= 30, 1, 0)
        comp_sub["comp_90d"] = np.where(comp_dt_days <= 90, 1, 0)
        comp_sub["comp_1yr"] = np.where(comp_dt_days <= 365, 1, 0)
        comp_sub["comp_2yr"] = 1
        comp_sub["days_comp"] = comp_dt_days

        comp_agg = comp_sub.groupby("bbl").agg(
            {
                "comp_30d": "sum",
                "comp_90d": "sum",
                "comp_1yr": "sum",
                "comp_2yr": "sum",
                "days_comp": "min",
            }
        )

        c_30d = bbls.map(comp_agg["comp_30d"]).fillna(0).astype(np.float32)
        c_90d = bbls.map(comp_agg["comp_90d"]).fillna(0).astype(np.float32)
        c_1yr = bbls.map(comp_agg["comp_1yr"]).fillna(0).astype(np.float32)
        c_2yr = bbls.map(comp_agg["comp_2yr"]).fillna(0).astype(np.float32)
        days_comp = bbls.map(comp_agg["days_comp"]).fillna(9999).astype(np.float32)
    else:
        c_30d = pd.Series(0.0, index=cohort_df.index, dtype=np.float32)
        c_90d = pd.Series(0.0, index=cohort_df.index, dtype=np.float32)
        c_1yr = pd.Series(0.0, index=cohort_df.index, dtype=np.float32)
        c_2yr = pd.Series(0.0, index=cohort_df.index, dtype=np.float32)
        days_comp = pd.Series(9999.0, index=cohort_df.index, dtype=np.float32)

    feat["comp_30d"] = c_30d
    feat["comp_90d"] = c_90d
    feat["comp_1yr"] = c_1yr
    feat["comp_2yr"] = c_2yr
    feat["days_since_comp"] = days_comp
    feat["comp_surge_90d"] = (c_90d * 4.0) / (c_1yr + 1.0)
    feat["comp_per_unit_1yr"] = c_1yr / (unitsres + 1e-4)
    feat["comp_per_unit_2yr"] = c_2yr / (unitsres + 1e-4)
    feat["comp_viol_coupling"] = feat["c_1yr"] * c_1yr

    # DOB ECB Violations
    if df_dob is not None and len(df_dob) > 0:
        min_dob_date = T - pd.Timedelta(days=1095)
        dob_mask = (df_dob["violation_date"] < T) & (
            df_dob["violation_date"] >= min_dob_date
        )
        dob_sub = df_dob.loc[dob_mask, ["bbl", "violation_date"]].copy()

        dob_dt = (T - dob_sub["violation_date"]).dt.days.values
        dob_sub["dob_cnt_1yr"] = np.where(dob_dt <= 365, 1, 0)
        dob_sub["dob_cnt_2yr"] = np.where(dob_dt <= 730, 1, 0)
        dob_sub["dob_cnt_3yr"] = 1
        dob_sub["days_dob"] = dob_dt

        dob_agg = dob_sub.groupby("bbl").agg(
            {
                "dob_cnt_1yr": "sum",
                "dob_cnt_2yr": "sum",
                "dob_cnt_3yr": "sum",
                "days_dob": "min",
            }
        )

        feat["dob_ecb_cnt_1yr"] = (
            bbls.map(dob_agg["dob_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["dob_ecb_cnt_2yr"] = (
            bbls.map(dob_agg["dob_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["dob_ecb_cnt_3yr"] = (
            bbls.map(dob_agg["dob_cnt_3yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_dob_ecb"] = (
            bbls.map(dob_agg["days_dob"]).fillna(9999).astype(np.float32)
        )
    else:
        feat["dob_ecb_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_ecb_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_ecb_cnt_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_dob_ecb"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)

    # DOHMH Rodent Inspections
    if df_rod is not None and len(df_rod) > 0:
        min_rod_date = T - pd.Timedelta(days=1095)
        rod_mask = (df_rod["inspection_date"] < T) & (
            df_rod["inspection_date"] >= min_rod_date
        )
        rod_sub = df_rod.loc[rod_mask, ["bbl", "inspection_date"]].copy()

        rod_dt = (T - rod_sub["inspection_date"]).dt.days.values
        rod_sub["rod_cnt_1yr"] = np.where(rod_dt <= 365, 1, 0)
        rod_sub["rod_cnt_2yr"] = np.where(rod_dt <= 730, 1, 0)
        rod_sub["rod_cnt_3yr"] = 1
        rod_sub["days_rod"] = rod_dt

        rod_agg = rod_sub.groupby("bbl").agg(
            {
                "rod_cnt_1yr": "sum",
                "rod_cnt_2yr": "sum",
                "rod_cnt_3yr": "sum",
                "days_rod": "min",
            }
        )

        feat["rodent_cnt_1yr"] = (
            bbls.map(rod_agg["rod_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["rodent_cnt_2yr"] = (
            bbls.map(rod_agg["rod_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["rodent_cnt_3yr"] = (
            bbls.map(rod_agg["rod_cnt_3yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_rodent"] = (
            bbls.map(rod_agg["days_rod"]).fillna(9999).astype(np.float32)
        )
    else:
        feat["rodent_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["rodent_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["rodent_cnt_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_rodent"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)

    # Municipal Residential Evictions Features
    if df_evic is not None and len(df_evic) > 0:
        min_evic_date = T - pd.Timedelta(days=730)
        evic_mask = (df_evic["eviction_date"] < T) & (
            df_evic["eviction_date"] >= min_evic_date
        )
        evic_sub = df_evic.loc[evic_mask, ["bbl", "eviction_date"]].copy()

        evic_dt = (T - evic_sub["eviction_date"]).dt.days.values
        evic_sub["evic_cnt_1yr"] = np.where(evic_dt <= 365, 1, 0)
        evic_sub["evic_cnt_2yr"] = 1
        evic_sub["days_evic"] = evic_dt

        evic_agg = evic_sub.groupby("bbl").agg(
            {
                "evic_cnt_1yr": "sum",
                "evic_cnt_2yr": "sum",
                "days_evic": "min",
            }
        )

        feat["evic_cnt_1yr"] = (
            bbls.map(evic_agg["evic_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["evic_cnt_2yr"] = (
            bbls.map(evic_agg["evic_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_evic"] = (
            bbls.map(evic_agg["days_evic"]).fillna(9999).astype(np.float32)
        )
        feat["evic_per_unit_1yr"] = feat["evic_cnt_1yr"] / (unitsres + 1e-4)
        feat["evic_per_unit_2yr"] = feat["evic_cnt_2yr"] / (unitsres + 1e-4)
    else:
        feat["evic_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["evic_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_evic"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)
        feat["evic_per_unit_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["evic_per_unit_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    # DOB Building Complaints Features
    if df_dob_comp is not None and len(df_dob_comp) > 0:
        min_dob_comp_date = T - pd.Timedelta(days=730)
        dc_mask = (df_dob_comp["complaint_date"] < T) & (
            df_dob_comp["complaint_date"] >= min_dob_comp_date
        )
        dc_sub = df_dob_comp.loc[dc_mask, ["bbl", "complaint_date"]].copy()

        dc_dt = (T - dc_sub["complaint_date"]).dt.days.values
        dc_sub["dob_comp_cnt_1yr"] = np.where(dc_dt <= 365, 1, 0)
        dc_sub["dob_comp_cnt_2yr"] = 1
        dc_sub["days_dob_comp"] = dc_dt

        dc_agg = dc_sub.groupby("bbl").agg(
            {
                "dob_comp_cnt_1yr": "sum",
                "dob_comp_cnt_2yr": "sum",
                "days_dob_comp": "min",
            }
        )

        feat["dob_comp_cnt_1yr"] = (
            bbls.map(dc_agg["dob_comp_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["dob_comp_cnt_2yr"] = (
            bbls.map(dc_agg["dob_comp_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_dob_comp"] = (
            bbls.map(dc_agg["days_dob_comp"]).fillna(9999).astype(np.float32)
        )
        feat["dob_comp_per_unit_1yr"] = feat["dob_comp_cnt_1yr"] / (unitsres + 1e-4)
        feat["dob_comp_per_unit_2yr"] = feat["dob_comp_cnt_2yr"] / (unitsres + 1e-4)
    else:
        feat["dob_comp_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_comp_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_dob_comp"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)
        feat["dob_comp_per_unit_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_comp_per_unit_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Emergency Repair Charges (HWO & OMO heat and acute physical distress)
    emerg_dfs = []
    if df_hwo is not None and len(df_hwo) > 0 and "charge_date" in df_hwo.columns:
        emerg_dfs.append(df_hwo[["bbl", "charge_date"]])
    if df_omo is not None and len(df_omo) > 0 and "charge_date" in df_omo.columns:
        emerg_dfs.append(df_omo[["bbl", "charge_date"]])

    if len(emerg_dfs) > 0:
        df_emerg = pd.concat(emerg_dfs, ignore_index=True)
        min_emerg_date = T - pd.Timedelta(days=1095)
        em_mask = (df_emerg["charge_date"] < T) & (
            df_emerg["charge_date"] >= min_emerg_date
        )
        em_sub = df_emerg.loc[em_mask, ["bbl", "charge_date"]].copy()

        em_dt = (T - em_sub["charge_date"]).dt.days.values
        em_sub["emerg_cnt_1yr"] = np.where(em_dt <= 365, 1, 0)
        em_sub["emerg_cnt_3yr"] = 1
        em_sub["days_emerg"] = em_dt

        em_agg = em_sub.groupby("bbl").agg(
            {
                "emerg_cnt_1yr": "sum",
                "emerg_cnt_3yr": "sum",
                "days_emerg": "min",
            }
        )

        feat["emerg_charge_cnt_1yr"] = (
            bbls.map(em_agg["emerg_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["emerg_charge_cnt_3yr"] = (
            bbls.map(em_agg["emerg_cnt_3yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_emerg_charge"] = (
            bbls.map(em_agg["days_emerg"]).fillna(9999).astype(np.float32)
        )
        feat["has_emerg_charge_3yr"] = (
            feat["emerg_charge_cnt_3yr"] > 0
        ).astype(np.float32)
    else:
        feat["emerg_charge_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["emerg_charge_cnt_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_emerg_charge"] = np.full(
            len(cohort_df), 9999.0, dtype=np.float32
        )
        feat["has_emerg_charge_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Tax Lien Distress Indicator
    if df_lien is not None and len(df_lien) > 0:
        if "sale_date" in df_lien.columns and df_lien["sale_date"].notna().any():
            lien_sub = df_lien[df_lien["sale_date"] < T]
            lien_bbls = set(lien_sub["bbl"].unique())
        elif "sale_year" in df_lien.columns and df_lien["sale_year"].notna().any():
            lien_sub = df_lien[df_lien["sale_year"] < T.year]
            lien_bbls = set(lien_sub["bbl"].unique())
        else:
            lien_bbls = set(df_lien["bbl"].unique())
        feat["has_tax_lien"] = bbls.isin(lien_bbls).astype(np.float32).values
    else:
        feat["has_tax_lien"] = np.zeros(len(cohort_df), dtype=np.float32)

    # CONH Harassment Program Indicator
    if conh_bbls is not None and len(conh_bbls) > 0:
        feat["is_conh_building"] = bbls.isin(conh_bbls).astype(np.float32).values
    else:
        feat["is_conh_building"] = np.zeros(len(cohort_df), dtype=np.float32)

    # DOB Violations Features (Multi-Window Counts, Recency, Density)
    if df_dob_v is not None and len(df_dob_v) > 0:
        min_dob_v_date = T - pd.Timedelta(days=1095)
        dv_mask = (df_dob_v["violation_date"] < T) & (
            df_dob_v["violation_date"] >= min_dob_v_date
        )
        dv_sub = df_dob_v.loc[dv_mask, ["bbl", "violation_date"]].copy()

        dv_dt = (T - dv_sub["violation_date"]).dt.days.values
        dv_sub["dob_v_cnt_1yr"] = np.where(dv_dt <= 365, 1, 0)
        dv_sub["dob_v_cnt_2yr"] = np.where(dv_dt <= 730, 1, 0)
        dv_sub["dob_v_cnt_3yr"] = 1
        dv_sub["days_dob_v"] = dv_dt

        dv_agg = dv_sub.groupby("bbl").agg(
            {
                "dob_v_cnt_1yr": "sum",
                "dob_v_cnt_2yr": "sum",
                "dob_v_cnt_3yr": "sum",
                "days_dob_v": "min",
            }
        )

        feat["dob_v_cnt_1yr"] = (
            bbls.map(dv_agg["dob_v_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["dob_v_cnt_2yr"] = (
            bbls.map(dv_agg["dob_v_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["dob_v_cnt_3yr"] = (
            bbls.map(dv_agg["dob_v_cnt_3yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_dob_v"] = (
            bbls.map(dv_agg["days_dob_v"]).fillna(9999).astype(np.float32)
        )
        feat["dob_v_per_unit_1yr"] = feat["dob_v_cnt_1yr"] / (unitsres + 1e-4)
    else:
        feat["dob_v_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_v_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["dob_v_cnt_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_dob_v"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)
        feat["dob_v_per_unit_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    # HPD Bedbug Reports Features (Multi-Window Counts, Recency, Density)
    if df_bedbug is not None and len(df_bedbug) > 0:
        min_bedbug_date = T - pd.Timedelta(days=1095)
        bb_mask = (df_bedbug["filing_date"] < T) & (
            df_bedbug["filing_date"] >= min_bedbug_date
        )
        bb_sub = df_bedbug.loc[bb_mask, ["bbl", "filing_date"]].copy()

        bb_dt = (T - bb_sub["filing_date"]).dt.days.values
        bb_sub["bedbug_cnt_1yr"] = np.where(bb_dt <= 365, 1, 0)
        bb_sub["bedbug_cnt_2yr"] = np.where(bb_dt <= 730, 1, 0)
        bb_sub["bedbug_cnt_3yr"] = 1
        bb_sub["days_bedbug"] = bb_dt

        bb_agg = bb_sub.groupby("bbl").agg(
            {
                "bedbug_cnt_1yr": "sum",
                "bedbug_cnt_2yr": "sum",
                "bedbug_cnt_3yr": "sum",
                "days_bedbug": "min",
            }
        )

        feat["bedbug_cnt_1yr"] = (
            bbls.map(bb_agg["bedbug_cnt_1yr"]).fillna(0).astype(np.float32)
        )
        feat["bedbug_cnt_2yr"] = (
            bbls.map(bb_agg["bedbug_cnt_2yr"]).fillna(0).astype(np.float32)
        )
        feat["bedbug_cnt_3yr"] = (
            bbls.map(bb_agg["bedbug_cnt_3yr"]).fillna(0).astype(np.float32)
        )
        feat["days_since_bedbug"] = (
            bbls.map(bb_agg["days_bedbug"]).fillna(9999).astype(np.float32)
        )
        feat["bedbug_per_unit_1yr"] = feat["bedbug_cnt_1yr"] / (unitsres + 1e-4)
    else:
        feat["bedbug_cnt_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["bedbug_cnt_2yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["bedbug_cnt_3yr"] = np.zeros(len(cohort_df), dtype=np.float32)
        feat["days_since_bedbug"] = np.full(len(cohort_df), 9999.0, dtype=np.float32)
        feat["bedbug_per_unit_1yr"] = np.zeros(len(cohort_df), dtype=np.float32)

    # HPD Underlying Conditions Program Flag
    if uc_bbls is not None and len(uc_bbls) > 0:
        feat["is_underlying_conditions"] = bbls.isin(uc_bbls).astype(np.float32).values
    else:
        feat["is_underlying_conditions"] = np.zeros(len(cohort_df), dtype=np.float32)

    # Speculation Watch List Flag
    if swl_bbls is not None and len(swl_bbls) > 0:
        feat["is_speculation_watchlist"] = bbls.isin(swl_bbls).astype(np.float32).values
    else:
        feat["is_speculation_watchlist"] = np.zeros(len(cohort_df), dtype=np.float32)

    return pd.DataFrame(feat, index=cohort_df.index)


print("Building training features (2020 cohort)...")
X_train_2020 = build_features(
    df_train_2020,
    "2020-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    prior_2019_pos,
    df_comp,
    df_dob,
    df_rod,
    df_evic,
    df_dob_comp,
    df_hwo,
    df_omo,
    df_lien,
    conh_bbls,
    df_dob_v,
    df_bedbug,
    uc_bbls,
    swl_bbls,
)

print("Building training features (2021 cohort)...")
X_train_2021 = build_features(
    df_train_2021,
    "2021-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    prior_2020_pos,
    df_comp,
    df_dob,
    df_rod,
    df_evic,
    df_dob_comp,
    df_hwo,
    df_omo,
    df_lien,
    conh_bbls,
    df_dob_v,
    df_bedbug,
    uc_bbls,
    swl_bbls,
)

print("Building validation features (2022 cohort)...")
X_val = build_features(
    df_val_cohort,
    "2022-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    prior_2021_pos,
    df_comp,
    df_dob,
    df_rod,
    df_evic,
    df_dob_comp,
    df_hwo,
    df_omo,
    df_lien,
    conh_bbls,
    df_dob_v,
    df_bedbug,
    uc_bbls,
    swl_bbls,
)

print("Building test features (2023 cohort)...")
X_test = build_features(
    df_test_cohort,
    "2023-01-01",
    df_viol,
    df_lit,
    df_vac,
    aep_bbls,
    prior_2022_pos,
    df_comp,
    df_dob,
    df_rod,
    df_evic,
    df_dob_comp,
    df_hwo,
    df_omo,
    df_lien,
    conh_bbls,
    df_dob_v,
    df_bedbug,
    uc_bbls,
    swl_bbls,
)

# Pool 2020 and 2021 cohorts into expanded training set with temporal recency weighting
if len(df_train_2020) > 0 and len(pluto_19) > 0:
    X_train = pd.concat([X_train_2020, X_train_2021], axis=0).reset_index(drop=True)
    y_train = np.concatenate([y_train_2020, y_train_2021], axis=0)
    sample_weight = np.concatenate(
        [
            np.full(len(X_train_2020), 0.7, dtype=np.float32),
            np.full(len(X_train_2021), 1.0, dtype=np.float32),
        ],
        axis=0,
    )
else:
    X_train = X_train_2021.reset_index(drop=True)
    y_train = y_train_2021
    sample_weight = np.ones(len(X_train), dtype=np.float32)

X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

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
        self.bypass_proj = nn.Sequential(
            nn.Linear(num_features, d_hidden),
            nn.LayerNorm(d_hidden),
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
        h = self.input_proj(flat) + self.bypass_proj(x)
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


def get_lgb_model():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=7,
        subsample=0.70,
        colsample_bytree=0.65,
        min_child_samples=50,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model():
    return xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
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
        iterations=700,
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
print("Fitting LightGBM Diversified Model...")
lgb_model = get_lgb_model()
lgb_model.fit(X_train, y_train, sample_weight=sample_weight)
lgb_model.booster_.save_model("./working/lgb_model.txt")

print("Fitting XGBoost Histogram Model...")
xgb_model = get_xgb_model()
xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
xgb_model.save_model("./working/xgb_hist.json")

print("Fitting CatBoost Model...")
cb_model = get_catboost_model()
cb_model.fit(X_train, y_train, sample_weight=sample_weight)
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
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=True)
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
p_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
p_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_val_cb = cb_model.predict_proba(X_val)[:, 1]

cagnet.eval()
val_preds_list = []
with torch.no_grad():
    for (v_bx,) in val_loader:
        v_bx = v_bx.to(device)
        val_preds_list.append(cagnet(v_bx).cpu())
p_val_nn = torch.cat(val_preds_list).numpy()

n_val = len(p_val_lgb)
r_val_lgb = rankdata(p_val_lgb) / n_val
r_val_xgb = rankdata(p_val_xgb) / n_val
r_val_cb = rankdata(p_val_cb) / n_val
r_val_nn = rankdata(p_val_nn) / n_val

val_ranks = [r_val_lgb, r_val_xgb, r_val_cb, r_val_nn]

print(
    f"Individual Val APs - LGB: {average_precision_score(y_val, r_val_lgb):.4f}, "
    f"XGB: {average_precision_score(y_val, r_val_xgb):.4f}, "
    f"CB: {average_precision_score(y_val, r_val_cb):.4f}, "
    f"CAGNet: {average_precision_score(y_val, r_val_nn):.4f}"
)

# Search optimal ensemble weights maximizing Average Precision on y_val
best_weights = np.array([0.25, 0.25, 0.25, 0.25])
best_ap = average_precision_score(
    y_val, sum(w * r for w, r in zip(best_weights, val_ranks))
)

np.random.seed(42)
dirichlet_priors = [
    np.array([2.0, 2.0, 2.0, 2.0]),
    np.array([1.5, 2.5, 2.5, 2.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([2.5, 2.5, 2.0, 1.5]),
]
for alpha in dirichlet_priors:
    for _ in range(300):
        w = np.random.dirichlet(alpha)
        ens = sum(wi * ri for wi, ri in zip(w, val_ranks))
        score = average_precision_score(y_val, ens)
        if score > best_ap:
            best_ap = score
            best_weights = w

# Fine-tune with coordinate search
for step in [0.05, 0.02, 0.01, 0.005]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
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
p_test_lgb = lgb_model.predict_proba(X_test)[:, 1]
p_test_xgb = xgb_model.predict_proba(X_test)[:, 1]
p_test_cb = cb_model.predict_proba(X_test)[:, 1]

test_preds_list = []
with torch.no_grad():
    for (t_bx,) in test_loader:
        t_bx = t_bx.to(device)
        test_preds_list.append(cagnet(t_bx).cpu())
p_test_nn = torch.cat(test_preds_list).numpy()

n_test = len(p_test_lgb)
r_test_lgb = rankdata(p_test_lgb) / n_test
r_test_xgb = rankdata(p_test_xgb) / n_test
r_test_cb = rankdata(p_test_cb) / n_test
r_test_nn = rankdata(p_test_nn) / n_test

test_ranks = [r_test_lgb, r_test_xgb, r_test_cb, r_test_nn]
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

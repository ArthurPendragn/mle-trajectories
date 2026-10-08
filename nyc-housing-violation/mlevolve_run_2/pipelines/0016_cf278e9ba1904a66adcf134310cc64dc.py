import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings
from catboost import CatBoostClassifier
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

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

T_TRAIN_2020 = pd.Timestamp("2020-01-01")
T_TRAIN_2021 = pd.Timestamp("2021-01-01")
T_TRAIN = T_TRAIN_2021
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
        boro_raw = df[boro_col]
        boro_n = pd.to_numeric(boro_raw, errors="coerce")
        if boro_n.isna().all() or boro_raw.dtype == object:
            boro_map = {
                "MANHATTAN": 1,
                "MN": 1,
                "NEW YORK": 1,
                "NY": 1,
                "BRONX": 2,
                "BX": 2,
                "BROOKLYN": 3,
                "BK": 3,
                "KINGS": 3,
                "QUEENS": 4,
                "QN": 4,
                "STATEN ISLAND": 5,
                "SI": 5,
                "RICHMOND": 5,
            }
            boro_mapped = boro_raw.astype(str).str.upper().str.strip().map(boro_map)
            boro_n = boro_n.fillna(boro_mapped)

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


def get_columns_from_data_dict(table_name: str) -> List[str]:
    """Extracts column names from DATA_DICTIONARY.md for a given table."""
    for path in ["input/DATA_DICTIONARY.md", "DATA_DICTIONARY.md"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                lower_content = content.lower()
                target = table_name.lower()
                idx = lower_content.find(f"`{target}`")
                if idx == -1:
                    idx = lower_content.find(f" {target} ")
                if idx == -1:
                    idx = lower_content.find(f"#{target}")
                if idx != -1:
                    sub = content[idx : idx + 4000]
                    lines = sub.split("\n")
                    cols = []
                    for line in lines[1:]:
                        if line.startswith("#"):
                            break
                        if "|" in line:
                            cells = [c.strip().strip("`* ") for c in line.split("|")]
                            cells = [c for c in cells if c]
                            if cells and cells[0].lower() not in [
                                "column",
                                "name",
                                "field",
                                "---",
                                "--",
                                "key",
                            ]:
                                cols.append(cells[0])
                    if cols:
                        return cols
            except Exception as e:
                print(f"Note: Error parsing data dictionary: {e}")
    return []


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
                if len(avail) == len(columns):
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

# 2.3b Load HPD Complaints
cmp_url = f"{LAKE_FULL_URL}/hpd_complaints"
cmp_schema_cols = []
try:
    import fsspec
    import pyarrow.parquet as pq

    fs, p_path = fsspec.core.url_to_fs(cmp_url, **(STORAGE_OPTIONS or {}))
    cmp_ds = pq.ParquetDataset(p_path, filesystem=fs)
    cmp_schema_cols = [f.name for f in cmp_ds.schema]
except Exception:
    pass

if not cmp_schema_cols:
    cmp_schema_cols = get_columns_from_data_dict("hpd_complaints")

date_candidates = [
    "receiveddate",
    "received_date",
    "status_date",
    "statusdate",
    "complaint_date",
    "dateentered",
]
cmp_date_col = None
if cmp_schema_cols:
    col_lower_map = {c.lower(): c for c in cmp_schema_cols}
    for cand in date_candidates:
        if cand in col_lower_map:
            cmp_date_col = col_lower_map[cand]
            break
    if cmp_date_col is None:
        for c in cmp_schema_cols:
            if any(k in c.lower() for k in ["receive", "date", "status"]):
                cmp_date_col = c
                break

cols_to_load = None
if cmp_schema_cols and cmp_date_col:
    bbl_cands = [
        c
        for c in cmp_schema_cols
        if c.lower() in ["bbl", "boroid", "borough", "block", "lot"]
    ]
    extra_cands = [
        c
        for c in cmp_schema_cols
        if any(
            k in c.lower()
            for k in [
                "majorcategory",
                "major_category",
                "category",
                "status",
                "statusdate",
                "status_date",
            ]
        )
    ]
    if bbl_cands:
        cols_to_load = list(set(bbl_cands + [cmp_date_col] + extra_cands))

df_cmp = safe_read_parquet(
    cmp_url, columns=cols_to_load, storage_options=STORAGE_OPTIONS
)
if not df_cmp.empty:
    df_cmp.columns = [c.lower() for c in df_cmp.columns]
    if cmp_date_col and cmp_date_col.lower() in df_cmp.columns:
        actual_date_col = cmp_date_col.lower()
    else:
        actual_date_col = next(
            (
                c
                for c in df_cmp.columns
                if any(k in c for k in ["receive", "date", "status"])
            ),
            None,
        )

    cat_col = next(
        (c for c in df_cmp.columns if "major" in c or "category" in c),
        None,
    )
    status_col = next(
        (
            c
            for c in df_cmp.columns
            if c in ["status", "complaintstatus", "complaint_status"]
        ),
        None,
    )
    statusdate_col = next(
        (
            c
            for c in df_cmp.columns
            if "statusdate" in c or "status_date" in c
        ),
        None,
    )

    if actual_date_col is not None:
        df_cmp["receiveddate"] = pd.to_datetime(
            df_cmp[actual_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
        df_cmp["bbl"] = normalize_bbl_series(df_cmp)
        df_cmp = df_cmp[df_cmp["bbl"].notna() & df_cmp["receiveddate"].notna()]

        if cat_col is not None:
            df_cmp["majorcategory"] = (
                df_cmp[cat_col].astype(str).str.upper().str.strip()
            )
        else:
            df_cmp["majorcategory"] = ""

        if status_col is not None:
            df_cmp["status"] = (
                df_cmp[status_col].astype(str).str.upper().str.strip()
            )
        else:
            df_cmp["status"] = ""

        if statusdate_col is not None:
            df_cmp["statusdate"] = pd.to_datetime(
                df_cmp[statusdate_col], errors="coerce", utc=True
            ).dt.tz_localize(None)
        else:
            df_cmp["statusdate"] = pd.NaT

        df_cmp = df_cmp[
            ["bbl", "receiveddate", "majorcategory", "status", "statusdate"]
        ]
    else:
        df_cmp = pd.DataFrame()

print(f"Loaded and standardized HPD complaints: {df_cmp.shape}")

# 2.3c Load DOB Violations
dob_url = f"{LAKE_FULL_URL}/dob_violations"
dob_schema_cols = []
try:
    import fsspec
    import pyarrow.parquet as pq

    fs, p_path = fsspec.core.url_to_fs(dob_url, **(STORAGE_OPTIONS or {}))
    dob_ds = pq.ParquetDataset(p_path, filesystem=fs)
    dob_schema_cols = [f.name for f in dob_ds.schema]
except Exception:
    pass

if not dob_schema_cols:
    dob_schema_cols = get_columns_from_data_dict("dob_violations")

dob_date_cands = [
    "issue_date",
    "issuedate",
    "violation_date",
    "violationdate",
    "issue_dt",
    "inspection_date",
    "inspectiondate",
    "entry_date",
]
dob_date_col = None
if dob_schema_cols:
    col_lower_map = {c.lower(): c for c in dob_schema_cols}
    for cand in dob_date_cands:
        if cand in col_lower_map:
            dob_date_col = col_lower_map[cand]
            break
    if dob_date_col is None:
        for c in dob_schema_cols:
            if any(k in c.lower() for k in ["issue", "violation_date", "date"]):
                dob_date_col = c
                break

cols_to_load_dob = None
if dob_schema_cols and dob_date_col:
    bbl_cands = [
        c
        for c in dob_schema_cols
        if c.lower() in ["bbl", "boroid", "borough", "boro", "block", "lot"]
    ]
    if bbl_cands:
        cols_to_load_dob = list(set(bbl_cands + [dob_date_col]))

df_dob = safe_read_parquet(
    dob_url, columns=cols_to_load_dob, storage_options=STORAGE_OPTIONS
)
if not df_dob.empty:
    df_dob.columns = [c.lower() for c in df_dob.columns]
    if dob_date_col and dob_date_col.lower() in df_dob.columns:
        actual_dob_date = dob_date_col.lower()
    else:
        actual_dob_date = next(
            (
                c
                for c in df_dob.columns
                if any(k in c for k in ["issue", "violation_date", "date"])
            ),
            None,
        )

    if actual_dob_date is not None:
        df_dob["event_date"] = pd.to_datetime(
            df_dob[actual_dob_date], errors="coerce", utc=True
        ).dt.tz_localize(None)

        boro_col = next(
            (c for c in df_dob.columns if c in ["boroid", "borough", "boro"]),
            "boroid",
        )
        df_dob["bbl"] = normalize_bbl_series(
            df_dob,
            bbl_col="bbl",
            boro_col=boro_col,
            block_col="block",
            lot_col="lot",
        )
        df_dob = df_dob[df_dob["bbl"].notna() & df_dob["event_date"].notna()]
        df_dob = df_dob[["bbl", "event_date"]]
    else:
        df_dob = pd.DataFrame()

print(f"Loaded and standardized DOB violations: {df_dob.shape}")

# 2.3d Load DOB ECB Violations (Environmental Control Board Summonses)
dob_ecb_url = f"{LAKE_FULL_URL}/dob_ecb_violations"
dob_ecb_schema_cols = []
try:
    import fsspec
    import pyarrow.parquet as pq

    fs, p_path = fsspec.core.url_to_fs(dob_ecb_url, **(STORAGE_OPTIONS or {}))
    dob_ecb_ds = pq.ParquetDataset(p_path, filesystem=fs)
    dob_ecb_schema_cols = [f.name for f in dob_ecb_ds.schema]
except Exception:
    pass

if not dob_ecb_schema_cols:
    dob_ecb_schema_cols = get_columns_from_data_dict("dob_ecb_violations")

dob_ecb_date_cands = [
    "issue_date",
    "issuedate",
    "violation_date",
    "violationdate",
    "issue_dt",
    "served_date",
    "hearing_date",
    "inspection_date",
    "inspectiondate",
    "entry_date",
]
dob_ecb_date_col = None
if dob_ecb_schema_cols:
    col_lower_map = {c.lower(): c for c in dob_ecb_schema_cols}
    for cand in dob_ecb_date_cands:
        if cand in col_lower_map:
            dob_ecb_date_col = col_lower_map[cand]
            break
    if dob_ecb_date_col is None:
        for c in dob_ecb_schema_cols:
            if any(k in c.lower() for k in ["issue", "violation_date", "date", "served"]):
                dob_ecb_date_col = c
                break

cols_to_load_ecb = None
if dob_ecb_schema_cols and dob_ecb_date_col:
    bbl_cands = [
        c
        for c in dob_ecb_schema_cols
        if c.lower() in ["bbl", "boroid", "borough", "boro", "block", "lot"]
    ]
    if bbl_cands:
        cols_to_load_ecb = list(set(bbl_cands + [dob_ecb_date_col]))

df_dob_ecb = safe_read_parquet(
    dob_ecb_url, columns=cols_to_load_ecb, storage_options=STORAGE_OPTIONS
)
if not df_dob_ecb.empty:
    df_dob_ecb.columns = [c.lower() for c in df_dob_ecb.columns]
    if dob_ecb_date_col and dob_ecb_date_col.lower() in df_dob_ecb.columns:
        actual_ecb_date = dob_ecb_date_col.lower()
    else:
        actual_ecb_date = next(
            (
                c
                for c in df_dob_ecb.columns
                if any(k in c for k in ["issue", "violation_date", "date", "served"])
            ),
            None,
        )

    if actual_ecb_date is not None:
        df_dob_ecb["event_date"] = pd.to_datetime(
            df_dob_ecb[actual_ecb_date], errors="coerce", utc=True
        ).dt.tz_localize(None)

        boro_col = next(
            (c for c in df_dob_ecb.columns if c in ["boroid", "borough", "boro"]),
            "boroid",
        )
        df_dob_ecb["bbl"] = normalize_bbl_series(
            df_dob_ecb,
            bbl_col="bbl",
            boro_col=boro_col,
            block_col="block",
            lot_col="lot",
        )
        df_dob_ecb = df_dob_ecb[
            df_dob_ecb["bbl"].notna() & df_dob_ecb["event_date"].notna()
        ]
        df_dob_ecb = df_dob_ecb[["bbl", "event_date"]]
    else:
        df_dob_ecb = pd.DataFrame()

print(f"Loaded and standardized DOB ECB violations: {df_dob_ecb.shape}")

# 2.3e Load HPD Registrations (Institutional Landlord Portfolios)
reg_url = f"{LAKE_FULL_URL}/hpd_registrations"
reg_schema_cols = []
try:
    import fsspec
    import pyarrow.parquet as pq

    fs, p_path = fsspec.core.url_to_fs(reg_url, **(STORAGE_OPTIONS or {}))
    reg_ds = pq.ParquetDataset(p_path, filesystem=fs)
    reg_schema_cols = [f.name for f in reg_ds.schema]
except Exception:
    pass

if not reg_schema_cols:
    reg_schema_cols = get_columns_from_data_dict("hpd_registrations")

cols_to_load_reg = None
if reg_schema_cols:
    bbl_cands = [
        c
        for c in reg_schema_cols
        if c.lower() in ["bbl", "boroid", "borough", "boro", "block", "lot"]
    ]
    id_cands = [
        c
        for c in reg_schema_cols
        if any(
            k in c.lower()
            for k in ["registrationid", "registration_id", "regid", "reg_id", "id"]
        )
    ]
    date_cands = [c for c in reg_schema_cols if "date" in c.lower()]
    if bbl_cands and id_cands:
        cols_to_load_reg = list(set(bbl_cands + id_cands + date_cands))

df_reg = safe_read_parquet(
    reg_url, columns=cols_to_load_reg, storage_options=STORAGE_OPTIONS
)
if not df_reg.empty:
    df_reg.columns = [c.lower() for c in df_reg.columns]
    boro_col = next(
        (c for c in df_reg.columns if c in ["boroid", "borough", "boro"]),
        "boroid",
    )
    df_reg["bbl"] = normalize_bbl_series(
        df_reg,
        bbl_col=next((c for c in df_reg.columns if c == "bbl"), "bbl"),
        boro_col=boro_col,
        block_col=next((c for c in df_reg.columns if c == "block"), "block"),
        lot_col=next((c for c in df_reg.columns if c == "lot"), "lot"),
    )
    reg_id_col = next(
        (
            c
            for c in df_reg.columns
            if "registrationid" in c or ("reg" in c and "id" in c)
        ),
        None,
    )
    if reg_id_col is not None:
        df_reg["registrationid"] = (
            df_reg[reg_id_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df_reg = df_reg[
            df_reg["bbl"].notna()
            & df_reg["registrationid"].notna()
            & (df_reg["registrationid"] != "")
            & (df_reg["registrationid"] != "0")
            & (df_reg["registrationid"] != "nan")
            & (df_reg["registrationid"] != "None")
        ]
        reg_date_col = next((c for c in df_reg.columns if "date" in c), None)
        if reg_date_col is not None:
            df_reg["event_date"] = pd.to_datetime(
                df_reg[reg_date_col], errors="coerce", utc=True
            ).dt.tz_localize(None)
        else:
            df_reg["event_date"] = pd.NaT
        df_reg = df_reg[["bbl", "registrationid", "event_date"]]
    else:
        df_reg = pd.DataFrame(columns=["bbl", "registrationid", "event_date"])
else:
    df_reg = pd.DataFrame(columns=["bbl", "registrationid", "event_date"])

print(f"Loaded and standardized HPD registrations: {df_reg.shape}")

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

# Load Habitability and Statutory Distress Tables
df_rodent = safe_read_parquet(
    f"{LAKE_FULL_URL}/dohmh_rodent_inspections", storage_options=STORAGE_OPTIONS
)
if not df_rodent.empty:
    df_rodent["bbl"] = normalize_bbl_series(
        df_rodent,
        bbl_col=next((c for c in df_rodent.columns if c.lower() == "bbl"), "bbl"),
        boro_col=next((c for c in df_rodent.columns if "boro" in c.lower()), "boroid"),
        block_col=next((c for c in df_rodent.columns if "block" in c.lower()), "block"),
        lot_col=next((c for c in df_rodent.columns if "lot" in c.lower()), "lot"),
    )
    rod_date_col = next(
        (c for c in df_rodent.columns if "date" in c.lower() or "insp" in c.lower()),
        None,
    )
    if rod_date_col is not None:
        df_rodent["event_date"] = pd.to_datetime(
            df_rodent[rod_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_rodent["event_date"] = pd.NaT
    res_col = next((c for c in df_rodent.columns if "result" in c.lower()), None)
    if res_col is not None:
        res_str = df_rodent[res_col].astype(str).str.lower()
        df_rodent["is_fail"] = (
            res_str.str.contains("rat|mouse|problem|fail|active|sign")
            & ~res_str.str.contains("pass|passed|clean")
        ).astype(np.float32)
    else:
        df_rodent["is_fail"] = 1.0
    df_rodent = df_rodent[df_rodent["bbl"].notna() & df_rodent["event_date"].notna()]
    df_rodent = df_rodent[["bbl", "event_date", "is_fail"]]
else:
    df_rodent = pd.DataFrame(columns=["bbl", "event_date", "is_fail"])

df_bb = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_bedbug_reports", storage_options=STORAGE_OPTIONS
)
if not df_bb.empty:
    df_bb["bbl"] = normalize_bbl_series(
        df_bb,
        bbl_col=next((c for c in df_bb.columns if c.lower() == "bbl"), "bbl"),
        boro_col=next((c for c in df_bb.columns if "boro" in c.lower()), "boroid"),
        block_col=next((c for c in df_bb.columns if "block" in c.lower()), "block"),
        lot_col=next((c for c in df_bb.columns if "lot" in c.lower()), "lot"),
    )
    bb_date_col = next(
        (c for c in df_bb.columns if "date" in c.lower() or "filing" in c.lower()),
        None,
    )
    if bb_date_col is not None:
        df_bb["event_date"] = pd.to_datetime(
            df_bb[bb_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_bb["event_date"] = pd.NaT
    unit_col = next(
        (
            c
            for c in df_bb.columns
            if any(k in c.lower() for k in ["infested", "dwelling", "unit"])
        ),
        None,
    )
    if unit_col is not None:
        df_bb["infested_units"] = (
            pd.to_numeric(df_bb[unit_col], errors="coerce").fillna(1.0).clip(lower=0)
        )
    else:
        df_bb["infested_units"] = 1.0
    df_bb = df_bb[df_bb["bbl"].notna() & df_bb["event_date"].notna()]
    df_bb = df_bb[["bbl", "event_date", "infested_units"]]
else:
    df_bb = pd.DataFrame(columns=["bbl", "event_date", "infested_units"])

df_evict = safe_read_parquet(
    f"{LAKE_FULL_URL}/evictions", storage_options=STORAGE_OPTIONS
)
if not df_evict.empty:
    df_evict["bbl"] = normalize_bbl_series(
        df_evict,
        bbl_col=next((c for c in df_evict.columns if c.lower() == "bbl"), "bbl"),
        boro_col=next((c for c in df_evict.columns if "boro" in c.lower()), "boroid"),
        block_col=next((c for c in df_evict.columns if "block" in c.lower()), "block"),
        lot_col=next((c for c in df_evict.columns if "lot" in c.lower()), "lot"),
    )
    ev_date_col = next(
        (c for c in df_evict.columns if "date" in c.lower() or "executed" in c.lower()),
        None,
    )
    if ev_date_col is not None:
        df_evict["event_date"] = pd.to_datetime(
            df_evict[ev_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_evict["event_date"] = pd.NaT
    df_evict = df_evict[df_evict["bbl"].notna() & df_evict["event_date"].notna()]
    df_evict = df_evict[["bbl", "event_date"]]
else:
    df_evict = pd.DataFrame(columns=["bbl", "event_date"])

df_uc = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_underlying_conditions", storage_options=STORAGE_OPTIONS
)
if not df_uc.empty:
    df_uc["bbl"] = normalize_bbl_series(df_uc)
    uc_date_col = next((c for c in df_uc.columns if "date" in c.lower()), None)
    if uc_date_col is not None:
        df_uc["event_date"] = pd.to_datetime(
            df_uc[uc_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_uc["event_date"] = pd.NaT
    df_uc = df_uc[df_uc["bbl"].notna()]
else:
    df_uc = pd.DataFrame(columns=["bbl", "event_date"])

df_conh = safe_read_parquet(
    f"{LAKE_FULL_URL}/hpd_conh_buildings", storage_options=STORAGE_OPTIONS
)
if not df_conh.empty:
    df_conh["bbl"] = normalize_bbl_series(df_conh)
    conh_date_col = next((c for c in df_conh.columns if "date" in c.lower()), None)
    if conh_date_col is not None:
        df_conh["event_date"] = pd.to_datetime(
            df_conh[conh_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_conh["event_date"] = pd.NaT
    df_conh = df_conh[df_conh["bbl"].notna()]
else:
    df_conh = pd.DataFrame(columns=["bbl", "event_date"])

df_swl = safe_read_parquet(
    f"{LAKE_FULL_URL}/speculation_watch_list", storage_options=STORAGE_OPTIONS
)
if not df_swl.empty:
    df_swl["bbl"] = normalize_bbl_series(df_swl)
    swl_date_col = next((c for c in df_swl.columns if "date" in c.lower()), None)
    if swl_date_col is not None:
        df_swl["event_date"] = pd.to_datetime(
            df_swl[swl_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_swl["event_date"] = pd.NaT
    df_swl = df_swl[df_swl["bbl"].notna()]
else:
    df_swl = pd.DataFrame(columns=["bbl", "event_date"])

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

    train_2021_mask = v_series.str.startswith("20v7") & (
        pd.to_numeric(df_pluto["unitsres"], errors="coerce") >= 3
    )
    train_2021_bbls = df_pluto.loc[train_2021_mask, "bbl"].dropna().unique()
    if len(train_2021_bbls) == 0:
        train_2021_bbls = test_bbls

    train_2020_mask = (
        v_series.str.startswith("19v2") | v_series.str.startswith("19v")
    ) & (pd.to_numeric(df_pluto["unitsres"], errors="coerce") >= 3)
    train_2020_bbls = df_pluto.loc[train_2020_mask, "bbl"].dropna().unique()
    if len(train_2020_bbls) == 0:
        train_2020_bbls = train_2021_bbls
    train_bbls = train_2021_bbls
else:
    val_bbls = test_bbls
    train_2021_bbls = test_bbls
    train_2020_bbls = test_bbls
    train_bbls = test_bbls

print(
    f"Entities extracted: Train2020={len(train_2020_bbls)}, Train2021={len(train_2021_bbls)}, "
    f"Val={len(val_bbls)}, Test={len(test_bbls)}"
)

cohort_bbls = (
    set(train_2020_bbls)
    | set(train_2021_bbls)
    | set(val_bbls)
    | set(test_bbls)
)
df_vio = df_vio[df_vio["bbl"].isin(cohort_bbls)]
if not df_cmp.empty:
    df_cmp = df_cmp[df_cmp["bbl"].isin(cohort_bbls)]
if not df_dob.empty:
    df_dob = df_dob[df_dob["bbl"].isin(cohort_bbls)]
if not df_dob_ecb.empty:
    df_dob_ecb = df_dob_ecb[df_dob_ecb["bbl"].isin(cohort_bbls)]
if not df_rodent.empty:
    df_rodent = df_rodent[df_rodent["bbl"].isin(cohort_bbls)]
if not df_bb.empty:
    df_bb = df_bb[df_bb["bbl"].isin(cohort_bbls)]
if not df_evict.empty:
    df_evict = df_evict[df_evict["bbl"].isin(cohort_bbls)]
if not df_uc.empty:
    df_uc = df_uc[df_uc["bbl"].isin(cohort_bbls)]
if not df_conh.empty:
    df_conh = df_conh[df_conh["bbl"].isin(cohort_bbls)]
if not df_swl.empty:
    df_swl = df_swl[df_swl["bbl"].isin(cohort_bbls)]
if not df_reg.empty:
    df_reg = df_reg[df_reg["bbl"].isin(cohort_bbls)]


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

    if "cd" in df_feat.columns:
        df_feat["cd"] = (
            pd.to_numeric(df_feat["cd"], errors="coerce").fillna(0).astype(np.int32)
        )
    else:
        df_feat["cd"] = 0

    # 3.2 HPD Violations Point-in-Time Trajectory (Class C, B Precursors, & Recency Windows)
    v_prior = df_vio[df_vio["inspectiondate"] < cutoff]

    w30 = cutoff - pd.Timedelta(days=30)
    w60 = cutoff - pd.Timedelta(days=60)
    w90 = cutoff - pd.Timedelta(days=90)
    w180 = cutoff - pd.Timedelta(days=180)
    w365 = cutoff - pd.Timedelta(days=365)
    w730 = cutoff - pd.Timedelta(days=730)
    w1095 = cutoff - pd.Timedelta(days=1095)

    vc_all = v_prior[v_prior["class"] == "C"]
    vc_30 = vc_all[vc_all["inspectiondate"] >= w30]
    vc_60 = vc_all[vc_all["inspectiondate"] >= w60]
    vc_90 = vc_all[vc_all["inspectiondate"] >= w90]
    vc_180 = vc_all[vc_all["inspectiondate"] >= w180]
    vc_365 = vc_all[vc_all["inspectiondate"] >= w365]
    vc_730 = vc_all[vc_all["inspectiondate"] >= w730]
    vc_1095 = vc_all[vc_all["inspectiondate"] >= w1095]

    vb_all = v_prior[v_prior["class"] == "B"]
    vb_90 = vb_all[vb_all["inspectiondate"] >= w90]
    vb_180 = vb_all[vb_all["inspectiondate"] >= w180]
    vb_365 = vb_all[vb_all["inspectiondate"] >= w365]

    va_365 = v_prior[(v_prior["class"] == "A") & (v_prior["inspectiondate"] >= w365)]
    vall_365 = v_prior[v_prior["inspectiondate"] >= w365]
    vall_730 = v_prior[v_prior["inspectiondate"] >= w730]

    cnt_c_30 = vc_30.groupby("bbl").size().rename("vio_c_count_30d")
    cnt_c_60 = vc_60.groupby("bbl").size().rename("vio_c_count_60d")
    cnt_c_90 = vc_90.groupby("bbl").size().rename("vio_c_count_90d")
    cnt_c_180 = vc_180.groupby("bbl").size().rename("vio_c_count_180d")
    cnt_c_365 = vc_365.groupby("bbl").size().rename("vio_c_count_12m")
    cnt_c_730 = vc_730.groupby("bbl").size().rename("vio_c_count_24m")
    cnt_c_1095 = vc_1095.groupby("bbl").size().rename("vio_c_count_36m")
    cnt_c_all = vc_all.groupby("bbl").size().rename("vio_c_count_all")

    cnt_b_90 = vb_90.groupby("bbl").size().rename("vio_b_count_90d")
    cnt_b_180 = vb_180.groupby("bbl").size().rename("vio_b_count_180d")
    cnt_b_365 = vb_365.groupby("bbl").size().rename("vio_b_count_12m")
    cnt_b_all = vb_all.groupby("bbl").size().rename("vio_b_count_all")

    cnt_a_365 = va_365.groupby("bbl").size().rename("vio_a_count_12m")
    cnt_all_365 = vall_365.groupby("bbl").size().rename("vio_all_count_12m")
    cnt_all_730 = vall_730.groupby("bbl").size().rename("vio_all_count_24m")
    cnt_all_total = v_prior.groupby("bbl").size().rename("vio_all_count_total")

    last_c_date = vc_all.groupby("bbl")["inspectiondate"].max()
    days_since_c = ((cutoff - last_c_date).dt.total_seconds() / 86400.0).rename(
        "days_since_last_vio_c"
    )

    last_b_date = vb_all.groupby("bbl")["inspectiondate"].max()
    days_since_b = ((cutoff - last_b_date).dt.total_seconds() / 86400.0).rename(
        "days_since_last_vio_b"
    )

    last_any_date = v_prior.groupby("bbl")["inspectiondate"].max()
    days_since_any = ((cutoff - last_any_date).dt.total_seconds() / 86400.0).rename(
        "days_since_last_vio_any"
    )

    # Distinct inspection visit dates
    vc_365_insp = pd.DataFrame(
        {
            "bbl": vc_365["bbl"].values,
            "insp_day": vc_365["inspectiondate"].dt.floor("D").values,
        }
    )
    cnt_dist_c_12m = (
        vc_365_insp.drop_duplicates(subset=["bbl", "insp_day"])
        .groupby("bbl")
        .size()
        .rename("distinct_c_inspection_days_12m")
    )

    vc_1095_insp = pd.DataFrame(
        {
            "bbl": vc_1095["bbl"].values,
            "insp_day": vc_1095["inspectiondate"].dt.floor("D").values,
        }
    )
    cnt_dist_c_36m = (
        vc_1095_insp.drop_duplicates(subset=["bbl", "insp_day"])
        .groupby("bbl")
        .size()
        .rename("distinct_c_inspection_days_36m")
    )

    v_open = v_prior[v_prior["violationstatus"].astype(str).str.lower() == "open"]
    cnt_open_all = v_open.groupby("bbl").size().rename("open_violations_all")
    v_open_c = v_open[v_open["class"] == "C"]
    cnt_open_c = v_open_c.groupby("bbl").size().rename("open_violations_c")
    cnt_open_b = (
        v_open[v_open["class"] == "B"].groupby("bbl").size().rename("open_violations_b")
    )

    # Oldest open Class C violation age in days
    oldest_open_c_date = v_open_c.groupby("bbl")["inspectiondate"].min()
    oldest_open_c_age = (
        (cutoff - oldest_open_c_date).dt.total_seconds() / 86400.0
    ).rename("oldest_open_c_age_days")

    v_aggs = pd.concat(
        [
            cnt_c_30,
            cnt_c_60,
            cnt_c_90,
            cnt_c_180,
            cnt_c_365,
            cnt_c_730,
            cnt_c_1095,
            cnt_c_all,
            cnt_b_90,
            cnt_b_180,
            cnt_b_365,
            cnt_b_all,
            cnt_a_365,
            cnt_all_365,
            cnt_all_730,
            cnt_all_total,
            days_since_c,
            days_since_b,
            days_since_any,
            cnt_dist_c_12m,
            cnt_dist_c_36m,
            cnt_open_all,
            cnt_open_c,
            cnt_open_b,
            oldest_open_c_age,
        ],
        axis=1,
    )
    v_aggs.index.name = "bbl"
    v_aggs = v_aggs.reset_index()

    df_feat = df_feat.merge(v_aggs, on="bbl", how="left")

    count_cols = [
        c
        for c in v_aggs.columns
        if c
        not in [
            "bbl",
            "days_since_last_vio_c",
            "days_since_last_vio_b",
            "days_since_last_vio_any",
            "oldest_open_c_age_days",
        ]
    ]
    for c in count_cols:
        df_feat[c] = df_feat[c].fillna(0).astype(np.float32)

    df_feat["days_since_last_vio_c"] = (
        df_feat["days_since_last_vio_c"]
        .fillna(3650.0)
        .clip(0, 3650.0)
        .astype(np.float32)
    )
    df_feat["days_since_last_vio_b"] = (
        df_feat["days_since_last_vio_b"]
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
    df_feat["oldest_open_c_age_days"] = (
        df_feat["oldest_open_c_age_days"]
        .fillna(0.0)
        .clip(0, 3650.0)
        .astype(np.float32)
    )
    df_feat["oldest_open_c_violation_days"] = df_feat["oldest_open_c_age_days"]
    df_feat["days_since_oldest_open_c"] = df_feat["oldest_open_c_age_days"]

    # Historical multi-year Class C annual recurrence flags and aggregate persistence
    df_feat["c_positive_y1"] = (df_feat["vio_c_count_12m"] > 0).astype(np.float32)
    df_feat["c_positive_y2"] = (
        (df_feat["vio_c_count_24m"] - df_feat["vio_c_count_12m"]) > 0
    ).astype(np.float32)
    df_feat["c_positive_y3"] = (
        (df_feat["vio_c_count_36m"] - df_feat["vio_c_count_24m"]) > 0
    ).astype(np.float32)
    df_feat["c_annual_persistence_3yr"] = (
        df_feat["c_positive_y1"] + df_feat["c_positive_y2"] + df_feat["c_positive_y3"]
    ).astype(np.float32)
    df_feat["distinct_c_inspection_ratio_12m"] = (
        df_feat["distinct_c_inspection_days_12m"]
        / (df_feat["vio_c_count_12m"] + 1.0)
    ).astype(np.float32)

    df_feat["class_c_ratio_12m"] = (
        df_feat["vio_c_count_12m"] / (df_feat["vio_all_count_12m"] + 1.0)
    ).astype(np.float32)
    df_feat["class_b_ratio_12m"] = (
        df_feat["vio_b_count_12m"] / (df_feat["vio_all_count_12m"] + 1.0)
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
    df_feat["open_ratio_b"] = (
        df_feat["open_violations_b"] / (df_feat["vio_b_count_all"] + 1.0)
    ).astype(np.float32)
    df_feat["open_backlog_rate"] = (
        df_feat["open_violations_all"] / (df_feat["vio_all_count_total"] + 1.0)
    ).astype(np.float32)
    df_feat["acute_vio_c_velocity"] = (
        df_feat["vio_c_count_30d"]
        / (
            ((df_feat["vio_c_count_90d"] - df_feat["vio_c_count_30d"]).clip(lower=0) / 2.0)
            + 1.0
        )
    ).astype(np.float32)

    # 3.2a Landlord Portfolio Features via HPD Registrations
    if not df_reg.empty:
        if "event_date" in df_reg.columns and df_reg["event_date"].notna().any():
            reg_prior = df_reg[
                df_reg["event_date"].isna() | (df_reg["event_date"] < cutoff)
            ]
            reg_prior = reg_prior.sort_values("event_date")
        else:
            reg_prior = df_reg
        bbl_to_reg = reg_prior.drop_duplicates(subset=["bbl"], keep="last")[
            ["bbl", "registrationid"]
        ]
        reg_lot_cnt = (
            bbl_to_reg.groupby("registrationid")["bbl"]
            .nunique()
            .rename("portfolio_lot_count")
        )
        vc_365_with_reg = vc_365.merge(bbl_to_reg, on="bbl", how="inner")
        reg_vio_cnt = (
            vc_365_with_reg.groupby("registrationid")
            .size()
            .rename("portfolio_vio_c_12m")
        )
        port_df = pd.concat([reg_lot_cnt, reg_vio_cnt], axis=1).fillna(0)
        port_df["portfolio_mean_vio_c_rate"] = (
            port_df["portfolio_vio_c_12m"] / (port_df["portfolio_lot_count"] + 1e-4)
        ).astype(np.float32)
        port_df = port_df.reset_index()

        df_feat = df_feat.merge(bbl_to_reg, on="bbl", how="left")
        df_feat = df_feat.merge(port_df, on="registrationid", how="left")
        df_feat = df_feat.drop(columns=["registrationid"], errors="ignore")

        df_feat["portfolio_lot_count"] = (
            df_feat["portfolio_lot_count"].fillna(1.0).astype(np.float32)
        )
        df_feat["portfolio_vio_c_12m"] = (
            df_feat["portfolio_vio_c_12m"]
            .fillna(df_feat["vio_c_count_12m"])
            .astype(np.float32)
        )
        df_feat["portfolio_mean_vio_c_rate"] = (
            df_feat["portfolio_mean_vio_c_rate"]
            .fillna(df_feat["vio_c_count_12m"])
            .astype(np.float32)
        )
        df_feat["is_portfolio_landlord"] = (
            (df_feat["portfolio_lot_count"] > 1.0).astype(np.float32)
        )
    else:
        df_feat["portfolio_lot_count"] = np.float32(1.0)
        df_feat["portfolio_vio_c_12m"] = df_feat["vio_c_count_12m"].astype(np.float32)
        df_feat["portfolio_mean_vio_c_rate"] = df_feat["vio_c_count_12m"].astype(
            np.float32
        )
        df_feat["is_portfolio_landlord"] = np.float32(0.0)

    # 3.2b HPD Complaints Point-in-Time Trajectory (Acute Heat, Status & Velocity)
    if not df_cmp.empty and "receiveddate" in df_cmp.columns:
        cmp_prior = df_cmp[df_cmp["receiveddate"] < cutoff]
        cw30 = cutoff - pd.Timedelta(days=30)
        cw60 = cutoff - pd.Timedelta(days=60)
        cw90 = cutoff - pd.Timedelta(days=90)
        cw365 = cutoff - pd.Timedelta(days=365)

        c_30 = cmp_prior[cmp_prior["receiveddate"] >= cw30]
        c_60 = cmp_prior[cmp_prior["receiveddate"] >= cw60]
        c_90 = cmp_prior[cmp_prior["receiveddate"] >= cw90]
        c_365 = cmp_prior[cmp_prior["receiveddate"] >= cw365]

        cnt_cmp_30 = c_30.groupby("bbl").size().rename("complaints_30d")
        cnt_cmp_60 = c_60.groupby("bbl").size().rename("complaints_60d")
        cnt_cmp_90 = c_90.groupby("bbl").size().rename("complaints_90d")
        cnt_cmp_365 = c_365.groupby("bbl").size().rename("complaints_12m")

        last_cmp_date = cmp_prior.groupby("bbl")["receiveddate"].max()
        days_since_cmp = (
            (cutoff - last_cmp_date).dt.total_seconds() / 86400.0
        ).rename("days_since_last_complaint")

        # Acute Physical Failure Complaints Extraction
        is_heat = (
            cmp_prior["majorcategory"].str.contains(
                "HEAT|HOT WATER", case=False, na=False
            )
            if "majorcategory" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )
        is_plumbing = (
            cmp_prior["majorcategory"].str.contains(
                "PLUMB|LEAK|WATER LEAK", case=False, na=False
            )
            if "majorcategory" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )
        is_paint = (
            cmp_prior["majorcategory"].str.contains(
                "PAINT|PLASTER", case=False, na=False
            )
            if "majorcategory" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )
        is_electric = (
            cmp_prior["majorcategory"].str.contains("ELECTRIC", case=False, na=False)
            if "majorcategory" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )

        cmp_heat = cmp_prior[is_heat]
        cmp_plumb = cmp_prior[is_plumbing]
        cmp_paint = cmp_prior[is_paint]
        cmp_elec = cmp_prior[is_electric]

        heat_30 = cmp_heat[cmp_heat["receiveddate"] >= cw30]
        heat_90 = cmp_heat[cmp_heat["receiveddate"] >= cw90]
        heat_365 = cmp_heat[cmp_heat["receiveddate"] >= cw365]

        cnt_heat_30 = heat_30.groupby("bbl").size().rename("heat_complaints_30d")
        cnt_heat_90 = heat_90.groupby("bbl").size().rename("heat_complaints_90d")
        cnt_heat_365 = heat_365.groupby("bbl").size().rename("heat_complaints_12m")

        cnt_plumb_90 = (
            cmp_plumb[cmp_plumb["receiveddate"] >= cw90]
            .groupby("bbl")
            .size()
            .rename("plumbing_complaints_90d")
        )
        cnt_plumb_365 = (
            cmp_plumb[cmp_plumb["receiveddate"] >= cw365]
            .groupby("bbl")
            .size()
            .rename("plumbing_complaints_12m")
        )

        cnt_paint_90 = (
            cmp_paint[cmp_paint["receiveddate"] >= cw90]
            .groupby("bbl")
            .size()
            .rename("paint_complaints_90d")
        )
        cnt_paint_365 = (
            cmp_paint[cmp_paint["receiveddate"] >= cw365]
            .groupby("bbl")
            .size()
            .rename("paint_complaints_12m")
        )

        cnt_elec_90 = (
            cmp_elec[cmp_elec["receiveddate"] >= cw90]
            .groupby("bbl")
            .size()
            .rename("electric_complaints_90d")
        )
        cnt_elec_365 = (
            cmp_elec[cmp_elec["receiveddate"] >= cw365]
            .groupby("bbl")
            .size()
            .rename("electric_complaints_12m")
        )

        # Pending Open Complaints at Cutoff
        is_open_status = (
            cmp_prior["status"].str.contains("OPEN|PEND", case=False, na=False)
            if "status" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )
        has_future_statusdate = (
            (cmp_prior["statusdate"].notna() & (cmp_prior["statusdate"] >= cutoff))
            if "statusdate" in cmp_prior.columns
            else pd.Series(False, index=cmp_prior.index)
        )
        is_pending = is_open_status | has_future_statusdate
        cmp_pending = cmp_prior[is_pending]

        cnt_pending_all = (
            cmp_pending.groupby("bbl").size().rename("pending_complaints_count")
        )
        cnt_open_all = (
            cmp_pending.groupby("bbl").size().rename("open_complaints_count")
        )
        cnt_pending_90d = (
            cmp_pending[cmp_pending["receiveddate"] >= cw90]
            .groupby("bbl")
            .size()
            .rename("pending_complaints_90d")
        )
        cnt_pending_heat = (
            cmp_pending[
                cmp_pending["majorcategory"].str.contains(
                    "HEAT|HOT WATER", case=False, na=False
                )
            ]
            .groupby("bbl")
            .size()
            .rename("pending_heat_complaints_count")
        )

        cmp_aggs = pd.concat(
            [
                cnt_cmp_30,
                cnt_cmp_60,
                cnt_cmp_90,
                cnt_cmp_365,
                days_since_cmp,
                cnt_heat_30,
                cnt_heat_90,
                cnt_heat_365,
                cnt_plumb_90,
                cnt_plumb_365,
                cnt_paint_90,
                cnt_paint_365,
                cnt_elec_90,
                cnt_elec_365,
                cnt_pending_all,
                cnt_open_all,
                cnt_pending_90d,
                cnt_pending_heat,
            ],
            axis=1,
        )
        cmp_aggs.index.name = "bbl"
        cmp_aggs = cmp_aggs.reset_index()

        df_feat = df_feat.merge(cmp_aggs, on="bbl", how="left")
        for col in [
            "complaints_30d",
            "complaints_60d",
            "complaints_90d",
            "complaints_12m",
            "heat_complaints_30d",
            "heat_complaints_90d",
            "heat_complaints_12m",
            "plumbing_complaints_90d",
            "plumbing_complaints_12m",
            "paint_complaints_90d",
            "paint_complaints_12m",
            "electric_complaints_90d",
            "electric_complaints_12m",
            "pending_complaints_count",
            "open_complaints_count",
            "pending_complaints_90d",
            "pending_heat_complaints_count",
        ]:
            df_feat[col] = df_feat[col].fillna(0.0).astype(np.float32)

        df_feat["days_since_last_complaint"] = (
            df_feat["days_since_last_complaint"]
            .fillna(3650.0)
            .clip(0, 3650.0)
            .astype(np.float32)
        )
    else:
        for col in [
            "complaints_30d",
            "complaints_60d",
            "complaints_90d",
            "complaints_12m",
            "heat_complaints_30d",
            "heat_complaints_90d",
            "heat_complaints_12m",
            "plumbing_complaints_90d",
            "plumbing_complaints_12m",
            "paint_complaints_90d",
            "paint_complaints_12m",
            "electric_complaints_90d",
            "electric_complaints_12m",
            "pending_complaints_count",
            "open_complaints_count",
            "pending_complaints_90d",
            "pending_heat_complaints_count",
        ]:
            df_feat[col] = np.float32(0.0)
        df_feat["days_since_last_complaint"] = np.float32(3650.0)

    # Acute Winter Heat Ratios and Intensities
    df_feat["winter_heat_share"] = (
        df_feat["heat_complaints_90d"] / (df_feat["heat_complaints_12m"] + 1.0)
    ).astype(np.float32)
    df_feat["heat_per_unit"] = (
        df_feat["heat_complaints_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["winter_heat_per_unit"] = (
        df_feat["heat_complaints_90d"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["heat_complaint_share_90d"] = (
        df_feat["heat_complaints_90d"] / (df_feat["complaints_90d"] + 1.0)
    ).astype(np.float32)
    df_feat["plumbing_per_unit"] = (
        df_feat["plumbing_complaints_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["paint_per_unit"] = (
        df_feat["paint_complaints_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["electric_per_unit"] = (
        df_feat["electric_complaints_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)

    df_feat["complaint_momentum"] = (
        df_feat["complaints_90d"]
        / (
            (df_feat["complaints_12m"] - df_feat["complaints_90d"]).clip(
                lower=0
            )
            + 1.0
        )
    ).astype(np.float32)
    df_feat["acute_complaint_velocity"] = (
        df_feat["complaints_30d"]
        / (
            ((df_feat["complaints_90d"] - df_feat["complaints_30d"]).clip(lower=0) / 2.0)
            + 1.0
        )
    ).astype(np.float32)
    df_feat["complaint_intensity"] = (
        df_feat["complaints_12m"] / df_feat["unitsres"]
    ).astype(np.float32)
    df_feat["complaint_intensity_res"] = (
        df_feat["complaints_12m"] / df_feat["unitsres"]
    ).astype(np.float32)
    df_feat["complaint_to_vio_c_ratio"] = (
        df_feat["vio_c_count_12m"] / (df_feat["complaints_12m"] + 1.0)
    ).astype(np.float32)

    # 3.2c DOB Violations Point-in-Time Trajectory
    if not df_dob.empty and "event_date" in df_dob.columns:
        dob_prior = df_dob[df_dob["event_date"] < cutoff]
        w365 = cutoff - pd.Timedelta(days=365)
        w1095 = cutoff - pd.Timedelta(days=1095)

        cnt_dob_total = dob_prior.groupby("bbl").size().rename("dob_viol_total")
        cnt_dob_12m = (
            dob_prior[dob_prior["event_date"] >= w365]
            .groupby("bbl")
            .size()
            .rename("dob_viol_12m")
        )
        cnt_dob_36m = (
            dob_prior[dob_prior["event_date"] >= w1095]
            .groupby("bbl")
            .size()
            .rename("dob_viol_36m")
        )
        last_dob_date = dob_prior.groupby("bbl")["event_date"].max()
        days_since_dob = (
            (cutoff - last_dob_date).dt.total_seconds() / 86400.0
        ).rename("days_since_last_dob_viol")

        dob_aggs = pd.concat(
            [cnt_dob_total, cnt_dob_12m, cnt_dob_36m, days_since_dob], axis=1
        )
        dob_aggs.index.name = "bbl"
        dob_aggs = dob_aggs.reset_index()

        df_feat = df_feat.merge(dob_aggs, on="bbl", how="left")
        df_feat["dob_viol_total"] = (
            df_feat["dob_viol_total"].fillna(0.0).astype(np.float32)
        )
        df_feat["dob_viol_12m"] = (
            df_feat["dob_viol_12m"].fillna(0.0).astype(np.float32)
        )
        df_feat["dob_viol_36m"] = (
            df_feat["dob_viol_36m"].fillna(0.0).astype(np.float32)
        )
        df_feat["days_since_last_dob_viol"] = (
            df_feat["days_since_last_dob_viol"]
            .fillna(3650.0)
            .clip(0, 3650.0)
            .astype(np.float32)
        )
    else:
        df_feat["dob_viol_total"] = np.float32(0.0)
        df_feat["dob_viol_12m"] = np.float32(0.0)
        df_feat["dob_viol_36m"] = np.float32(0.0)
        df_feat["days_since_last_dob_viol"] = np.float32(3650.0)

    # 3.2d DOB ECB Violations Point-in-Time Trajectory (Judicial Summonses)
    if not df_dob_ecb.empty and "event_date" in df_dob_ecb.columns:
        ecb_prior = df_dob_ecb[df_dob_ecb["event_date"] < cutoff]
        w365 = cutoff - pd.Timedelta(days=365)
        w1095 = cutoff - pd.Timedelta(days=1095)

        cnt_ecb_total = ecb_prior.groupby("bbl").size().rename("dob_ecb_viol_total")
        cnt_ecb_12m = (
            ecb_prior[ecb_prior["event_date"] >= w365]
            .groupby("bbl")
            .size()
            .rename("dob_ecb_viol_12m")
        )
        cnt_ecb_36m = (
            ecb_prior[ecb_prior["event_date"] >= w1095]
            .groupby("bbl")
            .size()
            .rename("dob_ecb_viol_36m")
        )
        last_ecb_date = ecb_prior.groupby("bbl")["event_date"].max()
        days_since_ecb = (
            (cutoff - last_ecb_date).dt.total_seconds() / 86400.0
        ).rename("days_since_last_dob_ecb_viol")

        ecb_aggs = pd.concat(
            [cnt_ecb_total, cnt_ecb_12m, cnt_ecb_36m, days_since_ecb], axis=1
        )
        ecb_aggs.index.name = "bbl"
        ecb_aggs = ecb_aggs.reset_index()

        df_feat = df_feat.merge(ecb_aggs, on="bbl", how="left")
        df_feat["dob_ecb_viol_total"] = (
            df_feat["dob_ecb_viol_total"].fillna(0.0).astype(np.float32)
        )
        df_feat["dob_ecb_viol_12m"] = (
            df_feat["dob_ecb_viol_12m"].fillna(0.0).astype(np.float32)
        )
        df_feat["dob_ecb_viol_36m"] = (
            df_feat["dob_ecb_viol_36m"].fillna(0.0).astype(np.float32)
        )
        df_feat["days_since_last_dob_ecb_viol"] = (
            df_feat["days_since_last_dob_ecb_viol"]
            .fillna(3650.0)
            .clip(0, 3650.0)
            .astype(np.float32)
        )
    else:
        df_feat["dob_ecb_viol_total"] = np.float32(0.0)
        df_feat["dob_ecb_viol_12m"] = np.float32(0.0)
        df_feat["dob_ecb_viol_36m"] = np.float32(0.0)
        df_feat["days_since_last_dob_ecb_viol"] = np.float32(3650.0)

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

    # 3.4 Point-in-Time Rodent, Bedbug & Eviction Features
    if not df_rodent.empty and "event_date" in df_rodent.columns:
        rod_prior = df_rodent[df_rodent["event_date"] < cutoff]
        rod_fails = rod_prior[rod_prior["is_fail"] > 0]
        w365 = cutoff - pd.Timedelta(days=365)
        w1095 = cutoff - pd.Timedelta(days=1095)
        cnt_rod_12m = (
            rod_fails[rod_fails["event_date"] >= w365]
            .groupby("bbl")
            .size()
            .rename("rodent_failures_12m")
        )
        cnt_rod_36m = (
            rod_fails[rod_fails["event_date"] >= w1095]
            .groupby("bbl")
            .size()
            .rename("rodent_failures_36m")
        )
        cnt_rod_all = rod_fails.groupby("bbl").size().rename("rodent_failures_all")
        rod_aggs = pd.concat(
            [cnt_rod_12m, cnt_rod_36m, cnt_rod_all], axis=1
        ).reset_index()
        df_feat = df_feat.merge(rod_aggs, on="bbl", how="left")
        df_feat["rodent_failures_12m"] = (
            df_feat["rodent_failures_12m"].fillna(0.0).astype(np.float32)
        )
        df_feat["rodent_failures_36m"] = (
            df_feat["rodent_failures_36m"].fillna(0.0).astype(np.float32)
        )
        df_feat["rodent_failures_all"] = (
            df_feat["rodent_failures_all"].fillna(0.0).astype(np.float32)
        )
    else:
        df_feat["rodent_failures_12m"] = np.float32(0.0)
        df_feat["rodent_failures_36m"] = np.float32(0.0)
        df_feat["rodent_failures_all"] = np.float32(0.0)

    if not df_bb.empty and "event_date" in df_bb.columns:
        bb_prior = df_bb[df_bb["event_date"] < cutoff]
        w365 = cutoff - pd.Timedelta(days=365)
        bb_12m = bb_prior[bb_prior["event_date"] >= w365]
        cnt_bb_units_12m = (
            bb_12m.groupby("bbl")["infested_units"]
            .sum()
            .rename("bedbug_infested_units_12m")
        )
        cnt_bb_reports_12m = bb_12m.groupby("bbl").size().rename("bedbug_reports_12m")
        bb_aggs = pd.concat(
            [cnt_bb_units_12m, cnt_bb_reports_12m], axis=1
        ).reset_index()
        df_feat = df_feat.merge(bb_aggs, on="bbl", how="left")
        df_feat["bedbug_infested_units_12m"] = (
            df_feat["bedbug_infested_units_12m"].fillna(0.0).astype(np.float32)
        )
        df_feat["bedbug_reports_12m"] = (
            df_feat["bedbug_reports_12m"].fillna(0.0).astype(np.float32)
        )
    else:
        df_feat["bedbug_infested_units_12m"] = np.float32(0.0)
        df_feat["bedbug_reports_12m"] = np.float32(0.0)

    if not df_evict.empty and "event_date" in df_evict.columns:
        ev_prior = df_evict[df_evict["event_date"] < cutoff]
        w730 = cutoff - pd.Timedelta(days=730)
        ev_24m = ev_prior[ev_prior["event_date"] >= w730]
        cnt_evict_24m = ev_24m.groupby("bbl").size().rename("evictions_count_24m")
        cnt_evict_all = ev_prior.groupby("bbl").size().rename("evictions_count_total")
        ev_aggs = pd.concat([cnt_evict_24m, cnt_evict_all], axis=1).reset_index()
        df_feat = df_feat.merge(ev_aggs, on="bbl", how="left")
        df_feat["evictions_count_24m"] = (
            df_feat["evictions_count_24m"].fillna(0.0).astype(np.float32)
        )
        df_feat["evictions_count_total"] = (
            df_feat["evictions_count_total"].fillna(0.0).astype(np.float32)
        )
    else:
        df_feat["evictions_count_24m"] = np.float32(0.0)
        df_feat["evictions_count_total"] = np.float32(0.0)

    # 3.5 Statutory Distress Programs
    if not df_uc.empty:
        uc_prior = df_uc[df_uc["event_date"].isna() | (df_uc["event_date"] < cutoff)]
        uc_bbls = set(uc_prior["bbl"].dropna().unique())
        df_feat["is_underlying_conditions"] = (
            df_feat["bbl"].isin(uc_bbls).astype(np.float32)
        )
    else:
        df_feat["is_underlying_conditions"] = np.float32(0.0)

    if not df_conh.empty:
        conh_prior = df_conh[
            df_conh["event_date"].isna() | (df_conh["event_date"] < cutoff)
        ]
        conh_bbls = set(conh_prior["bbl"].dropna().unique())
        df_feat["is_conh_building"] = df_feat["bbl"].isin(conh_bbls).astype(np.float32)
    else:
        df_feat["is_conh_building"] = np.float32(0.0)

    if not df_swl.empty:
        swl_prior = df_swl[
            df_swl["event_date"].isna() | (df_swl["event_date"] < cutoff)
        ]
        swl_bbls = set(swl_prior["bbl"].dropna().unique())
        df_feat["is_speculation_watchlist"] = (
            df_feat["bbl"].isin(swl_bbls).astype(np.float32)
        )
    else:
        df_feat["is_speculation_watchlist"] = np.float32(0.0)

    df_feat["statutory_distress_total"] = (
        df_feat["has_vacate_order"]
        + df_feat["is_aep_distressed"]
        + df_feat["is_underlying_conditions"]
        + df_feat["is_conh_building"]
        + df_feat["is_speculation_watchlist"]
    ).astype(np.float32)

    # 3.6 Unit-Normalized Rates & Winter Season Velocity Dynamics
    df_feat["vio_c_rate_12m"] = (
        df_feat["vio_c_count_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["vio_b_rate_12m"] = (
        df_feat["vio_b_count_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["open_vio_c_rate"] = (
        df_feat["open_violations_c"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["open_vio_b_rate"] = (
        df_feat["open_violations_b"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["vio_all_rate_12m"] = (
        df_feat["vio_all_count_12m"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["open_vio_all_rate"] = (
        df_feat["open_violations_all"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)

    # Winter heat-season (Q4 / 90d) velocity dynamics
    df_feat["winter_complaint_acceleration"] = (
        df_feat["complaints_90d"]
        / (
            ((df_feat["complaints_12m"] - df_feat["complaints_90d"]).clip(lower=0) / 3.0)
            + 1.0
        )
    ).astype(np.float32)
    df_feat["winter_complaint_rate"] = (
        df_feat["complaints_90d"] / (df_feat["unitsres"] + 1e-4)
    ).astype(np.float32)
    df_feat["winter_complaint_share"] = (
        df_feat["complaints_90d"] / (df_feat["complaints_12m"] + 1.0)
    ).astype(np.float32)
    df_feat["winter_vio_c_acceleration"] = (
        df_feat["vio_c_count_90d"]
        / (
            ((df_feat["vio_c_count_12m"] - df_feat["vio_c_count_90d"]).clip(lower=0) / 3.0)
            + 1.0
        )
    ).astype(np.float32)

    return df_feat


def compute_cohort_target(
    target_bbls: np.ndarray,
    cutoff: pd.Timestamp,
    return_severity: bool = False,
) -> Any:
    """Computes ground truth target(s) for the 12-month forward window [cutoff, cutoff + 365d)."""
    w_start = cutoff
    w_end = cutoff + pd.DateOffset(years=1)
    c_forward = df_vio[
        (df_vio["class"] == "C")
        & (df_vio["inspectiondate"] >= w_start)
        & (df_vio["inspectiondate"] < w_end)
    ]
    cnt_map = c_forward.groupby("bbl").size().to_dict()
    counts = pd.Series(target_bbls).map(cnt_map).fillna(0).values.astype(np.float32)
    binary_target = (counts > 0).astype(np.int32)
    if return_severity:
        forward_c_severity = np.log1p(counts).astype(np.float32)
        return binary_target, forward_c_severity
    return binary_target


# Construct Datasets
print("Engineering features for Train Cohort 2020 (Cutoff: 2020-01-01)...")
df_train_2020 = extract_temporal_features(
    train_2020_bbls, T_TRAIN_2020, pluto_release_ver="19v2"
)
(
    df_train_2020["target"],
    df_train_2020["forward_c_severity"],
) = compute_cohort_target(train_2020_bbls, T_TRAIN_2020, return_severity=True)
print(
    f"Train 2020 cohort ready: {df_train_2020.shape} | Positive rate:"
    f" {df_train_2020['target'].mean():.4f}"
)

print("Engineering features for Train Cohort 2021 (Cutoff: 2021-01-01)...")
df_train_2021 = extract_temporal_features(
    train_2021_bbls, T_TRAIN_2021, pluto_release_ver="20v7"
)
(
    df_train_2021["target"],
    df_train_2021["forward_c_severity"],
) = compute_cohort_target(train_2021_bbls, T_TRAIN_2021, return_severity=True)
print(
    f"Train 2021 cohort ready: {df_train_2021.shape} | Positive rate:"
    f" {df_train_2021['target'].mean():.4f}"
)

df_train = pd.concat([df_train_2020, df_train_2021], ignore_index=True)
print(
    f"Pooled df_train ready: {df_train.shape} | Positive rate:"
    f" {df_train['target'].mean():.4f}"
)

print("Engineering features for Validation Cohort (Cutoff: 2022-01-01)...")
df_val = extract_temporal_features(val_bbls, T_VAL, pluto_release_ver="21v4")
df_val["target"], df_val["forward_c_severity"] = compute_cohort_target(
    val_bbls, T_VAL, return_severity=True
)
print(
    f"Validation cohort ready: {df_val.shape} | Positive rate:"
    f" {df_val['target'].mean():.4f}"
)

print("Engineering features for Test Cohort (Cutoff: 2023-01-01)...")
df_test = extract_temporal_features(test_bbls, T_TEST, pluto_release_ver="22v3")
print(f"Test cohort ready: {df_test.shape}")

# Hierarchical spatial distress aggregations strictly computed from training distribution
for df_c in [df_train, df_val, df_test]:
    df_c["block_id"] = df_c["bbl"].astype(str).str.zfill(10).str[:6]
    if "zipcode" in df_c.columns:
        df_c["zipcode"] = df_c["zipcode"].astype(str)
    if "cd" not in df_c.columns:
        df_c["cd"] = 0

block_c_risk_map = df_train.groupby("block_id")["vio_c_count_12m"].mean().to_dict()
default_block_c_risk = float(df_train["vio_c_count_12m"].mean())

cd_c_risk_map = df_train.groupby("cd")["vio_c_count_12m"].mean().to_dict()
default_cd_c_risk = float(df_train["vio_c_count_12m"].mean())

zip_risk_map = df_train.groupby("zipcode")["vio_c_count_12m"].mean().to_dict()
default_zip_risk = float(df_train["vio_c_count_12m"].mean())

for df_c in [df_train, df_val, df_test]:
    df_c["block_historical_c_risk"] = (
        df_c["block_id"]
        .map(block_c_risk_map)
        .fillna(default_block_c_risk)
        .astype(np.float32)
    )
    df_c["cd_historical_c_risk"] = (
        df_c["cd"]
        .map(cd_c_risk_map)
        .fillna(default_cd_c_risk)
        .astype(np.float32)
    )
    df_c["zip_historical_c_risk"] = (
        df_c["zipcode"]
        .map(zip_risk_map)
        .fillna(default_zip_risk)
        .astype(np.float32)
    )

# Align feature sets
exclude_cols = {
    "bbl",
    "target",
    "forward_c_severity",
    "registrationid",
    "bldgclass",
    "zipcode",
    "cd",
    "block_id",
}
feature_columns = sorted([c for c in df_train.columns if c not in exclude_cols])

for c in feature_columns:
    if c not in df_val.columns:
        df_val[c] = np.float32(0.0)
    if c not in df_test.columns:
        df_test[c] = np.float32(0.0)
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
# 4. MODEL ARCHITECTURES & HYPERPARAMETER CONFIGURATION
# ==============================================================================
def get_lgbm_model_config(
    pos_scale_weight: float = 2.5, random_state: int = 42
) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "average_precision",
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


def get_xgb_model_config(
    pos_scale_weight: float = 2.5, random_state: int = 42
) -> Dict[str, Any]:
    return {
        "tree_method": "hist",
        "max_depth": 7,
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "scale_pos_weight": float(pos_scale_weight),
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "eval_metric": "aucpr",
        "random_state": random_state,
        "n_jobs": -1,
    }


def get_catboost_model_config(
    pos_scale_weight: float = 2.5, random_state: int = 42
) -> Dict[str, Any]:
    return {
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "scale_pos_weight": float(pos_scale_weight),
        "eval_metric": "PRAUC",
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }


def get_lgbm_severity_regressor_config(
    random_state: int = 42,
) -> Dict[str, Any]:
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": 7,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
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

# Temporal recency sample weighting: discount 2020 pandemic cohort to align enforcement regime
sample_weight = np.concatenate(
    [
        np.full(len(df_train_2020), 0.6, dtype=np.float32),
        np.full(len(df_train_2021), 1.0, dtype=np.float32),
    ]
)
print(
    f"Constructed temporal sample weights: shape={sample_weight.shape} | "
    f"2020 weight=0.6 ({len(df_train_2020)}), 2021 weight=1.0 ({len(df_train_2021)})"
)

y_train_severity = df_train["forward_c_severity"].values.astype(np.float32)

X_val = np.nan_to_num(
    df_val[feature_columns].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_val = df_val["target"].values.astype(np.float32)
y_val_severity = df_val["forward_c_severity"].values.astype(np.float32)

X_test = np.nan_to_num(
    df_test[feature_columns].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# 5.1 Train LightGBM
print("Training LightGBM Classifier with Temporal Sample Weighting...")
lgb_config = get_lgbm_model_config(pos_scale_weight=2.5, random_state=42)
lgb_model = lgb.LGBMClassifier(**lgb_config)

lgb_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
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

# 5.2 Train Depthwise XGBoost Classifier
print("Training Regularized Depthwise XGBoost Classifier with Temporal Sample Weighting...")
xgb_config = get_xgb_model_config(pos_scale_weight=2.5, random_state=42)
try:
    xgb_model = xgb.XGBClassifier(early_stopping_rounds=50, **xgb_config)
    xgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
except (TypeError, ValueError):
    xgb_model = xgb.XGBClassifier(**xgb_config)
    xgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_val_ap = average_precision_score(y_val, val_preds_xgb)
print(f"XGBoost Validation Average Precision: {xgb_val_ap:.5f}")

joblib.dump(xgb_model, os.path.join(WORKING_DIR, "xgb_best_model.pkl"))

# 5.3 Train CatBoost Classifier with Oblivious Decision Trees
print("Training CatBoost Classifier with Temporal Sample Weighting...")
cat_config = get_catboost_model_config(pos_scale_weight=2.5, random_state=42)
cat_model = CatBoostClassifier(early_stopping_rounds=50, **cat_config)
cat_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    eval_set=(X_val, y_val),
    verbose=False,
)
val_preds_cat = cat_model.predict_proba(X_val)[:, 1]
test_preds_cat = cat_model.predict_proba(X_test)[:, 1]
cat_val_ap = average_precision_score(y_val, val_preds_cat)
print(f"CatBoost Validation Average Precision: {cat_val_ap:.5f}")

joblib.dump(cat_model, os.path.join(WORKING_DIR, "cat_best_model.pkl"))

# 5.4 Train LightGBM Severity Regressor with Temporal Sample Weighting
print("Training LightGBM Severity Regressor with Temporal Sample Weighting...")
reg_config = get_lgbm_severity_regressor_config(random_state=42)
lgb_reg_model = lgb.LGBMRegressor(**reg_config)
lgb_reg_model.fit(
    X_train,
    y_train_severity,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val_severity)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
val_preds_reg = lgb_reg_model.predict(X_val)
test_preds_reg = lgb_reg_model.predict(X_test)
reg_val_ap = average_precision_score(y_val, val_preds_reg)
print(f"LightGBM Severity Regressor Validation Average Precision: {reg_val_ap:.5f}")

joblib.dump(lgb_reg_model, os.path.join(WORKING_DIR, "lgb_reg_model.pkl"))

# 5.5 4-Way Optimal Rank Normalization Ensemble
rank_val_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)
rank_val_xgb = rankdata(val_preds_xgb) / len(val_preds_xgb)
rank_val_cat = rankdata(val_preds_cat) / len(val_preds_cat)
rank_val_reg = rankdata(val_preds_reg) / len(val_preds_reg)

rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
rank_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
rank_test_cat = rankdata(test_preds_cat) / len(test_preds_cat)
rank_test_reg = rankdata(test_preds_reg) / len(test_preds_reg)

best_ap = -1.0
best_weights = (0.35, 0.25, 0.20, 0.20)
step = 0.05
steps = int(round(1.0 / step)) + 1
for i in range(steps):
    w1 = round(i * step, 4)
    rem1 = round(1.0 - w1, 4)
    for j in range(int(round(rem1 / step)) + 1):
        w2 = round(j * step, 4)
        rem2 = round(rem1 - w2, 4)
        for k in range(int(round(rem2 / step)) + 1):
            w3 = round(k * step, 4)
            w4 = max(0.0, round(rem2 - w3, 4))
            blend_cand = (
                w1 * rank_val_lgb
                + w2 * rank_val_xgb
                + w3 * rank_val_cat
                + w4 * rank_val_reg
            )
            cand_score = average_precision_score(y_val, blend_cand)
            if cand_score > best_ap:
                best_ap = cand_score
                best_weights = (float(w1), float(w2), float(w3), float(w4))

# Fine local simplex search around initial optimum
bw1, bw2, bw3, bw4 = best_weights
delta = 0.04
w1_range = np.arange(max(0.0, bw1 - delta), min(1.0, bw1 + delta + 1e-5), 0.01)
w2_range = np.arange(max(0.0, bw2 - delta), min(1.0, bw2 + delta + 1e-5), 0.01)
w3_range = np.arange(max(0.0, bw3 - delta), min(1.0, bw3 + delta + 1e-5), 0.01)
for nw1 in w1_range:
    for nw2 in w2_range:
        for nw3 in w3_range:
            nw4 = round(1.0 - nw1 - nw2 - nw3, 4)
            if 0.0 <= nw4 <= 1.0 and abs(nw4 - bw4) <= delta + 1e-5:
                blend_cand = (
                    nw1 * rank_val_lgb
                    + nw2 * rank_val_xgb
                    + nw3 * rank_val_cat
                    + nw4 * rank_val_reg
                )
                cand_score = average_precision_score(y_val, blend_cand)
                if cand_score > best_ap:
                    best_ap = cand_score
                    best_weights = (float(nw1), float(nw2), float(nw3), float(nw4))

w_lgb, w_xgb, w_cat, w_reg = best_weights
print(
    f"Optimal Ensemble Weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, CAT={w_cat:.3f}, REG={w_reg:.3f}"
    f" | Best Val AP: {best_ap:.5f}"
)

val_ensemble = (
    w_lgb * rank_val_lgb
    + w_xgb * rank_val_xgb
    + w_cat * rank_val_cat
    + w_reg * rank_val_reg
)
test_ensemble = (
    w_lgb * rank_test_lgb
    + w_xgb * rank_test_xgb
    + w_cat * rank_test_cat
    + w_reg * rank_test_reg
)

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

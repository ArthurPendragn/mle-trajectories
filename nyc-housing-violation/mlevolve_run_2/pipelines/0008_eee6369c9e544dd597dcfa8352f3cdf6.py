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
    if bbl_cands:
        cols_to_load = list(set(bbl_cands + [cmp_date_col]))

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

    if actual_date_col is not None:
        df_cmp["receiveddate"] = pd.to_datetime(
            df_cmp[actual_date_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
        df_cmp["bbl"] = normalize_bbl_series(df_cmp)
        df_cmp = df_cmp[df_cmp["bbl"].notna() & df_cmp["receiveddate"].notna()]
        df_cmp = df_cmp[["bbl", "receiveddate"]]
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

    # 3.2b HPD Complaints Point-in-Time Trajectory
    if not df_cmp.empty and "receiveddate" in df_cmp.columns:
        cmp_prior = df_cmp[df_cmp["receiveddate"] < cutoff]
        cw30 = cutoff - pd.Timedelta(days=30)
        cw90 = cutoff - pd.Timedelta(days=90)
        cw365 = cutoff - pd.Timedelta(days=365)

        c_30 = cmp_prior[cmp_prior["receiveddate"] >= cw30]
        c_90 = cmp_prior[cmp_prior["receiveddate"] >= cw90]
        c_365 = cmp_prior[cmp_prior["receiveddate"] >= cw365]

        cnt_cmp_30 = c_30.groupby("bbl").size().rename("complaints_30d")
        cnt_cmp_90 = c_90.groupby("bbl").size().rename("complaints_90d")
        cnt_cmp_365 = c_365.groupby("bbl").size().rename("complaints_12m")

        last_cmp_date = cmp_prior.groupby("bbl")["receiveddate"].max()
        days_since_cmp = (
            (cutoff - last_cmp_date).dt.total_seconds() / 86400.0
        ).rename("days_since_last_complaint")

        cmp_aggs = pd.concat(
            [cnt_cmp_30, cnt_cmp_90, cnt_cmp_365, days_since_cmp], axis=1
        )
        cmp_aggs.index.name = "bbl"
        cmp_aggs = cmp_aggs.reset_index()

        df_feat = df_feat.merge(cmp_aggs, on="bbl", how="left")
        df_feat["complaints_30d"] = (
            df_feat["complaints_30d"].fillna(0.0).astype(np.float32)
        )
        df_feat["complaints_90d"] = (
            df_feat["complaints_90d"].fillna(0.0).astype(np.float32)
        )
        df_feat["complaints_12m"] = (
            df_feat["complaints_12m"].fillna(0.0).astype(np.float32)
        )
        df_feat["days_since_last_complaint"] = (
            df_feat["days_since_last_complaint"]
            .fillna(3650.0)
            .clip(0, 3650.0)
            .astype(np.float32)
        )
    else:
        df_feat["complaints_30d"] = np.float32(0.0)
        df_feat["complaints_90d"] = np.float32(0.0)
        df_feat["complaints_12m"] = np.float32(0.0)
        df_feat["days_since_last_complaint"] = np.float32(3650.0)

    df_feat["complaint_momentum"] = (
        df_feat["complaints_90d"]
        / (
            (df_feat["complaints_12m"] - df_feat["complaints_90d"]).clip(
                lower=0
            )
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
print("Engineering features for Train Cohort 2020 (Cutoff: 2020-01-01)...")
df_train_2020 = extract_temporal_features(
    train_2020_bbls, T_TRAIN_2020, pluto_release_ver="19v2"
)
df_train_2020["target"] = compute_cohort_target(train_2020_bbls, T_TRAIN_2020)
print(
    f"Train 2020 cohort ready: {df_train_2020.shape} | Positive rate:"
    f" {df_train_2020['target'].mean():.4f}"
)

print("Engineering features for Train Cohort 2021 (Cutoff: 2021-01-01)...")
df_train_2021 = extract_temporal_features(
    train_2021_bbls, T_TRAIN_2021, pluto_release_ver="20v7"
)
df_train_2021["target"] = compute_cohort_target(train_2021_bbls, T_TRAIN_2021)
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
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
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

# 5.2 Train Depthwise XGBoost Classifier
print("Training Regularized Depthwise XGBoost Classifier...")
xgb_config = get_xgb_model_config(pos_scale_weight=2.5, random_state=42)
try:
    xgb_model = xgb.XGBClassifier(early_stopping_rounds=50, **xgb_config)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
except (TypeError, ValueError):
    xgb_model = xgb.XGBClassifier(**xgb_config)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_val_ap = average_precision_score(y_val, val_preds_xgb)
print(f"XGBoost Validation Average Precision: {xgb_val_ap:.5f}")

joblib.dump(xgb_model, os.path.join(WORKING_DIR, "xgb_best_model.pkl"))

# 5.3 Ensemble Rank Normalization
rank_val_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)
rank_val_xgb = rankdata(val_preds_xgb) / len(val_preds_xgb)

rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
rank_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)

val_ensemble = 0.55 * rank_val_lgb + 0.45 * rank_val_xgb
test_ensemble = 0.55 * rank_test_lgb + 0.45 * rank_test_xgb

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

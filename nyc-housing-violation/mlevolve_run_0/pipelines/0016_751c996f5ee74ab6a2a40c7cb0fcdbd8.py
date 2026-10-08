import copy
import json
import os
import re
from catboost import CatBoostClassifier
import gcsfs
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
        boro_map = {
            "MANHATTAN": "1",
            "MN": "1",
            "BRONX": "2",
            "BX": "2",
            "BROOKLYN": "3",
            "BK": "3",
            "QUEENS": "4",
            "QN": "4",
            "STATEN ISLAND": "5",
            "SI": "5",
        }
        b_raw = df[boro_col].astype(str).str.strip().str.upper()
        b_mapped = b_raw.map(boro_map)
        b_num = (
            pd.to_numeric(df[boro_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
            .str.strip()
            .str.zfill(1)
        )
        b = b_mapped.fillna(b_num)
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
    "ownername",
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
df_train_2021_entities = load_pluto_release("20v7", df_test)
df_train_2020_entities = load_pluto_release("19v2", df_test)
df_train_2019_entities = load_pluto_release("19v2", df_test)

# ---------------------------------------------------------
# Load HPD Violations (2015-2022)
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
    "apartment",
    "originalcorrectbydate",
]

for cols_candidate in [
    desired_viol_cols,
    [c for c in desired_viol_cols if c not in ["apartment", "originalcorrectbydate"]],
    None,
]:
    for flt in [[("year", ">=", 2015)], [("year", ">=", 2016)], None]:
        try:
            if flt:
                df_viol = pd.read_parquet(
                    hpd_viol_path,
                    columns=cols_candidate,
                    filters=flt,
                    storage_options=storage_options,
                )
            else:
                df_viol = pd.read_parquet(
                    hpd_viol_path,
                    columns=cols_candidate,
                    storage_options=storage_options,
                )
            if len(df_viol) > 0:
                break
        except Exception:
            continue
    if "df_viol" in locals() and len(df_viol) > 0:
        break

if "df_viol" not in locals() or len(df_viol) == 0:
    df_viol = pd.read_parquet(hpd_viol_path, storage_options=storage_options)

df_viol.columns = [c.lower() for c in df_viol.columns]
df_viol["clean_bbl"] = clean_bbl_series(df_viol)
df_viol["class_clean"] = df_viol["class"].astype(str).str.strip().str.upper()
df_viol["inspectiondate_dt"] = pd.to_datetime(
    df_viol["inspectiondate"], errors="coerce"
)
df_viol = df_viol.dropna(subset=["inspectiondate_dt"]).copy()

if "apartment" in df_viol.columns:
    df_viol["apartment"] = df_viol["apartment"].fillna("").astype(str).str.strip().str.upper()
else:
    df_viol["apartment"] = ""

if "originalcorrectbydate" in df_viol.columns:
    df_viol["correctby_dt"] = pd.to_datetime(
        df_viol["originalcorrectbydate"], errors="coerce"
    )
else:
    df_viol["correctby_dt"] = pd.NaT

# ---------------------------------------------------------
# Load HPD Complaints
# ---------------------------------------------------------
def load_complaints_table():
    """Discover and ingest HPD complaints table from lake."""
    fs = gcsfs.GCSFileSystem(token=TOKEN_PATH if os.path.exists(TOKEN_PATH) else None)
    candidate_paths = [
        f"{LAKE_FULL}/hpd_complaints",
        f"{LAKE_FULL}/complaints",
        f"{LAKE_FULL}/hpd_complaint_problems",
        f"{LAKE_FULL}/hpd_complaints_problems",
    ]
    try:
        entries = fs.ls("mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full")
        for entry in entries:
            name = entry.split("/")[-1].lower()
            if "complaint" in name:
                candidate_paths.insert(0, f"gs://{entry}")
    except Exception:
        pass

    for p in candidate_paths:
        for flt in [[("year", ">=", 2015)], [("year", ">=", 2016)], None]:
            try:
                if flt:
                    df = pd.read_parquet(p, filters=flt, storage_options=storage_options)
                else:
                    df = pd.read_parquet(p, storage_options=storage_options)
                if len(df) > 0:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


df_complaints = load_complaints_table()
if len(df_complaints) > 0:
    df_complaints.columns = [c.lower() for c in df_complaints.columns]
    df_complaints["clean_bbl"] = clean_bbl_series(df_complaints)
    date_col = next(
        (
            c
            for c in [
                "receiveddate",
                "dateentered",
                "complaintdate",
                "statusdate",
                "inspectiondate",
                "date",
            ]
            if c in df_complaints.columns
        ),
        None,
    )
    if date_col is None:
        date_col = next((c for c in df_complaints.columns if "date" in c), None)

    if date_col is not None:
        df_complaints["complaint_dt"] = pd.to_datetime(
            df_complaints[date_col], errors="coerce"
        )
        df_complaints = df_complaints.dropna(subset=["complaint_dt"]).copy()
    else:
        df_complaints["complaint_dt"] = pd.NaT

    status_col = next(
        (
            c
            for c in [
                "status",
                "complaintstatus",
                "statusdescription",
                "currentstatus",
            ]
            if c in df_complaints.columns
        ),
        None,
    )
    if status_col is not None:
        df_complaints["is_open"] = (
            df_complaints[status_col]
            .astype(str)
            .str.upper()
            .str.contains("OPEN|ACTIVE|PENDING", regex=True)
        )
    else:
        df_complaints["is_open"] = False


# ---------------------------------------------------------
# Load HPD Litigations
# ---------------------------------------------------------
def load_litigations_table():
    """Discover and ingest HPD Housing Court litigations table from lake."""
    fs = gcsfs.GCSFileSystem(token=TOKEN_PATH if os.path.exists(TOKEN_PATH) else None)
    candidate_paths = [
        f"{LAKE_FULL}/hpd_litigations",
        f"{LAKE_FULL}/litigations",
        f"{LAKE_FULL}/hpd_litigation",
        f"{LAKE_FULL}/housing_litigations",
    ]
    try:
        entries = fs.ls("mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full")
        for entry in entries:
            name = entry.split("/")[-1].lower()
            if "litig" in name:
                candidate_paths.insert(0, f"gs://{entry}")
    except Exception:
        pass

    for p in candidate_paths:
        for flt in [None, [("year", ">=", 2015)], [("year", ">=", 2016)]]:
            try:
                if flt:
                    df = pd.read_parquet(p, filters=flt, storage_options=storage_options)
                else:
                    df = pd.read_parquet(p, storage_options=storage_options)
                if len(df) > 0:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


df_litigations = load_litigations_table()
if len(df_litigations) > 0:
    df_litigations.columns = [c.lower() for c in df_litigations.columns]
    df_litigations["clean_bbl"] = clean_bbl_series(df_litigations)
    date_col = next(
        (
            c
            for c in [
                "caseopendate",
                "opendate",
                "litigationdate",
                "dateentered",
                "litdate",
                "date",
            ]
            if c in df_litigations.columns
        ),
        None,
    )
    if date_col is None:
        date_col = next(
            (c for c in df_litigations.columns if "open" in c and "date" in c), None
        )
    if date_col is None:
        date_col = next((c for c in df_litigations.columns if "date" in c), None)

    if date_col is not None:
        df_litigations["lit_dt"] = pd.to_datetime(
            df_litigations[date_col], errors="coerce"
        )
        df_litigations = df_litigations.dropna(subset=["lit_dt"]).copy()
    else:
        df_litigations["lit_dt"] = pd.NaT

    status_col = next(
        (
            c
            for c in [
                "casestatus",
                "status",
                "litigationstatus",
                "currentstatus",
            ]
            if c in df_litigations.columns
        ),
        None,
    )
    close_col = next(
        (
            c
            for c in [
                "caseclosedate",
                "closedate",
                "dateclosed",
            ]
            if c in df_litigations.columns
        ),
        None,
    )
    if close_col is not None:
        df_litigations["lit_close_dt"] = pd.to_datetime(
            df_litigations[close_col], errors="coerce"
        )
    else:
        df_litigations["lit_close_dt"] = pd.NaT

    if status_col is not None:
        df_litigations["is_open_lit"] = (
            df_litigations[status_col]
            .astype(str)
            .str.upper()
            .str.contains("OPEN|ACTIVE|PENDING", regex=True)
        )
    else:
        if close_col is not None:
            df_litigations["is_open_lit"] = df_litigations["lit_close_dt"].isna()
        else:
            df_litigations["is_open_lit"] = False


# ---------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------
def extract_point_in_time_features(df_entities, df_violations, cutoff_str):
    """Strictly point-in-time feature extraction for entities at a specified cutoff date."""
    t_cutoff = pd.Timestamp(cutoff_str)
    cutoff_year = t_cutoff.year

    t_30d = t_cutoff - pd.Timedelta(days=30)
    t_60d = t_cutoff - pd.Timedelta(days=60)
    t_3m = t_cutoff - pd.DateOffset(months=3)
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

    if "correctby_dt" in df_h.columns:
        is_overdue = is_open & df_h["correctby_dt"].notna() & (df_h["correctby_dt"] < t_cutoff)
    else:
        is_overdue = pd.Series(False, index=df_h.index)

    dates = df_h["inspectiondate_dt"]
    is_30d = dates >= t_30d
    is_60d = dates >= t_60d
    is_3m = dates >= t_3m
    is_6m = dates >= t_6m
    is_1y = dates >= t_1y
    is_2y = dates >= t_2y
    is_3y = dates >= t_3y
    is_5y = dates >= t_5y

    df_h["c_30d"] = (is_c & is_30d).astype(np.int32)
    df_h["c_60d"] = (is_c & is_60d).astype(np.int32)
    df_h["c_3m"] = (is_c & is_3m).astype(np.int32)
    df_h["c_6m"] = (is_c & is_6m).astype(np.int32)
    df_h["c_1y"] = (is_c & is_1y).astype(np.int32)
    df_h["c_2y"] = (is_c & is_2y).astype(np.int32)
    df_h["c_3y"] = (is_c & is_3y).astype(np.int32)
    df_h["c_5y"] = (is_c & is_5y).astype(np.int32)
    df_h["c_all"] = is_c.astype(np.int32)
    df_h["c_open"] = (is_c & is_open).astype(np.int32)
    df_h["c_overdue"] = (is_c & is_overdue).astype(np.int32)

    df_h["b_30d"] = (is_b & is_30d).astype(np.int32)
    df_h["b_60d"] = (is_b & is_60d).astype(np.int32)
    df_h["b_3m"] = (is_b & is_3m).astype(np.int32)
    df_h["b_6m"] = (is_b & is_6m).astype(np.int32)
    df_h["b_1y"] = (is_b & is_1y).astype(np.int32)
    df_h["b_2y"] = (is_b & is_2y).astype(np.int32)
    df_h["b_3y"] = (is_b & is_3y).astype(np.int32)
    df_h["b_5y"] = (is_b & is_5y).astype(np.int32)
    df_h["b_all"] = is_b.astype(np.int32)
    df_h["b_open"] = (is_b & is_open).astype(np.int32)
    df_h["b_overdue"] = (is_b & is_overdue).astype(np.int32)

    df_h["a_1y"] = (is_a & is_1y).astype(np.int32)
    df_h["a_all"] = is_a.astype(np.int32)
    df_h["i_all"] = is_i.astype(np.int32)

    df_h["tot_30d"] = is_30d.astype(np.int32)
    df_h["tot_60d"] = is_60d.astype(np.int32)
    df_h["tot_6m"] = is_6m.astype(np.int32)
    df_h["tot_1y"] = is_1y.astype(np.int32)
    df_h["tot_2y"] = is_2y.astype(np.int32)
    df_h["tot_3y"] = is_3y.astype(np.int32)
    df_h["tot_all"] = 1
    df_h["tot_overdue"] = is_overdue.astype(np.int32)

    t_q4_start = pd.Timestamp(year=cutoff_year - 1, month=10, day=1)
    is_q4_heat = (dates >= t_q4_start) & (dates < t_cutoff)
    df_h["c_q4_heat"] = (is_c & is_q4_heat).astype(np.int32)
    df_h["tot_q4_heat"] = is_q4_heat.astype(np.int32)

    df_h["last_c_date"] = df_h["inspectiondate_dt"].where(is_c)
    df_h["last_any_date"] = df_h["inspectiondate_dt"]

    agg_funcs = {
        "c_30d": "sum",
        "c_60d": "sum",
        "c_3m": "sum",
        "c_6m": "sum",
        "c_1y": "sum",
        "c_2y": "sum",
        "c_3y": "sum",
        "c_5y": "sum",
        "c_all": "sum",
        "c_open": "sum",
        "c_overdue": "sum",
        "c_q4_heat": "sum",
        "b_30d": "sum",
        "b_60d": "sum",
        "b_3m": "sum",
        "b_6m": "sum",
        "b_1y": "sum",
        "b_2y": "sum",
        "b_3y": "sum",
        "b_5y": "sum",
        "b_all": "sum",
        "b_open": "sum",
        "b_overdue": "sum",
        "a_1y": "sum",
        "a_all": "sum",
        "i_all": "sum",
        "tot_30d": "sum",
        "tot_60d": "sum",
        "tot_6m": "sum",
        "tot_1y": "sum",
        "tot_2y": "sum",
        "tot_3y": "sum",
        "tot_all": "sum",
        "tot_overdue": "sum",
        "tot_q4_heat": "sum",
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

    df_c = df_h.loc[is_c, ["clean_bbl"]].copy()
    if len(df_c) > 0:
        df_c["viol_year"] = df_h.loc[is_c, "inspectiondate_dt"].dt.year
        c_years_agg = (
            df_c.drop_duplicates()
            .groupby("clean_bbl")["viol_year"]
            .count()
            .rename("c_active_years")
            .reset_index()
        )
        res = res.merge(c_years_agg, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])
        res["c_active_years"] = res["c_active_years"].fillna(0.0).astype(np.float32)
    else:
        res["c_active_years"] = np.zeros(len(res), dtype=np.float32)

    # Apartment-level breadth: distinct units with violations in past 1 year
    df_c_1y = df_h[is_c & is_1y & (df_h["apartment"] != "")]
    if len(df_c_1y) > 0:
        c_apts = (
            df_c_1y.groupby("clean_bbl")["apartment"]
            .nunique()
            .rename("c_unique_apts_1y")
            .reset_index()
        )
        res = res.merge(c_apts, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])
        res["c_unique_apts_1y"] = res["c_unique_apts_1y"].fillna(0.0).astype(np.float32)
    else:
        res["c_unique_apts_1y"] = np.zeros(len(res), dtype=np.float32)

    df_b_1y = df_h[is_b & is_1y & (df_h["apartment"] != "")]
    if len(df_b_1y) > 0:
        b_apts = (
            df_b_1y.groupby("clean_bbl")["apartment"]
            .nunique()
            .rename("b_unique_apts_1y")
            .reset_index()
        )
        res = res.merge(b_apts, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])
        res["b_unique_apts_1y"] = res["b_unique_apts_1y"].fillna(0.0).astype(np.float32)
    else:
        res["b_unique_apts_1y"] = np.zeros(len(res), dtype=np.float32)

    # De-clustered inspection visit metrics: unique inspection dates per BBL in past 1 year
    df_c_1y_sub = df_h[is_c & is_1y]
    if len(df_c_1y_sub) > 0:
        c_visits_agg = (
            df_c_1y_sub.groupby("clean_bbl")["inspectiondate_dt"]
            .nunique()
            .rename("c_inspection_visits_1y")
            .reset_index()
        )
        res = res.merge(c_visits_agg, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])
        res["c_inspection_visits_1y"] = (
            res["c_inspection_visits_1y"].fillna(0.0).astype(np.float32)
        )
    else:
        res["c_inspection_visits_1y"] = np.zeros(len(res), dtype=np.float32)

    df_tot_1y_sub = df_h[is_1y]
    if len(df_tot_1y_sub) > 0:
        tot_visits_agg = (
            df_tot_1y_sub.groupby("clean_bbl")["inspectiondate_dt"]
            .nunique()
            .rename("tot_inspection_visits_1y")
            .reset_index()
        )
        res = res.merge(tot_visits_agg, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])
        res["tot_inspection_visits_1y"] = (
            res["tot_inspection_visits_1y"].fillna(0.0).astype(np.float32)
        )
    else:
        res["tot_inspection_visits_1y"] = np.zeros(len(res), dtype=np.float32)

    count_cols = [
        "c_30d",
        "c_60d",
        "c_3m",
        "c_6m",
        "c_1y",
        "c_2y",
        "c_3y",
        "c_5y",
        "c_all",
        "c_open",
        "c_overdue",
        "c_q4_heat",
        "b_30d",
        "b_60d",
        "b_3m",
        "b_6m",
        "b_1y",
        "b_2y",
        "b_3y",
        "b_5y",
        "b_all",
        "b_open",
        "b_overdue",
        "a_1y",
        "a_all",
        "i_all",
        "tot_30d",
        "tot_60d",
        "tot_6m",
        "tot_1y",
        "tot_2y",
        "tot_3y",
        "tot_all",
        "tot_overdue",
        "tot_q4_heat",
    ]
    res[count_cols] = res[count_cols].fillna(0).astype(np.float32)

    # Point-in-time block-level violation aggregates (6-digit key: boro + block)
    df_h["block_key"] = df_h["clean_bbl"].str[:6]
    block_agg = (
        df_h.groupby("block_key")
        .agg(
            block_c_1y=("c_1y", "sum"),
            block_c_2y=("c_2y", "sum"),
            block_tot_1y=("tot_1y", "sum"),
            block_tot_2y=("tot_2y", "sum"),
        )
        .reset_index()
    )

    res["block_key"] = res["bbl"].str[:6]
    res = res.merge(block_agg, on="block_key", how="left")
    res = res.drop(columns=["block_key"])

    res["block_c_1y"] = res["block_c_1y"].fillna(0.0).astype(np.float32)
    res["block_c_2y"] = res["block_c_2y"].fillna(0.0).astype(np.float32)
    res["block_tot_1y"] = res["block_tot_1y"].fillna(0.0).astype(np.float32)
    res["block_tot_2y"] = res["block_tot_2y"].fillna(0.0).astype(np.float32)

    res["lot_to_block_c_1y"] = (res["c_1y"] / (res["block_c_1y"] + 0.5)).astype(np.float32)
    res["lot_to_block_tot_1y"] = (res["tot_1y"] / (res["block_tot_1y"] + 0.5)).astype(np.float32)
    res["block_c_density"] = (res["block_c_1y"] / (res["block_tot_1y"] + 1e-4)).astype(np.float32)

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
    c_prev_1y = np.maximum(0.0, res["c_2y"] - res["c_1y"])
    c_prev_2y = np.maximum(0.0, res["c_3y"] - res["c_2y"])
    res["c_trend_1y"] = (res["c_1y"] - c_prev_1y).astype(np.float32)
    res["c_accel"] = (res["c_1y"] / (c_prev_1y + 0.5)).astype(np.float32)
    res["c_trajectory_accel"] = ((res["c_1y"] - c_prev_1y) - (c_prev_1y - c_prev_2y)).astype(np.float32)
    res["open_c_ratio"] = (res["c_open"] / (res["c_all"] + 1e-4)).astype(np.float32)
    res["c_overdue_ratio"] = (res["c_overdue"] / (res["c_open"] + 1e-4)).astype(np.float32)
    res["c_velocity"] = (res["c_3m"] / (res["c_1y"] / 4.0 + 0.1)).astype(np.float32)
    res["c_velocity_30d"] = (res["c_30d"] / (res["c_3m"] / 3.0 + 0.1)).astype(np.float32)
    res["c_violations_per_visit_1y"] = (
        res["c_1y"] / (res["c_inspection_visits_1y"] + 1e-4)
    ).astype(np.float32)
    res["c_chronic_flag"] = ((res["c_1y"] > 0) & (c_prev_1y > 0)).astype(np.float32)
    res["c_consecutive_3y_flag"] = (
        (res["c_1y"] > 0) & (c_prev_1y > 0) & (c_prev_2y > 0)
    ).astype(np.float32)
    res["c_resolution_efficiency_1y"] = (
        (res["c_1y"] - res["c_open"]) / (res["c_1y"] + 1e-4)
    ).astype(np.float32)
    res["bc_1y_sum"] = (res["c_1y"] + res["b_1y"]).astype(np.float32)
    res["c_to_b_ratio"] = (res["c_1y"] / (res["b_1y"] + 1e-4)).astype(np.float32)

    # Landlord resolution rates & chronic unresolved violations
    res["closed_c_ratio"] = (
        (res["c_all"] - res["c_open"]) / (res["c_all"] + 1e-4)
    ).astype(np.float32)
    res["chronic_unresolved_flag"] = (
        (res["c_open"] > 0) & (res["c_chronic_flag"] > 0)
    ).astype(np.float32)
    res["chronic_unresolved_c_count"] = np.where(
        res["c_chronic_flag"] > 0, res["c_open"], 0.0
    ).astype(np.float32)
    res["c_q4_ratio"] = (
        res["c_q4_heat"] / (res["c_1y"] + 1e-4)
    ).astype(np.float32)

    # Multi-class escalation dynamics
    b_prev_1y = np.maximum(0.0, res["b_2y"] - res["b_1y"])
    b_prev_2y = np.maximum(0.0, res["b_3y"] - res["b_2y"])
    res["b_accel"] = (res["b_1y"] / (b_prev_1y + 0.5)).astype(np.float32)
    res["b_trajectory_accel"] = ((res["b_1y"] - b_prev_1y) - (b_prev_1y - b_prev_2y)).astype(np.float32)
    res["b_to_c_escalation"] = (res["b_open"] / (res["c_open"] + 1.0)).astype(np.float32)
    res["b_open_ratio"] = (res["b_open"] / (res["b_all"] + 1e-4)).astype(np.float32)
    res["b_overdue_ratio"] = (res["b_overdue"] / (res["b_open"] + 1e-4)).astype(np.float32)

    # Ingest Point-in-time HPD tenant complaints
    if len(df_complaints) > 0 and "complaint_dt" in df_complaints.columns:
        mask_comp = df_complaints["complaint_dt"] < t_cutoff
        df_c_hist = df_complaints[mask_comp].copy()

        t_comp_30d = t_cutoff - pd.Timedelta(days=30)
        t_comp_90d = t_cutoff - pd.Timedelta(days=90)
        t_comp_1y = t_cutoff - pd.DateOffset(years=1)
        t_comp_3y = t_cutoff - pd.DateOffset(years=3)

        dates_c = df_c_hist["complaint_dt"]
        df_c_hist["comp_30d"] = (dates_c >= t_comp_30d).astype(np.int32)
        df_c_hist["comp_90d"] = (dates_c >= t_comp_90d).astype(np.int32)
        df_c_hist["comp_1y"] = (dates_c >= t_comp_1y).astype(np.int32)
        df_c_hist["comp_3y"] = (dates_c >= t_comp_3y).astype(np.int32)
        df_c_hist["comp_open"] = df_c_hist["is_open"].astype(np.int32)
        df_c_hist["last_comp_date"] = dates_c

        comp_agg = (
            df_c_hist.groupby("clean_bbl")
            .agg(
                {
                    "comp_30d": "sum",
                    "comp_90d": "sum",
                    "comp_1y": "sum",
                    "comp_3y": "sum",
                    "comp_open": "sum",
                    "last_comp_date": "max",
                }
            )
            .reset_index()
        )

        res = res.merge(comp_agg, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])

        res["comp_30d"] = res["comp_30d"].fillna(0.0).astype(np.float32)
        res["comp_90d"] = res["comp_90d"].fillna(0.0).astype(np.float32)
        res["comp_1y"] = res["comp_1y"].fillna(0.0).astype(np.float32)
        res["comp_3y"] = res["comp_3y"].fillna(0.0).astype(np.float32)
        res["comp_open"] = res["comp_open"].fillna(0.0).astype(np.float32)

        res["comp_velocity_90d"] = (
            res["comp_90d"] / (res["comp_1y"] / 4.0 + 0.1)
        ).astype(np.float32)
        res["comp_surge_30d"] = (
            res["comp_30d"] / (res["comp_1y"] / 12.0 + 0.1)
        ).astype(np.float32)
        res["comp_open_to_c_ratio"] = (
            res["comp_open"] / (res["c_1y"] + 1.0)
        ).astype(np.float32)
        res["comp_open_to_tot_ratio"] = (
            res["comp_open"] / (res["tot_1y"] + 1.0)
        ).astype(np.float32)
        res["comp_to_c_ratio_90d"] = (
            (res["comp_90d"] + 1.0) / (res["c_3m"] + 1.0)
        ).astype(np.float32)
        res["winter_comp_surge"] = (
            res["comp_90d"] / (res["comp_1y"] + 0.1)
        ).astype(np.float32)

        days_comp = (t_cutoff - res["last_comp_date"]).dt.days
        res["days_since_last_complaint"] = (
            days_comp.fillna(3650.0).clip(lower=0).astype(np.float32)
        )
        res["recency_decay_complaint"] = np.exp(
            -res["days_since_last_complaint"] / 365.0
        ).astype(np.float32)
        res = res.drop(columns=["last_comp_date"])
    else:
        res["comp_30d"] = np.zeros(len(res), dtype=np.float32)
        res["comp_90d"] = np.zeros(len(res), dtype=np.float32)
        res["comp_1y"] = np.zeros(len(res), dtype=np.float32)
        res["comp_3y"] = np.zeros(len(res), dtype=np.float32)
        res["comp_open"] = np.zeros(len(res), dtype=np.float32)
        res["comp_velocity_90d"] = np.zeros(len(res), dtype=np.float32)
        res["comp_surge_30d"] = np.zeros(len(res), dtype=np.float32)
        res["comp_open_to_c_ratio"] = np.zeros(len(res), dtype=np.float32)
        res["comp_open_to_tot_ratio"] = np.zeros(len(res), dtype=np.float32)
        res["comp_to_c_ratio_90d"] = (
            1.0 / (res["c_3m"] + 1.0)
        ).astype(np.float32)
        res["winter_comp_surge"] = np.zeros(len(res), dtype=np.float32)
        res["days_since_last_complaint"] = np.full(
            len(res), 3650.0, dtype=np.float32
        )
        res["recency_decay_complaint"] = np.zeros(len(res), dtype=np.float32)

    # Ingest Point-in-time HPD Housing Court litigations
    if len(df_litigations) > 0 and "lit_dt" in df_litigations.columns:
        mask_lit = df_litigations["lit_dt"] < t_cutoff
        df_l_hist = df_litigations[mask_lit].copy()

        t_lit_1y = t_cutoff - pd.DateOffset(years=1)
        df_l_hist["lit_1y"] = (df_l_hist["lit_dt"] >= t_lit_1y).astype(np.int32)
        is_still_open = (
            (df_l_hist["lit_close_dt"].isna() | (df_l_hist["lit_close_dt"] >= t_cutoff))
            & (
                df_l_hist["is_open_lit"]
                | df_l_hist["lit_close_dt"].isna()
                | (df_l_hist["lit_close_dt"] >= t_cutoff)
            )
        )
        df_l_hist["lit_open"] = is_still_open.astype(np.int32)

        lit_agg = (
            df_l_hist.groupby("clean_bbl")
            .agg(
                lit_open_count=("lit_open", "sum"),
                lit_1y_count=("lit_1y", "sum"),
            )
            .reset_index()
        )
        res = res.merge(lit_agg, left_on="bbl", right_on="clean_bbl", how="left")
        if "clean_bbl" in res.columns:
            res = res.drop(columns=["clean_bbl"])

        res["lit_open_count"] = res["lit_open_count"].fillna(0.0).astype(np.float32)
        res["lit_1y_count"] = res["lit_1y_count"].fillna(0.0).astype(np.float32)
        res["has_active_litigation"] = (res["lit_open_count"] > 0).astype(np.float32)
    else:
        res["lit_open_count"] = np.zeros(len(res), dtype=np.float32)
        res["lit_1y_count"] = np.zeros(len(res), dtype=np.float32)
        res["has_active_litigation"] = np.zeros(len(res), dtype=np.float32)

    if "unitsres" in res.columns:
        units = (
            pd.to_numeric(res["unitsres"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)
        )
    else:
        units = pd.Series(1.0, index=res.index, dtype=np.float32)

    # Landlord portfolio distress metrics strictly point-in-time
    if "ownername" in res.columns:
        owner_raw = res["ownername"].fillna("").astype(str).str.strip().str.upper()
        generic_owners = {
            "", "NAN", "NONE", "UNKNOWN", "N/A", "NULL", "NYC", "CITY OF NEW YORK",
            "NYC HOUSING AUTHORITY", "NYCHA", "NEW YORK CITY HOUSING AUTHORITY",
            "DEPARTMENT OF HOUSING", "HPD", "PARKS AND RECREATION", "DCAS",
            "BOARD OF EDUCATION", "MTA", "NEW YORK CITY", "CITY OF NY", "DEPT OF PARKS",
            "UNITED STATES OF AMERICA", "FEDERAL", "STATE OF NEW YORK", "HOUSING PRESERVATION",
        }
        valid_owner_mask = (~owner_raw.isin(generic_owners)) & (owner_raw.str.len() >= 3)
        owner_clean = owner_raw.where(valid_owner_mask, other="")

        valid_df = pd.DataFrame(
            {
                "owner": owner_clean[valid_owner_mask],
                "units": units[valid_owner_mask],
                "c_1y": res.loc[valid_owner_mask, "c_1y"],
            }
        )

        if len(valid_df) > 0:
            owner_stats = valid_df.groupby("owner").agg(
                owner_bldg_count=("units", "count"),
                owner_unitsres_sum=("units", "sum"),
                owner_c_1y_sum=("c_1y", "sum"),
            )
            res["owner_bldg_count"] = (
                owner_clean.map(owner_stats["owner_bldg_count"]).fillna(1.0).astype(np.float32)
            )
            res["owner_unitsres_sum"] = (
                owner_clean.map(owner_stats["owner_unitsres_sum"]).fillna(units).astype(np.float32)
            )
            res["owner_c_1y_sum"] = (
                owner_clean.map(owner_stats["owner_c_1y_sum"]).fillna(res["c_1y"]).astype(np.float32)
            )
        else:
            res["owner_bldg_count"] = np.ones(len(res), dtype=np.float32)
            res["owner_unitsres_sum"] = units.astype(np.float32)
            res["owner_c_1y_sum"] = res["c_1y"].astype(np.float32)

        res["owner_c_per_unit"] = (
            res["owner_c_1y_sum"] / (res["owner_unitsres_sum"] + 1.0)
        ).astype(np.float32)
        res = res.drop(columns=["ownername"])
    else:
        res["owner_bldg_count"] = np.ones(len(res), dtype=np.float32)
        res["owner_unitsres_sum"] = units.astype(np.float32)
        res["owner_c_1y_sum"] = res["c_1y"].astype(np.float32)
        res["owner_c_per_unit"] = (
            res["owner_c_1y_sum"] / (res["owner_unitsres_sum"] + 1.0)
        ).astype(np.float32)

    res["c_open_per_unit"] = (res["c_open"] / units).astype(np.float32)
    res["b_open_per_unit"] = (res["b_open"] / units).astype(np.float32)
    res["c_overdue_per_unit"] = (res["c_overdue"] / units).astype(np.float32)
    res["tot_overdue_per_unit"] = (res["tot_overdue"] / units).astype(np.float32)
    res["c_apt_spread_ratio"] = (res["c_unique_apts_1y"] / units).astype(np.float32)
    res["b_apt_spread_ratio"] = (res["b_unique_apts_1y"] / units).astype(np.float32)
    res["c_q4_heat_per_unit"] = (res["c_q4_heat"] / units).astype(np.float32)
    res["comp_per_unit_1y"] = (res["comp_1y"] / units).astype(np.float32)

    # Community District (cd) violation density per residential unit
    if "cd" in res.columns:
        cd_series = pd.to_numeric(res["cd"], errors="coerce").fillna(0).astype("int32")
        cd_c_sum = res["c_1y"].groupby(cd_series).transform("sum")
        cd_u_sum = units.groupby(cd_series).transform("sum")
        res["cd_c_per_unit"] = (cd_c_sum / (cd_u_sum + 10.0)).fillna(0.0).astype(np.float32)
    else:
        res["cd_c_per_unit"] = np.zeros(len(res), dtype=np.float32)

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
        assessed = pd.Series(0.0, index=res.index, dtype=np.float32)
        res["log_assesstot"] = np.zeros(len(res), dtype=np.float32)
        res["assess_per_unit"] = np.zeros(len(res), dtype=np.float32)

    res["assessed_value_per_sqft"] = (assessed / (bldgarea + 10.0)).astype(np.float32)
    res["prewar_walkup_flag"] = (
        (res["is_prewar"] > 0) & (numfloors <= 6)
    ).astype(np.float32)

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
        bldgclass = res["bldgclass"].fillna("UNK").astype(str).str.strip().str.upper()
        res["bldgclass_2char"] = bldgclass.str[:2]
        res["bldgclass_major"] = bldgclass.str[:1]
        res = res.drop(columns=["bldgclass"])
    else:
        res["bldgclass_2char"] = "UN"
        res["bldgclass_major"] = "U"

    if "zipcode" in res.columns:
        res["zipcode_str"] = (
            pd.to_numeric(res["zipcode"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        res = res.drop(columns=["zipcode"])
    else:
        res["zipcode_str"] = "00000"

    boro = res["bbl"].str[:1]
    res["borocode"] = (
        pd.to_numeric(boro, errors="coerce").fillna(0).astype(np.int32)
    )

    # Explicitly convert all engineered numeric features safely
    categorical_cols = ["bbl", "bldgclass_major", "bldgclass_2char", "zipcode_str"]
    for col in res.columns:
        if col not in categorical_cols:
            res[col] = (
                pd.to_numeric(res[col], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )

    return res.copy()


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


# Extract cohorts (Test 2023, Val 2022, Train 2021, 2020, 2019)
X_test_df = extract_point_in_time_features(df_test, df_viol, "2023-01-01")
X_val_df = extract_point_in_time_features(df_val_entities, df_viol, "2022-01-01")
y_val = compute_target(df_val_entities, df_viol, "2022-01-01")
X_val_df["target"] = y_val

X_train_2021_df = extract_point_in_time_features(
    df_train_2021_entities, df_viol, "2021-01-01"
)
y_train_2021 = compute_target(df_train_2021_entities, df_viol, "2021-01-01")
X_train_2021_df["target"] = y_train_2021

X_train_2020_df = extract_point_in_time_features(
    df_train_2020_entities, df_viol, "2020-01-01"
)
y_train_2020 = compute_target(df_train_2020_entities, df_viol, "2020-01-01")
X_train_2020_df["target"] = y_train_2020

X_train_2019_df = extract_point_in_time_features(
    df_train_2019_entities, df_viol, "2019-01-01"
)
y_train_2019 = compute_target(df_train_2019_entities, df_viol, "2019-01-01")
X_train_2019_df["target"] = y_train_2019

X_train_df = pd.concat(
    [X_train_2019_df, X_train_2020_df, X_train_2021_df], ignore_index=True
)
sample_weights_np = np.concatenate(
    [
        np.full(len(X_train_2019_df), 0.8, dtype=np.float32),
        np.full(len(X_train_2020_df), 0.5, dtype=np.float32),
        np.full(len(X_train_2021_df), 1.0, dtype=np.float32),
    ]
)
y_train = X_train_df["target"]

# Strict train-only target encoding
global_target_mean = float(y_train.mean())
prior_weight = 50.0

X_train_df = X_train_df.copy()
X_val_df = X_val_df.copy()
X_test_df = X_test_df.copy()

for cat_col in ["bldgclass_2char", "bldgclass_major", "zipcode_str", "borocode"]:
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
            pd.to_numeric(X_train_df[cat_col].map(te_map).fillna(global_target_mean), errors="coerce")
            .fillna(global_target_mean)
            .astype(np.float32)
        )
        X_val_df[col_name] = (
            pd.to_numeric(X_val_df[cat_col].map(te_map).fillna(global_target_mean), errors="coerce")
            .fillna(global_target_mean)
            .astype(np.float32)
        )
        X_test_df[col_name] = (
            pd.to_numeric(X_test_df[cat_col].map(te_map).fillna(global_target_mean), errors="coerce")
            .fillna(global_target_mean)
            .astype(np.float32)
        )

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
    """Residual block with LayerNorm, SiLU activations, SE channel attention gating, and Dropout."""

    def __init__(self, dim, dropout=0.2, se_reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

        red_dim = max(8, dim // se_reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, red_dim),
            nn.SiLU(),
            nn.Linear(red_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out * self.se(out)
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
                "c_30d": 0.5,
                "c_60d": 0.4,
                "c_3m": 0.5,
                "b_open_ratio": 0.2,
                "c_open_per_unit": 0.5,
                "c_active_years": 0.25,
                "block_c_1y": 0.3,
                "b_to_c_escalation": 0.3,
                "c_q4_heat": 0.4,
                "closed_c_ratio": -0.2,
                "chronic_unresolved_flag": 0.3,
                "comp_1y": 0.3,
                "comp_90d": 0.3,
                "comp_30d": 0.3,
                "comp_velocity_90d": 0.2,
                "c_unique_apts_1y": 0.4,
                "c_apt_spread_ratio": 0.5,
                "c_overdue": 0.4,
                "c_overdue_per_unit": 0.4,
                "c_trajectory_accel": 0.3,
                "comp_surge_30d": 0.25,
                "comp_open_to_c_ratio": 0.2,
                "lit_open_count": 0.35,
                "lit_1y_count": 0.25,
                "c_inspection_visits_1y": 0.35,
                "owner_c_per_unit": 0.4,
                "owner_bldg_count": 0.2,
                "c_consecutive_3y_flag": 0.3,
                "c_resolution_efficiency_1y": -0.2,
                "comp_to_c_ratio_90d": 0.25,
                "winter_comp_surge": 0.2,
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

    def forward(self, logits, targets, weights=None):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean() if weights is None else loss.sum() / weights.sum()
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

# Configure positive monotonic constraints for core cumulative violation predictors
monotone_features = {"c_1y", "c_2y", "c_open"}
monotone_lgb = [1 if c in monotone_features else 0 for c in common_features]
monotone_xgb = tuple(1 if c in monotone_features else 0 for c in common_features)

# 1. Train Gradient Boosted Decision Tree (LightGBM) with Sample Weighting & AP metric
lgb_model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=2.0,
    subsample=0.85,
    colsample_bytree=0.75,
    eval_metric="average_precision",
    monotone_constraints=monotone_lgb,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_np,
    y_train_np,
    sample_weight=sample_weights_np,
    eval_set=[(X_val_np, y_val_np)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
val_lgb_probs = lgb_model.predict_proba(X_val_np)[:, 1].astype(float)
test_lgb_probs = lgb_model.predict_proba(X_test_np)[:, 1].astype(float)
lgb_val_ap = float(average_precision_score(y_val_np, val_lgb_probs))
print(f"LightGBM Val AP: {lgb_val_ap:.6f}")

# 2. Train Symmetric Oblivious Tree Model (CatBoost) with Sample Weighting
cat_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=8.0,
    eval_metric="PRAUC",
    random_seed=42,
    verbose=False,
)
cat_model.fit(
    X_train_np,
    y_train_np,
    sample_weight=sample_weights_np,
    eval_set=(X_val_np, y_val_np),
    early_stopping_rounds=50,
    verbose=False,
)
val_cat_probs = cat_model.predict_proba(X_val_np)[:, 1].astype(float)
test_cat_probs = cat_model.predict_proba(X_test_np)[:, 1].astype(float)
cat_val_ap = float(average_precision_score(y_val_np, val_cat_probs))
print(f"CatBoost Val AP: {cat_val_ap:.6f}")

# 3. Train Loss-Guided Histogram Tree Model (XGBoost) with Sample Weighting
xgb_model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.04,
    max_depth=5,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=3.0,
    tree_method="hist",
    eval_metric="aucpr",
    monotone_constraints=monotone_xgb,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(
    X_train_np,
    y_train_np,
    sample_weight=sample_weights_np,
    eval_set=[(X_val_np, y_val_np)],
    verbose=False,
)
val_xgb_probs = xgb_model.predict_proba(X_val_np)[:, 1].astype(float)
test_xgb_probs = xgb_model.predict_proba(X_test_np)[:, 1].astype(float)
xgb_val_ap = float(average_precision_score(y_val_np, val_xgb_probs))
print(f"XGBoost Val AP: {xgb_val_ap:.6f}")

# 4. Train TabularResNet with Non-Linear Log1p Scaling, LR Warmup, Sample Weights, & Checkpoint Restoration
X_train_nn = X_train_np.copy()
X_val_nn = X_val_np.copy()
X_test_nn = X_test_np.copy()

for idx, col in enumerate(common_features):
    is_count_col = (
        col.startswith(("c_", "b_", "a_", "i_", "tot_", "comp_", "lit_", "block_"))
        or "count" in col
        or "visits" in col
        or "units" in col
    )
    is_valuation_area_col = (
        "area" in col
        or "assess" in col
        or col in ["bldgarea", "lotarea", "assesstot", "assessland"]
    )
    is_excluded = (
        "trend" in col
        or "trajectory" in col
        or "decay" in col
        or "ratio" in col
        or "log_" in col
        or "flag" in col
    )
    if (is_count_col or is_valuation_area_col) and not is_excluded:
        if np.all(X_train_nn[:, idx] >= 0.0):
            X_train_nn[:, idx] = np.log1p(np.maximum(0.0, X_train_nn[:, idx]))
            X_val_nn[:, idx] = np.log1p(np.maximum(0.0, X_val_nn[:, idx]))
            X_test_nn[:, idx] = np.log1p(np.maximum(0.0, X_test_nn[:, idx]))

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_nn).astype(np.float32)
X_val_norm = scaler.transform(X_val_nn).astype(np.float32)
X_test_norm = scaler.transform(X_test_nn).astype(np.float32)

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

scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 1, eta_min=1e-5)
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_cosine],
    milestones=[1],
)

batch_size = 2048
eval_batch_size = 16384
train_dataset = TensorDataset(
    torch.from_numpy(X_train_norm),
    torch.from_numpy(y_train_np),
    torch.from_numpy(sample_weights_np),
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

    for batch_x, batch_y, batch_w in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y, weights=batch_w)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
print(f"TabularResNet Best Val AP: {val_nn_ap:.6f}")

# ---------------------------------------------------------
# Dynamic Simplex 4-Way Ensemble & Submission Generation
# ---------------------------------------------------------
val_lgb_rank = (rankdata(val_lgb_probs) / len(val_lgb_probs)).astype(np.float32)
test_lgb_rank = (rankdata(test_lgb_probs) / len(test_lgb_probs)).astype(np.float32)

val_cat_rank = (rankdata(val_cat_probs) / len(val_cat_probs)).astype(np.float32)
test_cat_rank = (rankdata(test_cat_probs) / len(test_cat_probs)).astype(np.float32)

val_xgb_rank = (rankdata(val_xgb_probs) / len(val_xgb_probs)).astype(np.float32)
test_xgb_rank = (rankdata(test_xgb_probs) / len(test_xgb_probs)).astype(np.float32)

val_nn_rank = (rankdata(val_nn_probs) / len(val_nn_probs)).astype(np.float32)
test_nn_rank = (rankdata(test_nn_probs) / len(test_nn_probs)).astype(np.float32)

best_ens_ap = -1.0
best_weights = [0.25, 0.25, 0.25, 0.25]
step = 0.05
grid = np.arange(0.0, 1.0 + 1e-5, step)

for w_lgb in grid:
    for w_cat in np.arange(0.0, 1.0 - w_lgb + 1e-5, step):
        for w_xgb in np.arange(0.0, 1.0 - w_lgb - w_cat + 1e-5, step):
            w_nn = max(0.0, 1.0 - w_lgb - w_cat - w_xgb)
            ens_val = (
                w_lgb * val_lgb_rank
                + w_cat * val_cat_rank
                + w_xgb * val_xgb_rank
                + w_nn * val_nn_rank
            )
            score = float(average_precision_score(y_val_np, ens_val))
            if score > best_ens_ap:
                best_ens_ap = score
                best_weights = [float(w_lgb), float(w_cat), float(w_xgb), float(w_nn)]

# Continuous Simplex Nelder-Mead Optimization on Validation Average Precision
def softmax(theta):
    e_theta = np.exp(theta - np.max(theta))
    return e_theta / np.sum(e_theta)

def ensemble_objective(theta):
    w = softmax(theta)
    blend_val = (
        w[0] * val_lgb_rank
        + w[1] * val_cat_rank
        + w[2] * val_xgb_rank
        + w[3] * val_nn_rank
    )
    return -float(average_precision_score(y_val_np, blend_val))

init_theta = np.log(np.array(best_weights) + 1e-4)
opt_res = minimize(
    ensemble_objective,
    x0=init_theta,
    method="Nelder-Mead",
    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
)

opt_weights = softmax(opt_res.x)
opt_val_ap = -float(opt_res.fun)

if opt_val_ap > best_ens_ap:
    best_ens_ap = opt_val_ap
    best_w_lgb, best_w_cat, best_w_xgb, best_w_nn = (
        float(opt_weights[0]),
        float(opt_weights[1]),
        float(opt_weights[2]),
        float(opt_weights[3]),
    )
else:
    best_w_lgb, best_w_cat, best_w_xgb, best_w_nn = best_weights

print(
    f"Optimal Ensemble Weights: LGB={best_w_lgb:.4f}, CAT={best_w_cat:.4f}, XGB={best_w_xgb:.4f}, NN={best_w_nn:.4f} -> Val AP: {best_ens_ap:.6f}"
)

final_val_score = best_ens_ap
final_test_scores = (
    best_w_lgb * test_lgb_rank
    + best_w_cat * test_cat_rank
    + best_w_xgb * test_xgb_rank
    + best_w_nn * test_nn_rank
)

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
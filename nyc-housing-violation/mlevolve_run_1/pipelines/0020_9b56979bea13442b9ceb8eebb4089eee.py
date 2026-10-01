import copy
import gc
import os
import warnings

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. Configuration & Storage Setup
# -------------------------------------------------------------------------
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
STORAGE_OPTIONS = {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else {}

GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
LAKE_FULL = f"{GCS_BASE}/lake/full"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

CUTOFF_TRAIN_2020 = pd.Timestamp("2020-01-01")
CUTOFF_TRAIN_2021 = pd.Timestamp("2021-01-01")
CUTOFF_TRAIN = CUTOFF_TRAIN_2021
CUTOFF_VAL = pd.Timestamp("2022-01-01")
CUTOFF_TEST = pd.Timestamp("2023-01-01")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# 2. Ingestion: Test Entities, PLUTO, HPD Violations, Complaints, Litigations
# -------------------------------------------------------------------------
df_test_entities = pd.read_parquet(
    f"{GCS_BASE}/test_entities.parquet",
    storage_options=STORAGE_OPTIONS,
)
df_test_entities["bbl"] = (
    df_test_entities["bbl"].astype(str).str.split(".").str[0].str.zfill(10)
)
test_bbls = df_test_entities["bbl"].values

pluto_cols = [
    "bbl",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "bldgclass",
    "numfloors",
    "bldgarea",
    "assesstot",
    "borough",
    "zipcode",
    "cd",
]
try:
    df_pluto = pd.read_parquet(
        f"{LAKE_FULL}/pluto",
        columns=pluto_cols,
        storage_options=STORAGE_OPTIONS,
    )
except Exception:
    df_pluto = pd.read_parquet(
        f"{LAKE_FULL}/pluto",
        columns=[c for c in pluto_cols if c != "cd"],
        storage_options=STORAGE_OPTIONS,
    )
    df_pluto["cd"] = df_pluto["borough"].astype(str)

df_pluto["bbl"] = df_pluto["bbl"].astype(str).str.split(".").str[0].str.zfill(10)
df_pluto = df_pluto.drop_duplicates(subset=["bbl"], keep="last")

df_pluto["unitsres"] = (
    pd.to_numeric(df_pluto["unitsres"], errors="coerce").fillna(0).astype(np.float32)
)
df_pluto["unitstotal"] = (
    pd.to_numeric(df_pluto["unitstotal"], errors="coerce").fillna(0).astype(np.float32)
)
df_pluto["yearbuilt"] = (
    pd.to_numeric(df_pluto["yearbuilt"], errors="coerce").fillna(0).astype(np.float32)
)
df_pluto["numfloors"] = (
    pd.to_numeric(df_pluto["numfloors"], errors="coerce").fillna(0).astype(np.float32)
)
df_pluto["bldgarea"] = (
    pd.to_numeric(df_pluto["bldgarea"], errors="coerce").fillna(0).astype(np.float32)
)
df_pluto["assesstot"] = (
    pd.to_numeric(df_pluto["assesstot"], errors="coerce").fillna(0).astype(np.float32)
)

df_pluto["is_prewar"] = (
    (df_pluto["yearbuilt"] > 1800) & (df_pluto["yearbuilt"] < 1940)
).astype(np.int32)
df_pluto["area_per_unit"] = (
    df_pluto["bldgarea"] / (df_pluto["unitsres"] + 1.0)
).astype(np.float32)
df_pluto["assess_per_unit"] = (
    df_pluto["assesstot"] / (df_pluto["unitsres"] + 1.0)
).astype(np.float32)
df_pluto["res_unit_ratio"] = (
    df_pluto["unitsres"] / (df_pluto["unitstotal"] + 1.0)
).astype(np.float32)
df_pluto["block_prefix"] = df_pluto["bbl"].astype(str).str[:6]

df_pluto["borough"] = (
    df_pluto["borough"].astype(str).fillna("UNKNOWN").astype("category")
)
df_pluto["bldgclass"] = (
    df_pluto["bldgclass"].astype(str).fillna("UNKNOWN").astype("category")
)
df_pluto["zipcode"] = (
    df_pluto["zipcode"].astype(str).fillna("UNKNOWN").astype("category")
)
if "cd" in df_pluto.columns:
    df_pluto["cd"] = (
        df_pluto["cd"].astype(str).fillna("UNKNOWN").astype("category")
    )

try:
    df_aep = pd.read_parquet(
        f"{LAKE_FULL}/hpd_aep_buildings",
        columns=["bbl"],
        storage_options=STORAGE_OPTIONS,
    )
    aep_bbls = set(
        df_aep["bbl"].dropna().astype(str).str.split(".").str[0].str.zfill(10)
    )
except Exception:
    aep_bbls = set()

try:
    df_conh = pd.read_parquet(
        f"{LAKE_FULL}/hpd_conh_buildings",
        storage_options=STORAGE_OPTIONS,
    )
    bbl_col_conh = next((c for c in df_conh.columns if c.lower() == "bbl"), None)
    if bbl_col_conh is not None:
        conh_bbls = set(
            df_conh[bbl_col_conh].dropna().astype(str).str.split(".").str[0].str.zfill(10)
        )
    else:
        b_col = next((c for c in df_conh.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]), None)
        blk_col = next((c for c in df_conh.columns if c.lower() == "block"), None)
        lot_col = next((c for c in df_conh.columns if c.lower() == "lot"), None)
        if b_col and blk_col and lot_col:
            boro_map = {
                "MANHATTAN": "1",
                "BRONX": "2",
                "BROOKLYN": "3",
                "QUEENS": "4",
                "STATEN ISLAND": "5",
                "MN": "1",
                "BX": "2",
                "BK": "3",
                "QN": "4",
                "SI": "5",
            }
            s_boro = df_conh[b_col].astype(str).str.upper().str.strip().replace(boro_map)
            b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
            blk_str = pd.to_numeric(df_conh[blk_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
            lot_str = pd.to_numeric(df_conh[lot_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
            conh_bbls = set((b_str + blk_str + lot_str).values)
        else:
            conh_bbls = set()
except Exception:
    conh_bbls = set()

def load_distress_watchlist_bbls(table_name):
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/{table_name}",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = next((c for c in df.columns if c.lower() == "bbl"), None)
        if bbl_col is not None:
            return set(
                df[bbl_col]
                .dropna()
                .astype(str)
                .str.split(".")
                .str[0]
                .str.strip()
                .str.zfill(10)
            )
        b_col = next(
            (c for c in df.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]),
            None,
        )
        blk_col = next((c for c in df.columns if c.lower() == "block"), None)
        lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
        if b_col and blk_col and lot_col:
            boro_map = {
                "MANHATTAN": "1",
                "BRONX": "2",
                "BROOKLYN": "3",
                "QUEENS": "4",
                "STATEN ISLAND": "5",
                "MN": "1",
                "BX": "2",
                "BK": "3",
                "QN": "4",
                "SI": "5",
            }
            s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
            b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
            blk_str = pd.to_numeric(df[blk_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
            lot_str = pd.to_numeric(df[lot_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
            return set((b_str + blk_str + lot_str).values)
        return set()
    except Exception:
        return set()

underlying_bbls = load_distress_watchlist_bbls("hpd_underlying_conditions")
speculation_bbls = load_distress_watchlist_bbls("speculation_watch_list")

try:
    df_lit = pd.read_parquet(
        f"{LAKE_FULL}/hpd_litigations",
        columns=["bbl", "caseopendate"],
        storage_options=STORAGE_OPTIONS,
    )
    df_lit["bbl"] = df_lit["bbl"].astype(str).str.split(".").str[0].str.zfill(10)
    df_lit["date"] = pd.to_datetime(df_lit["caseopendate"], errors="coerce")
    df_lit = df_lit.dropna(subset=["bbl", "date"])
except Exception:
    df_lit = pd.DataFrame(columns=["bbl", "date"])

try:
    df_comp = pd.read_parquet(
        f"{LAKE_FULL}/hpd_complaints",
        storage_options=STORAGE_OPTIONS,
    )
    bbl_col = next((c for c in df_comp.columns if c.lower() == "bbl"), None)
    if bbl_col is None:
        b_col = next((c for c in df_comp.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]), None)
        blk_col = next((c for c in df_comp.columns if c.lower() == "block"), None)
        lot_col = next((c for c in df_comp.columns if c.lower() == "lot"), None)
        if b_col and blk_col and lot_col:
            boro_map = {
                "MANHATTAN": "1",
                "BRONX": "2",
                "BROOKLYN": "3",
                "QUEENS": "4",
                "STATEN ISLAND": "5",
                "MN": "1",
                "BX": "2",
                "BK": "3",
                "QN": "4",
                "SI": "5",
            }
            s_boro = df_comp[b_col].astype(str).str.upper().str.strip().replace(boro_map)
            b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
            blk_str = pd.to_numeric(df_comp[blk_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
            lot_str = pd.to_numeric(df_comp[lot_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
            df_comp["bbl"] = b_str + blk_str + lot_str
            bbl_col = "bbl"

    df_comp["bbl"] = df_comp[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
    date_col = next((c for c in df_comp.columns if "received" in c.lower() or c.lower() == "date"), None)
    if date_col is None:
        date_col = next((c for c in df_comp.columns if "date" in c.lower()), None)
    df_comp["date"] = pd.to_datetime(df_comp[date_col], errors="coerce")
    if hasattr(df_comp["date"].dt, "tz") and df_comp["date"].dt.tz is not None:
        df_comp["date"] = df_comp["date"].dt.tz_localize(None)
    df_comp = df_comp.dropna(subset=["bbl", "date"])
    df_comp = df_comp[df_comp["date"] >= pd.Timestamp("2014-01-01")]

    heat_mask = pd.Series(False, index=df_comp.index)
    potential_cat_cols = [c for c in df_comp.columns if c.lower() not in ["bbl", "date", str(date_col).lower()]]
    for c in potential_cat_cols:
        if df_comp[c].dtype == object or str(df_comp[c].dtype) == "category" or df_comp[c].dtype == "string":
            col_str = df_comp[c].astype(str).str.upper()
            m = col_str.str.contains("HEAT|HOT WATER|HOT_WATER|HEATING", regex=True, na=False)
            if m.any():
                heat_mask = heat_mask | m

    df_comp["is_heat"] = heat_mask.astype(bool)
    df_comp = df_comp[["bbl", "date", "is_heat"]].copy()
except Exception:
    df_comp = pd.DataFrame(columns=["bbl", "date", "is_heat"])


def load_emergency_table(table_name, priority_dates=None):
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/{table_name}",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = None
        for c in df.columns:
            if c.lower() == "bbl":
                bbl_col = c
                break
        if bbl_col is None:
            b_col = next((c for c in df.columns if c.lower() in ["boroid", "borough", "boro"]), None)
            blk_col = next((c for c in df.columns if c.lower() == "block"), None)
            lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
            if b_col and blk_col and lot_col:
                boro_map = {
                    "MANHATTAN": "1",
                    "BRONX": "2",
                    "BROOKLYN": "3",
                    "QUEENS": "4",
                    "STATEN ISLAND": "5",
                    "MN": "1",
                    "BX": "2",
                    "BK": "3",
                    "QN": "4",
                    "SI": "5",
                }
                s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
                b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
                blk_str = pd.to_numeric(df[blk_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
                lot_str = pd.to_numeric(df[lot_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
                df["bbl"] = b_str + blk_str + lot_str
                bbl_col = "bbl"
        if not bbl_col:
            return pd.DataFrame(columns=["bbl", "date"])

        df["bbl"] = df[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
        df = df[df["bbl"].str.len() == 10]

        chosen_col = None
        if priority_dates:
            for p in priority_dates:
                matches = [c for c in df.columns if c.lower().replace("_", "") == p.lower().replace("_", "")]
                if matches:
                    chosen_col = matches[0]
                    break

        if chosen_col is None:
            cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            clean_cands = [
                c for c in cands
                if not any(k in c.lower() for k in ["mod", "extract", "created", "end", "rescind", "close"])
            ]
            chosen_col = clean_cands[0] if clean_cands else (cands[0] if cands else None)

        if chosen_col is None:
            return pd.DataFrame(columns=["bbl", "date"])

        df["date"] = pd.to_datetime(df[chosen_col], errors="coerce")
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.dropna(subset=["bbl", "date"])
        return df[["bbl", "date"]].copy()
    except Exception as e:
        print(f"Warning: could not load {table_name}: {e}")
        return pd.DataFrame(columns=["bbl", "date"])


def load_dob_violations():
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/dob_violations",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = None
        for c in df.columns:
            if c.lower() == "bbl":
                bbl_col = c
                break
        if bbl_col is None:
            b_col = next(
                (c for c in df.columns if c.lower() in ["boro", "borough", "boroid", "boro_code"]),
                None,
            )
            blk_col = next((c for c in df.columns if c.lower() == "block"), None)
            lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
            if b_col and blk_col and lot_col:
                boro_map = {
                    "MANHATTAN": "1",
                    "BRONX": "2",
                    "BROOKLYN": "3",
                    "QUEENS": "4",
                    "STATEN ISLAND": "5",
                    "MN": "1",
                    "BX": "2",
                    "BK": "3",
                    "QN": "4",
                    "SI": "5",
                }
                s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
                b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
                blk_str = (
                    pd.to_numeric(df[blk_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(5)
                )
                lot_str = (
                    pd.to_numeric(df[lot_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(4)
                )
                df["bbl"] = b_str + blk_str + lot_str
                bbl_col = "bbl"
        if not bbl_col:
            return pd.DataFrame(columns=["bbl", "date"])

        df["bbl"] = df[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
        df = df[df["bbl"].str.len() == 10]

        priority_dates = ["issue_date", "issuedate", "inspection_date", "date"]
        chosen_col = None
        for p in priority_dates:
            matches = [
                c
                for c in df.columns
                if c.lower().replace("_", "") == p.lower().replace("_", "")
            ]
            if matches:
                chosen_col = matches[0]
                break

        if chosen_col is None:
            cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            clean_cands = [
                c
                for c in cands
                if not any(
                    k in c.lower()
                    for k in ["mod", "extract", "created", "end", "rescind", "close"]
                )
            ]
            chosen_col = clean_cands[0] if clean_cands else (cands[0] if cands else None)

        if chosen_col is None:
            return pd.DataFrame(columns=["bbl", "date"])

        df["date"] = pd.to_datetime(df[chosen_col], errors="coerce")
        df = df.dropna(subset=["bbl", "date"])
        df = df[["bbl", "date"]].copy()
        df = df[df["date"] >= pd.Timestamp("2010-01-01")]
        return df
    except Exception as e:
        print(f"Warning: could not load dob_violations: {e}")
        return pd.DataFrame(columns=["bbl", "date"])


df_dob = load_dob_violations()
df_dob_safety = load_emergency_table(
    "dob_safety_violations",
    priority_dates=[
        "issue_date",
        "issuedate",
        "violation_date",
        "inspection_date",
        "date",
    ],
)
df_vacate = load_emergency_table(
    "hpd_vacate_orders",
    priority_dates=[
        "vacate_effective_date",
        "vacateeffectivedate",
        "effective_date",
        "order_date",
        "issued_date",
    ],
)
df_hwo = load_emergency_table(
    "hpd_hwo_charges",
    priority_dates=[
        "fee_date",
        "feedate",
        "charge_date",
        "chargedate",
        "order_date",
        "invoice_date",
    ],
)
df_omo = load_emergency_table(
    "hpd_omo_charges",
    priority_dates=[
        "fee_date",
        "feedate",
        "charge_date",
        "chargedate",
        "order_date",
        "invoice_date",
    ],
)
df_bedbug = load_emergency_table(
    "hpd_bedbug_reports",
    priority_dates=[
        "filing_date",
        "filingdate",
        "received_date",
        "inspection_date",
        "date",
    ],
)
df_evict = load_emergency_table(
    "evictions",
    priority_dates=[
        "executed_date",
        "executeddate",
        "eviction_date",
        "date",
    ],
)
df_ecb = load_emergency_table(
    "dob_ecb_violations",
    priority_dates=[
        "issue_date",
        "issuedate",
        "violation_date",
        "hearing_date",
        "date",
    ],
)


def load_rodent_inspections():
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/dohmh_rodent_inspections",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = next((c for c in df.columns if c.lower() == "bbl"), None)
        if bbl_col is None:
            b_col = next(
                (c for c in df.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]),
                None,
            )
            blk_col = next((c for c in df.columns if c.lower() == "block"), None)
            lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
            if b_col and blk_col and lot_col:
                boro_map = {
                    "MANHATTAN": "1",
                    "BRONX": "2",
                    "BROOKLYN": "3",
                    "QUEENS": "4",
                    "STATEN ISLAND": "5",
                    "MN": "1",
                    "BX": "2",
                    "BK": "3",
                    "QN": "4",
                    "SI": "5",
                }
                s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
                b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
                blk_str = pd.to_numeric(df[blk_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
                lot_str = pd.to_numeric(df[lot_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
                df["bbl"] = b_str + blk_str + lot_str
                bbl_col = "bbl"
        if not bbl_col:
            return pd.DataFrame(columns=["bbl", "date"])

        df["bbl"] = df[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
        df = df[df["bbl"].str.len() == 10]

        priority_dates = ["inspection_date", "inspectiondate", "date"]
        chosen_col = None
        for p in priority_dates:
            matches = [
                c
                for c in df.columns
                if c.lower().replace("_", "") == p.lower().replace("_", "")
            ]
            if matches:
                chosen_col = matches[0]
                break

        if chosen_col is None:
            cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            clean_cands = [
                c
                for c in cands
                if not any(
                    k in c.lower()
                    for k in ["mod", "extract", "created", "end", "rescind", "close"]
                )
            ]
            chosen_col = clean_cands[0] if clean_cands else (cands[0] if cands else None)

        if chosen_col is None:
            return pd.DataFrame(columns=["bbl", "date"])

        df["date"] = pd.to_datetime(df[chosen_col], errors="coerce")
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.dropna(subset=["bbl", "date"])

        result_col = next((c for c in df.columns if "result" in c.lower() or "status" in c.lower()), None)
        if result_col is not None:
            res_str = df[result_col].astype(str).str.upper()
            fail_mask = res_str.str.contains("ACTIVE|RAT|FAIL|PROBLEM", regex=True, na=False)
            df = df[fail_mask]

        df = df[["bbl", "date"]].copy()
        df = df[df["date"] >= pd.Timestamp("2010-01-01")]
        return df
    except Exception as e:
        print(f"Warning: could not load dohmh_rodent_inspections: {e}")
        return pd.DataFrame(columns=["bbl", "date"])


df_rodent = load_rodent_inspections()


def load_hpd_registrations():
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/hpd_registrations",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = next((c for c in df.columns if c.lower() == "bbl"), None)
        if bbl_col is None:
            b_col = next(
                (c for c in df.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]),
                None,
            )
            blk_col = next((c for c in df.columns if c.lower() == "block"), None)
            lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
            if b_col and blk_col and lot_col:
                boro_map = {
                    "MANHATTAN": "1",
                    "BRONX": "2",
                    "BROOKLYN": "3",
                    "QUEENS": "4",
                    "STATEN ISLAND": "5",
                    "MN": "1",
                    "BX": "2",
                    "BK": "3",
                    "QN": "4",
                    "SI": "5",
                }
                s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
                b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
                blk_str = (
                    pd.to_numeric(df[blk_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(5)
                )
                lot_str = (
                    pd.to_numeric(df[lot_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(4)
                )
                df["bbl"] = b_str + blk_str + lot_str
                bbl_col = "bbl"
        if not bbl_col:
            return pd.DataFrame(columns=["bbl", "registrationid"])

        df["bbl"] = df[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
        df = df[df["bbl"].str.len() == 10]
        reg_col = next(
            (c for c in df.columns if "registration" in c.lower() or c.lower() == "regid"),
            None,
        )
        if reg_col is None:
            return pd.DataFrame(columns=["bbl", "registrationid"])
        df["registrationid"] = df[reg_col].astype(str).str.strip()
        df = df[df["registrationid"] != ""].drop_duplicates(subset=["bbl"], keep="last")
        return df[["bbl", "registrationid"]].copy()
    except Exception as e:
        print(f"Warning: could not load hpd_registrations: {e}")
        return pd.DataFrame(columns=["bbl", "registrationid"])


def load_tax_lien_sales():
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/dof_tax_lien_sales",
            storage_options=STORAGE_OPTIONS,
        )
        bbl_col = next((c for c in df.columns if c.lower() == "bbl"), None)
        if bbl_col is None:
            b_col = next(
                (c for c in df.columns if c.lower() in ["boroid", "borough", "boro", "boro_code"]),
                None,
            )
            blk_col = next((c for c in df.columns if c.lower() == "block"), None)
            lot_col = next((c for c in df.columns if c.lower() == "lot"), None)
            if b_col and blk_col and lot_col:
                boro_map = {
                    "MANHATTAN": "1",
                    "BRONX": "2",
                    "BROOKLYN": "3",
                    "QUEENS": "4",
                    "STATEN ISLAND": "5",
                    "MN": "1",
                    "BX": "2",
                    "BK": "3",
                    "QN": "4",
                    "SI": "5",
                }
                s_boro = df[b_col].astype(str).str.upper().str.strip().replace(boro_map)
                b_str = pd.to_numeric(s_boro, errors="coerce").fillna(0).astype(int).astype(str)
                blk_str = (
                    pd.to_numeric(df[blk_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(5)
                )
                lot_str = (
                    pd.to_numeric(df[lot_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                    .str.zfill(4)
                )
                df["bbl"] = b_str + blk_str + lot_str
                bbl_col = "bbl"
        if not bbl_col:
            return pd.DataFrame(columns=["bbl", "date"])

        df["bbl"] = df[bbl_col].astype(str).str.split(".").str[0].str.strip().str.zfill(10)
        df = df[df["bbl"].str.len() == 10]

        date_col = next((c for c in df.columns if "date" in c.lower() or "year" in c.lower()), None)
        if date_col is not None:
            if "year" in date_col.lower() and "date" not in date_col.lower():
                df["date"] = pd.to_datetime(df[date_col].astype(str) + "-01-01", errors="coerce")
            else:
                df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["date"] = pd.Timestamp("2000-01-01")

        df["date"] = df["date"].fillna(pd.Timestamp("2000-01-01"))
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        return df[["bbl", "date"]].copy()
    except Exception as e:
        print(f"Warning: could not load dof_tax_lien_sales: {e}")
        return pd.DataFrame(columns=["bbl", "date"])


df_reg = load_hpd_registrations()
bbl_to_reg_map = dict(zip(df_reg["bbl"], df_reg["registrationid"])) if not df_reg.empty else {}
df_tax_lien = load_tax_lien_sales()

viol_cols = [
    "bbl",
    "boroid",
    "block",
    "lot",
    "class",
    "inspectiondate",
    "currentstatus",
]
df_viol = pd.read_parquet(
    f"{LAKE_FULL}/hpd_violations",
    columns=viol_cols,
    storage_options=STORAGE_OPTIONS,
)

bbl_raw = df_viol["bbl"].astype(str).str.split(".").str[0].str.strip()
is_valid_bbl = (bbl_raw.str.len() == 10) & bbl_raw.str.isnumeric()
boroid = df_viol["boroid"].fillna(0).astype(int).astype(str)
block = df_viol["block"].fillna(0).astype(int).astype(str).str.zfill(5)
lot = df_viol["lot"].fillna(0).astype(int).astype(str).str.zfill(4)
fallback_bbl = boroid + block + lot
df_viol["bbl"] = np.where(is_valid_bbl, bbl_raw, fallback_bbl)

df_viol["inspectiondate"] = pd.to_datetime(df_viol["inspectiondate"], errors="coerce")
df_viol = df_viol.dropna(subset=["bbl", "inspectiondate"])
df_viol["class"] = df_viol["class"].astype(str).str.upper().str.strip()

status_str = df_viol["currentstatus"].astype(str).str.upper()
df_viol["is_closed"] = status_str.str.contains("CLOSE", na=False)
df_viol["is_open"] = ~df_viol["is_closed"]
df_viol = df_viol[df_viol["inspectiondate"] >= pd.Timestamp("2014-01-01")]

gc.collect()


# -------------------------------------------------------------------------
# 3. Point-in-Time Feature Engineering & Target Computation
# -------------------------------------------------------------------------
def build_point_in_time_dataset(entities, cutoff_date, label_window_months=12):
    df_base = pd.DataFrame({"bbl": entities}).drop_duplicates().reset_index(drop=True)

    v_hist = df_viol[df_viol["inspectiondate"] < cutoff_date]
    v_c = v_hist[v_hist["class"] == "C"]
    v_b = v_hist[v_hist["class"] == "B"]
    v_a = v_hist[v_hist["class"] == "A"]

    t_30d = cutoff_date - pd.Timedelta(days=30)
    t_60d = cutoff_date - pd.Timedelta(days=60)
    t_90d = cutoff_date - pd.Timedelta(days=90)
    t_180d = cutoff_date - pd.Timedelta(days=180)
    t_1y = cutoff_date - pd.Timedelta(days=365)
    t_2y = cutoff_date - pd.Timedelta(days=730)
    t_3y = cutoff_date - pd.Timedelta(days=1095)
    t_5y = cutoff_date - pd.Timedelta(days=1825)

    c_30d = v_c[v_c["inspectiondate"] >= t_30d].groupby("bbl").size().rename("viol_c_30d")
    c_60d = v_c[v_c["inspectiondate"] >= t_60d].groupby("bbl").size().rename("viol_c_60d")
    c_90d = v_c[v_c["inspectiondate"] >= t_90d].groupby("bbl").size().rename("viol_c_90d")
    c_180d = v_c[v_c["inspectiondate"] >= t_180d].groupby("bbl").size().rename("viol_c_180d")
    c_1y = v_c[v_c["inspectiondate"] >= t_1y].groupby("bbl").size().rename("viol_c_1y")
    c_2y = v_c[v_c["inspectiondate"] >= t_2y].groupby("bbl").size().rename("viol_c_2y")
    c_3y = v_c[v_c["inspectiondate"] >= t_3y].groupby("bbl").size().rename("viol_c_3y")
    c_5y = v_c[v_c["inspectiondate"] >= t_5y].groupby("bbl").size().rename("viol_c_5y")
    c_all = v_c.groupby("bbl").size().rename("viol_c_all")
    c_max_date = v_c.groupby("bbl")["inspectiondate"].max().rename("viol_c_max_date")

    b_30d = v_b[v_b["inspectiondate"] >= t_30d].groupby("bbl").size().rename("viol_b_30d")
    b_90d = v_b[v_b["inspectiondate"] >= t_90d].groupby("bbl").size().rename("viol_b_90d")
    b_180d = v_b[v_b["inspectiondate"] >= t_180d].groupby("bbl").size().rename("viol_b_180d")
    b_1y = v_b[v_b["inspectiondate"] >= t_1y].groupby("bbl").size().rename("viol_b_1y")
    b_2y = v_b[v_b["inspectiondate"] >= t_2y].groupby("bbl").size().rename("viol_b_2y")
    b_all = v_b.groupby("bbl").size().rename("viol_b_all")
    b_max_date = v_b.groupby("bbl")["inspectiondate"].max().rename("viol_b_max_date")

    a_1y = v_a[v_a["inspectiondate"] >= t_1y].groupby("bbl").size().rename("viol_a_1y")
    a_all = v_a.groupby("bbl").size().rename("viol_a_all")

    all_30d = (
        v_hist[v_hist["inspectiondate"] >= t_30d]
        .groupby("bbl")
        .size()
        .rename("viol_all_30d")
    )
    all_90d = (
        v_hist[v_hist["inspectiondate"] >= t_90d]
        .groupby("bbl")
        .size()
        .rename("viol_all_90d")
    )
    all_180d = (
        v_hist[v_hist["inspectiondate"] >= t_180d]
        .groupby("bbl")
        .size()
        .rename("viol_all_180d")
    )
    all_1y = (
        v_hist[v_hist["inspectiondate"] >= t_1y]
        .groupby("bbl")
        .size()
        .rename("viol_all_1y")
    )
    all_2y = (
        v_hist[v_hist["inspectiondate"] >= t_2y]
        .groupby("bbl")
        .size()
        .rename("viol_all_2y")
    )
    all_all = v_hist.groupby("bbl").size().rename("viol_all_all")
    all_max_date = (
        v_hist.groupby("bbl")["inspectiondate"].max().rename("viol_all_max_date")
    )

    open_all = v_hist[v_hist["is_open"]].groupby("bbl").size().rename("viol_open_count")
    open_c = (
        v_hist[(v_hist["class"] == "C") & v_hist["is_open"]]
        .groupby("bbl")
        .size()
        .rename("viol_c_open_count")
    )

    viol_agg = pd.concat(
        [
            c_30d,
            c_60d,
            c_90d,
            c_180d,
            c_1y,
            c_2y,
            c_3y,
            c_5y,
            c_all,
            c_max_date,
            b_30d,
            b_90d,
            b_180d,
            b_1y,
            b_2y,
            b_all,
            b_max_date,
            a_1y,
            a_all,
            all_30d,
            all_90d,
            all_180d,
            all_1y,
            all_2y,
            all_all,
            all_max_date,
            open_all,
            open_c,
        ],
        axis=1,
    ).reset_index()

    df_base = df_base.merge(viol_agg, on="bbl", how="left")

    if not df_comp.empty:
        c_hist = df_comp[df_comp["date"] < cutoff_date]
        comp_30d = (
            c_hist[c_hist["date"] >= t_30d].groupby("bbl").size().rename("comp_30d")
        )
        comp_60d = (
            c_hist[c_hist["date"] >= t_60d].groupby("bbl").size().rename("comp_60d")
        )
        comp_90d = (
            c_hist[c_hist["date"] >= t_90d].groupby("bbl").size().rename("comp_90d")
        )
        comp_180d = (
            c_hist[c_hist["date"] >= t_180d].groupby("bbl").size().rename("comp_180d")
        )
        comp_1y = c_hist[c_hist["date"] >= t_1y].groupby("bbl").size().rename("comp_1y")
        comp_3y = c_hist[c_hist["date"] >= t_3y].groupby("bbl").size().rename("comp_3y")
        comp_max_date = c_hist.groupby("bbl")["date"].max().rename("comp_max_date")

        c_heat = c_hist[c_hist["is_heat"]]
        comp_heat_30d = (
            c_heat[c_heat["date"] >= t_30d].groupby("bbl").size().rename("comp_heat_30d")
        )
        comp_heat_60d = (
            c_heat[c_heat["date"] >= t_60d].groupby("bbl").size().rename("comp_heat_60d")
        )
        comp_heat_90d = (
            c_heat[c_heat["date"] >= t_90d].groupby("bbl").size().rename("comp_heat_90d")
        )
        comp_heat_1y = (
            c_heat[c_heat["date"] >= t_1y].groupby("bbl").size().rename("comp_heat_1y")
        )
        comp_heat_all = c_heat.groupby("bbl").size().rename("comp_heat_all")
        comp_heat_max_date = c_heat.groupby("bbl")["date"].max().rename("comp_heat_max_date")

        comp_agg = pd.concat(
            [
                comp_30d,
                comp_60d,
                comp_90d,
                comp_180d,
                comp_1y,
                comp_3y,
                comp_max_date,
                comp_heat_30d,
                comp_heat_60d,
                comp_heat_90d,
                comp_heat_1y,
                comp_heat_all,
                comp_heat_max_date,
            ],
            axis=1,
        ).reset_index()
        df_base = df_base.merge(comp_agg, on="bbl", how="left")
    else:
        df_base["comp_30d"] = 0
        df_base["comp_60d"] = 0
        df_base["comp_90d"] = 0
        df_base["comp_180d"] = 0
        df_base["comp_1y"] = 0
        df_base["comp_3y"] = 0
        df_base["comp_max_date"] = pd.NaT
        df_base["comp_heat_30d"] = 0
        df_base["comp_heat_60d"] = 0
        df_base["comp_heat_90d"] = 0
        df_base["comp_heat_1y"] = 0
        df_base["comp_heat_all"] = 0
        df_base["comp_heat_max_date"] = pd.NaT

    if not df_lit.empty:
        l_hist = df_lit[df_lit["date"] < cutoff_date]
        lit_3y = (
            l_hist[l_hist["date"] >= (cutoff_date - pd.Timedelta(days=1095))]
            .groupby("bbl")
            .size()
            .rename("lit_3y")
        )
        lit_all = l_hist.groupby("bbl").size().rename("lit_all")
        lit_agg = pd.concat([lit_3y, lit_all], axis=1).reset_index()
        df_base = df_base.merge(lit_agg, on="bbl", how="left")
    else:
        df_base["lit_3y"] = 0
        df_base["lit_all"] = 0

    if not df_vacate.empty:
        vac_hist = df_vacate[df_vacate["date"] < cutoff_date]
        vac_1y = vac_hist[vac_hist["date"] >= t_1y].groupby("bbl").size().rename("vacate_1y")
        vac_all = vac_hist.groupby("bbl").size().rename("vacate_all")
        vac_max_date = vac_hist.groupby("bbl")["date"].max().rename("vacate_max_date")
        vac_agg = pd.concat([vac_1y, vac_all, vac_max_date], axis=1).reset_index()
        df_base = df_base.merge(vac_agg, on="bbl", how="left")
    else:
        df_base["vacate_1y"] = 0
        df_base["vacate_all"] = 0
        df_base["vacate_max_date"] = pd.NaT

    if not df_hwo.empty:
        hwo_hist = df_hwo[df_hwo["date"] < cutoff_date]
        hwo_1y = hwo_hist[hwo_hist["date"] >= t_1y].groupby("bbl").size().rename("hwo_1y")
        hwo_all = hwo_hist.groupby("bbl").size().rename("hwo_all")
        hwo_max_date = hwo_hist.groupby("bbl")["date"].max().rename("hwo_max_date")
        hwo_agg = pd.concat([hwo_1y, hwo_all, hwo_max_date], axis=1).reset_index()
        df_base = df_base.merge(hwo_agg, on="bbl", how="left")
    else:
        df_base["hwo_1y"] = 0
        df_base["hwo_all"] = 0
        df_base["hwo_max_date"] = pd.NaT

    if not df_omo.empty:
        omo_hist = df_omo[df_omo["date"] < cutoff_date]
        omo_1y = omo_hist[omo_hist["date"] >= t_1y].groupby("bbl").size().rename("omo_1y")
        omo_all = omo_hist.groupby("bbl").size().rename("omo_all")
        omo_max_date = omo_hist.groupby("bbl")["date"].max().rename("omo_max_date")
        omo_agg = pd.concat([omo_1y, omo_all, omo_max_date], axis=1).reset_index()
        df_base = df_base.merge(omo_agg, on="bbl", how="left")
    else:
        df_base["omo_1y"] = 0
        df_base["omo_all"] = 0
        df_base["omo_max_date"] = pd.NaT

    if not df_dob.empty:
        d_hist = df_dob[df_dob["date"] < cutoff_date]
        dob_1y = d_hist[d_hist["date"] >= t_1y].groupby("bbl").size().rename("dob_1y")
        dob_all = d_hist.groupby("bbl").size().rename("dob_all")
        dob_max_date = d_hist.groupby("bbl")["date"].max().rename("dob_max_date")
        dob_agg = pd.concat([dob_1y, dob_all, dob_max_date], axis=1).reset_index()
        df_base = df_base.merge(dob_agg, on="bbl", how="left")
    else:
        df_base["dob_1y"] = 0
        df_base["dob_all"] = 0
        df_base["dob_max_date"] = pd.NaT

    if not df_dob_safety.empty:
        ds_hist = df_dob_safety[df_dob_safety["date"] < cutoff_date]
        dob_safety_1y = ds_hist[ds_hist["date"] >= t_1y].groupby("bbl").size().rename("dob_safety_1y")
        dob_safety_all = ds_hist.groupby("bbl").size().rename("dob_safety_all")
        dob_safety_max_date = ds_hist.groupby("bbl")["date"].max().rename("dob_safety_max_date")
        dob_safety_agg = pd.concat([dob_safety_1y, dob_safety_all, dob_safety_max_date], axis=1).reset_index()
        df_base = df_base.merge(dob_safety_agg, on="bbl", how="left")
    else:
        df_base["dob_safety_1y"] = 0
        df_base["dob_safety_all"] = 0
        df_base["dob_safety_max_date"] = pd.NaT

    if not df_bedbug.empty:
        bb_hist = df_bedbug[df_bedbug["date"] < cutoff_date]
        bedbug_1y = bb_hist[bb_hist["date"] >= t_1y].groupby("bbl").size().rename("bedbug_1y")
        bedbug_all = bb_hist.groupby("bbl").size().rename("bedbug_all")
        bedbug_max_date = bb_hist.groupby("bbl")["date"].max().rename("bedbug_max_date")
        bb_agg = pd.concat([bedbug_1y, bedbug_all, bedbug_max_date], axis=1).reset_index()
        df_base = df_base.merge(bb_agg, on="bbl", how="left")
    else:
        df_base["bedbug_1y"] = 0
        df_base["bedbug_all"] = 0
        df_base["bedbug_max_date"] = pd.NaT

    if not df_evict.empty:
        ev_hist = df_evict[df_evict["date"] < cutoff_date]
        evict_1y = ev_hist[ev_hist["date"] >= t_1y].groupby("bbl").size().rename("evict_1y")
        evict_all = ev_hist.groupby("bbl").size().rename("evict_all")
        evict_max_date = ev_hist.groupby("bbl")["date"].max().rename("evict_max_date")
        ev_agg = pd.concat([evict_1y, evict_all, evict_max_date], axis=1).reset_index()
        df_base = df_base.merge(ev_agg, on="bbl", how="left")
    else:
        df_base["evict_1y"] = 0
        df_base["evict_all"] = 0
        df_base["evict_max_date"] = pd.NaT

    if not df_ecb.empty:
        ecb_hist = df_ecb[df_ecb["date"] < cutoff_date]
        ecb_1y = ecb_hist[ecb_hist["date"] >= t_1y].groupby("bbl").size().rename("ecb_1y")
        ecb_all = ecb_hist.groupby("bbl").size().rename("ecb_all")
        ecb_max_date = ecb_hist.groupby("bbl")["date"].max().rename("ecb_max_date")
        ecb_agg = pd.concat([ecb_1y, ecb_all, ecb_max_date], axis=1).reset_index()
        df_base = df_base.merge(ecb_agg, on="bbl", how="left")
    else:
        df_base["ecb_1y"] = 0
        df_base["ecb_all"] = 0
        df_base["ecb_max_date"] = pd.NaT

    if not df_rodent.empty:
        r_hist = df_rodent[df_rodent["date"] < cutoff_date]
        rodent_fail_1y = (
            r_hist[r_hist["date"] >= t_1y].groupby("bbl").size().rename("rodent_fail_1y")
        )
        rodent_fail_all = r_hist.groupby("bbl").size().rename("rodent_fail_all")
        rodent_fail_max_date = (
            r_hist.groupby("bbl")["date"].max().rename("rodent_fail_max_date")
        )
        rodent_agg = pd.concat(
            [rodent_fail_1y, rodent_fail_all, rodent_fail_max_date], axis=1
        ).reset_index()
        df_base = df_base.merge(rodent_agg, on="bbl", how="left")
    else:
        df_base["rodent_fail_1y"] = 0
        df_base["rodent_fail_all"] = 0
        df_base["rodent_fail_max_date"] = pd.NaT

    if not df_tax_lien.empty:
        tl_hist = df_tax_lien[df_tax_lien["date"] < cutoff_date]
        tax_lien_count = tl_hist.groupby("bbl").size().rename("tax_lien_count")
        tax_lien_max_date = tl_hist.groupby("bbl")["date"].max().rename("tax_lien_max_date")
        tl_agg = pd.concat([tax_lien_count, tax_lien_max_date], axis=1).reset_index()
        df_base = df_base.merge(tl_agg, on="bbl", how="left")
    else:
        df_base["tax_lien_count"] = 0
        df_base["tax_lien_max_date"] = pd.NaT

    df_base = df_base.merge(df_pluto, on="bbl", how="left")

    count_cols = [
        "viol_c_30d",
        "viol_c_60d",
        "viol_c_90d",
        "viol_c_180d",
        "viol_c_1y",
        "viol_c_2y",
        "viol_c_3y",
        "viol_c_5y",
        "viol_c_all",
        "viol_b_30d",
        "viol_b_90d",
        "viol_b_180d",
        "viol_b_1y",
        "viol_b_2y",
        "viol_b_all",
        "viol_a_1y",
        "viol_a_all",
        "viol_all_30d",
        "viol_all_90d",
        "viol_all_180d",
        "viol_all_1y",
        "viol_all_2y",
        "viol_all_all",
        "viol_open_count",
        "viol_c_open_count",
        "comp_30d",
        "comp_60d",
        "comp_90d",
        "comp_180d",
        "comp_1y",
        "comp_3y",
        "comp_heat_30d",
        "comp_heat_60d",
        "comp_heat_90d",
        "comp_heat_1y",
        "comp_heat_all",
        "lit_3y",
        "lit_all",
        "vacate_1y",
        "vacate_all",
        "hwo_1y",
        "hwo_all",
        "omo_1y",
        "omo_all",
        "dob_1y",
        "dob_all",
        "dob_safety_1y",
        "dob_safety_all",
        "bedbug_1y",
        "bedbug_all",
        "evict_1y",
        "evict_all",
        "ecb_1y",
        "ecb_all",
        "rodent_fail_1y",
        "rodent_fail_all",
        "tax_lien_count",
    ]
    for col in count_cols:
        if col in df_base.columns:
            df_base[col] = df_base[col].fillna(0).astype(np.float32)

    df_base["has_tax_lien"] = (df_base["tax_lien_count"] > 0).astype(np.float32)
    df_base["has_repeat_tax_lien"] = (df_base["tax_lien_count"] > 1).astype(np.float32)

    df_base["viol_c_days_since"] = (
        (cutoff_date - df_base["viol_c_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["viol_b_days_since"] = (
        (cutoff_date - df_base["viol_b_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["viol_all_days_since"] = (
        (cutoff_date - df_base["viol_all_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["comp_days_since"] = (
        (cutoff_date - df_base["comp_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["comp_heat_days_since"] = (
        (cutoff_date - df_base["comp_heat_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["vacate_days_since"] = (
        (cutoff_date - df_base["vacate_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["hwo_days_since"] = (
        (cutoff_date - df_base["hwo_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["omo_days_since"] = (
        (cutoff_date - df_base["omo_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["dob_days_since"] = (
        (cutoff_date - df_base["dob_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["dob_safety_days_since"] = (
        (cutoff_date - df_base["dob_safety_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["bedbug_days_since"] = (
        (cutoff_date - df_base["bedbug_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["evict_days_since"] = (
        (cutoff_date - df_base["evict_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["ecb_days_since"] = (
        (cutoff_date - df_base["ecb_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["rodent_fail_days_since"] = (
        (cutoff_date - df_base["rodent_fail_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base["tax_lien_days_since"] = (
        (cutoff_date - df_base["tax_lien_max_date"])
        .dt.days.fillna(3650.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )
    df_base = df_base.drop(
        columns=[
            "viol_c_max_date",
            "viol_b_max_date",
            "viol_all_max_date",
            "comp_max_date",
            "comp_heat_max_date",
            "vacate_max_date",
            "hwo_max_date",
            "omo_max_date",
            "dob_max_date",
            "dob_safety_max_date",
            "bedbug_max_date",
            "evict_max_date",
            "ecb_max_date",
            "rodent_fail_max_date",
            "tax_lien_max_date",
        ],
        errors="ignore",
    )

    df_base["has_prior_c"] = (df_base["viol_c_all"] > 0).astype(np.int32)
    df_base["viol_c_trend"] = (
        df_base["viol_c_1y"] / (df_base["viol_c_2y"] - df_base["viol_c_1y"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_accel_90d"] = (
        df_base["viol_c_90d"] - (df_base["viol_c_180d"] - df_base["viol_c_90d"])
    ).astype(np.float32)
    df_base["comp_velocity"] = (
        df_base["comp_90d"] * 4.0 / (df_base["comp_1y"] + 1.0)
    ).astype(np.float32)
    df_base["comp_velocity_30d"] = (
        (df_base["comp_30d"] * 12.0) / (df_base["comp_1y"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_velocity_30d"] = (
        (df_base["comp_heat_30d"] * 12.0) / (df_base["comp_heat_1y"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_accel_30_90"] = (
        df_base["comp_heat_30d"] / (df_base["comp_heat_90d"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_velocity_30_vs_90"] = (
        (df_base["comp_heat_30d"] * 3.0) / (df_base["comp_heat_90d"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_accel"] = (
        df_base["comp_heat_30d"] * 3.0 - df_base["comp_heat_90d"]
    ).astype(np.float32)
    df_base["comp_heat_ratio_1y"] = (
        df_base["comp_heat_1y"] / (df_base["comp_1y"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_ratio_30d"] = (
        df_base["comp_heat_30d"] / (df_base["comp_30d"] + 1.0)
    ).astype(np.float32)
    df_base["comp_heat_per_res_unit"] = (
        df_base["comp_heat_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)

    df_base["viol_c_has_y1"] = (df_base["viol_c_1y"] > 0).astype(np.float32)
    df_base["viol_c_has_y2"] = (
        (df_base["viol_c_2y"] - df_base["viol_c_1y"]) > 0
    ).astype(np.float32)
    df_base["viol_c_has_y3"] = (
        (df_base["viol_c_3y"] - df_base["viol_c_2y"]) > 0
    ).astype(np.float32)
    df_base["viol_c_active_years"] = (
        df_base["viol_c_has_y1"] + df_base["viol_c_has_y2"] + df_base["viol_c_has_y3"]
    ).astype(np.float32)

    df_base["viol_c_60d_to_180d_ratio"] = (
        df_base["viol_c_60d"] / (df_base["viol_c_180d"] + 1.0)
    ).astype(np.float32)
    df_base["comp_60d_to_180d_ratio"] = (
        df_base["comp_60d"] / (df_base["comp_180d"] + 1.0)
    ).astype(np.float32)

    c_1y_dict = c_1y.to_dict()
    pluto_c = df_pluto["bbl"].map(c_1y_dict).fillna(0.0)
    zip_c_map = (
        pluto_c
        .groupby(df_pluto["zipcode"].astype(str))
        .mean()
    )
    df_base["zip_viol_c_1y_mean"] = (
        df_base["zipcode"].astype(str).map(zip_c_map).fillna(0.0).astype(np.float32)
    )

    base_block = df_base["bbl"].astype(str).str[:6]
    block_c_sum_map = pluto_c.groupby(df_pluto["block_prefix"]).sum()
    block_count_map = df_pluto.groupby("block_prefix")["bbl"].count()

    block_c_sum = base_block.map(block_c_sum_map).fillna(0.0).astype(np.float32)
    block_count = base_block.map(block_count_map).fillna(0.0).astype(np.float32)
    lot_c_1y = df_base["viol_c_1y"].fillna(0.0).astype(np.float32)

    df_base["block_viol_c_1y_mean"] = np.clip(
        (block_c_sum - lot_c_1y) / np.maximum(block_count - 1.0, 1.0),
        0.0,
        None,
    ).fillna(0.0).astype(np.float32)

    if "cd" in df_base.columns and "cd" in df_pluto.columns:
        cd_c_sum_map = pluto_c.groupby(df_pluto["cd"].astype(str)).sum()
        cd_count_map = df_pluto.groupby(df_pluto["cd"].astype(str))["bbl"].count()
        base_cd = df_base["cd"].astype(str)
        cd_c_sum = base_cd.map(cd_c_sum_map).fillna(0.0).astype(np.float32)
        cd_count = base_cd.map(cd_count_map).fillna(0.0).astype(np.float32)
        df_base["cd_viol_c_1y_mean"] = np.clip(
            (cd_c_sum - lot_c_1y) / np.maximum(cd_count - 1.0, 1.0),
            0.0,
            None,
        ).fillna(0.0).astype(np.float32)
    else:
        df_base["cd_viol_c_1y_mean"] = 0.0

    if not df_comp.empty:
        comp_1y_dict = comp_1y.to_dict()
        pluto_comp = df_pluto["bbl"].map(comp_1y_dict).fillna(0.0)
        block_comp_sum_map = pluto_comp.groupby(df_pluto["block_prefix"]).sum()
        block_comp_sum = base_block.map(block_comp_sum_map).fillna(0.0).astype(np.float32)
        lot_comp_1y = df_base["comp_1y"].fillna(0.0).astype(np.float32)
        df_base["block_comp_1y_mean"] = np.clip(
            (block_comp_sum - lot_comp_1y) / np.maximum(block_count - 1.0, 1.0),
            0.0,
            None,
        ).fillna(0.0).astype(np.float32)
    else:
        df_base["block_comp_1y_mean"] = 0.0

    df_base["viol_close_ratio"] = (
        (df_base["viol_all_all"] - df_base["viol_open_count"])
        / (df_base["viol_all_all"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_open_ratio"] = (
        df_base["viol_c_open_count"] / (df_base["viol_c_all"] + 1.0)
    ).astype(np.float32)
    df_base["b_to_c_ratio"] = (
        df_base["viol_b_1y"] / (df_base["viol_c_1y"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_to_all_1y"] = (
        df_base["viol_c_1y"] / (df_base["viol_all_1y"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_to_all_all"] = (
        df_base["viol_c_all"] / (df_base["viol_all_all"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_to_all_90d"] = (
        df_base["viol_c_90d"] / (df_base["viol_all_90d"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_to_all_30d"] = (
        df_base["viol_c_30d"] / (df_base["viol_all_30d"] + 1.0)
    ).astype(np.float32)
    df_base["viol_open_ratio"] = (
        df_base["viol_open_count"] / (df_base["viol_all_all"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_open_share"] = (
        df_base["viol_c_open_count"] / (df_base["viol_open_count"] + 1.0)
    ).astype(np.float32)
    df_base["viol_c_open_per_unit"] = (
        df_base["viol_c_open_count"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["viol_open_per_unit"] = (
        df_base["viol_open_count"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)

    df_base["viol_c_per_res_unit"] = (
        df_base["viol_c_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["viol_all_per_res_unit"] = (
        df_base["viol_all_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["comp_per_res_unit"] = (
        df_base["comp_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["vacate_per_res_unit"] = (
        df_base["vacate_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["hwo_per_res_unit"] = (
        df_base["hwo_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["omo_per_res_unit"] = (
        df_base["omo_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)
    df_base["erp_1y"] = (df_base["hwo_1y"] + df_base["omo_1y"]).astype(np.float32)
    df_base["erp_all"] = (df_base["hwo_all"] + df_base["omo_all"]).astype(np.float32)
    df_base["erp_per_res_unit"] = (
        df_base["erp_1y"] / (df_base["unitsres"] + 1.0)
    ).astype(np.float32)

    if bbl_to_reg_map and not df_reg.empty:
        reg_bbls_df = df_reg[["bbl", "registrationid"]].copy()
        reg_bbls_df["c_1y"] = reg_bbls_df["bbl"].map(c_1y).fillna(0.0).astype(np.float32)
        hwo_map = hwo_1y.to_dict() if ("hwo_1y" in locals() and not df_hwo.empty) else {}
        omo_map = omo_1y.to_dict() if ("omo_1y" in locals() and not df_omo.empty) else {}
        reg_bbls_df["erp_1y"] = (
            reg_bbls_df["bbl"].map(hwo_map).fillna(0.0)
            + reg_bbls_df["bbl"].map(omo_map).fillna(0.0)
        ).astype(np.float32)

        port_bldg_cnt = reg_bbls_df.groupby("registrationid")["bbl"].nunique()
        port_c_sum = reg_bbls_df.groupby("registrationid")["c_1y"].sum()
        port_erp_sum = reg_bbls_df.groupby("registrationid")["erp_1y"].sum()

        bbl_port_cnt = reg_bbls_df["registrationid"].map(port_bldg_cnt).astype(np.float32)
        bbl_port_c = reg_bbls_df["registrationid"].map(port_c_sum).astype(np.float32)
        bbl_port_erp = reg_bbls_df["registrationid"].map(port_erp_sum).astype(np.float32)

        cnt_dict = dict(zip(reg_bbls_df["bbl"], bbl_port_cnt))
        c_dict = dict(zip(reg_bbls_df["bbl"], bbl_port_c))
        erp_dict = dict(zip(reg_bbls_df["bbl"], bbl_port_erp))

        df_base["portfolio_bldg_count"] = df_base["bbl"].map(cnt_dict).fillna(1.0).astype(np.float32)
        df_base["portfolio_viol_c_1y"] = df_base["bbl"].map(c_dict).fillna(df_base["viol_c_1y"]).astype(np.float32)
        df_base["portfolio_erp_1y"] = df_base["bbl"].map(erp_dict).fillna(df_base["erp_1y"]).astype(np.float32)
        df_base["portfolio_c_per_bldg"] = (
            df_base["portfolio_viol_c_1y"] / df_base["portfolio_bldg_count"]
        ).astype(np.float32)
    else:
        df_base["portfolio_bldg_count"] = 1.0
        df_base["portfolio_viol_c_1y"] = df_base["viol_c_1y"].astype(np.float32)
        df_base["portfolio_erp_1y"] = df_base["erp_1y"].astype(np.float32)
        df_base["portfolio_c_per_bldg"] = df_base["viol_c_1y"].astype(np.float32)

    df_base["has_vacate"] = (df_base["vacate_all"] > 0).astype(np.int32)
    df_base["has_erp"] = (df_base["erp_all"] > 0).astype(np.int32)
    df_base["in_aep"] = df_base["bbl"].isin(aep_bbls).astype(np.int32)
    df_base["in_conh"] = df_base["bbl"].isin(conh_bbls).astype(np.int32)
    df_base["in_underlying_cond"] = df_base["bbl"].isin(underlying_bbls).astype(np.int32)
    df_base["in_speculation_list"] = df_base["bbl"].isin(speculation_bbls).astype(np.int32)

    bldg_age = cutoff_date.year - df_base["yearbuilt"]
    df_base["bldg_age"] = np.where(
        (df_base["yearbuilt"] > 1800) & (bldg_age >= 0), bldg_age, np.nan
    ).astype(np.float32)

    if label_window_months is not None:
        next_cutoff = cutoff_date + pd.DateOffset(months=label_window_months)
        pos_bbls = set(
            df_viol[
                (df_viol["class"] == "C")
                & (df_viol["inspectiondate"] >= cutoff_date)
                & (df_viol["inspectiondate"] < next_cutoff)
            ]["bbl"].unique()
        )
        df_base["target"] = df_base["bbl"].isin(pos_bbls).astype(np.int32)

    return df_base


cohort_bbls = df_pluto[df_pluto["unitsres"] >= 3]["bbl"].values

df_train_2020 = build_point_in_time_dataset(
    cohort_bbls, CUTOFF_TRAIN_2020, label_window_months=12
)
df_train_2021 = build_point_in_time_dataset(
    cohort_bbls, CUTOFF_TRAIN_2021, label_window_months=12
)
df_train = pd.concat([df_train_2020, df_train_2021], ignore_index=True)
del df_train_2020, df_train_2021
gc.collect()

df_val = build_point_in_time_dataset(cohort_bbls, CUTOFF_VAL, label_window_months=12)
df_test = build_point_in_time_dataset(test_bbls, CUTOFF_TEST, label_window_months=None)
df_test = pd.DataFrame({"bbl": test_bbls}).merge(df_test, on="bbl", how="left")

del (
    df_viol,
    df_comp,
    df_lit,
    df_vacate,
    df_hwo,
    df_omo,
    df_dob,
    df_dob_safety,
    df_bedbug,
    df_evict,
    df_ecb,
    df_rodent,
    df_pluto,
    df_reg,
    df_tax_lien,
    bbl_to_reg_map,
)
gc.collect()

feature_cols = [c for c in df_train.columns if c not in ["bbl", "target"]]
cat_cols = [
    c
    for c in feature_cols
    if df_train[c].dtype == "category" or df_train[c].dtype == "object"
]
num_cols = [c for c in feature_cols if c not in cat_cols]

for c in cat_cols:
    combined_cats = (
        pd.concat([df_train[c].dropna(), df_val[c].dropna(), df_test[c].dropna()])
        .astype(str)
        .unique()
    )
    cat_dtype = pd.CategoricalDtype(categories=sorted(combined_cats))
    df_train[c] = (
        df_train[c]
        .astype(str)
        .where(df_train[c].notna(), np.nan)
        .astype(cat_dtype)
    )
    df_val[c] = (
        df_val[c]
        .astype(str)
        .where(df_val[c].notna(), np.nan)
        .astype(cat_dtype)
    )
    df_test[c] = (
        df_test[c]
        .astype(str)
        .where(df_test[c].notna(), np.nan)
        .astype(cat_dtype)
    )

X_train, y_train = df_train[feature_cols], df_train["target"].values
X_val, y_val = df_val[feature_cols], df_val["target"].values
X_test = df_test[feature_cols]


# -------------------------------------------------------------------------
# 4. Neural Architecture & Loss Definition
# -------------------------------------------------------------------------
class TabResBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return residual + out


class TabResNet(nn.Module):
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list = None,
        cat_embed_dims: list = None,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities or []

        self.embeddings = nn.ModuleList()
        total_cat_dim = 0
        if cat_cardinalities:
            if cat_embed_dims is None:
                cat_embed_dims = [
                    min(32, max(4, int(card**0.5) * 2)) for card in cat_cardinalities
                ]
            for card, edim in zip(cat_cardinalities, cat_embed_dims):
                self.embeddings.append(nn.Embedding(card, edim, padding_idx=0))
                total_cat_dim += edim

        self.num_bn = nn.BatchNorm1d(num_numerical)
        self.num_proj = nn.Linear(num_numerical, hidden_dim)

        fusion_dim = hidden_dim + total_cat_dim
        self.in_proj = nn.Linear(fusion_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        # Feature-wise gating linear unit
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim * 2)

        self.blocks = nn.ModuleList(
            [
                TabResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        x_num_emb = self.num_proj(self.num_bn(x_num))

        if len(self.embeddings) > 0 and x_cat is not None and x_cat.size(1) > 0:
            cat_embeds = [
                emb(x_cat[:, i].long().clamp(0, emb.num_embeddings - 1))
                for i, emb in enumerate(self.embeddings)
            ]
            fused = torch.cat([x_num_emb] + cat_embeds, dim=-1)
        else:
            fused = x_num_emb

        x = self.in_norm(self.in_proj(fused))
        gate_out = self.gate_fc(x)
        val, gate = gate_out.chunk(2, dim=-1)
        x = val * torch.sigmoid(gate)

        for block in self.blocks:
            x = block(x)

        logits = self.head(x)
        return logits.squeeze(-1)


# -------------------------------------------------------------------------
# 5. Model Instantiation & Training
# -------------------------------------------------------------------------
if cat_cols:
    X_train_cat = np.column_stack(
        [
            np.maximum(df_train[c].cat.codes.values + 1, 0).astype(np.int64)
            for c in cat_cols
        ]
    )
    X_val_cat = np.column_stack(
        [
            np.maximum(df_val[c].cat.codes.values + 1, 0).astype(np.int64)
            for c in cat_cols
        ]
    )
    X_test_cat = np.column_stack(
        [
            np.maximum(df_test[c].cat.codes.values + 1, 0).astype(np.int64)
            for c in cat_cols
        ]
    )
    cat_cardinalities = [len(df_train[c].cat.categories) + 2 for c in cat_cols]
else:
    X_train_cat = np.zeros((len(df_train), 0), dtype=np.int64)
    X_val_cat = np.zeros((len(df_val), 0), dtype=np.int64)
    X_test_cat = np.zeros((len(df_test), 0), dtype=np.int64)
    cat_cardinalities = []

def transform_num_for_nn(df, cols):
    arr = df[cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values.copy()
    for j, c in enumerate(cols):
        c_lower = c.lower()
        if any(
            k in c_lower
            for k in [
                "viol",
                "comp",
                "lit",
                "vacate",
                "hwo",
                "omo",
                "dob",
                "bedbug",
                "evict",
                "ecb",
                "rodent",
                "unit",
                "area",
                "assess",
                "ratio",
                "count",
                "trend",
                "velocity",
                "mean",
                "erp",
                "portfolio",
                "lien",
            ]
        ):
            col_vals = arr[:, j]
            if np.all(col_vals >= 0.0):
                arr[:, j] = np.log1p(col_vals)
    return arr

X_train_num_nn = transform_num_for_nn(df_train, num_cols)
X_val_num_nn = transform_num_for_nn(df_val, num_cols)
X_test_num_nn = transform_num_for_nn(df_test, num_cols)

scaler = StandardScaler()
scaler.fit(X_train_num_nn)

X_train_num_scaled = np.nan_to_num(
    scaler.transform(X_train_num_nn), nan=0.0, posinf=0.0, neginf=0.0
).astype(np.float32)
X_val_num_scaled = np.nan_to_num(
    scaler.transform(X_val_num_nn), nan=0.0, posinf=0.0, neginf=0.0
).astype(np.float32)
X_test_num_scaled = np.nan_to_num(
    scaler.transform(X_test_num_nn), nan=0.0, posinf=0.0, neginf=0.0
).astype(np.float32)

y_train_arr = df_train["target"].values.astype(np.float32)
y_val_arr = df_val["target"].values.astype(np.float32)

model = TabResNet(
    num_numerical=len(num_cols),
    cat_cardinalities=cat_cardinalities,
    hidden_dim=128,
    num_blocks=3,
    dropout_rate=0.2,
)

criterion = nn.BCEWithLogitsLoss()

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "norm" in name or "bn" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer=optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)

batch_size = 512
train_dataset = TensorDataset(
    torch.from_numpy(X_train_num_scaled),
    torch.from_numpy(X_train_cat),
    torch.from_numpy(y_train_arr),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_num_scaled),
    torch.from_numpy(X_val_cat),
    torch.from_numpy(y_val_arr),
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_num_scaled),
    torch.from_numpy(X_test_cat),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

model.to(device)
criterion.to(device)

num_epochs = 12
patience = 4
patience_counter = 0
best_val_ap_nn = -1.0
best_model_weights = None

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_num_b, x_cat_b, targets_b in train_loader:
        x_num_b = x_num_b.to(device)
        x_cat_b = x_cat_b.to(device)
        targets_b = targets_b.to(device)

        optimizer.zero_grad()
        logits = model(x_num_b, x_cat_b)
        loss = criterion(logits, targets_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * targets_b.size(0)
        total_samples += targets_b.size(0)

    scheduler.step()
    epoch_loss = running_loss / max(1, total_samples)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for x_num_b, x_cat_b, _ in val_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            probs = torch.sigmoid(model(x_num_b, x_cat_b))
            val_preds_list.append(probs.cpu().numpy())

    val_preds_nn = np.concatenate(val_preds_list)
    val_ap_nn = float(average_precision_score(y_val_arr, val_preds_nn))

    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} - Train Loss: {epoch_loss:.4f} - Val AP: {val_ap_nn:.5f}"
    )

    if val_ap_nn > best_val_ap_nn:
        best_val_ap_nn = val_ap_nn
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
model.eval()

val_preds_list = []
with torch.no_grad():
    for x_num_b, x_cat_b, _ in val_loader:
        probs = torch.sigmoid(model(x_num_b.to(device), x_cat_b.to(device)))
        val_preds_list.append(probs.cpu().numpy())
val_preds_nn = np.concatenate(val_preds_list)
final_val_ap_nn = float(average_precision_score(y_val_arr, val_preds_nn))

# -------------------------------------------------------------------------
# 6. Complementary Tree Model & Ensembling
# -------------------------------------------------------------------------
lgb_model_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 47,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.70,
    "subsample_freq": 1,
    "colsample_bytree": 0.70,
    "scale_pos_weight": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

dtrain = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
)
dval = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=cat_cols,
    reference=dtrain,
    free_raw_data=False,
)


def lgb_ap_eval(preds, train_data):
    labels = train_data.get_label()
    score = average_precision_score(labels, preds)
    return "average_precision", score, True


lgb_model = lgb.train(
    lgb_model_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    feval=lgb_ap_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=60, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_preds_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
val_ap_lgb = float(average_precision_score(y_val_arr, val_preds_lgb))

xgb_model_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 20.0,
    "subsample": 0.70,
    "colsample_bytree": 0.70,
    "scale_pos_weight": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_estimators": 1000,
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}

xgb_model = xgb.XGBClassifier(
    **xgb_model_params,
    early_stopping_rounds=60,
)
try:
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_params = copy.deepcopy(xgb_model_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=60,
        verbose=False,
    )

val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
val_ap_xgb = float(average_precision_score(y_val_arr, val_preds_xgb))

cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=5.0,
    eval_metric="PRAUC",
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
    thread_count=-1,
    cat_features=cat_cols,
)
cb_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=60,
    verbose=False,
)

val_preds_cb = cb_model.predict_proba(X_val)[:, 1]
val_ap_cb = float(average_precision_score(y_val_arr, val_preds_cb))

print(f"Validation AP - LightGBM:  {val_ap_lgb:.5f}")
print(f"Validation AP - XGBoost:   {val_ap_xgb:.5f}")
print(f"Validation AP - CatBoost:  {val_ap_cb:.5f}")
print(f"Validation AP - TabResNet: {final_val_ap_nn:.5f}")

rank_nn = rankdata(val_preds_nn) / len(val_preds_nn)
rank_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)
rank_xgb = rankdata(val_preds_xgb) / len(val_preds_xgb)
rank_cb = rankdata(val_preds_cb) / len(val_preds_cb)

# Coarse grid search for robust starting weights
coarse_best_val_ap = -1.0
coarse_best_weights = (0.25, 0.40, 0.20, 0.15)

for i in range(21):
    w_lgb = i * 0.05
    term_lgb = w_lgb * rank_lgb
    for j in range(21 - i):
        w_xgb = j * 0.05
        term_lx = term_lgb + w_xgb * rank_xgb
        for k in range(21 - i - j):
            w_cb = k * 0.05
            l = 20 - i - j - k
            w_nn = l * 0.05
            val_blend = term_lx + w_cb * rank_cb + w_nn * rank_nn
            score = float(average_precision_score(y_val_arr, val_blend))
            if score > coarse_best_val_ap:
                coarse_best_val_ap = score
                coarse_best_weights = (w_lgb, w_xgb, w_cb, w_nn)


# Continuous optimization via Nelder-Mead to directly maximize validation AP
def ensemble_loss_fn(weights_raw):
    w = np.maximum(weights_raw, 0.0)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1.0
    w = w / w_sum
    val_blend = (
        w[0] * rank_lgb
        + w[1] * rank_xgb
        + w[2] * rank_cb
        + w[3] * rank_nn
    )
    return -float(average_precision_score(y_val_arr, val_blend))


opt_res = minimize(
    ensemble_loss_fn,
    x0=np.array(coarse_best_weights, dtype=np.float64),
    method="Nelder-Mead",
    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5},
)

opt_weights = np.maximum(opt_res.x, 0.0)
opt_weights = opt_weights / np.sum(opt_weights)
best_weights = tuple(float(x) for x in opt_weights)
best_val_ap = -float(opt_res.fun)

print(
    f"Optimal Weights (LGB: {best_weights[0]:.4f}, XGB: {best_weights[1]:.4f}, CB: {best_weights[2]:.4f}, NN: {best_weights[3]:.4f}) - Val AP: {best_val_ap:.5f}"
)

test_preds_list = []
with torch.no_grad():
    for x_num_b, x_cat_b in test_loader:
        probs = torch.sigmoid(model(x_num_b.to(device), x_cat_b.to(device)))
        test_preds_list.append(probs.cpu().numpy())
test_preds_nn = np.concatenate(test_preds_list)
test_preds_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
test_preds_cb = cb_model.predict_proba(X_test)[:, 1]

test_rank_nn = rankdata(test_preds_nn) / len(test_preds_nn)
test_rank_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
test_rank_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
test_rank_cb = rankdata(test_preds_cb) / len(test_preds_cb)

w_lgb, w_xgb, w_cb, w_nn = best_weights
final_test_scores = (
    w_lgb * test_rank_lgb
    + w_xgb * test_rank_xgb
    + w_cb * test_rank_cb
    + w_nn * test_rank_nn
)
best_score = best_val_ap

# -------------------------------------------------------------------------
# 7. Submission Verification & Final Metric
# -------------------------------------------------------------------------
df_submission = pd.DataFrame(
    {"bbl": df_test["bbl"].values, "score": final_test_scores.astype(float)}
)

assert len(df_submission) == len(
    df_test_entities
), f"Submission row mismatch: {len(df_submission)} vs {len(df_test_entities)}"
assert df_submission["bbl"].equals(
    df_test_entities["bbl"]
), "Test entity BBL order mismatch!"
assert (
    not df_submission["score"].isna().any()
), "Submission contains invalid NaN scores!"

submission_file = f"{SUBMISSION_DIR}/submission.csv"
df_submission.to_csv(submission_file, index=False)

print(f"Final Validation Score: {best_score}")

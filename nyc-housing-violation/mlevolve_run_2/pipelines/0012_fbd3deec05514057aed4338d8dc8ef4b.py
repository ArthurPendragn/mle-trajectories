import copy
import gc
import json
import os
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 0. Global Setup & Path Configurations
# =========================================================================
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
storage_options = (
    {"token": TOKEN_PATH}
    if os.path.exists(TOKEN_PATH)
    else (
        {"token": "anon"}
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        else {}
    )
)

GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
LAKE_FULL = f"{GCS_BASE}/lake/full"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


BORO_LOOKUP = {
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "MANHATTAN": "1", "MN": "1",
    "BRONX": "2", "BX": "2",
    "BROOKLYN": "3", "BK": "3",
    "QUEENS": "4", "QN": "4",
    "STATEN ISLAND": "5", "STATEN IS": "5", "SI": "5",
}


def clean_bbl_series(
    df,
    bbl_col="bbl",
    boro_col="boroid",
    block_col="block",
    lot_col="lot",
):
    """Standardizes BBL into a clean 10-digit string according to task specifications."""
    cols_lower = {c.lower(): c for c in df.columns}
    actual_bbl_col = cols_lower.get(bbl_col.lower(), bbl_col)
    has_bbl = actual_bbl_col in df.columns
    if has_bbl:
        s = df[actual_bbl_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(10)
        valid_mask = (s.str.len() == 10) & s.str.isdigit() & (s.str[0].isin(["1", "2", "3", "4", "5"]))
    else:
        s = pd.Series("", index=df.index)
        valid_mask = pd.Series(False, index=df.index)

    boro_cols = [cols_lower[c] for c in [boro_col.lower(), "borough", "borocode", "boro"] if c in cols_lower]
    actual_block_col = cols_lower.get(block_col.lower(), block_col)
    actual_lot_col = cols_lower.get(lot_col.lower(), lot_col)
    if len(boro_cols) > 0 and actual_block_col in df.columns and actual_lot_col in df.columns:
        boro_raw = df[boro_cols[0]].astype(str).str.strip().str.upper()
        boro = boro_raw.map(BORO_LOOKUP).fillna(
            pd.to_numeric(df[boro_cols[0]], errors="coerce").fillna(0).astype(int).astype(str)
        )
        block = (
            pd.to_numeric(df[actual_block_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .apply(lambda x: f"{x:05d}")
        )
        lot = (
            pd.to_numeric(df[actual_lot_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .apply(lambda x: f"{x:04d}")
        )
        fallback = boro + block + lot
        return s.where(valid_mask, fallback)
    return s


def load_auxiliary_table(table_name, date_keywords=None):
    """Loads parquet table case-insensitively, resolving BBL and date columns."""
    try:
        df = pd.read_parquet(
            f"{LAKE_FULL}/{table_name}",
            storage_options=storage_options,
        )
    except Exception as e:
        print(f"Warning loading {table_name}: {e}")
        return pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

    cols_lower = {c.lower(): c for c in df.columns}
    bbl_c = None
    for cand in ["bbl", "bbl_number", "lot_bbl"]:
        if cand in cols_lower:
            bbl_c = cols_lower[cand]
            break

    boro_c = None
    for cand in ["boroid", "borough", "boro", "borocode"]:
        if cand in cols_lower:
            boro_c = cols_lower[cand]
            break

    block_c = cols_lower.get("block", "block")
    lot_c = cols_lower.get("lot", "lot")

    df["bbl"] = clean_bbl_series(
        df,
        bbl_col=bbl_c or "bbl",
        boro_col=boro_c or "boroid",
        block_col=block_c,
        lot_col=lot_c,
    )

    date_col = None
    if date_keywords:
        for kw in date_keywords:
            for c in df.columns:
                if kw in c.lower():
                    date_col = c
                    break
            if date_col is not None:
                break

    if date_col is None:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break

    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        if getattr(df["date"].dtype, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    else:
        df["date"] = pd.NaT

    df = df.dropna(subset=["bbl"])
    df = df[(df["bbl"].str.len() == 10) & (df["bbl"].str.isdigit())]
    print(f"Loaded {table_name}: {len(df)} records (date col: {date_col})")
    return df[["bbl", "date"]]


# =========================================================================
# 1. Ingestion: Cohort Construction & Raw Lake Tables
# =========================================================================
print("Starting data ingestion and feature engineering...")

# Test Entities
try:
    df_test_raw = pd.read_parquet(
        f"{GCS_BASE}/test_entities.parquet", storage_options=storage_options
    )
    df_test = pd.DataFrame({"bbl": clean_bbl_series(df_test_raw, bbl_col="bbl")})
except Exception:
    df_sample = pd.read_parquet(
        f"{GCS_BASE}/sample_submission.parquet", storage_options=storage_options
    )
    df_test = pd.DataFrame({"bbl": clean_bbl_series(df_sample, bbl_col="bbl")})

print(f"Loaded test entities: {len(df_test)} lots")

# PLUTO Table
pluto_cols = [
    "bbl",
    "borough",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "bldgarea",
    "numfloors",
    "lotarea",
    "assessland",
    "assesstot",
    "bldgclass",
    "landuse",
    "zipcode",
    "cd",
    "council",
    "builtfar",
    "residfar",
    "yearalter1",
    "yearalter2",
]

try:
    df_pluto = pd.read_parquet(
        f"{LAKE_FULL}/pluto",
        columns=pluto_cols,
        storage_options=storage_options,
    )
except Exception:
    df_pluto = pd.read_parquet(
        f"{LAKE_FULL}/pluto",
        storage_options=storage_options,
    )
    for c in pluto_cols:
        if c not in df_pluto.columns:
            df_pluto[c] = np.nan
    df_pluto = df_pluto[[c for c in pluto_cols if c in df_pluto.columns]]

df_pluto["bbl"] = clean_bbl_series(
    df_pluto, bbl_col="bbl", boro_col="borough", block_col="block", lot_col="lot"
)
df_pluto = df_pluto.drop(columns=["lot"], errors="ignore")
df_pluto = df_pluto.drop_duplicates(subset=["bbl"], keep="last")
print(f"Loaded PLUTO records: {len(df_pluto)} unique lots")

# Create Train (T = 2019-01-01, 2020-01-01, 2021-01-01) and Val (T = 2022-01-01) Cohorts (unitsres >= 3)
pluto_md = df_pluto[df_pluto["unitsres"].fillna(0) >= 3].copy()
all_md_bbls = pluto_md["bbl"].unique()

df_val = pd.DataFrame({"bbl": all_md_bbls})
df_train_2019 = pd.DataFrame({"bbl": all_md_bbls})
df_train_2020 = pd.DataFrame({"bbl": all_md_bbls})
df_train_2021 = pd.DataFrame({"bbl": all_md_bbls})
print(
    f"Constructed train 2019 ({len(df_train_2019)} lots), train 2020 ({len(df_train_2020)} lots), "
    f"train 2021 ({len(df_train_2021)} lots), and val cohort ({len(df_val)} lots)"
)

# HPD Violations Table
viol_cols = ["bbl", "boroid", "block", "lot", "class", "inspectiondate"]
df_viol = pd.read_parquet(
    f"{LAKE_FULL}/hpd_violations",
    columns=viol_cols,
    storage_options=storage_options,
)
df_viol["bbl"] = clean_bbl_series(
    df_viol, bbl_col="bbl", boro_col="boroid", block_col="block", lot_col="lot"
)
df_viol = df_viol.drop(columns=["boroid", "block", "lot"], errors="ignore")
df_viol["inspectiondate"] = pd.to_datetime(df_viol["inspectiondate"], errors="coerce")
if getattr(df_viol["inspectiondate"].dtype, "tz", None) is not None:
    df_viol["inspectiondate"] = df_viol["inspectiondate"].dt.tz_localize(None)

df_viol = df_viol.dropna(subset=["inspectiondate", "bbl"])
df_viol = df_viol[(df_viol["bbl"].str.len() == 10) & (df_viol["bbl"].str.isdigit())]
df_viol["class"] = df_viol["class"].astype(str).str.strip().str.upper()
print(f"Loaded HPD Violations: {len(df_viol)} records")

# Compute Ground-Truth Target Labels
c_violations = df_viol[df_viol["class"] == "C"]

train_2019_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2019-01-01")
        & (c_violations["inspectiondate"] < "2020-01-01")
    ]["bbl"].unique()
)
df_train_2019["target"] = df_train_2019["bbl"].isin(train_2019_pos_bbls).astype(np.int32)

train_2020_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2020-01-01")
        & (c_violations["inspectiondate"] < "2021-01-01")
    ]["bbl"].unique()
)
df_train_2020["target"] = df_train_2020["bbl"].isin(train_2020_pos_bbls).astype(np.int32)

train_2021_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2021-01-01")
        & (c_violations["inspectiondate"] < "2022-01-01")
    ]["bbl"].unique()
)
df_train_2021["target"] = df_train_2021["bbl"].isin(train_2021_pos_bbls).astype(np.int32)

val_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2022-01-01")
        & (c_violations["inspectiondate"] < "2023-01-01")
    ]["bbl"].unique()
)
df_val["target"] = df_val["bbl"].isin(val_pos_bbls).astype(np.int32)
print(
    f"Train 2019 positive rate: {df_train_2019['target'].mean():.4f} | "
    f"Train 2020 positive rate: {df_train_2020['target'].mean():.4f} | "
    f"Train 2021 positive rate: {df_train_2021['target'].mean():.4f} | "
    f"Val positive rate: {df_val['target'].mean():.4f}"
)

# Auxiliary Indicators
try:
    df_complaints = pd.read_parquet(
        f"{LAKE_FULL}/hpd_complaints",
        columns=["bbl", "receiveddate"],
        storage_options=storage_options,
    )
    date_col = "receiveddate"
except Exception:
    try:
        df_complaints = pd.read_parquet(
            f"{LAKE_FULL}/hpd_complaints",
            columns=["bbl", "dateentered"],
            storage_options=storage_options,
        )
        date_col = "dateentered"
    except Exception:
        df_complaints = pd.DataFrame(columns=["bbl", "receiveddate"])
        date_col = "receiveddate"

df_complaints["bbl"] = clean_bbl_series(df_complaints, bbl_col="bbl")
df_complaints["date"] = pd.to_datetime(df_complaints[date_col], errors="coerce")
if getattr(df_complaints["date"].dtype, "tz", None) is not None:
    df_complaints["date"] = df_complaints["date"].dt.tz_localize(None)
df_complaints = df_complaints.dropna(subset=["bbl", "date"])
df_complaints = df_complaints[
    (df_complaints["bbl"].str.len() == 10) & (df_complaints["bbl"].str.isdigit())
]

try:
    df_lit = pd.read_parquet(
        f"{LAKE_FULL}/hpd_litigations",
        columns=["bbl", "caseopendate"],
        storage_options=storage_options,
    )
    df_lit["bbl"] = clean_bbl_series(df_lit, bbl_col="bbl")
    df_lit["date"] = pd.to_datetime(df_lit["caseopendate"], errors="coerce")
    if getattr(df_lit["date"].dtype, "tz", None) is not None:
        df_lit["date"] = df_lit["date"].dt.tz_localize(None)
    df_lit = df_lit.dropna(subset=["bbl", "date"])
    df_lit = df_lit[(df_lit["bbl"].str.len() == 10) & (df_lit["bbl"].str.isdigit())]
except Exception:
    df_lit = pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

try:
    df_vacate = pd.read_parquet(
        f"{LAKE_FULL}/hpd_vacate_orders",
        storage_options=storage_options,
    )
    df_vacate["bbl"] = clean_bbl_series(df_vacate, bbl_col="bbl")
    v_date_col = [c for c in ["vacate_effective_date", "effective_date", "date"] if c in df_vacate.columns]
    df_vacate["date"] = pd.to_datetime(df_vacate[v_date_col[0]], errors="coerce") if v_date_col else pd.NaT
    if getattr(df_vacate["date"].dtype, "tz", None) is not None:
        df_vacate["date"] = df_vacate["date"].dt.tz_localize(None)
except Exception:
    df_vacate = pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

try:
    df_omo = pd.read_parquet(
        f"{LAKE_FULL}/hpd_omo_charges",
        storage_options=storage_options,
    )
    df_omo["bbl"] = clean_bbl_series(df_omo, bbl_col="bbl", boro_col="boro", block_col="block", lot_col="lot")
    omo_date_col = [c for c in ["invoicedate", "charge_date", "date"] if c in df_omo.columns]
    df_omo["date"] = pd.to_datetime(df_omo[omo_date_col[0]], errors="coerce") if omo_date_col else pd.NaT
    if getattr(df_omo["date"].dtype, "tz", None) is not None:
        df_omo["date"] = df_omo["date"].dt.tz_localize(None)
except Exception:
    df_omo = pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

try:
    df_aep = pd.read_parquet(
        f"{LAKE_FULL}/hpd_aep_buildings",
        storage_options=storage_options,
    )
    df_aep["bbl"] = clean_bbl_series(df_aep, bbl_col="bbl", boro_col="boro", block_col="block", lot_col="lot")
    aep_date_col = [c for c in ["start_date", "date"] if c in df_aep.columns]
    df_aep["date"] = pd.to_datetime(df_aep[aep_date_col[0]], errors="coerce") if aep_date_col else pd.NaT
    if getattr(df_aep["date"].dtype, "tz", None) is not None:
        df_aep["date"] = df_aep["date"].dt.tz_localize(None)
except Exception:
    df_aep = pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

# HPD Heat & Hot Water Charges (hpd_hwo_charges - robust column discovery)
df_hwo = load_auxiliary_table("hpd_hwo_charges", ["invoice", "charge", "issue", "date"])

# DOB Safety Violations
df_dob_safety = load_auxiliary_table("dob_safety_violations", ["violation", "issue", "inspection", "date"])

# NYC Evictions
df_evictions = load_auxiliary_table("evictions", ["executed", "execution", "date"])

# HPD Bedbug Reports
df_bedbug = load_auxiliary_table("hpd_bedbug_reports", ["filing", "file", "date"])

# DOF Tax Lien Sales
df_tax_lien = load_auxiliary_table("dof_tax_lien_sales", ["sale", "notice", "date", "year"])

# DOB Violations Table
try:
    df_dob = pd.read_parquet(
        f"{LAKE_FULL}/dob_violations",
        storage_options=storage_options,
    )
    df_dob["bbl"] = clean_bbl_series(
        df_dob, bbl_col="bbl", boro_col="boro", block_col="block", lot_col="lot"
    )
    dob_date_cols = [
        c
        for c in [
            "issue_date",
            "issuedate",
            "inspection_date",
            "inspectiondate",
            "violation_date",
        ]
        if c in df_dob.columns
    ]
    dob_date_col = (
        dob_date_cols[0]
        if dob_date_cols
        else [c for c in df_dob.columns if "date" in c.lower()][0]
    )
    df_dob["date"] = pd.to_datetime(df_dob[dob_date_col], errors="coerce")
    if getattr(df_dob["date"].dtype, "tz", None) is not None:
        df_dob["date"] = df_dob["date"].dt.tz_localize(None)
    df_dob = df_dob.dropna(subset=["bbl", "date"])
    df_dob = df_dob[
        (df_dob["bbl"].str.len() == 10) & (df_dob["bbl"].str.isdigit())
    ]
    df_dob = df_dob[["bbl", "date"]].copy()
    print(f"Loaded DOB Violations: {len(df_dob)} records")
except Exception as e:
    print(f"Warning loading DOB violations: {e}")
    df_dob = pd.DataFrame(columns=["bbl", "date"]).astype({"date": "datetime64[ns]"})

# DOHMH Rodent Inspections Table
try:
    df_rodent = pd.read_parquet(
        f"{LAKE_FULL}/dohmh_rodent_inspections",
        storage_options=storage_options,
    )
    df_rodent["bbl"] = clean_bbl_series(
        df_rodent, bbl_col="bbl", boro_col="boro", block_col="block", lot_col="lot"
    )
    rodent_date_cols = [
        c
        for c in ["inspection_date", "inspectiondate", "date", "approved_date"]
        if c in df_rodent.columns
    ]
    rodent_date_col = (
        rodent_date_cols[0]
        if rodent_date_cols
        else [c for c in df_rodent.columns if "date" in c.lower()][0]
    )
    df_rodent["date"] = pd.to_datetime(df_rodent[rodent_date_col], errors="coerce")
    if getattr(df_rodent["date"].dtype, "tz", None) is not None:
        df_rodent["date"] = df_rodent["date"].dt.tz_localize(None)
    df_rodent = df_rodent.dropna(subset=["bbl", "date"])
    df_rodent = df_rodent[
        (df_rodent["bbl"].str.len() == 10) & (df_rodent["bbl"].str.isdigit())
    ]
    df_rodent = df_rodent[["bbl", "date"]].copy()
    print(f"Loaded DOHMH Rodent Inspections: {len(df_rodent)} records")
except Exception as e:
    print(f"Warning loading rodent inspections: {e}")
    df_rodent = pd.DataFrame(columns=["bbl", "date"]).astype(
        {"date": "datetime64[ns]"}
    )


# =========================================================================
# 2. Point-in-Time Feature Engineering
# =========================================================================
def extract_cohort_features(entity_df, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)
    cutoff_year = cutoff.year
    res = entity_df[["bbl"]].copy()

    # Static PLUTO Metadata
    res = res.merge(df_pluto, on="bbl", how="left")

    res["unitsres"] = res["unitsres"].fillna(0).astype(np.float32)
    res["unitstotal"] = res["unitstotal"].fillna(0).astype(np.float32)
    res["bldgarea"] = res["bldgarea"].fillna(0).astype(np.float32)
    res["lotarea"] = res["lotarea"].fillna(0).astype(np.float32)
    res["numfloors"] = res["numfloors"].fillna(0).astype(np.float32)
    res["assessland"] = res["assessland"].fillna(0).astype(np.float32)
    res["assesstot"] = res["assesstot"].fillna(0).astype(np.float32)
    res["yearbuilt"] = (
        pd.to_numeric(res["yearbuilt"], errors="coerce")
        .fillna(1950)
        .astype(np.float32)
    )
    res["landuse"] = (
        pd.to_numeric(res["landuse"], errors="coerce").fillna(-1).astype(np.float32)
    )

    res["building_age"] = (
        (cutoff_year - res["yearbuilt"]).clip(0, 200).astype(np.float32)
    )
    # PLUTO Alteration & FAR density metrics
    yalter1 = pd.to_numeric(res.get("yearalter1", 0), errors="coerce").fillna(0)
    yalter2 = pd.to_numeric(res.get("yearalter2", 0), errors="coerce").fillna(0)
    max_alter = np.maximum(yalter1, yalter2)
    res["years_since_alteration"] = np.where(
        max_alter > 1800,
        (cutoff_year - max_alter).clip(0, 200),
        res["building_age"],
    ).astype(np.float32)
    res["has_alteration"] = (max_alter > 1800).astype(np.float32)

    builtfar = pd.to_numeric(res.get("builtfar", 0), errors="coerce").fillna(0).astype(np.float32)
    residfar = pd.to_numeric(res.get("residfar", 0), errors="coerce").fillna(0).astype(np.float32)
    res["builtfar"] = builtfar
    res["residfar"] = residfar
    res["far_ratio"] = (builtfar / (residfar + 0.01)).astype(np.float32)
    res["far_over_built"] = (builtfar > (residfar + 0.05)).astype(np.float32)

    res["res_unit_ratio"] = (res["unitsres"] / (res["unitstotal"] + 1e-4)).astype(
        np.float32
    )
    res["area_per_unit"] = (res["bldgarea"] / (res["unitsres"] + 1.0)).astype(
        np.float32
    )
    res["lot_per_unit"] = (res["lotarea"] / (res["unitsres"] + 1.0)).astype(np.float32)
    res["floors_per_unit"] = (res["numfloors"] / (res["unitsres"] + 1.0)).astype(
        np.float32
    )
    res["assess_per_unit"] = (res["assesstot"] / (res["unitsres"] + 1.0)).astype(
        np.float32
    )
    res["land_value_share"] = (res["assessland"] / (res["assesstot"] + 1e-4)).astype(
        np.float32
    )
    # PLUTO Valuation Density Ratios
    res["assess_per_sqft"] = (res["assesstot"] / (res["bldgarea"] + 1.0)).astype(np.float32)
    res["land_assess_per_lot_sqft"] = (res["assessland"] / (res["lotarea"] + 1.0)).astype(np.float32)
    bldg_val = np.maximum(0.0, res["assesstot"] - res["assessland"])
    res["bldg_assess_per_sqft"] = (bldg_val / (res["bldgarea"] + 1.0)).astype(np.float32)
    res["bldg_to_land_value_ratio"] = (bldg_val / (res["assessland"] + 1.0)).astype(np.float32)

    res["bldgclass_major"] = res["bldgclass"].astype(str).str[:1].fillna("Missing")
    res["bldgclass"] = res["bldgclass"].astype(str).fillna("Missing")
    res["borough"] = res["borough"].astype(str).fillna("Missing")
    res["cd"] = (
        pd.to_numeric(res["cd"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    res["council"] = (
        pd.to_numeric(res["council"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    res["zipcode"] = (
        pd.to_numeric(res["zipcode"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # Temporal Violations Aggregations (< cutoff)
    prior_viol = df_viol[df_viol["inspectiondate"] < cutoff]
    t_14d = cutoff - pd.DateOffset(days=14)
    t_30d = cutoff - pd.DateOffset(days=30)
    t_60d = cutoff - pd.DateOffset(days=60)
    t_90d = cutoff - pd.DateOffset(days=90)
    t_180d = cutoff - pd.DateOffset(days=180)
    t_1y = cutoff - pd.DateOffset(years=1)
    t_2y = cutoff - pd.DateOffset(years=2)
    t_3y = cutoff - pd.DateOffset(years=3)
    t_5y = cutoff - pd.DateOffset(years=5)

    v_14d = prior_viol[prior_viol["inspectiondate"] >= t_14d]
    v_30d = prior_viol[prior_viol["inspectiondate"] >= t_30d]
    v_60d = prior_viol[prior_viol["inspectiondate"] >= t_60d]
    v_90d = prior_viol[prior_viol["inspectiondate"] >= t_90d]
    v_180d = prior_viol[prior_viol["inspectiondate"] >= t_180d]
    v_1y = prior_viol[prior_viol["inspectiondate"] >= t_1y]
    v_2y = prior_viol[prior_viol["inspectiondate"] >= t_2y]
    v_3y = prior_viol[prior_viol["inspectiondate"] >= t_3y]
    v_5y = prior_viol[prior_viol["inspectiondate"] >= t_5y]

    vc_all = prior_viol[prior_viol["class"] == "C"]
    vc_14d = v_14d[v_14d["class"] == "C"]
    vc_30d = v_30d[v_30d["class"] == "C"]
    vc_60d = v_60d[v_60d["class"] == "C"]
    vc_90d = v_90d[v_90d["class"] == "C"]
    vc_180d = v_180d[v_180d["class"] == "C"]
    vc_1y = v_1y[v_1y["class"] == "C"]
    vc_2y = v_2y[v_2y["class"] == "C"]
    vc_3y = v_3y[v_3y["class"] == "C"]
    vc_5y = v_5y[v_5y["class"] == "C"]

    # Annual Longitudinal slices for recurrence flags (t-2: [cutoff-2y, cutoff-1y), t-3: [cutoff-3y, cutoff-2y))
    vc_y2 = prior_viol[
        (prior_viol["class"] == "C")
        & (prior_viol["inspectiondate"] >= t_2y)
        & (prior_viol["inspectiondate"] < t_1y)
    ]
    vc_y3 = prior_viol[
        (prior_viol["class"] == "C")
        & (prior_viol["inspectiondate"] >= t_3y)
        & (prior_viol["inspectiondate"] < t_2y)
    ]

    vb_all = prior_viol[prior_viol["class"] == "B"]
    vb_30d = v_30d[v_30d["class"] == "B"]
    vb_90d = v_90d[v_90d["class"] == "B"]
    vb_180d = v_180d[v_180d["class"] == "B"]
    vb_1y = v_1y[v_1y["class"] == "B"]
    vb_3y = v_3y[v_3y["class"] == "B"]

    va_all = prior_viol[prior_viol["class"] == "A"]
    va_1y = v_1y[v_1y["class"] == "A"]

    def get_counts(df_sub, col_name):
        return (
            df_sub.groupby("bbl")
            .size()
            .rename(col_name)
            .astype(np.float32)
            .reset_index()
        )

    # Distinct Inspection Cadence (Unique inspection dates per lot)
    def get_unique_inspection_counts(df_sub, col_name):
        return (
            df_sub.groupby("bbl")["inspectiondate"]
            .nunique()
            .rename(col_name)
            .astype(np.float32)
            .reset_index()
        )

    counts_list = [
        get_counts(vc_14d, "viol_c_cnt_14d"),
        get_counts(vc_30d, "viol_c_cnt_30d"),
        get_counts(vc_60d, "viol_c_cnt_60d"),
        get_counts(vc_90d, "viol_c_cnt_90d"),
        get_counts(vc_180d, "viol_c_cnt_180d"),
        get_counts(vc_1y, "viol_c_cnt_1y"),
        get_counts(vc_2y, "viol_c_cnt_2y"),
        get_counts(vc_3y, "viol_c_cnt_3y"),
        get_counts(vc_5y, "viol_c_cnt_5y"),
        get_counts(vc_all, "viol_c_cnt_all"),
        get_counts(vc_y2, "viol_c_cnt_t2"),
        get_counts(vc_y3, "viol_c_cnt_t3"),
        get_counts(vb_30d, "viol_b_cnt_30d"),
        get_counts(vb_90d, "viol_b_cnt_90d"),
        get_counts(vb_180d, "viol_b_cnt_180d"),
        get_counts(vb_1y, "viol_b_cnt_1y"),
        get_counts(vb_3y, "viol_b_cnt_3y"),
        get_counts(vb_all, "viol_b_cnt_all"),
        get_counts(va_1y, "viol_a_cnt_1y"),
        get_counts(va_all, "viol_a_cnt_all"),
        get_counts(v_14d, "viol_all_cnt_14d"),
        get_counts(v_30d, "viol_all_cnt_30d"),
        get_counts(v_60d, "viol_all_cnt_60d"),
        get_counts(v_90d, "viol_all_cnt_90d"),
        get_counts(v_180d, "viol_all_cnt_180d"),
        get_counts(v_1y, "viol_all_cnt_1y"),
        get_counts(v_3y, "viol_all_cnt_3y"),
        get_counts(prior_viol, "viol_all_cnt_all"),
        get_unique_inspection_counts(v_1y, "unique_inspections_1y"),
        get_unique_inspection_counts(v_2y, "unique_inspections_2y"),
        get_unique_inspection_counts(v_3y, "unique_inspections_3y"),
        get_unique_inspection_counts(vc_1y, "unique_c_inspections_1y"),
        get_unique_inspection_counts(vc_3y, "unique_c_inspections_3y"),
    ]

    for c_df in counts_list:
        res = res.merge(c_df, on="bbl", how="left")
        col = c_df.columns[1]
        res[col] = res[col].fillna(0).astype(np.float32)

    # Violations per inspection visit
    res["viol_c_per_inspection_1y"] = (
        res["viol_c_cnt_1y"] / (res["unique_inspections_1y"] + 0.1)
    ).astype(np.float32)
    res["viol_all_per_inspection_1y"] = (
        res["viol_all_cnt_1y"] / (res["unique_inspections_1y"] + 0.1)
    ).astype(np.float32)

    # Tax-block Historical Class C Violation Risk (< cutoff)
    prior_viol_block = prior_viol.copy()
    prior_viol_block["tax_block"] = prior_viol_block["bbl"].str[:6]
    block_vc_1y = (
        prior_viol_block[
            (prior_viol_block["class"] == "C")
            & (prior_viol_block["inspectiondate"] >= t_1y)
        ]
        .groupby("tax_block")
        .size()
        .rename("block_viol_c_cnt_1y")
        .astype(np.float32)
        .reset_index()
    )
    block_vc_3y = (
        prior_viol_block[
            (prior_viol_block["class"] == "C")
            & (prior_viol_block["inspectiondate"] >= t_3y)
        ]
        .groupby("tax_block")
        .size()
        .rename("block_viol_c_cnt_3y")
        .astype(np.float32)
        .reset_index()
    )

    res["tax_block"] = res["bbl"].str[:6]
    res = res.merge(block_vc_1y, on="tax_block", how="left")
    res = res.merge(block_vc_3y, on="tax_block", how="left")
    res["block_viol_c_cnt_1y"] = res["block_viol_c_cnt_1y"].fillna(0).astype(np.float32)
    res["block_viol_c_cnt_3y"] = res["block_viol_c_cnt_3y"].fillna(0).astype(np.float32)
    res["block_spillover_c_1y"] = np.maximum(0, res["block_viol_c_cnt_1y"] - res["viol_c_cnt_1y"]).astype(np.float32)
    res = res.drop(columns=["tax_block"])

    # Violation Recency
    max_date_all = (
        prior_viol.groupby("bbl")["inspectiondate"]
        .max()
        .rename("last_viol_date")
        .reset_index()
    )
    res = res.merge(max_date_all, on="bbl", how="left")
    res["days_since_last_violation"] = (
        (cutoff - res["last_viol_date"]).dt.days.fillna(3650).astype(np.float32)
    )
    res = res.drop(columns=["last_viol_date"])

    max_date_c = (
        vc_all.groupby("bbl")["inspectiondate"]
        .max()
        .rename("last_viol_c_date")
        .reset_index()
    )
    res = res.merge(max_date_c, on="bbl", how="left")
    res["days_since_last_viol_c"] = (
        (cutoff - res["last_viol_c_date"]).dt.days.fillna(3650).astype(np.float32)
    )
    res = res.drop(columns=["last_viol_c_date"])

    # Rates, Accelerations, and Ratios
    res["viol_c_ratio_1y"] = (
        res["viol_c_cnt_1y"] / (res["viol_all_cnt_1y"] + 1e-4)
    ).astype(np.float32)
    res["viol_c_ratio_all"] = (
        res["viol_c_cnt_all"] / (res["viol_all_cnt_all"] + 1e-4)
    ).astype(np.float32)
    res["viol_c_velocity"] = (
        res["viol_c_cnt_1y"] / (res["viol_c_cnt_3y"] / 3.0 + 0.1)
    ).astype(np.float32)
    res["viol_all_velocity"] = (
        res["viol_all_cnt_1y"] / (res["viol_all_cnt_3y"] / 3.0 + 0.1)
    ).astype(np.float32)
    res["viol_c_velocity_90d_1y"] = (
        (res["viol_c_cnt_90d"] * 4.0) / (res["viol_c_cnt_1y"] + 0.1)
    ).astype(np.float32)
    res["viol_c_velocity_180d_1y"] = (
        (res["viol_c_cnt_180d"] * 2.0) / (res["viol_c_cnt_1y"] + 0.1)
    ).astype(np.float32)
    res["viol_all_velocity_90d_1y"] = (
        (res["viol_all_cnt_90d"] * 4.0) / (res["viol_all_cnt_1y"] + 0.1)
    ).astype(np.float32)
    res["viol_all_velocity_180d_1y"] = (
        (res["viol_all_cnt_180d"] * 2.0) / (res["viol_all_cnt_1y"] + 0.1)
    ).astype(np.float32)
    res["viol_c_yoy_diff"] = (
        res["viol_c_cnt_1y"] - (res["viol_c_cnt_2y"] - res["viol_c_cnt_1y"])
    ).astype(np.float32)
    res["viol_c_per_unit_1y"] = (res["viol_c_cnt_1y"] / (res["unitsres"] + 1.0)).astype(
        np.float32
    )
    res["viol_c_per_unit_14d"] = (
        res["viol_c_cnt_14d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_c_per_unit_30d"] = (
        res["viol_c_cnt_30d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_c_per_unit_60d"] = (
        res["viol_c_cnt_60d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_c_per_unit_90d"] = (
        res["viol_c_cnt_90d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_c_per_unit_180d"] = (
        res["viol_c_cnt_180d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_all_per_unit_1y"] = (
        res["viol_all_cnt_1y"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_all_per_unit_30d"] = (
        res["viol_all_cnt_30d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)
    res["viol_all_per_unit_90d"] = (
        res["viol_all_cnt_90d"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)

    # Annual Longitudinal Recurrence Flags (Class C in t-1, t-2, and t-3)
    res["has_viol_c_t1"] = (res["viol_c_cnt_1y"] > 0).astype(np.float32)
    res["has_viol_c_t2"] = (res["viol_c_cnt_t2"] > 0).astype(np.float32)
    res["has_viol_c_t3"] = (res["viol_c_cnt_t3"] > 0).astype(np.float32)
    res["recurrent_c_2yr"] = (
        (res["has_viol_c_t1"] > 0) & (res["has_viol_c_t2"] > 0)
    ).astype(np.float32)
    res["recurrent_c_3yr"] = (
        (res["has_viol_c_t1"] > 0) & (res["has_viol_c_t2"] > 0) & (res["has_viol_c_t3"] > 0)
    ).astype(np.float32)
    res["recurrence_c_sum"] = (
        res["has_viol_c_t1"] + res["has_viol_c_t2"] + res["has_viol_c_t3"]
    ).astype(np.float32)

    # Complaints Aggregations (< cutoff)
    if len(df_complaints) > 0:
        prior_comp = df_complaints[df_complaints["date"] < cutoff]
        comp_14d = prior_comp[prior_comp["date"] >= t_14d]
        comp_30d = prior_comp[prior_comp["date"] >= t_30d]
        comp_60d = prior_comp[prior_comp["date"] >= t_60d]
        comp_90d = prior_comp[prior_comp["date"] >= t_90d]
        comp_180d = prior_comp[prior_comp["date"] >= t_180d]
        comp_1y = prior_comp[prior_comp["date"] >= t_1y]
        comp_3y = prior_comp[prior_comp["date"] >= t_3y]

        res = res.merge(get_counts(comp_14d, "complaint_cnt_14d"), on="bbl", how="left")
        res = res.merge(get_counts(comp_30d, "complaint_cnt_30d"), on="bbl", how="left")
        res = res.merge(get_counts(comp_60d, "complaint_cnt_60d"), on="bbl", how="left")
        res = res.merge(get_counts(comp_90d, "complaint_cnt_90d"), on="bbl", how="left")
        res = res.merge(get_counts(comp_180d, "complaint_cnt_180d"), on="bbl", how="left")
        res = res.merge(get_counts(comp_1y, "complaint_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(comp_3y, "complaint_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_comp, "complaint_cnt_all"), on="bbl", how="left")

        res["complaint_cnt_14d"] = res["complaint_cnt_14d"].fillna(0).astype(np.float32)
        res["complaint_cnt_30d"] = res["complaint_cnt_30d"].fillna(0).astype(np.float32)
        res["complaint_cnt_60d"] = res["complaint_cnt_60d"].fillna(0).astype(np.float32)
        res["complaint_cnt_90d"] = res["complaint_cnt_90d"].fillna(0).astype(np.float32)
        res["complaint_cnt_180d"] = res["complaint_cnt_180d"].fillna(0).astype(np.float32)
        res["complaint_cnt_1y"] = res["complaint_cnt_1y"].fillna(0).astype(np.float32)
        res["complaint_cnt_3y"] = res["complaint_cnt_3y"].fillna(0).astype(np.float32)
        res["complaint_cnt_all"] = res["complaint_cnt_all"].fillna(0).astype(np.float32)

        max_comp_date = (
            prior_comp.groupby("bbl")["date"]
            .max()
            .rename("last_comp_date")
            .reset_index()
        )
        res = res.merge(max_comp_date, on="bbl", how="left")
        res["days_since_last_complaint"] = (
            (cutoff - res["last_comp_date"]).dt.days.fillna(3650).astype(np.float32)
        )
        res = res.drop(columns=["last_comp_date"])

        res["complaint_velocity"] = (
            res["complaint_cnt_1y"] / (res["complaint_cnt_3y"] / 3.0 + 0.1)
        ).astype(np.float32)
        res["complaint_velocity_30d_1y"] = (
            (res["complaint_cnt_30d"] * 12.0) / (res["complaint_cnt_1y"] + 0.1)
        ).astype(np.float32)
        res["complaint_velocity_90d_1y"] = (
            (res["complaint_cnt_90d"] * 4.0) / (res["complaint_cnt_1y"] + 0.1)
        ).astype(np.float32)
        res["complaint_per_unit_1y"] = (
            res["complaint_cnt_1y"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)
        res["complaint_per_unit_30d"] = (
            res["complaint_cnt_30d"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)
        res["complaint_per_unit_90d"] = (
            res["complaint_cnt_90d"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)

        # Complaint-to-violation substantiation ratios
        res["complaint_to_viol_ratio_1y"] = (
            res["complaint_cnt_1y"] / (res["viol_all_cnt_1y"] + 1.0)
        ).astype(np.float32)
        res["viol_to_complaint_ratio_1y"] = (
            res["viol_all_cnt_1y"] / (res["complaint_cnt_1y"] + 1.0)
        ).astype(np.float32)
        res["viol_c_to_complaint_ratio_1y"] = (
            res["viol_c_cnt_1y"] / (res["complaint_cnt_1y"] + 1.0)
        ).astype(np.float32)
        res["unsubstantiated_complaint_flag"] = (
            (res["complaint_cnt_1y"] > 2) & (res["viol_all_cnt_1y"] == 0)
        ).astype(np.float32)
    else:
        for c in [
            "complaint_cnt_14d",
            "complaint_cnt_30d",
            "complaint_cnt_60d",
            "complaint_cnt_90d",
            "complaint_cnt_180d",
            "complaint_cnt_1y",
            "complaint_cnt_3y",
            "complaint_cnt_all",
            "days_since_last_complaint",
            "complaint_velocity",
            "complaint_velocity_30d_1y",
            "complaint_velocity_90d_1y",
            "complaint_per_unit_1y",
            "complaint_per_unit_30d",
            "complaint_per_unit_90d",
            "complaint_to_viol_ratio_1y",
            "viol_to_complaint_ratio_1y",
            "viol_c_to_complaint_ratio_1y",
            "unsubstantiated_complaint_flag",
        ]:
            res[c] = np.float32(0)

    # Litigations Aggregations
    if len(df_lit) > 0:
        prior_lit = df_lit[df_lit["date"] < cutoff]
        lit_2y = prior_lit[prior_lit["date"] >= t_2y]
        res = res.merge(get_counts(lit_2y, "litigation_cnt_2y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_lit, "litigation_cnt_all"), on="bbl", how="left")
        res["litigation_cnt_2y"] = res["litigation_cnt_2y"].fillna(0).astype(np.float32)
        res["litigation_cnt_all"] = (
            res["litigation_cnt_all"].fillna(0).astype(np.float32)
        )
        res["has_litigation"] = (res["litigation_cnt_all"] > 0).astype(np.float32)
    else:
        res["litigation_cnt_2y"] = np.float32(0)
        res["litigation_cnt_all"] = np.float32(0)
        res["has_litigation"] = np.float32(0)

    # DOB Violations Aggregations
    if len(df_dob) > 0:
        prior_dob = df_dob[df_dob["date"] < cutoff]
        dob_90d = prior_dob[prior_dob["date"] >= t_90d]
        dob_180d = prior_dob[prior_dob["date"] >= t_180d]
        dob_1y = prior_dob[prior_dob["date"] >= t_1y]
        dob_3y = prior_dob[prior_dob["date"] >= t_3y]

        res = res.merge(get_counts(dob_90d, "dob_viol_cnt_90d"), on="bbl", how="left")
        res = res.merge(get_counts(dob_180d, "dob_viol_cnt_180d"), on="bbl", how="left")
        res = res.merge(get_counts(dob_1y, "dob_viol_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(dob_3y, "dob_viol_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_dob, "dob_viol_cnt_all"), on="bbl", how="left")

        res["dob_viol_cnt_90d"] = res["dob_viol_cnt_90d"].fillna(0).astype(np.float32)
        res["dob_viol_cnt_180d"] = res["dob_viol_cnt_180d"].fillna(0).astype(np.float32)
        res["dob_viol_cnt_1y"] = res["dob_viol_cnt_1y"].fillna(0).astype(np.float32)
        res["dob_viol_cnt_3y"] = res["dob_viol_cnt_3y"].fillna(0).astype(np.float32)
        res["dob_viol_cnt_all"] = res["dob_viol_cnt_all"].fillna(0).astype(np.float32)

        max_dob_date = (
            prior_dob.groupby("bbl")["date"]
            .max()
            .rename("last_dob_date")
            .reset_index()
        )
        res = res.merge(max_dob_date, on="bbl", how="left")
        res["days_since_last_dob_viol"] = (
            (cutoff - res["last_dob_date"]).dt.days.fillna(3650).astype(np.float32)
        )
        res = res.drop(columns=["last_dob_date"])

        res["dob_velocity_90d_1y"] = (
            (res["dob_viol_cnt_90d"] * 4.0) / (res["dob_viol_cnt_1y"] + 0.1)
        ).astype(np.float32)
        res["dob_per_unit_1y"] = (
            res["dob_viol_cnt_1y"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)
    else:
        for c in [
            "dob_viol_cnt_90d",
            "dob_viol_cnt_180d",
            "dob_viol_cnt_1y",
            "dob_viol_cnt_3y",
            "dob_viol_cnt_all",
            "days_since_last_dob_viol",
            "dob_velocity_90d_1y",
            "dob_per_unit_1y",
        ]:
            res[c] = np.float32(0)

    # DOHMH Rodent Inspections Aggregations
    if len(df_rodent) > 0:
        prior_rodent = df_rodent[df_rodent["date"] < cutoff]
        rodent_90d = prior_rodent[prior_rodent["date"] >= t_90d]
        rodent_180d = prior_rodent[prior_rodent["date"] >= t_180d]
        rodent_1y = prior_rodent[prior_rodent["date"] >= t_1y]
        rodent_3y = prior_rodent[prior_rodent["date"] >= t_3y]

        res = res.merge(get_counts(rodent_90d, "rodent_cnt_90d"), on="bbl", how="left")
        res = res.merge(get_counts(rodent_180d, "rodent_cnt_180d"), on="bbl", how="left")
        res = res.merge(get_counts(rodent_1y, "rodent_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(rodent_3y, "rodent_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_rodent, "rodent_cnt_all"), on="bbl", how="left")

        res["rodent_cnt_90d"] = res["rodent_cnt_90d"].fillna(0).astype(np.float32)
        res["rodent_cnt_180d"] = res["rodent_cnt_180d"].fillna(0).astype(np.float32)
        res["rodent_cnt_1y"] = res["rodent_cnt_1y"].fillna(0).astype(np.float32)
        res["rodent_cnt_3y"] = res["rodent_cnt_3y"].fillna(0).astype(np.float32)
        res["rodent_cnt_all"] = res["rodent_cnt_all"].fillna(0).astype(np.float32)

        max_rodent_date = (
            prior_rodent.groupby("bbl")["date"]
            .max()
            .rename("last_rodent_date")
            .reset_index()
        )
        res = res.merge(max_rodent_date, on="bbl", how="left")
        res["days_since_last_rodent"] = (
            (cutoff - res["last_rodent_date"]).dt.days.fillna(3650).astype(np.float32)
        )
        res = res.drop(columns=["last_rodent_date"])

        res["rodent_velocity_90d_1y"] = (
            (res["rodent_cnt_90d"] * 4.0) / (res["rodent_cnt_1y"] + 0.1)
        ).astype(np.float32)
        res["rodent_per_unit_1y"] = (
            res["rodent_cnt_1y"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)
    else:
        for c in [
            "rodent_cnt_90d",
            "rodent_cnt_180d",
            "rodent_cnt_1y",
            "rodent_cnt_3y",
            "rodent_cnt_all",
            "days_since_last_rodent",
            "rodent_velocity_90d_1y",
            "rodent_per_unit_1y",
        ]:
            res[c] = np.float32(0)

    # HPD Heat & Hot Water Charges Aggregations (hpd_hwo_charges)
    if len(df_hwo) > 0:
        prior_hwo = df_hwo[df_hwo["date"] < cutoff]
        hwo_1y = prior_hwo[prior_hwo["date"] >= t_1y]
        hwo_3y = prior_hwo[prior_hwo["date"] >= t_3y]

        res = res.merge(get_counts(hwo_1y, "hwo_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(hwo_3y, "hwo_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_hwo, "hwo_cnt_all"), on="bbl", how="left")
        res["hwo_cnt_1y"] = res["hwo_cnt_1y"].fillna(0).astype(np.float32)
        res["hwo_cnt_3y"] = res["hwo_cnt_3y"].fillna(0).astype(np.float32)
        res["hwo_cnt_all"] = res["hwo_cnt_all"].fillna(0).astype(np.float32)

        max_hwo_date = (
            prior_hwo.groupby("bbl")["date"]
            .max()
            .rename("last_hwo_date")
            .reset_index()
        )
        res = res.merge(max_hwo_date, on="bbl", how="left")
        res["days_since_last_hwo"] = (
            (cutoff - res["last_hwo_date"]).dt.days.fillna(3650).astype(np.float32)
        )
        res = res.drop(columns=["last_hwo_date"])
        res["has_hwo_charges"] = (res["hwo_cnt_all"] > 0).astype(np.float32)
    else:
        res["hwo_cnt_1y"] = np.float32(0)
        res["hwo_cnt_3y"] = np.float32(0)
        res["hwo_cnt_all"] = np.float32(0)
        res["days_since_last_hwo"] = np.float32(3650)
        res["has_hwo_charges"] = np.float32(0)

    # DOB Safety Violations Aggregations
    if len(df_dob_safety) > 0:
        prior_dob_s = df_dob_safety[df_dob_safety["date"].isna() | (df_dob_safety["date"] < cutoff)]
        dob_s_1y = prior_dob_s[prior_dob_s["date"] >= t_1y]
        dob_s_3y = prior_dob_s[prior_dob_s["date"] >= t_3y]
        res = res.merge(get_counts(dob_s_1y, "dob_safety_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(dob_s_3y, "dob_safety_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_dob_s, "dob_safety_cnt_all"), on="bbl", how="left")
        res["dob_safety_cnt_1y"] = res["dob_safety_cnt_1y"].fillna(0).astype(np.float32)
        res["dob_safety_cnt_3y"] = res["dob_safety_cnt_3y"].fillna(0).astype(np.float32)
        res["dob_safety_cnt_all"] = res["dob_safety_cnt_all"].fillna(0).astype(np.float32)
        res["has_dob_safety"] = (res["dob_safety_cnt_all"] > 0).astype(np.float32)
    else:
        res["dob_safety_cnt_1y"] = np.float32(0)
        res["dob_safety_cnt_3y"] = np.float32(0)
        res["dob_safety_cnt_all"] = np.float32(0)
        res["has_dob_safety"] = np.float32(0)

    # NYC Evictions Aggregations
    if len(df_evictions) > 0:
        prior_evict = df_evictions[df_evictions["date"].isna() | (df_evictions["date"] < cutoff)]
        evict_1y = prior_evict[prior_evict["date"] >= t_1y]
        evict_3y = prior_evict[prior_evict["date"] >= t_3y]
        res = res.merge(get_counts(evict_1y, "eviction_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(evict_3y, "eviction_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_evict, "eviction_cnt_all"), on="bbl", how="left")
        res["eviction_cnt_1y"] = res["eviction_cnt_1y"].fillna(0).astype(np.float32)
        res["eviction_cnt_3y"] = res["eviction_cnt_3y"].fillna(0).astype(np.float32)
        res["eviction_cnt_all"] = res["eviction_cnt_all"].fillna(0).astype(np.float32)
        res["has_eviction"] = (res["eviction_cnt_all"] > 0).astype(np.float32)
    else:
        res["eviction_cnt_1y"] = np.float32(0)
        res["eviction_cnt_3y"] = np.float32(0)
        res["eviction_cnt_all"] = np.float32(0)
        res["has_eviction"] = np.float32(0)

    # HPD Bedbug Reports Aggregations
    if len(df_bedbug) > 0:
        prior_bb = df_bedbug[df_bedbug["date"].isna() | (df_bedbug["date"] < cutoff)]
        bb_1y = prior_bb[prior_bb["date"] >= t_1y]
        bb_3y = prior_bb[prior_bb["date"] >= t_3y]
        res = res.merge(get_counts(bb_1y, "bedbug_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(bb_3y, "bedbug_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_bb, "bedbug_cnt_all"), on="bbl", how="left")
        res["bedbug_cnt_1y"] = res["bedbug_cnt_1y"].fillna(0).astype(np.float32)
        res["bedbug_cnt_3y"] = res["bedbug_cnt_3y"].fillna(0).astype(np.float32)
        res["bedbug_cnt_all"] = res["bedbug_cnt_all"].fillna(0).astype(np.float32)
        res["has_bedbug"] = (res["bedbug_cnt_all"] > 0).astype(np.float32)
    else:
        res["bedbug_cnt_1y"] = np.float32(0)
        res["bedbug_cnt_3y"] = np.float32(0)
        res["bedbug_cnt_all"] = np.float32(0)
        res["has_bedbug"] = np.float32(0)

    # DOF Tax Lien Sales Aggregations
    if len(df_tax_lien) > 0:
        prior_lien = df_tax_lien[df_tax_lien["date"].isna() | (df_tax_lien["date"] < cutoff)]
        res = res.merge(get_counts(prior_lien, "tax_lien_cnt_all"), on="bbl", how="left")
        res["tax_lien_cnt_all"] = res["tax_lien_cnt_all"].fillna(0).astype(np.float32)
        res["has_tax_lien"] = (res["tax_lien_cnt_all"] > 0).astype(np.float32)
    else:
        res["tax_lien_cnt_all"] = np.float32(0)
        res["has_tax_lien"] = np.float32(0)

    # Multi-year Chronic Recidivism Flags
    res["chronic_viol_c_flag"] = (
        (res["viol_c_cnt_1y"] > 0) & (res["viol_c_cnt_3y"] > res["viol_c_cnt_1y"])
    ).astype(np.float32)
    res["chronic_recidivist_flag"] = (res["viol_c_cnt_5y"] >= 3).astype(np.float32)
    res["chronic_complaint_flag"] = (
        (res["complaint_cnt_1y"] > 0) & (res["complaint_cnt_3y"] > res["complaint_cnt_1y"])
    ).astype(np.float32)

    # Enforcement & Multi-Agency Flags (Point-in-Time)
    v_prior = df_vacate[df_vacate["date"].isna() | (df_vacate["date"] < cutoff)]
    omo_prior = df_omo[df_omo["date"].isna() | (df_omo["date"] < cutoff)]
    aep_prior = df_aep[df_aep["date"].isna() | (df_aep["date"] < cutoff)]
    res["has_vacate_order"] = res["bbl"].isin(set(v_prior["bbl"])).astype(np.float32)
    res["has_emergency_charges"] = res["bbl"].isin(set(omo_prior["bbl"])).astype(np.float32)
    res["is_aep_building"] = res["bbl"].isin(set(aep_prior["bbl"])).astype(np.float32)
    res["has_dob_violation"] = (res["dob_viol_cnt_all"] > 0).astype(np.float32)
    res["has_rodent_inspection"] = (res["rodent_cnt_all"] > 0).astype(np.float32)
    res["multi_agency_acute"] = (
        (res["viol_c_cnt_90d"] > 0).astype(int)
        + (res["complaint_cnt_90d"] > 0).astype(int)
        + (res["dob_viol_cnt_1y"] > 0).astype(int)
        + (res["rodent_cnt_1y"] > 0).astype(int)
        + (res["dob_safety_cnt_1y"] > 0).astype(int)
        + (res["eviction_cnt_1y"] > 0).astype(int)
        + (res["hwo_cnt_1y"] > 0).astype(int)
    ).astype(np.float32)

    cols_to_drop = ["block", "lot", "bbl_clean", "version"]
    res = res.drop(columns=[c for c in cols_to_drop if c in res.columns])
    return res


print("Extracting features for Training cohort (Cutoff: 2019-01-01)...")
X_train_2019 = extract_cohort_features(df_train_2019, "2019-01-01")
X_train_2019["target"] = df_train_2019["target"].values

print("Extracting features for Training cohort (Cutoff: 2020-01-01)...")
X_train_2020 = extract_cohort_features(df_train_2020, "2020-01-01")
X_train_2020["target"] = df_train_2020["target"].values

print("Extracting features for Training cohort (Cutoff: 2021-01-01)...")
X_train_2021 = extract_cohort_features(df_train_2021, "2021-01-01")
X_train_2021["target"] = df_train_2021["target"].values

print("Concatenating into pooled longitudinal training panel...")
X_train = pd.concat([X_train_2019, X_train_2020, X_train_2021], ignore_index=True)
del X_train_2019, X_train_2020, X_train_2021, df_train_2019, df_train_2020, df_train_2021
gc.collect()
print(f"Pooled training panel shape: {X_train.shape}")

print("Extracting features for Validation cohort (Cutoff: 2022-01-01)...")
X_val = extract_cohort_features(df_val, "2022-01-01")
X_val["target"] = df_val["target"].values

print("Extracting features for Test cohort (Cutoff: 2023-01-01)...")
X_test = extract_cohort_features(df_test, "2023-01-01")

# Clean up raw event logs to free memory
del df_viol, c_violations, df_complaints, df_lit, df_dob, df_rodent, df_hwo, df_dob_safety, df_evictions, df_bedbug, df_tax_lien
gc.collect()

# Fit Spatial Community District Aggregations Strictly on Training Set
cd_stats = (
    X_train.groupby("cd")
    .agg(
        cd_mean_viol_c_1y=("viol_c_cnt_1y", "mean"),
        cd_mean_complaints_1y=("complaint_cnt_1y", "mean"),
        cd_target_rate=("target", "mean"),
    )
    .reset_index()
)

global_cd_c_mean = float(X_train["viol_c_cnt_1y"].mean())
global_cd_comp_mean = float(X_train["complaint_cnt_1y"].mean())
global_cd_target_rate = float(X_train["target"].mean())


def apply_cd_stats(df):
    df = df.merge(cd_stats, on="cd", how="left")
    df["cd_mean_viol_c_1y"] = (
        df["cd_mean_viol_c_1y"].fillna(global_cd_c_mean).astype(np.float32)
    )
    df["cd_mean_complaints_1y"] = (
        df["cd_mean_complaints_1y"].fillna(global_cd_comp_mean).astype(np.float32)
    )
    df["cd_target_rate"] = (
        df["cd_target_rate"].fillna(global_cd_target_rate).astype(np.float32)
    )
    df["relative_risk_cd"] = (
        df["viol_c_cnt_1y"] / (df["cd_mean_viol_c_1y"] + 0.1)
    ).astype(np.float32)
    return df


X_train = apply_cd_stats(X_train)
X_val = apply_cd_stats(X_val)
X_test = apply_cd_stats(X_test)

# Laplace-Smoothed Target Encodings (Fit strictly on Training Set)
global_mean = float(X_train["target"].mean())
smooth_m = 10.0

zip_stats = X_train.groupby("zipcode")["target"].agg(["count", "sum"]).reset_index()
zip_stats["zipcode_target_enc"] = (
    (zip_stats["sum"] + smooth_m * global_mean) / (zip_stats["count"] + smooth_m)
).astype(np.float32)
zip_enc_map = dict(zip(zip_stats["zipcode"], zip_stats["zipcode_target_enc"]))

bldg_stats = X_train.groupby("bldgclass")["target"].agg(["count", "sum"]).reset_index()
bldg_stats["bldgclass_target_enc"] = (
    (bldg_stats["sum"] + smooth_m * global_mean) / (bldg_stats["count"] + smooth_m)
).astype(np.float32)
bldg_enc_map = dict(zip(bldg_stats["bldgclass"], bldg_stats["bldgclass_target_enc"]))

for df in [X_train, X_val, X_test]:
    df["zipcode_target_enc"] = df["zipcode"].map(zip_enc_map).fillna(global_mean).astype(np.float32)
    df["bldgclass_target_enc"] = df["bldgclass"].map(bldg_enc_map).fillna(global_mean).astype(np.float32)

# Categorical Frequency and Ordinal Encodings (Fit on Training Set)
categorical_cols = [
    "borough",
    "bldgclass_major",
    "bldgclass",
    "cd",
    "council",
    "zipcode",
]
for cat in categorical_cols:
    cat_freq = X_train[cat].value_counts().to_dict()
    X_train[f"{cat}_freq"] = X_train[cat].map(cat_freq).fillna(0).astype(np.float32)
    X_val[f"{cat}_freq"] = X_val[cat].map(cat_freq).fillna(0).astype(np.float32)
    X_test[f"{cat}_freq"] = X_test[cat].map(cat_freq).fillna(0).astype(np.float32)

    categories = sorted(list(X_train[cat].astype(str).unique()))
    cat_mapping = {val: idx for idx, val in enumerate(categories)}
    X_train[cat] = X_train[cat].astype(str).map(cat_mapping).fillna(-1).astype(np.int32)
    X_val[cat] = X_val[cat].astype(str).map(cat_mapping).fillna(-1).astype(np.int32)
    X_test[cat] = X_test[cat].astype(str).map(cat_mapping).fillna(-1).astype(np.int32)

exclude_cols = {"bbl", "target"}
feature_names = [c for c in X_train.columns if c not in exclude_cols]
with open(os.path.join(WORKING_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

print(f"Feature engineering completed: {len(feature_names)} features.")

# =========================================================================
# 3. Model Architecture & Loss Definitions
# =========================================================================
# LightGBM Classifier (Leaf-wise GBDT - Unweighted, parity capacity)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 25,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# XGBoost Classifier (Depth-wise GBDT - Unweighted)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_estimators": 1000,
    "early_stopping_rounds": 40,
    "random_state": 42,
    "n_jobs": -1,
}
xgb_model = xgb.XGBClassifier(**xgb_params)

# CatBoost Classifier (Symmetric/Oblivious Trees)
cb_params = {
    "iterations": 1000,
    "learning_rate": 0.04,
    "depth": 6,
    "eval_metric": "Logloss",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 40,
}
cb_model = CatBoostClassifier(**cb_params)

# =========================================================================
# 4. Training, Ensembling & Validation
# =========================================================================
X_tr_mat = np.nan_to_num(
    X_train[feature_names].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_tr = X_train["target"].values.astype(np.float32)

X_val_mat = np.nan_to_num(
    X_val[feature_names].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_val = X_val["target"].values.astype(np.float32)

X_te_mat = np.nan_to_num(
    X_test[feature_names].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

print(
    f"Training LightGBM model on {X_tr_mat.shape[0]} samples with"
    f" {X_tr_mat.shape[1]} features..."
)
callbacks_lgb = [
    lgb.early_stopping(stopping_rounds=40, verbose=False),
    lgb.log_evaluation(period=0),
]
lgb_model.fit(
    X_tr_mat,
    y_tr,
    eval_set=[(X_val_mat, y_val)],
    callbacks=callbacks_lgb,
)
lgb_model.booster_.save_model(os.path.join(WORKING_DIR, "lgbm_model.txt"))
val_preds_lgb = lgb_model.predict_proba(X_val_mat)[:, 1]
test_preds_lgb = lgb_model.predict_proba(X_te_mat)[:, 1]
gc.collect()

print("Training XGBoost model...")
try:
    xgb_model.fit(
        X_tr_mat,
        y_tr,
        eval_set=[(X_val_mat, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_params_fallback = {
        k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"
    }
    xgb_model = xgb.XGBClassifier(**xgb_params_fallback)
    xgb_model.fit(
        X_tr_mat,
        y_tr,
        eval_set=[(X_val_mat, y_val)],
        early_stopping_rounds=40,
        verbose=False,
    )
xgb_model.save_model(os.path.join(WORKING_DIR, "xgb_model.json"))
val_preds_xgb = xgb_model.predict_proba(X_val_mat)[:, 1]
test_preds_xgb = xgb_model.predict_proba(X_te_mat)[:, 1]
gc.collect()

print("Training CatBoost model...")
cb_model.fit(
    X_tr_mat,
    y_tr,
    eval_set=(X_val_mat, y_val),
    verbose=False,
)
cb_model.save_model(os.path.join(WORKING_DIR, "cb_model.cbm"))
val_preds_cb = cb_model.predict_proba(X_val_mat)[:, 1]
test_preds_cb = cb_model.predict_proba(X_te_mat)[:, 1]

gc.collect()

# Rank Normalization and Metric-Optimized Blending
rank_val_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)
rank_val_xgb = rankdata(val_preds_xgb) / len(val_preds_xgb)
rank_val_cb = rankdata(val_preds_cb) / len(val_preds_cb)

rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
rank_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
rank_test_cb = rankdata(test_preds_cb) / len(test_preds_cb)

val_ap_lgb = float(average_precision_score(y_val, rank_val_lgb))
val_ap_xgb = float(average_precision_score(y_val, rank_val_xgb))
val_ap_cb = float(average_precision_score(y_val, rank_val_cb))
print(
    f"Individual Model Val AP - LightGBM: {val_ap_lgb:.4f} | XGBoost:"
    f" {val_ap_xgb:.4f} | CatBoost: {val_ap_cb:.4f}"
)

# 3-Way Grid Search over Simplex Weights to directly maximize Average Precision
best_score = -1.0
best_weights = (0.34, 0.33, 0.33)
grid_steps = np.linspace(0.0, 1.0, 21)
for w1 in grid_steps:
    for w2 in grid_steps:
        if w1 + w2 > 1.0:
            continue
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue
        blended_val = w1 * rank_val_lgb + w2 * rank_val_xgb + w3 * rank_val_cb
        ap_score = float(average_precision_score(y_val, blended_val))
        if ap_score > best_score:
            best_score = ap_score
            best_weights = (float(w1), float(w2), float(w3))

w_lgb, w_xgb, w_cb = best_weights
print(
    f"Optimal Tri-Model Ensemble Weights - LGBM: {w_lgb:.2f}, XGB: {w_xgb:.2f}, CB: {w_cb:.2f} | Blended AP: {best_score:.4f}"
)

final_val_preds = (
    w_lgb * rank_val_lgb + w_xgb * rank_val_xgb + w_cb * rank_val_cb
)
final_test_preds = (
    w_lgb * rank_test_lgb + w_xgb * rank_test_xgb + w_cb * rank_test_cb
)

official_val_ap = float(average_precision_score(y_val, final_val_preds))
val_roc_auc = float(roc_auc_score(y_val, final_val_preds))

print(
    f"Validation Summary | Official AP: {official_val_ap:.4f} | ROC-AUC:"
    f" {val_roc_auc:.4f}"
)

for k_pct in [1, 5, 10]:
    k_count = int(len(final_val_preds) * k_pct / 100.0)
    top_indices = np.argsort(final_val_preds)[-k_count:]
    top_positives = y_val[top_indices].sum()
    prec_k = top_positives / k_count
    rec_k = top_positives / y_val.sum()
    print(
        f"Top {k_pct:02d}% Inspection Tier - Precision: {prec_k:.4f}, Recall:"
        f" {rec_k:.4f}"
    )

# =========================================================================
# 5. Submission Generation & Audit
# =========================================================================
submission_df = pd.DataFrame(
    {
        "bbl": df_test["bbl"].astype(str).str.zfill(10),
        "score": final_test_preds.astype(float),
    }
)

submission_file = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_file, index=False)

assert len(submission_df) == len(
    X_test
), f"Row count mismatch: expected {len(X_test)}, got {len(submission_df)}"
assert (
    not submission_df["score"].isna().any()
), "Encountered NaN values in test prediction scores!"
assert (
    submission_df["bbl"].str.len() == 10
).all(), "Detected malformed BBL keys in submission!"

score = official_val_ap
print(f"Final Validation Score: {score}")
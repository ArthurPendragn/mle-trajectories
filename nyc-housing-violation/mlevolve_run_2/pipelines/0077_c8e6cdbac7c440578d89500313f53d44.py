import os
import gc
import sys
import glob
import math
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as pds
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
np.random.seed(42)

# ---------------------------------------------------------
# 1. GCS Storage & Path Configuration
# ---------------------------------------------------------
GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
TOKEN_PATHS = [
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json",
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
]

storage_options = None
for p in TOKEN_PATHS:
    if p and os.path.exists(p):
        storage_options = {"token": p}
        break
if storage_options is None:
    cands = (
        glob.glob("/home/*/*/*key*.json")
        + glob.glob("/home/*/*/*lake*.json")
        + glob.glob("./*key*.json")
    )
    if cands:
        storage_options = {"token": cands[0]}
    else:
        storage_options = {}

print("Resolved GCS storage credentials.")


# ---------------------------------------------------------
# 2. Schema-Aware Safe Data Ingestion
# ---------------------------------------------------------
def safe_series_numeric(df, col_name, default=0.0):
    """Safely extract and cast a column to float32 series without type errors."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default).astype(np.float32)
    return pd.Series(default, index=df.index, dtype=np.float32)


def clean_bbl_series(bbl_s=None, boro_s=None, blk_s=None, lot_s=None):
    """Vectorized cleaning and zero-padded standardization of 10-digit BBL."""
    ref_idx = None
    for item in [bbl_s, boro_s, blk_s, lot_s]:
        if isinstance(item, pd.Series):
            ref_idx = item.index
            break

    if bbl_s is not None and isinstance(bbl_s, pd.Series):
        s = bbl_s.astype(str).str.strip().str.split(".").str[0].str.zfill(10)
        valid = (s.str.len() == 10) & (s.str.isnumeric()) & (~s.str.startswith("0"))
    else:
        if ref_idx is not None:
            valid = pd.Series(False, index=ref_idx)
            s = pd.Series("", index=ref_idx)
        else:
            return pd.Series([], dtype=str)

    if (~valid).any() and boro_s is not None and blk_s is not None and lot_s is not None:
        b_clean = boro_s.astype(str).str.strip().str.split(".").str[0].str.zfill(1)
        blk_clean = blk_s.astype(str).str.strip().str.split(".").str[0].str.zfill(5)
        lot_clean = lot_s.astype(str).str.strip().str.split(".").str[0].str.zfill(4)
        synth = b_clean + blk_clean + lot_clean
        s = pd.Series(np.where(valid, s, synth), index=ref_idx).astype(str)
    return pd.Series(s, index=ref_idx).astype(str)


def get_parquet_cols(full_path):
    try:
        ds_fs = pds.dataset(full_path, storage_options=storage_options)
        return ds_fs.schema.names
    except Exception:
        return []


def safe_read_parquet(table_rel_path, target_cols, optional_cols=None, col_patterns=None):
    full_path = f"{GCS_BASE}/{table_rel_path}"
    avail_cols = get_parquet_cols(full_path)
    if not avail_cols:
        try:
            sample_df = pd.read_parquet(full_path, storage_options=storage_options)
            avail_cols = list(sample_df.columns)
            del sample_df
        except Exception:
            return pd.DataFrame()

    avail_lower = {c.lower(): c for c in avail_cols}
    selected = []
    for c in target_cols:
        if c.lower() in avail_lower:
            selected.append(avail_lower[c.lower()])

    if optional_cols:
        for c in optional_cols:
            if c.lower() in avail_lower and avail_lower[c.lower()] not in selected:
                selected.append(avail_lower[c.lower()])

    if col_patterns:
        for pat in col_patterns:
            pat_lower = pat.lower()
            for c_low, c_orig in avail_lower.items():
                if pat_lower in c_low and c_orig not in selected:
                    selected.append(c_orig)

    if not selected:
        return pd.DataFrame()

    try:
        df = pd.read_parquet(
            full_path, columns=selected, storage_options=storage_options
        )
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"Warning: could not read {table_rel_path}: {e}")
        return pd.DataFrame()


print("Loading test entities...")
df_test_entities = pd.read_parquet(
    f"{GCS_BASE}/test_entities.parquet", storage_options=storage_options
)
test_bbls = clean_bbl_series(df_test_entities["bbl"]).values
print(f"Test lots: {len(test_bbls)}")

# ---------------------------------------------------------
# 3. Loading Tables from NYC Lake
# ---------------------------------------------------------
t0 = time.time()
print("Ingesting PLUTO releases...")
pluto_cols = [
    "bbl",
    "version",
    "unitsres",
    "yearbuilt",
    "bldgarea",
    "numfloors",
    "borocode",
    "block",
    "lot",
    "zipcode",
    "assesstot",
    "bldgclass",
    "lotarea",
    "yearalter1",
]
df_pluto = safe_read_parquet(
    "lake/full/pluto", pluto_cols, ["borough", "boroid", "latitude", "longitude"]
)
df_pluto["clean_bbl"] = clean_bbl_series(
    df_pluto.get("bbl"),
    df_pluto.get("borocode", df_pluto.get("borough", df_pluto.get("boroid"))),
    df_pluto.get("block"),
    df_pluto.get("lot"),
)
df_pluto["unitsres"] = safe_series_numeric(df_pluto, "unitsres", 0.0)
df_pluto["yearbuilt"] = safe_series_numeric(df_pluto, "yearbuilt", 0.0)
df_pluto["bldgarea"] = safe_series_numeric(df_pluto, "bldgarea", 0.0)
df_pluto["numfloors"] = safe_series_numeric(df_pluto, "numfloors", 0.0)
df_pluto["assesstot"] = safe_series_numeric(df_pluto, "assesstot", 0.0)
df_pluto["lotarea"] = safe_series_numeric(df_pluto, "lotarea", 0.0)
df_pluto["yearalter1"] = safe_series_numeric(df_pluto, "yearalter1", 0.0)
if "bldgclass" in df_pluto.columns:
    df_pluto["bldgclass"] = df_pluto["bldgclass"].astype(str).str.strip().str.upper()
else:
    df_pluto["bldgclass"] = pd.Series("", index=df_pluto.index, dtype=str)
df_pluto["version"] = df_pluto.get("version", "").astype(str).str.lower()
print(f"PLUTO loaded: {len(df_pluto):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
print("Ingesting HPD Violations...")
viol_cols = [
    "bbl",
    "boroid",
    "block",
    "lot",
    "class",
    "inspectiondate",
    "ordernumber",
    "violationstatus",
    "currentstatusdate",
]
df_viol = safe_read_parquet(
    "lake/full/hpd_violations", viol_cols, ["boro", "novdescription"]
)
df_viol["clean_bbl"] = clean_bbl_series(
    df_viol.get("bbl"),
    df_viol.get("boroid", df_viol.get("boro")),
    df_viol.get("block"),
    df_viol.get("lot"),
)
df_viol["insp_dt"] = pd.to_datetime(df_viol.get("inspectiondate"), errors="coerce")
df_viol["class"] = df_viol.get("class", "").astype(str).str.upper().str.strip()
ord_s = df_viol.get("ordernumber", "").astype(str)
nov_upper = df_viol.get("novdescription", "").astype(str).str.upper()
df_viol["is_heat"] = ord_s.str.contains(
    "501|502|503|504|505", regex=True, na=False
) | nov_upper.str.contains("HEAT|HOT WATER|BOILER", regex=True, na=False)
df_viol["is_lead"] = ord_s.str.contains(
    "616|617|618|619|620", regex=True, na=False
) | nov_upper.str.contains("LEAD", regex=True, na=False)
df_viol["is_mold_leak"] = nov_upper.str.contains(
    "MOLD|LEAK", regex=True, na=False
)
df_viol["is_alarm_safety"] = ord_s.str.contains(
    "506|507|508|509|510", regex=True, na=False
) | nov_upper.str.contains("SMOKE DETECTOR|CARBON MONOXIDE|ALARM", regex=True, na=False)
del ord_s, nov_upper
if "novdescription" in df_viol.columns:
    df_viol.drop(columns=["novdescription"], inplace=True)
df_viol["status_dt"] = pd.to_datetime(df_viol.get("currentstatusdate"), errors="coerce")
df_viol["is_open"] = (
    df_viol.get("violationstatus", "")
    .astype(str)
    .str.lower()
    .str.contains("open", na=False)
)
df_viol = df_viol.dropna(subset=["insp_dt"])
print(f"HPD Violations loaded: {len(df_viol):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
print("Ingesting HPD Complaints...")
comp_cols = ["bbl", "borough", "block", "lot", "receiveddate"]
df_comp = safe_read_parquet(
    "lake/full/hpd_complaints",
    comp_cols,
    ["statusdate", "status_date", "received_date", "majorcategory"],
)
date_col = next(
    (
        c
        for c in ["receiveddate", "received_date", "statusdate", "status_date"]
        if c in df_comp.columns
    ),
    None,
)
if date_col:
    df_comp["dt"] = pd.to_datetime(df_comp[date_col], errors="coerce")
else:
    df_comp["dt"] = pd.NaT
df_comp["clean_bbl"] = clean_bbl_series(
    df_comp.get("bbl"), df_comp.get("borough"), df_comp.get("block"), df_comp.get("lot")
)
df_comp = df_comp.dropna(subset=["dt"])
# Filter complaints from 2017 onwards for efficiency
df_comp = df_comp[df_comp["dt"] >= "2017-01-01"]
if "majorcategory" in df_comp.columns:
    df_comp["is_heat_comp"] = (
        df_comp["majorcategory"]
        .astype(str)
        .str.contains("HEAT|HOT WATER", case=False, na=False)
    )
else:
    df_comp["is_heat_comp"] = False
print(f"HPD Complaints loaded: {len(df_comp):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
print("Ingesting Auxiliary Multi-Agency Enforcement Tables...")
df_ecb = safe_read_parquet(
    "lake/full/dob_ecb_violations",
    ["boro", "block", "lot", "severity", "issue_date", "penalty_balance_due"],
    ["bbl"],
)
if len(df_ecb) > 0:
    df_ecb["clean_bbl"] = clean_bbl_series(
        df_ecb.get("bbl"), df_ecb.get("boro"), df_ecb.get("block"), df_ecb.get("lot")
    )
    df_ecb["dt"] = pd.to_datetime(df_ecb["issue_date"], errors="coerce") if "issue_date" in df_ecb.columns else pd.NaT
    df_ecb["is_class1"] = (
        df_ecb["severity"]
        .astype(str)
        .str.contains("CLASS 1|HAZARDOUS", case=False, na=False)
        if "severity" in df_ecb.columns
        else False
    )
    df_ecb["penalty"] = safe_series_numeric(df_ecb, "penalty_balance_due", 0.0)

df_safety = safe_read_parquet(
    "lake/full/dob_safety_violations",
    ["bbl", "borough", "block", "lot", "issue_date"],
    ["violation_category", "device_type"],
)
if len(df_safety) > 0:
    df_safety["clean_bbl"] = clean_bbl_series(
        df_safety.get("bbl"),
        df_safety.get("borough"),
        df_safety.get("block"),
        df_safety.get("lot"),
    )
    df_safety["dt"] = pd.to_datetime(df_safety["issue_date"], errors="coerce") if "issue_date" in df_safety.columns else pd.NaT
    cat_str = (
        (df_safety["violation_category"].astype(str) if "violation_category" in df_safety.columns else "")
        + " "
        + (df_safety["device_type"].astype(str) if "device_type" in df_safety.columns else "")
    )
    df_safety["is_boiler"] = cat_str.str.contains("BOILER", case=False, na=False)
    df_safety["is_elevator"] = cat_str.str.contains("ELEVATOR", case=False, na=False)

df_hwo = safe_read_parquet(
    "lake/full/hpd_hwo_charges",
    ["bbl", "boroid", "boro", "block", "lot"],
    col_patterns=["date", "amount", "fee", "charge", "cost"],
)
if len(df_hwo) > 0:
    df_hwo["clean_bbl"] = clean_bbl_series(
        df_hwo.get("bbl"),
        df_hwo.get("boroid", df_hwo.get("boro")),
        df_hwo.get("block"),
        df_hwo.get("lot"),
    )
    hwo_dt_col = next(
        (c for c in df_hwo.columns if "date" in c or c.endswith("dt")), None
    )
    df_hwo["dt"] = (
        pd.to_datetime(df_hwo[hwo_dt_col], errors="coerce") if hwo_dt_col else pd.NaT
    )
    hwo_amt_col = next(
        (c for c in df_hwo.columns if any(k in c for k in ["amount", "fee", "charge", "cost"])), None
    )
    if hwo_amt_col:
        df_hwo["amount"] = pd.to_numeric(df_hwo[hwo_amt_col], errors="coerce").fillna(0.0).astype(np.float32)
    else:
        df_hwo["amount"] = pd.Series(1.0, index=df_hwo.index, dtype=np.float32)

df_omo = safe_read_parquet(
    "lake/full/hpd_omo_charges",
    ["bbl", "boroid", "boro", "block", "lot"],
    col_patterns=["date", "amount", "fee", "charge", "cost"],
)
if len(df_omo) > 0:
    df_omo["clean_bbl"] = clean_bbl_series(
        df_omo.get("bbl"),
        df_omo.get("boroid", df_omo.get("boro")),
        df_omo.get("block"),
        df_omo.get("lot"),
    )
    omo_dt_col = next(
        (c for c in df_omo.columns if "date" in c or c.endswith("dt")), None
    )
    df_omo["dt"] = (
        pd.to_datetime(df_omo[omo_dt_col], errors="coerce") if omo_dt_col else pd.NaT
    )
    omo_amt_col = next(
        (c for c in df_omo.columns if any(k in c for k in ["amount", "fee", "charge", "cost"])), None
    )
    if omo_amt_col:
        df_omo["amount"] = pd.to_numeric(df_omo[omo_amt_col], errors="coerce").fillna(0.0).astype(np.float32)
    else:
        df_omo["amount"] = pd.Series(1.0, index=df_omo.index, dtype=np.float32)

df_vacate = safe_read_parquet(
    "lake/full/hpd_vacate_orders",
    ["bbl", "boroid", "boro", "block", "lot"],
    col_patterns=["date"],
)
if len(df_vacate) > 0:
    df_vacate["clean_bbl"] = clean_bbl_series(
        df_vacate.get("bbl"),
        df_vacate.get("boroid", df_vacate.get("boro")),
        df_vacate.get("block"),
        df_vacate.get("lot"),
    )
    vac_dt_col = next(
        (c for c in df_vacate.columns if "date" in c or c.endswith("dt")), None
    )
    df_vacate["dt"] = (
        pd.to_datetime(df_vacate[vac_dt_col], errors="coerce") if vac_dt_col else pd.NaT
    )

df_lit = safe_read_parquet(
    "lake/full/hpd_litigations",
    ["bbl", "boroid", "block", "lot"],
    ["caseopendate", "casetype"],
)
if len(df_lit) > 0:
    df_lit["clean_bbl"] = clean_bbl_series(
        df_lit.get("bbl"), df_lit.get("boroid"), df_lit.get("block"), df_lit.get("lot")
    )
    df_lit["dt"] = pd.to_datetime(df_lit["caseopendate"], errors="coerce") if "caseopendate" in df_lit.columns else pd.NaT

df_evict = safe_read_parquet(
    "lake/full/evictions", ["bbl", "executed_date"], ["borough"]
)
if len(df_evict) > 0:
    df_evict["clean_bbl"] = clean_bbl_series(df_evict.get("bbl"))
    df_evict["dt"] = pd.to_datetime(df_evict["executed_date"], errors="coerce") if "executed_date" in df_evict.columns else pd.NaT

df_rodent = safe_read_parquet(
    "lake/full/dohmh_rodent_inspections", ["bbl", "inspection_date", "result"]
)
if len(df_rodent) > 0:
    df_rodent["clean_bbl"] = clean_bbl_series(df_rodent.get("bbl"))
    df_rodent["dt"] = pd.to_datetime(df_rodent["inspection_date"], errors="coerce") if "inspection_date" in df_rodent.columns else pd.NaT
    df_rodent["is_fail"] = (
        df_rodent["result"]
        .astype(str)
        .str.contains("ACTIVE RAT SIGNS|FAIL", case=False, na=False)
        if "result" in df_rodent.columns
        else False
    )

df_dob_viol = safe_read_parquet(
    "lake/full/dob_violations",
    ["boro", "block", "lot", "issue_date"],
    ["bbl", "violation_date", "issued_date"],
    col_patterns=["date"],
)
if len(df_dob_viol) > 0:
    df_dob_viol["clean_bbl"] = clean_bbl_series(
        df_dob_viol.get("bbl"),
        df_dob_viol.get("boro"),
        df_dob_viol.get("block"),
        df_dob_viol.get("lot"),
    )
    dob_dt_col = next(
        (c for c in df_dob_viol.columns if "issue" in c or "date" in c), None
    )
    df_dob_viol["dt"] = (
        pd.to_datetime(df_dob_viol[dob_dt_col], errors="coerce")
        if dob_dt_col
        else pd.NaT
    )

df_bedbug = safe_read_parquet(
    "lake/full/hpd_bedbug_reports",
    ["bbl", "borough", "block", "lot", "filing_date"],
    ["infested_dwelling_unit_count", "total_dwelling_units"],
    col_patterns=["date", "infested"],
)
if len(df_bedbug) > 0:
    df_bedbug["clean_bbl"] = clean_bbl_series(
        df_bedbug.get("bbl"),
        df_bedbug.get("borough"),
        df_bedbug.get("block"),
        df_bedbug.get("lot"),
    )
    bb_dt_col = next(
        (c for c in df_bedbug.columns if "filing" in c or "date" in c), None
    )
    df_bedbug["dt"] = (
        pd.to_datetime(df_bedbug[bb_dt_col], errors="coerce")
        if bb_dt_col
        else pd.NaT
    )
    bb_inf_col = next(
        (c for c in df_bedbug.columns if "infested" in c), None
    )
    if bb_inf_col:
        df_bedbug["infested_units"] = safe_series_numeric(df_bedbug, bb_inf_col, 0.0)
    else:
        df_bedbug["infested_units"] = pd.Series(0.0, index=df_bedbug.index, dtype=np.float32)

df_aep = safe_read_parquet("lake/full/hpd_aep_buildings", ["bbl"])
aep_bbl_set = (
    set(clean_bbl_series(df_aep.get("bbl")).unique()) if len(df_aep) > 0 else set()
)

df_conh = safe_read_parquet("lake/full/hpd_conh_buildings", ["bbl"])
conh_bbl_set = (
    set(clean_bbl_series(df_conh.get("bbl")).unique()) if len(df_conh) > 0 else set()
)

df_spec = safe_read_parquet("lake/full/speculation_watch_list", ["bbl"])
spec_bbl_set = (
    set(clean_bbl_series(df_spec.get("bbl")).unique()) if len(df_spec) > 0 else set()
)

df_lien = safe_read_parquet(
    "lake/full/dof_tax_lien_sales",
    ["bbl", "borough", "boro", "boroid", "block", "lot"],
    col_patterns=["date", "sale", "year", "notice", "cycle"],
)
if len(df_lien) > 0:
    df_lien["clean_bbl"] = clean_bbl_series(
        df_lien.get("bbl"),
        df_lien.get("boroid", df_lien.get("boro", df_lien.get("borough"))),
        df_lien.get("block"),
        df_lien.get("lot"),
    )
    lien_dt_col = next(
        (c for c in df_lien.columns if "date" in c or c.endswith("dt")), None
    )
    if lien_dt_col:
        df_lien["dt"] = pd.to_datetime(df_lien[lien_dt_col], errors="coerce")
    else:
        lien_yr_col = next((c for c in df_lien.columns if "year" in c), None)
        if lien_yr_col:
            df_lien["dt"] = pd.to_datetime(
                df_lien[lien_yr_col].astype(str) + "-01-01", errors="coerce"
            )
        else:
            df_lien["dt"] = pd.NaT

print("All auxiliary enforcement datasets loaded.")
gc.collect()


# ---------------------------------------------------------
# 4. Feature Extraction Engine
# ---------------------------------------------------------
def extract_cohort_features(bbl_list, pluto_sub, cutoff_dt):
    """Point-in-time domain feature extraction strictly prior to cutoff_dt."""
    cutoff_ts = pd.Timestamp(cutoff_dt)
    n_lots = len(bbl_list)
    res_df = pd.DataFrame({"bbl": bbl_list})

    # Join PLUTO features
    pluto_sub_dedup = pluto_sub.drop_duplicates(subset=["clean_bbl"])
    pluto_merge_cols = [
        "clean_bbl",
        "unitsres",
        "yearbuilt",
        "bldgarea",
        "numfloors",
        "block",
        "zipcode",
        "assesstot",
        "bldgclass",
        "lotarea",
        "yearalter1",
    ]
    pluto_merge_cols = [c for c in pluto_merge_cols if c in pluto_sub_dedup.columns]
    merged = res_df.merge(
        pluto_sub_dedup[pluto_merge_cols],
        left_on="bbl",
        right_on="clean_bbl",
        how="left",
    )
    units = merged["unitsres"].fillna(1).clip(lower=1).values
    yb = merged["yearbuilt"].fillna(1950).values
    yb = np.where((yb < 1800) | (yb > cutoff_ts.year), 1950, yb)
    age = np.maximum(0, cutoff_ts.year - yb)
    bldgarea = merged["bldgarea"].fillna(0).clip(lower=0).values

    assesstot = safe_series_numeric(merged, "assesstot", 0.0).values
    lotarea = safe_series_numeric(merged, "lotarea", 0.0).values
    yearalter1 = safe_series_numeric(merged, "yearalter1", 0.0).values
    bldgclass_s = (
        merged["bldgclass"].fillna("").astype(str).str.strip().str.upper()
        if "bldgclass" in merged.columns
        else pd.Series("", index=merged.index)
    )
    is_walkup = bldgclass_s.str.startswith("C").values

    res_df["unitsres"] = units.astype(np.float32)
    res_df["log_unitsres"] = np.log1p(units).astype(np.float32)
    res_df["age"] = age.astype(np.float32)
    res_df["log_bldgarea"] = np.log1p(bldgarea).astype(np.float32)
    res_df["area_per_unit"] = (bldgarea / (units + 1e-4)).astype(np.float32)
    res_df["numfloors"] = merged["numfloors"].fillna(3).astype(np.float32)
    res_df["age_x_log_units"] = (np.log1p(age) * np.log1p(units)).astype(np.float32)
    res_df["block"] = (
        pd.to_numeric(merged["block"], errors="coerce").fillna(0).astype(np.int32)
    )
    res_df["zipcode"] = (
        pd.to_numeric(merged["zipcode"], errors="coerce").fillna(0).astype(np.int32)
    )
    res_df["boro"] = (
        pd.to_numeric(res_df["bbl"].str[:1], errors="coerce").fillna(1).astype(np.int32)
    )

    # PLUTO economic & alteration valuation features
    res_df["assesstot_per_unit"] = (assesstot / (units + 1e-4)).astype(np.float32)
    res_df["assesstot_per_sqft"] = (assesstot / (bldgarea + 1e-4)).astype(np.float32)
    res_df["log_assesstot"] = np.log1p(np.maximum(0, assesstot)).astype(np.float32)
    res_df["log_lotarea"] = np.log1p(np.maximum(0, lotarea)).astype(np.float32)
    res_df["lotarea_per_unit"] = (lotarea / (units + 1e-4)).astype(np.float32)

    # Pre-war unrenovated flags & tenement indicator
    is_prewar = (yb < 1940) & (yb > 1800)
    is_unrenovated = yearalter1 == 0
    res_df["is_prewar_unrenovated"] = (is_prewar & is_unrenovated).astype(np.float32)
    res_df["is_unrenovated_prewar_walkup"] = (
        is_prewar & is_unrenovated & is_walkup
    ).astype(np.float32)
    res_df["yearalter1"] = yearalter1.astype(np.float32)

    # 1. HPD Violations strictly prior to cutoff
    v_sub = df_viol[df_viol["insp_dt"] < cutoff_ts]
    c_sub = v_sub[v_sub["class"] == "C"]

    dt_30d = cutoff_ts - pd.Timedelta(days=30)
    dt_90d = cutoff_ts - pd.Timedelta(days=90)
    dt_180d = cutoff_ts - pd.Timedelta(days=180)
    dt_1y = cutoff_ts - pd.Timedelta(days=365)
    dt_2y = cutoff_ts - pd.Timedelta(days=730)
    dt_3y = cutoff_ts - pd.Timedelta(days=1095)
    dt_5y = cutoff_ts - pd.Timedelta(days=1825)

    c_30 = c_sub[c_sub["insp_dt"] >= dt_30d].groupby("clean_bbl").size()
    c_90 = c_sub[c_sub["insp_dt"] >= dt_90d].groupby("clean_bbl").size()
    c_180 = c_sub[c_sub["insp_dt"] >= dt_180d].groupby("clean_bbl").size()
    c_1y = c_sub[c_sub["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    c_2y = c_sub[c_sub["insp_dt"] >= dt_2y].groupby("clean_bbl").size()
    c_3y = c_sub[c_sub["insp_dt"] >= dt_3y].groupby("clean_bbl").size()
    c_5y = c_sub[c_sub["insp_dt"] >= dt_5y].groupby("clean_bbl").size()
    c_life = c_sub.groupby("clean_bbl").size()

    # Recency & Persistence
    c_last = c_sub.groupby("clean_bbl")["insp_dt"].max()
    c_sub_yr = c_sub.copy()
    c_sub_yr["yr"] = c_sub_yr["insp_dt"].dt.year
    c_years_active = c_sub_yr.groupby("clean_bbl")["yr"].nunique()

    # Granular Class C Hazard Phenotypes
    # Heat-specific Class C
    c_heat = c_sub[c_sub["is_heat"]]
    c_heat_90 = c_heat[c_heat["insp_dt"] >= dt_90d].groupby("clean_bbl").size()
    c_heat_1y = c_heat[c_heat["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    c_heat_3y = c_heat[c_heat["insp_dt"] >= dt_3y].groupby("clean_bbl").size()

    # Lead-based paint (orders 616-620)
    c_lead = c_sub[c_sub["is_lead"]]
    c_lead_1y = c_lead[c_lead["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    c_lead_3y = c_lead[c_lead["insp_dt"] >= dt_3y].groupby("clean_bbl").size()
    c_lead_life = c_lead.groupby("clean_bbl").size()

    # Mold and water leaks
    c_mold = c_sub[c_sub["is_mold_leak"]]
    c_mold_1y = c_mold[c_mold["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    c_mold_3y = c_mold[c_mold["insp_dt"] >= dt_3y].groupby("clean_bbl").size()
    c_mold_life = c_mold.groupby("clean_bbl").size()

    # Life-safety alarms (smoke & carbon monoxide)
    c_alarm = c_sub[c_sub["is_alarm_safety"]]
    c_alarm_1y = c_alarm[c_alarm["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    c_alarm_3y = c_alarm[c_alarm["insp_dt"] >= dt_3y].groupby("clean_bbl").size()

    # Other classes & open backlog
    b_1y = (
        v_sub[(v_sub["class"] == "B") & (v_sub["insp_dt"] >= dt_1y)]
        .groupby("clean_bbl")
        .size()
    )
    b_3y = (
        v_sub[(v_sub["class"] == "B") & (v_sub["insp_dt"] >= dt_3y)]
        .groupby("clean_bbl")
        .size()
    )
    a_1y = (
        v_sub[(v_sub["class"] == "A") & (v_sub["insp_dt"] >= dt_1y)]
        .groupby("clean_bbl")
        .size()
    )
    all_last = v_sub.groupby("clean_bbl")["insp_dt"].max()

    open_mask = (
        (v_sub["is_open"])
        | (v_sub["status_dt"].isna())
        | (v_sub["status_dt"] >= cutoff_ts)
    )
    open_c_records = v_sub[open_mask & (v_sub["class"] == "C")].copy()
    open_c = open_c_records.groupby("clean_bbl").size()
    open_all = v_sub[open_mask].groupby("clean_bbl").size()

    # Open Class C dwell time statistics (mean and max unresolved days prior to cutoff)
    if len(open_c_records) > 0:
        open_c_records["dwell_days"] = (
            cutoff_ts - open_c_records["insp_dt"]
        ).dt.days.clip(lower=0)
        open_c_dwell_mean = open_c_records.groupby("clean_bbl")["dwell_days"].mean()
        open_c_dwell_max = open_c_records.groupby("clean_bbl")["dwell_days"].max()
        chronic_open_c = open_c_records[open_c_records["dwell_days"] > 90].groupby(
            "clean_bbl"
        ).size()
    else:
        open_c_dwell_mean = pd.Series(dtype=np.float32)
        open_c_dwell_max = pd.Series(dtype=np.float32)
        chronic_open_c = pd.Series(dtype=np.float32)

    # Distinct HPD inspection encounter counts over trailing 1y and 3y
    v_1y_sub = v_sub[v_sub["insp_dt"] >= dt_1y]
    insp_enc_1y = (
        v_1y_sub[["clean_bbl", "insp_dt"]]
        .drop_duplicates()
        .groupby("clean_bbl")
        .size()
    )
    v_3y_sub = v_sub[v_sub["insp_dt"] >= dt_3y]
    insp_enc_3y = (
        v_3y_sub[["clean_bbl", "insp_dt"]]
        .drop_duplicates()
        .groupby("clean_bbl")
        .size()
    )

    # Map onto res_df
    bbl_series = res_df["bbl"]
    res_df["viol_c_30d"] = bbl_series.map(c_30).fillna(0).astype(np.float32)
    res_df["viol_c_90d"] = bbl_series.map(c_90).fillna(0).astype(np.float32)
    res_df["viol_c_180d"] = bbl_series.map(c_180).fillna(0).astype(np.float32)
    res_df["viol_c_1y"] = bbl_series.map(c_1y).fillna(0).astype(np.float32)
    res_df["viol_c_2y"] = bbl_series.map(c_2y).fillna(0).astype(np.float32)
    res_df["viol_c_3y"] = bbl_series.map(c_3y).fillna(0).astype(np.float32)
    res_df["viol_c_5y"] = bbl_series.map(c_5y).fillna(0).astype(np.float32)
    res_df["viol_c_life"] = bbl_series.map(c_life).fillna(0).astype(np.float32)

    last_dt_c = bbl_series.map(c_last)
    res_df["days_since_viol_c"] = (
        (cutoff_ts - last_dt_c).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
    )
    last_dt_all = bbl_series.map(all_last)
    res_df["days_since_viol_any"] = (
        (cutoff_ts - last_dt_all).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
    )
    res_df["c_years_active"] = (
        bbl_series.map(c_years_active).fillna(0).astype(np.float32)
    )

    res_df["viol_c_heat_90d"] = bbl_series.map(c_heat_90).fillna(0).astype(np.float32)
    res_df["viol_c_heat_1y"] = bbl_series.map(c_heat_1y).fillna(0).astype(np.float32)
    res_df["viol_c_heat_3y"] = bbl_series.map(c_heat_3y).fillna(0).astype(np.float32)

    res_df["viol_c_lead_1y"] = bbl_series.map(c_lead_1y).fillna(0).astype(np.float32)
    res_df["viol_c_lead_3y"] = bbl_series.map(c_lead_3y).fillna(0).astype(np.float32)
    res_df["viol_c_lead_life"] = bbl_series.map(c_lead_life).fillna(0).astype(np.float32)

    res_df["viol_c_mold_1y"] = bbl_series.map(c_mold_1y).fillna(0).astype(np.float32)
    res_df["viol_c_mold_3y"] = bbl_series.map(c_mold_3y).fillna(0).astype(np.float32)
    res_df["viol_c_mold_life"] = bbl_series.map(c_mold_life).fillna(0).astype(np.float32)

    res_df["viol_c_alarm_1y"] = bbl_series.map(c_alarm_1y).fillna(0).astype(np.float32)
    res_df["viol_c_alarm_3y"] = bbl_series.map(c_alarm_3y).fillna(0).astype(np.float32)

    res_df["open_c_dwell_mean"] = (
        bbl_series.map(open_c_dwell_mean).fillna(0).astype(np.float32)
    )
    res_df["open_c_dwell_max"] = (
        bbl_series.map(open_c_dwell_max).fillna(0).astype(np.float32)
    )
    res_df["open_c_dwell_mean_log"] = np.log1p(res_df["open_c_dwell_mean"]).astype(
        np.float32
    )
    res_df["open_c_dwell_max_log"] = np.log1p(res_df["open_c_dwell_max"]).astype(
        np.float32
    )

    res_df["viol_b_1y"] = bbl_series.map(b_1y).fillna(0).astype(np.float32)
    res_df["viol_b_3y"] = bbl_series.map(b_3y).fillna(0).astype(np.float32)
    res_df["viol_a_1y"] = bbl_series.map(a_1y).fillna(0).astype(np.float32)
    res_df["open_viol_c"] = bbl_series.map(open_c).fillna(0).astype(np.float32)
    res_df["open_viol_all"] = bbl_series.map(open_all).fillna(0).astype(np.float32)

    res_df["chronic_open_viol_c"] = (
        bbl_series.map(chronic_open_c).fillna(0).astype(np.float32)
    )
    res_df["chronic_open_c_ratio"] = (
        res_df["chronic_open_viol_c"] / (res_df["open_viol_c"] + 1.0)
    ).astype(np.float32)
    res_df["chronic_open_c_per_unit"] = (
        res_df["chronic_open_viol_c"] / (units + 1e-4)
    ).astype(np.float32)

    res_df["insp_encounters_1y"] = (
        bbl_series.map(insp_enc_1y).fillna(0).astype(np.float32)
    )
    res_df["insp_encounters_3y"] = (
        bbl_series.map(insp_enc_3y).fillna(0).astype(np.float32)
    )
    res_df["insp_encounter_velocity"] = (
        res_df["insp_encounters_1y"] / (res_df["insp_encounters_3y"] / 3.0 + 1e-4)
    ).astype(np.float32)

    # Dynamic Ratios
    res_df["viol_c_per_unit_1y"] = (res_df["viol_c_1y"] / (units + 1e-4)).astype(
        np.float32
    )
    res_df["viol_c_per_unit_3y"] = (res_df["viol_c_3y"] / (units + 1e-4)).astype(
        np.float32
    )
    res_df["open_c_per_unit"] = (res_df["open_viol_c"] / (units + 1e-4)).astype(
        np.float32
    )
    res_df["c_accel_90d"] = (
        res_df["viol_c_90d"] / (res_df["viol_c_1y"] / 4.0 + 1e-4)
    ).astype(np.float32)
    res_df["c_ratio_1y_to_3y"] = (
        res_df["viol_c_1y"] / (res_df["viol_c_3y"] / 3.0 + 1e-4)
    ).astype(np.float32)
    viol_total_1y = res_df["viol_c_1y"] + res_df["viol_b_1y"] + res_df["viol_a_1y"]
    res_df["viol_c_share_1y"] = (res_df["viol_c_1y"] / (viol_total_1y + 1e-4)).astype(
        np.float32
    )
    res_df["viol_c_accel_to_backlog"] = (
        res_df["viol_c_90d"] * np.log1p(res_df["open_viol_c"])
    ).astype(np.float32)

    # Disjoint Yearly Class C and Momentum Ratios
    c_y1 = res_df["viol_c_1y"]
    c_prev1y = (res_df["viol_c_2y"] - res_df["viol_c_1y"]).clip(lower=0)
    c_prev2y = (res_df["viol_c_3y"] - res_df["viol_c_2y"]).clip(lower=0)
    res_df["viol_c_disjoint_prev1y"] = c_prev1y.astype(np.float32)
    res_df["viol_c_disjoint_prev2y"] = c_prev2y.astype(np.float32)
    res_df["viol_c_yoy_delta"] = (c_y1 - c_prev1y).astype(np.float32)
    res_df["viol_c_momentum_1y"] = ((c_y1 + 0.1) / (c_prev1y + 0.1)).astype(np.float32)
    res_df["viol_c_momentum_2y"] = ((c_prev1y + 0.1) / (c_prev2y + 0.1)).astype(np.float32)

    # Compliance Resolution & Open Backlog Ratios
    res_df["c_compliance_resolution_1y"] = (
        (res_df["viol_c_1y"] - res_df["open_viol_c"]).clip(lower=0)
        / (res_df["viol_c_1y"] + 1.0)
    ).astype(np.float32)
    res_df["c_compliance_resolution_life"] = (
        (res_df["viol_c_life"] - res_df["open_viol_c"]).clip(lower=0)
        / (res_df["viol_c_life"] + 1.0)
    ).astype(np.float32)
    res_df["open_backlog_ratio_1y"] = (
        res_df["open_viol_c"] / (res_df["viol_c_1y"] + 1.0)
    ).astype(np.float32)
    res_df["open_backlog_ratio_life"] = (
        res_df["open_viol_c"] / (res_df["viol_c_life"] + 1.0)
    ).astype(np.float32)

    # 2. HPD Complaints
    cmp_sub = df_comp[df_comp["dt"] < cutoff_ts]
    cmp_30 = cmp_sub[cmp_sub["dt"] >= dt_30d].groupby("clean_bbl").size()
    cmp_90 = cmp_sub[cmp_sub["dt"] >= dt_90d].groupby("clean_bbl").size()
    cmp_1y = cmp_sub[cmp_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
    cmp_2y = cmp_sub[cmp_sub["dt"] >= dt_2y].groupby("clean_bbl").size()
    cmp_3y = cmp_sub[cmp_sub["dt"] >= dt_3y].groupby("clean_bbl").size()
    cmp_last = cmp_sub.groupby("clean_bbl")["dt"].max()

    res_df["comp_30d"] = bbl_series.map(cmp_30).fillna(0).astype(np.float32)
    res_df["comp_90d"] = bbl_series.map(cmp_90).fillna(0).astype(np.float32)
    res_df["comp_1y"] = bbl_series.map(cmp_1y).fillna(0).astype(np.float32)
    res_df["comp_2y"] = bbl_series.map(cmp_2y).fillna(0).astype(np.float32)
    res_df["comp_3y"] = bbl_series.map(cmp_3y).fillna(0).astype(np.float32)
    last_dt_cmp = bbl_series.map(cmp_last)
    res_df["days_since_comp"] = (
        (cutoff_ts - last_dt_cmp).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
    )
    res_df["comp_per_unit_1y"] = (res_df["comp_1y"] / (units + 1e-4)).astype(np.float32)
    res_df["comp_accel_90d"] = (
        res_df["comp_90d"] / (res_df["comp_1y"] / 4.0 + 1e-4)
    ).astype(np.float32)

    # Disjoint Yearly Complaints and Momentum Ratios
    cmp_y1 = res_df["comp_1y"]
    cmp_prev1y = (res_df["comp_2y"] - res_df["comp_1y"]).clip(lower=0)
    cmp_prev2y = (res_df["comp_3y"] - res_df["comp_2y"]).clip(lower=0)
    res_df["comp_disjoint_prev1y"] = cmp_prev1y.astype(np.float32)
    res_df["comp_disjoint_prev2y"] = cmp_prev2y.astype(np.float32)
    res_df["comp_yoy_delta"] = (cmp_y1 - cmp_prev1y).astype(np.float32)
    res_df["comp_momentum_1y"] = ((cmp_y1 + 0.1) / (cmp_prev1y + 0.1)).astype(np.float32)
    res_df["comp_momentum_2y"] = ((cmp_prev1y + 0.1) / (cmp_prev2y + 0.1)).astype(np.float32)

    if "is_heat_comp" in cmp_sub.columns:
        cmp_heat = cmp_sub[cmp_sub["is_heat_comp"]]
        cmp_heat_90 = cmp_heat[cmp_heat["dt"] >= dt_90d].groupby("clean_bbl").size()
        res_df["comp_heat_90d"] = (
            bbl_series.map(cmp_heat_90).fillna(0).astype(np.float32)
        )
    else:
        res_df["comp_heat_90d"] = np.float32(0.0)

    # 3. DOB ECB Violations
    if len(df_ecb) > 0 and "dt" in df_ecb.columns:
        ecb_sub = df_ecb[df_ecb["dt"] < cutoff_ts]
        ecb_c1 = ecb_sub[ecb_sub["is_class1"]].groupby("clean_bbl").size()
        ecb_c1_1y = (
            ecb_sub[(ecb_sub["is_class1"]) & (ecb_sub["dt"] >= dt_1y)]
            .groupby("clean_bbl")
            .size()
        )
        ecb_pen = ecb_sub.groupby("clean_bbl")["penalty"].sum()
        res_df["dob_ecb_class1_life"] = (
            bbl_series.map(ecb_c1).fillna(0).astype(np.float32)
        )
        res_df["dob_ecb_class1_1y"] = (
            bbl_series.map(ecb_c1_1y).fillna(0).astype(np.float32)
        )
        res_df["dob_ecb_penalty_log"] = np.log1p(
            bbl_series.map(ecb_pen).fillna(0).clip(lower=0)
        ).astype(np.float32)
    else:
        res_df["dob_ecb_class1_life"] = np.float32(0.0)
        res_df["dob_ecb_class1_1y"] = np.float32(0.0)
        res_df["dob_ecb_penalty_log"] = np.float32(0.0)

    # 4. DOB Safety Mechanical Mandates (Boiler/Elevator)
    if len(df_safety) > 0 and "dt" in df_safety.columns:
        sft_sub = df_safety[df_safety["dt"] < cutoff_ts]
        sft_b = sft_sub[sft_sub["is_boiler"]].groupby("clean_bbl").size()
        sft_e = sft_sub[sft_sub["is_elevator"]].groupby("clean_bbl").size()
        res_df["dob_boiler_viol"] = bbl_series.map(sft_b).fillna(0).astype(np.float32)
        res_df["dob_elevator_viol"] = bbl_series.map(sft_e).fillna(0).astype(np.float32)
    else:
        res_df["dob_boiler_viol"] = np.float32(0.0)
        res_df["dob_elevator_viol"] = np.float32(0.0)

    # 5. Emergency Municipal Repairs (HWO & OMO Charges) & Vacate Orders
    if len(df_hwo) > 0 and "dt" in df_hwo.columns:
        hwo_sub = df_hwo[df_hwo["dt"] < cutoff_ts]
        hwo_cnt_1y = hwo_sub[hwo_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
        hwo_amt_1y = (
            hwo_sub[hwo_sub["dt"] >= dt_1y].groupby("clean_bbl")["amount"].sum()
        )
        res_df["hpd_hwo_cnt_1y"] = (
            bbl_series.map(hwo_cnt_1y).fillna(0).astype(np.float32)
        )
        res_df["hpd_hwo_amt_log_1y"] = np.log1p(
            bbl_series.map(hwo_amt_1y).fillna(0).clip(lower=0)
        ).astype(np.float32)
    else:
        res_df["hpd_hwo_cnt_1y"] = np.float32(0.0)
        res_df["hpd_hwo_amt_log_1y"] = np.float32(0.0)

    if len(df_omo) > 0 and "dt" in df_omo.columns:
        omo_sub = df_omo[df_omo["dt"] < cutoff_ts]
        omo_cnt_life = omo_sub.groupby("clean_bbl").size()
        omo_amt_life = omo_sub.groupby("clean_bbl")["amount"].sum()
        res_df["hpd_omo_cnt_life"] = (
            bbl_series.map(omo_cnt_life).fillna(0).astype(np.float32)
        )
        res_df["hpd_omo_amt_log_life"] = np.log1p(
            bbl_series.map(omo_amt_life).fillna(0).clip(lower=0)
        ).astype(np.float32)
    else:
        res_df["hpd_omo_cnt_life"] = np.float32(0.0)
        res_df["hpd_omo_amt_log_life"] = np.float32(0.0)

    if len(df_vacate) > 0 and "dt" in df_vacate.columns:
        vac_sub = df_vacate[df_vacate["dt"] < cutoff_ts]
        vac_last = vac_sub.groupby("clean_bbl")["dt"].max()
        vac_cnt_life = vac_sub.groupby("clean_bbl").size()
        last_dt_vac = bbl_series.map(vac_last)
        res_df["days_since_vacate_order"] = (
            (cutoff_ts - last_dt_vac).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
        )
        res_df["hpd_vacate_cnt_life"] = (
            bbl_series.map(vac_cnt_life).fillna(0).astype(np.float32)
        )
    else:
        res_df["days_since_vacate_order"] = np.float32(3650.0)
        res_df["hpd_vacate_cnt_life"] = np.float32(0.0)

    # 6. HPD Litigations
    if len(df_lit) > 0 and "dt" in df_lit.columns:
        lit_sub = df_lit[df_lit["dt"] < cutoff_ts]
        lit_1y = lit_sub[lit_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
        lit_life = lit_sub.groupby("clean_bbl").size()
        res_df["hpd_lit_1y"] = bbl_series.map(lit_1y).fillna(0).astype(np.float32)
        res_df["hpd_lit_life"] = bbl_series.map(lit_life).fillna(0).astype(np.float32)
    else:
        res_df["hpd_lit_1y"] = np.float32(0.0)
        res_df["hpd_lit_life"] = np.float32(0.0)

    # 7. Habitability: Evictions & Rodents
    if len(df_evict) > 0 and "dt" in df_evict.columns:
        ev_sub = df_evict[(df_evict["dt"] < cutoff_ts) & (df_evict["dt"] >= dt_2y)]
        ev_cnt = ev_sub.groupby("clean_bbl").size()
        res_df["evictions_2y"] = bbl_series.map(ev_cnt).fillna(0).astype(np.float32)
    else:
        res_df["evictions_2y"] = np.float32(0.0)

    if len(df_rodent) > 0 and "dt" in df_rodent.columns:
        rd_sub = df_rodent[
            (df_rodent["dt"] < cutoff_ts)
            & (df_rodent["dt"] >= dt_1y)
            & (df_rodent["is_fail"])
        ]
        rd_cnt = rd_sub.groupby("clean_bbl").size()
        res_df["rodent_failures_1y"] = (
            bbl_series.map(rd_cnt).fillna(0).astype(np.float32)
        )
    else:
        res_df["rodent_failures_1y"] = np.float32(0.0)

    # Multi-Agency: DOB Building Code Violations
    if len(df_dob_viol) > 0 and "dt" in df_dob_viol.columns:
        dob_sub = df_dob_viol[df_dob_viol["dt"] < cutoff_ts]
        dob_1y = dob_sub[dob_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
        dob_life = dob_sub.groupby("clean_bbl").size()
        res_df["dob_viol_1y"] = bbl_series.map(dob_1y).fillna(0).astype(np.float32)
        res_df["dob_viol_life"] = bbl_series.map(dob_life).fillna(0).astype(np.float32)
        res_df["dob_viol_per_unit_1y"] = (
            res_df["dob_viol_1y"] / (units + 1e-4)
        ).astype(np.float32)
    else:
        res_df["dob_viol_1y"] = np.float32(0.0)
        res_df["dob_viol_life"] = np.float32(0.0)
        res_df["dob_viol_per_unit_1y"] = np.float32(0.0)

    # Multi-Agency: HPD Bedbug Infestation Reports
    if len(df_bedbug) > 0 and "dt" in df_bedbug.columns:
        bb_sub = df_bedbug[df_bedbug["dt"] < cutoff_ts]
        bb_1y = bb_sub[bb_sub["dt"] >= dt_1y]
        bb_cnt_1y = bb_1y.groupby("clean_bbl").size()
        bb_inf_1y = bb_1y.groupby("clean_bbl")["infested_units"].sum()
        bb_cnt_life = bb_sub.groupby("clean_bbl").size()
        res_df["bedbug_reports_1y"] = (
            bbl_series.map(bb_cnt_1y).fillna(0).astype(np.float32)
        )
        res_df["bedbug_reports_life"] = (
            bbl_series.map(bb_cnt_life).fillna(0).astype(np.float32)
        )
        res_df["bedbug_infested_units_1y"] = (
            bbl_series.map(bb_inf_1y).fillna(0).astype(np.float32)
        )
        res_df["bedbug_infested_ratio_1y"] = (
            res_df["bedbug_infested_units_1y"] / (units + 1e-4)
        ).astype(np.float32)
    else:
        res_df["bedbug_reports_1y"] = np.float32(0.0)
        res_df["bedbug_reports_life"] = np.float32(0.0)
        res_df["bedbug_infested_units_1y"] = np.float32(0.0)
        res_df["bedbug_infested_ratio_1y"] = np.float32(0.0)

    # 8. Statutory Distress Programs
    res_df["flag_aep"] = bbl_series.isin(aep_bbl_set).astype(np.float32)
    res_df["flag_conh"] = bbl_series.isin(conh_bbl_set).astype(np.float32)
    res_df["flag_speculation"] = bbl_series.isin(spec_bbl_set).astype(np.float32)

    # 9. Spatial Tax Block and ZIP Code Class C Density (Leave-One-Out approximation)
    blk_sum = res_df.groupby("block")["viol_c_1y"].transform("sum")
    blk_cnt = res_df.groupby("block")["viol_c_1y"].transform("count")
    res_df["block_c_density"] = (
        ((blk_sum - res_df["viol_c_1y"]) / (blk_cnt - 1 + 1e-4))
        .clip(lower=0)
        .astype(np.float32)
    )

    zip_sum = res_df.groupby("zipcode")["viol_c_1y"].transform("sum")
    zip_cnt = res_df.groupby("zipcode")["viol_c_1y"].transform("count")
    zip_risk = ((zip_sum - res_df["viol_c_1y"]) / np.maximum(zip_cnt - 1.0, 1.0)).clip(lower=0)
    zip_risk = np.where(res_df["zipcode"] == 0, 0.0, zip_risk)
    res_df["zip_c_density"] = zip_risk.astype(np.float32)

    # 10. DOF Tax Lien Sales (Municipal tax delinquency)
    if len(df_lien) > 0 and "dt" in df_lien.columns and df_lien["dt"].notna().any():
        lien_sub = df_lien[df_lien["dt"] < cutoff_ts]
        lien_cnt_life = lien_sub.groupby("clean_bbl").size()
        lien_cnt_3y = lien_sub[lien_sub["dt"] >= dt_3y].groupby("clean_bbl").size()
        lien_last = lien_sub.groupby("clean_bbl")["dt"].max()
        res_df["tax_lien_cnt_life"] = (
            bbl_series.map(lien_cnt_life).fillna(0).astype(np.float32)
        )
        res_df["tax_lien_cnt_3y"] = (
            bbl_series.map(lien_cnt_3y).fillna(0).astype(np.float32)
        )
        last_dt_lien = bbl_series.map(lien_last)
        res_df["days_since_tax_lien"] = (
            (cutoff_ts - last_dt_lien)
            .dt.days.fillna(3650)
            .clip(0, 3650)
            .astype(np.float32)
        )
        res_df["has_tax_lien"] = (res_df["tax_lien_cnt_life"] > 0).astype(np.float32)
    elif len(df_lien) > 0 and "clean_bbl" in df_lien.columns:
        lien_cnt_life = df_lien.groupby("clean_bbl").size()
        res_df["tax_lien_cnt_life"] = (
            bbl_series.map(lien_cnt_life).fillna(0).astype(np.float32)
        )
        res_df["tax_lien_cnt_3y"] = np.float32(0.0)
        res_df["days_since_tax_lien"] = np.float32(3650.0)
        res_df["has_tax_lien"] = (res_df["tax_lien_cnt_life"] > 0).astype(np.float32)
    else:
        res_df["tax_lien_cnt_life"] = np.float32(0.0)
        res_df["tax_lien_cnt_3y"] = np.float32(0.0)
        res_df["days_since_tax_lien"] = np.float32(3650.0)
        res_df["has_tax_lien"] = np.float32(0.0)

    # Drop non-feature join columns
    res_df = res_df.drop(columns=["bbl", "block", "zipcode"])
    return res_df


def compute_ground_truth_labels(bbl_list, start_dt, end_dt):
    """Exact competition target definition."""
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    c_in_win = df_viol[
        (df_viol["class"] == "C")
        & (df_viol["insp_dt"] >= start_ts)
        & (df_viol["insp_dt"] < end_ts)
    ]
    pos_set = set(c_in_win["clean_bbl"].unique())
    y = np.array([1 if b in pos_set else 0 for b in bbl_list], dtype=np.int32)
    return y


# ---------------------------------------------------------
# 5. Cohort Generation (Longitudinal Train, Val, Test)
# ---------------------------------------------------------
print("\n--- Constructing Cohorts ---")
# 2020 Cohort: latest PLUTO before 2020-01-01 is 19v2
pluto_19 = df_pluto[
    df_pluto["version"].str.contains("19") & (df_pluto["unitsres"] >= 3)
]
bbls_2020 = pluto_19["clean_bbl"].unique()
print(f"Cohort 2020: {len(bbls_2020):,} lots")
X_2020 = extract_cohort_features(bbls_2020, pluto_19, "2020-01-01")
y_2020 = compute_ground_truth_labels(bbls_2020, "2020-01-01", "2021-01-01")
w_2020 = np.full(len(y_2020), 0.60, dtype=np.float32)

# 2021 Cohort: latest PLUTO before 2021-01-01 is 20v7
pluto_20 = df_pluto[
    df_pluto["version"].str.contains("20") & (df_pluto["unitsres"] >= 3)
]
bbls_2021 = pluto_20["clean_bbl"].unique()
print(f"Cohort 2021: {len(bbls_2021):,} lots")
X_2021 = extract_cohort_features(bbls_2021, pluto_20, "2021-01-01")
y_2021 = compute_ground_truth_labels(bbls_2021, "2021-01-01", "2022-01-01")
w_2021 = np.full(len(y_2021), 1.00, dtype=np.float32)

# Combine into Pooled Training Set
feature_cols = list(X_2021.columns)
X_train = pd.concat([X_2020[feature_cols], X_2021[feature_cols]], axis=0).reset_index(
    drop=True
)
y_train = np.concatenate([y_2020, y_2021])
sample_weights_train = np.concatenate([w_2020, w_2021])
print(
    f"Pooled Training Set: {len(X_train):,} samples (Positives: {y_train.sum():,}, Rate: {y_train.mean():.4f})"
)

# 2022 Validation Cohort: latest PLUTO before 2022-01-01 is 21v4
pluto_21 = df_pluto[
    df_pluto["version"].str.contains("21") & (df_pluto["unitsres"] >= 3)
]
bbls_2022 = pluto_21["clean_bbl"].unique()
print(f"Validation Cohort 2022: {len(bbls_2022):,} lots")
X_val = extract_cohort_features(bbls_2022, pluto_21, "2022-01-01")[
    feature_cols
].reset_index(drop=True)
y_val = compute_ground_truth_labels(bbls_2022, "2022-01-01", "2023-01-01")
print(
    f"Validation Set: {len(X_val):,} samples (Positives: {y_val.sum():,}, Rate: {y_val.mean():.4f})"
)

# 2023 Test Cohort: test entities from PLUTO 22v3
pluto_22 = df_pluto[
    df_pluto["version"].str.contains("22") & (df_pluto["unitsres"] >= 3)
]
print(f"Test Cohort 2023: {len(test_bbls):,} lots")
X_test = extract_cohort_features(test_bbls, pluto_22, "2023-01-01")[
    feature_cols
].reset_index(drop=True)

# Free up memory
del df_pluto, pluto_19, pluto_20, pluto_21, pluto_22, X_2020, X_2021
gc.collect()

print(f"Feature count: {len(feature_cols)}")

# Convert to numpy arrays
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# Impute any edge NaNs/Infs
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------------------------------------
# 6. Structurally Diverse Model Fleet
# ---------------------------------------------------------
val_preds = []
test_preds = []
model_names = []

# --- Model 1: Deep Leaf-Wise LightGBM ---
print("\n[1/5] Training Deep Leaf-Wise LightGBM...")
t0 = time.time()
lgb_deep = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=128,
    max_depth=8,
    min_child_samples=35,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.60,
    scale_pos_weight=1.5,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_deep.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_np, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
p_val_1 = lgb_deep.predict_proba(X_val_np)[:, 1]
p_test_1 = lgb_deep.predict_proba(X_test_np)[:, 1]
score_1 = average_precision_score(y_val, p_val_1)
print(f"LGBM Deep Validation AP: {score_1:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_1)
test_preds.append(p_test_1)
model_names.append("LGBM_Deep")

# --- Model 2: Extremely Randomized LightGBM ---
print("\n[2/5] Training Extremely Randomized LightGBM (ExtraTrees)...")
t0 = time.time()
lgb_et = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    extra_trees=True,
    n_estimators=850,
    learning_rate=0.035,
    num_leaves=64,
    max_depth=7,
    min_child_samples=50,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.50,
    scale_pos_weight=1.2,
    reg_alpha=0.5,
    reg_lambda=3.0,
    random_state=123,
    n_jobs=-1,
    verbose=-1,
)
lgb_et.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_np, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
p_val_2 = lgb_et.predict_proba(X_val_np)[:, 1]
p_test_2 = lgb_et.predict_proba(X_test_np)[:, 1]
score_2 = average_precision_score(y_val, p_val_2)
print(
    f"LGBM ExtraTrees Validation AP: {score_2:.5f} (trained in {time.time()-t0:.1f}s)"
)
val_preds.append(p_val_2)
test_preds.append(p_test_2)
model_names.append("LGBM_ExtraTrees")

# --- Model 3: Histogram XGBoost ---
print("\n[3/5] Training Multi-Depth Histogram XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=850,
    learning_rate=0.035,
    max_depth=7,
    subsample=0.80,
    colsample_bytree=0.60,
    scale_pos_weight=1.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    early_stopping_rounds=50,
    random_state=789,
    n_jobs=-1,
)
xgb_model.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_np, y_val)],
    verbose=False,
)
p_val_3 = xgb_model.predict_proba(X_val_np)[:, 1]
p_test_3 = xgb_model.predict_proba(X_test_np)[:, 1]
score_3 = average_precision_score(y_val, p_val_3)
print(f"XGBoost Validation AP: {score_3:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_3)
test_preds.append(p_test_3)
model_names.append("XGB_Hist")

# --- Model 4: Subspace Symmetric Oblivious CatBoost ---
print("\n[4/5] Training Subspace Symmetric Oblivious CatBoost...")
t0 = time.time()
cb_model = CatBoostClassifier(
    iterations=750,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=4.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    scale_pos_weight=1.3,
    rsm=0.80,
    subsample=0.85,
    bootstrap_type="Bernoulli",
    early_stopping_rounds=50,
    random_seed=456,
    verbose=False,
    thread_count=-1,
)
cb_model.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=(X_val_np, y_val),
    verbose=False,
)
p_val_4 = cb_model.predict_proba(X_val_np)[:, 1]
p_test_4 = cb_model.predict_proba(X_test_np)[:, 1]
score_4 = average_precision_score(y_val, p_val_4)
print(f"CatBoost Subspace Validation AP: {score_4:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_4)
test_preds.append(p_test_4)
model_names.append("CatBoost_Subspace")

# --- Model 5: Non-Greedy Tree-Dropout LightGBM (DART) ---
print("\n[5/5] Training Non-Greedy Tree-Dropout LightGBM (DART)...")
t0 = time.time()
lgb_dart = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="dart",
    drop_rate=0.10,
    skip_drop=0.50,
    n_estimators=650,
    learning_rate=0.04,
    num_leaves=64,
    max_depth=7,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.60,
    scale_pos_weight=1.5,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=321,
    n_jobs=-1,
    verbose=-1,
)
lgb_dart.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
)
p_val_5 = lgb_dart.predict_proba(X_val_np)[:, 1]
p_test_5 = lgb_dart.predict_proba(X_test_np)[:, 1]
score_5 = average_precision_score(y_val, p_val_5)
print(f"LGBM DART Validation AP: {score_5:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_5)
test_preds.append(p_test_5)
model_names.append("LGBM_DART")

# ---------------------------------------------------------
# 7. Simplex Rank Optimization on Validation AP
# ---------------------------------------------------------
print("\n--- Model Fleet Summary ---")
for name, sc in zip(model_names, [score_1, score_2, score_3, score_4, score_5]):
    print(f"  {name:20s}: {sc:.5f}")

# Convert predictions to fractional percentile ranks
val_ranks = np.array([rankdata(p) / len(p) for p in val_preds])
test_ranks = np.array([rankdata(p) / len(p) for p in test_preds])


def optimize_simplex_weights(ranks, y_true):
    n = len(ranks)
    # Initialize with uniform weights
    best_w = np.ones(n) / n
    best_score = average_precision_score(y_true, np.dot(best_w, ranks))

    # Evaluate single model defaults
    for i in range(n):
        single_w = np.zeros(n)
        single_w[i] = 1.0
        score = average_precision_score(y_true, ranks[i])
        if score > best_score:
            best_score = score
            best_w = single_w

    # Multi-scale coordinate descent
    for step in [0.08, 0.04, 0.02, 0.01, 0.005]:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if best_w[i] >= step:
                        cand_w = best_w.copy()
                        cand_w[i] -= step
                        cand_w[j] += step
                        cand_w /= cand_w.sum()
                        score = average_precision_score(y_true, np.dot(cand_w, ranks))
                        if score > best_score + 1e-6:
                            best_score = score
                            best_w = cand_w
                            improved = True
    return best_w, best_score


print("\nOptimizing ensemble simplex weights against Average Precision...")
opt_weights, final_val_score = optimize_simplex_weights(val_ranks, y_val)
print("Optimal Weights:")
for name, w in zip(model_names, opt_weights):
    print(f"  {name:20s}: {w:.4f}")

# Compute holdout metrics
val_ensemble_scores = np.dot(opt_weights, val_ranks)
val_roc_auc = roc_auc_score(y_val, val_ensemble_scores)
print(f"Ensemble Holdout Validation ROC-AUC: {val_roc_auc:.5f}")

# ---------------------------------------------------------
# 8. Test Inference & Submission Generation
# ---------------------------------------------------------
test_ensemble_scores = np.dot(opt_weights, test_ranks)

os.makedirs("./submission", exist_ok=True)
sub_df = pd.DataFrame(
    {"bbl": test_bbls, "score": test_ensemble_scores.astype(np.float64)}
)

# Submission verification checks
assert len(sub_df) == 171587, f"Expected 171587 rows, got {len(sub_df)}"
assert sub_df["bbl"].nunique() == 171587, "Duplicate BBLs detected in submission!"
assert not sub_df["score"].isna().any(), "NaN values found in submission score!"
assert not np.isinf(sub_df["score"]).any(), "Infinite values found in submission score!"

sub_df.to_csv("./submission/submission.csv", index=False)
print(f"Successfully generated ./submission/submission.csv ({len(sub_df):,} rows).")

# ---------------------------------------------------------
# 9. Final Validation Metric Output
# ---------------------------------------------------------
print(f"Final Validation Score: {final_val_score}")

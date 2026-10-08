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
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    "unitstotal",
    "yearbuilt",
    "bldgarea",
    "numfloors",
    "borocode",
    "block",
    "lot",
    "zipcode",
    "cd",
    "assesstot",
    "assessland",
    "builtfar",
    "residfar",
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
df_pluto["unitstotal"] = safe_series_numeric(df_pluto, "unitstotal", 0.0)
df_pluto["yearbuilt"] = safe_series_numeric(df_pluto, "yearbuilt", 0.0)
df_pluto["bldgarea"] = safe_series_numeric(df_pluto, "bldgarea", 0.0)
df_pluto["numfloors"] = safe_series_numeric(df_pluto, "numfloors", 0.0)
df_pluto["assesstot"] = safe_series_numeric(df_pluto, "assesstot", 0.0)
df_pluto["assessland"] = safe_series_numeric(df_pluto, "assessland", 0.0)
df_pluto["builtfar"] = safe_series_numeric(df_pluto, "builtfar", 0.0)
df_pluto["residfar"] = safe_series_numeric(df_pluto, "residfar", 0.0)
df_pluto["lotarea"] = safe_series_numeric(df_pluto, "lotarea", 0.0)
df_pluto["yearalter1"] = safe_series_numeric(df_pluto, "yearalter1", 0.0)
if "cd" in df_pluto.columns:
    df_pluto["cd"] = pd.to_numeric(df_pluto["cd"], errors="coerce").fillna(0).astype(np.int32)
else:
    df_pluto["cd"] = pd.Series(0, index=df_pluto.index, dtype=np.int32)
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
    "correctbydate",
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
df_viol["is_door_safety"] = ord_s.str.contains(
    "591|592|593", regex=True, na=False
) | nov_upper.str.contains("SELF-CLOSING|SELF CLOSING|FIRE DOOR", regex=True, na=False)
del ord_s, nov_upper
if "novdescription" in df_viol.columns:
    df_viol.drop(columns=["novdescription"], inplace=True)
df_viol["status_dt"] = pd.to_datetime(df_viol.get("currentstatusdate"), errors="coerce")
df_viol["correct_dt"] = pd.to_datetime(df_viol.get("correctbydate"), errors="coerce")
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
    [
        "statusdate",
        "status_date",
        "received_date",
        "majorcategory",
        "minorcategory",
        "apartment",
        "apt",
        "unit",
    ],
    col_patterns=["apartment", "category", "apt"],
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

# Apartment identifier cleaning
apt_col = next(
    (c for c in ["apartment", "apt", "unit", "apartment_number"] if c in df_comp.columns),
    None,
)
if apt_col:
    df_comp["clean_apt"] = (
        df_comp[apt_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(["", "NAN", "NONE", "NULL", "UNKNOWN", "N/A"], np.nan)
    )
else:
    df_comp["clean_apt"] = np.nan

# Category parsing
cat_cols = [c for c in ["majorcategory", "minorcategory"] if c in df_comp.columns]
if cat_cols:
    cat_text = df_comp[cat_cols[0]].fillna("").astype(str).str.upper()
    for extra_c in cat_cols[1:]:
        cat_text = cat_text + " " + df_comp[extra_c].fillna("").astype(str).str.upper()
    df_comp["is_heat_comp"] = cat_text.str.contains("HEAT|HOT WATER", regex=True, na=False)
    df_comp["is_plumb_leak"] = cat_text.str.contains("PLUMB|LEAK|WATER", regex=True, na=False)
    df_comp["is_paint_plaster"] = cat_text.str.contains("PAINT|PLASTER", regex=True, na=False)
    df_comp["is_door_window"] = cat_text.str.contains("DOOR|WINDOW|SAFETY|LOCK|FIRE ESCAPE|GUARD", regex=True, na=False)
    df_comp["is_pest_comp"] = cat_text.str.contains("PEST|RODENT|VERMIN|MICE|RATS|BEDBUG|ROACH", regex=True, na=False)
    df_comp["is_safety_comp"] = cat_text.str.contains("FIRE|COLLAPSE|ELECTRICAL|GAS|BOILER|STRUCTURAL|SMOKE", regex=True, na=False)
else:
    df_comp["is_heat_comp"] = False
    df_comp["is_plumb_leak"] = False
    df_comp["is_paint_plaster"] = False
    df_comp["is_door_window"] = False
    df_comp["is_pest_comp"] = False
    df_comp["is_safety_comp"] = False
print(f"HPD Complaints loaded: {len(df_comp):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
print("Ingesting HPD Registrations & Contacts...")
reg_contact_cols = [
    "registrationid",
    "corporationname",
    "businesshousenumber",
    "businessstreetname",
    "businesszip",
]
df_reg_contacts = safe_read_parquet(
    "lake/full/hpd_registration_contacts",
    reg_contact_cols,
    ["registration_id", "regid", "type", "contacttype"],
)

reg_cols = ["bbl", "boroid", "boro", "block", "lot", "registrationid"]
df_reg = safe_read_parquet(
    "lake/full/hpd_registrations",
    reg_cols,
    ["registration_id", "regid", "lastregistrationdate"],
    col_patterns=["registrationdate", "lastregistration"],
)
if len(df_reg) == 0:
    df_reg = safe_read_parquet(
        "lake/full/hpd_registration_summary",
        reg_cols,
        ["registration_id", "regid", "lastregistrationdate"],
        col_patterns=["registrationdate", "lastregistration"],
    )

reg_to_owner = {}
if len(df_reg_contacts) > 0:
    reg_id_c = next(
        (c for c in ["registrationid", "registration_id", "regid"] if c in df_reg_contacts.columns),
        None,
    )
    if reg_id_c:
        df_reg_contacts["reg_id"] = (
            pd.to_numeric(df_reg_contacts[reg_id_c], errors="coerce").fillna(0).astype(np.int64)
        )
    else:
        df_reg_contacts["reg_id"] = 0

    corp = (
        df_reg_contacts.get("corporationname", pd.Series("", index=df_reg_contacts.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    b_num = (
        df_reg_contacts.get("businesshousenumber", pd.Series("", index=df_reg_contacts.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    b_street = (
        df_reg_contacts.get("businessstreetname", pd.Series("", index=df_reg_contacts.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    b_zip = (
        df_reg_contacts.get("businesszip", pd.Series("", index=df_reg_contacts.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
        .str.zfill(5)
    )

    bad_toks = {"", "NAN", "NONE", "NULL", "UNKNOWN", "N/A", "N / A", "00000", "0"}
    corp_clean = corp.apply(lambda x: "" if x in bad_toks else x)
    b_num_clean = b_num.apply(lambda x: "" if x in bad_toks else x)
    b_street_clean = b_street.apply(lambda x: "" if x in bad_toks else x)
    b_zip_clean = b_zip.apply(lambda x: "" if x in bad_toks else x)

    addr_clean = (b_num_clean + " " + b_street_clean + " " + b_zip_clean).str.strip()
    owner_key = np.where(corp_clean != "", corp_clean, addr_clean)
    df_reg_contacts["owner_key"] = owner_key

    valid_c = df_reg_contacts[(df_reg_contacts["reg_id"] > 0) & (df_reg_contacts["owner_key"] != "")]
    if len(valid_c) > 0:
        reg_to_owner = (
            valid_c.groupby("reg_id")["owner_key"]
            .agg(lambda s: s.value_counts().index[0] if len(s) > 0 else "")
            .to_dict()
        )
    del df_reg_contacts
    gc.collect()

if len(df_reg) > 0:
    df_reg["clean_bbl"] = clean_bbl_series(
        df_reg.get("bbl"),
        df_reg.get("boroid", df_reg.get("boro")),
        df_reg.get("block"),
        df_reg.get("lot"),
    )
    reg_dt_c = next(
        (c for c in df_reg.columns if "registrationdate" in c or "regdate" in c),
        None,
    )
    if reg_dt_c:
        df_reg["reg_dt"] = pd.to_datetime(df_reg[reg_dt_c], errors="coerce")
    else:
        df_reg["reg_dt"] = pd.NaT

    reg_id_col = next(
        (c for c in ["registrationid", "registration_id", "regid"] if c in df_reg.columns),
        None,
    )
    if reg_id_col:
        df_reg["reg_id"] = (
            pd.to_numeric(df_reg[reg_id_col], errors="coerce").fillna(0).astype(np.int64)
        )
    else:
        df_reg["reg_id"] = 0

    df_reg["landlord_id"] = df_reg["reg_id"].map(reg_to_owner)
    df_reg["landlord_id"] = np.where(
        df_reg["landlord_id"].isna() | (df_reg["landlord_id"] == ""),
        np.where(df_reg["reg_id"] > 0, "REG_" + df_reg["reg_id"].astype(str), ""),
        df_reg["landlord_id"],
    )

    valid_regs = df_reg[(df_reg["clean_bbl"] != "") & (df_reg["reg_id"] > 0)]
    valid_regs_dedup = valid_regs.drop_duplicates(subset=["clean_bbl"], keep="last")
    bbl_to_reg = dict(zip(valid_regs_dedup["clean_bbl"], valid_regs_dedup["reg_id"]))
    bbl_to_landlord = dict(zip(valid_regs_dedup["clean_bbl"], valid_regs_dedup["landlord_id"]))
    landlord_to_port_size = (
        valid_regs[valid_regs["landlord_id"] != ""]
        .groupby("landlord_id")["clean_bbl"]
        .nunique()
        .to_dict()
    )
    reg_to_port_size = valid_regs.groupby("reg_id")["clean_bbl"].nunique().to_dict()
else:
    bbl_to_reg = {}
    bbl_to_landlord = {}
    landlord_to_port_size = {}
    reg_to_port_size = {}
print(f"HPD Registrations loaded: {len(bbl_to_reg):,} mapped lots, {len(landlord_to_port_size):,} corporate landlords in {time.time()-t0:.1f}s")

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
    ct_upper = df_lit.get("casetype", "").astype(str).str.upper()
    df_lit["is_hp"] = ct_upper.str.contains("HP", regex=False, na=False)
    df_lit["is_heat_lit"] = ct_upper.str.contains("HEAT", regex=False, na=False)

df_underlying = safe_read_parquet(
    "lake/full/hpd_underlying_conditions",
    ["bbl", "boroid", "boro", "block", "lot"],
    col_patterns=["date", "order"],
)
if len(df_underlying) > 0:
    df_underlying["clean_bbl"] = clean_bbl_series(
        df_underlying.get("bbl"),
        df_underlying.get("boroid", df_underlying.get("boro")),
        df_underlying.get("block"),
        df_underlying.get("lot"),
    )
    und_dt_col = next(
        (c for c in df_underlying.columns if "date" in c or c.endswith("dt")), None
    )
    df_underlying["dt"] = (
        pd.to_datetime(df_underlying[und_dt_col], errors="coerce") if und_dt_col else pd.NaT
    )
    underlying_bbl_set = set(df_underlying["clean_bbl"].unique())
else:
    underlying_bbl_set = set()

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

df_sales = safe_read_parquet(
    "lake/full/dof_annualized_sales",
    ["borough", "block", "lot", "sale_date", "sale_price"],
    ["bbl", "saledate", "saleprice"],
    col_patterns=["sale", "price", "date"],
)
if len(df_sales) > 0:
    df_sales["clean_bbl"] = clean_bbl_series(
        df_sales.get("bbl"),
        df_sales.get("borough"),
        df_sales.get("block"),
        df_sales.get("lot"),
    )
    sale_dt_c = next(
        (c for c in ["sale_date", "saledate", "date"] if c in df_sales.columns), None
    )
    df_sales["dt"] = (
        pd.to_datetime(df_sales[sale_dt_c], errors="coerce") if sale_dt_c else pd.NaT
    )
    sale_pr_c = next(
        (c for c in ["sale_price", "saleprice", "price"] if c in df_sales.columns), None
    )
    if sale_pr_c:
        df_sales["price"] = pd.to_numeric(
            df_sales[sale_pr_c].astype(str).str.replace("$", "").str.replace(",", ""),
            errors="coerce",
        ).fillna(0.0).astype(np.float32)
    else:
        df_sales["price"] = pd.Series(0.0, index=df_sales.index, dtype=np.float32)
    df_sales = df_sales.dropna(subset=["dt"])
    df_sales = df_sales[df_sales["clean_bbl"] != ""]
else:
    df_sales = pd.DataFrame(columns=["clean_bbl", "dt", "price"])

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
        "unitstotal",
        "yearbuilt",
        "bldgarea",
        "numfloors",
        "block",
        "zipcode",
        "cd",
        "assesstot",
        "assessland",
        "builtfar",
        "residfar",
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
    unitstot = safe_series_numeric(merged, "unitstotal", 0.0).values
    unitstot = np.maximum(unitstot, units)
    comm_units = np.maximum(0.0, unitstot - units)

    yb = merged["yearbuilt"].fillna(1950).values
    yb = np.where((yb < 1800) | (yb > cutoff_ts.year), 1950, yb)
    age = np.maximum(0, cutoff_ts.year - yb)
    bldgarea = merged["bldgarea"].fillna(0).clip(lower=0).values

    assesstot = safe_series_numeric(merged, "assesstot", 0.0).values
    assessland = safe_series_numeric(merged, "assessland", 0.0).values
    builtfar = safe_series_numeric(merged, "builtfar", 0.0).values
    residfar = safe_series_numeric(merged, "residfar", 0.0).values
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
    res_df["cd"] = (
        pd.to_numeric(merged["cd"], errors="coerce").fillna(0).astype(np.int32)
        if "cd" in merged.columns
        else pd.Series(0, index=res_df.index, dtype=np.int32)
    )
    res_df["boro"] = (
        pd.to_numeric(res_df["bbl"].str[:1], errors="coerce").fillna(1).astype(np.int32)
    )

    # PLUTO building class archetype family indicators
    res_df["is_class_c"] = bldgclass_s.str.startswith("C").astype(np.float32)
    res_df["is_class_d"] = bldgclass_s.str.startswith("D").astype(np.float32)
    res_df["is_class_r"] = bldgclass_s.str.startswith("R").astype(np.float32)
    res_df["is_class_s"] = bldgclass_s.str.startswith("S").astype(np.float32)

    # PLUTO economic & alteration valuation features
    res_df["unitstotal"] = unitstot.astype(np.float32)
    res_df["units_commercial"] = comm_units.astype(np.float32)
    res_df["has_commercial"] = (comm_units > 0).astype(np.float32)
    res_df["commercial_ratio"] = (comm_units / (unitstot + 1e-4)).astype(np.float32)

    res_df["assesstot_per_unit"] = (assesstot / (units + 1e-4)).astype(np.float32)
    res_df["assesstot_per_sqft"] = (assesstot / (bldgarea + 1e-4)).astype(np.float32)
    res_df["log_assesstot"] = np.log1p(np.maximum(0, assesstot)).astype(np.float32)
    res_df["assessland"] = assessland.astype(np.float32)
    res_df["land_deprec_ratio"] = (assessland / (assesstot + 1e-4)).clip(0.0, 10.0).astype(np.float32)
    res_df["builtfar"] = builtfar.astype(np.float32)
    res_df["residfar"] = residfar.astype(np.float32)
    res_df["far_utilization"] = (builtfar / (residfar + 1e-4)).clip(0.0, 20.0).astype(np.float32)
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

    # Multi-scale continuous exponential time-decay kernels for Class C violations
    ln2 = np.log(2.0)
    delta_days_c = (
        (cutoff_ts - c_sub["insp_dt"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).values
    c_decay_map = {}
    for hl in [30, 90, 180, 365]:
        s_c = pd.Series(np.exp(-delta_days_c * (ln2 / hl)), index=c_sub["clean_bbl"])
        c_decay_map[hl] = s_c.groupby(level=0).sum()

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

    # Continuous exponential decay kernels for Heat violations
    delta_days_heat = (
        (cutoff_ts - c_heat["insp_dt"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).values
    heat_decay_map = {}
    for hl in [30, 90, 180, 365]:
        s_heat = pd.Series(np.exp(-delta_days_heat * (ln2 / hl)), index=c_heat["clean_bbl"])
        heat_decay_map[hl] = s_heat.groupby(level=0).sum()

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

    # Self-closing door safety violations (orders 591-593)
    c_door = (
        c_sub[c_sub["is_door_safety"]]
        if "is_door_safety" in c_sub.columns
        else pd.DataFrame()
    )
    if len(c_door) > 0:
        c_door_1y = c_door[c_door["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
        c_door_life = c_door.groupby("clean_bbl").size()
    else:
        c_door_1y = pd.Series(dtype=np.float32)
        c_door_life = pd.Series(dtype=np.float32)

    # Comprehensive Class B Trajectories and Decays
    b_sub = v_sub[v_sub["class"] == "B"]
    b_30 = b_sub[b_sub["insp_dt"] >= dt_30d].groupby("clean_bbl").size()
    b_90 = b_sub[b_sub["insp_dt"] >= dt_90d].groupby("clean_bbl").size()
    b_180 = b_sub[b_sub["insp_dt"] >= dt_180d].groupby("clean_bbl").size()
    b_1y = b_sub[b_sub["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    b_2y = b_sub[b_sub["insp_dt"] >= dt_2y].groupby("clean_bbl").size()
    b_3y = b_sub[b_sub["insp_dt"] >= dt_3y].groupby("clean_bbl").size()
    b_life = b_sub.groupby("clean_bbl").size()

    delta_days_b = (
        (cutoff_ts - b_sub["insp_dt"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).values
    b_decay_map = {}
    for hl in [30, 90, 180, 365]:
        s_b = pd.Series(np.exp(-delta_days_b * (ln2 / hl)), index=b_sub["clean_bbl"])
        b_decay_map[hl] = s_b.groupby(level=0).sum()

    a_1y = (
        v_sub[(v_sub["class"] == "A") & (v_sub["insp_dt"] >= dt_1y)]
        .groupby("clean_bbl")
        .size()
    )

    # Multi-Class Total Violation Volumes (A+B+C+I)
    tot_30 = v_sub[v_sub["insp_dt"] >= dt_30d].groupby("clean_bbl").size()
    tot_90 = v_sub[v_sub["insp_dt"] >= dt_90d].groupby("clean_bbl").size()
    tot_180 = v_sub[v_sub["insp_dt"] >= dt_180d].groupby("clean_bbl").size()
    tot_1y = v_sub[v_sub["insp_dt"] >= dt_1y].groupby("clean_bbl").size()
    tot_3y = v_sub[v_sub["insp_dt"] >= dt_3y].groupby("clean_bbl").size()
    tot_life = v_sub.groupby("clean_bbl").size()
    all_last = v_sub.groupby("clean_bbl")["insp_dt"].max()

    open_mask = (
        (v_sub["is_open"])
        | (v_sub["status_dt"].isna())
        | (v_sub["status_dt"] >= cutoff_ts)
    )
    open_c_records = v_sub[open_mask & (v_sub["class"] == "C")].copy()
    open_c = open_c_records.groupby("clean_bbl").size()
    open_b_records = v_sub[open_mask & (v_sub["class"] == "B")].copy()
    open_b = open_b_records.groupby("clean_bbl").size()
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
        # Statutory cure delinquency duration tracking past mandatory deadline (correctbydate)
        if "correct_dt" in open_c_records.columns:
            c_past_due = open_c_records[
                open_c_records["correct_dt"].notna() & (open_c_records["correct_dt"] < cutoff_ts)
            ].copy()
            if len(c_past_due) > 0:
                c_past_due["delinq_days"] = (
                    (cutoff_ts - c_past_due["correct_dt"]).dt.total_seconds() / 86400.0
                ).clip(lower=0.0)
                c_delinq_mean = c_past_due.groupby("clean_bbl")["delinq_days"].mean()
                c_delinq_max = c_past_due.groupby("clean_bbl")["delinq_days"].max()
                c_delinq_chronic = (
                    c_past_due[c_past_due["delinq_days"] > 30.0]
                    .groupby("clean_bbl")
                    .size()
                )
            else:
                c_delinq_mean = pd.Series(dtype=np.float32)
                c_delinq_max = pd.Series(dtype=np.float32)
                c_delinq_chronic = pd.Series(dtype=np.float32)
        else:
            c_delinq_mean = pd.Series(dtype=np.float32)
            c_delinq_max = pd.Series(dtype=np.float32)
            c_delinq_chronic = pd.Series(dtype=np.float32)
    else:
        open_c_dwell_mean = pd.Series(dtype=np.float32)
        open_c_dwell_max = pd.Series(dtype=np.float32)
        chronic_open_c = pd.Series(dtype=np.float32)
        c_delinq_mean = pd.Series(dtype=np.float32)
        c_delinq_max = pd.Series(dtype=np.float32)
        c_delinq_chronic = pd.Series(dtype=np.float32)

    # Open Class B dwell time statistics
    if len(open_b_records) > 0:
        open_b_records["dwell_days"] = (
            cutoff_ts - open_b_records["insp_dt"]
        ).dt.days.clip(lower=0)
        open_b_dwell_mean = open_b_records.groupby("clean_bbl")["dwell_days"].mean()
        chronic_open_b = open_b_records[open_b_records["dwell_days"] > 90].groupby(
            "clean_bbl"
        ).size()
    else:
        open_b_dwell_mean = pd.Series(dtype=np.float32)
        chronic_open_b = pd.Series(dtype=np.float32)

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

    # Exponential decay Class C hazard features
    for hl in [30, 90, 180, 365]:
        res_df[f"viol_c_exp_decay_{hl}d"] = (
            bbl_series.map(c_decay_map[hl]).fillna(0.0).astype(np.float32)
        )
    res_df["viol_c_decay_velocity"] = (
        res_df["viol_c_exp_decay_30d"] / (res_df["viol_c_exp_decay_365d"] / 12.0 + 1e-4)
    ).astype(np.float32)

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
    for hl in [30, 90, 180, 365]:
        res_df[f"viol_heat_exp_decay_{hl}d"] = (
            bbl_series.map(heat_decay_map[hl]).fillna(0.0).astype(np.float32)
        )

    res_df["viol_c_lead_1y"] = bbl_series.map(c_lead_1y).fillna(0).astype(np.float32)
    res_df["viol_c_lead_3y"] = bbl_series.map(c_lead_3y).fillna(0).astype(np.float32)
    res_df["viol_c_lead_life"] = bbl_series.map(c_lead_life).fillna(0).astype(np.float32)

    res_df["viol_c_mold_1y"] = bbl_series.map(c_mold_1y).fillna(0).astype(np.float32)
    res_df["viol_c_mold_3y"] = bbl_series.map(c_mold_3y).fillna(0).astype(np.float32)
    res_df["viol_c_mold_life"] = bbl_series.map(c_mold_life).fillna(0).astype(np.float32)

    res_df["viol_c_alarm_1y"] = bbl_series.map(c_alarm_1y).fillna(0).astype(np.float32)
    res_df["viol_c_alarm_3y"] = bbl_series.map(c_alarm_3y).fillna(0).astype(np.float32)
    res_df["viol_c_door_1y"] = bbl_series.map(c_door_1y).fillna(0).astype(np.float32)
    res_df["viol_c_door_life"] = bbl_series.map(c_door_life).fillna(0).astype(np.float32)

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
    res_df["delinq_c_mean_days"] = bbl_series.map(c_delinq_mean).fillna(0.0).astype(np.float32)
    res_df["delinq_c_max_days"] = bbl_series.map(c_delinq_max).fillna(0.0).astype(np.float32)
    res_df["delinq_c_chronic_cnt"] = bbl_series.map(c_delinq_chronic).fillna(0.0).astype(np.float32)

    res_df["viol_b_30d"] = bbl_series.map(b_30).fillna(0).astype(np.float32)
    res_df["viol_b_90d"] = bbl_series.map(b_90).fillna(0).astype(np.float32)
    res_df["viol_b_180d"] = bbl_series.map(b_180).fillna(0).astype(np.float32)
    res_df["viol_b_1y"] = bbl_series.map(b_1y).fillna(0).astype(np.float32)
    res_df["viol_b_2y"] = bbl_series.map(b_2y).fillna(0).astype(np.float32)
    res_df["viol_b_3y"] = bbl_series.map(b_3y).fillna(0).astype(np.float32)
    res_df["viol_b_life"] = bbl_series.map(b_life).fillna(0).astype(np.float32)
    for hl in [30, 90, 180, 365]:
        res_df[f"viol_b_exp_decay_{hl}d"] = (
            bbl_series.map(b_decay_map[hl]).fillna(0.0).astype(np.float32)
        )
    res_df["viol_b_accel_90d"] = (
        res_df["viol_b_90d"] / (res_df["viol_b_1y"] / 4.0 + 1e-4)
    ).astype(np.float32)

    res_df["viol_total_30d"] = bbl_series.map(tot_30).fillna(0).astype(np.float32)
    res_df["viol_total_90d"] = bbl_series.map(tot_90).fillna(0).astype(np.float32)
    res_df["viol_total_180d"] = bbl_series.map(tot_180).fillna(0).astype(np.float32)
    res_df["viol_total_1y"] = bbl_series.map(tot_1y).fillna(0).astype(np.float32)
    res_df["viol_total_3y"] = bbl_series.map(tot_3y).fillna(0).astype(np.float32)
    res_df["viol_total_life"] = bbl_series.map(tot_life).fillna(0).astype(np.float32)

    res_df["severe_hazard_ratio_1y"] = (
        (res_df["viol_c_1y"] + res_df["viol_b_1y"]) / (res_df["viol_total_1y"] + 1e-4)
    ).clip(0.0, 1.0).astype(np.float32)
    res_df["severe_hazard_ratio_3y"] = (
        (res_df["viol_c_3y"] + res_df["viol_b_3y"]) / (res_df["viol_total_3y"] + 1e-4)
    ).clip(0.0, 1.0).astype(np.float32)

    res_df["viol_a_1y"] = bbl_series.map(a_1y).fillna(0).astype(np.float32)
    res_df["open_viol_c"] = bbl_series.map(open_c).fillna(0).astype(np.float32)
    res_df["open_viol_b"] = bbl_series.map(open_b).fillna(0).astype(np.float32)
    res_df["open_viol_all"] = bbl_series.map(open_all).fillna(0).astype(np.float32)

    res_df["open_b_dwell_mean"] = (
        bbl_series.map(open_b_dwell_mean).fillna(0).astype(np.float32)
    )
    res_df["chronic_open_viol_b"] = (
        bbl_series.map(chronic_open_b).fillna(0).astype(np.float32)
    )
    res_df["open_b_per_unit"] = (res_df["open_viol_b"] / (units + 1e-4)).astype(np.float32)
    res_df["viol_b_per_unit_1y"] = (res_df["viol_b_1y"] / (units + 1e-4)).astype(np.float32)
    res_df["open_hazard_ratio_b_to_c"] = (
        (res_df["open_viol_b"] + 0.1) / (res_df["open_viol_c"] + 0.1)
    ).astype(np.float32)
    res_df["open_hazard_ratio_c_to_b"] = (
        (res_df["open_viol_c"] + 0.1) / (res_df["open_viol_b"] + 0.1)
    ).astype(np.float32)
    res_df["total_open_hazard_bc"] = (
        res_df["open_viol_c"] + res_df["open_viol_b"]
    ).astype(np.float32)

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
    res_df["insp_viol_c_hit_rate_1y"] = (
        res_df["viol_c_1y"] / (res_df["insp_encounters_1y"] + 1.0)
    ).astype(np.float32)
    res_df["insp_viol_c_hit_rate_3y"] = (
        res_df["viol_c_3y"] / (res_df["insp_encounters_3y"] + 1.0)
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

    # 2. HPD Complaints & Q4 Heat Season Emergency Dynamics
    cmp_sub = df_comp[df_comp["dt"] < cutoff_ts]
    dt_14d = cutoff_ts - pd.Timedelta(days=14)
    dt_92d = cutoff_ts - pd.Timedelta(days=92)  # Statutory Q4 pre-cutoff window (Oct 1 to Dec 31)
    cmp_sub_14d = cmp_sub[cmp_sub["dt"] >= dt_14d]
    cmp_sub_30d = cmp_sub[cmp_sub["dt"] >= dt_30d]
    cmp_sub_90d = cmp_sub[cmp_sub["dt"] >= dt_90d]
    cmp_sub_92d = cmp_sub[cmp_sub["dt"] >= dt_92d]

    cmp_14 = cmp_sub_14d.groupby("clean_bbl").size()
    cmp_30 = cmp_sub_30d.groupby("clean_bbl").size()
    cmp_90 = cmp_sub_90d.groupby("clean_bbl").size()
    cmp_1y = cmp_sub[cmp_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
    cmp_2y = cmp_sub[cmp_sub["dt"] >= dt_2y].groupby("clean_bbl").size()
    cmp_3y = cmp_sub[cmp_sub["dt"] >= dt_3y].groupby("clean_bbl").size()
    cmp_last = cmp_sub.groupby("clean_bbl")["dt"].max()

    # Complaint persistence: distinct filing dates in 30d and 90d
    cmp_30_dt = cmp_sub_30d[["clean_bbl", "dt"]].copy()
    cmp_30_dt["day"] = cmp_30_dt["dt"].dt.floor("D")
    cmp_dates_30 = cmp_30_dt.drop_duplicates(subset=["clean_bbl", "day"]).groupby("clean_bbl").size()

    cmp_90_dt = cmp_sub_90d[["clean_bbl", "dt"]].copy()
    cmp_90_dt["day"] = cmp_90_dt["dt"].dt.floor("D")
    cmp_dates_90 = cmp_90_dt.drop_duplicates(subset=["clean_bbl", "day"]).groupby("clean_bbl").size()

    # Continuous exponential time-decay kernels for tenant complaints
    delta_days_cmp = (
        (cutoff_ts - cmp_sub["dt"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).values
    cmp_decay_map = {}
    for hl in [30, 90, 180, 365]:
        s_cmp = pd.Series(np.exp(-delta_days_cmp * (ln2 / hl)), index=cmp_sub["clean_bbl"])
        cmp_decay_map[hl] = s_cmp.groupby(level=0).sum()

    res_df["comp_14d"] = bbl_series.map(cmp_14).fillna(0).astype(np.float32)
    res_df["comp_30d"] = bbl_series.map(cmp_30).fillna(0).astype(np.float32)
    res_df["comp_90d"] = bbl_series.map(cmp_90).fillna(0).astype(np.float32)
    res_df["comp_1y"] = bbl_series.map(cmp_1y).fillna(0).astype(np.float32)
    res_df["comp_2y"] = bbl_series.map(cmp_2y).fillna(0).astype(np.float32)
    res_df["comp_3y"] = bbl_series.map(cmp_3y).fillna(0).astype(np.float32)
    res_df["comp_distinct_days_30d"] = bbl_series.map(cmp_dates_30).fillna(0).astype(np.float32)
    res_df["comp_distinct_days_90d"] = bbl_series.map(cmp_dates_90).fillna(0).astype(np.float32)
    res_df["comp_persistence_ratio_90d"] = (
        res_df["comp_distinct_days_90d"] / (res_df["comp_90d"] + 1e-4)
    ).clip(0.0, 1.0).astype(np.float32)
    for hl in [30, 90, 180, 365]:
        res_df[f"comp_exp_decay_{hl}d"] = (
            bbl_series.map(cmp_decay_map[hl]).fillna(0.0).astype(np.float32)
        )
    res_df["comp_decay_velocity"] = (
        res_df["comp_exp_decay_30d"] / (res_df["comp_exp_decay_365d"] / 12.0 + 1e-4)
    ).astype(np.float32)
    last_dt_cmp = bbl_series.map(cmp_last)
    res_df["days_since_comp"] = (
        (cutoff_ts - last_dt_cmp).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
    )
    res_df["comp_per_unit_1y"] = (res_df["comp_1y"] / (units + 1e-4)).astype(np.float32)
    res_df["comp_accel_90d"] = (
        res_df["comp_90d"] / (res_df["comp_1y"] / 4.0 + 1e-4)
    ).astype(np.float32)

    # Pending inspection dispatch queue: complaints filed after latest inspection
    dt_60d = cutoff_ts - pd.Timedelta(days=60)
    cmp_sub_60d = cmp_sub[cmp_sub["dt"] >= dt_60d].copy()
    if len(cmp_sub_60d) > 0:
        last_insp_mapped = cmp_sub_60d["clean_bbl"].map(all_last)
        is_post_insp = last_insp_mapped.isna() | (cmp_sub_60d["dt"] > last_insp_mapped)
        cmp_post_60 = cmp_sub_60d[is_post_insp]
        comp_post_60_cnt = cmp_post_60.groupby("clean_bbl").size()
        comp_post_30_cnt = cmp_post_60[cmp_post_60["dt"] >= dt_30d].groupby("clean_bbl").size()
        res_df["comp_post_insp_30d"] = bbl_series.map(comp_post_30_cnt).fillna(0).astype(np.float32)
        res_df["comp_post_insp_60d"] = bbl_series.map(comp_post_60_cnt).fillna(0).astype(np.float32)
    else:
        res_df["comp_post_insp_30d"] = np.float32(0.0)
        res_df["comp_post_insp_60d"] = np.float32(0.0)
    res_df["has_uninspected_comp"] = (res_df["comp_post_insp_60d"] > 0).astype(np.float32)

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
        cmp_heat_14 = cmp_heat[cmp_heat["dt"] >= dt_14d].groupby("clean_bbl").size()
        cmp_heat_90 = cmp_heat[cmp_heat["dt"] >= dt_90d].groupby("clean_bbl").size()
        cmp_heat_92 = cmp_heat[cmp_heat["dt"] >= dt_92d].groupby("clean_bbl").size()
        cmp_heat_1y = cmp_heat[cmp_heat["dt"] >= dt_1y].groupby("clean_bbl").size()
        res_df["comp_heat_14d"] = (
            bbl_series.map(cmp_heat_14).fillna(0).astype(np.float32)
        )
        res_df["comp_heat_90d"] = (
            bbl_series.map(cmp_heat_90).fillna(0).astype(np.float32)
        )
        res_df["comp_heat_92d"] = (
            bbl_series.map(cmp_heat_92).fillna(0).astype(np.float32)
        )
        res_df["comp_heat_1y"] = (
            bbl_series.map(cmp_heat_1y).fillna(0).astype(np.float32)
        )
        res_df["heat_comp_velocity"] = (
            res_df["comp_heat_14d"] / (res_df["comp_heat_92d"] / 6.5 + 1e-4)
        ).astype(np.float32)
    else:
        res_df["comp_heat_14d"] = np.float32(0.0)
        res_df["comp_heat_90d"] = np.float32(0.0)
        res_df["comp_heat_92d"] = np.float32(0.0)
        res_df["comp_heat_1y"] = np.float32(0.0)
        res_df["heat_comp_velocity"] = np.float32(0.0)

    # Granular Complaint Category Volumes (plumbing/leaks, paint/plaster, door/window safety)
    cmp_1y_sub = cmp_sub[cmp_sub["dt"] >= dt_1y]
    cmp_3y_sub = cmp_sub[cmp_sub["dt"] >= dt_3y]

    if "is_plumb_leak" in cmp_sub.columns:
        plumb_1y = cmp_1y_sub[cmp_1y_sub["is_plumb_leak"]].groupby("clean_bbl").size()
        plumb_3y = cmp_3y_sub[cmp_3y_sub["is_plumb_leak"]].groupby("clean_bbl").size()
        res_df["comp_plumb_1y"] = bbl_series.map(plumb_1y).fillna(0).astype(np.float32)
        res_df["comp_plumb_3y"] = bbl_series.map(plumb_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_plumb_1y"] = np.float32(0.0)
        res_df["comp_plumb_3y"] = np.float32(0.0)

    if "is_paint_plaster" in cmp_sub.columns:
        paint_1y = cmp_1y_sub[cmp_1y_sub["is_paint_plaster"]].groupby("clean_bbl").size()
        paint_3y = cmp_3y_sub[cmp_3y_sub["is_paint_plaster"]].groupby("clean_bbl").size()
        res_df["comp_paint_1y"] = bbl_series.map(paint_1y).fillna(0).astype(np.float32)
        res_df["comp_paint_3y"] = bbl_series.map(paint_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_paint_1y"] = np.float32(0.0)
        res_df["comp_paint_3y"] = np.float32(0.0)

    if "is_door_window" in cmp_sub.columns:
        door_1y = cmp_1y_sub[cmp_1y_sub["is_door_window"]].groupby("clean_bbl").size()
        door_3y = cmp_3y_sub[cmp_3y_sub["is_door_window"]].groupby("clean_bbl").size()
        res_df["comp_door_window_1y"] = bbl_series.map(door_1y).fillna(0).astype(np.float32)
        res_df["comp_door_window_3y"] = bbl_series.map(door_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_door_window_1y"] = np.float32(0.0)
        res_df["comp_door_window_3y"] = np.float32(0.0)

    if "is_pest_comp" in cmp_sub.columns:
        pest_1y = cmp_1y_sub[cmp_1y_sub["is_pest_comp"]].groupby("clean_bbl").size()
        pest_3y = cmp_3y_sub[cmp_3y_sub["is_pest_comp"]].groupby("clean_bbl").size()
        res_df["comp_pest_1y"] = bbl_series.map(pest_1y).fillna(0).astype(np.float32)
        res_df["comp_pest_3y"] = bbl_series.map(pest_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_pest_1y"] = np.float32(0.0)
        res_df["comp_pest_3y"] = np.float32(0.0)

    if "is_safety_comp" in cmp_sub.columns:
        sft_1y = cmp_1y_sub[cmp_1y_sub["is_safety_comp"]].groupby("clean_bbl").size()
        sft_3y = cmp_3y_sub[cmp_3y_sub["is_safety_comp"]].groupby("clean_bbl").size()
        res_df["comp_safety_1y"] = bbl_series.map(sft_1y).fillna(0).astype(np.float32)
        res_df["comp_safety_3y"] = bbl_series.map(sft_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_safety_1y"] = np.float32(0.0)
        res_df["comp_safety_3y"] = np.float32(0.0)

    # Unique Complaining Apartments & Diffusion Ratios
    has_apt = "clean_apt" in cmp_sub.columns and cmp_sub["clean_apt"].notna().any()
    if has_apt:
        cmp_apt_1y = cmp_1y_sub.dropna(subset=["clean_apt"])
        cmp_apt_3y = cmp_3y_sub.dropna(subset=["clean_apt"])
        apts_1y = cmp_apt_1y.groupby("clean_bbl")["clean_apt"].nunique()
        apts_3y = cmp_apt_3y.groupby("clean_bbl")["clean_apt"].nunique()
        res_df["comp_unique_apts_1y"] = bbl_series.map(apts_1y).fillna(0).astype(np.float32)
        res_df["comp_unique_apts_3y"] = bbl_series.map(apts_3y).fillna(0).astype(np.float32)
    else:
        res_df["comp_unique_apts_1y"] = np.minimum(res_df["comp_1y"], units).astype(np.float32)
        res_df["comp_unique_apts_3y"] = np.minimum(res_df["comp_3y"], units).astype(np.float32)

    res_df["comp_diffusion_ratio_1y"] = (
        res_df["comp_unique_apts_1y"] / (units + 1e-4)
    ).clip(upper=5.0).astype(np.float32)
    res_df["comp_diffusion_ratio_3y"] = (
        res_df["comp_unique_apts_3y"] / (units + 1e-4)
    ).clip(upper=5.0).astype(np.float32)

    # Trailing Complaint-to-Violation Conversion Efficiency
    res_df["comp_to_viol_c_conversion_1y"] = (
        res_df["viol_c_1y"] / (res_df["comp_1y"] + 1.0)
    ).astype(np.float32)
    res_df["comp_to_viol_c_conversion_3y"] = (
        res_df["viol_c_3y"] / (res_df["comp_3y"] + 1.0)
    ).astype(np.float32)

    # Composite tenement hazard pressure ratios
    res_df["open_c_to_comp_90d_ratio"] = (
        res_df["open_viol_c"] / (res_df["comp_90d"] + 1.0)
    ).astype(np.float32)
    res_df["open_viol_all_per_unit"] = (
        res_df["open_viol_all"] / (res_df["unitsres"] + 1.0)
    ).astype(np.float32)
    res_df["open_viol_c_per_floor"] = (
        res_df["open_viol_c"] / (res_df["numfloors"] + 1.0)
    ).astype(np.float32)
    res_df["viol_c_per_floor_density"] = (
        res_df["viol_c_1y"] / (res_df["numfloors"] + 1.0)
    ).astype(np.float32)

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

        # Statutory tenant HP actions and heat/hot water court orders
        lit_hp = lit_sub[lit_sub.get("is_hp", False)] if "is_hp" in lit_sub.columns else pd.DataFrame()
        if len(lit_hp) > 0:
            lit_hp_1y = lit_hp[lit_hp["dt"] >= dt_1y].groupby("clean_bbl").size()
            lit_hp_life = lit_hp.groupby("clean_bbl").size()
            res_df["hpd_lit_hp_1y"] = bbl_series.map(lit_hp_1y).fillna(0).astype(np.float32)
            res_df["hpd_lit_hp_life"] = bbl_series.map(lit_hp_life).fillna(0).astype(np.float32)
        else:
            res_df["hpd_lit_hp_1y"] = np.float32(0.0)
            res_df["hpd_lit_hp_life"] = np.float32(0.0)

        lit_heat = lit_sub[lit_sub.get("is_heat_lit", False)] if "is_heat_lit" in lit_sub.columns else pd.DataFrame()
        if len(lit_heat) > 0:
            lit_heat_1y = lit_heat[lit_heat["dt"] >= dt_1y].groupby("clean_bbl").size()
            lit_heat_life = lit_heat.groupby("clean_bbl").size()
            res_df["hpd_lit_heat_1y"] = bbl_series.map(lit_heat_1y).fillna(0).astype(np.float32)
            res_df["hpd_lit_heat_life"] = bbl_series.map(lit_heat_life).fillna(0).astype(np.float32)
        else:
            res_df["hpd_lit_heat_1y"] = np.float32(0.0)
            res_df["hpd_lit_heat_life"] = np.float32(0.0)
    else:
        res_df["hpd_lit_1y"] = np.float32(0.0)
        res_df["hpd_lit_life"] = np.float32(0.0)
        res_df["hpd_lit_hp_1y"] = np.float32(0.0)
        res_df["hpd_lit_hp_life"] = np.float32(0.0)
        res_df["hpd_lit_heat_1y"] = np.float32(0.0)
        res_df["hpd_lit_heat_life"] = np.float32(0.0)

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

    # 8. DOF Annualized Property Sales Distress Indicators
    if len(df_sales) > 0 and "dt" in df_sales.columns and df_sales["dt"].notna().any():
        sales_sub = df_sales[df_sales["dt"] < cutoff_ts]
        if len(sales_sub) > 0:
            last_sales = sales_sub.sort_values("dt").groupby("clean_bbl").last()
            sales_last_dt = last_sales["dt"]
            sales_last_price = last_sales["price"]
            sales_cnt_1y = sales_sub[sales_sub["dt"] >= dt_1y].groupby("clean_bbl").size()
            sales_cnt_2y = sales_sub[sales_sub["dt"] >= dt_2y].groupby("clean_bbl").size()
            distressed_sales = sales_sub[(sales_sub["dt"] >= dt_2y) & (sales_sub["price"] > 0) & (sales_sub["price"] < 1000.0)].groupby("clean_bbl").size()

            last_sale_date_mapped = bbl_series.map(sales_last_dt)
            res_df["days_since_sale"] = (
                (cutoff_ts - last_sale_date_mapped).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
            )
            res_df["sale_1y"] = (bbl_series.map(sales_cnt_1y).fillna(0) > 0).astype(np.float32)
            res_df["sale_2y"] = (bbl_series.map(sales_cnt_2y).fillna(0) > 0).astype(np.float32)
            res_df["sale_price_log"] = np.log1p(
                bbl_series.map(sales_last_price).fillna(0.0).clip(lower=0.0)
            ).astype(np.float32)
            res_df["sale_distressed_flag"] = (
                bbl_series.map(distressed_sales).fillna(0) > 0
            ).astype(np.float32)
        else:
            res_df["days_since_sale"] = np.float32(3650.0)
            res_df["sale_1y"] = np.float32(0.0)
            res_df["sale_2y"] = np.float32(0.0)
            res_df["sale_price_log"] = np.float32(0.0)
            res_df["sale_distressed_flag"] = np.float32(0.0)
    else:
        res_df["days_since_sale"] = np.float32(3650.0)
        res_df["sale_1y"] = np.float32(0.0)
        res_df["sale_2y"] = np.float32(0.0)
        res_df["sale_price_log"] = np.float32(0.0)
        res_df["sale_distressed_flag"] = np.float32(0.0)

    # 9. Statutory Distress Programs
    res_df["flag_aep"] = bbl_series.isin(aep_bbl_set).astype(np.float32)
    res_df["flag_conh"] = bbl_series.isin(conh_bbl_set).astype(np.float32)
    res_df["flag_speculation"] = bbl_series.isin(spec_bbl_set).astype(np.float32)
    if len(df_underlying) > 0 and "dt" in df_underlying.columns and df_underlying["dt"].notna().any():
        und_sub = df_underlying[df_underlying["dt"] < cutoff_ts]
        res_df["flag_underlying_conditions"] = bbl_series.isin(set(und_sub["clean_bbl"].unique())).astype(np.float32)
    else:
        res_df["flag_underlying_conditions"] = bbl_series.isin(underlying_bbl_set).astype(np.float32)

    # 10. Authentic Landlord Portfolio Risk from HPD Registrations & Contacts
    reg_ids = bbl_series.map(bbl_to_reg).fillna(0).astype(np.int64)
    landlord_ids = bbl_series.map(bbl_to_landlord).fillna("")
    has_landlord = (landlord_ids != "") & (landlord_ids != "REG_0")
    res_df["is_unregistered_md"] = ((units >= 3) & (reg_ids == 0)).astype(np.float32)

    # Statutory registration delinquency features
    if len(df_reg) > 0 and "reg_dt" in df_reg.columns and df_reg["reg_dt"].notna().any():
        reg_sub = df_reg[df_reg["reg_dt"].notna() & (df_reg["reg_dt"] < cutoff_ts)]
        if len(reg_sub) > 0:
            reg_last = reg_sub.groupby("clean_bbl")["reg_dt"].max()
            reg_last_mapped = bbl_series.map(reg_last)
            res_df["days_since_last_registration"] = (
                (cutoff_ts - reg_last_mapped).dt.days.fillna(3650).clip(0, 3650).astype(np.float32)
            )
            res_df["is_registration_overdue_flag"] = (
                (units >= 3) & ((res_df["days_since_last_registration"] > 365) | reg_last_mapped.isna())
            ).astype(np.float32)
        else:
            res_df["days_since_last_registration"] = np.float32(3650.0)
            res_df["is_registration_overdue_flag"] = (units >= 3).astype(np.float32)
    else:
        res_df["days_since_last_registration"] = np.float32(3650.0)
        res_df["is_registration_overdue_flag"] = (units >= 3).astype(np.float32)

    # Citywide authentic portfolio size and leave-one-out size
    city_port_size = landlord_ids.map(landlord_to_port_size).fillna(0).astype(np.float32)
    res_df["landlord_portfolio_size"] = np.where(has_landlord, city_port_size, 0.0).astype(np.float32)
    res_df["portfolio_loo_size"] = np.where(has_landlord, np.maximum(0.0, city_port_size - 1.0), 0.0).astype(np.float32)

    # Within-cohort authentic portfolio Class C aggregation & leave-one-out co-owned Class C rates
    port_c_1y_sum = res_df.groupby(landlord_ids)["viol_c_1y"].transform("sum")
    port_bbl_cnt = res_df.groupby(landlord_ids)["viol_c_1y"].transform("count")
    res_df["portfolio_viol_c_1y"] = np.where(has_landlord, port_c_1y_sum, 0.0).astype(np.float32)
    res_df["portfolio_loo_c_rate"] = np.where(
        has_landlord & (port_bbl_cnt > 1),
        (port_c_1y_sum - res_df["viol_c_1y"]) / np.maximum(port_bbl_cnt - 1.0, 1.0),
        0.0,
    ).astype(np.float32)

    port_c_3y_sum = res_df.groupby(landlord_ids)["viol_c_3y"].transform("sum")
    res_df["portfolio_loo_c_3y_rate"] = np.where(
        has_landlord & (port_bbl_cnt > 1),
        (port_c_3y_sum - res_df["viol_c_3y"]) / np.maximum(port_bbl_cnt - 1.0, 1.0),
        0.0,
    ).astype(np.float32)

    # 11. Spatial Tax Block Disambiguation: group on 6-digit borough-block prefix (bbl[:6])
    boro_block_prefix = res_df["bbl"].str[:6]
    blk_sum = res_df.groupby(boro_block_prefix)["viol_c_1y"].transform("sum")
    blk_cnt = res_df.groupby(boro_block_prefix)["viol_c_1y"].transform("count")
    res_df["block_c_density"] = np.where(
        blk_cnt > 1,
        ((blk_sum - res_df["viol_c_1y"]) / np.maximum(blk_cnt - 1.0, 1.0)).clip(lower=0),
        0.0,
    ).astype(np.float32)

    zip_sum = res_df.groupby("zipcode")["viol_c_1y"].transform("sum")
    zip_cnt = res_df.groupby("zipcode")["viol_c_1y"].transform("count")
    zip_risk = ((zip_sum - res_df["viol_c_1y"]) / np.maximum(zip_cnt - 1.0, 1.0)).clip(lower=0)
    zip_risk = np.where(res_df["zipcode"] == 0, 0.0, zip_risk)
    res_df["zip_c_density"] = zip_risk.astype(np.float32)

    zip_units_sum = res_df.groupby("zipcode")["unitsres"].transform("sum")
    loo_zip_units = np.maximum(zip_units_sum - res_df["unitsres"], 1.0)
    zip_c_per_unit = ((zip_sum - res_df["viol_c_1y"]) / loo_zip_units).clip(lower=0)
    zip_c_per_unit = np.where(res_df["zipcode"] == 0, 0.0, zip_c_per_unit)
    res_df["zip_viol_c_per_unit"] = zip_c_per_unit.astype(np.float32)

    # 12. Community District (cd) leave-one-out administrative hazard densities per residential unit
    cd_c_sum = res_df.groupby("cd")["viol_c_1y"].transform("sum")
    cd_units_sum = res_df.groupby("cd")["unitsres"].transform("sum")
    cd_comp_sum = res_df.groupby("cd")["comp_1y"].transform("sum")
    cd_cnt = res_df.groupby("cd")["viol_c_1y"].transform("count")

    loo_cd_units = np.maximum(cd_units_sum - res_df["unitsres"], 1.0)
    loo_cd_c_rate = np.maximum(0.0, cd_c_sum - res_df["viol_c_1y"]) / loo_cd_units
    res_df["cd_viol_c_per_unit"] = np.where(res_df["cd"] > 0, loo_cd_c_rate, 0.0).astype(np.float32)

    loo_cd_comp_rate = np.maximum(0.0, cd_comp_sum - res_df["comp_1y"]) / loo_cd_units
    res_df["cd_comp_per_unit"] = np.where(res_df["cd"] > 0, loo_cd_comp_rate, 0.0).astype(np.float32)

    loo_cd_lot_cnt = np.maximum(cd_cnt - 1.0, 1.0)
    loo_cd_c_lot_rate = np.maximum(0.0, cd_c_sum - res_df["viol_c_1y"]) / loo_cd_lot_cnt
    res_df["cd_viol_c_lot_density"] = np.where(res_df["cd"] > 0, loo_cd_c_lot_rate, 0.0).astype(np.float32)

    # 13. DOF Tax Lien Sales (Municipal tax delinquency)
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

    # Drop non-feature join columns and cast cleanly to float32
    res_df = res_df.drop(
        columns=["bbl", "block", "zipcode", "cd", "landlord_reg_id"], errors="ignore"
    )
    return res_df.fillna(0.0).astype(np.float32)


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
def safe_predict_scores(model, X):
    """Extract monotonic continuous 1D ranking scores in [0, 1]."""
    try:
        proba = model.predict_proba(X)
        if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] >= 2:
            p = proba[:, 1].astype(np.float32)
            if p.min() >= 0.0 and p.max() <= 1.0 and (p.max() - p.min()) > 0:
                return p
            return (1.0 / (1.0 + np.exp(-np.clip(p, -30.0, 30.0)))).astype(np.float32)
        proba = np.asarray(proba, dtype=np.float32).ravel()
        if proba.min() >= 0.0 and proba.max() <= 1.0 and (proba.max() - proba.min()) > 0:
            return proba
        return (1.0 / (1.0 + np.exp(-np.clip(proba, -30.0, 30.0)))).astype(np.float32)
    except Exception:
        pass
    try:
        raw = model.predict(X, output_margin=True)
        raw = np.asarray(raw, dtype=np.float32).ravel()
        return (1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))).astype(np.float32)
    except Exception:
        pass
    raw = model.predict(X)
    raw = np.asarray(raw, dtype=np.float32).ravel()
    return (1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))).astype(np.float32)


val_preds = []
test_preds = []
model_names = []

# --- Model 1: Balanced Depth-Limited LightGBM ---
print("\n[1/6] Training Balanced Depth-Limited LightGBM with Fast Logloss Early Stopping...")
t0 = time.time()
lgb_balanced = lgb.LGBMClassifier(
    objective="binary",
    eval_metric="binary_logloss",
    boosting_type="gbdt",
    scale_pos_weight=1.20,
    n_estimators=1000,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=7,
    min_child_samples=40,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.60,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_balanced.fit(
    X_train_np,
    y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_np, y_val)],
    callbacks=[lgb.early_stopping(60, verbose=False)],
)
p_val_1 = safe_predict_scores(lgb_balanced, X_val_np)
p_test_1 = safe_predict_scores(lgb_balanced, X_test_np)
score_1 = average_precision_score(y_val, p_val_1)
print(f"LGBM Balanced Validation AP: {score_1:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_1)
test_preds.append(p_test_1)
model_names.append("LGBM_Balanced")

# --- Model 2: Extremely Randomized LightGBM ---
print("\n[2/6] Training Extremely Randomized LightGBM (ExtraTrees) with Fast Logloss Early Stopping...")
t0 = time.time()
lgb_et = lgb.LGBMClassifier(
    objective="binary",
    eval_metric="binary_logloss",
    boosting_type="gbdt",
    extra_trees=True,
    scale_pos_weight=1.20,
    n_estimators=950,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=7,
    min_child_samples=50,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.50,
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
    callbacks=[lgb.early_stopping(60, verbose=False)],
)
p_val_2 = safe_predict_scores(lgb_et, X_val_np)
p_test_2 = safe_predict_scores(lgb_et, X_test_np)
score_2 = average_precision_score(y_val, p_val_2)
print(
    f"LGBM ExtraTrees Validation AP: {score_2:.5f} (trained in {time.time()-t0:.1f}s)"
)
val_preds.append(p_val_2)
test_preds.append(p_test_2)
model_names.append("LGBM_ExtraTrees")

# --- Model 3: Deep Node-Subsampled Histogram XGBoost ---
print("\n[3/6] Training Deep Node-Subsampled Histogram XGBoost with PR-AUC Objective...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=1.20,
    n_estimators=950,
    learning_rate=0.035,
    max_depth=8,
    colsample_bynode=0.75,
    colsample_bytree=0.60,
    subsample=0.80,
    reg_alpha=1.0,
    reg_lambda=5.0,
    early_stopping_rounds=60,
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
p_val_3 = safe_predict_scores(xgb_model, X_val_np)
p_test_3 = safe_predict_scores(xgb_model, X_test_np)
score_3 = average_precision_score(y_val, p_val_3)
print(f"XGBoost Validation AP: {score_3:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_3)
test_preds.append(p_test_3)
model_names.append("XGB_Hist")

# --- Model 4: Subspace Symmetric Oblivious CatBoost ---
print("\n[4/6] Training Subspace Symmetric Oblivious CatBoost with PR-AUC Metric...")
t0 = time.time()
cb_model = CatBoostClassifier(
    iterations=850,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=4.0,
    loss_function="Logloss",
    eval_metric="PRAUC",
    scale_pos_weight=1.15,
    rsm=0.80,
    subsample=0.85,
    bootstrap_type="Bernoulli",
    early_stopping_rounds=60,
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
p_val_4 = safe_predict_scores(cb_model, X_val_np)
p_test_4 = safe_predict_scores(cb_model, X_test_np)
score_4 = average_precision_score(y_val, p_val_4)
print(f"CatBoost Subspace Validation AP: {score_4:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_4)
test_preds.append(p_test_4)
model_names.append("CatBoost_Subspace")

# --- Model 5: Non-Greedy Tree-Dropout LightGBM (DART) ---
print("\n[5/6] Training Non-Greedy Tree-Dropout LightGBM (DART)...")
t0 = time.time()
lgb_dart = lgb.LGBMClassifier(
    objective="binary",
    eval_metric="binary_logloss",
    boosting_type="dart",
    scale_pos_weight=1.20,
    drop_rate=0.10,
    skip_drop=0.50,
    n_estimators=700,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=7,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.60,
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
p_val_5 = safe_predict_scores(lgb_dart, X_val_np)
p_test_5 = safe_predict_scores(lgb_dart, X_test_np)
score_5 = average_precision_score(y_val, p_val_5)
print(f"LGBM DART Validation AP: {score_5:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_5)
test_preds.append(p_test_5)
model_names.append("LGBM_DART")

# --- Model 6: Regularized Tabular DCN-v2 ResNet with Asymmetric Focal Loss ---
print("\n[6/6] Training Regularized Tabular DCN-v2 ResNet with Asymmetric Focal Loss...")
t0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Tabular ResNet compute device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

is_bin_col = []
for c_idx in range(X_train_np.shape[1]):
    u = np.unique(X_train_np[:, c_idx])
    if len(u) <= 2 and np.all(np.isin(u, [0.0, 1.0])):
        is_bin_col.append(True)
    else:
        is_bin_col.append(False)
is_bin_col = np.array(is_bin_col)
cont_idx = np.where(~is_bin_col)[0]
bin_idx = np.where(is_bin_col)[0]

# Zero-preserving Log1p continuous feature transformation followed by StandardScaler normalization
scaler = StandardScaler()
X_tr_cont_log = np.log1p(np.maximum(0.0, X_train_np[:, cont_idx]))
X_v_cont_log = np.log1p(np.maximum(0.0, X_val_np[:, cont_idx]))
X_te_cont_log = np.log1p(np.maximum(0.0, X_test_np[:, cont_idx]))

X_tr_cont = scaler.fit_transform(X_tr_cont_log).astype(np.float32)
X_v_cont = scaler.transform(X_v_cont_log).astype(np.float32)
X_te_cont = scaler.transform(X_te_cont_log).astype(np.float32)

X_train_scaled = np.hstack([X_tr_cont, X_train_np[:, bin_idx]]).astype(np.float32)
X_val_scaled = np.hstack([X_v_cont, X_val_np[:, bin_idx]]).astype(np.float32)
X_test_scaled = np.hstack([X_te_cont, X_test_np[:, bin_idx]]).astype(np.float32)


class CrossLayer(nn.Module):
    """Explicit DCN-v2 feature crossing layer: x_{l+1} = x_0 * (W * x_l + b) + x_l"""
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features, bias=True)

    def forward(self, x0, xl):
        return x0 * self.linear(xl) + xl


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """DCN-v2 ResNet combining degree-2 and degree-3 feature crossing with residual continuous pathways."""
    def __init__(self, in_features, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.cross1 = CrossLayer(in_features)
        self.cross2 = CrossLayer(in_features)
        self.input_layer = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x1 = self.cross1(x, x)
        x2 = self.cross2(x, x1)
        x_comb = torch.cat([x, x2], dim=-1)
        h = self.input_layer(x_comb)
        h = self.res1(h)
        h = self.res2(h)
        out = self.head(h)
        return out.squeeze(-1)


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal BCE Loss focusing neural parameter updates on difficult positive Class C cases."""
    def __init__(self, gamma=2.0, pos_weight=1.5, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        pos_loss = -self.pos_weight * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        neg_loss = -(probs ** self.gamma) * torch.log(1.0 - probs)
        return targets * pos_loss + (1.0 - targets) * neg_loss


tab_model = TabularResNet(in_features=X_train_scaled.shape[1], hidden_dim=256, dropout=0.25).to(device)

train_dataset = TensorDataset(
    torch.from_numpy(X_train_scaled),
    torch.from_numpy(y_train.astype(np.float32)),
    torch.from_numpy(sample_weights_train.astype(np.float32)),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

optimizer = optim.AdamW(tab_model.parameters(), lr=1.5e-3, weight_decay=1e-3)
epochs = 12
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = AsymmetricFocalLoss(gamma=2.0, pos_weight=1.5)

best_val_ap = 0.0
best_model_weights = None
patience = 4
patience_counter = 0

for epoch in range(1, epochs + 1):
    tab_model.train()
    total_loss = 0.0
    for bx, by, bw in train_loader:
        bx = bx.to(device)
        by = by.to(device)
        bw = bw.to(device)

        optimizer.zero_grad()
        out = tab_model(bx)
        loss = (criterion(out, by) * bw).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(tab_model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation evaluation after each epoch for checkpointing
    tab_model.eval()
    val_preds_epoch = []
    with torch.no_grad():
        for i in range(0, len(X_val_scaled), 8192):
            batch = torch.from_numpy(X_val_scaled[i : i + 8192]).to(device)
            p = torch.sigmoid(tab_model(batch)).cpu().numpy()
            val_preds_epoch.append(p)
    val_p_arr = np.concatenate(val_preds_epoch)
    epoch_ap = average_precision_score(y_val, val_p_arr)
    if epoch_ap > best_val_ap:
        best_val_ap = epoch_ap
        best_model_weights = {k: v.cpu().clone() for k, v in tab_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Tabular ResNet early stopping triggered at epoch {epoch}")
            break

if best_model_weights is not None:
    tab_model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

tab_model.eval()
with torch.no_grad():
    val_preds_list = []
    for i in range(0, len(X_val_scaled), 8192):
        batch = torch.from_numpy(X_val_scaled[i : i + 8192]).to(device)
        val_preds_list.append(torch.sigmoid(tab_model(batch)).cpu().numpy())
    p_val_6 = np.concatenate(val_preds_list).astype(np.float32)

    test_preds_list = []
    for i in range(0, len(X_test_scaled), 8192):
        batch = torch.from_numpy(X_test_scaled[i : i + 8192]).to(device)
        test_preds_list.append(torch.sigmoid(tab_model(batch)).cpu().numpy())
    p_test_6 = np.concatenate(test_preds_list).astype(np.float32)

score_6 = average_precision_score(y_val, p_val_6)
print(f"Tabular DCN ResNet Validation AP: {score_6:.5f} (trained in {time.time()-t0:.1f}s)")
val_preds.append(p_val_6)
test_preds.append(p_test_6)
model_names.append("Tabular_DCN_ResNet")

del X_train_scaled, X_val_scaled, X_test_scaled, train_loader, train_dataset
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 7. Simplex Rank Optimization on Validation AP
# ---------------------------------------------------------
print("\n--- Model Fleet Summary ---")
for name, sc in zip(model_names, [score_1, score_2, score_3, score_4, score_5, score_6]):
    print(f"  {name:20s}: {sc:.5f}")

# Convert predictions to fractional percentile ranks
val_ranks = np.array([rankdata(p) / len(p) for p in val_preds])
test_ranks = np.array([rankdata(p) / len(p) for p in test_preds])


def optimize_simplex_weights(ranks, y_true, n_dirichlet=3000, seed=42):
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

    # Multi-start Dirichlet exploration
    np.random.seed(seed)
    for alpha_val in [0.2, 0.4, 0.7, 1.0, 1.5, 2.5]:
        samples = np.random.dirichlet(np.full(n, alpha_val), size=n_dirichlet // 6)
        for cand_w in samples:
            score = average_precision_score(y_true, np.dot(cand_w, ranks))
            if score > best_score:
                best_score = score
                best_w = cand_w.copy()

    # Multi-scale fine-grained coordinate descent (step sizes 0.08 down to 0.001)
    for step in [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]:
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

import gc
import json
import math
import os
import warnings
import gcsfs
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. HARDWARE DETERMINISM & CLOUD STORAGE CONNECTION
# ---------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)

token_val = (
    TOKEN_PATH
    if os.path.exists(TOKEN_PATH)
    else os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
)
storage_options = {"token": token_val} if token_val else {}
fs = gcsfs.GCSFileSystem(token=token_val) if token_val else gcsfs.GCSFileSystem()

os.makedirs("submission", exist_ok=True)
os.makedirs("working", exist_ok=True)


# ---------------------------------------------------------
# 2. ROBUST BBL STANDARDIZATION & LAKE HELPER
# ---------------------------------------------------------
def clean_bbl_to_int(
    df, bbl_col="bbl", boro_col="boroid", block_col="block", lot_col="lot"
):
    """Vectorized conversion of heterogeneous NYC tax lot identifiers into int64."""
    n = len(df)
    res = np.full(n, -1, dtype=np.int64)

    if bbl_col in df.columns:
        bbl_num = pd.to_numeric(df[bbl_col], errors="coerce").fillna(-1).values
        valid_mask = (bbl_num >= 1000000000) & (bbl_num < 6000000000)
        res[valid_mask] = bbl_num[valid_mask].astype(np.int64)

    need_fallback = res == -1
    if need_fallback.any():
        actual_boro_col = None
        for cand in [boro_col, "boroid", "boro", "borocode", "borough"]:
            if cand in df.columns:
                actual_boro_col = cand
                break

        if (
            actual_boro_col is not None
            and block_col in df.columns
            and lot_col in df.columns
        ):
            b_vals = (
                pd.to_numeric(df[actual_boro_col], errors="coerce").fillna(-1).values
            )
            if np.all(b_vals == -1) and df[actual_boro_col].dtype == object:
                boro_map = {
                    "MANHATTAN": 1,
                    "MN": 1,
                    "BRONX": 2,
                    "BX": 2,
                    "BROOKLYN": 3,
                    "BK": 3,
                    "QUEENS": 4,
                    "QN": 4,
                    "STATEN ISLAND": 5,
                    "SI": 5,
                }
                b_vals = (
                    df[actual_boro_col]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .map(boro_map)
                    .fillna(-1)
                    .values
                )

            blk_vals = pd.to_numeric(df[block_col], errors="coerce").fillna(-1).values
            lot_vals = pd.to_numeric(df[lot_col], errors="coerce").fillna(-1).values

            fb_mask = (
                need_fallback
                & (b_vals >= 1)
                & (b_vals <= 5)
                & (blk_vals >= 0)
                & (blk_vals <= 99999)
                & (lot_vals >= 0)
                & (lot_vals <= 9999)
            )

            if fb_mask.any():
                res[fb_mask] = (
                    b_vals[fb_mask].astype(np.int64) * 1_000_000_000
                    + blk_vals[fb_mask].astype(np.int64) * 10_000
                    + lot_vals[fb_mask].astype(np.int64)
                )

    return res


def get_available_columns(gcs_table_path):
    path_no_gs = gcs_table_path.replace("gs://", "")
    dataset = ds.dataset(path_no_gs, filesystem=fs, format="parquet")
    return dataset.schema.names


# ---------------------------------------------------------
# 3. LOAD DATA FROM NYC LAKE (PLUTO & HPD VIOLATIONS)
# ---------------------------------------------------------
print("Loading PLUTO releases...")
pluto_path = f"{GCS_BASE}/lake/full/pluto"
pluto_avail_cols = get_available_columns(pluto_path)
desired_pluto_cols = [
    "bbl",
    "borough",
    "borocode",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "bldgclass",
    "landuse",
    "bldgarea",
    "resarea",
    "numfloors",
    "lotarea",
    "assessland",
    "assesstot",
    "zipcode",
    "cd",
    "version",
    "release",
    "pluto_version",
]
pluto_cols = [c for c in desired_pluto_cols if c in pluto_avail_cols]
df_pluto = pd.read_parquet(
    pluto_path, storage_options=storage_options, columns=pluto_cols
)
df_pluto["bbl_int"] = clean_bbl_to_int(df_pluto)
df_pluto = df_pluto[df_pluto["bbl_int"] > 0].copy()

version_col = next(
    (
        c
        for c in df_pluto.columns
        if c.lower() in ["version", "release", "pluto_version"]
    ),
    None,
)
if version_col is None:
    version_col = next(
        (c for c in df_pluto.columns if "ver" in c.lower() or "rel" in c.lower()),
        None,
    )

print("Loading test entities...")
df_test = pd.read_parquet(
    f"{GCS_BASE}/test_entities.parquet", storage_options=storage_options
)
df_test["bbl_int"] = clean_bbl_to_int(df_test)
df_test["bbl_str"] = df_test["bbl_int"].apply(lambda x: f"{x:010d}")

print("Loading HPD violations...")
viol_path = f"{GCS_BASE}/lake/full/hpd_violations"
viol_avail_cols = get_available_columns(viol_path)
desired_viol_cols = [
    "bbl",
    "boroid",
    "boro",
    "block",
    "lot",
    "class",
    "inspectiondate",
    "violationstatus",
]
viol_cols = [c for c in desired_viol_cols if c in viol_avail_cols]
df_viol = pd.read_parquet(viol_path, storage_options=storage_options, columns=viol_cols)
df_viol["bbl_int"] = clean_bbl_to_int(df_viol)
df_viol = df_viol[df_viol["bbl_int"] > 0].copy()
df_viol["insp_dt"] = pd.to_datetime(
    df_viol["inspectiondate"], errors="coerce", utc=True
).dt.tz_localize(None)
df_viol = df_viol[df_viol["insp_dt"].notna()].copy()
df_viol["class_clean"] = df_viol["class"].astype(str).str.strip().str.upper()
if "violationstatus" in df_viol.columns:
    df_viol["status_clean"] = df_viol["violationstatus"].fillna("").astype(str).str.strip().str.upper()
else:
    df_viol["status_clean"] = "OPEN"

# ---------------------------------------------------------
# 4. LOAD AUXILIARY MUNICIPAL DISTRESS DATA
# ---------------------------------------------------------
print("Loading auxiliary municipal distress indicators...")


def safe_load_aux_bbl(table_name, date_cands=None):
    try:
        t_path = f"{GCS_BASE}/lake/full/{table_name}"
        cols = get_available_columns(t_path)
        cols_norm_map = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
        target_keys = [
            "bbl",
            "boroid",
            "boro",
            "borocode",
            "borough",
            "block",
            "lot",
            "buildingid",
            "actualchargeamount",
            "chargeamount",
            "amount",
            "totalamount",
            "saleprice",
            "price",
            "grosssquarefeet",
            "grosssqft",
            "totalunits",
            "residentialunits",
            "infesteddwellingunitcount",
            "infestedunits",
            "result",
            "inspectionresult",
        ]
        keep = []
        for k in target_keys:
            if k in cols_norm_map and cols_norm_map[k] not in keep:
                keep.append(cols_norm_map[k])

        dt_col = None
        if date_cands:
            for dc in date_cands:
                dc_norm = dc.lower().replace(" ", "").replace("_", "")
                for c in cols:
                    c_norm = c.lower().replace(" ", "").replace("_", "")
                    if "flag" in c_norm:
                        continue
                    if c_norm == dc_norm:
                        if c not in keep:
                            keep.append(c)
                        dt_col = c
                        break
                if dt_col:
                    break
        df_aux = pd.read_parquet(t_path, storage_options=storage_options, columns=keep)
        df_aux.columns = [c.lower().replace(" ", "_") for c in df_aux.columns]
        df_aux["bbl_int"] = clean_bbl_to_int(df_aux)
        df_aux = df_aux[df_aux["bbl_int"] > 0].copy()
        if dt_col:
            dt_col_clean = dt_col.lower().replace(" ", "_")
            df_aux["event_dt"] = pd.to_datetime(
                df_aux[dt_col_clean], errors="coerce", utc=True
            ).dt.tz_localize(None)
        else:
            df_aux["event_dt"] = pd.NaT

        amt_col = next(
            (
                c
                for c in df_aux.columns
                if any(
                    k in c.replace("_", "")
                    for k in [
                        "actualchargeamount",
                        "chargeamount",
                        "saleprice",
                        "price",
                        "amount",
                    ]
                )
            ),
            None,
        )
        if amt_col:
            if df_aux[amt_col].dtype == object:
                clean_s = df_aux[amt_col].astype(str).str.replace(r"[$,]", "", regex=True)
                df_aux["amount"] = pd.to_numeric(clean_s, errors="coerce").fillna(0.0)
            else:
                df_aux["amount"] = pd.to_numeric(df_aux[amt_col], errors="coerce").fillna(0.0)
        else:
            df_aux["amount"] = 0.0

        inf_col = next(
            (c for c in df_aux.columns if "infested" in c or "inf_units" in c),
            None,
        )
        if inf_col:
            df_aux["infested_units"] = pd.to_numeric(
                df_aux[inf_col], errors="coerce"
            ).fillna(0.0)
        else:
            df_aux["infested_units"] = 0.0

        return df_aux
    except Exception as e:
        print(f"Notice loading {table_name}: {e}")
        return pd.DataFrame(
            columns=["bbl_int", "event_dt", "amount", "infested_units"]
        )


df_lit = safe_load_aux_bbl(
    "hpd_litigations", ["caseopendate", "case_open_date", "opendate"]
)
df_vacate = safe_load_aux_bbl(
    "hpd_vacate_orders",
    ["vacate_effective_date", "effective_date", "vacatedate"],
)
df_aep = safe_load_aux_bbl("hpd_aep_buildings")
df_conh = safe_load_aux_bbl("hpd_conh_buildings")
df_hwo = safe_load_aux_bbl(
    "hpd_hwo_charges", ["chargedate", "approvaldate", "fee_date"]
)
df_evict = safe_load_aux_bbl(
    "evictions", ["executed_date", "eviction_date", "ejection_date"]
)
df_omo = safe_load_aux_bbl(
    "hpd_omo_charges",
    [
        "invoicedate",
        "chargedate",
        "invoice_date",
        "charge_date",
        "approvaldate",
        "approveddate",
    ],
)
df_bedbug = safe_load_aux_bbl(
    "hpd_bedbug_reports",
    [
        "filingdate",
        "filing_date",
        "receiveddate",
        "received_date",
        "inspectiondate",
        "inspection_date",
        "reportdate",
        "report_date",
        "date",
    ],
)

# Cross-agency distress datasets: DOB violations, DOB ECB violations, and Rodent inspections
print("Loading DOB violations, DOB ECB violations, and DOHMH rodent inspections...")
df_dob = safe_load_aux_bbl(
    "dob_violations",
    ["issue_date", "issuedate", "violation_date", "date", "inspection_date", "inspectiondate"],
)
df_dob_ecb = safe_load_aux_bbl(
    "dob_ecb_violations",
    ["issue_date", "issuedate", "served_date", "violation_date", "hearing_date", "date"],
)
df_rodent = safe_load_aux_bbl(
    "dohmh_rodent_inspections",
    ["inspection_date", "inspectiondate", "date", "inspect_date"],
)

if "result" in df_rodent.columns or "inspection_result" in df_rodent.columns:
    res_c = "result" if "result" in df_rodent.columns else "inspection_result"
    is_failed = df_rodent[res_c].astype(str).str.lower().str.contains("fail|rat|problem|active|violation|action")
    df_rodent_fail = df_rodent[is_failed].copy() if is_failed.sum() > 0 else df_rodent.copy()
else:
    df_rodent_fail = df_rodent

print("Loading DOB safety violations, DOF tax lien sales, and HPD underlying conditions...")
df_dob_safety = safe_load_aux_bbl(
    "dob_safety_violations",
    ["issue_date", "issuedate", "violation_date", "date", "inspection_date", "inspectiondate"],
)
df_tax_lien = safe_load_aux_bbl(
    "dof_tax_lien_sales",
    ["sale_date", "saledate", "lien_sale_date", "notice_date", "date", "year"],
)
df_underlying = safe_load_aux_bbl(
    "hpd_underlying_conditions",
    ["order_date", "orderdate", "notice_date", "date", "inspection_date"],
)

print("Loading DOF annualized sales and speculation watch list...")
df_sales = safe_load_aux_bbl(
    "dof_annualized_sales",
    ["sale_date", "saledate", "date", "deed_date", "document_date"],
)
df_speculation = safe_load_aux_bbl(
    "speculation_watch_list",
    ["date", "record_date", "as_of_date"],
)

print("Loading HPD registrations...")
try:
    reg_path = f"{GCS_BASE}/lake/full/hpd_registrations"
    reg_avail_cols = get_available_columns(reg_path)
    reg_id_col = next(
        (
            c
            for c in reg_avail_cols
            if any(
                k in c.lower()
                for k in ["registrationid", "registration_id", "regid", "reg_id"]
            )
        ),
        None,
    )
    reg_dt_cands = [
        "lastregistrationdate",
        "last_registration_date",
        "registrationdate",
        "registration_date",
        "date",
    ]
    reg_dt_col = next(
        (c for c in reg_dt_cands if c.lower() in [col.lower() for col in reg_avail_cols]),
        None,
    )

    desired_reg_cols = [
        c
        for c in ["bbl", "boroid", "boro", "block", "lot"]
        if c in reg_avail_cols
    ]
    if reg_id_col and reg_id_col not in desired_reg_cols:
        desired_reg_cols.append(reg_id_col)
    if reg_dt_col and reg_dt_col not in desired_reg_cols:
        desired_reg_cols.append(reg_dt_col)

    df_reg = pd.read_parquet(
        reg_path, storage_options=storage_options, columns=desired_reg_cols
    )
    df_reg.columns = [c.lower() for c in df_reg.columns]
    df_reg["bbl_int"] = clean_bbl_to_int(df_reg)
    df_reg = df_reg[df_reg["bbl_int"] > 0].copy()
    if reg_id_col:
        reg_id_lower = reg_id_col.lower()
        df_reg["registrationid"] = (
            pd.to_numeric(df_reg[reg_id_lower], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
        )
    else:
        df_reg["registrationid"] = -1

    if reg_dt_col:
        df_reg["event_dt"] = pd.to_datetime(
            df_reg[reg_dt_col.lower()], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df_reg["event_dt"] = pd.NaT
except Exception as e:
    print(f"Notice loading registrations: {e}")
    df_reg = pd.DataFrame(columns=["bbl_int", "registrationid", "event_dt"])

print("Loading HPD complaints...")
complaint_path = f"{GCS_BASE}/lake/full/hpd_complaints"
try:
    comp_avail_cols = get_available_columns(complaint_path)
    date_cands = [
        "receiveddate",
        "received_date",
        "dateentered",
        "date_entered",
        "complaintdate",
        "complaint_date",
        "statusdate",
        "status_date",
    ]
    comp_dt_col = next((c for c in date_cands if c in comp_avail_cols), None)
    desired_comp_cols = [
        c
        for c in [
            "bbl",
            "boroid",
            "boro",
            "block",
            "lot",
            "complaintid",
            "complaint_id",
            "status",
        ]
        if c in comp_avail_cols
    ]
    if comp_dt_col:
        desired_comp_cols.append(comp_dt_col)

    comp_cat_cols = [
        c
        for c in comp_avail_cols
        if any(k in c.lower() for k in ["category", "type", "desc", "problem", "code"])
        and c not in desired_comp_cols
    ]
    desired_comp_cols.extend(comp_cat_cols)

    df_comp = pd.read_parquet(
        complaint_path, storage_options=storage_options, columns=desired_comp_cols
    )
    df_comp["bbl_int"] = clean_bbl_to_int(df_comp)
    df_comp = df_comp[df_comp["bbl_int"] > 0].copy()
    if comp_dt_col:
        df_comp["event_dt"] = pd.to_datetime(
            df_comp[comp_dt_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
        df_comp = df_comp[df_comp["event_dt"].notna()].copy()
    else:
        df_comp["event_dt"] = pd.NaT

    heat_mask = pd.Series(False, index=df_comp.index)
    for c in comp_cat_cols:
        c_low = c.lower()
        if c_low in df_comp.columns:
            s_up = df_comp[c_low].astype(str).str.upper()
            heat_mask = heat_mask | s_up.str.contains("HEAT|HOT WATER|HOT_WATER")
    df_comp["is_heat"] = heat_mask
except Exception as e:
    print(f"Notice loading complaints: {e}")
    df_comp = pd.DataFrame(columns=["bbl_int", "event_dt", "is_heat"])

# ---------------------------------------------------------
# 5. COHORT FORMATION & TARGET DEFINITION
# ---------------------------------------------------------
print("Constructing training, validation, and test cohorts...")


def get_pluto_release_df(target_release):
    if version_col is not None and version_col in df_pluto.columns:
        mask = (
            df_pluto[version_col]
            .astype(str)
            .str.lower()
            .str.contains(target_release.lower())
        )
        sub = df_pluto[mask].copy()
    else:
        sub = df_pluto.copy()
    sub = sub[pd.to_numeric(sub["unitsres"], errors="coerce").fillna(0) >= 3]
    return sub.drop_duplicates(subset=["bbl_int"])


df_pluto_train_2020 = get_pluto_release_df("19v2")
df_pluto_train_2021 = get_pluto_release_df("20v7")
df_pluto_val = get_pluto_release_df("21v4")
df_pluto_test = get_pluto_release_df("22v3")

train_bbls_2020 = df_pluto_train_2020["bbl_int"].unique()
train_bbls_2021 = df_pluto_train_2021["bbl_int"].unique()
val_bbls = df_pluto_val["bbl_int"].unique()
test_bbls = df_test["bbl_int"].values

print(
    f"Cohort Sizes -> Train 2020: {len(train_bbls_2020)}, Train 2021: {len(train_bbls_2021)}, "
    f"Val (2022): {len(val_bbls)}, Test (2023): {len(test_bbls)}"
)


def get_positive_bbls(start_dt, end_dt):
    mask = (
        (df_viol["class_clean"] == "C")
        & (df_viol["insp_dt"] >= start_dt)
        & (df_viol["insp_dt"] < end_dt)
    )
    return set(df_viol.loc[mask, "bbl_int"].unique())


pos_train_2020 = get_positive_bbls(
    pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")
)
pos_train_2021 = get_positive_bbls(
    pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")
)
pos_val = get_positive_bbls(
    pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")
)


# ---------------------------------------------------------
# 6. FEATURE ENGINEERING PIPELINE
# ---------------------------------------------------------
def extract_features(bbl_array, cutoff_dt, pluto_release_df):
    cohort_df = pd.DataFrame({"bbl_int": bbl_array})
    cohort_df = cohort_df.merge(
        pluto_release_df, on="bbl_int", how="left"
    ).drop_duplicates(subset=["bbl_int"])

    # Morphological features
    unitsres = pd.to_numeric(cohort_df["unitsres"], errors="coerce").fillna(3)
    unitstot = pd.to_numeric(cohort_df["unitstotal"], errors="coerce").fillna(unitsres)
    yearbuilt = (
        pd.to_numeric(cohort_df["yearbuilt"], errors="coerce")
        .fillna(1940)
        .clip(1800, cutoff_dt.year)
    )
    bldgarea = (
        pd.to_numeric(cohort_df["bldgarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    resarea = (
        pd.to_numeric(cohort_df["resarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    lotarea = (
        pd.to_numeric(cohort_df["lotarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    numfloors = (
        pd.to_numeric(cohort_df["numfloors"], errors="coerce").fillna(3).clip(1, 120)
    )
    assessland = (
        pd.to_numeric(cohort_df["assessland"], errors="coerce").fillna(0).clip(lower=0)
    )
    assesstot = (
        pd.to_numeric(cohort_df["assesstot"], errors="coerce").fillna(0).clip(lower=0)
    )

    feat = pd.DataFrame(index=cohort_df.index)
    feat["bbl_int"] = cohort_df["bbl_int"]
    feat["unitsres"] = unitsres
    feat["unitstotal"] = unitstot
    feat["res_unit_share"] = unitsres / (unitstot + 1e-4)
    feat["building_age"] = (cutoff_dt.year - yearbuilt).clip(0, 160)
    feat["is_prewar"] = (yearbuilt < 1940).astype(np.float32)
    feat["is_midcentury"] = ((yearbuilt >= 1940) & (yearbuilt <= 1975)).astype(
        np.float32
    )
    feat["numfloors"] = numfloors
    feat["units_per_floor"] = unitsres / (numfloors + 1e-4)
    feat["bldgarea"] = bldgarea
    feat["resarea"] = resarea
    feat["lotarea"] = lotarea
    feat["area_per_unit"] = resarea / (unitsres + 1e-4)
    feat["lot_coverage"] = bldgarea / (lotarea * numfloors + 1e-4)
    feat["assessed_val_per_unit"] = assesstot / (unitsres + 1e-4)
    feat["land_val_ratio"] = assessland / (assesstot + 1e-4)
    feat["borough"] = (
        pd.to_numeric(cohort_df["borocode"], errors="coerce")
        .fillna(cohort_df["bbl_int"] // 1_000_000_000)
        .astype(int)
    )
    feat["cd"] = pd.to_numeric(cohort_df["cd"], errors="coerce").fillna(-1)

    # Building typology representations
    if "bldgclass" in cohort_df.columns:
        bldgclass_str = (
            cohort_df["bldgclass"].fillna("").astype(str).str.strip().str.upper()
        )
        prefix = bldgclass_str.str[0].fillna("")
        archetype_map = {
            "C": 1.0,
            "D": 2.0,
            "S": 3.0,
            "A": 4.0,
            "B": 5.0,
            "R": 6.0,
            "K": 7.0,
            "O": 8.0,
            "L": 9.0,
            "H": 10.0,
        }
        feat["bldgclass_archetype"] = (
            prefix.map(archetype_map).fillna(0).astype(np.float32)
        )
        feat["is_bldgclass_walkup"] = (prefix == "C").astype(np.float32)
        feat["is_bldgclass_elevator"] = (prefix == "D").astype(np.float32)
        feat["is_bldgclass_mixed"] = (prefix == "S").astype(np.float32)
        bldg_counts = cohort_df["bldgclass"].astype(str).value_counts(normalize=True)
        feat["bldgclass_freq"] = (
            cohort_df["bldgclass"].astype(str).map(bldg_counts).fillna(0).astype(np.float32)
        )
    else:
        feat["bldgclass_archetype"] = 0.0
        feat["is_bldgclass_walkup"] = 0.0
        feat["is_bldgclass_elevator"] = 0.0
        feat["is_bldgclass_mixed"] = 0.0
        feat["bldgclass_freq"] = 0.0

    if "landuse" in cohort_df.columns:
        landuse_counts = cohort_df["landuse"].astype(str).value_counts(normalize=True)
        feat["landuse_freq"] = (
            cohort_df["landuse"].astype(str).map(landuse_counts).fillna(0).astype(np.float32)
        )
        feat["landuse_code"] = (
            pd.to_numeric(cohort_df["landuse"], errors="coerce")
            .fillna(-1)
            .astype(np.float32)
        )
    else:
        feat["landuse_freq"] = 0.0
        feat["landuse_code"] = -1.0

    # Historical Violation Features (strictly insp_dt < cutoff_dt)
    v_past = df_viol[df_viol["insp_dt"] < cutoff_dt]
    dt_30d = cutoff_dt - pd.Timedelta(days=30)
    dt_90d = cutoff_dt - pd.Timedelta(days=90)
    dt_180d = cutoff_dt - pd.Timedelta(days=180)
    dt_1y = cutoff_dt - pd.Timedelta(days=365)
    dt_2y = cutoff_dt - pd.Timedelta(days=730)
    dt_3y = cutoff_dt - pd.Timedelta(days=1095)

    v_30d = v_past[v_past["insp_dt"] >= dt_30d]
    v_90d = v_past[v_past["insp_dt"] >= dt_90d]
    v_180d = v_past[v_past["insp_dt"] >= dt_180d]
    v_1y = v_past[v_past["insp_dt"] >= dt_1y]
    v_2y = v_past[v_past["insp_dt"] >= dt_2y]
    v_3y = v_past[v_past["insp_dt"] >= dt_3y]

    def count_by_bbl(df_slice, filter_c=False, filter_b=False, filter_a=False):
        if filter_c:
            df_slice = df_slice[df_slice["class_clean"] == "C"]
        elif filter_b:
            df_slice = df_slice[df_slice["class_clean"] == "B"]
        elif filter_a:
            df_slice = df_slice[df_slice["class_clean"] == "A"]
        return df_slice.groupby("bbl_int").size()

    c_30d = count_by_bbl(v_30d, filter_c=True)
    b_30d = count_by_bbl(v_30d, filter_b=True)
    tot_30d = count_by_bbl(v_30d)

    c_90d = count_by_bbl(v_90d, filter_c=True)
    b_90d = count_by_bbl(v_90d, filter_b=True)
    tot_90d = count_by_bbl(v_90d)

    c_180d = count_by_bbl(v_180d, filter_c=True)
    b_180d = count_by_bbl(v_180d, filter_b=True)
    tot_180d = count_by_bbl(v_180d)

    c_1y = count_by_bbl(v_1y, filter_c=True)
    b_1y = count_by_bbl(v_1y, filter_b=True)
    a_1y = count_by_bbl(v_1y, filter_a=True)
    tot_1y = count_by_bbl(v_1y)

    c_2y = count_by_bbl(v_2y, filter_c=True)
    b_2y = count_by_bbl(v_2y, filter_b=True)
    tot_2y = count_by_bbl(v_2y)

    b_3y = count_by_bbl(v_3y, filter_b=True)
    c_3y = count_by_bbl(v_3y, filter_c=True)
    tot_3y = count_by_bbl(v_3y)

    c_life = count_by_bbl(v_past, filter_c=True)
    b_life = count_by_bbl(v_past, filter_b=True)
    tot_life = count_by_bbl(v_past)

    # Acute 60-day early-winter surge features (Nov 1 to Dec 31 pre-cutoff)
    dt_winter_start = cutoff_dt - pd.Timedelta(days=61)
    v_winter = v_past[v_past["insp_dt"] >= dt_winter_start]
    c_winter = count_by_bbl(v_winter, filter_c=True)
    b_winter = count_by_bbl(v_winter, filter_b=True)
    tot_winter = count_by_bbl(v_winter)

    # Open violation backlog features
    v_past_open = v_past[v_past["status_clean"].str.contains("OPEN")]
    c_open = count_by_bbl(v_past_open, filter_c=True)
    b_open = count_by_bbl(v_past_open, filter_b=True)
    tot_open = count_by_bbl(v_past_open)

    feat["viol_c_30d"] = feat["bbl_int"].map(c_30d).fillna(0)
    feat["viol_b_30d"] = feat["bbl_int"].map(b_30d).fillna(0)
    feat["viol_tot_30d"] = feat["bbl_int"].map(tot_30d).fillna(0)

    feat["viol_c_90d"] = feat["bbl_int"].map(c_90d).fillna(0)
    feat["viol_b_90d"] = feat["bbl_int"].map(b_90d).fillna(0)
    feat["viol_tot_90d"] = feat["bbl_int"].map(tot_90d).fillna(0)

    feat["viol_c_180d"] = feat["bbl_int"].map(c_180d).fillna(0)
    feat["viol_b_180d"] = feat["bbl_int"].map(b_180d).fillna(0)
    feat["viol_tot_180d"] = feat["bbl_int"].map(tot_180d).fillna(0)

    feat["viol_c_1y"] = feat["bbl_int"].map(c_1y).fillna(0)
    feat["viol_b_1y"] = feat["bbl_int"].map(b_1y).fillna(0)
    feat["viol_a_1y"] = feat["bbl_int"].map(a_1y).fillna(0)
    feat["viol_tot_1y"] = feat["bbl_int"].map(tot_1y).fillna(0)

    feat["viol_c_2y"] = feat["bbl_int"].map(c_2y).fillna(0)
    feat["viol_b_2y"] = feat["bbl_int"].map(b_2y).fillna(0)
    feat["viol_tot_2y"] = feat["bbl_int"].map(tot_2y).fillna(0)

    feat["viol_c_3y"] = feat["bbl_int"].map(c_3y).fillna(0)
    feat["viol_b_3y"] = feat["bbl_int"].map(b_3y).fillna(0)
    feat["viol_tot_3y"] = feat["bbl_int"].map(tot_3y).fillna(0)

    feat["viol_c_life"] = feat["bbl_int"].map(c_life).fillna(0)
    feat["viol_b_life"] = feat["bbl_int"].map(b_life).fillna(0)
    feat["viol_tot_life"] = feat["bbl_int"].map(tot_life).fillna(0)

    # Inspection visit cadence and per-visit violation densities
    v_1y_c = v_1y[v_1y["class_clean"] == "C"]
    v_3y_c = v_3y[v_3y["class_clean"] == "C"]
    insp_visits_c_1y = v_1y_c.groupby("bbl_int")["insp_dt"].nunique()
    insp_visits_tot_1y = v_1y.groupby("bbl_int")["insp_dt"].nunique()
    insp_visits_c_3y = v_3y_c.groupby("bbl_int")["insp_dt"].nunique()
    insp_visits_tot_3y = v_3y.groupby("bbl_int")["insp_dt"].nunique()

    feat["insp_visits_c_1y"] = feat["bbl_int"].map(insp_visits_c_1y).fillna(0).astype(np.float32)
    feat["insp_visits_tot_1y"] = feat["bbl_int"].map(insp_visits_tot_1y).fillna(0).astype(np.float32)
    feat["insp_visits_c_3y"] = feat["bbl_int"].map(insp_visits_c_3y).fillna(0).astype(np.float32)
    feat["insp_visits_tot_3y"] = feat["bbl_int"].map(insp_visits_tot_3y).fillna(0).astype(np.float32)
    feat["viol_c_per_visit_1y"] = (
        feat["viol_c_1y"] / np.maximum(feat["insp_visits_c_1y"], 1.0)
    ).astype(np.float32)
    feat["viol_tot_per_visit_1y"] = (
        feat["viol_tot_1y"] / np.maximum(feat["insp_visits_tot_1y"], 1.0)
    ).astype(np.float32)

    # Intra-year 180-day momentum (H1: older half [-365d, -180d), H2: recent half [-180d, 0d))
    v_h1 = v_past[(v_past["insp_dt"] >= dt_1y) & (v_past["insp_dt"] < dt_180d)]
    c_h1 = count_by_bbl(v_h1, filter_c=True)
    c_h2 = c_180d
    feat["viol_c_h1"] = feat["bbl_int"].map(c_h1).fillna(0).astype(np.float32)
    feat["viol_c_h2"] = feat["bbl_int"].map(c_h2).fillna(0).astype(np.float32)
    feat["viol_c_h2_to_h1_accel"] = (
        feat["viol_c_h2"] / (feat["viol_c_h1"] + 1.0)
    ).astype(np.float32)

    # 60-day winter surge metrics & velocities
    feat["viol_c_winter_surge"] = feat["bbl_int"].map(c_winter).fillna(0).astype(np.float32)
    feat["viol_b_winter_surge"] = feat["bbl_int"].map(b_winter).fillna(0).astype(np.float32)
    feat["viol_tot_winter_surge"] = feat["bbl_int"].map(tot_winter).fillna(0).astype(np.float32)
    feat["viol_c_winter_velocity"] = ((feat["viol_c_winter_surge"] * 6.0) / (feat["viol_c_1y"] + 1.0)).astype(np.float32)
    feat["viol_tot_winter_velocity"] = ((feat["viol_tot_winter_surge"] * 6.0) / (feat["viol_tot_1y"] + 1.0)).astype(np.float32)
    feat["has_winter_c_surge"] = (feat["viol_c_winter_surge"] > 0).astype(np.float32)

    # Backlog counts and ratios
    feat["viol_c_open_backlog"] = feat["bbl_int"].map(c_open).fillna(0).astype(np.float32)
    feat["viol_b_open_backlog"] = feat["bbl_int"].map(b_open).fillna(0).astype(np.float32)
    feat["viol_tot_open_backlog"] = feat["bbl_int"].map(tot_open).fillna(0).astype(np.float32)
    feat["backlog_ratio_c"] = (feat["viol_c_open_backlog"] / (feat["viol_c_life"] + 1.0)).astype(np.float32)
    feat["backlog_ratio_tot"] = (feat["viol_tot_open_backlog"] / (feat["viol_tot_life"] + 1.0)).astype(np.float32)
    feat["backlog_b_to_c_ratio"] = (feat["viol_c_open_backlog"] / (feat["viol_b_open_backlog"] + 1.0)).astype(np.float32)

    feat["viol_c_accel_30_180"] = (feat["viol_c_30d"] * 6.0) / (
        feat["viol_c_180d"] + 1.0
    )
    feat["viol_tot_accel_30_180"] = (feat["viol_tot_30d"] * 6.0) / (
        feat["viol_tot_180d"] + 1.0
    )
    feat["viol_b_accel_30_180"] = (feat["viol_b_30d"] * 6.0) / (
        feat["viol_b_180d"] + 1.0
    )

    feat["viol_c_accel_1y_2y"] = feat["viol_c_1y"] / (
        feat["viol_c_2y"] - feat["viol_c_1y"] + 1.0
    )
    feat["viol_c_net_accel_1y_2y"] = (
        feat["viol_c_1y"] - (feat["viol_c_2y"] - feat["viol_c_1y"])
    ).astype(np.float32)
    feat["is_chronic_repeat"] = (
        (feat["viol_c_1y"] > 0) & (feat["viol_c_2y"] > feat["viol_c_1y"])
    ).astype(np.float32)
    feat["viol_b_to_c_ratio"] = (
        feat["viol_c_1y"] / (feat["viol_b_1y"] + 1.0)
    ).astype(np.float32)
    feat["b_to_c_escalation_1y"] = (
        feat["viol_c_1y"] / (feat["viol_b_1y"] + 1.0)
    ).astype(np.float32)
    feat["b_to_c_escalation_3y"] = (
        feat["viol_c_3y"] / (feat["viol_b_3y"] + 1.0)
    ).astype(np.float32)
    feat["viol_tot_accel_1y_2y"] = feat["viol_tot_1y"] / (
        feat["viol_tot_2y"] - feat["viol_tot_1y"] + 1.0
    )
    feat["viol_c_share_1y"] = feat["viol_c_1y"] / (feat["viol_tot_1y"] + 1e-4)
    feat["viol_c_share_life"] = feat["viol_c_life"] / (feat["viol_tot_life"] + 1e-4)
    feat["has_viol_c_30d"] = (feat["viol_c_30d"] > 0).astype(np.float32)
    feat["has_viol_c_90d"] = (feat["viol_c_90d"] > 0).astype(np.float32)
    feat["has_viol_c_1y"] = (feat["viol_c_1y"] > 0).astype(np.float32)
    feat["has_viol_c_2y"] = (feat["viol_c_2y"] > 0).astype(np.float32)
    feat["viol_c_per_unit_1y"] = feat["viol_c_1y"] / (feat["unitsres"] + 1e-4)
    feat["viol_tot_per_unit_1y"] = feat["viol_tot_1y"] / (feat["unitsres"] + 1e-4)

    # Pre-cutoff Tenant Complaints
    if len(df_comp) > 0 and df_comp["event_dt"].notna().any():
        c_past = df_comp[df_comp["event_dt"] < cutoff_dt]
        cp_30d = c_past[c_past["event_dt"] >= dt_30d]
        cp_90d = c_past[c_past["event_dt"] >= dt_90d]
        cp_180d = c_past[c_past["event_dt"] >= dt_180d]
        cp_1y = c_past[c_past["event_dt"] >= dt_1y]

        comp_30d_cnt = cp_30d.groupby("bbl_int").size()
        comp_90d_cnt = cp_90d.groupby("bbl_int").size()
        comp_180d_cnt = cp_180d.groupby("bbl_int").size()
        comp_1y_cnt = cp_1y.groupby("bbl_int").size()
        comp_life_cnt = c_past.groupby("bbl_int").size()

        feat["comp_30d"] = feat["bbl_int"].map(comp_30d_cnt).fillna(0)
        feat["comp_90d"] = feat["bbl_int"].map(comp_90d_cnt).fillna(0)
        feat["comp_180d"] = feat["bbl_int"].map(comp_180d_cnt).fillna(0)
        feat["comp_1y"] = feat["bbl_int"].map(comp_1y_cnt).fillna(0)
        feat["comp_life"] = feat["bbl_int"].map(comp_life_cnt).fillna(0)

        cp_winter = c_past[c_past["event_dt"] >= dt_winter_start]
        comp_winter_cnt = cp_winter.groupby("bbl_int").size()
        feat["comp_winter_surge"] = feat["bbl_int"].map(comp_winter_cnt).fillna(0).astype(np.float32)
        feat["comp_winter_velocity"] = ((feat["comp_winter_surge"] * 6.0) / (feat["comp_1y"] + 1.0)).astype(np.float32)

        feat["comp_accel_30_180"] = (feat["comp_30d"] * 6.0) / (
            feat["comp_180d"] + 1.0
        )
        feat["has_recent_comp"] = (feat["comp_90d"] > 0).astype(np.float32)
        feat["comp_per_unit_1y"] = feat["comp_1y"] / (feat["unitsres"] + 1e-4)

        max_dt_comp = c_past.groupby("bbl_int")["event_dt"].max()
        days_since_comp = (cutoff_dt - feat["bbl_int"].map(max_dt_comp)).dt.days.fillna(9999)
        feat["days_since_last_comp"] = days_since_comp.clip(0, 9999)
        feat["recency_decay_comp"] = np.exp(-0.003 * feat["days_since_last_comp"])

        # Heat and hot water emergency complaint aggregations
        if "is_heat" in c_past.columns and c_past["is_heat"].any():
            cp_heat = c_past[c_past["is_heat"]]
            cp_heat_30d = cp_heat[cp_heat["event_dt"] >= dt_30d]
            cp_heat_winter = cp_heat[cp_heat["event_dt"] >= dt_winter_start]
            cp_heat_90d = cp_heat[cp_heat["event_dt"] >= dt_90d]
            cp_heat_1y = cp_heat[cp_heat["event_dt"] >= dt_1y]

            feat["comp_heat_30d"] = feat["bbl_int"].map(cp_heat_30d.groupby("bbl_int").size()).fillna(0).astype(np.float32)
            feat["comp_heat_winter_surge"] = feat["bbl_int"].map(cp_heat_winter.groupby("bbl_int").size()).fillna(0).astype(np.float32)
            feat["comp_heat_90d"] = feat["bbl_int"].map(cp_heat_90d.groupby("bbl_int").size()).fillna(0).astype(np.float32)
            feat["comp_heat_1y"] = feat["bbl_int"].map(cp_heat_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
            feat["comp_heat_share_1y"] = (feat["comp_heat_1y"] / (feat["comp_1y"] + 1e-4)).astype(np.float32)
            feat["has_winter_heat_comp"] = (feat["comp_heat_winter_surge"] > 0).astype(np.float32)
        else:
            feat["comp_heat_30d"] = 0.0
            feat["comp_heat_winter_surge"] = 0.0
            feat["comp_heat_90d"] = 0.0
            feat["comp_heat_1y"] = 0.0
            feat["comp_heat_share_1y"] = 0.0
            feat["has_winter_heat_comp"] = 0.0
    else:
        feat["comp_30d"] = 0.0
        feat["comp_90d"] = 0.0
        feat["comp_180d"] = 0.0
        feat["comp_1y"] = 0.0
        feat["comp_life"] = 0.0
        feat["comp_winter_surge"] = 0.0
        feat["comp_winter_velocity"] = 0.0
        feat["comp_accel_30_180"] = 0.0
        feat["has_recent_comp"] = 0.0
        feat["comp_per_unit_1y"] = 0.0
        feat["days_since_last_comp"] = 9999.0
        feat["recency_decay_comp"] = 0.0
        feat["comp_heat_30d"] = 0.0
        feat["comp_heat_winter_surge"] = 0.0
        feat["comp_heat_90d"] = 0.0
        feat["comp_heat_1y"] = 0.0
        feat["comp_heat_share_1y"] = 0.0
        feat["has_winter_heat_comp"] = 0.0

    # Micro-spatial Tax Block Leave-One-Out Risk Aggregations
    block_id = cohort_df["bbl_int"] // 10000
    block_viol_sum = feat.groupby(block_id)["viol_c_1y"].transform("sum")
    block_cnt = feat.groupby(block_id)["viol_c_1y"].transform("count")
    feat["block_viol_c_density"] = (
        (block_viol_sum - feat["viol_c_1y"]) / np.maximum(block_cnt - 1.0, 1.0)
    ).fillna(0.0).astype(np.float32)

    block_comp_sum = feat.groupby(block_id)["comp_1y"].transform("sum")
    feat["block_comp_density"] = (
        (block_comp_sum - feat["comp_1y"]) / np.maximum(block_cnt - 1.0, 1.0)
    ).fillna(0.0).astype(np.float32)

    # Recency features
    max_dt_all = v_past.groupby("bbl_int")["insp_dt"].max()
    max_dt_c = v_past[v_past["class_clean"] == "C"].groupby("bbl_int")["insp_dt"].max()

    days_since_viol = (cutoff_dt - feat["bbl_int"].map(max_dt_all)).dt.days.fillna(9999)
    days_since_c = (cutoff_dt - feat["bbl_int"].map(max_dt_c)).dt.days.fillna(9999)
    feat["days_since_last_viol"] = days_since_viol.clip(0, 9999)
    feat["days_since_last_c"] = days_since_c.clip(0, 9999)
    feat["recency_decay_c"] = np.exp(-0.003 * feat["days_since_last_c"])

    # Auxiliary Distress Signals
    if len(df_lit) > 0:
        lit_past = df_lit[
            (df_lit["event_dt"].isna()) | (df_lit["event_dt"] < cutoff_dt)
        ]
        lit_2y = lit_past[lit_past["event_dt"].isna() | (lit_past["event_dt"] >= dt_2y)]
        feat["lit_count_life"] = (
            feat["bbl_int"].map(lit_past.groupby("bbl_int").size()).fillna(0)
        )
        feat["lit_count_2y"] = (
            feat["bbl_int"].map(lit_2y.groupby("bbl_int").size()).fillna(0)
        )
    else:
        feat["lit_count_life"] = 0
        feat["lit_count_2y"] = 0
    feat["has_litigation"] = (feat["lit_count_life"] > 0).astype(np.float32)

    feat["has_vacate_order"] = (
        feat["bbl_int"].isin(set(df_vacate["bbl_int"])).astype(np.float32)
    )
    feat["is_aep_building"] = (
        feat["bbl_int"].isin(set(df_aep["bbl_int"])).astype(np.float32)
    )
    feat["is_conh_building"] = (
        feat["bbl_int"].isin(set(df_conh["bbl_int"])).astype(np.float32)
    )

    if len(df_hwo) > 0:
        hwo_past = df_hwo[
            (df_hwo["event_dt"].isna()) | (df_hwo["event_dt"] < cutoff_dt)
        ]
        feat["hwo_count"] = (
            feat["bbl_int"].map(hwo_past.groupby("bbl_int").size()).fillna(0)
        )
    else:
        feat["hwo_count"] = 0

    if len(df_evict) > 0:
        evict_past = df_evict[
            (df_evict["event_dt"].isna()) | (df_evict["event_dt"] < cutoff_dt)
        ]
        feat["eviction_count"] = (
            feat["bbl_int"].map(evict_past.groupby("bbl_int").size()).fillna(0)
        )
    else:
        feat["eviction_count"] = 0

    # DOB Violations
    if len(df_dob) > 0:
        dob_past = df_dob[(df_dob["event_dt"].isna()) | (df_dob["event_dt"] < cutoff_dt)]
        dob_1y = df_dob[(df_dob["event_dt"].notna()) & (df_dob["event_dt"] < cutoff_dt) & (df_dob["event_dt"] >= dt_1y)]
        dob_3y = df_dob[(df_dob["event_dt"].notna()) & (df_dob["event_dt"] < cutoff_dt) & (df_dob["event_dt"] >= dt_3y)]
        feat["dob_viol_1y"] = feat["bbl_int"].map(dob_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_viol_3y"] = feat["bbl_int"].map(dob_3y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_viol_life"] = feat["bbl_int"].map(dob_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
    else:
        feat["dob_viol_1y"] = 0.0
        feat["dob_viol_3y"] = 0.0
        feat["dob_viol_life"] = 0.0

    # DOB ECB Violations
    if len(df_dob_ecb) > 0:
        ecb_past = df_dob_ecb[(df_dob_ecb["event_dt"].isna()) | (df_dob_ecb["event_dt"] < cutoff_dt)]
        ecb_1y = df_dob_ecb[(df_dob_ecb["event_dt"].notna()) & (df_dob_ecb["event_dt"] < cutoff_dt) & (df_dob_ecb["event_dt"] >= dt_1y)]
        ecb_3y = df_dob_ecb[(df_dob_ecb["event_dt"].notna()) & (df_dob_ecb["event_dt"] < cutoff_dt) & (df_dob_ecb["event_dt"] >= dt_3y)]
        feat["dob_ecb_viol_1y"] = feat["bbl_int"].map(ecb_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_ecb_viol_3y"] = feat["bbl_int"].map(ecb_3y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_ecb_viol_life"] = feat["bbl_int"].map(ecb_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
    else:
        feat["dob_ecb_viol_1y"] = 0.0
        feat["dob_ecb_viol_3y"] = 0.0
        feat["dob_ecb_viol_life"] = 0.0

    # DOHMH Rodent Inspections
    if len(df_rodent_fail) > 0:
        rod_past = df_rodent_fail[(df_rodent_fail["event_dt"].isna()) | (df_rodent_fail["event_dt"] < cutoff_dt)]
        rod_1y = df_rodent_fail[(df_rodent_fail["event_dt"].notna()) & (df_rodent_fail["event_dt"] < cutoff_dt) & (df_rodent_fail["event_dt"] >= dt_1y)]
        rod_3y = df_rodent_fail[(df_rodent_fail["event_dt"].notna()) & (df_rodent_fail["event_dt"] < cutoff_dt) & (df_rodent_fail["event_dt"] >= dt_3y)]
        feat["rodent_fail_1y"] = feat["bbl_int"].map(rod_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["rodent_fail_3y"] = feat["bbl_int"].map(rod_3y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["rodent_fail_life"] = feat["bbl_int"].map(rod_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
    else:
        feat["rodent_fail_1y"] = 0.0
        feat["rodent_fail_3y"] = 0.0
        feat["rodent_fail_life"] = 0.0

    # DOB Safety Violations
    if len(df_dob_safety) > 0:
        safety_past = df_dob_safety[(df_dob_safety["event_dt"].isna()) | (df_dob_safety["event_dt"] < cutoff_dt)]
        safety_1y = df_dob_safety[(df_dob_safety["event_dt"].notna()) & (df_dob_safety["event_dt"] < cutoff_dt) & (df_dob_safety["event_dt"] >= dt_1y)]
        safety_3y = df_dob_safety[(df_dob_safety["event_dt"].notna()) & (df_dob_safety["event_dt"] < cutoff_dt) & (df_dob_safety["event_dt"] >= dt_3y)]
        feat["dob_safety_viol_1y"] = feat["bbl_int"].map(safety_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_safety_viol_3y"] = feat["bbl_int"].map(safety_3y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["dob_safety_viol_life"] = feat["bbl_int"].map(safety_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
    else:
        feat["dob_safety_viol_1y"] = 0.0
        feat["dob_safety_viol_3y"] = 0.0
        feat["dob_safety_viol_life"] = 0.0

    feat["multi_agency_distress_1y"] = (
        feat["dob_viol_1y"] + feat["dob_ecb_viol_1y"] + feat["rodent_fail_1y"] + feat["dob_safety_viol_1y"]
    ).astype(np.float32)
    feat["multi_agency_distress_3y"] = (
        feat["dob_viol_3y"] + feat["dob_ecb_viol_3y"] + feat["rodent_fail_3y"] + feat["dob_safety_viol_3y"]
    ).astype(np.float32)
    feat["has_multi_agency_distress"] = (feat["multi_agency_distress_1y"] > 0).astype(np.float32)

    # DOF Tax Lien Sales
    if len(df_tax_lien) > 0:
        lien_past = df_tax_lien[(df_tax_lien["event_dt"].isna()) | (df_tax_lien["event_dt"] < cutoff_dt)]
        feat["has_tax_lien_sale"] = feat["bbl_int"].isin(set(lien_past["bbl_int"])).astype(np.float32)
    else:
        feat["has_tax_lien_sale"] = 0.0

    # HPD Underlying Conditions Program
    if len(df_underlying) > 0:
        und_past = df_underlying[(df_underlying["event_dt"].isna()) | (df_underlying["event_dt"] < cutoff_dt)]
        feat["is_underlying_conditions_bldg"] = feat["bbl_int"].isin(set(und_past["bbl_int"])).astype(np.float32)
    else:
        feat["is_underlying_conditions_bldg"] = 0.0

    # DOF Annualized Sales Features (strictly event_dt < cutoff_dt)
    if len(df_sales) > 0 and df_sales["event_dt"].notna().any():
        sales_past = df_sales[df_sales["event_dt"] < cutoff_dt]
        sales_1y = sales_past[sales_past["event_dt"] >= dt_1y]
        sales_3y = sales_past[sales_past["event_dt"] >= dt_3y]

        feat["sale_count_1y"] = feat["bbl_int"].map(sales_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["sale_count_3y"] = feat["bbl_int"].map(sales_3y.groupby("bbl_int").size()).fillna(0).astype(np.float32)

        max_dt_sale = sales_past.groupby("bbl_int")["event_dt"].max()
        days_since_sale = (cutoff_dt - feat["bbl_int"].map(max_dt_sale)).dt.days.fillna(9999)
        feat["days_since_last_sale"] = days_since_sale.clip(0, 9999).astype(np.float32)

        sales_past_sorted = sales_past.sort_values("event_dt").drop_duplicates(subset=["bbl_int"], keep="last")
        last_sale_map = sales_past_sorted.set_index("bbl_int")["amount"].to_dict()
        feat["last_sale_price"] = feat["bbl_int"].map(last_sale_map).fillna(0.0).astype(np.float32)
        feat["sale_price_per_unit"] = (feat["last_sale_price"] / (feat["unitsres"] + 1e-4)).astype(np.float32)
    else:
        feat["sale_count_1y"] = 0.0
        feat["sale_count_3y"] = 0.0
        feat["days_since_last_sale"] = 9999.0
        feat["last_sale_price"] = 0.0
        feat["sale_price_per_unit"] = 0.0

    # Speculation Watch List Flag
    if len(df_speculation) > 0:
        feat["is_speculation_watch_bldg"] = feat["bbl_int"].isin(set(df_speculation["bbl_int"])).astype(np.float32)
    else:
        feat["is_speculation_watch_bldg"] = 0.0

    # Landlord Registration Portfolio Contagion Dynamics
    if len(df_reg) > 0:
        reg_past = df_reg[(df_reg["event_dt"].isna()) | (df_reg["event_dt"] < cutoff_dt)]
        if len(reg_past) > 0:
            if "event_dt" in reg_past.columns and reg_past["event_dt"].notna().any():
                reg_valid = (
                    reg_past[reg_past["registrationid"] > 0]
                    .sort_values("event_dt")
                    .drop_duplicates(subset=["bbl_int"], keep="last")
                )
            else:
                reg_valid = reg_past[reg_past["registrationid"] > 0].drop_duplicates(
                    subset=["bbl_int"], keep="last"
                )

            bbl_to_reg = dict(zip(reg_valid["bbl_int"], reg_valid["registrationid"]))
            cohort_reg = cohort_df["bbl_int"].map(bbl_to_reg).fillna(-1).astype(np.int64)
            feat["is_unregistered_owner"] = (cohort_reg <= 0).astype(np.float32)

            reg_valid["viol_c_1y"] = reg_valid["bbl_int"].map(c_1y).fillna(0.0)
            port_size_map = reg_valid.groupby("registrationid")["bbl_int"].count().to_dict()
            port_viol_map = reg_valid.groupby("registrationid")["viol_c_1y"].sum().to_dict()

            port_size = cohort_reg.map(port_size_map).fillna(0.0).astype(np.float32)
            port_c_sum = cohort_reg.map(port_viol_map).fillna(0.0).astype(np.float32)

            loo_c = np.maximum(port_c_sum - feat["viol_c_1y"], 0.0).astype(np.float32)
            feat["portfolio_lot_count"] = np.where(cohort_reg > 0, port_size, 0.0).astype(np.float32)
            feat["portfolio_c_viol_loo_1y"] = np.where(cohort_reg > 0, loo_c, 0.0).astype(np.float32)
            rem_lots = np.maximum(port_size - 1.0, 1.0)
            feat["portfolio_viol_density"] = np.where(
                (cohort_reg > 0) & (port_size > 1), loo_c / rem_lots, 0.0
            ).astype(np.float32)
        else:
            feat["is_unregistered_owner"] = 1.0
            feat["portfolio_lot_count"] = 0.0
            feat["portfolio_c_viol_loo_1y"] = 0.0
            feat["portfolio_viol_density"] = 0.0
    else:
        feat["is_unregistered_owner"] = 1.0
        feat["portfolio_lot_count"] = 0.0
        feat["portfolio_c_viol_loo_1y"] = 0.0
        feat["portfolio_viol_density"] = 0.0

    # HPD OMO (Open Market Order) Emergency Contractor Interventions
    if len(df_omo) > 0:
        omo_past = df_omo[(df_omo["event_dt"].isna()) | (df_omo["event_dt"] < cutoff_dt)]
        omo_1y = df_omo[(df_omo["event_dt"].notna()) & (df_omo["event_dt"] < cutoff_dt) & (df_omo["event_dt"] >= dt_1y)]
        feat["omo_count_life"] = feat["bbl_int"].map(omo_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["omo_count_1y"] = feat["bbl_int"].map(omo_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["has_omo"] = (feat["omo_count_life"] > 0).astype(np.float32)
        feat["omo_amount_life"] = feat["bbl_int"].map(omo_past.groupby("bbl_int")["amount"].sum()).fillna(0.0).astype(np.float32)
        feat["omo_amount_1y"] = feat["bbl_int"].map(omo_1y.groupby("bbl_int")["amount"].sum()).fillna(0.0).astype(np.float32)
        feat["omo_amount_per_unit"] = (feat["omo_amount_1y"] / (feat["unitsres"] + 1e-4)).astype(np.float32)
    else:
        feat["omo_count_life"] = 0.0
        feat["omo_count_1y"] = 0.0
        feat["has_omo"] = 0.0
        feat["omo_amount_life"] = 0.0
        feat["omo_amount_1y"] = 0.0
        feat["omo_amount_per_unit"] = 0.0

    # HPD Bedbug Infestation Reports
    if len(df_bedbug) > 0:
        bb_past = df_bedbug[(df_bedbug["event_dt"].isna()) | (df_bedbug["event_dt"] < cutoff_dt)]
        bb_1y = df_bedbug[(df_bedbug["event_dt"].notna()) & (df_bedbug["event_dt"] < cutoff_dt) & (df_bedbug["event_dt"] >= dt_1y)]
        bb_2y = df_bedbug[(df_bedbug["event_dt"].notna()) & (df_bedbug["event_dt"] < cutoff_dt) & (df_bedbug["event_dt"] >= dt_2y)]
        feat["bedbug_reports_life"] = feat["bbl_int"].map(bb_past.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["bedbug_reports_1y"] = feat["bbl_int"].map(bb_1y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["bedbug_reports_2y"] = feat["bbl_int"].map(bb_2y.groupby("bbl_int").size()).fillna(0).astype(np.float32)
        feat["has_bedbug_reports"] = (feat["bedbug_reports_life"] > 0).astype(np.float32)
        feat["bedbug_infested_units_1y"] = feat["bbl_int"].map(bb_1y.groupby("bbl_int")["infested_units"].sum()).fillna(0.0).astype(np.float32)
        feat["bedbug_infested_units_life"] = feat["bbl_int"].map(bb_past.groupby("bbl_int")["infested_units"].sum()).fillna(0.0).astype(np.float32)
    else:
        feat["bedbug_reports_life"] = 0.0
        feat["bedbug_reports_1y"] = 0.0
        feat["bedbug_reports_2y"] = 0.0
        feat["has_bedbug_reports"] = 0.0
        feat["bedbug_infested_units_1y"] = 0.0
        feat["bedbug_infested_units_life"] = 0.0

    # Composite Distress Score with Cross-Agency, Contractor Intervention & Winter Surge weighting
    feat["composite_distress"] = (
        feat["viol_c_1y"] * 2.5
        + feat["viol_c_90d"] * 3.5
        + feat["comp_90d"] * 1.5
        + feat["has_viol_c_1y"] * 3.0
        + feat["lit_count_2y"] * 2.0
        + feat["is_aep_building"] * 4.0
        + feat["has_vacate_order"] * 3.5
        + (feat["hwo_count"] > 0) * 2.5
        + feat["has_litigation"] * 1.5
        + feat["dob_viol_1y"] * 1.5
        + feat["dob_ecb_viol_1y"] * 1.5
        + feat["rodent_fail_1y"] * 1.0
        + feat["viol_c_winter_surge"] * 2.0
        + feat["comp_heat_winter_surge"] * 2.5
        + feat["omo_count_1y"] * 3.0
        + (feat["bedbug_reports_1y"] > 0) * 1.5
        + feat["dob_safety_viol_1y"] * 1.5
        + feat["has_tax_lien_sale"] * 2.0
        + feat["is_underlying_conditions_bldg"] * 2.5
        + feat["portfolio_viol_density"] * 2.0
        + feat["is_speculation_watch_bldg"] * 2.5
        + (feat["sale_count_1y"] > 0) * 1.5
        + feat["insp_visits_c_1y"] * 1.5
    )

    # 5-digit ZIP Code Empirical Bayes Smoothed Violation Risk
    if "zipcode" in cohort_df.columns:
        zip_series = pd.to_numeric(cohort_df["zipcode"], errors="coerce").fillna(0).astype(int)
        global_mean_c = feat["viol_c_2y"].mean()
        prior_weight = 25.0
        zip_sum = feat.groupby(zip_series)["viol_c_2y"].transform("sum")
        zip_count = feat.groupby(zip_series)["viol_c_2y"].transform("count")
        feat["zip_smoothed_risk"] = ((zip_sum + prior_weight * global_mean_c) / (zip_count + prior_weight)).astype(np.float32)
        feat["lot_to_zip_risk_ratio"] = (feat["viol_c_1y"] / (feat["zip_smoothed_risk"] + 1e-4)).astype(np.float32)
    else:
        feat["zip_smoothed_risk"] = 0.0
        feat["lot_to_zip_risk_ratio"] = 0.0

    # Spatial Community District Empirical Risk
    cd_rates = feat.groupby("cd")["viol_c_2y"].mean().to_dict()
    feat["cd_hist_c_density"] = feat["cd"].map(cd_rates).fillna(0)
    feat["lot_vs_cd_risk"] = feat["viol_c_1y"] / (feat["cd_hist_c_density"] + 1e-4)

    feat = feat.drop(columns=["bbl_int"])
    return feat


print("Engineering features for Train (2020 & 2021), Val (2022), and Test (2023)...")
X_train_2020 = extract_features(
    train_bbls_2020, pd.Timestamp("2020-01-01"), df_pluto_train_2020
)
y_train_2020 = pd.Series(train_bbls_2020).isin(pos_train_2020).astype(int).values

X_train_2021 = extract_features(
    train_bbls_2021, pd.Timestamp("2021-01-01"), df_pluto_train_2021
)
y_train_2021 = pd.Series(train_bbls_2021).isin(pos_train_2021).astype(int).values

X_val = extract_features(val_bbls, pd.Timestamp("2022-01-01"), df_pluto_val)
y_val = pd.Series(val_bbls).isin(pos_val).astype(int).values

X_test = extract_features(test_bbls, pd.Timestamp("2023-01-01"), df_pluto_test)

if len(X_train_2020) > 0:
    X_train = pd.concat([X_train_2020, X_train_2021], ignore_index=True)
    y_train = np.concatenate([y_train_2020, y_train_2021])
    sample_weight_train = np.concatenate([
        np.full(len(train_bbls_2020), 0.6, dtype=np.float32),
        np.full(len(train_bbls_2021), 1.0, dtype=np.float32),
    ])
else:
    X_train = X_train_2021
    y_train = y_train_2021
    sample_weight_train = np.ones(len(train_bbls_2021), dtype=np.float32)

del X_train_2020, X_train_2021, df_pluto_train_2020, df_pluto_train_2021
gc.collect()

# Align feature columns and cast all features to float32
feature_cols = [
    c
    for c in X_train.columns
    if c in X_val.columns and c in X_test.columns and c not in ["cd", "zipcode"]
]
X_train = X_train[feature_cols].fillna(0).astype(np.float32)
X_val = X_val[feature_cols].fillna(0).astype(np.float32)
X_test = X_test[feature_cols].fillna(0).astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

print(
    f"Extracted {len(feature_cols)} features. Pooled Train size: {len(X_train)} "
    f"(prevalence: {y_train.mean():.4f}), Val size: {len(X_val)} (prevalence: {y_val.mean():.4f})"
)

# ---------------------------------------------------------
# 7. MULTI-DEPTH LIGHTGBM & DEPTHWISE XGBOOST MODELS
# ---------------------------------------------------------
print("Training multi-depth regularized LightGBM models (shallow generalizer & deep interaction)...")

params_shallow = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "extra_trees": True,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 6,
    "scale_pos_weight": 1.2,
    "feature_fraction": 0.60,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "min_child_samples": 40,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

params_deep = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": 8,
    "scale_pos_weight": 1.2,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "random_state": 1337,
    "n_jobs": -1,
    "verbose": -1,
}

dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model_shallow = lgb.train(
    params_shallow,
    dtrain,
    num_boost_round=1200,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

model_deep = lgb.train(
    params_deep,
    dtrain,
    num_boost_round=1200,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

val_preds_shallow = model_shallow.predict(X_val)
val_preds_deep = model_deep.predict(X_val)
test_preds_shallow = model_shallow.predict(X_test)
test_preds_deep = model_deep.predict(X_test)

print("Training dual-scale regularized XGBoost models (shallow generalizer & deep interaction)...")
try:
    xgb_shallow = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.04,
        scale_pos_weight=1.2,
        subsample=0.85,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_shallow.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_shallow = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.04,
        scale_pos_weight=1.2,
        subsample=0.85,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_shallow.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
xgb_shallow.predict = lambda X: xgb_shallow.predict_proba(X)[:, 1]

try:
    xgb_deep = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.035,
        scale_pos_weight=1.2,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=1337,
        eval_metric="logloss",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_deep.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_deep = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.035,
        scale_pos_weight=1.2,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=1337,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_deep.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
xgb_deep.predict = lambda X: xgb_deep.predict_proba(X)[:, 1]

# Maintain backward compatibility reference
xgb_model = xgb_deep

val_preds_xgb_shallow = xgb_shallow.predict_proba(X_val)[:, 1]
test_preds_xgb_shallow = xgb_shallow.predict_proba(X_test)[:, 1]
val_preds_xgb_deep = xgb_deep.predict_proba(X_val)[:, 1]
test_preds_xgb_deep = xgb_deep.predict_proba(X_test)[:, 1]
val_preds_xgb = 0.5 * (val_preds_xgb_shallow + val_preds_xgb_deep)
test_preds_xgb = 0.5 * (test_preds_xgb_shallow + test_preds_xgb_deep)


# ---------------------------------------------------------
# 8. NEURAL FT-TRANSFORMER ARCHITECTURE & LOSS FUNCTIONS
# ---------------------------------------------------------
class FeatureTokenizer(nn.Module):
    """Projects continuous tabular features into uniform token embedding space."""

    def __init__(self, num_numerical_features: int, d_token: int):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.d_token = d_token

        self.weight = nn.Parameter(torch.Tensor(num_numerical_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_numerical_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.d_token) if self.d_token > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerEncoderBlock(nn.Module):
    """Transformer block with Pre-LayerNorm, Multi-Head Attention, and Gated SwiGLU MLP."""

    def __init__(
        self,
        d_token: int,
        n_heads: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_token)
        hidden_dim = int(d_token * ffn_mult)
        self.w_gate = nn.Linear(d_token, hidden_dim)
        self.w_up = nn.Linear(d_token, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, d_token)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)

        norm_x = self.norm2(x)
        gate = F.silu(self.w_gate(norm_x))
        up = self.w_up(norm_x)
        ffn_out = self.w_down(gate * up)
        x = x + self.dropout2(ffn_out)
        return x


class TabularRiskTransformer(nn.Module):
    """Deep Tabular Attention Network with [CLS] risk aggregation for violation ranking."""

    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.15,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        b = tokens.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1)

        for layer in self.layers:
            x_seq = layer(x_seq)

        x_seq = self.final_norm(x_seq)
        cls_repr = x_seq[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)
        return logits


class SmoothAPLoss(nn.Module):
    """Differentiable surrogate of Average Precision (Soft-AP)."""

    def __init__(self, tau: float = 0.15):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        pos_mask = targets == 1
        n_pos = pos_mask.sum()
        if n_pos == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pos_logits = logits[pos_mask]
        all_logits = logits

        diff_all = all_logits.unsqueeze(0) - pos_logits.unsqueeze(1)
        sim_all = torch.sigmoid(diff_all / self.tau)

        diff_pos = pos_logits.unsqueeze(0) - pos_logits.unsqueeze(1)
        sim_pos = torch.sigmoid(diff_pos / self.tau)

        if weight is not None:
            w_all = weight.unsqueeze(0)
            w_pos = weight[pos_mask]
            rank_all = (
                1.0
                + torch.sum(sim_all * w_all, dim=1)
                - torch.sigmoid(torch.zeros(1, device=logits.device)) * w_pos
            )
            rank_pos = (
                1.0
                + torch.sum(sim_pos * w_pos.unsqueeze(0), dim=1)
                - torch.sigmoid(torch.zeros(1, device=logits.device)) * w_pos
            )
            precision_at_i = rank_pos / (rank_all + 1e-8)
            smooth_ap = torch.sum(precision_at_i * w_pos) / (
                torch.sum(w_pos) + 1e-8
            )
        else:
            rank_all = (
                1.0
                + torch.sum(sim_all, dim=1)
                - torch.sigmoid(torch.zeros(1, device=logits.device))
            )
            rank_pos = (
                1.0
                + torch.sum(sim_pos, dim=1)
                - torch.sigmoid(torch.zeros(1, device=logits.device))
            )
            precision_at_i = rank_pos / (rank_all + 1e-8)
            smooth_ap = torch.mean(precision_at_i)

        return 1.0 - smooth_ap


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Cross Entropy to prioritize hard positive violation risks."""

    def __init__(
        self,
        gamma_pos: float = 0.5,
        gamma_neg: float = 2.5,
        clip: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (1.0 - probs).clamp(min=self.eps, max=1.0 - self.eps)

        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = (
            targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        )
        loss_neg = (
            (1.0 - targets)
            * torch.pow(1.0 - probs_neg, self.gamma_neg)
            * torch.log(probs_neg)
        )
        loss_elem = -(loss_pos + loss_neg)

        if weight is not None:
            return torch.sum(loss_elem * weight) / (torch.sum(weight) + 1e-8)
        return torch.mean(loss_elem)


class HybridAPRankingLoss(nn.Module):
    """Combines Smooth-AP direct metric optimization with Asymmetric Focal anchor."""

    def __init__(
        self, ap_weight: float = 0.7, focal_weight: float = 0.3, tau: float = 0.15
    ):
        super().__init__()
        self.ap_weight = ap_weight
        self.focal_weight = focal_weight
        self.smooth_ap = SmoothAPLoss(tau=tau)
        self.focal = AsymmetricFocalLoss(gamma_pos=0.5, gamma_neg=2.5)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        loss_ap = self.smooth_ap(logits, targets, weight=weight)
        loss_focal = self.focal(logits, targets, weight=weight)
        return self.ap_weight * loss_ap + self.focal_weight * loss_focal


def build_model_and_optimizer(
    num_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    tau: float = 0.15,
    dev: torch.device = device,
    epochs: int = 8,
):
    model = TabularRiskTransformer(
        num_features=num_features,
        d_token=64,
        n_layers=3,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.15,
        head_dropout=0.2,
    ).to(dev)

    criterion = HybridAPRankingLoss(ap_weight=0.7, focal_weight=0.3, tau=tau).to(dev)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    warmup_epochs = 2
    try:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-5,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    except Exception:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs,
            T_mult=1,
            eta_min=1e-5,
        )

    return model, criterion, optimizer, scheduler


# ---------------------------------------------------------
# 9. NEURAL MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
print("Preparing tensor datasets and training FT-Transformer with log1p stabilization...")

X_tr_log = np.log1p(np.maximum(np.nan_to_num(X_train.values, nan=0.0), 0.0))
X_val_log = np.log1p(np.maximum(np.nan_to_num(X_val.values, nan=0.0), 0.0))
X_te_log = np.log1p(np.maximum(np.nan_to_num(X_test.values, nan=0.0), 0.0))

scaler = StandardScaler()
X_tr_norm = np.clip(scaler.fit_transform(X_tr_log), -10.0, 10.0)
X_val_norm = np.clip(scaler.transform(X_val_log), -10.0, 10.0)
X_te_norm = np.clip(scaler.transform(X_te_log), -10.0, 10.0)

train_dataset = TensorDataset(
    torch.tensor(X_tr_norm, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
    torch.tensor(sample_weight_train, dtype=torch.float32),
)
val_dataset = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_te_norm, dtype=torch.float32))

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


def predict_dataloader(net, loader, dev):
    net.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(dev)
            logits = net(x_b)
            all_preds.append(logits.detach().cpu().numpy())
    return np.concatenate(all_preds)


nn_model, criterion, optimizer, scheduler = build_model_and_optimizer(
    num_features=X_train.shape[1],
    lr=1e-3,
    weight_decay=1e-4,
    dev=device,
)

epochs = 8
best_val_ap = -1.0
best_model_path = "working/best_tabular_transformer.pt"

for epoch in range(epochs):
    nn_model.train()
    running_loss = 0.0
    total_batches = 0

    for x_batch, y_batch, w_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)

        optimizer.zero_grad()
        logits = nn_model(x_batch)
        loss = criterion(logits, y_batch, weight=w_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    scheduler.step()
    epoch_loss = running_loss / max(1, total_batches)

    val_preds_nn = predict_dataloader(nn_model, val_loader, device)
    current_val_ap = average_precision_score(y_val, val_preds_nn)

    if current_val_ap > best_val_ap:
        best_val_ap = current_val_ap
        torch.save(nn_model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val AP:"
        f" {current_val_ap:.5f}"
    )

# ---------------------------------------------------------
# 10. ENSEMBLING, VALIDATION EVALUATION & TEST INFERENCE
# ---------------------------------------------------------
print("Computing rank-normalized ensemble predictions...")
nn_model.load_state_dict(torch.load(best_model_path, map_location=device))
nn_model.eval()

# Bind standardized continuous predict method to nn_model with log1p transform
def nn_predict(X_input):
    nn_model.eval()
    if isinstance(X_input, pd.DataFrame):
        arr = X_input.values
    else:
        arr = X_input
    arr_log = np.log1p(np.maximum(np.nan_to_num(arr, nan=0.0), 0.0))
    arr_scaled = np.clip(scaler.transform(arr_log), -10.0, 10.0)
    eval_ds = TensorDataset(torch.tensor(arr_scaled, dtype=torch.float32))
    eval_ld = DataLoader(eval_ds, batch_size=2048, shuffle=False)
    return predict_dataloader(nn_model, eval_ld, device)

nn_model.predict = nn_predict
nn_model.predict_proba = lambda X: np.column_stack(
    [1.0 - 1.0 / (1.0 + np.exp(-nn_predict(X))), 1.0 / (1.0 + np.exp(-nn_predict(X)))]
)

val_preds_nn = predict_dataloader(nn_model, val_loader, device)
test_preds_nn = predict_dataloader(nn_model, test_loader, device)

val_rank_nn = rankdata(val_preds_nn) / len(val_preds_nn)
test_rank_nn = rankdata(test_preds_nn) / len(test_preds_nn)

val_rank_shallow = rankdata(val_preds_shallow) / len(val_preds_shallow)
test_rank_shallow = rankdata(test_preds_shallow) / len(test_preds_shallow)

val_rank_deep = rankdata(val_preds_deep) / len(val_preds_deep)
test_rank_deep = rankdata(test_preds_deep) / len(test_preds_deep)

val_rank_xgb_s = rankdata(val_preds_xgb_shallow) / len(val_preds_xgb_shallow)
test_rank_xgb_s = rankdata(test_preds_xgb_shallow) / len(test_preds_xgb_shallow)

val_rank_xgb_d = rankdata(val_preds_xgb_deep) / len(val_preds_xgb_deep)
test_rank_xgb_d = rankdata(test_preds_xgb_deep) / len(test_preds_xgb_deep)

val_rank_lgb = 0.5 * (val_rank_shallow + val_rank_deep)
test_rank_lgb = 0.5 * (test_rank_shallow + test_rank_deep)

val_rank_xgb = 0.5 * (val_rank_xgb_s + val_rank_xgb_d)
test_rank_xgb = 0.5 * (test_rank_xgb_s + test_rank_xgb_d)

shallow_val_ap = average_precision_score(y_val, val_rank_shallow)
deep_val_ap = average_precision_score(y_val, val_rank_deep)
lgb_val_ap = average_precision_score(y_val, val_rank_lgb)
xgb_s_val_ap = average_precision_score(y_val, val_rank_xgb_s)
xgb_d_val_ap = average_precision_score(y_val, val_rank_xgb_d)
xgb_val_ap = average_precision_score(y_val, val_rank_xgb)
nn_val_ap = average_precision_score(y_val, val_rank_nn)
print(
    f"Validation AP -> Shallow LGB: {shallow_val_ap:.5f}, Deep LGB: {deep_val_ap:.5f}, "
    f"Combined LGB: {lgb_val_ap:.5f}, Shallow XGB: {xgb_s_val_ap:.5f}, Deep XGB: {xgb_d_val_ap:.5f}, "
    f"Combined XGB: {xgb_val_ap:.5f}, FT-Transformer: {nn_val_ap:.5f}"
)

# 5-way simplex grid search for optimal ensemble weights
print("Optimizing 5-way ensemble weights over holdout validation AP...")
best_score = -1.0
best_weights = (0.20, 0.20, 0.20, 0.20, 0.20)
step_coarse = 0.10
N_c = int(round(1.0 / step_coarse))
for i in range(N_c + 1):
    for j in range(N_c + 1 - i):
        for k in range(N_c + 1 - i - j):
            for l in range(N_c + 1 - i - j - k):
                m = N_c - i - j - k - l
                w1, w2, w3, w4, w5 = (
                    i * step_coarse,
                    j * step_coarse,
                    k * step_coarse,
                    l * step_coarse,
                    m * step_coarse,
                )
                blend = (
                    w1 * val_rank_shallow
                    + w2 * val_rank_deep
                    + w3 * val_rank_xgb_s
                    + w4 * val_rank_xgb_d
                    + w5 * val_rank_nn
                )
                score = average_precision_score(y_val, blend)
                if score > best_score:
                    best_score = score
                    best_weights = (w1, w2, w3, w4, w5)

# Fine refinement around best coarse weights (+/- 0.06 with step = 0.02)
b1, b2, b3, b4, b5 = best_weights
fine_delta = 0.06
step_fine = 0.02
for dw1 in np.arange(-fine_delta, fine_delta + 1e-5, step_fine):
    for dw2 in np.arange(-fine_delta, fine_delta + 1e-5, step_fine):
        for dw3 in np.arange(-fine_delta, fine_delta + 1e-5, step_fine):
            for dw4 in np.arange(-fine_delta, fine_delta + 1e-5, step_fine):
                w1 = round(b1 + dw1, 4)
                w2 = round(b2 + dw2, 4)
                w3 = round(b3 + dw3, 4)
                w4 = round(b4 + dw4, 4)
                w5 = round(1.0 - (w1 + w2 + w3 + w4), 4)
                if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0 or w5 < 0:
                    continue
                blend = (
                    w1 * val_rank_shallow
                    + w2 * val_rank_deep
                    + w3 * val_rank_xgb_s
                    + w4 * val_rank_xgb_d
                    + w5 * val_rank_nn
                )
                score = average_precision_score(y_val, blend)
                if score > best_score:
                    best_score = score
                    best_weights = (w1, w2, w3, w4, w5)

w_lgb_s, w_lgb_d, w_xgb_s, w_xgb_d, w_nn = best_weights
print(
    f"Optimal 5-way weights: Shallow LGB={w_lgb_s:.2f}, Deep LGB={w_lgb_d:.2f}, "
    f"Shallow XGB={w_xgb_s:.2f}, Deep XGB={w_xgb_d:.2f}, NN={w_nn:.2f} -> Validation AP: {best_score:.5f}"
)

final_score = best_score
final_test_scores = (
    w_lgb_s * test_rank_shallow
    + w_lgb_d * test_rank_deep
    + w_xgb_s * test_rank_xgb_s
    + w_xgb_d * test_rank_xgb_d
    + w_nn * test_rank_nn
)

# ---------------------------------------------------------
# 11. SUBMISSION GENERATION & VALIDATION CHECKS
# ---------------------------------------------------------
submission_df = pd.DataFrame(
    {"bbl": df_test["bbl_str"].values, "score": final_test_scores}
)
submission_df.to_csv("submission/submission.csv", index=False)

assert (
    len(submission_df) == 171587
), f"Expected 171587 test samples, found {len(submission_df)}"
assert (
    submission_df["score"].isna().sum() == 0
), "Detected NaN values in test prediction scores!"
assert (
    submission_df["bbl"].isna().sum() == 0
), "Detected NaN values in submission BBL identifiers!"
assert (
    submission_df["bbl"].str.len() == 10
).all(), "Malformed BBL identifiers detected!"

print(f"Final Validation Score: {final_score}")

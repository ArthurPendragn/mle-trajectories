import copy
import gc
import glob
import json
import os
import sys
import gcsfs
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Setup directories & Storage Authentication
# ---------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
LAKE_FULL = f"{GCS_BASE}/lake/full"

DEFAULT_TOKEN = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
TOKEN = DEFAULT_TOKEN if os.path.exists(DEFAULT_TOKEN) else None
if TOKEN is None:
    found_tokens = glob.glob("/home/**/nyc-lake-agent-key.json", recursive=True)
    if found_tokens:
        TOKEN = found_tokens[0]
    elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and os.path.exists(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    ):
        TOKEN = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

storage_options = {"token": TOKEN} if TOKEN else {}
fs = gcsfs.GCSFileSystem(token=TOKEN) if TOKEN else gcsfs.GCSFileSystem()


def standardize_bbl(
    df, bbl_col="bbl", boro_col="boroid", block_col="block", lot_col="lot"
):
    """Standardizes BBL to a clean 10-character string:

    1 digit boro + 5 digit block + 4 digit lot. Follows official rule: use bbl
    if 10-digit, else construct from boro, block, lot.
    """
    if bbl_col in df.columns:
        s = df[bbl_col]
        if pd.api.types.is_float_dtype(s):
            bbl_str = s.fillna(0).astype("int64").astype(str)
        else:
            bbl_str = s.astype(str).str.split(".").str[0].str.strip()
        valid = (bbl_str.str.len() == 10) & (
            bbl_str.str[0].isin(["1", "2", "3", "4", "5"])
        )
    else:
        valid = pd.Series(False, index=df.index)
        bbl_str = pd.Series("", index=df.index)

    alt_boro_cols = [
        c for c in [boro_col, "borough", "borocode", "boro"] if c in df.columns
    ]
    alt_block_cols = [c for c in [block_col, "block"] if c in df.columns]
    alt_lot_cols = [c for c in [lot_col, "lot"] if c in df.columns]

    if (~valid).any() and alt_boro_cols and alt_block_cols and alt_lot_cols:
        b_col, blk_col, lt_col = (
            alt_boro_cols[0],
            alt_block_cols[0],
            alt_lot_cols[0],
        )
        idx = ~valid
        boro_val = df.loc[idx, b_col]
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
        if pd.api.types.is_string_dtype(boro_val) or pd.api.types.is_object_dtype(
            boro_val
        ):
            boro_digit = boro_val.astype(str).str.upper().map(boro_map)
            boro_digit = boro_digit.fillna(
                pd.to_numeric(boro_val, errors="coerce")
                .fillna(0)
                .astype("int64")
                .astype(str)
            )
        else:
            boro_digit = (
                pd.to_numeric(boro_val, errors="coerce")
                .fillna(0)
                .astype("int64")
                .astype(str)
            )

        block_val = (
            pd.to_numeric(df.loc[idx, blk_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .apply(lambda x: f"{x:05d}")
        )
        lot_val = (
            pd.to_numeric(df.loc[idx, lt_col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .apply(lambda x: f"{x:04d}")
        )

        constructed = boro_digit + block_val + lot_val
        bbl_str = bbl_str.copy()
        bbl_str.loc[idx] = constructed

    return bbl_str


def get_table_schema(table_name, gcs_path=None):
    """Dynamically resolves column names for a lake table using PyArrow schema or DATA_DICTIONARY.md."""
    if gcs_path:
        clean_path = gcs_path.replace("gs://", "")
        try:
            ds = pq.ParquetDataset(clean_path, filesystem=fs)
            if ds.schema and ds.schema.names:
                return list(ds.schema.names)
        except Exception:
            pass

    for dict_path in ["./input/DATA_DICTIONARY.md", "DATA_DICTIONARY.md"]:
        if os.path.exists(dict_path):
            try:
                with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                import re
                pattern = re.compile(
                    rf"#+\s*[`*_]*{re.escape(table_name)}[`*_]*\b(.*?)(?=\n#+ |\Z)",
                    re.DOTALL | re.IGNORECASE,
                )
                m = pattern.search(content)
                if m:
                    sec = m.group(1)
                    cols = re.findall(r"-\s*`([^`]+)`", sec)
                    if cols:
                        return cols
            except Exception:
                pass
    return []


def find_matching_col(candidates, available_cols):
    """Finds the first candidate column present in available_cols (case-insensitive)."""
    avail_lower = {c.lower(): c for c in available_cols}
    for cand in candidates:
        if cand.lower() in avail_lower:
            return avail_lower[cand.lower()]
    return None


def safe_read_parquet(gcs_path, preferred_cols):
    """Safely reads parquet dataset from GCS by projecting only available columns."""
    clean_path = gcs_path.replace("gs://", "")
    try:
        dataset = pq.ParquetDataset(clean_path, filesystem=fs)
        avail_map = {c.lower(): c for c in dataset.schema.names}
        cols_to_load = [avail_map[c.lower()] for c in preferred_cols if c.lower() in avail_map]
        if not cols_to_load:
            cols_to_load = None
        df = pd.read_parquet(
            gcs_path, columns=cols_to_load, storage_options=storage_options
        )
        return df
    except Exception:
        df = pd.read_parquet(gcs_path, storage_options=storage_options)
        avail_map = {c.lower(): c for c in df.columns}
        cols = [avail_map[c.lower()] for c in preferred_cols if c.lower() in avail_map]
        return df[cols] if cols else df


# ---------------------------------------------------------------------------
# 1. Load Entities & PLUTO Building Morphology
# ---------------------------------------------------------------------------
test_entities_df = pd.read_parquet(
    f"{GCS_BASE}/test_entities.parquet", storage_options=storage_options
)
test_entities_df["bbl"] = standardize_bbl(test_entities_df)
test_bbls = test_entities_df["bbl"].unique()

pluto_cols = [
    "bbl",
    "borocode",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "bldgclass",
    "numfloors",
    "bldgarea",
    "resarea",
    "lotarea",
    "cd",
    "version",
]
df_pluto = safe_read_parquet(f"{LAKE_FULL}/pluto", pluto_cols)
df_pluto["bbl"] = standardize_bbl(df_pluto)
df_pluto = df_pluto[df_pluto["bbl"].str.len() == 10].copy()

if "version" in df_pluto.columns:
    df_pluto = df_pluto.sort_values("version").drop_duplicates("bbl", keep="last")
else:
    df_pluto = df_pluto.drop_duplicates("bbl", keep="last")

df_pluto["unitsres"] = (
    pd.to_numeric(df_pluto["unitsres"], errors="coerce").fillna(1.0).clip(lower=1.0)
)
df_pluto["unitstotal"] = (
    pd.to_numeric(df_pluto["unitstotal"], errors="coerce").fillna(1.0).clip(lower=1.0)
)
df_pluto["yearbuilt"] = (
    pd.to_numeric(df_pluto["yearbuilt"], errors="coerce")
    .fillna(1950.0)
    .clip(lower=1800, upper=2023)
)
df_pluto["numfloors"] = (
    pd.to_numeric(df_pluto["numfloors"], errors="coerce").fillna(3.0).clip(lower=1.0)
)
df_pluto["bldgarea"] = (
    pd.to_numeric(df_pluto["bldgarea"], errors="coerce").fillna(0.0).clip(lower=0.0)
)
df_pluto["resarea"] = (
    pd.to_numeric(df_pluto["resarea"], errors="coerce").fillna(0.0).clip(lower=0.0)
)
df_pluto["lotarea"] = (
    pd.to_numeric(df_pluto["lotarea"], errors="coerce").fillna(0.0).clip(lower=0.0)
)
df_pluto["borocode"] = (
    pd.to_numeric(df_pluto["borocode"], errors="coerce")
    .fillna(df_pluto["bbl"].str[0].astype(float))
    .fillna(1.0)
)
df_pluto["cd"] = pd.to_numeric(df_pluto["cd"], errors="coerce").fillna(
    df_pluto["borocode"] * 100 + 1.0
)
df_pluto["bldgclass_first"] = df_pluto["bldgclass"].astype(str).str[0].fillna("C")

all_pluto_cds = sorted([float(x) for x in df_pluto["cd"].dropna().unique() if x > 0])
cd_to_idx = {cd_val: i for i, cd_val in enumerate(all_pluto_cds)}

entity_universe_df = df_pluto[df_pluto["unitsres"] >= 3].copy()
universe_bbls = set(entity_universe_df["bbl"]).union(set(test_bbls))

pluto_features = df_pluto.set_index("bbl")[
    [
        "borocode",
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "numfloors",
        "bldgarea",
        "resarea",
        "lotarea",
        "cd",
        "bldgclass_first",
    ]
].to_dict("index")

del df_pluto
gc.collect()

# ---------------------------------------------------------------------------
# 2. Load Administrative Distress & Auxiliary Datasets
# ---------------------------------------------------------------------------
try:
    aep_schema = get_table_schema("hpd_aep_buildings", f"{LAKE_FULL}/hpd_aep_buildings")
    aep_cols = [c for c in ["bbl", "boroid", "boro", "block", "lot"] if find_matching_col([c], aep_schema)]
    df_aep = pd.read_parquet(
        f"{LAKE_FULL}/hpd_aep_buildings",
        columns=aep_cols if aep_cols else None,
        storage_options=storage_options,
    )
    df_aep["bbl"] = standardize_bbl(df_aep)
    aep_bbl_set = set(df_aep["bbl"].unique())
    del df_aep
except Exception:
    aep_bbl_set = set()

try:
    vacate_schema = get_table_schema("hpd_vacate_orders", f"{LAKE_FULL}/hpd_vacate_orders")
    vacate_date_col = find_matching_col(
        ["vacate_effective_date", "effective_date", "vacate_date", "vacatedate"],
        vacate_schema,
    )
    vacate_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "block", "lot"]:
        m = find_matching_col([c_cand], vacate_schema)
        if m and m not in vacate_cols_to_load:
            vacate_cols_to_load.append(m)
    if vacate_date_col and vacate_date_col not in vacate_cols_to_load:
        vacate_cols_to_load.append(vacate_date_col)

    df_vacate = pd.read_parquet(
        f"{LAKE_FULL}/hpd_vacate_orders",
        columns=vacate_cols_to_load if vacate_cols_to_load else None,
        storage_options=storage_options,
    )
    df_vacate["bbl"] = standardize_bbl(df_vacate)
    if vacate_date_col and vacate_date_col in df_vacate.columns:
        df_vacate["vacate_effective_date"] = pd.to_datetime(df_vacate[vacate_date_col], format="mixed", errors="coerce")
    else:
        df_vacate["vacate_effective_date"] = pd.NaT
    df_vacate = df_vacate[df_vacate["bbl"].isin(universe_bbls)][["bbl", "vacate_effective_date"]].copy()
except Exception:
    df_vacate = pd.DataFrame(columns=["bbl", "vacate_effective_date"])

try:
    lit_schema = get_table_schema("hpd_litigations", f"{LAKE_FULL}/hpd_litigations")
    lit_date_col = find_matching_col(
        ["caseopendate", "case_open_date", "case_date", "date_opened", "opendate"],
        lit_schema,
    )
    lit_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "block", "lot"]:
        m = find_matching_col([c_cand], lit_schema)
        if m and m not in lit_cols_to_load:
            lit_cols_to_load.append(m)
    if lit_date_col and lit_date_col not in lit_cols_to_load:
        lit_cols_to_load.append(lit_date_col)

    df_lit = pd.read_parquet(
        f"{LAKE_FULL}/hpd_litigations",
        columns=lit_cols_to_load if lit_cols_to_load else None,
        storage_options=storage_options,
    )
    df_lit["bbl"] = standardize_bbl(df_lit)
    if lit_date_col and lit_date_col in df_lit.columns:
        df_lit["caseopendate"] = pd.to_datetime(df_lit[lit_date_col], format="mixed", errors="coerce")
    else:
        df_lit["caseopendate"] = pd.NaT
    df_lit = df_lit[df_lit["bbl"].isin(universe_bbls)][["bbl", "caseopendate"]].copy()
except Exception:
    df_lit = pd.DataFrame(columns=["bbl", "caseopendate"])

try:
    complaint_schema = get_table_schema("hpd_complaints", f"{LAKE_FULL}/hpd_complaints")
    complaint_date_col = find_matching_col(
        [
            "receiveddate", "received_date", "date_received",
            "status_date", "statusdate", "complaint_date",
            "complaintdate", "date_entered", "entereddate",
        ],
        complaint_schema,
    )
    if not complaint_date_col:
        complaint_date_col = next((c for c in complaint_schema if "date" in c.lower()), None)

    complaint_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "block", "lot"]:
        m = find_matching_col([c_cand], complaint_schema)
        if m and m not in complaint_cols_to_load:
            complaint_cols_to_load.append(m)
    if complaint_date_col and complaint_date_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(complaint_date_col)

    df_complaints = pd.read_parquet(
        f"{LAKE_FULL}/hpd_complaints",
        columns=complaint_cols_to_load if complaint_cols_to_load else None,
        storage_options=storage_options,
    )
    df_complaints["bbl"] = standardize_bbl(df_complaints)
    if complaint_date_col and complaint_date_col in df_complaints.columns:
        df_complaints["receiveddate"] = pd.to_datetime(df_complaints[complaint_date_col], format="mixed", errors="coerce")
    else:
        df_complaints["receiveddate"] = pd.NaT

    df_complaints = df_complaints[
        (df_complaints["bbl"].isin(universe_bbls))
        & (df_complaints["receiveddate"] >= "2016-01-01")
    ][["bbl", "receiveddate"]].copy()
except Exception:
    df_complaints = pd.DataFrame(columns=["bbl", "receiveddate"])

# Load Emergency Repair Program (ERP) records from HWO and OMO charges
erp_dfs = []
for erp_tbl in ["hpd_hwo_charges", "hpd_omo_charges"]:
    try:
        tbl_schema = get_table_schema(erp_tbl, f"{LAKE_FULL}/{erp_tbl}")
        date_col = find_matching_col(
            [
                "invoicedate", "invoice_date", "chargedate", "charge_date",
                "orderdate", "order_date", "issuedate", "issue_date",
                "servicedate", "service_date",
            ],
            tbl_schema,
        )
        if not date_col:
            date_col = next(
                (
                    c for c in tbl_schema
                    if "date" in c.lower()
                    and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
                ),
                None,
            )
        cols_to_load = []
        for c_cand in ["bbl", "boroid", "borough", "block", "lot"]:
            m = find_matching_col([c_cand], tbl_schema)
            if m and m not in cols_to_load:
                cols_to_load.append(m)
        if date_col and date_col not in cols_to_load:
            cols_to_load.append(date_col)

        df_temp = pd.read_parquet(
            f"{LAKE_FULL}/{erp_tbl}",
            columns=cols_to_load if cols_to_load else None,
            storage_options=storage_options,
        )
        df_temp["bbl"] = standardize_bbl(df_temp)
        if date_col and date_col in df_temp.columns:
            df_temp["charge_date"] = pd.to_datetime(df_temp[date_col], format="mixed", errors="coerce")
        else:
            df_temp["charge_date"] = pd.NaT
        df_temp = df_temp[
            df_temp["bbl"].isin(universe_bbls)
            & df_temp["charge_date"].notna()
            & (df_temp["charge_date"] >= "2016-01-01")
        ][["bbl", "charge_date"]].copy()
        erp_dfs.append(df_temp)
    except Exception:
        pass

if erp_dfs:
    df_erp = pd.concat(erp_dfs, ignore_index=True)
else:
    df_erp = pd.DataFrame(columns=["bbl", "charge_date"])
del erp_dfs
gc.collect()

# Load Department of Buildings (DOB) Violations
try:
    dob_schema = get_table_schema("dob_violations", f"{LAKE_FULL}/dob_violations")
    dob_date_col = find_matching_col(
        [
            "issue_date", "issuedate", "nov_issued_date", "violation_date",
            "issue_dt", "date", "inspection_date", "novissue_date", "novissueddate",
        ],
        dob_schema,
    )
    if not dob_date_col:
        dob_date_col = next(
            (c for c in dob_schema if "issue" in c.lower() and "date" in c.lower()),
            None,
        )
    if not dob_date_col:
        dob_date_col = next(
            (
                c for c in dob_schema
                if "date" in c.lower()
                and not any(
                    bad in c.lower()
                    for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"]
                )
            ),
            None,
        )

    dob_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], dob_schema)
        if m and m not in dob_cols_to_load:
            dob_cols_to_load.append(m)
    if dob_date_col and dob_date_col not in dob_cols_to_load:
        dob_cols_to_load.append(dob_date_col)

    df_dob = pd.read_parquet(
        f"{LAKE_FULL}/dob_violations",
        columns=dob_cols_to_load if dob_cols_to_load else None,
        storage_options=storage_options,
    )
    df_dob["bbl"] = standardize_bbl(df_dob)
    if dob_date_col and dob_date_col in df_dob.columns:
        df_dob["issue_date"] = pd.to_datetime(df_dob[dob_date_col], format="mixed", errors="coerce")
    else:
        df_dob["issue_date"] = pd.NaT

    if hasattr(df_dob["issue_date"], "dt") and df_dob["issue_date"].dt.tz is not None:
        try:
            df_dob["issue_date"] = df_dob["issue_date"].dt.tz_localize(None)
        except TypeError:
            df_dob["issue_date"] = df_dob["issue_date"].dt.tz_convert(None)

    df_dob = df_dob[
        (df_dob["bbl"].isin(universe_bbls))
        & (df_dob["issue_date"].notna())
        & (df_dob["issue_date"] >= "2010-01-01")
    ][["bbl", "issue_date"]].copy()
except Exception:
    df_dob = pd.DataFrame(columns=["bbl", "issue_date"])

# Load DOHMH Rodent Inspections
try:
    rodent_schema = get_table_schema("dohmh_rodent_inspections", f"{LAKE_FULL}/dohmh_rodent_inspections")
    rodent_date_col = find_matching_col(
        [
            "inspection_date", "inspectiondate", "approved_date", "approveddate",
            "inspection_date_time", "date", "date_inspected", "created_date",
        ],
        rodent_schema,
    )
    if not rodent_date_col:
        rodent_date_col = next(
            (c for c in rodent_schema if "inspect" in c.lower() and "date" in c.lower()),
            None,
        )
    if not rodent_date_col:
        rodent_date_col = next(
            (
                c for c in rodent_schema
                if "date" in c.lower()
                and not any(
                    bad in c.lower()
                    for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"]
                )
            ),
            None,
        )

    rodent_result_col = find_matching_col(
        ["result", "inspection_result", "result_type", "type", "rodent_result"],
        rodent_schema,
    )

    rodent_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], rodent_schema)
        if m and m not in rodent_cols_to_load:
            rodent_cols_to_load.append(m)
    if rodent_date_col and rodent_date_col not in rodent_cols_to_load:
        rodent_cols_to_load.append(rodent_date_col)
    if rodent_result_col and rodent_result_col not in rodent_cols_to_load:
        rodent_cols_to_load.append(rodent_result_col)

    df_rodent = pd.read_parquet(
        f"{LAKE_FULL}/dohmh_rodent_inspections",
        columns=rodent_cols_to_load if rodent_cols_to_load else None,
        storage_options=storage_options,
    )
    df_rodent["bbl"] = standardize_bbl(df_rodent)
    if rodent_date_col and rodent_date_col in df_rodent.columns:
        df_rodent["inspection_date"] = pd.to_datetime(df_rodent[rodent_date_col], format="mixed", errors="coerce")
    else:
        df_rodent["inspection_date"] = pd.NaT

    if hasattr(df_rodent["inspection_date"], "dt") and df_rodent["inspection_date"].dt.tz is not None:
        try:
            df_rodent["inspection_date"] = df_rodent["inspection_date"].dt.tz_localize(None)
        except TypeError:
            df_rodent["inspection_date"] = df_rodent["inspection_date"].dt.tz_convert(None)

    if rodent_result_col and rodent_result_col in df_rodent.columns:
        df_rodent["is_active"] = (
            df_rodent[rodent_result_col]
            .astype(str)
            .str.contains("Active|Fail|Rat|Problem", case=False, na=False)
            .astype(float)
        )
    else:
        df_rodent["is_active"] = 0.0

    df_rodent = df_rodent[
        (df_rodent["bbl"].isin(universe_bbls))
        & (df_rodent["inspection_date"].notna())
        & (df_rodent["inspection_date"] >= "2010-01-01")
    ][["bbl", "inspection_date", "is_active"]].copy()
except Exception:
    df_rodent = pd.DataFrame(columns=["bbl", "inspection_date", "is_active"])

# ---------------------------------------------------------------------------
# 3. Load HPD Violations (Core Signal)
# ---------------------------------------------------------------------------
viol_schema = get_table_schema("hpd_violations", f"{LAKE_FULL}/hpd_violations")
insp_date_col = find_matching_col(
    ["inspectiondate", "inspection_date", "inspection_date_time", "novissueddate"],
    viol_schema,
) or "inspectiondate"

status_date_col = find_matching_col(
    ["currentstatusdate", "current_status_date", "status_date", "statusdate"],
    viol_schema,
)

class_col = find_matching_col(["class", "violationstatus", "violation_class"], viol_schema) or "class"

viol_cols_to_load = []
for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
    m = find_matching_col([c_cand], viol_schema)
    if m and m not in viol_cols_to_load:
        viol_cols_to_load.append(m)

for c in [class_col, insp_date_col, status_date_col]:
    if c and c in viol_schema and c not in viol_cols_to_load:
        viol_cols_to_load.append(c)

df_violations = pd.read_parquet(
    f"{LAKE_FULL}/hpd_violations",
    columns=viol_cols_to_load if viol_cols_to_load else None,
    storage_options=storage_options,
)
df_violations["bbl"] = standardize_bbl(df_violations)

if class_col in df_violations.columns:
    df_violations["class"] = df_violations[class_col].astype(str).str.upper().str.strip()
else:
    df_violations["class"] = ""

if insp_date_col in df_violations.columns:
    df_violations["inspectiondate"] = pd.to_datetime(df_violations[insp_date_col], format="mixed", errors="coerce")
    if hasattr(df_violations["inspectiondate"], "dt") and df_violations["inspectiondate"].dt.tz is not None:
        try:
            df_violations["inspectiondate"] = df_violations["inspectiondate"].dt.tz_localize(None)
        except TypeError:
            df_violations["inspectiondate"] = df_violations["inspectiondate"].dt.tz_convert(None)
else:
    df_violations["inspectiondate"] = pd.NaT

if status_date_col and status_date_col in df_violations.columns:
    df_violations["currentstatusdate"] = pd.to_datetime(df_violations[status_date_col], format="mixed", errors="coerce")
    if hasattr(df_violations["currentstatusdate"], "dt") and df_violations["currentstatusdate"].dt.tz is not None:
        try:
            df_violations["currentstatusdate"] = df_violations["currentstatusdate"].dt.tz_localize(None)
        except TypeError:
            df_violations["currentstatusdate"] = df_violations["currentstatusdate"].dt.tz_convert(None)
else:
    df_violations["currentstatusdate"] = pd.NaT

df_violations = df_violations[
    (df_violations["bbl"].isin(universe_bbls))
    & (df_violations["inspectiondate"] >= "2016-01-01")
][["bbl", "class", "inspectiondate", "currentstatusdate"]].copy()
gc.collect()


# ---------------------------------------------------------------------------
# 4. Point-In-Time Feature Extraction Engine
# ---------------------------------------------------------------------------
def compute_cohort_dataset(cutoff_str, target_bbl_list, is_test=False):
    cutoff = pd.Timestamp(cutoff_str)
    cutoff_30d_prior = cutoff - pd.DateOffset(days=30)
    cutoff_60d_prior = cutoff - pd.DateOffset(days=60)
    cutoff_90d_prior = cutoff - pd.DateOffset(days=90)
    cutoff_180d_prior = cutoff - pd.DateOffset(days=180)
    cutoff_1y_prior = cutoff - pd.DateOffset(days=365)
    cutoff_2y_prior = cutoff - pd.DateOffset(days=730)
    cutoff_3y_prior = cutoff - pd.DateOffset(days=1095)

    df_cohort = (
        pd.DataFrame({"bbl": target_bbl_list})
        .drop_duplicates("bbl")
        .reset_index(drop=True)
    )
    bbl_set = set(df_cohort["bbl"])

    v_hist = df_violations[
        (df_violations["bbl"].isin(bbl_set))
        & (df_violations["inspectiondate"] < cutoff)
    ]

    v_c = v_hist[v_hist["class"] == "C"]
    v_b = v_hist[v_hist["class"] == "B"]
    v_a = v_hist[v_hist["class"] == "A"]

    # Short and multi-year horizon Class C windows
    c_30d = (
        v_c[v_c["inspectiondate"] >= cutoff_30d_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_30d")
    )
    c_90d = (
        v_c[v_c["inspectiondate"] >= cutoff_90d_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_90d")
    )
    c_180d = (
        v_c[v_c["inspectiondate"] >= cutoff_180d_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_180d")
    )
    c_1y = (
        v_c[v_c["inspectiondate"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_1y")
    )
    c_2y = (
        v_c[v_c["inspectiondate"] >= cutoff_2y_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_2y")
    )
    c_3y = (
        v_c[v_c["inspectiondate"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_3y")
    )
    c_all = v_c.groupby("bbl").size().rename("c_viol_all")

    # Short and multi-year horizon Class B windows
    b_30d = (
        v_b[v_b["inspectiondate"] >= cutoff_30d_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_30d")
    )
    b_90d = (
        v_b[v_b["inspectiondate"] >= cutoff_90d_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_90d")
    )
    b_180d = (
        v_b[v_b["inspectiondate"] >= cutoff_180d_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_180d")
    )
    b_1y = (
        v_b[v_b["inspectiondate"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_1y")
    )
    b_2y = (
        v_b[v_b["inspectiondate"] >= cutoff_2y_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_2y")
    )
    b_3y = (
        v_b[v_b["inspectiondate"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("b_viol_3y")
    )
    b_all = v_b.groupby("bbl").size().rename("b_viol_all")

    a_1y = (
        v_a[v_a["inspectiondate"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("a_viol_1y")
    )
    a_2y = (
        v_a[v_a["inspectiondate"] >= cutoff_2y_prior]
        .groupby("bbl")
        .size()
        .rename("a_viol_2y")
    )

    # Short and multi-year total violation windows
    tot_30d = (
        v_hist[v_hist["inspectiondate"] >= cutoff_30d_prior]
        .groupby("bbl")
        .size()
        .rename("total_viol_30d")
    )
    tot_90d = (
        v_hist[v_hist["inspectiondate"] >= cutoff_90d_prior]
        .groupby("bbl")
        .size()
        .rename("total_viol_90d")
    )
    tot_180d = (
        v_hist[v_hist["inspectiondate"] >= cutoff_180d_prior]
        .groupby("bbl")
        .size()
        .rename("total_viol_180d")
    )
    tot_all = v_hist.groupby("bbl").size().rename("total_viol_all")

    # Exponential recency-decayed violation intensity scores (180d and 365d half-lives)
    if len(v_c) > 0:
        c_dt_days = ((cutoff - v_c["inspectiondate"]).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_c_180 = np.exp(-0.69314718056 * c_dt_days / 180.0)
        w_c_365 = np.exp(-0.69314718056 * c_dt_days / 365.0)
        c_decay_180 = w_c_180.groupby(v_c["bbl"]).sum().rename("c_decay_180d")
        c_decay_365 = w_c_365.groupby(v_c["bbl"]).sum().rename("c_decay_365d")
    else:
        c_decay_180 = pd.Series(dtype=float, name="c_decay_180d")
        c_decay_365 = pd.Series(dtype=float, name="c_decay_365d")

    if len(v_hist) > 0:
        tot_dt_days = ((cutoff - v_hist["inspectiondate"]).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_tot_180 = np.exp(-0.69314718056 * tot_dt_days / 180.0)
        w_tot_365 = np.exp(-0.69314718056 * tot_dt_days / 365.0)
        tot_decay_180 = w_tot_180.groupby(v_hist["bbl"]).sum().rename("total_decay_180d")
        tot_decay_365 = w_tot_365.groupby(v_hist["bbl"]).sum().rename("total_decay_365d")
    else:
        tot_decay_180 = pd.Series(dtype=float, name="total_decay_180d")
        tot_decay_365 = pd.Series(dtype=float, name="total_decay_365d")

    # Multi-year violation persistence: distinct calendar years with Class C citations
    if len(v_c) > 0:
        v_c_years = v_c.assign(insp_year=v_c["inspectiondate"].dt.year)
        c_years_with_viol = (
            v_c_years.groupby("bbl")["insp_year"].nunique().rename("c_years_with_viol")
        )
    else:
        c_years_with_viol = pd.Series(dtype=int, name="c_years_with_viol")

    # Hazardous open violation backlog duration and lingering count (>90 days unresolved)
    open_c_records = v_c[
        (v_c["currentstatusdate"].isna()) | (v_c["currentstatusdate"] >= cutoff)
    ]
    open_c = open_c_records.groupby("bbl").size().rename("open_c_viol")

    if len(open_c_records) > 0:
        open_c_age_days = (
            (cutoff - open_c_records["inspectiondate"]).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        open_c_max_age_days = (
            open_c_age_days.groupby(open_c_records["bbl"]).max().rename("open_c_max_age_days")
        )
        open_c_lingering_90d = (
            (open_c_age_days > 90.0)
            .astype(float)
            .groupby(open_c_records["bbl"])
            .sum()
            .rename("open_c_lingering_90d")
        )
    else:
        open_c_max_age_days = pd.Series(dtype=float, name="open_c_max_age_days")
        open_c_lingering_90d = pd.Series(dtype=float, name="open_c_lingering_90d")

    open_b = (
        v_b[(v_b["currentstatusdate"].isna()) | (v_b["currentstatusdate"] >= cutoff)]
        .groupby("bbl")
        .size()
        .rename("open_b_viol")
    )

    if len(v_c) > 0:
        recency_c = ((cutoff - v_c.groupby("bbl")["inspectiondate"].max()).dt.days).rename(
            "days_since_last_c"
        )
    else:
        recency_c = pd.Series(dtype=float, name="days_since_last_c")

    if len(v_hist) > 0:
        recency_any = (
            (cutoff - v_hist.groupby("bbl")["inspectiondate"].max()).dt.days
        ).rename("days_since_last_any")
    else:
        recency_any = pd.Series(dtype=float, name="days_since_last_any")

    # Winter heating season surge (October 1 to December 31 = cutoff minus 92 days)
    cutoff_winter_prior = cutoff - pd.DateOffset(days=92)
    c_winter = (
        v_c[v_c["inspectiondate"] >= cutoff_winter_prior]
        .groupby("bbl")
        .size()
        .rename("c_viol_winter")
    )

    # Complaints features
    c_hist = df_complaints[
        (df_complaints["bbl"].isin(bbl_set)) & (df_complaints["receiveddate"] < cutoff)
    ]
    comp_30d = (
        c_hist[c_hist["receiveddate"] >= cutoff_30d_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_30d")
    )
    comp_60d = (
        c_hist[c_hist["receiveddate"] >= cutoff_60d_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_60d")
    )
    comp_90d = (
        c_hist[c_hist["receiveddate"] >= cutoff_90d_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_90d")
    )
    comp_1y = (
        c_hist[c_hist["receiveddate"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_1y")
    )
    comp_2y = (
        c_hist[c_hist["receiveddate"] >= cutoff_2y_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_2y")
    )
    comp_3y = (
        c_hist[c_hist["receiveddate"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_3y")
    )
    comp_winter = (
        c_hist[c_hist["receiveddate"] >= cutoff_winter_prior]
        .groupby("bbl")
        .size()
        .rename("complaints_winter")
    )
    if len(c_hist) > 0:
        recency_comp = (
            (cutoff - c_hist.groupby("bbl")["receiveddate"].max()).dt.days
        ).rename("days_since_last_comp")
    else:
        recency_comp = pd.Series(dtype=float, name="days_since_last_comp")

    # ERP charges
    erp_hist = df_erp[
        (df_erp["bbl"].isin(bbl_set)) & (df_erp["charge_date"] < cutoff)
    ]
    erp_1y = (
        erp_hist[erp_hist["charge_date"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("erp_charges_1y")
    )
    erp_3y = (
        erp_hist[erp_hist["charge_date"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("erp_charges_3y")
    )
    erp_total = (
        erp_hist.groupby("bbl")
        .size()
        .rename("erp_charges_total")
    )

    # Vacate orders
    vacate_cnt = (
        df_vacate[
            (df_vacate["bbl"].isin(bbl_set))
            & (df_vacate["vacate_effective_date"] < cutoff)
        ]
        .groupby("bbl")
        .size()
        .rename("vacate_orders_hist")
    )

    # Litigations
    lit_1y = (
        df_lit[
            (df_lit["bbl"].isin(bbl_set))
            & (df_lit["caseopendate"] >= cutoff_1y_prior)
            & (df_lit["caseopendate"] < cutoff)
        ]
        .groupby("bbl")
        .size()
        .rename("litigations_1y")
    )
    lit_3y = (
        df_lit[
            (df_lit["bbl"].isin(bbl_set))
            & (df_lit["caseopendate"] >= cutoff_3y_prior)
            & (df_lit["caseopendate"] < cutoff)
        ]
        .groupby("bbl")
        .size()
        .rename("litigations_3y")
    )

    # DOB Violations
    dob_hist = df_dob[
        (df_dob["bbl"].isin(bbl_set)) & (df_dob["issue_date"] < cutoff)
    ]
    dob_1y = (
        dob_hist[dob_hist["issue_date"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("dob_viol_1y")
    )
    dob_3y = (
        dob_hist[dob_hist["issue_date"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("dob_viol_3y")
    )
    dob_all = dob_hist.groupby("bbl").size().rename("dob_viol_all")
    if len(dob_hist) > 0:
        recency_dob = (
            (cutoff - dob_hist.groupby("bbl")["issue_date"].max()).dt.days
        ).rename("days_since_last_dob")
    else:
        recency_dob = pd.Series(dtype=float, name="days_since_last_dob")

    # DOHMH Rodent Inspections
    rodent_hist = df_rodent[
        (df_rodent["bbl"].isin(bbl_set)) & (df_rodent["inspection_date"] < cutoff)
    ]
    rodent_1y = (
        rodent_hist[rodent_hist["inspection_date"] >= cutoff_1y_prior]
        .groupby("bbl")
        .size()
        .rename("rodent_insp_1y")
    )
    rodent_3y = (
        rodent_hist[rodent_hist["inspection_date"] >= cutoff_3y_prior]
        .groupby("bbl")
        .size()
        .rename("rodent_insp_3y")
    )
    rodent_all = rodent_hist.groupby("bbl").size().rename("rodent_insp_all")
    if len(rodent_hist) > 0:
        recency_rodent = (
            (cutoff - rodent_hist.groupby("bbl")["inspection_date"].max()).dt.days
        ).rename("days_since_last_rodent")
    else:
        recency_rodent = pd.Series(dtype=float, name="days_since_last_rodent")
    rodent_active_1y = (
        rodent_hist[(rodent_hist["inspection_date"] >= cutoff_1y_prior) & (rodent_hist["is_active"] > 0)]
        .groupby("bbl")
        .size()
        .rename("rodent_active_1y")
    )
    rodent_active_all = (
        rodent_hist[rodent_hist["is_active"] > 0]
        .groupby("bbl")
        .size()
        .rename("rodent_active_all")
    )

    feature_dict = {
        "c_viol_30d": c_30d,
        "c_viol_90d": c_90d,
        "c_viol_180d": c_180d,
        "c_viol_1y": c_1y,
        "c_viol_2y": c_2y,
        "c_viol_3y": c_3y,
        "c_viol_all": c_all,
        "b_viol_30d": b_30d,
        "b_viol_90d": b_90d,
        "b_viol_180d": b_180d,
        "b_viol_1y": b_1y,
        "b_viol_2y": b_2y,
        "b_viol_3y": b_3y,
        "b_viol_all": b_all,
        "a_viol_1y": a_1y,
        "a_viol_2y": a_2y,
        "total_viol_30d": tot_30d,
        "total_viol_90d": tot_90d,
        "total_viol_180d": tot_180d,
        "total_viol_all": tot_all,
        "c_decay_180d": c_decay_180,
        "c_decay_365d": c_decay_365,
        "total_decay_180d": tot_decay_180,
        "total_decay_365d": tot_decay_365,
        "c_years_with_viol": c_years_with_viol,
        "open_c_viol": open_c,
        "open_c_max_age_days": open_c_max_age_days,
        "open_c_lingering_90d": open_c_lingering_90d,
        "open_b_viol": open_b,
        "days_since_last_c": recency_c,
        "days_since_last_any": recency_any,
        "c_viol_winter": c_winter,
        "complaints_30d": comp_30d,
        "complaints_60d": comp_60d,
        "complaints_90d": comp_90d,
        "complaints_1y": comp_1y,
        "complaints_2y": comp_2y,
        "complaints_3y": comp_3y,
        "complaints_winter": comp_winter,
        "days_since_last_comp": recency_comp,
        "erp_charges_1y": erp_1y,
        "erp_charges_3y": erp_3y,
        "erp_charges_total": erp_total,
        "vacate_orders_hist": vacate_cnt,
        "litigations_1y": lit_1y,
        "litigations_3y": lit_3y,
        "dob_viol_1y": dob_1y,
        "dob_viol_3y": dob_3y,
        "dob_viol_all": dob_all,
        "days_since_last_dob": recency_dob,
        "rodent_insp_1y": rodent_1y,
        "rodent_insp_3y": rodent_3y,
        "rodent_insp_all": rodent_all,
        "days_since_last_rodent": recency_rodent,
        "rodent_active_1y": rodent_active_1y,
        "rodent_active_all": rodent_active_all,
    }

    recency_cols = {
        "days_since_last_c",
        "days_since_last_any",
        "days_since_last_comp",
        "days_since_last_dob",
        "days_since_last_rodent",
    }

    for col_name, s in feature_dict.items():
        if col_name in recency_cols:
            df_cohort[col_name] = df_cohort["bbl"].map(s).fillna(3650.0).clip(0, 3650)
        else:
            df_cohort[col_name] = df_cohort["bbl"].map(s).fillna(0.0)

    df_cohort["is_aep_building"] = df_cohort["bbl"].isin(aep_bbl_set).astype(float)

    morph_records = [
        pluto_features.get(
            bbl,
            {
                "borocode": float(bbl[0]) if bbl and bbl[0].isdigit() else 1.0,
                "unitsres": 1.0,
                "unitstotal": 1.0,
                "yearbuilt": 1950.0,
                "numfloors": 3.0,
                "bldgarea": 0.0,
                "resarea": 0.0,
                "lotarea": 0.0,
                "cd": 101.0,
                "bldgclass_first": "C",
            },
        )
        for bbl in df_cohort["bbl"]
    ]
    df_morph = pd.DataFrame(morph_records, index=df_cohort.index)
    df_cohort = pd.concat([df_cohort, df_morph], axis=1)

    unitsres_safe = df_cohort["unitsres"].clip(lower=1.0)
    floors_safe = df_cohort["numfloors"].clip(lower=1.0)
    bldgarea_safe = df_cohort["bldgarea"].clip(lower=1.0)

    cutoff_year = cutoff.year
    df_cohort["building_age"] = (cutoff_year - df_cohort["yearbuilt"]).clip(0, 200)
    df_cohort["is_prewar"] = (df_cohort["yearbuilt"] < 1940).astype(float)
    df_cohort["is_pre1960"] = (df_cohort["yearbuilt"] < 1960).astype(float)
    df_cohort["area_per_unit"] = (df_cohort["bldgarea"] / unitsres_safe).clip(0, 10000)
    df_cohort["units_per_floor"] = (unitsres_safe / floors_safe).clip(0, 100)
    df_cohort["res_area_ratio"] = (df_cohort["resarea"] / bldgarea_safe).clip(0, 1.0)
    df_cohort["log_unitsres"] = np.log1p(unitsres_safe)
    df_cohort["log_bldgarea"] = np.log1p(df_cohort["bldgarea"])

    df_cohort["total_viol_1y"] = (
        df_cohort["c_viol_1y"] + df_cohort["b_viol_1y"] + df_cohort["a_viol_1y"]
    )
    df_cohort["total_viol_2y"] = (
        df_cohort["c_viol_2y"] + df_cohort["b_viol_2y"] + df_cohort["a_viol_2y"]
    )
    df_cohort["total_viol_3y"] = df_cohort["c_viol_3y"] + df_cohort["b_viol_3y"]

    # Per-unit metrics
    df_cohort["c_viol_30d_per_unit"] = df_cohort["c_viol_30d"] / unitsres_safe
    df_cohort["c_viol_90d_per_unit"] = df_cohort["c_viol_90d"] / unitsres_safe
    df_cohort["c_viol_180d_per_unit"] = df_cohort["c_viol_180d"] / unitsres_safe
    df_cohort["c_viol_1y_per_unit"] = df_cohort["c_viol_1y"] / unitsres_safe
    df_cohort["c_viol_3y_per_unit"] = df_cohort["c_viol_3y"] / unitsres_safe

    df_cohort["total_viol_90d_per_unit"] = df_cohort["total_viol_90d"] / unitsres_safe
    df_cohort["total_viol_1y_per_unit"] = df_cohort["total_viol_1y"] / unitsres_safe

    df_cohort["c_decay_180d_per_unit"] = df_cohort["c_decay_180d"] / unitsres_safe
    df_cohort["c_decay_365d_per_unit"] = df_cohort["c_decay_365d"] / unitsres_safe
    df_cohort["total_decay_180d_per_unit"] = df_cohort["total_decay_180d"] / unitsres_safe
    df_cohort["total_decay_365d_per_unit"] = df_cohort["total_decay_365d"] / unitsres_safe

    df_cohort["complaints_30d_per_unit"] = df_cohort["complaints_30d"] / unitsres_safe
    df_cohort["complaints_60d_per_unit"] = df_cohort["complaints_60d"] / unitsres_safe
    df_cohort["complaints_90d_per_unit"] = df_cohort["complaints_90d"] / unitsres_safe
    df_cohort["complaints_1y_per_unit"] = df_cohort["complaints_1y"] / unitsres_safe

    df_cohort["open_c_viol_per_unit"] = df_cohort["open_c_viol"] / unitsres_safe
    df_cohort["open_b_viol_per_unit"] = df_cohort["open_b_viol"] / unitsres_safe
    df_cohort["erp_charges_1y_per_unit"] = df_cohort["erp_charges_1y"] / unitsres_safe
    df_cohort["erp_charges_total_per_unit"] = df_cohort["erp_charges_total"] / unitsres_safe

    df_cohort["dob_viol_1y_per_unit"] = df_cohort["dob_viol_1y"] / unitsres_safe
    df_cohort["dob_viol_all_per_unit"] = df_cohort["dob_viol_all"] / unitsres_safe
    df_cohort["rodent_insp_1y_per_unit"] = df_cohort["rodent_insp_1y"] / unitsres_safe
    df_cohort["rodent_insp_all_per_unit"] = df_cohort["rodent_insp_all"] / unitsres_safe

    # Short-term velocity and acceleration metrics
    df_cohort["c_viol_velocity_90d"] = (df_cohort["c_viol_90d"] * 4.0 + 0.1) / (
        df_cohort["c_viol_1y"] + 0.1
    )
    df_cohort["c_viol_velocity_30d"] = (df_cohort["c_viol_30d"] * 12.0 + 0.1) / (
        df_cohort["c_viol_1y"] + 0.1
    )
    df_cohort["c_viol_velocity_180d"] = (df_cohort["c_viol_180d"] * 2.0 + 0.1) / (
        df_cohort["c_viol_1y"] + 0.1
    )
    df_cohort["c_viol_accel_30_90"] = (df_cohort["c_viol_30d"] * 3.0 + 0.1) / (
        df_cohort["c_viol_90d"] + 0.1
    )

    df_cohort["total_viol_velocity_90d"] = (df_cohort["total_viol_90d"] * 4.0 + 0.1) / (
        df_cohort["total_viol_1y"] + 0.1
    )
    df_cohort["total_viol_accel_30_90"] = (df_cohort["total_viol_30d"] * 3.0 + 0.1) / (
        df_cohort["total_viol_90d"] + 0.1
    )

    c_viol_prior_year = (df_cohort["c_viol_2y"] - df_cohort["c_viol_1y"]).clip(
        lower=0.0
    )
    df_cohort["c_viol_acceleration"] = (df_cohort["c_viol_1y"] + 0.1) / (
        c_viol_prior_year + 0.1
    )
    df_cohort["complaint_velocity"] = (df_cohort["complaints_90d"] * 4.0 + 0.1) / (
        df_cohort["complaints_1y"] + 0.1
    )
    df_cohort["complaint_accel_30d"] = (df_cohort["complaints_30d"] * 12.0 + 0.1) / (
        df_cohort["complaints_1y"] + 0.1
    )
    df_cohort["complaint_accel_60d"] = (df_cohort["complaints_60d"] * 6.0 + 0.1) / (
        df_cohort["complaints_1y"] + 0.1
    )
    df_cohort["unresolved_viol_ratio"] = (
        df_cohort["open_c_viol"] + df_cohort["open_b_viol"]
    ) / (df_cohort["c_viol_all"] + df_cohort["b_viol_all"] + 1.0)

    df_cohort["c_severity_share_90d"] = df_cohort["c_viol_90d"] / (
        df_cohort["total_viol_90d"] + 1.0
    )
    df_cohort["c_severity_share_1y"] = df_cohort["c_viol_1y"] / (
        df_cohort["total_viol_1y"] + 1.0
    )
    df_cohort["c_severity_share_3y"] = df_cohort["c_viol_3y"] / (
        df_cohort["total_viol_3y"] + 1.0
    )

    # Multi-agency distress index
    df_cohort["cross_agency_distress_1y"] = (
        df_cohort["c_viol_1y"]
        + df_cohort["dob_viol_1y"]
        + df_cohort["erp_charges_1y"]
        + df_cohort["litigations_1y"]
    )

    # Winter heating season escalation and chronic persistence features
    df_cohort["winter_c_ratio"] = df_cohort["c_viol_winter"] / (
        df_cohort["c_viol_1y"] + 1.0
    )
    df_cohort["winter_comp_ratio"] = df_cohort["complaints_winter"] / (
        df_cohort["complaints_1y"] + 1.0
    )
    df_cohort["c_viol_winter_per_unit"] = (
        df_cohort["c_viol_winter"] / unitsres_safe
    )
    df_cohort["complaints_winter_per_unit"] = (
        df_cohort["complaints_winter"] / unitsres_safe
    )
    df_cohort["c_to_comp_ratio_1y"] = (df_cohort["c_viol_1y"] + 0.1) / (
        df_cohort["complaints_1y"] + 0.1
    )
    df_cohort["total_to_comp_ratio_1y"] = (df_cohort["total_viol_1y"] + 0.1) / (
        df_cohort["complaints_1y"] + 0.1
    )
    df_cohort["winter_c_to_comp_ratio"] = (df_cohort["c_viol_winter"] + 0.1) / (
        df_cohort["complaints_winter"] + 0.1
    )
    df_cohort["open_c_lingering_ratio"] = df_cohort["open_c_lingering_90d"] / (
        df_cohort["open_c_viol"] + 1.0
    )

    cd_mean = df_cohort.groupby("cd")["c_viol_1y_per_unit"].transform("mean")
    df_cohort["relative_c_viol_risk"] = df_cohort["c_viol_1y_per_unit"] / (
        cd_mean + 1e-4
    )

    # Convert discrete spatial and typology codes to compact 0-indexed integers
    df_cohort["borocode"] = (
        pd.to_numeric(df_cohort["borocode"], errors="coerce")
        .fillna(1.0)
        .astype(int)
        - 1
    ).clip(0, 4)
    df_cohort["cd"] = (
        df_cohort["cd"]
        .map(lambda x: cd_to_idx.get(float(x) if pd.notna(x) else 101.0, 0))
        .astype(int)
    )

    class_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "R": 9,
        "S": 10,
    }
    df_cohort["bldgclass_code"] = (
        df_cohort["bldgclass_first"].map(class_map).fillna(2).astype(int)
    )
    df_cohort = df_cohort.drop(columns=["bldgclass_first"])

    if not is_test:
        window_end = cutoff + pd.DateOffset(years=1)
        v_target = df_violations[
            (df_violations["class"] == "C")
            & (df_violations["inspectiondate"] >= cutoff)
            & (df_violations["inspectiondate"] < window_end)
        ]
        pos_bbls = set(v_target["bbl"].unique())
        df_cohort["target"] = df_cohort["bbl"].isin(pos_bbls).astype(int)

    return df_cohort


train_entities = entity_universe_df["bbl"].unique()
val_entities = entity_universe_df["bbl"].unique()

train_df_2020 = compute_cohort_dataset("2020-01-01", train_entities, is_test=False)
train_df_2021 = compute_cohort_dataset("2021-01-01", train_entities, is_test=False)
train_df = pd.concat([train_df_2020, train_df_2021], ignore_index=True)
del train_df_2020, train_df_2021
gc.collect()

val_df = compute_cohort_dataset("2022-01-01", val_entities, is_test=False)
test_df = compute_cohort_dataset("2023-01-01", test_bbls, is_test=True)

test_df = test_entities_df[["bbl"]].merge(test_df, on="bbl", how="left")

cat_cols = ["borocode", "cd", "bldgclass_code"]
cont_cols = [
    c for c in val_df.columns if c not in ["bbl", "target"] and c not in cat_cols
]
feature_cols = cont_cols + cat_cols

del df_violations, df_complaints, df_vacate, df_lit, df_erp, df_dob, df_rodent
gc.collect()

# ---------------------------------------------------------------------------
# 5. Model Architecture: Explicit Feature-Cross Network (DCN-v2)
# ---------------------------------------------------------------------------


class CrossNetwork(nn.Module):
    """Explicit bounded-degree feature crossing layer (DCN-v2 vector formulation).

    x_{l+1} = x_0 * (x_l @ W_l) + b_l + x_l
    """

    def __init__(self, in_features: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(in_features, 1) * 0.01)
                for _ in range(num_layers)
            ]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for w, b in zip(self.weights, self.biases):
            xl_w = torch.matmul(xl, w)
            xl = x0 * xl_w + b + xl
        return xl


class DeepNetwork(nn.Module):
    """Deep residual multilayer perceptron with batch normalization, SiLU activations, and dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DCNv2TabularRanker(nn.Module):
    """Combines explicit Cross Network and Deep Non-linear MLP with categorical embeddings to predict housing violation risk scores."""

    def __init__(
        self,
        num_features: int,
        cross_layers: int = 3,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
        num_boros: int = 8,
        boro_dim: int = 8,
        num_cds: int = 120,
        cd_dim: int = 16,
        num_bldgclasses: int = 20,
        bldgclass_dim: int = 8,
        num_cats: int = 3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.num_cont = num_features - num_cats
        self.num_boros = num_boros
        self.num_cds = num_cds
        self.num_bldgclasses = num_bldgclasses

        self.boro_embed = nn.Embedding(num_boros, boro_dim)
        self.cd_embed = nn.Embedding(num_cds, cd_dim)
        self.bldg_embed = nn.Embedding(num_bldgclasses, bldgclass_dim)

        self.input_norm = nn.BatchNorm1d(self.num_cont)
        total_in_features = self.num_cont + boro_dim + cd_dim + bldgclass_dim

        self.cross_net = CrossNetwork(total_in_features, num_layers=cross_layers)
        self.deep_net = DeepNetwork(
            total_in_features, hidden_dims=hidden_dims, dropout=dropout
        )

        combined_dim = total_in_features + self.deep_net.out_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, : self.num_cont]
        x_cat = x[:, self.num_cont :].long()

        boro_idx = torch.clamp(x_cat[:, 0], 0, self.num_boros - 1)
        cd_idx = torch.clamp(x_cat[:, 1], 0, self.num_cds - 1)
        bldg_idx = torch.clamp(x_cat[:, 2], 0, self.num_bldgclasses - 1)

        boro_emb = self.boro_embed(boro_idx)
        cd_emb = self.cd_embed(cd_idx)
        bldg_emb = self.bldg_embed(bldg_idx)

        x_cont_norm = self.input_norm(x_cont)
        x_dense = torch.cat([x_cont_norm, boro_emb, cd_emb, bldg_emb], dim=1)

        cross_out = self.cross_net(x_dense)
        deep_out = self.deep_net(x_dense)
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.head(combined)
        return logits.squeeze(-1)


class PairwiseRankingLoss(nn.Module):
    """Hybrid criterion combining temperature-scaled pairwise margin ranking loss with binary cross-entropy."""

    def __init__(
        self,
        temperature: float = 1.0,
        bce_weight: float = 0.3,
        max_pairs_per_batch: int = 1024,
    ):
        super().__init__()
        self.temperature = temperature
        self.bce_weight = bce_weight
        self.max_pairs_per_batch = max_pairs_per_batch
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        pos_mask = targets == 1.0
        neg_mask = targets == 0.0

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return bce_loss

        if pos_logits.numel() > self.max_pairs_per_batch // 2:
            perm_pos = torch.randperm(
                pos_logits.numel(), device=logits.device
            )[: self.max_pairs_per_batch // 2]
            pos_logits = pos_logits[perm_pos]
        if neg_logits.numel() > self.max_pairs_per_batch:
            perm_neg = torch.randperm(
                neg_logits.numel(), device=logits.device
            )[: self.max_pairs_per_batch]
            neg_logits = neg_logits[perm_neg]

        diff = (pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)) / self.temperature
        rank_loss = torch.mean(F.softplus(-diff))

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * rank_loss


# ---------------------------------------------------------------------------
# 6. Prepare Feature Matrices & Dataloaders
# ---------------------------------------------------------------------------
skew_keywords = [
    "viol", "decay", "complaint", "erp", "dob", "rodent",
    "litig", "vacate", "open", "age", "unit", "area"
]
skew_cols = [
    c for c in cont_cols
    if any(k in c.lower() for k in skew_keywords)
    and not any(k in c.lower() for k in ["is_", "ratio", "share", "relative"])
]

def prepare_nn_features(df_input, scaler=None, is_fit=False):
    df_cont = df_input[cont_cols].copy()
    for col in skew_cols:
        df_cont[col] = np.log1p(np.clip(df_cont[col].values, 0.0, None))
    X_cont = df_cont.values.astype(np.float32)
    X_cont = np.nan_to_num(X_cont, nan=0.0, posinf=0.0, neginf=0.0)
    if is_fit:
        scaler = StandardScaler()
        X_cont_scaled = scaler.fit_transform(X_cont).astype(np.float32)
    else:
        X_cont_scaled = scaler.transform(X_cont).astype(np.float32)
    X_cont_scaled = np.nan_to_num(X_cont_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_cat = df_input[cat_cols].values.astype(np.float32)
    X_nn = np.concatenate([X_cont_scaled, X_cat], axis=1).astype(np.float32)
    return X_nn, scaler

X_train_nn, scaler = prepare_nn_features(train_df, scaler=None, is_fit=True)
y_train = train_df["target"].values.astype(np.float32)

X_val_nn, _ = prepare_nn_features(val_df, scaler=scaler, is_fit=False)
y_val = val_df["target"].values.astype(np.float32)

X_test_nn, _ = prepare_nn_features(test_df, scaler=scaler, is_fit=False)

# LightGBM feature frames with explicit native category dtype
X_train_raw = train_df[feature_cols].copy()
X_val_raw = val_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

for col in cat_cols:
    X_train_raw[col] = X_train_raw[col].astype("category")
    X_val_raw[col] = X_val_raw[col].astype("category")
    X_test_raw[col] = X_test_raw[col].astype("category")

train_dataset = TensorDataset(torch.from_numpy(X_train_nn), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val_nn), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test_nn))

train_loader = DataLoader(
    train_dataset, batch_size=2048, shuffle=True, drop_last=False, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=4096, shuffle=False, drop_last=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=4096, shuffle=False, drop_last=False, num_workers=2
)

# ---------------------------------------------------------------------------
# 7. Train DCN-v2 Tabular Ranker
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DCNv2TabularRanker(
    num_features=len(feature_cols),
    cross_layers=3,
    hidden_dims=[128, 64],
    dropout=0.2,
).to(device)

criterion = PairwiseRankingLoss(temperature=1.0, bce_weight=0.3)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

best_val_ap = -1.0
best_model_weights = None
best_model_path = "./working/dcn_best_model.pt"

for epoch in range(1, 11):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * len(by)
        total_samples += len(by)

    scheduler.step()
    avg_train_loss = running_loss / max(1, total_samples)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            val_preds_list.append(logits.cpu().numpy())

    val_preds_epoch = np.concatenate(val_preds_list)
    val_ap_epoch = average_precision_score(y_val, val_preds_epoch)

    print(
        f"Epoch {epoch:02d}/10 | Train Loss: {avg_train_loss:.4f} | DCN Val AP: {val_ap_epoch:.5f}"
    )

    if val_ap_epoch > best_val_ap:
        best_val_ap = val_ap_epoch
        best_model_weights = copy.deepcopy(model.state_dict())

if best_model_weights is not None:
    torch.save(best_model_weights, best_model_path)
    model.load_state_dict(best_model_weights)
model.eval()

val_preds_nn = []
with torch.no_grad():
    for batch in val_loader:
        bx = batch[0].to(device)
        logits = model(bx)
        val_preds_nn.append(logits.cpu().numpy())
val_preds_nn = np.concatenate(val_preds_nn)

test_preds_nn = []
with torch.no_grad():
    for batch in test_loader:
        bx = batch[0].to(device)
        logits = model(bx)
        test_preds_nn.append(logits.cpu().numpy())
test_preds_nn = np.concatenate(test_preds_nn)

# ---------------------------------------------------------------------------
# 8. Train Complementary Gradient Boosted Trees (LightGBM)
# ---------------------------------------------------------------------------
lgb_params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "min_child_samples": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
try:
    lgb_model.fit(
        X_train_raw,
        y_train,
        eval_set=[(X_val_raw, y_val)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
except Exception:
    lgb_params["metric"] = "auc"
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_raw,
        y_train,
        eval_set=[(X_val_raw, y_val)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

val_preds_lgb = lgb_model.predict_proba(X_val_raw)[:, 1]
test_preds_lgb = lgb_model.predict_proba(X_test_raw)[:, 1]

# ---------------------------------------------------------------------------
# 9. Rank Normalization, Metric Optimization & Ensembling
# ---------------------------------------------------------------------------
val_rank_nn = (rankdata(val_preds_nn) - 1.0) / (len(val_preds_nn) - 1.0)
val_rank_lgb = (rankdata(val_preds_lgb) - 1.0) / (len(val_preds_lgb) - 1.0)

test_rank_nn = (rankdata(test_preds_nn) - 1.0) / (len(test_preds_nn) - 1.0)
test_rank_lgb = (rankdata(test_preds_lgb) - 1.0) / (len(test_preds_lgb) - 1.0)

best_score = -1.0
best_weight = 0.5

for w in np.linspace(0.0, 1.0, 21):
    blended_val = w * val_rank_lgb + (1.0 - w) * val_rank_nn
    ap_score = average_precision_score(y_val, blended_val)
    if ap_score > best_score:
        best_score = ap_score
        best_weight = w

score = best_score
final_test_score = best_weight * test_rank_lgb + (1.0 - best_weight) * test_rank_nn

# ---------------------------------------------------------------------------
# 10. Export Verified Submission File
# ---------------------------------------------------------------------------
submission_df = pd.DataFrame({"bbl": test_df["bbl"].values, "score": final_test_score})
submission_df["score"] = submission_df["score"].fillna(0.0)

assert len(submission_df) == 171587, f"Expected 171587 rows, got {len(submission_df)}"
assert submission_df["bbl"].nunique() == 171587, "Duplicate BBLs found in submission"
assert not submission_df["score"].isna().any(), "NaN values found in submission"
assert not np.isinf(submission_df["score"]).any(), "Inf values found in submission"

submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score:.5f}")

import copy
import gc
import glob
import json
import os
import sys
from catboost import CatBoostClassifier
import gcsfs
import lightgbm as lgb
import numpy as np
import xgboost as xgb
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
    "zipcode",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "yearalter1",
    "bldgclass",
    "numfloors",
    "bldgarea",
    "resarea",
    "comarea",
    "lotarea",
    "assesstot",
    "assessland",
    "builtfar",
    "landuse",
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
df_pluto["yearalter1"] = (
    pd.to_numeric(df_pluto["yearalter1"], errors="coerce")
    .fillna(0.0)
    .clip(lower=0.0, upper=2023.0)
    if "yearalter1" in df_pluto.columns
    else 0.0
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
comarea_safe = (
    pd.to_numeric(df_pluto["comarea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "comarea" in df_pluto.columns
    else 0.0
)
df_pluto["commercial_share"] = (
    comarea_safe / df_pluto["bldgarea"].clip(lower=1.0)
).clip(0.0, 1.0)
df_pluto["lotarea"] = (
    pd.to_numeric(df_pluto["lotarea"], errors="coerce").fillna(0.0).clip(lower=0.0)
)
df_pluto["assesstot"] = (
    pd.to_numeric(df_pluto["assesstot"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "assesstot" in df_pluto.columns
    else 0.0
)
df_pluto["assessland"] = (
    pd.to_numeric(df_pluto["assessland"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "assessland" in df_pluto.columns
    else 0.0
)
df_pluto["builtfar"] = (
    pd.to_numeric(df_pluto["builtfar"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    if "builtfar" in df_pluto.columns
    else 0.0
)
df_pluto["landuse"] = (
    pd.to_numeric(df_pluto["landuse"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=20.0)
    if "landuse" in df_pluto.columns
    else 0.0
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
df_pluto["zipcode"] = (
    pd.to_numeric(df_pluto["zipcode"], errors="coerce")
    .fillna(0)
    .astype(int)
    if "zipcode" in df_pluto.columns
    else 0
)

all_pluto_cds = sorted([float(x) for x in df_pluto["cd"].dropna().unique() if x > 0])
cd_to_idx = {cd_val: i for i, cd_val in enumerate(all_pluto_cds)}

all_pluto_zips = sorted([int(x) for x in df_pluto["zipcode"].dropna().unique() if 10000 <= x <= 12000])
zip_to_idx = {z_val: i + 1 for i, z_val in enumerate(all_pluto_zips)}
num_zips = len(zip_to_idx) + 2

entity_universe_df = df_pluto[df_pluto["unitsres"] >= 3].copy()
universe_bbls = set(entity_universe_df["bbl"]).union(set(test_bbls))

pluto_features = df_pluto.set_index("bbl")[
    [
        "borocode",
        "zipcode",
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "yearalter1",
        "numfloors",
        "bldgarea",
        "resarea",
        "commercial_share",
        "lotarea",
        "assesstot",
        "assessland",
        "builtfar",
        "landuse",
        "cd",
        "bldgclass_first",
    ]
].to_dict("index")

df_pluto["block_id"] = df_pluto["bbl"].str[:6]
block_units_map = (
    df_pluto.groupby("block_id")["unitsres"]
    .sum()
    .clip(lower=1.0)
    .to_dict()
)

bbl_to_cd = df_pluto.set_index("bbl")["cd"].to_dict()
bbl_to_zip = df_pluto.set_index("bbl")["zipcode"].to_dict()
cd_units_map = (
    df_pluto.groupby("cd")["unitsres"]
    .sum()
    .clip(lower=1.0)
    .to_dict()
)
zip_units_map = (
    df_pluto.groupby("zipcode")["unitsres"]
    .sum()
    .clip(lower=1.0)
    .to_dict()
)

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
    spec_schema = get_table_schema("speculation_watch_list", f"{LAKE_FULL}/speculation_watch_list")
    spec_cols = [c for c in ["bbl", "boroid", "boro", "borough", "block", "lot"] if find_matching_col([c], spec_schema)]
    df_spec = pd.read_parquet(
        f"{LAKE_FULL}/speculation_watch_list",
        columns=spec_cols if spec_cols else None,
        storage_options=storage_options,
    )
    df_spec["bbl"] = standardize_bbl(df_spec)
    spec_bbl_set = set(df_spec["bbl"].unique())
    del df_spec
except Exception:
    spec_bbl_set = set()

try:
    conh_schema = get_table_schema("hpd_conh_buildings", f"{LAKE_FULL}/hpd_conh_buildings")
    conh_cols = [c for c in ["bbl", "boroid", "boro", "borough", "block", "lot"] if find_matching_col([c], conh_schema)]
    df_conh = pd.read_parquet(
        f"{LAKE_FULL}/hpd_conh_buildings",
        columns=conh_cols if conh_cols else None,
        storage_options=storage_options,
    )
    df_conh["bbl"] = standardize_bbl(df_conh)
    conh_bbl_set = set(df_conh["bbl"].unique())
    del df_conh
except Exception:
    conh_bbl_set = set()

try:
    uc_schema = get_table_schema("hpd_underlying_conditions", f"{LAKE_FULL}/hpd_underlying_conditions")
    uc_cols = [c for c in ["bbl", "boroid", "boro", "borough", "block", "lot"] if find_matching_col([c], uc_schema)]
    df_uc = pd.read_parquet(
        f"{LAKE_FULL}/hpd_underlying_conditions",
        columns=uc_cols if uc_cols else None,
        storage_options=storage_options,
    )
    df_uc["bbl"] = standardize_bbl(df_uc)
    uc_bbl_set = set(df_uc["bbl"].unique())
    del df_uc
except Exception:
    uc_bbl_set = set()

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

    comp_cat_col = find_matching_col(
        ["majorcategory", "major_category", "category", "complainttype", "complaint_type"],
        complaint_schema,
    )
    comp_apt_col = find_matching_col(
        ["apartment", "apt", "unit", "aptnum", "apartment_number"],
        complaint_schema,
    )

    complaint_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "block", "lot"]:
        m = find_matching_col([c_cand], complaint_schema)
        if m and m not in complaint_cols_to_load:
            complaint_cols_to_load.append(m)
    if complaint_date_col and complaint_date_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(complaint_date_col)
    if comp_cat_col and comp_cat_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(comp_cat_col)
    if comp_apt_col and comp_apt_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(comp_apt_col)

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

    if hasattr(df_complaints["receiveddate"], "dt") and df_complaints["receiveddate"].dt.tz is not None:
        try:
            df_complaints["receiveddate"] = df_complaints["receiveddate"].dt.tz_localize(None)
        except TypeError:
            df_complaints["receiveddate"] = df_complaints["receiveddate"].dt.tz_convert(None)

    if comp_cat_col and comp_cat_col in df_complaints.columns:
        df_complaints["majorcategory"] = df_complaints[comp_cat_col].astype(str).str.upper()
    else:
        df_complaints["majorcategory"] = ""

    if comp_apt_col and comp_apt_col in df_complaints.columns:
        df_complaints["apartment"] = df_complaints[comp_apt_col].astype(str).str.strip().str.upper()
        df_complaints.loc[
            df_complaints["apartment"].isin(["NAN", "", "NONE", "BLDG", "BUILDING", "0", "NULL"]),
            "apartment",
        ] = np.nan
    else:
        df_complaints["apartment"] = np.nan

    df_complaints["is_heat_comp"] = (
        df_complaints["majorcategory"]
        .str.contains("HEAT|HOT WATER", case=False, na=False)
        .astype(float)
    )
    df_complaints["is_leak_comp"] = (
        df_complaints["majorcategory"]
        .str.contains("WATER LEAK|LEAK|PLUMBING", case=False, na=False)
        .astype(float)
    )
    df_complaints["is_paint_comp"] = (
        df_complaints["majorcategory"]
        .str.contains("PAINT|PLASTER", case=False, na=False)
        .astype(float)
    )
    df_complaints["is_unsanitary_comp"] = (
        df_complaints["majorcategory"]
        .str.contains("UNSANITARY|PESTS|RODENT|MICE|RATS|ROACHES|GARBAGE", case=False, na=False)
        .astype(float)
    )

    df_complaints = df_complaints[
        (df_complaints["bbl"].isin(universe_bbls))
        & (df_complaints["receiveddate"] >= "2015-01-01")
    ][
        [
            "bbl",
            "receiveddate",
            "apartment",
            "is_heat_comp",
            "is_leak_comp",
            "is_paint_comp",
            "is_unsanitary_comp",
        ]
    ].copy()
    df_complaints["cd"] = df_complaints["bbl"].map(bbl_to_cd).fillna(101.0)
    df_complaints["zipcode"] = df_complaints["bbl"].map(bbl_to_zip).fillna(0).astype(int)
except Exception:
    df_complaints = pd.DataFrame(
        columns=[
            "bbl",
            "receiveddate",
            "apartment",
            "is_heat_comp",
            "is_leak_comp",
            "is_paint_comp",
            "is_unsanitary_comp",
            "cd",
            "zipcode",
        ]
    )

# Load Emergency Repair Program (ERP) records: separate HWO (Boiler/Heat) and OMO (General) charges
def load_charge_table(table_name):
    try:
        tbl_schema = get_table_schema(table_name, f"{LAKE_FULL}/{table_name}")
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
            f"{LAKE_FULL}/{table_name}",
            columns=cols_to_load if cols_to_load else None,
            storage_options=storage_options,
        )
        df_temp["bbl"] = standardize_bbl(df_temp)
        if date_col and date_col in df_temp.columns:
            df_temp["charge_date"] = pd.to_datetime(df_temp[date_col], format="mixed", errors="coerce")
        else:
            df_temp["charge_date"] = pd.NaT
        return df_temp[
            df_temp["bbl"].isin(universe_bbls)
            & df_temp["charge_date"].notna()
            & (df_temp["charge_date"] >= "2015-01-01")
        ][["bbl", "charge_date"]].copy()
    except Exception:
        return pd.DataFrame(columns=["bbl", "charge_date"])

df_hwo = load_charge_table("hpd_hwo_charges")
df_omo = load_charge_table("hpd_omo_charges")

if len(df_hwo) > 0 or len(df_omo) > 0:
    df_erp = pd.concat([df_hwo, df_omo], ignore_index=True)
else:
    df_erp = pd.DataFrame(columns=["bbl", "charge_date"])
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

# Load Department of Buildings (DOB) Complaints
try:
    dob_comp_schema = get_table_schema("dob_complaints", f"{LAKE_FULL}/dob_complaints")
    dob_comp_date_col = find_matching_col(
        [
            "date_entered", "dateentered", "entered_date", "entereddate",
            "complaint_date", "complaintdate", "dobrundate", "date",
            "received_date", "receiveddate", "inspection_date", "issue_date",
        ],
        dob_comp_schema,
    )
    if not dob_comp_date_col:
        dob_comp_date_col = next(
            (
                c for c in dob_comp_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    dob_comp_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], dob_comp_schema)
        if m and m not in dob_comp_cols_to_load:
            dob_comp_cols_to_load.append(m)
    if dob_comp_date_col and dob_comp_date_col not in dob_comp_cols_to_load:
        dob_comp_cols_to_load.append(dob_comp_date_col)

    df_dob_comp = pd.read_parquet(
        f"{LAKE_FULL}/dob_complaints",
        columns=dob_comp_cols_to_load if dob_comp_cols_to_load else None,
        storage_options=storage_options,
    )
    df_dob_comp["bbl"] = standardize_bbl(df_dob_comp)
    if dob_comp_date_col and dob_comp_date_col in df_dob_comp.columns:
        df_dob_comp["complaint_date"] = pd.to_datetime(df_dob_comp[dob_comp_date_col], format="mixed", errors="coerce")
    else:
        df_dob_comp["complaint_date"] = pd.NaT

    if hasattr(df_dob_comp["complaint_date"], "dt") and df_dob_comp["complaint_date"].dt.tz is not None:
        try:
            df_dob_comp["complaint_date"] = df_dob_comp["complaint_date"].dt.tz_localize(None)
        except TypeError:
            df_dob_comp["complaint_date"] = df_dob_comp["complaint_date"].dt.tz_convert(None)

    df_dob_comp = df_dob_comp[
        (df_dob_comp["bbl"].isin(universe_bbls))
        & (df_dob_comp["complaint_date"].notna())
        & (df_dob_comp["complaint_date"] >= "2010-01-01")
    ][["bbl", "complaint_date"]].copy()
except Exception:
    df_dob_comp = pd.DataFrame(columns=["bbl", "complaint_date"])

# Load Court Evictions
try:
    evict_schema = get_table_schema("evictions", f"{LAKE_FULL}/evictions")
    evict_date_col = find_matching_col(
        [
            "executed_date", "executeddate", "eviction_date", "evictiondate",
            "court_date", "courtdate", "date", "file_date", "filed_date",
        ],
        evict_schema,
    )
    if not evict_date_col:
        evict_date_col = next(
            (
                c for c in evict_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    evict_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], evict_schema)
        if m and m not in evict_cols_to_load:
            evict_cols_to_load.append(m)
    if evict_date_col and evict_date_col not in evict_cols_to_load:
        evict_cols_to_load.append(evict_date_col)

    df_evictions = pd.read_parquet(
        f"{LAKE_FULL}/evictions",
        columns=evict_cols_to_load if evict_cols_to_load else None,
        storage_options=storage_options,
    )
    df_evictions["bbl"] = standardize_bbl(df_evictions)
    if evict_date_col and evict_date_col in df_evictions.columns:
        df_evictions["eviction_date"] = pd.to_datetime(df_evictions[evict_date_col], format="mixed", errors="coerce")
    else:
        df_evictions["eviction_date"] = pd.NaT

    if hasattr(df_evictions["eviction_date"], "dt") and df_evictions["eviction_date"].dt.tz is not None:
        try:
            df_evictions["eviction_date"] = df_evictions["eviction_date"].dt.tz_localize(None)
        except TypeError:
            df_evictions["eviction_date"] = df_evictions["eviction_date"].dt.tz_convert(None)

    df_evictions = df_evictions[
        (df_evictions["bbl"].isin(universe_bbls))
        & (df_evictions["eviction_date"].notna())
        & (df_evictions["eviction_date"] >= "2010-01-01")
    ][["bbl", "eviction_date"]].copy()
except Exception:
    df_evictions = pd.DataFrame(columns=["bbl", "eviction_date"])

# Load HPD Bedbug Reports
try:
    bedbug_schema = get_table_schema("hpd_bedbug_reports", f"{LAKE_FULL}/hpd_bedbug_reports")
    bedbug_date_col = find_matching_col(
        [
            "filing_date", "filingdate", "file_date", "filedate",
            "period_end_date", "period_start_date", "date",
        ],
        bedbug_schema,
    )
    if not bedbug_date_col:
        bedbug_date_col = next(
            (
                c for c in bedbug_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    bedbug_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], bedbug_schema)
        if m and m not in bedbug_cols_to_load:
            bedbug_cols_to_load.append(m)
    if bedbug_date_col and bedbug_date_col not in bedbug_cols_to_load:
        bedbug_cols_to_load.append(bedbug_date_col)

    df_bedbugs = pd.read_parquet(
        f"{LAKE_FULL}/hpd_bedbug_reports",
        columns=bedbug_cols_to_load if bedbug_cols_to_load else None,
        storage_options=storage_options,
    )
    df_bedbugs["bbl"] = standardize_bbl(df_bedbugs)
    if bedbug_date_col and bedbug_date_col in df_bedbugs.columns:
        df_bedbugs["filing_date"] = pd.to_datetime(df_bedbugs[bedbug_date_col], format="mixed", errors="coerce")
    else:
        df_bedbugs["filing_date"] = pd.NaT

    if hasattr(df_bedbugs["filing_date"], "dt") and df_bedbugs["filing_date"].dt.tz is not None:
        try:
            df_bedbugs["filing_date"] = df_bedbugs["filing_date"].dt.tz_localize(None)
        except TypeError:
            df_bedbugs["filing_date"] = df_bedbugs["filing_date"].dt.tz_convert(None)

    df_bedbugs = df_bedbugs[
        (df_bedbugs["bbl"].isin(universe_bbls))
        & (df_bedbugs["filing_date"].notna())
        & (df_bedbugs["filing_date"] >= "2010-01-01")
    ][["bbl", "filing_date"]].copy()
except Exception:
    df_bedbugs = pd.DataFrame(columns=["bbl", "filing_date"])

# Load HPD Registrations and Registration Contacts for authentic multi-property Landlord Portfolio distress metrics
try:
    reg_schema = get_table_schema("hpd_registrations", f"{LAKE_FULL}/hpd_registrations")
    reg_id_col = find_matching_col(["registrationid", "registration_id", "regid", "hpd_registration_id"], reg_schema) or "registrationid"
    reg_date_col = find_matching_col(["lastregistrationdate", "registrationdate", "last_registration_date", "reg_date", "date"], reg_schema)

    reg_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], reg_schema)
        if m and m not in reg_cols_to_load:
            reg_cols_to_load.append(m)
    if reg_id_col and reg_id_col in reg_schema and reg_id_col not in reg_cols_to_load:
        reg_cols_to_load.append(reg_id_col)
    if reg_date_col and reg_date_col in reg_schema and reg_date_col not in reg_cols_to_load:
        reg_cols_to_load.append(reg_date_col)

    df_reg = pd.read_parquet(
        f"{LAKE_FULL}/hpd_registrations",
        columns=reg_cols_to_load if reg_cols_to_load else None,
        storage_options=storage_options,
    )
    df_reg["bbl"] = standardize_bbl(df_reg)
    df_reg = df_reg[df_reg["bbl"].isin(universe_bbls)].copy()
    if reg_date_col and reg_date_col in df_reg.columns:
        df_reg["reg_date"] = pd.to_datetime(df_reg[reg_date_col], format="mixed", errors="coerce")
        if hasattr(df_reg["reg_date"], "dt") and df_reg["reg_date"].dt.tz is not None:
            try:
                df_reg["reg_date"] = df_reg["reg_date"].dt.tz_localize(None)
            except TypeError:
                df_reg["reg_date"] = df_reg["reg_date"].dt.tz_convert(None)
    else:
        df_reg["reg_date"] = pd.NaT

    df_reg_dates = df_reg[df_reg["reg_date"].notna()][["bbl", "reg_date"]].copy()

    if reg_id_col and reg_id_col in df_reg.columns:
        df_reg = df_reg.dropna(subset=[reg_id_col])
        if "reg_date" in df_reg.columns:
            df_reg = df_reg.sort_values("reg_date").drop_duplicates("bbl", keep="last")
        else:
            df_reg = df_reg.drop_duplicates("bbl", keep="last")
        df_reg["reg_id_int"] = pd.to_numeric(df_reg[reg_id_col], errors="coerce").fillna(0).astype(np.int64)
        bbl_reg_pairs = df_reg[["bbl", "reg_id_int"]].copy()
    else:
        bbl_reg_pairs = pd.DataFrame(columns=["bbl", "reg_id_int"])
    del df_reg
except Exception:
    bbl_reg_pairs = pd.DataFrame(columns=["bbl", "reg_id_int"])
    df_reg_dates = pd.DataFrame(columns=["bbl", "reg_date"])
gc.collect()

bbl_to_landlord = {}
try:
    cont_schema = get_table_schema("hpd_registration_contacts", f"{LAKE_FULL}/hpd_registration_contacts")
    cont_reg_id_col = find_matching_col(["registrationid", "registration_id", "regid", "hpd_registration_id"], cont_schema) or "registrationid"
    corp_col = find_matching_col(["corporationname", "corporation_name", "corpname"], cont_schema)
    house_col = find_matching_col(["businesshousenumber", "business_house_number", "housenumber"], cont_schema)
    street_col = find_matching_col(["businessstreetname", "business_street_name", "streetname"], cont_schema)
    zip_col = find_matching_col(["businesszip", "business_zip", "zip", "zipcode"], cont_schema)
    type_col = find_matching_col(["type", "contacttype", "contactdescription"], cont_schema)

    cont_cols_to_load = [c for c in [cont_reg_id_col, corp_col, house_col, street_col, zip_col, type_col] if c and c in cont_schema]
    df_contacts = pd.read_parquet(
        f"{LAKE_FULL}/hpd_registration_contacts",
        columns=cont_cols_to_load if cont_cols_to_load else None,
        storage_options=storage_options,
    )
    if cont_reg_id_col in df_contacts.columns and len(bbl_reg_pairs) > 0:
        df_contacts["reg_id_int"] = pd.to_numeric(df_contacts[cont_reg_id_col], errors="coerce").fillna(0).astype(np.int64)
        valid_regs = set(bbl_reg_pairs["reg_id_int"].unique())
        df_contacts = df_contacts[df_contacts["reg_id_int"].isin(valid_regs)].copy()

        corp_s = df_contacts[corp_col].astype(str).str.strip().str.upper().replace(["NAN", "NONE", "NULL", "0", ""], "") if corp_col and corp_col in df_contacts.columns else pd.Series("", index=df_contacts.index)
        house_s = df_contacts[house_col].astype(str).str.strip().str.upper().replace(["NAN", "NONE", "NULL", "0", ""], "") if house_col and house_col in df_contacts.columns else pd.Series("", index=df_contacts.index)
        street_s = df_contacts[street_col].astype(str).str.strip().str.upper().replace(["NAN", "NONE", "NULL", "0", ""], "") if street_col and street_col in df_contacts.columns else pd.Series("", index=df_contacts.index)
        zip_s = df_contacts[zip_col].astype(str).str.strip().str.split(".").str[0].str[:5].replace(["NAN", "NONE", "NULL", "00000", "0", ""], "") if zip_col and zip_col in df_contacts.columns else pd.Series("", index=df_contacts.index)

        corp_clean = corp_s.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        street_clean = street_s.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        house_clean = house_s.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

        addr_clean = np.where(
            (house_clean != "") & (street_clean != ""),
            house_clean + " " + street_clean + " " + zip_s,
            np.where(street_clean != "", street_clean + " " + zip_s, np.where(zip_s != "", zip_s, ""))
        )
        addr_clean = pd.Series(addr_clean, index=df_contacts.index).str.strip()

        composite_key = np.where(
            (corp_clean != "") & (addr_clean != ""),
            corp_clean + "_" + addr_clean,
            np.where(corp_clean != "", corp_clean, addr_clean)
        )
        df_contacts["owner_key"] = composite_key

        q_score = (corp_clean != "").astype(int) * 3 + (addr_clean != "").astype(int) * 2 + (corp_clean.str.len() > 5).astype(int) * 2 + (addr_clean.str.len() > 8).astype(int) * 2
        if type_col and type_col in df_contacts.columns:
            type_s = df_contacts[type_col].astype(str).str.upper()
            q_score += type_s.str.contains("AGENT|HEADOFFICER|CORPORATE|OFFICER|MANAG", case=False, na=False).astype(int) * 2
        df_contacts["q_score"] = q_score

        df_contacts = df_contacts[df_contacts["owner_key"].str.len() > 4].copy()
        df_contacts = df_contacts.sort_values(["reg_id_int", "q_score"], ascending=[True, False]).drop_duplicates("reg_id_int", keep="first")

        reg_to_owner = dict(zip(df_contacts["reg_id_int"], df_contacts["owner_key"]))
        del df_contacts

        bbl_reg_pairs["owner_key"] = bbl_reg_pairs["reg_id_int"].map(reg_to_owner)
        valid_keys = bbl_reg_pairs["owner_key"].notna() & (bbl_reg_pairs["owner_key"].str.len() > 4)
        bbl_to_landlord = dict(zip(bbl_reg_pairs.loc[valid_keys, "bbl"], bbl_reg_pairs.loc[valid_keys, "owner_key"]))
    del bbl_reg_pairs
except Exception:
    bbl_to_landlord = {}
gc.collect()

# Load DOB Environmental Control Board (ECB) Violations
try:
    ecb_schema = get_table_schema("dob_ecb_violations", f"{LAKE_FULL}/dob_ecb_violations")
    ecb_date_col = find_matching_col(
        [
            "issue_date", "issuedate", "violation_date", "violationdate",
            "served_date", "serveddate", "date", "inspection_date",
        ],
        ecb_schema,
    )
    if not ecb_date_col:
        ecb_date_col = next(
            (c for c in ecb_schema if "issue" in c.lower() and "date" in c.lower()),
            None,
        )
    if not ecb_date_col:
        ecb_date_col = next(
            (
                c for c in ecb_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    ecb_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], ecb_schema)
        if m and m not in ecb_cols_to_load:
            ecb_cols_to_load.append(m)
    if ecb_date_col and ecb_date_col not in ecb_cols_to_load:
        ecb_cols_to_load.append(ecb_date_col)

    df_ecb = pd.read_parquet(
        f"{LAKE_FULL}/dob_ecb_violations",
        columns=ecb_cols_to_load if ecb_cols_to_load else None,
        storage_options=storage_options,
    )
    df_ecb["bbl"] = standardize_bbl(df_ecb)
    if ecb_date_col and ecb_date_col in df_ecb.columns:
        df_ecb["ecb_date"] = pd.to_datetime(df_ecb[ecb_date_col], format="mixed", errors="coerce")
    else:
        df_ecb["ecb_date"] = pd.NaT

    if hasattr(df_ecb["ecb_date"], "dt") and df_ecb["ecb_date"].dt.tz is not None:
        try:
            df_ecb["ecb_date"] = df_ecb["ecb_date"].dt.tz_localize(None)
        except TypeError:
            df_ecb["ecb_date"] = df_ecb["ecb_date"].dt.tz_convert(None)

    df_ecb = df_ecb[
        (df_ecb["bbl"].isin(universe_bbls))
        & (df_ecb["ecb_date"].notna())
        & (df_ecb["ecb_date"] >= "2010-01-01")
    ][["bbl", "ecb_date"]].copy()
except Exception:
    df_ecb = pd.DataFrame(columns=["bbl", "ecb_date"])

# Load DOB Safety Violations (Boiler & Elevator Physical Safety Precursors)
try:
    safety_schema = get_table_schema("dob_safety_violations", f"{LAKE_FULL}/dob_safety_violations")
    safety_date_col = find_matching_col(
        [
            "issue_date", "violation_date", "issuedate", "inspection_date",
            "violationdate", "compliance_date", "date",
        ],
        safety_schema,
    )
    if not safety_date_col:
        safety_date_col = next(
            (
                c for c in safety_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    safety_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], safety_schema)
        if m and m not in safety_cols_to_load:
            safety_cols_to_load.append(m)
    if safety_date_col and safety_date_col not in safety_cols_to_load:
        safety_cols_to_load.append(safety_date_col)

    df_safety = pd.read_parquet(
        f"{LAKE_FULL}/dob_safety_violations",
        columns=safety_cols_to_load if safety_cols_to_load else None,
        storage_options=storage_options,
    )
    df_safety["bbl"] = standardize_bbl(df_safety)
    if safety_date_col and safety_date_col in df_safety.columns:
        df_safety["safety_date"] = pd.to_datetime(df_safety[safety_date_col], format="mixed", errors="coerce")
    else:
        df_safety["safety_date"] = pd.NaT

    if hasattr(df_safety["safety_date"], "dt") and df_safety["safety_date"].dt.tz is not None:
        try:
            df_safety["safety_date"] = df_safety["safety_date"].dt.tz_localize(None)
        except TypeError:
            df_safety["safety_date"] = df_safety["safety_date"].dt.tz_convert(None)

    df_safety = df_safety[
        (df_safety["bbl"].isin(universe_bbls))
        & (df_safety["safety_date"].notna())
        & (df_safety["safety_date"] >= "2010-01-01")
    ][["bbl", "safety_date"]].copy()
except Exception:
    df_safety = pd.DataFrame(columns=["bbl", "safety_date"])

# Load HPD Buildings for Class B SRO units and legal unit counts
try:
    bldg_schema = get_table_schema("hpd_buildings", f"{LAKE_FULL}/hpd_buildings")
    classb_col = find_matching_col(
        ["classbunits", "classb_units", "classb", "class_b_units", "class_b"],
        bldg_schema,
    )
    units_col = find_matching_col(
        ["totalunits", "total_units", "unitstotal", "units_total", "legalunits", "unitsres"],
        bldg_schema,
    )
    bldg_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], bldg_schema)
        if m and m not in bldg_cols_to_load:
            bldg_cols_to_load.append(m)
    if classb_col and classb_col not in bldg_cols_to_load:
        bldg_cols_to_load.append(classb_col)
    if units_col and units_col not in bldg_cols_to_load:
        bldg_cols_to_load.append(units_col)

    df_hpd_bldg = pd.read_parquet(
        f"{LAKE_FULL}/hpd_buildings",
        columns=bldg_cols_to_load if bldg_cols_to_load else None,
        storage_options=storage_options,
    )
    df_hpd_bldg["bbl"] = standardize_bbl(df_hpd_bldg)
    df_hpd_bldg = df_hpd_bldg[df_hpd_bldg["bbl"].isin(universe_bbls)].copy()

    if classb_col and classb_col in df_hpd_bldg.columns:
        df_hpd_bldg["hpd_classb_units"] = pd.to_numeric(df_hpd_bldg[classb_col], errors="coerce").fillna(0.0)
    else:
        df_hpd_bldg["hpd_classb_units"] = 0.0

    if units_col and units_col in df_hpd_bldg.columns:
        df_hpd_bldg["hpd_total_units"] = pd.to_numeric(df_hpd_bldg[units_col], errors="coerce").fillna(0.0)
    else:
        df_hpd_bldg["hpd_total_units"] = 0.0

    df_hpd_bldg = df_hpd_bldg.groupby("bbl")[["hpd_classb_units", "hpd_total_units"]].max().reset_index()
    hpd_classb_map = dict(zip(df_hpd_bldg["bbl"], df_hpd_bldg["hpd_classb_units"]))
    hpd_total_units_map = dict(zip(df_hpd_bldg["bbl"], df_hpd_bldg["hpd_total_units"]))
    del df_hpd_bldg
except Exception:
    hpd_classb_map = {}
    hpd_total_units_map = {}
gc.collect()

# Load DOF Tax Lien Sales
try:
    tax_lien_schema = get_table_schema("dof_tax_lien_sales", f"{LAKE_FULL}/dof_tax_lien_sales")
    tax_lien_date_col = find_matching_col(
        [
            "sale_date", "saledate", "date", "lien_sale_date",
            "notice_date", "tax_lien_sale_date", "month_year", "period",
        ],
        tax_lien_schema,
    )
    if not tax_lien_date_col:
        tax_lien_date_col = next(
            (
                c for c in tax_lien_schema
                if "date" in c.lower()
                and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])
            ),
            None,
        )

    tax_lien_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot", "year", "month"]:
        m = find_matching_col([c_cand], tax_lien_schema)
        if m and m not in tax_lien_cols_to_load:
            tax_lien_cols_to_load.append(m)
    if tax_lien_date_col and tax_lien_date_col not in tax_lien_cols_to_load:
        tax_lien_cols_to_load.append(tax_lien_date_col)

    df_tax_liens = pd.read_parquet(
        f"{LAKE_FULL}/dof_tax_lien_sales",
        columns=tax_lien_cols_to_load if tax_lien_cols_to_load else None,
        storage_options=storage_options,
    )
    df_tax_liens["bbl"] = standardize_bbl(df_tax_liens)
    if tax_lien_date_col and tax_lien_date_col in df_tax_liens.columns:
        df_tax_liens["sale_date"] = pd.to_datetime(df_tax_liens[tax_lien_date_col], format="mixed", errors="coerce")
    elif "year" in df_tax_liens.columns:
        year_s = pd.to_numeric(df_tax_liens["year"], errors="coerce").fillna(2018).astype(int)
        if "month" in df_tax_liens.columns:
            month_s = pd.to_numeric(df_tax_liens["month"], errors="coerce").fillna(6).astype(int).clip(1, 12)
        else:
            month_s = 6
        df_tax_liens["sale_date"] = pd.to_datetime(dict(year=year_s, month=month_s, day=1), errors="coerce")
    else:
        df_tax_liens["sale_date"] = pd.NaT

    if hasattr(df_tax_liens["sale_date"], "dt") and df_tax_liens["sale_date"].dt.tz is not None:
        try:
            df_tax_liens["sale_date"] = df_tax_liens["sale_date"].dt.tz_localize(None)
        except TypeError:
            df_tax_liens["sale_date"] = df_tax_liens["sale_date"].dt.tz_convert(None)

    df_tax_liens = df_tax_liens[
        (df_tax_liens["bbl"].isin(universe_bbls))
        & (df_tax_liens["sale_date"].notna())
        & (df_tax_liens["sale_date"] >= "2010-01-01")
    ][["bbl", "sale_date"]].copy()
except Exception:
    df_tax_liens = pd.DataFrame(columns=["bbl", "sale_date"])

# Load DOB Stalled Construction Sites (Physical Abandonment Indicator)
try:
    stalled_schema = get_table_schema("dob_stalled_construction", f"{LAKE_FULL}/dob_stalled_construction")
    stalled_date_col = find_matching_col(
        [
            "date", "stalled_date", "inspection_date", "issue_date",
            "status_date", "created_date", "date_stalled", "last_inspection_date",
        ],
        stalled_schema,
    )
    if not stalled_date_col:
        stalled_date_col = next(
            (c for c in stalled_schema if "date" in c.lower() and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])),
            None,
        )

    stalled_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], stalled_schema)
        if m and m not in stalled_cols_to_load:
            stalled_cols_to_load.append(m)
    if stalled_date_col and stalled_date_col not in stalled_cols_to_load:
        stalled_cols_to_load.append(stalled_date_col)

    df_stalled = pd.read_parquet(
        f"{LAKE_FULL}/dob_stalled_construction",
        columns=stalled_cols_to_load if stalled_cols_to_load else None,
        storage_options=storage_options,
    )
    df_stalled["bbl"] = standardize_bbl(df_stalled)
    if stalled_date_col and stalled_date_col in df_stalled.columns:
        df_stalled["stalled_date"] = pd.to_datetime(df_stalled[stalled_date_col], format="mixed", errors="coerce")
    else:
        df_stalled["stalled_date"] = pd.NaT

    if hasattr(df_stalled["stalled_date"], "dt") and df_stalled["stalled_date"].dt.tz is not None:
        try:
            df_stalled["stalled_date"] = df_stalled["stalled_date"].dt.tz_localize(None)
        except TypeError:
            df_stalled["stalled_date"] = df_stalled["stalled_date"].dt.tz_convert(None)

    df_stalled = df_stalled[df_stalled["bbl"].isin(universe_bbls)][["bbl", "stalled_date"]].copy()
except Exception:
    df_stalled = pd.DataFrame(columns=["bbl", "stalled_date"])

# Load DOF Annualized Sales (Ownership Turnover & Sales Distress)
try:
    sales_schema = get_table_schema("dof_annualized_sales", f"{LAKE_FULL}/dof_annualized_sales")
    sale_date_col = find_matching_col(
        ["sale_date", "saledate", "date_of_sale", "date"],
        sales_schema,
    )
    if not sale_date_col:
        sale_date_col = next(
            (c for c in sales_schema if "date" in c.lower() and not any(bad in c.lower() for bad in ["desc", "comment", "status", "type", "text", "name", "id", "by"])),
            None,
        )

    sales_cols_to_load = []
    for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
        m = find_matching_col([c_cand], sales_schema)
        if m and m not in sales_cols_to_load:
            sales_cols_to_load.append(m)
    if sale_date_col and sale_date_col not in sales_cols_to_load:
        sales_cols_to_load.append(sale_date_col)

    df_sales = pd.read_parquet(
        f"{LAKE_FULL}/dof_annualized_sales",
        columns=sales_cols_to_load if sales_cols_to_load else None,
        storage_options=storage_options,
    )
    df_sales["bbl"] = standardize_bbl(df_sales)
    if sale_date_col and sale_date_col in df_sales.columns:
        df_sales["sale_date"] = pd.to_datetime(df_sales[sale_date_col], format="mixed", errors="coerce")
    else:
        df_sales["sale_date"] = pd.NaT

    if hasattr(df_sales["sale_date"], "dt") and df_sales["sale_date"].dt.tz is not None:
        try:
            df_sales["sale_date"] = df_sales["sale_date"].dt.tz_localize(None)
        except TypeError:
            df_sales["sale_date"] = df_sales["sale_date"].dt.tz_convert(None)

    df_sales = df_sales[
        (df_sales["bbl"].isin(universe_bbls))
        & (df_sales["sale_date"].notna())
        & (df_sales["sale_date"] >= "2010-01-01")
    ][["bbl", "sale_date"]].copy()
except Exception:
    df_sales = pd.DataFrame(columns=["bbl", "sale_date"])

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

correct_date_col = find_matching_col(
    [
        "originalcorrectbydate", "original_correct_by_date",
        "correctbydate", "correct_by_date",
        "originalcertifybydate", "certifybydate",
    ],
    viol_schema,
)

desc_col = find_matching_col(
    ["novdescription", "nov_description", "description", "violationdescription"],
    viol_schema,
)

order_col = find_matching_col(
    ["ordernumber", "order_number", "orderno", "order_no"],
    viol_schema,
)

rent_col = find_matching_col(
    ["rentimpairing", "rent_impairing", "rentimpairingflag", "rent_impairing_flag"],
    viol_schema,
)

viol_apt_col = find_matching_col(
    ["apartment", "apt", "unit", "aptnum", "apartment_number"],
    viol_schema,
)

viol_cols_to_load = []
for c_cand in ["bbl", "boroid", "borough", "borocode", "boro", "block", "lot"]:
    m = find_matching_col([c_cand], viol_schema)
    if m and m not in viol_cols_to_load:
        viol_cols_to_load.append(m)

for c in [class_col, insp_date_col, status_date_col, correct_date_col, desc_col, order_col, rent_col, viol_apt_col]:
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

if correct_date_col and correct_date_col in df_violations.columns:
    df_violations["originalcorrectbydate"] = pd.to_datetime(df_violations[correct_date_col], format="mixed", errors="coerce")
    if hasattr(df_violations["originalcorrectbydate"], "dt") and df_violations["originalcorrectbydate"].dt.tz is not None:
        try:
            df_violations["originalcorrectbydate"] = df_violations["originalcorrectbydate"].dt.tz_localize(None)
        except TypeError:
            df_violations["originalcorrectbydate"] = df_violations["originalcorrectbydate"].dt.tz_convert(None)
else:
    df_violations["originalcorrectbydate"] = pd.NaT

if rent_col and rent_col in df_violations.columns:
    df_violations["is_rent_impairing"] = (
        df_violations[rent_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .isin(["Y", "1", "TRUE", "T"])
        .astype(float)
    )
else:
    df_violations["is_rent_impairing"] = 0.0

if viol_apt_col and viol_apt_col in df_violations.columns:
    df_violations["apartment"] = (
        df_violations[viol_apt_col].astype(str).str.strip().str.upper()
    )
    df_violations.loc[
        df_violations["apartment"].isin(
            ["NAN", "", "NONE", "BLDG", "BUILDING", "0", "NULL"]
        ),
        "apartment",
    ] = np.nan
else:
    df_violations["apartment"] = np.nan

if order_col and order_col in df_violations.columns:
    order_num_s = pd.to_numeric(
        df_violations[order_col].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).fillna(0).astype(int)
else:
    order_num_s = pd.Series(0, index=df_violations.index)

if desc_col and desc_col in df_violations.columns:
    nov_desc_s = df_violations[desc_col].astype(str).str.upper()
    df_violations["is_heat"] = (
        nov_desc_s.str.contains("HEAT|HOT WATER|BOILER", case=False, na=False)
        | order_num_s.between(501, 505)
    ).astype(float)
    df_violations["is_lead"] = (
        order_num_s.between(616, 620)
        | nov_desc_s.str.contains("LEAD", case=False, na=False)
    ).astype(float)
    df_violations["is_mold"] = (
        order_num_s.between(550, 559)
        | nov_desc_s.str.contains("MOLD|MOISTURE", case=False, na=False)
    ).astype(float)
    df_violations["is_doors_detectors"] = (
        order_num_s.between(509, 514)
        | nov_desc_s.str.contains("SELF-CLOSING|DETECTOR", case=False, na=False)
    ).astype(float)
    df_violations["is_pest"] = (
        order_num_s.between(570, 580)
        | nov_desc_s.str.contains("PEST|ROACH|MICE|MOUSE|RAT|VERMIN|BEDBUG", case=False, na=False)
    ).astype(float)
else:
    df_violations["is_heat"] = order_num_s.between(501, 505).astype(float)
    df_violations["is_lead"] = order_num_s.between(616, 620).astype(float)
    df_violations["is_mold"] = order_num_s.between(550, 559).astype(float)
    df_violations["is_doors_detectors"] = order_num_s.between(509, 514).astype(float)
    df_violations["is_pest"] = order_num_s.between(570, 580).astype(float)

df_violations = df_violations[
    (df_violations["bbl"].isin(universe_bbls))
    & (df_violations["inspectiondate"] >= "2015-01-01")
][
    [
        "bbl",
        "class",
        "inspectiondate",
        "currentstatusdate",
        "originalcorrectbydate",
        "apartment",
        "is_heat",
        "is_lead",
        "is_mold",
        "is_doors_detectors",
        "is_pest",
        "is_rent_impairing",
    ]
].copy()
df_violations["cd"] = df_violations["bbl"].map(bbl_to_cd).fillna(101.0)
df_violations["zipcode"] = df_violations["bbl"].map(bbl_to_zip).fillna(0).astype(int)
gc.collect()


# ---------------------------------------------------------------------------
# 4. Point-In-Time Feature Extraction Engine
# ---------------------------------------------------------------------------
def compute_cohort_dataset(cutoff_str, target_bbl_list, is_test=False):
    cutoff = pd.Timestamp(cutoff_str)
    cutoff_14d_prior = cutoff - pd.DateOffset(days=14)
    cutoff_30d_prior = cutoff - pd.DateOffset(days=30)
    cutoff_60d_prior = cutoff - pd.DateOffset(days=60)
    cutoff_90d_prior = cutoff - pd.DateOffset(days=90)
    cutoff_180d_prior = cutoff - pd.DateOffset(days=180)
    cutoff_1y_prior = cutoff - pd.DateOffset(days=365)
    cutoff_2y_prior = cutoff - pd.DateOffset(days=730)
    cutoff_3y_prior = cutoff - pd.DateOffset(days=1095)

    df_base = (
        pd.DataFrame({"bbl": target_bbl_list})
        .drop_duplicates("bbl")
        .reset_index(drop=True)
    )
    bbl_list = df_base["bbl"].tolist()
    bbl_set = set(bbl_list)
    bbl_series = df_base["bbl"]
    block_id_series = bbl_series.str[:6]

    v_hist = df_violations[
        (df_violations["bbl"].isin(bbl_set))
        & (df_violations["inspectiondate"] < cutoff)
    ]

    v_c = v_hist[v_hist["class"] == "C"]
    v_b = v_hist[v_hist["class"] == "B"]
    v_a = v_hist[v_hist["class"] == "A"]

    # Short and multi-year horizon Class C windows
    c_30d = v_c[v_c["inspectiondate"] >= cutoff_30d_prior].groupby("bbl").size()
    c_90d = v_c[v_c["inspectiondate"] >= cutoff_90d_prior].groupby("bbl").size()
    c_180d = v_c[v_c["inspectiondate"] >= cutoff_180d_prior].groupby("bbl").size()
    c_1y = v_c[v_c["inspectiondate"] >= cutoff_1y_prior].groupby("bbl").size()
    c_2y = v_c[v_c["inspectiondate"] >= cutoff_2y_prior].groupby("bbl").size()
    c_3y = v_c[v_c["inspectiondate"] >= cutoff_3y_prior].groupby("bbl").size()
    c_all = v_c.groupby("bbl").size()

    # Short and multi-year horizon Class B windows
    b_30d = v_b[v_b["inspectiondate"] >= cutoff_30d_prior].groupby("bbl").size()
    b_90d = v_b[v_b["inspectiondate"] >= cutoff_90d_prior].groupby("bbl").size()
    b_180d = v_b[v_b["inspectiondate"] >= cutoff_180d_prior].groupby("bbl").size()
    b_1y = v_b[v_b["inspectiondate"] >= cutoff_1y_prior].groupby("bbl").size()
    b_2y = v_b[v_b["inspectiondate"] >= cutoff_2y_prior].groupby("bbl").size()
    b_3y = v_b[v_b["inspectiondate"] >= cutoff_3y_prior].groupby("bbl").size()
    b_all = v_b.groupby("bbl").size()

    a_1y = v_a[v_a["inspectiondate"] >= cutoff_1y_prior].groupby("bbl").size()
    a_2y = v_a[v_a["inspectiondate"] >= cutoff_2y_prior].groupby("bbl").size()

    # Total violation windows
    tot_30d = v_hist[v_hist["inspectiondate"] >= cutoff_30d_prior].groupby("bbl").size()
    tot_90d = v_hist[v_hist["inspectiondate"] >= cutoff_90d_prior].groupby("bbl").size()
    tot_180d = v_hist[v_hist["inspectiondate"] >= cutoff_180d_prior].groupby("bbl").size()
    tot_all = v_hist.groupby("bbl").size()

    # Inspection visit cadence (unique inspection dates per BBL)
    v_hist_1y = v_hist[v_hist["inspectiondate"] >= cutoff_1y_prior]
    v_hist_3y = v_hist[v_hist["inspectiondate"] >= cutoff_3y_prior]
    insp_visits_1y = v_hist_1y.groupby("bbl")["inspectiondate"].nunique()
    insp_visits_3y = v_hist_3y.groupby("bbl")["inspectiondate"].nunique()

    # Exponential recency-decayed violation intensity scores (180d and 365d half-lives)
    if len(v_c) > 0:
        c_dt_days = ((cutoff - v_c["inspectiondate"]).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_c_180 = np.exp(-0.69314718056 * c_dt_days / 180.0)
        w_c_365 = np.exp(-0.69314718056 * c_dt_days / 365.0)
        c_decay_180 = w_c_180.groupby(v_c["bbl"]).sum()
        c_decay_365 = w_c_365.groupby(v_c["bbl"]).sum()
    else:
        c_decay_180 = pd.Series(dtype=float)
        c_decay_365 = pd.Series(dtype=float)

    if len(v_hist) > 0:
        tot_dt_days = ((cutoff - v_hist["inspectiondate"]).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_tot_180 = np.exp(-0.69314718056 * tot_dt_days / 180.0)
        w_tot_365 = np.exp(-0.69314718056 * tot_dt_days / 365.0)
        tot_decay_180 = w_tot_180.groupby(v_hist["bbl"]).sum()
        tot_decay_365 = w_tot_365.groupby(v_hist["bbl"]).sum()
    else:
        tot_decay_180 = pd.Series(dtype=float)
        tot_decay_365 = pd.Series(dtype=float)

    # Multi-year violation persistence: distinct calendar years with Class C citations
    if len(v_c) > 0:
        v_c_years = v_c.assign(insp_year=v_c["inspectiondate"].dt.year)
        c_years_with_viol = v_c_years.groupby("bbl")["insp_year"].nunique()
    else:
        c_years_with_viol = pd.Series(dtype=int)

    # Hazardous open violation backlog duration and lingering count (>90 days unresolved)
    open_c_records = v_c[
        (v_c["currentstatusdate"].isna()) | (v_c["currentstatusdate"] >= cutoff)
    ]
    open_c = open_c_records.groupby("bbl").size()

    if len(open_c_records) > 0:
        open_c_age_days = (
            (cutoff - open_c_records["inspectiondate"]).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        open_c_max_age_days = open_c_age_days.groupby(open_c_records["bbl"]).max()
        open_c_lingering_90d = (open_c_age_days > 90.0).astype(float).groupby(open_c_records["bbl"]).sum()
    else:
        open_c_max_age_days = pd.Series(dtype=float)
        open_c_lingering_90d = pd.Series(dtype=float)

    open_b = v_b[(v_b["currentstatusdate"].isna()) | (v_b["currentstatusdate"] >= cutoff)].groupby("bbl").size()
    open_a = v_a[(v_a["currentstatusdate"].isna()) | (v_a["currentstatusdate"] >= cutoff)].groupby("bbl").size()
    open_total = v_hist[(v_hist["currentstatusdate"].isna()) | (v_hist["currentstatusdate"] >= cutoff)].groupby("bbl").size()

    # Overdue uncertified Class C violation backlog & max days overdue past statutory deadline
    overdue_c_records = v_c[
        (v_c["originalcorrectbydate"].notna())
        & (v_c["originalcorrectbydate"] < cutoff)
        & ((v_c["currentstatusdate"].isna()) | (v_c["currentstatusdate"] >= cutoff))
    ]
    overdue_c = overdue_c_records.groupby("bbl").size()

    if len(overdue_c_records) > 0:
        overdue_c_days = (
            (cutoff - overdue_c_records["originalcorrectbydate"]).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        max_days_overdue_c = overdue_c_days.groupby(overdue_c_records["bbl"]).max()
    else:
        max_days_overdue_c = pd.Series(dtype=float)

    # Acute winter heating citation windows
    v_c_heat = v_c[v_c["is_heat"] > 0]
    heat_c_30d = v_c_heat[v_c_heat["inspectiondate"] >= cutoff_30d_prior].groupby("bbl").size()
    heat_c_60d = v_c_heat[v_c_heat["inspectiondate"] >= cutoff_60d_prior].groupby("bbl").size()
    heat_c_90d = v_c_heat[v_c_heat["inspectiondate"] >= cutoff_90d_prior].groupby("bbl").size()
    heat_c_1y = v_c_heat[v_c_heat["inspectiondate"] >= cutoff_1y_prior].groupby("bbl").size()
    heat_c_3y = v_c_heat[v_c_heat["inspectiondate"] >= cutoff_3y_prior].groupby("bbl").size()

    # Statutory Class C hazard families
    lead_c_1y = v_c[(v_c["is_lead"] > 0) & (v_c["inspectiondate"] >= cutoff_1y_prior)].groupby("bbl").size()
    lead_c_3y = v_c[(v_c["is_lead"] > 0) & (v_c["inspectiondate"] >= cutoff_3y_prior)].groupby("bbl").size()
    mold_c_1y = v_c[(v_c["is_mold"] > 0) & (v_c["inspectiondate"] >= cutoff_1y_prior)].groupby("bbl").size()
    mold_c_3y = v_c[(v_c["is_mold"] > 0) & (v_c["inspectiondate"] >= cutoff_3y_prior)].groupby("bbl").size()
    doors_c_1y = v_c[(v_c["is_doors_detectors"] > 0) & (v_c["inspectiondate"] >= cutoff_1y_prior)].groupby("bbl").size()
    doors_c_3y = v_c[(v_c["is_doors_detectors"] > 0) & (v_c["inspectiondate"] >= cutoff_3y_prior)].groupby("bbl").size()
    rent_c_1y = v_c[(v_c["is_rent_impairing"] > 0) & (v_c["inspectiondate"] >= cutoff_1y_prior)].groupby("bbl").size()
    pest_c_1y = v_c[(v_c["is_pest"] > 0) & (v_c["inspectiondate"] >= cutoff_1y_prior)].groupby("bbl").size()
    pest_c_3y = v_c[(v_c["is_pest"] > 0) & (v_c["inspectiondate"] >= cutoff_3y_prior)].groupby("bbl").size()
    v_c_1y = v_c[v_c["inspectiondate"] >= cutoff_1y_prior]
    c_apts_1y = v_c_1y[v_c_1y["apartment"].notna()].groupby("bbl")["apartment"].nunique()

    # Landlord resolution latency on cured Class C violations
    v_c_cured_1y = v_c[
        (v_c["currentstatusdate"].notna())
        & (v_c["currentstatusdate"] < cutoff)
        & (v_c["inspectiondate"] >= cutoff_1y_prior)
        & (v_c["currentstatusdate"] >= v_c["inspectiondate"])
    ]
    if len(v_c_cured_1y) > 0:
        c_cure_dur_1y = (
            (v_c_cured_1y["currentstatusdate"] - v_c_cured_1y["inspectiondate"]).dt.total_seconds()
            / 86400.0
        ).clip(0, 365)
        mean_cure_days_c_1y = c_cure_dur_1y.groupby(v_c_cured_1y["bbl"]).mean()
    else:
        mean_cure_days_c_1y = pd.Series(dtype=float)

    v_c_cured_3y = v_c[
        (v_c["currentstatusdate"].notna())
        & (v_c["currentstatusdate"] < cutoff)
        & (v_c["inspectiondate"] >= cutoff_3y_prior)
        & (v_c["currentstatusdate"] >= v_c["inspectiondate"])
    ]
    if len(v_c_cured_3y) > 0:
        c_cure_dur_3y = (
            (v_c_cured_3y["currentstatusdate"] - v_c_cured_3y["inspectiondate"]).dt.total_seconds()
            / 86400.0
        ).clip(0, 1095)
        mean_cure_days_c_3y = c_cure_dur_3y.groupby(v_c_cured_3y["bbl"]).mean()
    else:
        mean_cure_days_c_3y = pd.Series(dtype=float)

    recency_c = (cutoff - v_c.groupby("bbl")["inspectiondate"].max()).dt.days if len(v_c) > 0 else pd.Series(dtype=float)
    recency_b = (cutoff - v_b.groupby("bbl")["inspectiondate"].max()).dt.days if len(v_b) > 0 else pd.Series(dtype=float)
    recency_any = (cutoff - v_hist.groupby("bbl")["inspectiondate"].max()).dt.days if len(v_hist) > 0 else pd.Series(dtype=float)

    # Winter heating season surge
    cutoff_winter_prior = cutoff - pd.DateOffset(days=92)
    c_winter = v_c[v_c["inspectiondate"] >= cutoff_winter_prior].groupby("bbl").size()

    # Complaints features
    c_hist = df_complaints[
        (df_complaints["bbl"].isin(bbl_set)) & (df_complaints["receiveddate"] < cutoff)
    ]
    comp_14d = c_hist[c_hist["receiveddate"] >= cutoff_14d_prior].groupby("bbl").size()
    comp_30d = c_hist[c_hist["receiveddate"] >= cutoff_30d_prior].groupby("bbl").size()
    comp_60d = c_hist[c_hist["receiveddate"] >= cutoff_60d_prior].groupby("bbl").size()
    comp_90d = c_hist[c_hist["receiveddate"] >= cutoff_90d_prior].groupby("bbl").size()
    comp_1y = c_hist[c_hist["receiveddate"] >= cutoff_1y_prior].groupby("bbl").size()
    comp_2y = c_hist[c_hist["receiveddate"] >= cutoff_2y_prior].groupby("bbl").size()
    comp_3y = c_hist[c_hist["receiveddate"] >= cutoff_3y_prior].groupby("bbl").size()
    comp_winter = c_hist[c_hist["receiveddate"] >= cutoff_winter_prior].groupby("bbl").size()
    recency_comp = (cutoff - c_hist.groupby("bbl")["receiveddate"].max()).dt.days if len(c_hist) > 0 else pd.Series(dtype=float)

    c_hist_14d = c_hist[c_hist["receiveddate"] >= cutoff_14d_prior]
    c_hist_30d = c_hist[c_hist["receiveddate"] >= cutoff_30d_prior]
    c_hist_60d = c_hist[c_hist["receiveddate"] >= cutoff_60d_prior]
    c_hist_90d = c_hist[c_hist["receiveddate"] >= cutoff_90d_prior]
    c_hist_1y = c_hist[c_hist["receiveddate"] >= cutoff_1y_prior]
    c_hist_3y = c_hist[c_hist["receiveddate"] >= cutoff_3y_prior]

    comp_heat_14d = c_hist_14d[c_hist_14d["is_heat_comp"] > 0].groupby("bbl").size()
    comp_heat_30d = c_hist_30d[c_hist_30d["is_heat_comp"] > 0].groupby("bbl").size()
    comp_heat_60d = c_hist_60d[c_hist_60d["is_heat_comp"] > 0].groupby("bbl").size()
    comp_heat_90d = c_hist_90d[c_hist_90d["is_heat_comp"] > 0].groupby("bbl").size()
    comp_apts_30d = c_hist_30d[c_hist_30d["apartment"].notna()].groupby("bbl")["apartment"].nunique()
    comp_apts_1y = c_hist_1y[c_hist_1y["apartment"].notna()].groupby("bbl")["apartment"].nunique()
    comp_apts_3y = c_hist_3y[c_hist_3y["apartment"].notna()].groupby("bbl")["apartment"].nunique()
    comp_heat_1y = c_hist_1y[c_hist_1y["is_heat_comp"] > 0].groupby("bbl").size()
    comp_leak_1y = c_hist_1y[c_hist_1y["is_leak_comp"] > 0].groupby("bbl").size()
    comp_paint_1y = c_hist_1y[c_hist_1y["is_paint_comp"] > 0].groupby("bbl").size()
    comp_unsanitary_1y = c_hist_1y[c_hist_1y["is_unsanitary_comp"] > 0].groupby("bbl").size()

    # Latest code inspection date per lot and pending post-inspection complaint queue
    last_insp_date = v_hist.groupby("bbl")["inspectiondate"].max() if len(v_hist) > 0 else pd.Series(dtype="datetime64[ns]")
    if len(c_hist) > 0 and len(last_insp_date) > 0:
        c_bbl_insp = c_hist["bbl"].map(last_insp_date)
        c_is_post = c_bbl_insp.isna() | (c_hist["receiveddate"] > c_bbl_insp)
        c_post_insp = c_hist[c_is_post]
        comp_post_insp_14d = c_post_insp[c_post_insp["receiveddate"] >= cutoff_14d_prior].groupby("bbl").size()
        comp_post_insp_30d = c_post_insp[c_post_insp["receiveddate"] >= cutoff_30d_prior].groupby("bbl").size()
        comp_post_insp_60d = c_post_insp[c_post_insp["receiveddate"] >= cutoff_60d_prior].groupby("bbl").size()
        comp_post_insp_90d = c_post_insp[c_post_insp["receiveddate"] >= cutoff_90d_prior].groupby("bbl").size()
        comp_post_insp_any = c_post_insp.groupby("bbl").size()
    elif len(c_hist) > 0:
        comp_post_insp_14d = c_hist[c_hist["receiveddate"] >= cutoff_14d_prior].groupby("bbl").size()
        comp_post_insp_30d = c_hist[c_hist["receiveddate"] >= cutoff_30d_prior].groupby("bbl").size()
        comp_post_insp_60d = c_hist[c_hist["receiveddate"] >= cutoff_60d_prior].groupby("bbl").size()
        comp_post_insp_90d = c_hist[c_hist["receiveddate"] >= cutoff_90d_prior].groupby("bbl").size()
        comp_post_insp_any = c_hist.groupby("bbl").size()
    else:
        comp_post_insp_14d = pd.Series(dtype=int)
        comp_post_insp_30d = pd.Series(dtype=int)
        comp_post_insp_60d = pd.Series(dtype=int)
        comp_post_insp_90d = pd.Series(dtype=int)
        comp_post_insp_any = pd.Series(dtype=int)

    # ERP charges (Total, HWO emergency boiler charges, OMO general charges)
    erp_hist = df_erp[(df_erp["bbl"].isin(bbl_set)) & (df_erp["charge_date"] < cutoff)]
    erp_1y = erp_hist[erp_hist["charge_date"] >= cutoff_1y_prior].groupby("bbl").size()
    erp_3y = erp_hist[erp_hist["charge_date"] >= cutoff_3y_prior].groupby("bbl").size()
    erp_total = erp_hist.groupby("bbl").size()
    recency_erp = (cutoff - erp_hist.groupby("bbl")["charge_date"].max()).dt.days if len(erp_hist) > 0 else pd.Series(dtype=float)

    hwo_hist = df_hwo[(df_hwo["bbl"].isin(bbl_set)) & (df_hwo["charge_date"] < cutoff)]
    hwo_1y = hwo_hist[hwo_hist["charge_date"] >= cutoff_1y_prior].groupby("bbl").size()
    recency_hwo = (cutoff - hwo_hist.groupby("bbl")["charge_date"].max()).dt.days if len(hwo_hist) > 0 else pd.Series(dtype=float)

    omo_hist = df_omo[(df_omo["bbl"].isin(bbl_set)) & (df_omo["charge_date"] < cutoff)]
    omo_1y = omo_hist[omo_hist["charge_date"] >= cutoff_1y_prior].groupby("bbl").size()
    recency_omo = (cutoff - omo_hist.groupby("bbl")["charge_date"].max()).dt.days if len(omo_hist) > 0 else pd.Series(dtype=float)

    # HPD registration compliance & delinquency
    if len(df_reg_dates) > 0:
        reg_hist = df_reg_dates[(df_reg_dates["bbl"].isin(bbl_set)) & (df_reg_dates["reg_date"] < cutoff)]
        recency_reg = (cutoff - reg_hist.groupby("bbl")["reg_date"].max()).dt.days if len(reg_hist) > 0 else pd.Series(dtype=float)
    else:
        recency_reg = pd.Series(dtype=float)

    # Vacate orders & Litigations
    vacate_cnt = df_vacate[(df_vacate["bbl"].isin(bbl_set)) & (df_vacate["vacate_effective_date"] < cutoff)].groupby("bbl").size()
    lit_hist = df_lit[(df_lit["bbl"].isin(bbl_set)) & (df_lit["caseopendate"] < cutoff)]
    lit_1y = lit_hist[lit_hist["caseopendate"] >= cutoff_1y_prior].groupby("bbl").size()
    lit_3y = lit_hist[lit_hist["caseopendate"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_lit = (cutoff - lit_hist.groupby("bbl")["caseopendate"].max()).dt.days if len(lit_hist) > 0 else pd.Series(dtype=float)

    # DOB Violations
    dob_hist = df_dob[(df_dob["bbl"].isin(bbl_set)) & (df_dob["issue_date"] < cutoff)]
    dob_1y = dob_hist[dob_hist["issue_date"] >= cutoff_1y_prior].groupby("bbl").size()
    dob_3y = dob_hist[dob_hist["issue_date"] >= cutoff_3y_prior].groupby("bbl").size()
    dob_all = dob_hist.groupby("bbl").size()
    recency_dob = (cutoff - dob_hist.groupby("bbl")["issue_date"].max()).dt.days if len(dob_hist) > 0 else pd.Series(dtype=float)

    # DOHMH Rodent Inspections
    rodent_hist = df_rodent[(df_rodent["bbl"].isin(bbl_set)) & (df_rodent["inspection_date"] < cutoff)]
    rodent_1y = rodent_hist[rodent_hist["inspection_date"] >= cutoff_1y_prior].groupby("bbl").size()
    rodent_3y = rodent_hist[rodent_hist["inspection_date"] >= cutoff_3y_prior].groupby("bbl").size()
    rodent_all = rodent_hist.groupby("bbl").size()
    recency_rodent = (cutoff - rodent_hist.groupby("bbl")["inspection_date"].max()).dt.days if len(rodent_hist) > 0 else pd.Series(dtype=float)
    rodent_active_1y = rodent_hist[(rodent_hist["inspection_date"] >= cutoff_1y_prior) & (rodent_hist["is_active"] > 0)].groupby("bbl").size()
    rodent_active_all = rodent_hist[rodent_hist["is_active"] > 0].groupby("bbl").size()

    # DOB Complaints
    dob_c_hist = df_dob_comp[(df_dob_comp["bbl"].isin(bbl_set)) & (df_dob_comp["complaint_date"] < cutoff)]
    dob_comp_1y = dob_c_hist[dob_c_hist["complaint_date"] >= cutoff_1y_prior].groupby("bbl").size()
    dob_comp_3y = dob_c_hist[dob_c_hist["complaint_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_dob_comp = (cutoff - dob_c_hist.groupby("bbl")["complaint_date"].max()).dt.days if len(dob_c_hist) > 0 else pd.Series(dtype=float)

    # Court Evictions
    evict_hist = df_evictions[(df_evictions["bbl"].isin(bbl_set)) & (df_evictions["eviction_date"] < cutoff)]
    evict_1y = evict_hist[evict_hist["eviction_date"] >= cutoff_1y_prior].groupby("bbl").size()
    evict_3y = evict_hist[evict_hist["eviction_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_evict = (cutoff - evict_hist.groupby("bbl")["eviction_date"].max()).dt.days if len(evict_hist) > 0 else pd.Series(dtype=float)

    # Bedbugs
    bedbug_hist = df_bedbugs[(df_bedbugs["bbl"].isin(bbl_set)) & (df_bedbugs["filing_date"] < cutoff)]
    bedbug_1y = bedbug_hist[bedbug_hist["filing_date"] >= cutoff_1y_prior].groupby("bbl").size()
    bedbug_3y = bedbug_hist[bedbug_hist["filing_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_bedbug = (cutoff - bedbug_hist.groupby("bbl")["filing_date"].max()).dt.days if len(bedbug_hist) > 0 else pd.Series(dtype=float)

    # DOB ECB Violations
    ecb_hist = df_ecb[(df_ecb["bbl"].isin(bbl_set)) & (df_ecb["ecb_date"] < cutoff)]
    ecb_1y = ecb_hist[ecb_hist["ecb_date"] >= cutoff_1y_prior].groupby("bbl").size()
    ecb_3y = ecb_hist[ecb_hist["ecb_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_ecb = (cutoff - ecb_hist.groupby("bbl")["ecb_date"].max()).dt.days if len(ecb_hist) > 0 else pd.Series(dtype=float)

    # DOB Safety Violations
    safety_hist = df_safety[(df_safety["bbl"].isin(bbl_set)) & (df_safety["safety_date"] < cutoff)]
    safety_1y = safety_hist[safety_hist["safety_date"] >= cutoff_1y_prior].groupby("bbl").size()
    safety_3y = safety_hist[safety_hist["safety_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_safety = (cutoff - safety_hist.groupby("bbl")["safety_date"].max()).dt.days if len(safety_hist) > 0 else pd.Series(dtype=float)

    # DOF Tax Lien Sales
    tax_lien_hist = df_tax_liens[(df_tax_liens["bbl"].isin(bbl_set)) & (df_tax_liens["sale_date"] < cutoff)]
    tax_liens_1y = tax_lien_hist[tax_lien_hist["sale_date"] >= cutoff_1y_prior].groupby("bbl").size()
    tax_liens_total = tax_lien_hist.groupby("bbl").size()
    recency_tax_lien = (cutoff - tax_lien_hist.groupby("bbl")["sale_date"].max()).dt.days if len(tax_lien_hist) > 0 else pd.Series(dtype=float)

    # DOB Stalled Construction Sites (Physical Abandonment Indicator)
    stalled_hist = df_stalled[
        (df_stalled["bbl"].isin(bbl_set))
        & ((df_stalled["stalled_date"].isna()) | (df_stalled["stalled_date"] < cutoff))
    ]
    stalled_site_cnt = stalled_hist.groupby("bbl").size()
    if len(stalled_hist) > 0 and stalled_hist["stalled_date"].notna().any():
        recency_stalled = (cutoff - stalled_hist.dropna(subset=["stalled_date"]).groupby("bbl")["stalled_date"].max()).dt.days
    else:
        recency_stalled = pd.Series(dtype=float)

    # DOF Annualized Sales (Ownership Turnover & Sales Distress)
    sales_hist = df_sales[(df_sales["bbl"].isin(bbl_set)) & (df_sales["sale_date"] < cutoff)]
    sales_1y = sales_hist[sales_hist["sale_date"] >= cutoff_1y_prior].groupby("bbl").size()
    sales_3y = sales_hist[sales_hist["sale_date"] >= cutoff_3y_prior].groupby("bbl").size()
    recency_sales = (cutoff - sales_hist.groupby("bbl")["sale_date"].max()).dt.days if len(sales_hist) > 0 else pd.Series(dtype=float)

    # Historical tax block Class C violation rates pre-cutoff
    v_c_block = v_c.assign(block_id=v_c["bbl"].str[:6])
    block_c_1y = v_c_block[v_c_block["inspectiondate"] >= cutoff_1y_prior].groupby("block_id").size()
    block_c_3y = v_c_block[v_c_block["inspectiondate"] >= cutoff_3y_prior].groupby("block_id").size()

    # Assemble base features dictionary
    raw_feature_dict = {
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
        "insp_visits_1y": insp_visits_1y,
        "insp_visits_3y": insp_visits_3y,
        "c_decay_180d": c_decay_180,
        "c_decay_365d": c_decay_365,
        "total_decay_180d": tot_decay_180,
        "total_decay_365d": tot_decay_365,
        "c_years_with_viol": c_years_with_viol,
        "open_c_viol": open_c,
        "open_c_max_age_days": open_c_max_age_days,
        "open_c_lingering_90d": open_c_lingering_90d,
        "open_b_viol": open_b,
        "open_a_viol": open_a,
        "open_total_viol": open_total,
        "days_since_last_c": recency_c,
        "days_since_last_b": recency_b,
        "days_since_last_any": recency_any,
        "days_since_last_reg": recency_reg,
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
        "days_since_last_erp": recency_erp,
        "hwo_charges_1y": hwo_1y,
        "days_since_last_hwo": recency_hwo,
        "omo_charges_1y": omo_1y,
        "days_since_last_omo": recency_omo,
        "vacate_orders_hist": vacate_cnt,
        "litigations_1y": lit_1y,
        "litigations_3y": lit_3y,
        "days_since_last_lit": recency_lit,
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
        "dob_comp_1y": dob_comp_1y,
        "dob_comp_3y": dob_comp_3y,
        "days_since_last_dob_comp": recency_dob_comp,
        "evictions_1y": evict_1y,
        "evictions_3y": evict_3y,
        "days_since_last_eviction": recency_evict,
        "bedbug_reports_1y": bedbug_1y,
        "bedbug_reports_3y": bedbug_3y,
        "days_since_last_bedbug": recency_bedbug,
        "dob_ecb_viol_1y": ecb_1y,
        "dob_ecb_viol_3y": ecb_3y,
        "days_since_last_ecb": recency_ecb,
        "dob_safety_1y": safety_1y,
        "dob_safety_3y": safety_3y,
        "days_since_last_safety": recency_safety,
        "tax_liens_1y": tax_liens_1y,
        "tax_liens_total": tax_liens_total,
        "days_since_last_tax_lien": recency_tax_lien,
        "is_stalled_construction": stalled_site_cnt,
        "days_since_last_stalled": recency_stalled,
        "sales_1y": sales_1y,
        "sales_3y": sales_3y,
        "days_since_last_sale": recency_sales,
        "overdue_uncert_c_viol": overdue_c,
        "max_days_overdue_c": max_days_overdue_c,
        "heat_c_viol_30d": heat_c_30d,
        "heat_c_viol_60d": heat_c_60d,
        "heat_c_viol_90d": heat_c_90d,
        "heat_c_viol_1y": heat_c_1y,
        "heat_c_viol_3y": heat_c_3y,
        "lead_c_viol_1y": lead_c_1y,
        "lead_c_viol_3y": lead_c_3y,
        "mold_c_viol_1y": mold_c_1y,
        "mold_c_viol_3y": mold_c_3y,
        "doors_c_viol_1y": doors_c_1y,
        "doors_c_viol_3y": doors_c_3y,
        "rent_impairing_c_1y": rent_c_1y,
        "pest_c_viol_1y": pest_c_1y,
        "pest_c_viol_3y": pest_c_3y,
        "c_viol_distinct_apts_1y": c_apts_1y,
        "mean_cure_days_c_1y": mean_cure_days_c_1y,
        "mean_cure_days_c_3y": mean_cure_days_c_3y,
        "comp_heat_14d": comp_heat_14d,
        "comp_heat_30d": comp_heat_30d,
        "comp_heat_60d": comp_heat_60d,
        "comp_heat_90d": comp_heat_90d,
        "comp_distinct_apts_30d": comp_apts_30d,
        "comp_distinct_apts_1y": comp_apts_1y,
        "comp_distinct_apts_3y": comp_apts_3y,
        "comp_heat_1y": comp_heat_1y,
        "comp_leak_1y": comp_leak_1y,
        "comp_paint_1y": comp_paint_1y,
        "comp_unsanitary_1y": comp_unsanitary_1y,
        "comp_post_insp_14d": comp_post_insp_14d,
        "comp_post_insp_30d": comp_post_insp_30d,
        "comp_post_insp_60d": comp_post_insp_60d,
        "comp_post_insp_90d": comp_post_insp_90d,
    }

    recency_cols = {
        "days_since_last_c",
        "days_since_last_b",
        "days_since_last_any",
        "days_since_last_reg",
        "days_since_last_comp",
        "days_since_last_erp",
        "days_since_last_hwo",
        "days_since_last_omo",
        "days_since_last_lit",
        "days_since_last_dob",
        "days_since_last_rodent",
        "days_since_last_dob_comp",
        "days_since_last_eviction",
        "days_since_last_bedbug",
        "days_since_last_ecb",
        "days_since_last_safety",
        "days_since_last_tax_lien",
        "days_since_last_stalled",
        "days_since_last_sale",
    }

    # Consolidated dictionary batching to completely eliminate pandas fragmentation warnings
    cohort_dict = {"bbl": bbl_series}

    for col_name, s in raw_feature_dict.items():
        if col_name in recency_cols:
            cohort_dict[col_name] = bbl_series.map(s).fillna(3650.0).clip(0, 3650).values.astype(np.float32)
        else:
            cohort_dict[col_name] = bbl_series.map(s).fillna(0.0).values.astype(np.float32)

    # Administrative distress indicator watchlists
    cohort_dict["is_aep_building"] = bbl_series.isin(aep_bbl_set).values.astype(np.float32)
    cohort_dict["is_speculation_watchlist"] = bbl_series.isin(spec_bbl_set).values.astype(np.float32)
    cohort_dict["is_conh_building"] = bbl_series.isin(conh_bbl_set).values.astype(np.float32)
    cohort_dict["is_underlying_conditions"] = bbl_series.isin(uc_bbl_set).values.astype(np.float32)

    # Block-level contagion & unit-normalized contagion
    block_c_1y_val = block_id_series.map(block_c_1y).fillna(0.0).values.astype(np.float32)
    block_c_3y_val = block_id_series.map(block_c_3y).fillna(0.0).values.astype(np.float32)
    cohort_dict["block_c_viol_1y"] = block_c_1y_val
    cohort_dict["block_c_viol_3y"] = block_c_3y_val
    cohort_dict["block_neighbor_c_1y"] = np.maximum(block_c_1y_val - cohort_dict["c_viol_1y"], 0.0).astype(np.float32)
    cohort_dict["block_neighbor_c_3y"] = np.maximum(block_c_3y_val - cohort_dict["c_viol_3y"], 0.0).astype(np.float32)

    block_tot_units = block_id_series.map(block_units_map).fillna(1.0).clip(lower=1.0).values.astype(np.float32)
    cohort_dict["block_neighbor_c_per_unit_1y"] = (cohort_dict["block_neighbor_c_1y"] / block_tot_units).astype(np.float32)
    cohort_dict["block_neighbor_c_per_unit_3y"] = (cohort_dict["block_neighbor_c_3y"] / block_tot_units).astype(np.float32)

    # PLUTO morphology records
    pluto_defaults = {
        "borocode": 1.0,
        "zipcode": 0,
        "unitsres": 1.0,
        "unitstotal": 1.0,
        "yearbuilt": 1950.0,
        "yearalter1": 0.0,
        "numfloors": 3.0,
        "bldgarea": 0.0,
        "resarea": 0.0,
        "commercial_share": 0.0,
        "lotarea": 0.0,
        "assesstot": 0.0,
        "assessland": 0.0,
        "builtfar": 0.0,
        "landuse": 0.0,
        "cd": 101.0,
        "bldgclass_first": "C",
    }
    morph_records = [
        pluto_features.get(
            bbl,
            {
                "borocode": float(bbl[0]) if bbl and bbl[0].isdigit() else 1.0,
                "zipcode": 0,
                "unitsres": 1.0,
                "unitstotal": 1.0,
                "yearbuilt": 1950.0,
                "yearalter1": 0.0,
                "numfloors": 3.0,
                "bldgarea": 0.0,
                "resarea": 0.0,
                "commercial_share": 0.0,
                "lotarea": 0.0,
                "assesstot": 0.0,
                "assessland": 0.0,
                "builtfar": 0.0,
                "landuse": 0.0,
                "cd": 101.0,
                "bldgclass_first": "C",
            },
        )
        for bbl in bbl_list
    ]
    df_morph = pd.DataFrame(morph_records)

    unitsres_safe = np.maximum(df_morph["unitsres"].values.astype(np.float32), 1.0)
    floors_safe = np.maximum(df_morph["numfloors"].values.astype(np.float32), 1.0)
    bldgarea_safe = np.maximum(df_morph["bldgarea"].values.astype(np.float32), 1.0)
    cutoff_year = cutoff.year

    cohort_dict["unitsres"] = unitsres_safe
    cohort_dict["unitstotal"] = np.maximum(df_morph["unitstotal"].values.astype(np.float32), 1.0)
    cohort_dict["yearbuilt"] = df_morph["yearbuilt"].values.astype(np.float32)
    cohort_dict["yearalter1"] = df_morph["yearalter1"].values.astype(np.float32)
    cohort_dict["numfloors"] = floors_safe
    cohort_dict["bldgarea"] = df_morph["bldgarea"].values.astype(np.float32)
    cohort_dict["resarea"] = df_morph["resarea"].values.astype(np.float32)
    cohort_dict["commercial_share"] = df_morph["commercial_share"].values.astype(np.float32)
    cohort_dict["lotarea"] = df_morph["lotarea"].values.astype(np.float32)
    cohort_dict["assesstot"] = df_morph["assesstot"].values.astype(np.float32)
    cohort_dict["assessland"] = df_morph["assessland"].values.astype(np.float32)
    cohort_dict["builtfar"] = df_morph["builtfar"].values.astype(np.float32)
    cohort_dict["landuse"] = df_morph["landuse"].values.astype(np.float32)

    cohort_dict["building_age"] = np.clip(cutoff_year - cohort_dict["yearbuilt"], 0.0, 200.0).astype(np.float32)
    cohort_dict["is_prewar"] = (cohort_dict["yearbuilt"] < 1940.0).astype(np.float32)
    cohort_dict["is_pre1960"] = (cohort_dict["yearbuilt"] < 1960.0).astype(np.float32)
    cohort_dict["area_per_unit"] = np.clip(cohort_dict["bldgarea"] / unitsres_safe, 0.0, 10000.0).astype(np.float32)
    cohort_dict["units_per_floor"] = np.clip(unitsres_safe / floors_safe, 0.0, 100.0).astype(np.float32)
    cohort_dict["res_area_ratio"] = np.clip(cohort_dict["resarea"] / bldgarea_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict["log_unitsres"] = np.log1p(unitsres_safe).astype(np.float32)
    cohort_dict["log_bldgarea"] = np.log1p(cohort_dict["bldgarea"]).astype(np.float32)

    # Derived totals and visit cadence ratio
    cohort_dict["total_viol_1y"] = (cohort_dict["c_viol_1y"] + cohort_dict["b_viol_1y"] + cohort_dict["a_viol_1y"]).astype(np.float32)
    cohort_dict["total_viol_2y"] = (cohort_dict["c_viol_2y"] + cohort_dict["b_viol_2y"] + cohort_dict["a_viol_2y"]).astype(np.float32)
    cohort_dict["total_viol_3y"] = (cohort_dict["c_viol_3y"] + cohort_dict["b_viol_3y"]).astype(np.float32)
    cohort_dict["c_per_visit_1y"] = (cohort_dict["c_viol_1y"] / (cohort_dict["insp_visits_1y"] + 1e-4)).astype(np.float32)

    # Per-unit metrics
    cohort_dict["c_viol_30d_per_unit"] = (cohort_dict["c_viol_30d"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_viol_90d_per_unit"] = (cohort_dict["c_viol_90d"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_viol_180d_per_unit"] = (cohort_dict["c_viol_180d"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_viol_1y_per_unit"] = (cohort_dict["c_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_viol_3y_per_unit"] = (cohort_dict["c_viol_3y"] / unitsres_safe).astype(np.float32)

    cohort_dict["total_viol_90d_per_unit"] = (cohort_dict["total_viol_90d"] / unitsres_safe).astype(np.float32)
    cohort_dict["total_viol_1y_per_unit"] = (cohort_dict["total_viol_1y"] / unitsres_safe).astype(np.float32)

    cohort_dict["c_decay_180d_per_unit"] = (cohort_dict["c_decay_180d"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_decay_365d_per_unit"] = (cohort_dict["c_decay_365d"] / unitsres_safe).astype(np.float32)
    cohort_dict["total_decay_180d_per_unit"] = (cohort_dict["total_decay_180d"] / unitsres_safe).astype(np.float32)
    cohort_dict["total_decay_365d_per_unit"] = (cohort_dict["total_decay_365d"] / unitsres_safe).astype(np.float32)

    cohort_dict["complaints_30d_per_unit"] = (cohort_dict["complaints_30d"] / unitsres_safe).astype(np.float32)
    cohort_dict["complaints_60d_per_unit"] = (cohort_dict["complaints_60d"] / unitsres_safe).astype(np.float32)
    cohort_dict["complaints_90d_per_unit"] = (cohort_dict["complaints_90d"] / unitsres_safe).astype(np.float32)
    cohort_dict["complaints_1y_per_unit"] = (cohort_dict["complaints_1y"] / unitsres_safe).astype(np.float32)

    cohort_dict["open_c_viol_per_unit"] = (cohort_dict["open_c_viol"] / unitsres_safe).astype(np.float32)
    cohort_dict["open_b_viol_per_unit"] = (cohort_dict["open_b_viol"] / unitsres_safe).astype(np.float32)
    cohort_dict["open_total_per_unit"] = (cohort_dict["open_total_viol"] / unitsres_safe).astype(np.float32)
    cohort_dict["open_viol_ratio_total"] = (cohort_dict["open_total_viol"] / (cohort_dict["total_viol_all"] + 1.0)).astype(np.float32)
    cohort_dict["open_c_ratio_total"] = (cohort_dict["open_c_viol"] / (cohort_dict["c_viol_all"] + 1.0)).astype(np.float32)
    cohort_dict["is_reg_delinquent"] = (cohort_dict["days_since_last_reg"] > 365.0).astype(np.float32)
    cohort_dict["erp_charges_1y_per_unit"] = (cohort_dict["erp_charges_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["erp_charges_total_per_unit"] = (cohort_dict["erp_charges_total"] / unitsres_safe).astype(np.float32)
    cohort_dict["hwo_charges_1y_per_unit"] = (cohort_dict["hwo_charges_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["omo_charges_1y_per_unit"] = (cohort_dict["omo_charges_1y"] / unitsres_safe).astype(np.float32)

    cohort_dict["dob_viol_1y_per_unit"] = (cohort_dict["dob_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_viol_all_per_unit"] = (cohort_dict["dob_viol_all"] / unitsres_safe).astype(np.float32)
    cohort_dict["rodent_insp_1y_per_unit"] = (cohort_dict["rodent_insp_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["rodent_insp_all_per_unit"] = (cohort_dict["rodent_insp_all"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_comp_1y_per_unit"] = (cohort_dict["dob_comp_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_comp_3y_per_unit"] = (cohort_dict["dob_comp_3y"] / unitsres_safe).astype(np.float32)
    cohort_dict["evictions_1y_per_unit"] = (cohort_dict["evictions_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["evictions_3y_per_unit"] = (cohort_dict["evictions_3y"] / unitsres_safe).astype(np.float32)
    cohort_dict["bedbug_reports_1y_per_unit"] = (cohort_dict["bedbug_reports_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["bedbug_reports_3y_per_unit"] = (cohort_dict["bedbug_reports_3y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_ecb_viol_1y_per_unit"] = (cohort_dict["dob_ecb_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_ecb_viol_3y_per_unit"] = (cohort_dict["dob_ecb_viol_3y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_safety_1y_per_unit"] = (cohort_dict["dob_safety_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["dob_safety_3y_per_unit"] = (cohort_dict["dob_safety_3y"] / unitsres_safe).astype(np.float32)

    # Class B single-room occupancy and legal building unit features
    cohort_dict["hpd_classb_units"] = bbl_series.map(hpd_classb_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["hpd_total_units"] = bbl_series.map(hpd_total_units_map).fillna(0.0).values.astype(np.float32)
    hpd_units_denom = np.maximum(cohort_dict["hpd_total_units"], unitsres_safe)
    cohort_dict["classb_unit_ratio"] = np.clip(cohort_dict["hpd_classb_units"] / hpd_units_denom, 0.0, 1.0).astype(np.float32)
    cohort_dict["has_classb_units"] = (cohort_dict["hpd_classb_units"] > 0).astype(np.float32)

    # Delinquent uncertified backlog and heating citation metrics
    cohort_dict["overdue_uncert_c_per_unit"] = (cohort_dict["overdue_uncert_c_viol"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_viol_1y_per_unit"] = (cohort_dict["heat_c_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_viol_3y_per_unit"] = (cohort_dict["heat_c_viol_3y"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_ratio_1y"] = ((cohort_dict["heat_c_viol_1y"] + 0.1) / (cohort_dict["c_viol_1y"] + 0.1)).astype(np.float32)

    # Lead-lag recency delta and pending inspection queue indicators
    cohort_dict["has_uninspected_comp"] = (bbl_series.map(comp_post_insp_any).fillna(0.0).values > 0).astype(np.float32)
    cohort_dict["insp_comp_recency_diff"] = (cohort_dict["days_since_last_any"] - cohort_dict["days_since_last_comp"]).astype(np.float32)

    # Landlord portfolio distress metrics via authentic multi-building owner key mapping
    landlord_id_series = bbl_series.map(bbl_to_landlord)
    has_landlord = landlord_id_series.notna().values & (landlord_id_series.values != "")

    portfolio_lot_count = np.ones(len(bbl_list), dtype=np.float32)
    portfolio_c_1y_arr = cohort_dict["c_viol_1y"].copy()
    portfolio_c_3y_arr = cohort_dict["c_viol_3y"].copy()

    if has_landlord.any():
        temp_port_df = pd.DataFrame({
            "landlord_id": landlord_id_series[has_landlord],
            "c_1y": cohort_dict["c_viol_1y"][has_landlord],
            "c_3y": cohort_dict["c_viol_3y"][has_landlord],
        })
        port_size = temp_port_df.groupby("landlord_id")["c_1y"].transform("count").values
        port_c_1y = temp_port_df.groupby("landlord_id")["c_1y"].transform("sum").values
        port_c_3y = temp_port_df.groupby("landlord_id")["c_3y"].transform("sum").values

        portfolio_lot_count[has_landlord] = port_size.astype(np.float32)
        portfolio_c_1y_arr[has_landlord] = port_c_1y.astype(np.float32)
        portfolio_c_3y_arr[has_landlord] = port_c_3y.astype(np.float32)

    cohort_dict["portfolio_lot_count"] = portfolio_lot_count
    cohort_dict["portfolio_c_1y"] = portfolio_c_1y_arr
    cohort_dict["portfolio_c_3y"] = portfolio_c_3y_arr

    other_lots = np.maximum(portfolio_lot_count - 1.0, 0.0)
    cohort_dict["portfolio_loo_c_rate_1y"] = np.where(
        other_lots > 0,
        (portfolio_c_1y_arr - cohort_dict["c_viol_1y"]) / np.maximum(other_lots, 1e-4),
        0.0,
    ).astype(np.float32)
    cohort_dict["portfolio_loo_c_rate_3y"] = np.where(
        other_lots > 0,
        (portfolio_c_3y_arr - cohort_dict["c_viol_3y"]) / np.maximum(other_lots, 1e-4),
        0.0,
    ).astype(np.float32)

    # Short-term velocity and acceleration metrics
    cohort_dict["c_viol_velocity_90d"] = ((cohort_dict["c_viol_90d"] * 4.0 + 0.1) / (cohort_dict["c_viol_1y"] + 0.1)).astype(np.float32)
    cohort_dict["c_viol_velocity_30d"] = ((cohort_dict["c_viol_30d"] * 12.0 + 0.1) / (cohort_dict["c_viol_1y"] + 0.1)).astype(np.float32)
    cohort_dict["c_viol_velocity_180d"] = ((cohort_dict["c_viol_180d"] * 2.0 + 0.1) / (cohort_dict["c_viol_1y"] + 0.1)).astype(np.float32)
    cohort_dict["c_viol_accel_30_90"] = ((cohort_dict["c_viol_30d"] * 3.0 + 0.1) / (cohort_dict["c_viol_90d"] + 0.1)).astype(np.float32)

    cohort_dict["total_viol_velocity_90d"] = ((cohort_dict["total_viol_90d"] * 4.0 + 0.1) / (cohort_dict["total_viol_1y"] + 0.1)).astype(np.float32)
    cohort_dict["total_viol_accel_30_90"] = ((cohort_dict["total_viol_30d"] * 3.0 + 0.1) / (cohort_dict["total_viol_90d"] + 0.1)).astype(np.float32)

    # Longitudinal trajectory deltas across y1, y2, y3 and second-order acceleration
    c_y1 = cohort_dict["c_viol_1y"]
    c_y2 = np.maximum(cohort_dict["c_viol_2y"] - cohort_dict["c_viol_1y"], 0.0)
    c_y3 = np.maximum(cohort_dict["c_viol_3y"] - cohort_dict["c_viol_2y"], 0.0)
    cohort_dict["c_viol_y1"] = c_y1
    cohort_dict["c_viol_y2"] = c_y2
    cohort_dict["c_viol_y3"] = c_y3
    cohort_dict["c_viol_delta_1_2"] = (c_y1 - c_y2).astype(np.float32)
    cohort_dict["c_viol_delta_2_3"] = (c_y2 - c_y3).astype(np.float32)
    cohort_dict["c_viol_accel_2nd"] = (cohort_dict["c_viol_delta_1_2"] - cohort_dict["c_viol_delta_2_3"]).astype(np.float32)

    has_c_y1 = (c_y1 > 0).astype(int)
    has_c_y2 = (c_y2 > 0).astype(int)
    has_c_y3 = (c_y3 > 0).astype(int)
    cohort_dict["c_persistence_streak"] = np.where(
        has_c_y1,
        1 + np.where(has_c_y2, 1 + np.where(has_c_y3, 1, 0), 0),
        0,
    ).astype(np.float32)

    cohort_dict["heat_c_viol_30d_per_unit"] = (cohort_dict["heat_c_viol_30d"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_viol_60d_per_unit"] = (cohort_dict["heat_c_viol_60d"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_viol_90d_per_unit"] = (cohort_dict["heat_c_viol_90d"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_heat_30d_per_unit"] = (cohort_dict["comp_heat_30d"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_heat_60d_per_unit"] = (cohort_dict["comp_heat_60d"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_heat_90d_per_unit"] = (cohort_dict["comp_heat_90d"] / unitsres_safe).astype(np.float32)
    cohort_dict["heat_c_velocity_30d"] = ((cohort_dict["heat_c_viol_30d"] * 12.0 + 0.1) / (cohort_dict["heat_c_viol_1y"] + 0.1)).astype(np.float32)
    cohort_dict["comp_apt_dispersion_30d"] = np.clip(cohort_dict["comp_distinct_apts_30d"] / unitsres_safe, 0.0, 1.0).astype(np.float32)

    cohort_dict["c_viol_acceleration"] = ((cohort_dict["c_viol_1y"] + 0.1) / (c_y2 + 0.1)).astype(np.float32)
    cohort_dict["complaint_velocity"] = ((cohort_dict["complaints_90d"] * 4.0 + 0.1) / (cohort_dict["complaints_1y"] + 0.1)).astype(np.float32)
    cohort_dict["complaint_accel_30d"] = ((cohort_dict["complaints_30d"] * 12.0 + 0.1) / (cohort_dict["complaints_1y"] + 0.1)).astype(np.float32)
    cohort_dict["complaint_accel_60d"] = ((cohort_dict["complaints_60d"] * 6.0 + 0.1) / (cohort_dict["complaints_1y"] + 0.1)).astype(np.float32)
    cohort_dict["unresolved_viol_ratio"] = (
        (cohort_dict["open_c_viol"] + cohort_dict["open_b_viol"]) / (cohort_dict["c_viol_all"] + cohort_dict["b_viol_all"] + 1.0)
    ).astype(np.float32)

    cohort_dict["c_severity_share_90d"] = (cohort_dict["c_viol_90d"] / (cohort_dict["total_viol_90d"] + 1.0)).astype(np.float32)
    cohort_dict["c_severity_share_1y"] = (cohort_dict["c_viol_1y"] / (cohort_dict["total_viol_1y"] + 1.0)).astype(np.float32)
    cohort_dict["c_severity_share_3y"] = (cohort_dict["c_viol_3y"] / (cohort_dict["total_viol_3y"] + 1.0)).astype(np.float32)

    # Class C uncured ratio and multi-year severity escalation
    cohort_dict["c_uncured_ratio"] = (cohort_dict["open_c_viol"] / (cohort_dict["c_viol_1y"] + 1e-4)).astype(np.float32)
    cohort_dict["c_severity_escalation"] = (cohort_dict["c_severity_share_1y"] / (cohort_dict["c_severity_share_3y"] + 1e-4)).astype(np.float32)

    # Multi-agency distress index
    cohort_dict["cross_agency_distress_1y"] = (
        cohort_dict["c_viol_1y"]
        + cohort_dict["dob_viol_1y"]
        + cohort_dict["dob_ecb_viol_1y"]
        + cohort_dict["dob_comp_1y"]
        + cohort_dict["evictions_1y"]
        + cohort_dict["bedbug_reports_1y"]
        + cohort_dict["erp_charges_1y"]
        + cohort_dict["litigations_1y"]
    ).astype(np.float32)

    cohort_dict["winter_c_ratio"] = (cohort_dict["c_viol_winter"] / (cohort_dict["c_viol_1y"] + 1.0)).astype(np.float32)
    cohort_dict["winter_comp_ratio"] = (cohort_dict["complaints_winter"] / (cohort_dict["complaints_1y"] + 1.0)).astype(np.float32)
    cohort_dict["c_viol_winter_per_unit"] = (cohort_dict["c_viol_winter"] / unitsres_safe).astype(np.float32)
    cohort_dict["complaints_winter_per_unit"] = (cohort_dict["complaints_winter"] / unitsres_safe).astype(np.float32)
    cohort_dict["c_to_comp_ratio_1y"] = ((cohort_dict["c_viol_1y"] + 0.1) / (cohort_dict["complaints_1y"] + 0.1)).astype(np.float32)
    cohort_dict["total_to_comp_ratio_1y"] = ((cohort_dict["total_viol_1y"] + 0.1) / (cohort_dict["complaints_1y"] + 0.1)).astype(np.float32)
    cohort_dict["winter_c_to_comp_ratio"] = ((cohort_dict["c_viol_winter"] + 0.1) / (cohort_dict["complaints_winter"] + 0.1)).astype(np.float32)
    cohort_dict["open_c_lingering_ratio"] = (cohort_dict["open_c_lingering_90d"] / (cohort_dict["open_c_viol"] + 1.0)).astype(np.float32)

    cohort_dict["comp_apt_dispersion_1y"] = np.clip(cohort_dict["comp_distinct_apts_1y"] / unitsres_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict["comp_apt_dispersion_3y"] = np.clip(cohort_dict["comp_distinct_apts_3y"] / unitsres_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict["comp_heat_1y_per_unit"] = (cohort_dict["comp_heat_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_leak_1y_per_unit"] = (cohort_dict["comp_leak_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_paint_1y_per_unit"] = (cohort_dict["comp_paint_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_unsanitary_1y_per_unit"] = (cohort_dict["comp_unsanitary_1y"] / unitsres_safe).astype(np.float32)

    cohort_dict["lead_c_viol_1y_per_unit"] = (cohort_dict["lead_c_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["mold_c_viol_1y_per_unit"] = (cohort_dict["mold_c_viol_1y"] / unitsres_safe).astype(np.float32)
    cohort_dict["doors_c_viol_1y_per_unit"] = (cohort_dict["doors_c_viol_1y"] / unitsres_safe).astype(np.float32)

    cohort_dict["lingering_c_rate_90d"] = (cohort_dict["open_c_lingering_90d"] / (cohort_dict["open_c_viol"] + 1e-4)).astype(np.float32)
    cohort_dict["lingering_c_rate_1y"] = (cohort_dict["open_c_lingering_90d"] / (cohort_dict["c_viol_1y"] + 1.0)).astype(np.float32)

    cohort_dict["assessed_val_per_unit"] = np.clip(cohort_dict["assesstot"] / unitsres_safe, 0.0, 1e7).astype(np.float32)
    cohort_dict["land_to_total_assess_ratio"] = np.where(
        cohort_dict["assesstot"] > 0,
        np.clip(cohort_dict["assessland"] / np.maximum(cohort_dict["assesstot"], 1.0), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # Community district relative risk
    temp_cd_df = pd.DataFrame({"cd": df_morph["cd"].values, "c_viol_1y_per_unit": cohort_dict["c_viol_1y_per_unit"]})
    cd_mean = temp_cd_df.groupby("cd")["c_viol_1y_per_unit"].transform("mean").values.astype(np.float32)
    cohort_dict["relative_c_viol_risk"] = (cohort_dict["c_viol_1y_per_unit"] / (cd_mean + 1e-4)).astype(np.float32)

    # Pre-cutoff macro-spatial density aggregations (ZIP & CD rates per unit)
    v_c_all_1y = df_violations[(df_violations["class"] == "C") & (df_violations["inspectiondate"] >= cutoff_1y_prior) & (df_violations["inspectiondate"] < cutoff)]
    v_c_all_3y = df_violations[(df_violations["class"] == "C") & (df_violations["inspectiondate"] >= cutoff_3y_prior) & (df_violations["inspectiondate"] < cutoff)]
    comp_all_1y = df_complaints[(df_complaints["receiveddate"] >= cutoff_1y_prior) & (df_complaints["receiveddate"] < cutoff)]
    comp_all_3y = df_complaints[(df_complaints["receiveddate"] >= cutoff_3y_prior) & (df_complaints["receiveddate"] < cutoff)]

    cd_c_cnt_1y = v_c_all_1y.groupby("cd").size()
    cd_c_cnt_3y = v_c_all_3y.groupby("cd").size()
    zip_c_cnt_1y = v_c_all_1y.groupby("zipcode").size()
    zip_c_cnt_3y = v_c_all_3y.groupby("zipcode").size()

    cd_comp_cnt_1y = comp_all_1y.groupby("cd").size()
    cd_comp_cnt_3y = comp_all_3y.groupby("cd").size()
    zip_comp_cnt_1y = comp_all_1y.groupby("zipcode").size()
    zip_comp_cnt_3y = comp_all_3y.groupby("zipcode").size()

    cd_c_rate_1y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_c_cnt_1y.items()}
    cd_c_rate_3y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_c_cnt_3y.items()}
    zip_c_rate_1y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_c_cnt_1y.items()}
    zip_c_rate_3y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_c_cnt_3y.items()}

    cd_comp_rate_1y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_comp_cnt_1y.items()}
    cd_comp_rate_3y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_comp_cnt_3y.items()}
    zip_comp_rate_1y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_comp_cnt_1y.items()}
    zip_comp_rate_3y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_comp_cnt_3y.items()}

    cohort_dict["cd_c_viol_rate_1y"] = df_morph["cd"].map(cd_c_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["cd_c_viol_rate_3y"] = df_morph["cd"].map(cd_c_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["zip_c_viol_rate_1y"] = df_morph["zipcode"].map(zip_c_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["zip_c_viol_rate_3y"] = df_morph["zipcode"].map(zip_c_rate_3y_map).fillna(0.0).values.astype(np.float32)

    cohort_dict["cd_comp_rate_1y"] = df_morph["cd"].map(cd_comp_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["cd_comp_rate_3y"] = df_morph["cd"].map(cd_comp_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["zip_comp_rate_1y"] = df_morph["zipcode"].map(zip_comp_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict["zip_comp_rate_3y"] = df_morph["zipcode"].map(zip_comp_rate_3y_map).fillna(0.0).values.astype(np.float32)

    # Acute 14-day winter heating complaint dynamics and uninspected complaint queues
    cohort_dict["comp_heat_velocity_14d"] = (
        (cohort_dict["comp_heat_14d"] * 26.0 + 0.1) / (cohort_dict["comp_heat_1y"] + 0.1)
    ).astype(np.float32)
    cohort_dict["comp_heat_14d_per_unit"] = (cohort_dict["comp_heat_14d"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_post_insp_14d_per_unit"] = (cohort_dict["comp_post_insp_14d"] / unitsres_safe).astype(np.float32)
    cohort_dict["comp_post_insp_30d_per_unit"] = (cohort_dict["comp_post_insp_30d"] / unitsres_safe).astype(np.float32)
    cohort_dict["has_uninspected_comp_14d"] = (cohort_dict["comp_post_insp_14d"] > 0).astype(np.float32)
    cohort_dict["has_uninspected_comp_30d"] = (cohort_dict["comp_post_insp_30d"] > 0).astype(np.float32)

    # Discrete spatial and typology categorical integer codes
    cohort_dict["borocode"] = (
        pd.to_numeric(df_morph["borocode"], errors="coerce")
        .fillna(1.0)
        .astype(int)
        - 1
    ).clip(0, 4).values

    cohort_dict["cd"] = (
        df_morph["cd"]
        .map(lambda x: cd_to_idx.get(float(x) if pd.notna(x) else 101.0, 0))
        .astype(int)
        .values
    )

    class_map = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
        "G": 6, "H": 7, "I": 8, "R": 9, "S": 10,
    }
    cohort_dict["bldgclass_code"] = df_morph["bldgclass_first"].map(class_map).fillna(2).astype(int).values
    cohort_dict["zipcode"] = (
        df_morph["zipcode"]
        .map(lambda x: zip_to_idx.get(int(x) if pd.notna(x) else 0, 0))
        .astype(int)
        .values
    )

    # Instantiating the consolidated DataFrame in one shot to prevent DataFrame fragmentation
    df_cohort = pd.DataFrame(cohort_dict)

    if not is_test:
        window_end = cutoff + pd.DateOffset(years=1)
        v_target = df_violations[
            (df_violations["class"] == "C")
            & (df_violations["inspectiondate"] >= cutoff)
            & (df_violations["inspectiondate"] < window_end)
        ]
        c_counts = v_target.groupby("bbl").size()
        forward_c_count = bbl_series.map(c_counts).fillna(0.0).values.astype(np.float32)
        df_cohort["target"] = (forward_c_count > 0).astype(int)
        df_cohort["forward_c_severity"] = np.log1p(forward_c_count).astype(np.float32)

    return df_cohort


train_entities = entity_universe_df["bbl"].unique()
val_entities = entity_universe_df["bbl"].unique()

train_df_2019 = compute_cohort_dataset("2019-01-01", train_entities, is_test=False)
train_df_2019["sample_weight"] = 0.70

train_df_2020 = compute_cohort_dataset("2020-01-01", train_entities, is_test=False)
train_df_2020["sample_weight"] = 0.85

train_df_2021 = compute_cohort_dataset("2021-01-01", train_entities, is_test=False)
train_df_2021["sample_weight"] = 1.00

train_df = pd.concat([train_df_2019, train_df_2020, train_df_2021], ignore_index=True)
sample_weights = train_df["sample_weight"].values.astype(np.float32)
del train_df_2019, train_df_2020, train_df_2021
gc.collect()

val_df = compute_cohort_dataset("2022-01-01", val_entities, is_test=False)
test_df = compute_cohort_dataset("2023-01-01", test_bbls, is_test=True)

test_df = test_entities_df[["bbl"]].merge(test_df, on="bbl", how="left")

cat_cols = ["borocode", "cd", "bldgclass_code", "zipcode"]
cont_cols = [
    c for c in val_df.columns
    if c not in ["bbl", "target", "forward_c_severity", "sample_weight"] and c not in cat_cols
]
feature_cols = cont_cols + cat_cols

test_df[cont_cols] = test_df[cont_cols].fillna(0.0).astype(np.float32)
for c in cat_cols:
    test_df[c] = test_df[c].fillna(0).astype(int)

del df_violations, df_complaints, df_vacate, df_lit, df_erp, df_hwo, df_omo, df_reg_dates, df_dob, df_rodent, df_dob_comp, df_evictions, df_bedbugs, df_ecb, df_safety, df_tax_liens, df_stalled, df_sales
del aep_bbl_set, spec_bbl_set, conh_bbl_set, uc_bbl_set, hpd_classb_map, hpd_total_units_map, bbl_to_landlord
del bbl_to_cd, bbl_to_zip, cd_units_map, zip_units_map
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


class ResidualBlock(nn.Module):
    """Residual linear block with batch normalization, SiLU, and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class DeepNetwork(nn.Module):
    """Deep residual multilayer perceptron with batch normalization, SiLU activations, and dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout=dropout))
            prev_dim = h_dim
        self.mlp = nn.Sequential(*blocks)
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
        num_zips: int = 300,
        zip_dim: int = 16,
        num_cats: int = 4,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.num_cont = num_features - num_cats
        self.num_boros = num_boros
        self.num_cds = num_cds
        self.num_bldgclasses = num_bldgclasses
        self.num_zips = num_zips

        self.boro_embed = nn.Embedding(num_boros, boro_dim)
        self.cd_embed = nn.Embedding(num_cds, cd_dim)
        self.bldg_embed = nn.Embedding(num_bldgclasses, bldgclass_dim)
        self.zip_embed = nn.Embedding(num_zips, zip_dim)

        self.input_norm = nn.BatchNorm1d(self.num_cont)
        total_in_features = self.num_cont + boro_dim + cd_dim + bldgclass_dim + zip_dim

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
        zip_idx = torch.clamp(x_cat[:, 3], 0, self.num_zips - 1)

        boro_emb = self.boro_embed(boro_idx)
        cd_emb = self.cd_embed(cd_idx)
        bldg_emb = self.bldg_embed(bldg_idx)
        zip_emb = self.zip_embed(zip_idx)

        x_cont_norm = self.input_norm(x_cont)
        x_dense = torch.cat([x_cont_norm, boro_emb, cd_emb, bldg_emb, zip_emb], dim=1)

        cross_out = self.cross_net(x_dense)
        deep_out = self.deep_net(x_dense)
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.head(combined)
        return logits.squeeze(-1)


class PairwiseRankingLoss(nn.Module):
    """Hybrid criterion combining temperature-scaled pairwise margin ranking loss with binary cross-entropy,

    supporting sample- and severity-weighted pair ranking.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        bce_weight: float = 0.2,
        max_pairs_per_batch: int = 1024,
    ):
        super().__init__()
        self.temperature = temperature
        self.bce_weight = bce_weight
        self.max_pairs_per_batch = max_pairs_per_batch

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if weights is not None:
            bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            bce_loss = torch.sum(bce_raw * weights) / (torch.sum(weights) + 1e-6)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        pos_mask = targets == 1.0
        neg_mask = targets == 0.0

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return bce_loss

        pos_weights = weights[pos_mask] if weights is not None else None

        # Top-tier hard-negative mining: select top 50% highest-scoring negative logits
        k_neg = max(1, neg_logits.numel() // 2)
        hard_neg_logits, _ = torch.topk(neg_logits, k=k_neg)

        if pos_logits.numel() > self.max_pairs_per_batch // 2:
            perm_pos = torch.randperm(
                pos_logits.numel(), device=logits.device
            )[: self.max_pairs_per_batch // 2]
            pos_logits = pos_logits[perm_pos]
            if pos_weights is not None:
                pos_weights = pos_weights[perm_pos]

        if hard_neg_logits.numel() > self.max_pairs_per_batch:
            perm_neg = torch.randperm(
                hard_neg_logits.numel(), device=logits.device
            )[: self.max_pairs_per_batch]
            hard_neg_logits = hard_neg_logits[perm_neg]

        diff = (pos_logits.unsqueeze(1) - hard_neg_logits.unsqueeze(0)) / self.temperature
        pair_loss = F.softplus(-diff)

        if pos_weights is not None:
            w_matrix = pos_weights.unsqueeze(1)
            rank_loss = torch.sum(pair_loss * w_matrix) / (torch.sum(w_matrix) * pair_loss.shape[1] + 1e-6)
        else:
            rank_loss = torch.mean(pair_loss)

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * rank_loss


# ---------------------------------------------------------------------------
# 6. Prepare Feature Matrices & Dataloaders
# ---------------------------------------------------------------------------
skew_keywords = [
    "viol", "decay", "complaint", "erp", "dob", "rodent",
    "litig", "vacate", "open", "age", "unit", "area",
    "evict", "bedbug", "portfolio", "ecb", "safety", "overdue", "lien",
    "assess", "cure", "lead", "mold", "door", "visit", "sale",
]
skew_cols = [
    c for c in cont_cols
    if any(k in c.lower() for k in skew_keywords)
    and not any(k in c.lower() for k in ["is_", "ratio", "share", "relative", "loo"])
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
y_train_sev = train_df["forward_c_severity"].values.astype(np.float32)

X_val_nn, _ = prepare_nn_features(val_df, scaler=scaler, is_fit=False)
y_val = val_df["target"].values.astype(np.float32)
y_val_sev = val_df["forward_c_severity"].values.astype(np.float32)

X_test_nn, _ = prepare_nn_features(test_df, scaler=scaler, is_fit=False)

# LightGBM feature frames with explicit native category dtype
X_train_raw = train_df[feature_cols].copy()
X_val_raw = val_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

for col in cat_cols:
    X_train_raw[col] = X_train_raw[col].astype("category")
    X_val_raw[col] = X_val_raw[col].astype("category")
    X_test_raw[col] = X_test_raw[col].astype("category")

train_dataset = TensorDataset(
    torch.from_numpy(X_train_nn),
    torch.from_numpy(y_train),
    torch.from_numpy(sample_weights),
)
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
    num_zips=num_zips,
    num_cats=len(cat_cols),
).to(device)

criterion = PairwiseRankingLoss(temperature=0.5, bce_weight=0.2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

best_val_ap = -1.0
best_model_weights = None
best_model_path = "./working/dcn_best_model.pt"

for epoch in range(1, 11):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for bx, by, bw in train_loader:
        bx = bx.to(device)
        by = by.to(device)
        bw = bw.to(device)

        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by, bw)
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
    "num_leaves": 63,
    "max_depth": 7,
    "scale_pos_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "min_child_samples": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
try:
    lgb_model.fit(
        X_train_raw,
        y_train,
        sample_weight=sample_weights,
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
        sample_weight=sample_weights,
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
# 8b. Train Depth-Wise Gradient Boosted Decision Trees (XGBoost)
# ---------------------------------------------------------------------------
xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "aucpr",
    "enable_categorical": True,
    "early_stopping_rounds": 50,
}

xgb_model = xgb.XGBClassifier(**xgb_params)
try:
    xgb_model.fit(
        X_train_raw,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val_raw, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_params.pop("early_stopping_rounds", None)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_raw,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val_raw, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
except Exception:
    X_train_num = X_train_raw.copy()
    X_val_num = X_val_raw.copy()
    for col in cat_cols:
        X_train_num[col] = X_train_num[col].cat.codes.astype(np.int32)
        X_val_num[col] = X_val_num[col].cat.codes.astype(np.int32)
    xgb_params_clean = {k: v for k, v in xgb_params.items() if k not in ["enable_categorical", "early_stopping_rounds"]}
    xgb_model = xgb.XGBClassifier(**xgb_params_clean)
    try:
        xgb_model.fit(
            X_train_num,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val_num, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
    except Exception:
        xgb_model.fit(
            X_train_num,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val_num, y_val)],
            verbose=False,
        )

try:
    val_preds_xgb = xgb_model.predict_proba(X_val_raw)[:, 1]
    test_preds_xgb = xgb_model.predict_proba(X_test_raw)[:, 1]
except Exception:
    X_val_num = X_val_raw.copy()
    X_test_num = X_test_raw.copy()
    for col in cat_cols:
        X_val_num[col] = X_val_num[col].cat.codes.astype(np.int32)
        X_test_num[col] = X_test_num[col].cat.codes.astype(np.int32)
    val_preds_xgb = xgb_model.predict_proba(X_val_num)[:, 1]
    test_preds_xgb = xgb_model.predict_proba(X_test_num)[:, 1]

# ---------------------------------------------------------------------------
# 8c. Train Symmetric-Tree Gradient Boosted Classifier (CatBoost)
# ---------------------------------------------------------------------------
X_train_cb = X_train_raw.copy()
X_val_cb = X_val_raw.copy()
X_test_cb = X_test_raw.copy()
for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_val_cb[c] = X_val_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.04,
    depth=7,
    eval_metric="PRAUC",
    random_seed=42,
    verbose=False,
)
try:
    cb_model.fit(
        X_train_cb,
        y_train,
        sample_weight=sample_weights,
        eval_set=(X_val_cb, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )
except Exception:
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=7,
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
    )
    cb_model.fit(
        X_train_cb,
        y_train,
        sample_weight=sample_weights,
        eval_set=(X_val_cb, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )

val_preds_cb = cb_model.predict_proba(X_val_cb)[:, 1]
test_preds_cb = cb_model.predict_proba(X_test_cb)[:, 1]

# ---------------------------------------------------------------------------
# 9. Rank Normalization, Simplex Metric Optimization & Heterogeneous Ensembling
# ---------------------------------------------------------------------------
val_rank_nn = (rankdata(val_preds_nn) - 1.0) / (len(val_preds_nn) - 1.0)
val_rank_lgb = (rankdata(val_preds_lgb) - 1.0) / (len(val_preds_lgb) - 1.0)
val_rank_cb = (rankdata(val_preds_cb) - 1.0) / (len(val_preds_cb) - 1.0)
val_rank_xgb = (rankdata(val_preds_xgb) - 1.0) / (len(val_preds_xgb) - 1.0)

test_rank_nn = (rankdata(test_preds_nn) - 1.0) / (len(test_preds_nn) - 1.0)
test_rank_lgb = (rankdata(test_preds_lgb) - 1.0) / (len(test_preds_lgb) - 1.0)
test_rank_cb = (rankdata(test_preds_cb) - 1.0) / (len(test_preds_cb) - 1.0)
test_rank_xgb = (rankdata(test_preds_xgb) - 1.0) / (len(test_preds_xgb) - 1.0)

val_ranks = np.vstack([val_rank_lgb, val_rank_cb, val_rank_xgb, val_rank_nn])
test_ranks = np.vstack([test_rank_lgb, test_rank_cb, test_rank_xgb, test_rank_nn])


def evaluate_weights(w):
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s == 0:
        return 0.0
    w = w / s
    pred = np.dot(w, val_ranks)
    return average_precision_score(y_val, pred)


best_score = -1.0
best_weights = np.array([0.25, 0.25, 0.25, 0.25])

# Evaluate single models
for i in range(4):
    w = np.zeros(4)
    w[i] = 1.0
    sc = evaluate_weights(w)
    if sc > best_score:
        best_score = sc
        best_weights = w

# Evaluate candidate simplex points
candidates = [
    np.array([0.25, 0.25, 0.25, 0.25]),
    np.array([0.30, 0.30, 0.20, 0.20]),
    np.array([0.20, 0.30, 0.30, 0.20]),
    np.array([0.20, 0.35, 0.25, 0.20]),
    np.array([0.25, 0.35, 0.20, 0.20]),
    np.array([0.15, 0.35, 0.30, 0.20]),
    np.array([0.20, 0.25, 0.30, 0.25]),
    np.array([0.15, 0.30, 0.30, 0.25]),
    np.array([0.25, 0.25, 0.30, 0.20]),
]

np.random.seed(42)
for alpha in [1.0, 2.0, 0.5]:
    dir_samples = np.random.dirichlet([alpha] * 4, size=150)
    candidates.extend(list(dir_samples))

for c in candidates:
    sc = evaluate_weights(c)
    if sc > best_score:
        best_score = sc
        best_weights = c / c.sum()

# Multi-scale coordinate search refinement
current_w = best_weights.copy()
for delta in [0.08, 0.04, 0.02, 0.01, 0.005]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for d in [-delta, delta]:
                cand_w = current_w.copy()
                cand_w[i] += d
                cand_w = np.clip(cand_w, 0.0, None)
                if cand_w.sum() > 0:
                    cand_w /= cand_w.sum()
                    sc = evaluate_weights(cand_w)
                    if sc > best_score + 1e-6:
                        best_score = sc
                        best_weights = cand_w
                        current_w = cand_w
                        improved = True

best_weights = best_weights / best_weights.sum()
score = best_score
final_test_score = np.dot(best_weights, test_ranks)
print(
    f"Optimal Ensemble Weights: lgb={best_weights[0]:.3f}, cb={best_weights[1]:.3f}, xgb={best_weights[2]:.3f}, nn={best_weights[3]:.3f}"
)

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
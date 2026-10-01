import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

# ---------------------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)

TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
STORAGE_OPTIONS = (
    {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else {"token": "anon"}
)
GCS_BASE = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"
LAKE_BASE = f"{GCS_BASE}/lake/full"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Utility Functions
# ---------------------------------------------------------------------------
def clean_bbl_series(
    df,
    bbl_col="bbl",
    boro_col="boroid",
    block_col="block",
    lot_col="lot",
    boro_name_col="boro",
):
    """Standardize BBL representation to a 10-digit zero-padded string

    according to the official competition specification.
    """
    bbl_str = pd.Series("", index=df.index, dtype=str)
    is_valid_10 = pd.Series(False, index=df.index)

    if bbl_col in df.columns:
        str_bbl = (
            df[bbl_col].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        )
        is_valid_10 = (str_bbl.str.len() == 10) & (str_bbl.str.isdigit())
        bbl_str = str_bbl.where(is_valid_10, "")

    if (~is_valid_10).any():
        boro_id_str = None
        if boro_col in df.columns and df[boro_col].notna().any():
            boro_id_str = (
                pd.to_numeric(df[boro_col], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        elif boro_name_col in df.columns and df[boro_name_col].notna().any():
            boro_map = {
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "MN": "1",
                "MANHATTAN": "1",
                "BX": "2",
                "BRONX": "2",
                "BK": "3",
                "BROOKLYN": "3",
                "QN": "4",
                "QUEENS": "4",
                "SI": "5",
                "STATEN ISLAND": "5",
            }
            boro_id_str = (
                df[boro_name_col]
                .astype(str)
                .str.upper()
                .str.strip()
                .map(boro_map)
                .fillna("0")
            )

        if (
            boro_id_str is not None
            and block_col in df.columns
            and lot_col in df.columns
        ):
            block_str = (
                pd.to_numeric(df[block_col], errors="coerce")
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
            constructed = boro_id_str + block_str + lot_str
            bbl_str = bbl_str.where(is_valid_10, constructed)

    return bbl_str.astype(str).str.zfill(10)


def ensure_tz_naive(series):
    """Safely convert timestamps to tz-naive for consistent date comparisons."""
    dt_s = pd.to_datetime(series, errors="coerce")
    if dt_s.dt.tz is not None:
        dt_s = dt_s.dt.tz_convert(None)
    return dt_s


def extract_date_col(df):
    """Identify and parse available datetime columns into tz-naive format."""
    preferred = [
        "executeddate",
        "executed_date",
        "chargedate",
        "charge_date",
        "transdate",
        "trans_date",
        "transacteddate",
        "invoicedate",
        "invoice_date",
        "issuedate",
        "issue_date",
        "violation_date",
        "violationdate",
        "inspectiondate",
        "inspection_date",
        "createddate",
        "date",
    ]
    for p in preferred:
        if p in df.columns:
            dt = ensure_tz_naive(df[p])
            if dt.notna().any():
                return dt
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            dt = ensure_tz_naive(df[c])
            if dt.notna().any():
                return dt
    return None


# ---------------------------------------------------------------------------
# 2. Data Loading & Lake Preparation
# ---------------------------------------------------------------------------
print("Loading Test Entities...")
test_entities_path = f"{GCS_BASE}/test_entities.parquet"
df_test = pd.read_parquet(test_entities_path, storage_options=STORAGE_OPTIONS)
df_test["bbl"] = df_test["bbl"].astype(str).str.strip().str.zfill(10)

print("Loading PLUTO Data...")
pluto_cols = [
    "bbl",
    "borough",
    "block",
    "lot",
    "unitsres",
    "unitstotal",
    "yearbuilt",
    "bldgarea",
    "resarea",
    "numfloors",
    "lotarea",
    "bldgclass",
    "landuse",
    "latitude",
    "longitude",
    "zipcode",
]

try:
    pluto_raw = pd.read_parquet(
        f"{LAKE_BASE}/pluto/",
        columns=pluto_cols,
        storage_options=STORAGE_OPTIONS,
    )
except Exception:
    pluto_raw = pd.read_parquet(f"{LAKE_BASE}/pluto/", storage_options=STORAGE_OPTIONS)
    available_cols = [c for c in pluto_cols if c in pluto_raw.columns]
    pluto_raw = pluto_raw[available_cols]

pluto_raw["bbl"] = clean_bbl_series(pluto_raw)
pluto_df = pluto_raw.drop_duplicates(subset=["bbl"], keep="last").copy()
del pluto_raw
gc.collect()

# Spatial lookup dictionaries for empirical priors and smoothing
pluto_df["zip_clean"] = (
    pluto_df["zipcode"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)
bbl_to_zip = pluto_df.set_index("bbl")["zip_clean"].to_dict()
block_bbl_count = pluto_df["bbl"].str[:6].value_counts().to_dict()
zip_bbl_count = pluto_df["zip_clean"].value_counts().to_dict()

# Define multiple dwelling lots (unitsres >= 3)
pluto_res = pluto_df[pluto_df["unitsres"].fillna(0) >= 3].copy()
train_bbls = pluto_res["bbl"].unique()
val_bbls = pluto_res["bbl"].unique()
test_bbls = df_test["bbl"].values

print("Loading HPD Violations Data...")
violation_cols = [
    "bbl",
    "boroid",
    "boro",
    "block",
    "lot",
    "class",
    "inspectiondate",
    "violationstatus",
    "currentstatusdate",
]
try:
    hpd_viol = pd.read_parquet(
        f"{LAKE_BASE}/hpd_violations/",
        columns=violation_cols,
        storage_options=STORAGE_OPTIONS,
    )
except Exception:
    hpd_viol = pd.read_parquet(
        f"{LAKE_BASE}/hpd_violations/",
        storage_options=STORAGE_OPTIONS,
    )
hpd_viol["bbl"] = clean_bbl_series(hpd_viol)
hpd_viol["inspectiondate"] = ensure_tz_naive(hpd_viol["inspectiondate"])
if "currentstatusdate" in hpd_viol.columns:
    hpd_viol["currentstatusdate"] = ensure_tz_naive(hpd_viol["currentstatusdate"])
else:
    hpd_viol["currentstatusdate"] = pd.NaT
hpd_viol = hpd_viol[hpd_viol["inspectiondate"].notna()].copy()
hpd_viol["class"] = hpd_viol["class"].astype(str).str.upper().str.strip()
hpd_viol["violationstatus"] = (
    hpd_viol["violationstatus"].astype(str).str.strip().str.capitalize()
)

print("Loading HPD Complaints Data...")
try:
    try:
        hpd_comp_raw = pd.read_parquet(
            f"{LAKE_BASE}/hpd_complaints/",
            columns=["bbl", "receiveddate"],
            storage_options=STORAGE_OPTIONS,
        )
    except Exception:
        try:
            hpd_comp_raw = pd.read_parquet(
                f"{LAKE_BASE}/hpd_complaints/",
                columns=["bbl", "boroid", "block", "lot", "receiveddate"],
                storage_options=STORAGE_OPTIONS,
            )
        except Exception:
            hpd_comp_raw = pd.read_parquet(
                f"{LAKE_BASE}/hpd_complaints/",
                storage_options=STORAGE_OPTIONS,
            )

    hpd_comp_raw["bbl"] = clean_bbl_series(hpd_comp_raw)
    date_cols = [
        c
        for c in ["receiveddate", "dateentered", "complaintdate", "statusdate"]
        if c in hpd_comp_raw.columns
    ]
    if not date_cols:
        date_cols = [c for c in hpd_comp_raw.columns if "date" in c.lower()]
    comp_date_col = date_cols[0] if date_cols else None
    if comp_date_col:
        hpd_comp_raw["complaint_date"] = ensure_tz_naive(hpd_comp_raw[comp_date_col])
        hpd_comp = (
            hpd_comp_raw[hpd_comp_raw["complaint_date"].notna()][
                ["bbl", "complaint_date"]
            ].copy()
        )
    else:
        hpd_comp = pd.DataFrame(columns=["bbl", "complaint_date"])
    del hpd_comp_raw
    gc.collect()
except Exception as e:
    print(f"Warning: could not load hpd_complaints: {e}")
    hpd_comp = pd.DataFrame(columns=["bbl", "complaint_date"])

print("Loading Auxiliary Distress Datasets...")
# AEP Buildings
try:
    aep_df = pd.read_parquet(
        f"{LAKE_BASE}/hpd_aep_buildings/",
        columns=["bbl"],
        storage_options=STORAGE_OPTIONS,
    )
    aep_bbls = set(clean_bbl_series(aep_df).unique())
except Exception:
    aep_bbls = set()

# Vacate Orders
try:
    vacate_df = pd.read_parquet(
        f"{LAKE_BASE}/hpd_vacate_orders/",
        columns=["bbl", "vacate_effective_date"],
        storage_options=STORAGE_OPTIONS,
    )
    vacate_df["bbl"] = clean_bbl_series(vacate_df)
    vacate_df["vacate_effective_date"] = ensure_tz_naive(
        vacate_df["vacate_effective_date"]
    )
except Exception:
    vacate_df = pd.DataFrame(columns=["bbl", "vacate_effective_date"])

# Litigations
try:
    lit_df = pd.read_parquet(
        f"{LAKE_BASE}/hpd_litigations/",
        columns=["bbl", "caseopendate"],
        storage_options=STORAGE_OPTIONS,
    )
    lit_df["bbl"] = clean_bbl_series(lit_df)
    lit_df["caseopendate"] = ensure_tz_naive(lit_df["caseopendate"])
except Exception:
    lit_df = pd.DataFrame(columns=["bbl", "caseopendate"])

# Municipal Distress Datasets: OMO Charges, HWO Charges, DOB Violations
print("Loading Emergency Repair and DOB Distress Datasets...")
try:
    omo_df = pd.read_parquet(
        f"{LAKE_BASE}/hpd_omo_charges/",
        storage_options=STORAGE_OPTIONS,
    )
    omo_df.columns = [str(c).lower() for c in omo_df.columns]
    omo_df["bbl"] = clean_bbl_series(omo_df)
    dt_omo = extract_date_col(omo_df)
    if dt_omo is not None:
        omo_df["omo_date"] = dt_omo
        omo_df = omo_df[omo_df["omo_date"].notna()][["bbl", "omo_date"]].copy()
    else:
        omo_df = pd.DataFrame(columns=["bbl", "omo_date"])
except Exception as e:
    print(f"Warning: could not load hpd_omo_charges: {e}")
    omo_df = pd.DataFrame(columns=["bbl", "omo_date"])

try:
    hwo_df = pd.read_parquet(
        f"{LAKE_BASE}/hpd_hwo_charges/",
        storage_options=STORAGE_OPTIONS,
    )
    hwo_df.columns = [str(c).lower() for c in hwo_df.columns]
    hwo_df["bbl"] = clean_bbl_series(hwo_df)
    dt_hwo = extract_date_col(hwo_df)
    if dt_hwo is not None:
        hwo_df["hwo_date"] = dt_hwo
        hwo_df = hwo_df[hwo_df["hwo_date"].notna()][["bbl", "hwo_date"]].copy()
    else:
        hwo_df = pd.DataFrame(columns=["bbl", "hwo_date"])
except Exception as e:
    print(f"Warning: could not load hpd_hwo_charges: {e}")
    hwo_df = pd.DataFrame(columns=["bbl", "hwo_date"])

try:
    dob_df = pd.read_parquet(
        f"{LAKE_BASE}/dob_violations/",
        storage_options=STORAGE_OPTIONS,
    )
    dob_df.columns = [str(c).lower() for c in dob_df.columns]
    dob_df["bbl"] = clean_bbl_series(dob_df)
    dt_dob = extract_date_col(dob_df)
    if dt_dob is not None:
        dob_df["dob_date"] = dt_dob
        dob_df = dob_df[dob_df["dob_date"].notna()][["bbl", "dob_date"]].copy()
    else:
        dob_df = pd.DataFrame(columns=["bbl", "dob_date"])
except Exception as e:
    print(f"Warning: could not load dob_violations: {e}")
    dob_df = pd.DataFrame(columns=["bbl", "dob_date"])

print("Loading DOB ECB Violations Dataset...")
try:
    dob_ecb_df = pd.read_parquet(
        f"{LAKE_BASE}/dob_ecb_violations/",
        storage_options=STORAGE_OPTIONS,
    )
    dob_ecb_df.columns = [str(c).lower() for c in dob_ecb_df.columns]
    dob_ecb_df["bbl"] = clean_bbl_series(dob_ecb_df)
    dt_dob_ecb = extract_date_col(dob_ecb_df)
    if dt_dob_ecb is not None:
        dob_ecb_df["dob_ecb_date"] = dt_dob_ecb
        dob_ecb_df = dob_ecb_df[dob_ecb_df["dob_ecb_date"].notna()][
            ["bbl", "dob_ecb_date"]
        ].copy()
    else:
        dob_ecb_df = pd.DataFrame(columns=["bbl", "dob_ecb_date"])
except Exception as e:
    print(f"Warning: could not load dob_ecb_violations: {e}")
    dob_ecb_df = pd.DataFrame(columns=["bbl", "dob_ecb_date"])

print("Loading Evictions Dataset...")
try:
    try:
        evict_df = pd.read_parquet(
            f"{LAKE_BASE}/evictions/",
            storage_options=STORAGE_OPTIONS,
        )
    except Exception:
        evict_df = pd.read_parquet(
            f"{LAKE_BASE}/doi_evictions/",
            storage_options=STORAGE_OPTIONS,
        )
    evict_df.columns = [str(c).lower() for c in evict_df.columns]
    evict_df["bbl"] = clean_bbl_series(evict_df)
    dt_evict = extract_date_col(evict_df)
    if dt_evict is not None:
        evict_df["evict_date"] = dt_evict
        evict_df = evict_df[evict_df["evict_date"].notna()][["bbl", "evict_date"]].copy()
    else:
        evict_df = pd.DataFrame(columns=["bbl", "evict_date"])
except Exception as e:
    print(f"Warning: could not load evictions: {e}")
    evict_df = pd.DataFrame(columns=["bbl", "evict_date"])


# ---------------------------------------------------------------------------
# 3. Label Definition & Feature Extraction Pipeline
# ---------------------------------------------------------------------------
def compute_labels(bbl_list, cutoff_date, violations_df):
    """Compute binary label indicating >=1 Class C violation in [cutoff, cutoff + 12m)."""
    end_date = cutoff_date + pd.DateOffset(months=12)
    c_viols = violations_df[
        (violations_df["class"] == "C")
        & (violations_df["inspectiondate"] >= cutoff_date)
        & (violations_df["inspectiondate"] < end_date)
    ]
    pos_bbls = set(c_viols["bbl"].unique())
    labels = pd.Series(
        [1 if b in pos_bbls else 0 for b in bbl_list],
        index=bbl_list,
        name="target",
    )
    return labels


def extract_features(bbl_list, cutoff_date):
    """Extract point-in-time features strictly prior to cutoff_date."""
    features = pd.DataFrame({"bbl": bbl_list})

    # 1. Merge Static PLUTO Building Features
    features = features.merge(pluto_df, on="bbl", how="left")

    for col in [
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "bldgarea",
        "resarea",
        "lotarea",
        "numfloors",
        "borough",
        "latitude",
        "longitude",
        "bldgclass",
    ]:
        if col not in features.columns:
            features[col] = 0

    unitsres = (
        pd.to_numeric(features["unitsres"], errors="coerce").fillna(0).clip(lower=0)
    )
    unitstotal = (
        pd.to_numeric(features["unitstotal"], errors="coerce").fillna(0).clip(lower=0)
    )
    yearbuilt = pd.to_numeric(features["yearbuilt"], errors="coerce").fillna(0).values
    bldgarea = (
        pd.to_numeric(features["bldgarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    resarea = (
        pd.to_numeric(features["resarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    lotarea = (
        pd.to_numeric(features["lotarea"], errors="coerce").fillna(0).clip(lower=0)
    )
    numfloors = (
        pd.to_numeric(features["numfloors"], errors="coerce").fillna(0).clip(lower=0)
    )

    features["feat_log_unitsres"] = np.log1p(unitsres).astype(np.float32)
    features["feat_log_unitstotal"] = np.log1p(unitstotal).astype(np.float32)
    features["feat_res_share"] = (unitsres / (unitstotal + 1e-4)).astype(np.float32)
    features["feat_log_bldgarea"] = np.log1p(bldgarea).astype(np.float32)
    features["feat_log_resarea"] = np.log1p(resarea).astype(np.float32)
    features["feat_log_lotarea"] = np.log1p(lotarea).astype(np.float32)
    features["feat_numfloors"] = numfloors.astype(np.float32)
    features["feat_area_per_unit"] = (bldgarea / (unitsres + 1.0)).astype(np.float32)
    features["feat_floors_per_unit"] = (numfloors / (unitsres + 1.0)).astype(np.float32)

    # Building Vintage
    valid_year = (yearbuilt > 1800) & (yearbuilt <= cutoff_date.year)
    building_age = np.where(valid_year, cutoff_date.year - yearbuilt, -1).astype(
        np.float32
    )
    features["feat_building_age"] = building_age
    features["feat_is_prewar"] = ((yearbuilt > 1800) & (yearbuilt < 1940)).astype(
        np.float32
    )
    features["feat_is_postwar"] = ((yearbuilt >= 1940) & (yearbuilt < 1974)).astype(
        np.float32
    )

    # Geographic identifiers
    features["feat_borough"] = (
        pd.to_numeric(features["borough"], errors="coerce").fillna(0).astype(np.float32)
    )
    features["feat_latitude"] = (
        pd.to_numeric(features["latitude"], errors="coerce")
        .fillna(40.7)
        .astype(np.float32)
    )
    features["feat_longitude"] = (
        pd.to_numeric(features["longitude"], errors="coerce")
        .fillna(-73.9)
        .astype(np.float32)
    )

    # Building Class prefix (e.g. C=Walkup, D=Elevator)
    bldgclass_prefix = features["bldgclass"].fillna("").astype(str).str[:1].str.upper()
    class_map = {"C": 1, "D": 2, "A": 3, "B": 4, "S": 5, "O": 6, "R": 7}
    features["feat_bldgclass_code"] = (
        bldgclass_prefix.map(class_map).fillna(0).astype(np.float32)
    )

    # 2. Historical Violations (strictly prior to cutoff_date)
    prior_viols = hpd_viol[hpd_viol["inspectiondate"] < cutoff_date]

    w30d = cutoff_date - pd.Timedelta(days=30)
    w60d = cutoff_date - pd.Timedelta(days=60)
    w90d = cutoff_date - pd.Timedelta(days=90)
    w180d = cutoff_date - pd.Timedelta(days=180)
    w1y = cutoff_date - pd.Timedelta(days=365)
    w2y = cutoff_date - pd.Timedelta(days=730)
    w3y = cutoff_date - pd.Timedelta(days=1095)
    w5y = cutoff_date - pd.Timedelta(days=1825)

    v_30d = prior_viols[prior_viols["inspectiondate"] >= w30d]
    v_60d = prior_viols[prior_viols["inspectiondate"] >= w60d]
    v_90d = prior_viols[prior_viols["inspectiondate"] >= w90d]
    v_180d = prior_viols[prior_viols["inspectiondate"] >= w180d]
    v_1y = prior_viols[prior_viols["inspectiondate"] >= w1y]
    v_2y = prior_viols[prior_viols["inspectiondate"] >= w2y]
    v_3y = prior_viols[prior_viols["inspectiondate"] >= w3y]
    v_5y = prior_viols[prior_viols["inspectiondate"] >= w5y]

    def count_by_bbl(df_sub, c_val=None):
        if c_val is not None:
            df_sub = df_sub[df_sub["class"] == c_val]
        return df_sub.groupby("bbl").size()

    c_30d = count_by_bbl(v_30d, "C")
    c_60d = count_by_bbl(v_60d, "C")
    c_90d = count_by_bbl(v_90d, "C")
    c_180d = count_by_bbl(v_180d, "C")
    c_1y = count_by_bbl(v_1y, "C")
    c_2y = count_by_bbl(v_2y, "C")
    c_3y = count_by_bbl(v_3y, "C")
    c_5y = count_by_bbl(v_5y, "C")
    c_all = count_by_bbl(prior_viols, "C")

    b_30d = count_by_bbl(v_30d, "B")
    b_90d = count_by_bbl(v_90d, "B")
    b_1y = count_by_bbl(v_1y, "B")
    b_3y = count_by_bbl(v_3y, "B")
    a_1y = count_by_bbl(v_1y, "A")

    tot_30d = count_by_bbl(v_30d)
    tot_60d = count_by_bbl(v_60d)
    tot_180d = count_by_bbl(v_180d)
    tot_1y = count_by_bbl(v_1y)
    tot_2y = count_by_bbl(v_2y)
    tot_3y = count_by_bbl(v_3y)
    tot_all = count_by_bbl(prior_viols)

    insp_visits_1y = v_1y.groupby("bbl")["inspectiondate"].nunique()

    c_viols_only = prior_viols[prior_viols["class"] == "C"]
    max_c_date = c_viols_only.groupby("bbl")["inspectiondate"].max()
    max_any_date = prior_viols.groupby("bbl")["inspectiondate"].max()

    bbl_s = features["bbl"]
    features["feat_viol_c_30d"] = bbl_s.map(c_30d).fillna(0).astype(np.float32).values
    features["feat_viol_c_60d"] = bbl_s.map(c_60d).fillna(0).astype(np.float32).values
    features["feat_viol_c_180d"] = bbl_s.map(c_180d).fillna(0).astype(np.float32).values
    features["feat_viol_c_1y"] = bbl_s.map(c_1y).fillna(0).astype(np.float32).values
    features["feat_viol_c_2y"] = bbl_s.map(c_2y).fillna(0).astype(np.float32).values
    features["feat_viol_c_3y"] = bbl_s.map(c_3y).fillna(0).astype(np.float32).values
    features["feat_viol_c_5y"] = bbl_s.map(c_5y).fillna(0).astype(np.float32).values
    features["feat_viol_c_all"] = bbl_s.map(c_all).fillna(0).astype(np.float32).values

    features["feat_viol_c_90d"] = bbl_s.map(c_90d).fillna(0).astype(np.float32).values
    features["feat_viol_b_30d"] = bbl_s.map(b_30d).fillna(0).astype(np.float32).values
    features["feat_viol_b_90d"] = bbl_s.map(b_90d).fillna(0).astype(np.float32).values
    features["feat_viol_b_1y"] = bbl_s.map(b_1y).fillna(0).astype(np.float32).values
    features["feat_viol_b_3y"] = bbl_s.map(b_3y).fillna(0).astype(np.float32).values
    features["feat_viol_a_1y"] = bbl_s.map(a_1y).fillna(0).astype(np.float32).values

    features["feat_viol_tot_30d"] = bbl_s.map(tot_30d).fillna(0).astype(np.float32).values
    features["feat_viol_tot_60d"] = bbl_s.map(tot_60d).fillna(0).astype(np.float32).values
    features["feat_viol_tot_180d"] = bbl_s.map(tot_180d).fillna(0).astype(np.float32).values
    features["feat_viol_tot_1y"] = bbl_s.map(tot_1y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_2y"] = bbl_s.map(tot_2y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_3y"] = bbl_s.map(tot_3y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_all"] = (
        bbl_s.map(tot_all).fillna(0).astype(np.float32).values
    )
    features["feat_insp_visits_1y"] = (
        bbl_s.map(insp_visits_1y).fillna(0).astype(np.float32).values
    )

    # Tax block and ZIP Class C spatial density and empirical rate smoothing
    block_s = bbl_s.str[:6]
    v_1y_c = v_1y[v_1y["class"] == "C"]
    block_c_1y = v_1y_c.groupby(v_1y_c["bbl"].str[:6]).size()
    prior_c = prior_viols[prior_viols["class"] == "C"]
    block_c_all = prior_c.groupby(prior_c["bbl"].str[:6]).size()
    features["feat_block_viol_c_1y"] = (
        block_s.map(block_c_1y).fillna(0).astype(np.float32).values
    )
    features["feat_block_viol_c_all"] = (
        block_s.map(block_c_all).fillna(0).astype(np.float32).values
    )
    block_total_lots = block_s.map(block_bbl_count).fillna(1.0).astype(np.float32)
    features["feat_block_viol_c_rate_1y"] = (
        features["feat_block_viol_c_1y"] / (block_total_lots + 1.0)
    ).astype(np.float32)
    features["feat_block_viol_c_rate_all"] = (
        features["feat_block_viol_c_all"] / (block_total_lots + 1.0)
    ).astype(np.float32)

    zip_s = bbl_s.map(bbl_to_zip).fillna("")
    v_1y_c_zips = v_1y_c["bbl"].map(bbl_to_zip).fillna("")
    zip_c_1y = (
        v_1y_c[v_1y_c_zips != ""].groupby(v_1y_c_zips[v_1y_c_zips != ""]).size()
    )
    prior_c_zips = prior_c["bbl"].map(bbl_to_zip).fillna("")
    zip_c_all = (
        prior_c[prior_c_zips != ""].groupby(prior_c_zips[prior_c_zips != ""]).size()
    )

    feat_zip_1y = zip_s.map(zip_c_1y).fillna(0).astype(np.float32)
    feat_zip_1y[zip_s == ""] = 0.0
    features["feat_zip_viol_c_1y"] = feat_zip_1y.values

    feat_zip_all = zip_s.map(zip_c_all).fillna(0).astype(np.float32)
    feat_zip_all[zip_s == ""] = 0.0
    features["feat_zip_viol_c_all"] = feat_zip_all.values

    zip_total_lots = zip_s.map(zip_bbl_count).fillna(10.0).astype(np.float32)
    features["feat_zip_viol_c_rate_1y"] = (
        features["feat_zip_viol_c_1y"] / (zip_total_lots + 1.0)
    ).astype(np.float32)
    features["feat_zip_viol_c_rate_all"] = (
        features["feat_zip_viol_c_all"] / (zip_total_lots + 1.0)
    ).astype(np.float32)

    # Derived Ratios and Acceleration Metrics
    features["feat_viol_c_per_unit_1y"] = (
        features["feat_viol_c_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_viol_c_per_unit_3y"] = (
        features["feat_viol_c_3y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_viol_tot_per_unit_1y"] = (
        features["feat_viol_tot_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_viol_tot_per_unit_3y"] = (
        features["feat_viol_tot_3y"] / (unitsres + 1.0)
    ).astype(np.float32)

    features["feat_ratio_c_1y"] = (
        features["feat_viol_c_1y"] / (features["feat_viol_tot_1y"] + 1.0)
    ).astype(np.float32)
    features["feat_ratio_c_all"] = (
        features["feat_viol_c_all"] / (features["feat_viol_tot_all"] + 1.0)
    ).astype(np.float32)
    features["feat_accel_c"] = (
        features["feat_viol_c_1y"]
        - (features["feat_viol_c_2y"] - features["feat_viol_c_1y"])
    ).astype(np.float32)
    features["feat_accel_tot"] = (
        features["feat_viol_tot_1y"]
        - (features["feat_viol_tot_2y"] - features["feat_viol_tot_1y"])
    ).astype(np.float32)

    # Multi-scale hazard velocity metrics (30d and 90d Class C/B relative to annualized baseline)
    annual_c_30d = features["feat_viol_c_1y"] / 12.0
    annual_c_90d = features["feat_viol_c_1y"] / 4.0
    annual_b_30d = features["feat_viol_b_1y"] / 12.0
    annual_b_90d = features["feat_viol_b_1y"] / 4.0

    features["feat_viol_c_vel_30d"] = (
        features["feat_viol_c_30d"] - annual_c_30d
    ).astype(np.float32)
    features["feat_viol_c_vel_ratio_30d"] = (
        features["feat_viol_c_30d"] / (annual_c_30d + 1.0)
    ).astype(np.float32)
    features["feat_viol_c_vel_90d"] = (
        features["feat_viol_c_90d"] - annual_c_90d
    ).astype(np.float32)
    features["feat_viol_c_vel_ratio_90d"] = (
        features["feat_viol_c_90d"] / (annual_c_90d + 1.0)
    ).astype(np.float32)

    features["feat_viol_b_vel_30d"] = (
        features["feat_viol_b_30d"] - annual_b_30d
    ).astype(np.float32)
    features["feat_viol_b_vel_ratio_30d"] = (
        features["feat_viol_b_30d"] / (annual_b_30d + 1.0)
    ).astype(np.float32)
    features["feat_viol_b_vel_90d"] = (
        features["feat_viol_b_90d"] - annual_b_90d
    ).astype(np.float32)
    features["feat_viol_b_vel_ratio_90d"] = (
        features["feat_viol_b_90d"] / (annual_b_90d + 1.0)
    ).astype(np.float32)

    # Violation Recency
    last_c_days = (cutoff_date - bbl_s.map(max_c_date)).dt.days.fillna(3650)
    last_any_days = (cutoff_date - bbl_s.map(max_any_date)).dt.days.fillna(3650)
    features["feat_days_since_last_c"] = last_c_days.astype(np.float32).values
    features["feat_days_since_last_any"] = last_any_days.astype(np.float32).values
    features["feat_has_prior_c"] = (features["feat_viol_c_all"] > 0).astype(np.float32)

    # Active Violation Backlog (reconstructed point-in-time at cutoff_date)
    is_active = (prior_viols["violationstatus"] == "Open") | (
        prior_viols["currentstatusdate"].notna()
        & (prior_viols["currentstatusdate"] >= cutoff_date)
    )
    open_viols = prior_viols[is_active]
    open_c = open_viols[open_viols["class"] == "C"].groupby("bbl").size()
    open_b = open_viols[open_viols["class"] == "B"].groupby("bbl").size()
    open_tot = open_viols.groupby("bbl").size()

    features["feat_viol_open_c"] = bbl_s.map(open_c).fillna(0).astype(np.float32).values
    features["feat_viol_open_b"] = bbl_s.map(open_b).fillna(0).astype(np.float32).values
    features["feat_viol_open_tot"] = bbl_s.map(open_tot).fillna(0).astype(np.float32).values
    features["feat_viol_open_c_per_unit"] = (
        features["feat_viol_open_c"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_viol_open_b_per_unit"] = (
        features["feat_viol_open_b"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_viol_open_tot_per_unit"] = (
        features["feat_viol_open_tot"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_ratio_open_c"] = (
        features["feat_viol_open_c"] / (features["feat_viol_c_all"] + 1.0)
    ).astype(np.float32)
    features["feat_ratio_open_tot"] = (
        features["feat_viol_open_tot"] / (features["feat_viol_tot_all"] + 1.0)
    ).astype(np.float32)

    # Tenant Complaints (strictly prior to cutoff_date)
    prior_comp = hpd_comp[hpd_comp["complaint_date"] < cutoff_date]
    w30d = cutoff_date - pd.Timedelta(days=30)
    w90d = cutoff_date - pd.Timedelta(days=90)
    w365d = cutoff_date - pd.Timedelta(days=365)

    comp_30d = prior_comp[prior_comp["complaint_date"] >= w30d].groupby("bbl").size()
    comp_90d = prior_comp[prior_comp["complaint_date"] >= w90d].groupby("bbl").size()
    comp_365d = prior_comp[prior_comp["complaint_date"] >= w365d].groupby("bbl").size()

    features["feat_complaints_30d"] = bbl_s.map(comp_30d).fillna(0).astype(np.float32).values
    features["feat_complaints_90d"] = bbl_s.map(comp_90d).fillna(0).astype(np.float32).values
    features["feat_complaints_365d"] = bbl_s.map(comp_365d).fillna(0).astype(np.float32).values
    features["feat_complaints_per_unit_365d"] = (
        features["feat_complaints_365d"] / (unitsres + 1.0)
    ).astype(np.float32)
    annualized_comp_30d = features["feat_complaints_365d"] / 12.0
    features["feat_complaint_velocity_30d"] = (
        features["feat_complaints_30d"] - annualized_comp_30d
    ).astype(np.float32)
    features["feat_complaint_velocity_ratio_30d"] = (
        features["feat_complaints_30d"] / (annualized_comp_30d + 1.0)
    ).astype(np.float32)

    max_comp_date = prior_comp.groupby("bbl")["complaint_date"].max()
    last_comp_days = (cutoff_date - bbl_s.map(max_comp_date)).dt.days.fillna(3650)
    features["feat_days_since_last_complaint"] = last_comp_days.astype(np.float32).values

    # 3. High-Risk Auxiliary Programs Features
    features["feat_in_aep"] = bbl_s.isin(aep_bbls).astype(np.float32).values

    prior_vacates = vacate_df[vacate_df["vacate_effective_date"] < cutoff_date]
    vacate_counts = prior_vacates.groupby("bbl").size()
    features["feat_vacate_orders_count"] = (
        bbl_s.map(vacate_counts).fillna(0).astype(np.float32).values
    )

    prior_lits = lit_df[lit_df["caseopendate"] < cutoff_date]
    lit_counts_1y = prior_lits[prior_lits["caseopendate"] >= w1y].groupby("bbl").size()
    lit_counts_all = prior_lits.groupby("bbl").size()
    features["feat_litigations_1y"] = (
        bbl_s.map(lit_counts_1y).fillna(0).astype(np.float32).values
    )
    features["feat_litigations_all"] = (
        bbl_s.map(lit_counts_all).fillna(0).astype(np.float32).values
    )

    # Evictions (strictly prior to cutoff_date)
    prior_evict = evict_df[evict_df["evict_date"] < cutoff_date]
    evict_1y = prior_evict[prior_evict["evict_date"] >= w1y].groupby("bbl").size()
    evict_all = prior_evict.groupby("bbl").size()
    features["feat_evictions_1y"] = (
        bbl_s.map(evict_1y).fillna(0).astype(np.float32).values
    )
    features["feat_evictions_all"] = (
        bbl_s.map(evict_all).fillna(0).astype(np.float32).values
    )
    features["feat_evictions_per_unit_1y"] = (
        features["feat_evictions_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_evictions_per_unit_all"] = (
        features["feat_evictions_all"] / (unitsres + 1.0)
    ).astype(np.float32)

    # 4. Cross-Agency Emergency Distress Features (strictly prior to cutoff_date)
    prior_omo = omo_df[omo_df["omo_date"] < cutoff_date]
    omo_1y = prior_omo[prior_omo["omo_date"] >= w1y].groupby("bbl").size()
    omo_all = prior_omo.groupby("bbl").size()
    features["feat_omo_1y"] = bbl_s.map(omo_1y).fillna(0).astype(np.float32).values
    features["feat_omo_all"] = bbl_s.map(omo_all).fillna(0).astype(np.float32).values
    features["feat_omo_per_unit_1y"] = (
        features["feat_omo_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_omo_per_unit_all"] = (
        features["feat_omo_all"] / (unitsres + 1.0)
    ).astype(np.float32)

    prior_hwo = hwo_df[hwo_df["hwo_date"] < cutoff_date]
    hwo_1y = prior_hwo[prior_hwo["hwo_date"] >= w1y].groupby("bbl").size()
    hwo_all = prior_hwo.groupby("bbl").size()
    features["feat_hwo_1y"] = bbl_s.map(hwo_1y).fillna(0).astype(np.float32).values
    features["feat_hwo_all"] = bbl_s.map(hwo_all).fillna(0).astype(np.float32).values
    features["feat_hwo_per_unit_1y"] = (
        features["feat_hwo_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_hwo_per_unit_all"] = (
        features["feat_hwo_all"] / (unitsres + 1.0)
    ).astype(np.float32)

    prior_dob = dob_df[dob_df["dob_date"] < cutoff_date]
    dob_1y = prior_dob[prior_dob["dob_date"] >= w1y].groupby("bbl").size()
    dob_all = prior_dob.groupby("bbl").size()
    features["feat_dob_1y"] = bbl_s.map(dob_1y).fillna(0).astype(np.float32).values
    features["feat_dob_all"] = bbl_s.map(dob_all).fillna(0).astype(np.float32).values
    features["feat_dob_per_unit_1y"] = (
        features["feat_dob_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_dob_per_unit_all"] = (
        features["feat_dob_all"] / (unitsres + 1.0)
    ).astype(np.float32)

    prior_dob_ecb = dob_ecb_df[dob_ecb_df["dob_ecb_date"] < cutoff_date]
    dob_ecb_1y = (
        prior_dob_ecb[prior_dob_ecb["dob_ecb_date"] >= w1y].groupby("bbl").size()
    )
    dob_ecb_all = prior_dob_ecb.groupby("bbl").size()
    features["feat_dob_ecb_1y"] = (
        bbl_s.map(dob_ecb_1y).fillna(0).astype(np.float32).values
    )
    features["feat_dob_ecb_all"] = (
        bbl_s.map(dob_ecb_all).fillna(0).astype(np.float32).values
    )
    features["feat_dob_ecb_per_unit_1y"] = (
        features["feat_dob_ecb_1y"] / (unitsres + 1.0)
    ).astype(np.float32)
    features["feat_dob_ecb_per_unit_all"] = (
        features["feat_dob_ecb_all"] / (unitsres + 1.0)
    ).astype(np.float32)

    feature_cols = [c for c in features.columns if c.startswith("feat_")]
    return features[["bbl"] + feature_cols], feature_cols


# Generate Train, Validation, and Test Datasets (Multi-temporal cohort pooling)
t_train_2020 = pd.Timestamp("2020-01-01")
print(f"Extracting features for 2020 train cohort ({t_train_2020})...")
df_train_2020, feature_names = extract_features(train_bbls, t_train_2020)
df_train_2020["target"] = (
    compute_labels(train_bbls, t_train_2020, hpd_viol).astype(np.int32).values
)

t_train_2021 = pd.Timestamp("2021-01-01")
print(f"Extracting features for 2021 train cohort ({t_train_2021})...")
df_train_2021, _ = extract_features(train_bbls, t_train_2021)
df_train_2021["target"] = (
    compute_labels(train_bbls, t_train_2021, hpd_viol).astype(np.int32).values
)

df_train_feat = pd.concat([df_train_2020, df_train_2021], ignore_index=True)
del df_train_2020, df_train_2021
gc.collect()

t_val = pd.Timestamp("2022-01-01")
print(f"Extracting features for 2022 validation cohort ({t_val})...")
df_val_feat, _ = extract_features(val_bbls, t_val)
df_val_feat["target"] = (
    compute_labels(val_bbls, t_val, hpd_viol).astype(np.int32).values
)

t_test = pd.Timestamp("2023-01-01")
print(f"Extracting features for test cohort ({t_test})...")
df_test_feat, _ = extract_features(test_bbls, t_test)

# Persist datasets
df_train_feat.to_parquet(
    os.path.join(WORKING_DIR, "train_features.parquet"), index=False
)
df_val_feat.to_parquet(os.path.join(WORKING_DIR, "val_features.parquet"), index=False)
df_test_feat.to_parquet(os.path.join(WORKING_DIR, "test_features.parquet"), index=False)
with open(os.path.join(WORKING_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

num_features = len(feature_names)


# ---------------------------------------------------------------------------
# 4. Model Design & Hyperparameters
# ---------------------------------------------------------------------------
lgb_model_params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 7,
    "min_child_samples": 80,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 1.0,
    "reg_alpha": 3.0,
    "reg_lambda": 10.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_model_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 1.0,
    "reg_alpha": 3.0,
    "reg_lambda": 10.0,
    "min_child_weight": 8.0,
    "n_estimators": 2500,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# 5. Training, Evaluation, and Inference
# ---------------------------------------------------------------------------
X_train_raw = (
    df_train_feat[feature_names].fillna(0).replace([np.inf, -np.inf], 0).values
)
y_train = df_train_feat["target"].values.astype(np.float32)

X_val_raw = df_val_feat[feature_names].fillna(0).replace([np.inf, -np.inf], 0).values
y_val = df_val_feat["target"].values.astype(np.float32)

X_test_raw = df_test_feat[feature_names].fillna(0).replace([np.inf, -np.inf], 0).values

print("Training LightGBM Classifier...")
lgb_clf = lgb.LGBMClassifier(**lgb_model_params)
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
lgb_clf.fit(
    X_train_raw,
    y_train,
    eval_set=[(X_val_raw, y_val)],
    callbacks=callbacks,
)
val_preds_lgb = lgb_clf.predict_proba(X_val_raw)[:, 1]
val_ap_lgb = average_precision_score(y_val, val_preds_lgb)
print(f"LightGBM Validation AP: {val_ap_lgb:.5f}")

print("Training XGBoost Classifier...")
try:
    xgb_clf = xgb.XGBClassifier(**xgb_model_params, early_stopping_rounds=50)
    xgb_clf.fit(
        X_train_raw,
        y_train,
        eval_set=[(X_val_raw, y_val)],
        verbose=False,
    )
except TypeError:
    xgb_clf = xgb.XGBClassifier(**xgb_model_params)
    xgb_clf.fit(
        X_train_raw,
        y_train,
        eval_set=[(X_val_raw, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

val_preds_xgb = xgb_clf.predict_proba(X_val_raw)[:, 1]
val_ap_xgb = average_precision_score(y_val, val_preds_xgb)
print(f"XGBoost Validation AP: {val_ap_xgb:.5f}")


# Rank ensemble optimization against validation AP
def to_rank_percentile(arr):
    return rankdata(arr) / len(arr)


rank_val_lgb = to_rank_percentile(val_preds_lgb)
rank_val_xgb = to_rank_percentile(val_preds_xgb)

best_w = 0.5
best_ensemble_ap = -1.0

# Grid search optimal blend weight w in [0.0, 1.0] with step 0.05
for w_val in np.linspace(0.0, 1.0, 21):
    w_val = round(float(w_val), 2)
    val_blend = w_val * rank_val_lgb + (1.0 - w_val) * rank_val_xgb
    ap = average_precision_score(y_val, val_blend)
    if ap > best_ensemble_ap:
        best_ensemble_ap = ap
        best_w = w_val

print(
    f"Optimal Ensemble Blend: LightGBM weight = {best_w:.2f}, "
    f"XGBoost weight = {1.0 - best_w:.2f} | Validation AP: {best_ensemble_ap:.5f}"
)

val_ensemble = best_w * rank_val_lgb + (1.0 - best_w) * rank_val_xgb
final_val_ap = average_precision_score(y_val, val_ensemble)
val_auc = roc_auc_score(y_val, val_ensemble)

n_val = len(y_val)
sorted_indices = np.argsort(-val_ensemble)
total_positives = y_val.sum()

p1_cutoff = int(n_val * 0.01)
p5_cutoff = int(n_val * 0.05)
p10_cutoff = int(n_val * 0.10)

prec_at_1 = y_val[sorted_indices[:p1_cutoff]].mean()
rec_at_1 = y_val[sorted_indices[:p1_cutoff]].sum() / max(total_positives, 1.0)
prec_at_5 = y_val[sorted_indices[:p5_cutoff]].mean()
rec_at_5 = y_val[sorted_indices[:p5_cutoff]].sum() / max(total_positives, 1.0)
prec_at_10 = y_val[sorted_indices[:p10_cutoff]].mean()
rec_at_10 = y_val[sorted_indices[:p10_cutoff]].sum() / max(total_positives, 1.0)

print(
    f"Ensemble Validation Diagnostics: AP = {final_val_ap:.5f} | ROC AUC = {val_auc:.5f} | "
    f"Prec@1% = {prec_at_1:.4f} (Rec={rec_at_1:.4f}) | "
    f"Prec@5% = {prec_at_5:.4f} (Rec={rec_at_5:.4f}) | "
    f"Prec@10% = {prec_at_10:.4f} (Rec={rec_at_10:.4f})"
)

# Test Inference
test_preds_lgb = lgb_clf.predict_proba(X_test_raw)[:, 1]
test_preds_xgb = xgb_clf.predict_proba(X_test_raw)[:, 1]

rank_test_lgb = to_rank_percentile(test_preds_lgb)
rank_test_xgb = to_rank_percentile(test_preds_xgb)
test_ensemble = best_w * rank_test_lgb + (1.0 - best_w) * rank_test_xgb

# Submission Generation & Validation
sub_df = pd.DataFrame(
    {
        "bbl": df_test["bbl"].astype(str).str.strip().str.zfill(10),
        "score": test_ensemble.astype(float),
    }
)

submission_file = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(submission_file, index=False)

assert len(sub_df) == len(
    df_test
), f"Row count mismatch: expected {len(df_test)}, got {len(sub_df)}"
assert len(sub_df) == 171587, f"Expected 171587 rows, got {len(sub_df)}"
assert not sub_df["bbl"].duplicated().any(), "Duplicate BBLs detected in submission!"
assert not sub_df["score"].isna().any(), "NaN values found in submission score!"
assert (
    sub_df["bbl"].str.len() == 10
).all(), "Malformed BBL length detected in submission!"

print(f"Final Validation Score: {final_val_ap:.6f}")

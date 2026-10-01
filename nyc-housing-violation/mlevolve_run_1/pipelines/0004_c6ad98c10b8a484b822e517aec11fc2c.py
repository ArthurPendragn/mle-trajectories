import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
]
hpd_viol = pd.read_parquet(
    f"{LAKE_BASE}/hpd_violations/",
    columns=violation_cols,
    storage_options=STORAGE_OPTIONS,
)
hpd_viol["bbl"] = clean_bbl_series(hpd_viol)
hpd_viol["inspectiondate"] = ensure_tz_naive(hpd_viol["inspectiondate"])
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

    w1y = cutoff_date - pd.Timedelta(days=365)
    w2y = cutoff_date - pd.Timedelta(days=730)
    w3y = cutoff_date - pd.Timedelta(days=1095)
    w5y = cutoff_date - pd.Timedelta(days=1825)

    v_1y = prior_viols[prior_viols["inspectiondate"] >= w1y]
    v_2y = prior_viols[prior_viols["inspectiondate"] >= w2y]
    v_3y = prior_viols[prior_viols["inspectiondate"] >= w3y]
    v_5y = prior_viols[prior_viols["inspectiondate"] >= w5y]

    def count_by_bbl(df_sub, c_val=None):
        if c_val is not None:
            df_sub = df_sub[df_sub["class"] == c_val]
        return df_sub.groupby("bbl").size()

    c_1y = count_by_bbl(v_1y, "C")
    c_2y = count_by_bbl(v_2y, "C")
    c_3y = count_by_bbl(v_3y, "C")
    c_5y = count_by_bbl(v_5y, "C")
    c_all = count_by_bbl(prior_viols, "C")

    b_1y = count_by_bbl(v_1y, "B")
    b_3y = count_by_bbl(v_3y, "B")
    a_1y = count_by_bbl(v_1y, "A")

    tot_1y = count_by_bbl(v_1y)
    tot_2y = count_by_bbl(v_2y)
    tot_3y = count_by_bbl(v_3y)
    tot_all = count_by_bbl(prior_viols)

    c_viols_only = prior_viols[prior_viols["class"] == "C"]
    max_c_date = c_viols_only.groupby("bbl")["inspectiondate"].max()
    max_any_date = prior_viols.groupby("bbl")["inspectiondate"].max()

    bbl_s = features["bbl"]
    features["feat_viol_c_1y"] = bbl_s.map(c_1y).fillna(0).astype(np.float32).values
    features["feat_viol_c_2y"] = bbl_s.map(c_2y).fillna(0).astype(np.float32).values
    features["feat_viol_c_3y"] = bbl_s.map(c_3y).fillna(0).astype(np.float32).values
    features["feat_viol_c_5y"] = bbl_s.map(c_5y).fillna(0).astype(np.float32).values
    features["feat_viol_c_all"] = bbl_s.map(c_all).fillna(0).astype(np.float32).values

    features["feat_viol_b_1y"] = bbl_s.map(b_1y).fillna(0).astype(np.float32).values
    features["feat_viol_b_3y"] = bbl_s.map(b_3y).fillna(0).astype(np.float32).values
    features["feat_viol_a_1y"] = bbl_s.map(a_1y).fillna(0).astype(np.float32).values

    features["feat_viol_tot_1y"] = bbl_s.map(tot_1y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_2y"] = bbl_s.map(tot_2y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_3y"] = bbl_s.map(tot_3y).fillna(0).astype(np.float32).values
    features["feat_viol_tot_all"] = (
        bbl_s.map(tot_all).fillna(0).astype(np.float32).values
    )

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

    # Violation Recency
    last_c_days = (cutoff_date - bbl_s.map(max_c_date)).dt.days.fillna(3650)
    last_any_days = (cutoff_date - bbl_s.map(max_any_date)).dt.days.fillna(3650)
    features["feat_days_since_last_c"] = last_c_days.astype(np.float32).values
    features["feat_days_since_last_any"] = last_any_days.astype(np.float32).values
    features["feat_has_prior_c"] = (features["feat_viol_c_all"] > 0).astype(np.float32)

    # Active Violation Backlog (strictly prior to cutoff_date)
    open_viols = prior_viols[prior_viols["violationstatus"] == "Open"]
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

    feature_cols = [c for c in features.columns if c.startswith("feat_")]
    return features[["bbl"] + feature_cols], feature_cols


# Generate Train, Validation, and Test Datasets
t_train = pd.Timestamp("2021-01-01")
df_train_feat, feature_names = extract_features(train_bbls, t_train)
df_train_feat["target"] = (
    compute_labels(train_bbls, t_train, hpd_viol).astype(np.int32).values
)

t_val = pd.Timestamp("2022-01-01")
df_val_feat, _ = extract_features(val_bbls, t_val)
df_val_feat["target"] = (
    compute_labels(val_bbls, t_val, hpd_viol).astype(np.int32).values
)

t_test = pd.Timestamp("2023-01-01")
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
# 4. Neural Architecture & Loss Design
# ---------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    """Residual dense block with LayerNorm, Mish activations, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        return self.activation(out + residual)


class HousingRiskResNet(nn.Module):
    """Deep Tabular ResNet for building violation risk ranking."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits


class FocalBinaryCrossEntropyLoss(nn.Module):
    """Focal Binary Cross-Entropy Loss to handle class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 2.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.pos_weight + (1.0 - targets) * 1.0
        modulating_factor = (1.0 - p_t) ** self.gamma
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Auxiliary GBDT Configuration specification
lgb_model_params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 7,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 2.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Instantiate model, loss, optimizer, scheduler
model = HousingRiskResNet(
    in_features=num_features, hidden_dim=128, num_blocks=3, dropout_rate=0.25
).to(device)

criterion = FocalBinaryCrossEntropyLoss(gamma=2.0, pos_weight=2.5, reduction="mean").to(
    device
)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)

scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)


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

# Standardize features using strictly training-set statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32).clip(-10.0, 10.0)
X_val_scaled = scaler.transform(X_val_raw).astype(np.float32).clip(-10.0, 10.0)
X_test_scaled = scaler.transform(X_test_raw).astype(np.float32).clip(-10.0, 10.0)

batch_size = 2048
train_dataset = TensorDataset(
    torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)
)
val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test_scaled))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

epochs = 15
best_val_ap = -1.0
best_model_path = os.path.join(WORKING_DIR, "best_housing_risk_resnet.pt")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(batch_y)
        total_samples += len(batch_y)

    scheduler.step()
    epoch_train_loss = running_loss / max(total_samples, 1)

    model.eval()
    val_loss = 0.0
    val_samples = 0
    val_preds_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            val_loss += loss.item() * len(batch_y)
            val_samples += len(batch_y)
            val_preds_list.append(probs.cpu().numpy())

    epoch_val_loss = val_loss / max(val_samples, 1)
    val_preds = np.concatenate(val_preds_list)
    val_ap = average_precision_score(y_val, val_preds)

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} |"
        f" Val Loss: {epoch_val_loss:.4f} | Val AP: {val_ap:.5f}"
    )

# Train LightGBM model
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

# Restore best checkpoint for final evaluation and test scoring
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
final_val_preds_list = []

with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        final_val_preds_list.append(probs.cpu().numpy())

final_val_preds = np.concatenate(final_val_preds_list)
val_ap_resnet = average_precision_score(y_val, final_val_preds)
print(f"HousingRiskResNet Validation AP: {val_ap_resnet:.5f}")

# Fuse predictions via rank percentiles
def to_rank_percentile(arr):
    return rankdata(arr) / len(arr)

rank_val_resnet = to_rank_percentile(final_val_preds)
rank_val_lgb = to_rank_percentile(val_preds_lgb)
val_ensemble = 0.5 * rank_val_resnet + 0.5 * rank_val_lgb

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
test_preds_list = []

with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        test_preds_list.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds_list)
test_preds_lgb = lgb_clf.predict_proba(X_test_raw)[:, 1]

rank_test_resnet = to_rank_percentile(test_preds)
rank_test_lgb = to_rank_percentile(test_preds_lgb)
test_ensemble = 0.5 * rank_test_resnet + 0.5 * rank_test_lgb

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
assert not sub_df["bbl"].duplicated().any(), "Duplicate BBLs detected in submission!"
assert not sub_df["score"].isna().any(), "NaN values found in submission score!"
assert (
    sub_df["bbl"].str.len() == 10
).all(), "Malformed BBL length detected in submission!"

print(f"Final Validation Score: {final_val_ap:.6f}")

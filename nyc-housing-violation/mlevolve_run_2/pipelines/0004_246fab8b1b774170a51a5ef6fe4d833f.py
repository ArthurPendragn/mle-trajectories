import copy
import gc
import json
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_bbl_series(
    df,
    bbl_col="bbl",
    boro_col="boroid",
    block_col="block",
    lot_col="lot",
):
    """Standardizes BBL into a clean 10-digit string according to task specifications."""
    has_bbl = bbl_col in df.columns
    if has_bbl:
        s = df[bbl_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        valid_mask = (s.str.len() == 10) & s.str.isdigit()
    else:
        s = pd.Series("", index=df.index)
        valid_mask = pd.Series(False, index=df.index)

    boro_cols = [c for c in [boro_col, "borough", "borocode", "boro"] if c in df]
    if len(boro_cols) > 0 and block_col in df.columns and lot_col in df.columns:
        boro = (
            pd.to_numeric(df[boro_cols[0]], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
        block = (
            pd.to_numeric(df[block_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .apply(lambda x: f"{x:05d}")
        )
        lot = (
            pd.to_numeric(df[lot_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .apply(lambda x: f"{x:04d}")
        )
        fallback = boro + block + lot
        return s.where(valid_mask, fallback)
    return s


# =========================================================================
# 1. Ingestion: Cohort Construction & Raw Lake Tables
# =========================================================================
print("Starting data ingestion and feature engineering...")

# Test Entities
try:
    df_sample = pd.read_parquet(
        f"{GCS_BASE}/sample_submission.parquet", storage_options=storage_options
    )
    test_bbls = clean_bbl_series(df_sample, bbl_col="bbl")
    df_test = pd.DataFrame({"bbl": test_bbls})
except Exception:
    df_test_raw = pd.read_parquet(
        f"{GCS_BASE}/test_entities.parquet", storage_options=storage_options
    )
    test_bbls = clean_bbl_series(df_test_raw, bbl_col="bbl")
    df_test = (
        pd.DataFrame({"bbl": test_bbls})
        .drop_duplicates(subset=["bbl"])
        .reset_index(drop=True)
    )

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
]

df_pluto = pd.read_parquet(
    f"{LAKE_FULL}/pluto",
    columns=pluto_cols,
    storage_options=storage_options,
)

df_pluto["bbl"] = clean_bbl_series(
    df_pluto, bbl_col="bbl", boro_col="borough", block_col="block", lot_col="lot"
)
df_pluto = df_pluto.drop(columns=["block", "lot"], errors="ignore")
df_pluto = df_pluto.drop_duplicates(subset=["bbl"], keep="last")
print(f"Loaded PLUTO records: {len(df_pluto)} unique lots")

# Create Train (T = 2021-01-01) and Val (T = 2022-01-01) Cohorts (unitsres >= 3)
pluto_md = df_pluto[df_pluto["unitsres"].fillna(0) >= 3].copy()
all_md_bbls = pluto_md["bbl"].unique()

df_val = pd.DataFrame({"bbl": all_md_bbls})
df_train = pd.DataFrame({"bbl": all_md_bbls})
print(
    f"Constructed train cohort ({len(df_train)} lots) and val cohort"
    f" ({len(df_val)} lots)"
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

train_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2021-01-01")
        & (c_violations["inspectiondate"] < "2022-01-01")
    ]["bbl"].unique()
)
df_train["target"] = df_train["bbl"].isin(train_pos_bbls).astype(np.int32)

val_pos_bbls = set(
    c_violations[
        (c_violations["inspectiondate"] >= "2022-01-01")
        & (c_violations["inspectiondate"] < "2023-01-01")
    ]["bbl"].unique()
)
df_val["target"] = df_val["bbl"].isin(val_pos_bbls).astype(np.int32)
print(
    f"Train positive rate: {df_train['target'].mean():.4f} | Val positive rate:"
    f" {df_val['target'].mean():.4f}"
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
        columns=["bbl"],
        storage_options=storage_options,
    )
    vacate_bbl_set = set(clean_bbl_series(df_vacate, bbl_col="bbl").unique())
except Exception:
    vacate_bbl_set = set()

try:
    df_omo = pd.read_parquet(
        f"{LAKE_FULL}/hpd_omo_charges",
        columns=["bbl"],
        storage_options=storage_options,
    )
    omo_bbl_set = set(clean_bbl_series(df_omo, bbl_col="bbl").unique())
except Exception:
    omo_bbl_set = set()

try:
    df_aep = pd.read_parquet(
        f"{LAKE_FULL}/hpd_aep_buildings",
        columns=["bbl"],
        storage_options=storage_options,
    )
    aep_bbl_set = set(clean_bbl_series(df_aep, bbl_col="bbl").unique())
except Exception:
    aep_bbl_set = set()


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
    t_1y = cutoff - pd.DateOffset(years=1)
    t_2y = cutoff - pd.DateOffset(years=2)
    t_3y = cutoff - pd.DateOffset(years=3)
    t_5y = cutoff - pd.DateOffset(years=5)

    v_1y = prior_viol[prior_viol["inspectiondate"] >= t_1y]
    v_2y = prior_viol[prior_viol["inspectiondate"] >= t_2y]
    v_3y = prior_viol[prior_viol["inspectiondate"] >= t_3y]
    v_5y = prior_viol[prior_viol["inspectiondate"] >= t_5y]

    vc_all = prior_viol[prior_viol["class"] == "C"]
    vc_1y = v_1y[v_1y["class"] == "C"]
    vc_2y = v_2y[v_2y["class"] == "C"]
    vc_3y = v_3y[v_3y["class"] == "C"]
    vc_5y = v_5y[v_5y["class"] == "C"]

    vb_all = prior_viol[prior_viol["class"] == "B"]
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

    counts_list = [
        get_counts(vc_1y, "viol_c_cnt_1y"),
        get_counts(vc_2y, "viol_c_cnt_2y"),
        get_counts(vc_3y, "viol_c_cnt_3y"),
        get_counts(vc_5y, "viol_c_cnt_5y"),
        get_counts(vc_all, "viol_c_cnt_all"),
        get_counts(vb_1y, "viol_b_cnt_1y"),
        get_counts(vb_3y, "viol_b_cnt_3y"),
        get_counts(vb_all, "viol_b_cnt_all"),
        get_counts(va_1y, "viol_a_cnt_1y"),
        get_counts(va_all, "viol_a_cnt_all"),
        get_counts(v_1y, "viol_all_cnt_1y"),
        get_counts(v_3y, "viol_all_cnt_3y"),
        get_counts(prior_viol, "viol_all_cnt_all"),
    ]

    for c_df in counts_list:
        res = res.merge(c_df, on="bbl", how="left")
        col = c_df.columns[1]
        res[col] = res[col].fillna(0).astype(np.float32)

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
    res["viol_c_yoy_diff"] = (
        res["viol_c_cnt_1y"] - (res["viol_c_cnt_2y"] - res["viol_c_cnt_1y"])
    ).astype(np.float32)
    res["viol_c_per_unit_1y"] = (res["viol_c_cnt_1y"] / (res["unitsres"] + 1.0)).astype(
        np.float32
    )
    res["viol_all_per_unit_1y"] = (
        res["viol_all_cnt_1y"] / (res["unitsres"] + 1.0)
    ).astype(np.float32)

    # Complaints Aggregations
    if len(df_complaints) > 0:
        prior_comp = df_complaints[df_complaints["date"] < cutoff]
        comp_1y = prior_comp[prior_comp["date"] >= t_1y]
        comp_3y = prior_comp[prior_comp["date"] >= t_3y]

        res = res.merge(get_counts(comp_1y, "complaint_cnt_1y"), on="bbl", how="left")
        res = res.merge(get_counts(comp_3y, "complaint_cnt_3y"), on="bbl", how="left")
        res = res.merge(get_counts(prior_comp, "complaint_cnt_all"), on="bbl", how="left")

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
        res["complaint_per_unit_1y"] = (
            res["complaint_cnt_1y"] / (res["unitsres"] + 1.0)
        ).astype(np.float32)
        res["complaint_to_viol_ratio_1y"] = (
            res["complaint_cnt_1y"] / (res["viol_all_cnt_1y"] + 1.0)
        ).astype(np.float32)
    else:
        for c in [
            "complaint_cnt_1y",
            "complaint_cnt_3y",
            "complaint_cnt_all",
            "days_since_last_complaint",
            "complaint_velocity",
            "complaint_per_unit_1y",
            "complaint_to_viol_ratio_1y",
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

    # Enforcement Flags
    res["has_vacate_order"] = res["bbl"].isin(vacate_bbl_set).astype(np.float32)
    res["has_emergency_charges"] = res["bbl"].isin(omo_bbl_set).astype(np.float32)
    res["is_aep_building"] = res["bbl"].isin(aep_bbl_set).astype(np.float32)

    cols_to_drop = ["block", "lot", "bbl_clean", "version"]
    res = res.drop(columns=[c for c in cols_to_drop if c in res.columns])
    return res


print("Extracting features for Training cohort (Cutoff: 2021-01-01)...")
X_train = extract_cohort_features(df_train, "2021-01-01")
X_train["target"] = df_train["target"].values

print("Extracting features for Validation cohort (Cutoff: 2022-01-01)...")
X_val = extract_cohort_features(df_val, "2022-01-01")
X_val["target"] = df_val["target"].values

print("Extracting features for Test cohort (Cutoff: 2023-01-01)...")
X_test = extract_cohort_features(df_test, "2023-01-01")

# Clean up raw event logs to free memory
del df_viol, c_violations
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


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HousingRiskTabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.input_norm = nn.LayerNorm(in_features)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.input_norm(x)
        out = self.input_proj(out)
        for block in self.res_blocks:
            out = block(out)
        logits = self.head(out)
        return logits.squeeze(-1)


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits).float()
        probs = torch.sigmoid(logits)

        log_pos = F.logsigmoid(logits)
        log_neg = F.logsigmoid(-logits)

        pos_modulating = self.alpha * torch.pow(1.0 - probs, self.gamma)
        neg_modulating = (1.0 - self.alpha) * torch.pow(probs, self.gamma)

        loss = (
            -targets * pos_modulating * log_pos
            - (1.0 - targets) * neg_modulating * log_neg
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Initialize Models and Optimizers
num_input_features = len(feature_names)
nn_model = HousingRiskTabularResNet(
    in_features=num_input_features, hidden_dim=128, num_blocks=3, dropout=0.25
).to(device)

criterion = AsymmetricFocalLoss(gamma=2.0, alpha=0.75)
optimizer = torch.optim.AdamW(
    nn_model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=12, eta_min=1e-5
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5.0,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

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

train_dataset = TensorDataset(torch.from_numpy(X_tr_mat), torch.from_numpy(y_tr))
val_dataset = TensorDataset(torch.from_numpy(X_val_mat), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_te_mat))

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, drop_last=False)

# Train Deep Tabular ResNet
num_epochs = 12
best_val_ap_nn = -1.0
best_model_state = None

for epoch in range(num_epochs):
    nn_model.train()
    running_loss = 0.0
    total_samples = 0

    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        logits = nn_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(by)
        total_samples += len(by)

    scheduler.step()
    epoch_loss = running_loss / max(total_samples, 1)

    nn_model.eval()
    val_preds_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_preds_list.append(torch.sigmoid(nn_model(bx)).cpu().numpy())
    val_preds_nn_epoch = np.concatenate(val_preds_list)
    val_ap_nn_epoch = float(average_precision_score(y_val, val_preds_nn_epoch))

    if val_ap_nn_epoch > best_val_ap_nn:
        best_val_ap_nn = val_ap_nn_epoch
        best_model_state = copy.deepcopy(nn_model.state_dict())

    print(
        f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Focal Loss:"
        f" {epoch_loss:.4f} | Val AP: {val_ap_nn_epoch:.4f}"
    )

if best_model_state is not None:
    nn_model.load_state_dict(best_model_state)
torch.save(nn_model.state_dict(), os.path.join(WORKING_DIR, "best_tabular_resnet.pt"))

# Deep Model Predictions
nn_model.eval()
val_preds_list = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        val_preds_list.append(torch.sigmoid(nn_model(bx)).cpu().numpy())
val_preds_nn = np.concatenate(val_preds_list)

test_preds_list = []
with torch.no_grad():
    for batch in test_loader:
        bx = batch[0].to(device)
        test_preds_list.append(torch.sigmoid(nn_model(bx)).cpu().numpy())
test_preds_nn = np.concatenate(test_preds_list)

# Train LightGBM GBDT
callbacks = [
    lgb.early_stopping(stopping_rounds=40, verbose=False),
    lgb.log_evaluation(period=0),
]
lgb_model.fit(
    X_tr_mat,
    y_tr,
    eval_set=[(X_val_mat, y_val)],
    callbacks=callbacks,
)
lgb_model.booster_.save_model(os.path.join(WORKING_DIR, "lgbm_model.txt"))

val_preds_lgb = lgb_model.predict_proba(X_val_mat)[:, 1]
test_preds_lgb = lgb_model.predict_proba(X_te_mat)[:, 1]

# Rank Normalization and Blend Calibration
rank_val_nn = rankdata(val_preds_nn) / len(val_preds_nn)
rank_val_lgb = rankdata(val_preds_lgb) / len(val_preds_lgb)

rank_test_nn = rankdata(test_preds_nn) / len(test_preds_nn)
rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)

best_score = -1.0
best_weight_nn = 0.5
for w in np.linspace(0.0, 1.0, 21):
    blended_val = w * rank_val_nn + (1.0 - w) * rank_val_lgb
    ap_score = float(average_precision_score(y_val, blended_val))
    if ap_score > best_score:
        best_score = ap_score
        best_weight_nn = float(w)

final_val_preds = best_weight_nn * rank_val_nn + (1.0 - best_weight_nn) * rank_val_lgb
final_test_preds = (
    best_weight_nn * rank_test_nn + (1.0 - best_weight_nn) * rank_test_lgb
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

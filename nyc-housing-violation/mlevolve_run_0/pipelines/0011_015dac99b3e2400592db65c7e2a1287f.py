import copy
import json
import math
import os
import sys
import gcsfs
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# 1. Environment and Configuration Setup
# -------------------------------------------------------------------------
TOKEN_PATH = (
    "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
)
storage_options = {"token": TOKEN_PATH} if os.path.exists(TOKEN_PATH) else {}
fs = (
    gcsfs.GCSFileSystem(token=TOKEN_PATH)
    if os.path.exists(TOKEN_PATH)
    else gcsfs.GCSFileSystem()
)

LAKE_FULL = "mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full"
TEST_ENTITIES_PATH = (
    "gs://mle-nyc-lake/tasks/housing_violation_risk/v1/test_entities.parquet"
)

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 2. Key Standardization and Lake Ingestion Utilities
# -------------------------------------------------------------------------
def clean_bbl_series(s):
    """Converts a BBL series into clean 10-digit zero-padded string representation."""
    s_clean = (
        pd.to_numeric(s, errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.zfill(10)
    )
    return s_clean


def extract_violations_bbl(df_viol):
    """Implements exact official task definition for violation BBL:

    Row's BBL is 'bbl' when 10 digits; otherwise boroid (1 digit) +
    block (5 digits, zero-padded) + lot (4 digits, zero-padded).
    """
    if "bbl" in df_viol.columns:
        raw_bbl = (
            pd.to_numeric(df_viol["bbl"], errors="coerce")
            .fillna(0)
            .astype("int64")
            .astype(str)
        )
        is_10_digit = (raw_bbl.str.len() == 10) & (raw_bbl != "0000000000")
    else:
        is_10_digit = pd.Series(False, index=df_viol.index)
        raw_bbl = pd.Series("", index=df_viol.index)

    boro = (
        pd.to_numeric(df_viol["boroid"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
    )
    block = (
        pd.to_numeric(df_viol["block"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )
    lot = (
        pd.to_numeric(df_viol["lot"], errors="coerce")
        .fillna(0)
        .astype("int64")
        .astype(str)
        .str.strip()
        .str.zfill(4)
    )
    fallback = boro + block + lot
    final_bbl = raw_bbl.where(is_10_digit, fallback)

    return final_bbl.str.zfill(10)


def list_lake_tables():
    try:
        return fs.ls(LAKE_FULL)
    except Exception as e:
        return []


LAKE_TABLES = list_lake_tables()


def load_pluto_cohort(release_name):
    """Finds and loads the PLUTO release (19v2, 20v7, 21v4, 22v3)."""
    candidates = []
    for tbl in LAKE_TABLES:
        tbl_name = tbl.split("/")[-1].lower()
        if "pluto" in tbl_name:
            candidates.append(f"{tbl}/release={release_name}")
            candidates.append(f"{tbl}/{release_name}")
            if release_name.lower() in tbl_name:
                candidates.append(tbl)

    for p in candidates:
        try:
            if fs.exists(p):
                df = pd.read_parquet(f"gs://{p}", storage_options=storage_options)
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            continue

    for tbl in LAKE_TABLES:
        if "pluto" in tbl.lower():
            try:
                df = pd.read_parquet(
                    f"gs://{tbl}",
                    filters=[("release", "==", release_name)],
                    storage_options=storage_options,
                )
                if len(df) > 0:
                    df.columns = [c.lower() for c in df.columns]
                    return df
            except Exception:
                pass

    direct_paths = [
        f"{LAKE_FULL}/pluto/release={release_name}",
        f"{LAKE_FULL}/dcp_pluto/release={release_name}",
        f"{LAKE_FULL}/pluto_{release_name}",
    ]
    for dp in direct_paths:
        try:
            if fs.exists(dp):
                df = pd.read_parquet(f"gs://{dp}", storage_options=storage_options)
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            pass

    raise FileNotFoundError(f"Unable to locate PLUTO release {release_name} in lake.")


def load_hpd_violations(start_year=2017):
    """Efficiently loads HPD violations from lake."""
    viol_tbl = None
    for tbl in LAKE_TABLES:
        tbl_name = tbl.split("/")[-1].lower()
        if "hpd_violations" in tbl_name or (
            "hpd" in tbl_name and "violation" in tbl_name
        ):
            viol_tbl = tbl
            break
    if viol_tbl is None:
        viol_tbl = f"{LAKE_FULL}/hpd_violations"

    needed_cols = [
        "bbl",
        "boroid",
        "block",
        "lot",
        "class",
        "inspectiondate",
        "currentstatus",
    ]

    try:
        subdirs = fs.ls(viol_tbl)
    except Exception:
        subdirs = []

    year_partitions = [
        s
        for s in subdirs
        if "year=" in s and any(str(yr) in s for yr in range(start_year, 2024))
    ]

    if year_partitions:
        dfs = []
        for p in sorted(year_partitions):
            try:
                part = pd.read_parquet(
                    f"gs://{p}",
                    storage_options=storage_options,
                    columns=[c for c in needed_cols if c != "year"],
                )
                part.columns = [c.lower() for c in part.columns]
                dfs.append(part)
            except Exception:
                try:
                    part = pd.read_parquet(f"gs://{p}", storage_options=storage_options)
                    part.columns = [c.lower() for c in part.columns]
                    dfs.append(part)
                except Exception:
                    pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    try:
        df_v = pd.read_parquet(
            f"gs://{viol_tbl}",
            storage_options=storage_options,
            columns=needed_cols,
        )
    except Exception:
        df_v = pd.read_parquet(f"gs://{viol_tbl}", storage_options=storage_options)
    df_v.columns = [c.lower() for c in df_v.columns]
    return df_v


# -------------------------------------------------------------------------
# 3. Leak-Free Feature Engineering
# -------------------------------------------------------------------------
PHYSICAL_FEATURES = [
    "feat_building_age",
    "feat_is_prewar",
    "feat_units_per_floor",
    "feat_area_per_unit",
    "feat_assessed_per_unit",
    "feat_land_value_ratio",
    "feat_commercial_unit_ratio",
    "feat_unitsres",
    "feat_unitstotal",
    "feat_numfloors",
    "feat_bldgarea",
    "feat_lotarea",
    "feat_builtfar",
    "feat_residfar",
    "feat_assessland",
    "feat_assesstot",
]

RECENCY_SEVERITY_FEATURES = [
    "feat_days_since_last_c",
    "feat_days_since_last_any",
    "feat_has_prior_c_365d",
    "feat_has_prior_c_1095d",
    "feat_has_prior_any_365d",
    "feat_severity_index_365d",
    "feat_severity_index_90d",
    "feat_c_exp_decay_90d",
    "feat_c_exp_decay_365d",
    "feat_tot_exp_decay_90d",
    "feat_tot_exp_decay_365d",
    "feat_open_c_count",
    "feat_open_total",
    "feat_open_c_ratio",
    "feat_open_tot_ratio",
    "feat_c_active_years_3yr",
    "feat_c_consecutive_years",
    "feat_viol_per_unit_365d",
    "feat_c_viol_per_unit_365d",
]

TEMPORAL_VELOCITY_FEATURES = [
    "feat_c_count_30d",
    "feat_c_count_60d",
    "feat_c_count_prev_60d",
    "feat_c_count_90d",
    "feat_c_count_180d",
    "feat_c_count_365d",
    "feat_c_count_730d",
    "feat_c_count_1095d",
    "feat_b_count_90d",
    "feat_b_count_365d",
    "feat_b_count_1095d",
    "feat_a_count_365d",
    "feat_a_count_1095d",
    "feat_tot_viol_30d",
    "feat_tot_viol_90d",
    "feat_tot_viol_365d",
    "feat_tot_viol_730d",
    "feat_tot_viol_1095d",
    "feat_c_ratio_365d",
    "feat_b_ratio_365d",
    "feat_c_heat_diff_60d",
    "feat_c_heat_ratio_60d",
    "feat_c_yoy_diff",
    "feat_c_yoy_ratio",
    "feat_tot_yoy_diff",
    "feat_tot_yoy_ratio",
    "feat_c_velocity_90_365",
    "feat_tot_velocity_90_365",
]

SPATIAL_BASE_RATE_FEATURES = [
    "feat_cd_c_rate_365d",
    "feat_cd_c_mean_365d",
    "feat_cd_tot_rate_365d",
    "feat_block_c_rate_365d",
    "feat_block_c_mean_365d",
]

DOMAIN_FEATURE_GROUPS = [
    PHYSICAL_FEATURES,
    RECENCY_SEVERITY_FEATURES,
    TEMPORAL_VELOCITY_FEATURES,
    SPATIAL_BASE_RATE_FEATURES,
]
FEATURE_COLUMNS = (
    PHYSICAL_FEATURES
    + RECENCY_SEVERITY_FEATURES
    + TEMPORAL_VELOCITY_FEATURES
    + SPATIAL_BASE_RATE_FEATURES
)
GROUP_DIMS = [len(g) for g in DOMAIN_FEATURE_GROUPS]


def extract_features(entities_df, violations_df, cutoff_timestamp):
    """Computes point-in-time structural, temporal, velocity, and severity

    features for a given entity cohort strictly using data prior to
    cutoff_timestamp.
    """
    T = pd.Timestamp(cutoff_timestamp)

    df = entities_df.copy()
    if "bbl" not in df.columns:
        df["bbl"] = clean_bbl_series(df.index)
    else:
        df["bbl"] = clean_bbl_series(df["bbl"])

    pluto_num_defaults = {
        "unitsres": 3.0,
        "unitstotal": 3.0,
        "numfloors": 3.0,
        "bldgarea": 3000.0,
        "lotarea": 2500.0,
        "builtfar": 1.5,
        "residfar": 1.5,
        "yearbuilt": 1940.0,
        "assessland": 50000.0,
        "assesstot": 150000.0,
    }

    for col, default_val in pluto_num_defaults.items():
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(default_val)
                .clip(lower=0)
            )
        else:
            df[col] = default_val

    # Store raw units for denominator normalization
    raw_unitsres = np.maximum(df["unitsres"], 1.0)
    raw_unitstotal = np.maximum(df["unitstotal"], 1.0)
    raw_numfloors = np.maximum(df["numfloors"], 1.0)
    raw_assesstot = np.maximum(df["assesstot"], 1.0)

    # Physical Building Domain Features with Log1p transformations
    df["feat_building_age"] = (T.year - df["yearbuilt"]).clip(0, 200).astype(np.float32)
    df["feat_is_prewar"] = (df["yearbuilt"] < 1940).astype(np.float32)
    df["feat_units_per_floor"] = np.log1p(df["unitsres"] / raw_numfloors).astype(np.float32)
    df["feat_area_per_unit"] = np.log1p(df["bldgarea"] / raw_unitsres).astype(np.float32)
    df["feat_assessed_per_unit"] = np.log1p(df["assesstot"] / raw_unitsres).astype(np.float32)
    df["feat_land_value_ratio"] = (df["assessland"] / raw_assesstot).astype(np.float32)
    df["feat_commercial_unit_ratio"] = (
        np.maximum(df["unitstotal"] - df["unitsres"], 0.0) / raw_unitstotal
    ).astype(np.float32)
    df["feat_unitsres"] = np.log1p(df["unitsres"]).astype(np.float32)
    df["feat_unitstotal"] = np.log1p(df["unitstotal"]).astype(np.float32)
    df["feat_numfloors"] = np.log1p(df["numfloors"]).astype(np.float32)
    df["feat_bldgarea"] = np.log1p(df["bldgarea"]).astype(np.float32)
    df["feat_lotarea"] = np.log1p(df["lotarea"]).astype(np.float32)
    df["feat_builtfar"] = df["builtfar"].astype(np.float32)
    df["feat_residfar"] = df["residfar"].astype(np.float32)
    df["feat_assessland"] = np.log1p(df["assessland"]).astype(np.float32)
    df["feat_assesstot"] = np.log1p(df["assesstot"]).astype(np.float32)

    # Discrete categorical integer representations (borough, community district, building class)
    if "borough" in df.columns:
        boro_raw = pd.to_numeric(df["borough"], errors="coerce").fillna(0).astype(int)
    elif "borocode" in df.columns:
        boro_raw = pd.to_numeric(df["borocode"], errors="coerce").fillna(0).astype(int)
    else:
        boro_raw = pd.to_numeric(df["bbl"].str[0], errors="coerce").fillna(0).astype(int)
    df["cat_borough"] = np.where(boro_raw.isin([1, 2, 3, 4, 5]), boro_raw, 0).astype(np.int64)

    all_cds = [
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 164,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 226, 227, 228,
        301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 355, 356,
        401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 480, 481, 482, 483, 484,
        501, 502, 503, 595,
    ]
    cd_map = {cd: i + 1 for i, cd in enumerate(all_cds)}
    if "cd" in df.columns:
        raw_cd = pd.to_numeric(df["cd"], errors="coerce").fillna(0).astype(int)
    else:
        raw_cd = pd.Series(0, index=df.index)
    df["cat_cd"] = raw_cd.map(cd_map).fillna(0).astype(np.int64)

    if "bldgclass" in df.columns:
        major_class = (
            df["bldgclass"].fillna("").astype(str).str.strip().str[:1].str.upper()
        )
    else:
        major_class = pd.Series("", index=df.index)
    bldg_classes = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    bldg_class_map = {ch: i + 1 for i, ch in enumerate(bldg_classes)}
    df["cat_bldgclass"] = major_class.map(bldg_class_map).fillna(0).astype(np.int64)

    # Historical violation aggregations strictly prior to T
    v_hist = violations_df[violations_df["inspectiondate"] < T]
    t_3yr = T - pd.Timedelta(days=1095)
    v_3yr = v_hist[v_hist["inspectiondate"] >= t_3yr].copy()

    v_c = v_3yr[v_3yr["class"] == "C"].copy()
    v_b = v_3yr[v_3yr["class"] == "B"].copy()
    v_a = v_3yr[v_3yr["class"] == "A"].copy()

    # Vectorized exponential decay violation intensity scores with 90d and 365d half-lives
    dt_c = (T - v_c["inspectiondate"]).dt.total_seconds() / 86400.0
    v_c["decay_90"] = np.exp(-dt_c / 90.0)
    v_c["decay_365"] = np.exp(-dt_c / 365.0)
    exp_c_90 = v_c.groupby("clean_bbl")["decay_90"].sum().rename("feat_c_exp_decay_90d")
    exp_c_365 = v_c.groupby("decay_365")["decay_365"].sum().rename("feat_c_exp_decay_365d")
    exp_c_365 = v_c.groupby("clean_bbl")["decay_365"].sum().rename("feat_c_exp_decay_365d")

    dt_tot = (T - v_3yr["inspectiondate"]).dt.total_seconds() / 86400.0
    v_3yr["decay_90"] = np.exp(-dt_tot / 90.0)
    v_3yr["decay_365"] = np.exp(-dt_tot / 365.0)
    exp_tot_90 = v_3yr.groupby("clean_bbl")["decay_90"].sum().rename("feat_tot_exp_decay_90d")
    exp_tot_365 = v_3yr.groupby("clean_bbl")["decay_365"].sum().rename("feat_tot_exp_decay_365d")

    # Chronic violation recidivism: distinct active years with Class C in lookback
    v_c_copy = v_c.copy()
    v_c_copy["insp_year"] = v_c_copy["inspectiondate"].dt.year
    c_active_years = (
        v_c_copy.groupby("clean_bbl")["insp_year"]
        .nunique()
        .rename("feat_c_active_years_3yr")
    )

    c_30 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=30))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_30d")
    )
    c_60 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=60))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_60d")
    )
    c_prev_60 = (
        v_c[
            (v_c["inspectiondate"] >= (T - pd.Timedelta(days=120)))
            & (v_c["inspectiondate"] < (T - pd.Timedelta(days=60)))
        ]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_prev_60d")
    )
    c_90 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_90d")
    )
    c_180 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=180))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_180d")
    )
    c_365 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_365d")
    )
    c_730 = (
        v_c[v_c["inspectiondate"] >= (T - pd.Timedelta(days=730))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_c_count_730d")
    )
    c_1095 = v_c.groupby("clean_bbl").size().rename("feat_c_count_1095d")

    b_90 = (
        v_b[v_b["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_b_count_90d")
    )
    b_365 = (
        v_b[v_b["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_b_count_365d")
    )
    b_1095 = v_b.groupby("clean_bbl").size().rename("feat_b_count_1095d")

    a_365 = (
        v_a[v_a["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_a_count_365d")
    )
    a_1095 = v_a.groupby("clean_bbl").size().rename("feat_a_count_1095d")

    tot_30 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=30))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_30d")
    )
    tot_90 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=90))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_90d")
    )
    tot_365 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=365))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_365d")
    )
    tot_730 = (
        v_3yr[v_3yr["inspectiondate"] >= (T - pd.Timedelta(days=730))]
        .groupby("clean_bbl")
        .size()
        .rename("feat_tot_viol_730d")
    )
    tot_1095 = v_3yr.groupby("clean_bbl").size().rename("feat_tot_viol_1095d")

    last_c_date = v_c.groupby("clean_bbl")["inspectiondate"].max()
    days_since_c = ((T - last_c_date).dt.total_seconds() / 86400.0).rename(
        "feat_days_since_last_c"
    )

    last_tot_date = v_3yr.groupby("clean_bbl")["inspectiondate"].max()
    days_since_any = ((T - last_tot_date).dt.total_seconds() / 86400.0).rename(
        "feat_days_since_last_any"
    )

    if "currentstatus" in v_3yr.columns:
        is_open = (
            v_3yr["currentstatus"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains("OPEN")
        )
        open_c = (
            v_3yr[is_open & (v_3yr["class"] == "C")]
            .groupby("clean_bbl")
            .size()
            .rename("feat_open_c_count")
        )
        open_tot = v_3yr[is_open].groupby("clean_bbl").size().rename("feat_open_total")
    else:
        open_c = pd.Series(0, index=[], name="feat_open_c_count")
        open_tot = pd.Series(0, index=[], name="feat_open_total")

    df = df.set_index("bbl")
    for s in [
        c_30,
        c_60,
        c_prev_60,
        c_90,
        c_180,
        c_365,
        c_730,
        c_1095,
        c_active_years,
        b_90,
        b_365,
        b_1095,
        a_365,
        a_1095,
        tot_30,
        tot_90,
        tot_365,
        tot_730,
        tot_1095,
        open_c,
        open_tot,
        exp_c_90,
        exp_c_365,
        exp_tot_90,
        exp_tot_365,
    ]:
        df = df.join(s, how="left")
        df[s.name] = df[s.name].fillna(0.0).astype(np.float32)

    df = df.join(days_since_c, how="left")
    df["feat_days_since_last_c"] = (
        df["feat_days_since_last_c"].fillna(3650.0).astype(np.float32)
    )

    df = df.join(days_since_any, how="left")
    df["feat_days_since_last_any"] = (
        df["feat_days_since_last_any"].fillna(3650.0).astype(np.float32)
    )

    # Dynamic ratios and velocity interactions computed from uncompressed counts
    df["feat_has_prior_c_365d"] = (df["feat_c_count_365d"] > 0).astype(np.float32)
    df["feat_has_prior_c_1095d"] = (df["feat_c_count_1095d"] > 0).astype(np.float32)
    df["feat_has_prior_any_365d"] = (df["feat_tot_viol_365d"] > 0).astype(np.float32)

    df["feat_c_ratio_365d"] = (
        df["feat_c_count_365d"] / (df["feat_tot_viol_365d"] + 1.0)
    ).astype(np.float32)
    df["feat_b_ratio_365d"] = (
        df["feat_b_count_365d"] / (df["feat_tot_viol_365d"] + 1.0)
    ).astype(np.float32)

    df["feat_open_c_ratio"] = (
        df["feat_open_c_count"] / (df["feat_c_count_1095d"] + 1e-4)
    ).astype(np.float32)
    df["feat_open_tot_ratio"] = (
        df["feat_open_total"] / (df["feat_tot_viol_1095d"] + 1e-4)
    ).astype(np.float32)

    # Winter heating season acceleration (60d window vs prior 60d)
    df["feat_c_heat_diff_60d"] = (
        df["feat_c_count_60d"] - df["feat_c_count_prev_60d"]
    ).astype(np.float32)
    df["feat_c_heat_ratio_60d"] = (
        (df["feat_c_count_60d"] + 1.0) / (df["feat_c_count_prev_60d"] + 1.0)
    ).astype(np.float32)

    # Longitudinal YoY acceleration and difference: [T-365d, T) vs [T-730d, T-365d)
    c_prev_year = np.maximum(df["feat_c_count_730d"] - df["feat_c_count_365d"], 0.0)
    tot_prev_year = np.maximum(df["feat_tot_viol_730d"] - df["feat_tot_viol_365d"], 0.0)

    df["feat_c_consecutive_years"] = (
        (df["feat_c_count_365d"] > 0) & (c_prev_year > 0)
    ).astype(np.float32)

    df["feat_c_yoy_diff"] = (df["feat_c_count_365d"] - c_prev_year).astype(np.float32)
    df["feat_c_yoy_ratio"] = (
        (df["feat_c_count_365d"] + 1.0) / (c_prev_year + 1.0)
    ).astype(np.float32)

    df["feat_tot_yoy_diff"] = (df["feat_tot_viol_365d"] - tot_prev_year).astype(np.float32)
    df["feat_tot_yoy_ratio"] = (
        (df["feat_tot_viol_365d"] + 1.0) / (tot_prev_year + 1.0)
    ).astype(np.float32)

    df["feat_c_velocity_90_365"] = (
        (df["feat_c_count_90d"] * 4.0) / (df["feat_c_count_365d"] + 1.0)
    ).astype(np.float32)
    df["feat_tot_velocity_90_365"] = (
        (df["feat_tot_viol_90d"] * 4.0) / (df["feat_tot_viol_365d"] + 1.0)
    ).astype(np.float32)

    df["feat_severity_index_365d"] = (
        1.0 * df["feat_a_count_365d"]
        + 3.0 * df["feat_b_count_365d"]
        + 6.0 * df["feat_c_count_365d"]
    ).astype(np.float32)
    df["feat_severity_index_90d"] = (
        3.0 * df["feat_b_count_90d"] + 6.0 * df["feat_c_count_90d"]
    ).astype(np.float32)

    units_norm = np.maximum(df["unitsres"], 1.0)
    df["feat_viol_per_unit_365d"] = (df["feat_tot_viol_365d"] / units_norm).astype(np.float32)
    df["feat_c_viol_per_unit_365d"] = (df["feat_c_count_365d"] / units_norm).astype(np.float32)

    # Apply log1p pre-transformation to heavy-tailed count and intensity fields
    count_cols_to_log = [
        "feat_c_count_30d",
        "feat_c_count_60d",
        "feat_c_count_prev_60d",
        "feat_c_count_90d",
        "feat_c_count_180d",
        "feat_c_count_365d",
        "feat_c_count_730d",
        "feat_c_count_1095d",
        "feat_c_active_years_3yr",
        "feat_b_count_90d",
        "feat_b_count_365d",
        "feat_b_count_1095d",
        "feat_a_count_365d",
        "feat_a_count_1095d",
        "feat_tot_viol_30d",
        "feat_tot_viol_90d",
        "feat_tot_viol_365d",
        "feat_tot_viol_730d",
        "feat_tot_viol_1095d",
        "feat_open_c_count",
        "feat_open_total",
        "feat_severity_index_365d",
        "feat_severity_index_90d",
        "feat_c_exp_decay_90d",
        "feat_c_exp_decay_365d",
        "feat_tot_exp_decay_90d",
        "feat_tot_exp_decay_365d",
        "feat_viol_per_unit_365d",
        "feat_c_viol_per_unit_365d",
    ]
    for col in count_cols_to_log:
        df[col] = np.log1p(np.maximum(df[col], 0.0)).astype(np.float32)

    # Community district Class C base rates computed strictly before T
    cd_c_rate = df.groupby("cat_cd")["feat_has_prior_c_365d"].transform("mean")
    cd_c_mean = df.groupby("cat_cd")["feat_c_count_365d"].transform("mean")
    cd_tot_rate = df.groupby("cat_cd")["feat_has_prior_any_365d"].transform("mean")

    df["feat_cd_c_rate_365d"] = cd_c_rate.fillna(0.0).astype(np.float32)
    df["feat_cd_c_mean_365d"] = cd_c_mean.fillna(0.0).astype(np.float32)
    df["feat_cd_tot_rate_365d"] = cd_tot_rate.fillna(0.0).astype(np.float32)

    df = df.reset_index()

    # Empirical Bayes smoothed spatial base rates for tax blocks with leave-one-out self-exclusion
    m_factor = 10.0
    tax_block = df["bbl"].astype(str).str[:6]
    global_prior_rate = float(df["feat_has_prior_c_365d"].mean())
    global_prior_mean = float(df["feat_c_count_365d"].mean())

    block_count = df.groupby(tax_block)["feat_has_prior_c_365d"].transform("count")
    block_c_sum = df.groupby(tax_block)["feat_has_prior_c_365d"].transform("sum")
    block_cnt_sum = df.groupby(tax_block)["feat_c_count_365d"].transform("sum")

    other_count = (block_count - 1).clip(lower=0)
    other_c_sum = (block_c_sum - df["feat_has_prior_c_365d"]).clip(lower=0.0)
    other_cnt_sum = (block_cnt_sum - df["feat_c_count_365d"]).clip(lower=0.0)

    df["feat_block_c_rate_365d"] = (
        (other_c_sum + m_factor * global_prior_rate) / (other_count + m_factor)
    ).astype(np.float32)
    df["feat_block_c_mean_365d"] = (
        (other_cnt_sum + m_factor * global_prior_mean) / (other_count + m_factor)
    ).astype(np.float32)

    cat_cols = ["cat_borough", "cat_cd", "cat_bldgclass"]
    for col in cat_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .clip(lower=0)
        )

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float32)
            )
        else:
            df[col] = np.float32(0.0)

    return df[["bbl"] + cat_cols + FEATURE_COLUMNS]


# -------------------------------------------------------------------------
# 4. Neural Architecture and Loss Definitions
# -------------------------------------------------------------------------
class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss with sample weighting and probability margin clipping on negatives."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-7,
        pos_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        probs = torch.sigmoid(logits)

        loss_pos = (
            -self.pos_weight
            * targets
            * torch.pow(1.0 - probs, self.gamma_pos)
            * torch.log(probs.clamp(min=self.eps))
        )

        p_neg = (probs - self.clip).clamp(min=0.0)
        loss_neg = (
            -(1.0 - targets)
            * torch.pow(p_neg, self.gamma_neg)
            * torch.log((1.0 - p_neg).clamp(min=self.eps))
        )

        loss = loss_pos + loss_neg

        if weights is not None:
            w = weights.view(-1).float()
            loss = loss * w
            if self.reduction == "mean":
                return loss.sum() / (w.sum() + self.eps)
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FeatureTokenizer(nn.Module):
    """Continuous feature projection module with Gated Linear Unit (GLU) activation."""

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.empty(num_features, embed_dim * 2))
        self.bias = nn.Parameter(torch.empty(num_features, embed_dim * 2))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = 1
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        val, gate = projected.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GroupedFeatureTokenizer(nn.Module):
    """Projects continuous feature domain clusters into shared embedding dimension d."""

    def __init__(self, group_dims: list, embed_dim: int):
        super().__init__()
        self.group_dims = group_dims
        self.embed_dim = embed_dim
        self.slices = []
        cur = 0
        for d in group_dims:
            self.slices.append((cur, cur + d))
            cur += d
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, embed_dim * 2),
                nn.GLU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for dim in group_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for proj, (start_idx, end_idx) in zip(self.projections, self.slices):
            g_input = x[:, start_idx:end_idx]
            g_token = proj(g_input).unsqueeze(1)
            tokens.append(g_token)
        return torch.cat(tokens, dim=1)


class FiLMConditioner(nn.Module):
    """Maps pooled categorical embeddings to gamma and beta modulation parameters for continuous tokens."""

    def __init__(self, cat_embed_dim: int, num_tokens: int, token_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat_embed_dim, cat_embed_dim * 2),
            nn.GELU(),
            nn.Linear(cat_embed_dim * 2, num_tokens * token_dim * 2),
        )
        # Identity initialization: gamma=1, beta=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cat_pooled: torch.Tensor, cont_tokens: torch.Tensor) -> torch.Tensor:
        B = cat_pooled.size(0)
        params = self.mlp(cat_pooled).view(B, self.num_tokens, 2, self.token_dim)
        gamma = params[:, :, 0, :] + 1.0
        beta = params[:, :, 1, :]
        return gamma * cont_tokens + beta


class TabularFeatureInteractionNet(nn.Module):
    """Tabular Feature Interaction Network with Grouped Continuous Tokenization,

    Categorical FiLM Conditioning, Multi-Head Self-Attention, Multi-Scale Pooling,
    and Wide Highway.
    """

    def __init__(
        self,
        num_cont_features: int,
        cat_cardinalities: list,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.2,
        feature_groups: list = None,
    ):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.cat_cardinalities = cat_cardinalities
        self.embed_dim = embed_dim

        if feature_groups is None or sum(feature_groups) != num_cont_features:
            base = num_cont_features // 4
            rem = num_cont_features % 4
            feature_groups = [base + (1 if i < rem else 0) for i in range(4)]
        self.feature_groups = feature_groups

        self.input_norm = nn.LayerNorm(num_cont_features)
        self.tokenizer = GroupedFeatureTokenizer(self.feature_groups, embed_dim)

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
            for card in cat_cardinalities
        ])

        self.film = FiLMConditioner(
            cat_embed_dim=embed_dim,
            num_tokens=len(self.feature_groups),
            token_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.pool_norm = nn.LayerNorm(embed_dim)

        num_cat = len(cat_cardinalities)
        wide_in_dim = num_cont_features + num_cat * embed_dim
        self.wide_highway = nn.Sequential(
            nn.Linear(wide_in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        B = x_cont.size(0)
        x_norm = self.input_norm(x_cont)

        cont_tokens = self.tokenizer(x_norm)
        cat_tokens_list = [
            emb(x_cat[:, i]).unsqueeze(1)
            for i, emb in enumerate(self.cat_embeddings)
        ]
        cat_tokens = torch.cat(cat_tokens_list, dim=1)
        cat_pooled = cat_tokens.mean(dim=1)

        cont_tokens_mod = self.film(cat_pooled, cont_tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, cat_tokens, cont_tokens_mod], dim=1)

        tokens = self.transformer(tokens)
        cls_out = self.post_norm(tokens[:, 0, :])
        mean_pool = self.pool_norm(tokens[:, 1:, :].mean(dim=1))

        flat_cat = cat_tokens.view(B, -1)
        wide_in = torch.cat([x_norm, flat_cat], dim=-1)
        wide_out = self.wide_highway(wide_in)

        fused = torch.cat([cls_out, mean_pool, wide_out], dim=-1)
        logits = self.head(fused)
        return logits


def build_model(
    num_cont_features: int,
    cat_cardinalities: list,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    feature_groups: list = None,
) -> nn.Module:
    return TabularFeatureInteractionNet(
        num_cont_features=num_cont_features,
        cat_cardinalities=cat_cardinalities,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        feature_groups=feature_groups,
    )


class CompositeRankingLoss(nn.Module):
    """Composite loss coupling positive-reweighted Asymmetric Focal Loss with online hard-negative margin ranking."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        pos_weight: float = 2.5,
        margin: float = 0.5,
        ranking_weight: float = 0.2,
        max_pairs: int = 4096,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.focal = AsymmetricFocalLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            pos_weight=pos_weight,
            eps=eps,
            reduction="mean",
        )
        self.margin = margin
        self.ranking_weight = ranking_weight
        self.max_pairs = max_pairs

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        focal_loss = self.focal(logits, targets, weights=weights)

        pos_mask = targets > 0.5
        neg_mask = targets <= 0.5

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return focal_loss

        n_pos = pos_logits.numel()
        n_neg = neg_logits.numel()

        # Online Hard-Negative Mining: sort negative logits descending to select hardest false positives
        neg_sorted, _ = torch.sort(neg_logits, descending=True)
        k_neg = min(n_neg, max(1, self.max_pairs // n_pos))
        hard_neg = neg_sorted[:k_neg]

        p_sample = pos_logits.unsqueeze(1).expand(-1, k_neg).reshape(-1)
        n_sample = hard_neg.unsqueeze(0).expand(n_pos, -1).reshape(-1)

        target_ones = torch.ones_like(p_sample)
        rank_loss = F.margin_ranking_loss(
            p_sample, n_sample, target_ones, margin=self.margin, reduction="mean"
        )

        return focal_loss + self.ranking_weight * rank_loss


def build_loss(
    gamma_neg: float = 2.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    pos_weight: float = 2.5,
    margin: float = 0.5,
    ranking_weight: float = 0.2,
) -> nn.Module:
    return CompositeRankingLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
        pos_weight=pos_weight,
        margin=margin,
        ranking_weight=ranking_weight,
    )


def build_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "cls_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    pct_start: float = 0.1,
    min_lr_ratio: float = 1e-2,
):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1.0 / min_lr_ratio,
    )


# -------------------------------------------------------------------------
# 5. Cohort Generation and Feature Pipeline Execution
# -------------------------------------------------------------------------
print("Loading HPD violations dataset...")
df_violations = load_hpd_violations(start_year=2016)
df_violations["inspectiondate"] = pd.to_datetime(
    df_violations["inspectiondate"], errors="coerce"
)
df_violations = df_violations[df_violations["inspectiondate"].notna()].copy()
df_violations["clean_bbl"] = extract_violations_bbl(df_violations)
df_violations = df_violations[
    (df_violations["clean_bbl"].str.len() == 10)
    & (df_violations["clean_bbl"].str.isdigit())
].copy()

# Test Cohort (Cutoff: 2023-01-01)
print("Extracting test cohort features...")
df_test_raw = pd.read_parquet(TEST_ENTITIES_PATH, storage_options=storage_options)
df_test_raw.columns = [c.lower() for c in df_test_raw.columns]
df_test_raw["bbl"] = clean_bbl_series(df_test_raw["bbl"])

if len(df_test_raw.columns) <= 2:
    pluto_22v3 = load_pluto_cohort("22v3")
    pluto_22v3["bbl"] = clean_bbl_series(pluto_22v3["bbl"])
    pluto_22v3 = pluto_22v3[pluto_22v3["unitsres"] >= 3].drop_duplicates(subset=["bbl"])
    df_test_entities = df_test_raw[["bbl"]].merge(pluto_22v3, on="bbl", how="left")
else:
    df_test_entities = df_test_raw

test_features = extract_features(
    df_test_entities, df_violations, cutoff_timestamp="2023-01-01"
)

# Validation Cohort (Cutoff: 2022-01-01, Release: 21v4)
print("Extracting validation cohort (2022) features...")
pluto_21v4 = load_pluto_cohort("21v4")
pluto_21v4["bbl"] = clean_bbl_series(pluto_21v4["bbl"])
df_val_entities = (
    pluto_21v4[pluto_21v4["unitsres"] >= 3]
    .drop_duplicates(subset=["bbl"])
    .reset_index(drop=True)
)

val_features = extract_features(
    df_val_entities, df_violations, cutoff_timestamp="2022-01-01"
)
val_label_mask = (
    (df_violations["class"] == "C")
    & (df_violations["inspectiondate"] >= "2022-01-01")
    & (df_violations["inspectiondate"] < "2023-01-01")
)
val_positive_bbls = set(df_violations.loc[val_label_mask, "clean_bbl"].unique())
val_features["target"] = val_features["bbl"].isin(val_positive_bbls).astype(np.int32)

# Training Cohort 2021 (Cutoff: 2021-01-01, Release: 20v7)
print("Extracting training cohort 2021 features...")
pluto_20v7 = load_pluto_cohort("20v7")
pluto_20v7["bbl"] = clean_bbl_series(pluto_20v7["bbl"])
df_train_entities_2021 = (
    pluto_20v7[pluto_20v7["unitsres"] >= 3]
    .drop_duplicates(subset=["bbl"])
    .reset_index(drop=True)
)

train_features_2021 = extract_features(
    df_train_entities_2021, df_violations, cutoff_timestamp="2021-01-01"
)
train_label_mask_2021 = (
    (df_violations["class"] == "C")
    & (df_violations["inspectiondate"] >= "2021-01-01")
    & (df_violations["inspectiondate"] < "2022-01-01")
)
train_positive_bbls_2021 = set(df_violations.loc[train_label_mask_2021, "clean_bbl"].unique())
train_features_2021["target"] = (
    train_features_2021["bbl"].isin(train_positive_bbls_2021).astype(np.int32)
)
# Temporal cohort sample weighting: 1.0 for 2021
train_features_2021["weight"] = 1.0

# Training Cohort 2020 (Cutoff: 2020-01-01, Release: 19v2)
print("Extracting training cohort 2020 features...")
pluto_19v2 = load_pluto_cohort("19v2")
pluto_19v2["bbl"] = clean_bbl_series(pluto_19v2["bbl"])
df_train_entities_2020 = (
    pluto_19v2[pluto_19v2["unitsres"] >= 3]
    .drop_duplicates(subset=["bbl"])
    .reset_index(drop=True)
)

train_features_2020 = extract_features(
    df_train_entities_2020, df_violations, cutoff_timestamp="2020-01-01"
)
train_label_mask_2020 = (
    (df_violations["class"] == "C")
    & (df_violations["inspectiondate"] >= "2020-01-01")
    & (df_violations["inspectiondate"] < "2021-01-01")
)
train_positive_bbls_2020 = set(df_violations.loc[train_label_mask_2020, "clean_bbl"].unique())
train_features_2020["target"] = (
    train_features_2020["bbl"].isin(train_positive_bbls_2020).astype(np.int32)
)
# Temporal cohort sample weighting: 0.5 for 2020 pandemic cohort
train_features_2020["weight"] = 0.5

# Pool multi-cohort training data
print("Pooling multi-cohort training datasets...")
train_features = pd.concat([train_features_2020, train_features_2021], ignore_index=True)
print(f"Total pooled training records: {len(train_features)}")

cat_columns = ["cat_borough", "cat_cd", "cat_bldgclass"]
all_cds_count = 71
cat_cardinalities = [6, all_cds_count + 1, 27]

feature_columns = [
    c
    for c in FEATURE_COLUMNS
    if c in train_features.columns
]
num_cont_features = len(feature_columns)
print(f"Continuous features: {num_cont_features}, Categorical columns: {len(cat_columns)}")

# -------------------------------------------------------------------------
# 6. Data Scaling and PyTorch DataLoaders
# -------------------------------------------------------------------------
scaler = StandardScaler()
X_train_cont_raw = train_features[feature_columns].values.astype(np.float32)
X_val_cont_raw = val_features[feature_columns].values.astype(np.float32)
X_test_cont_raw = test_features[feature_columns].values.astype(np.float32)

X_train_cont = scaler.fit_transform(X_train_cont_raw)
X_val_cont = scaler.transform(X_val_cont_raw)
X_test_cont = scaler.transform(X_test_cont_raw)

np.nan_to_num(X_train_cont, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_val_cont, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test_cont, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_train_cat = train_features[cat_columns].values.astype(np.int64)
X_val_cat = val_features[cat_columns].values.astype(np.int64)
X_test_cat = test_features[cat_columns].values.astype(np.int64)

for i, card in enumerate(cat_cardinalities):
    X_train_cat[:, i] = np.clip(X_train_cat[:, i], 0, card - 1)
    X_val_cat[:, i] = np.clip(X_val_cat[:, i], 0, card - 1)
    X_test_cat[:, i] = np.clip(X_test_cat[:, i], 0, card - 1)

y_train = train_features["target"].values.astype(np.float32)
w_train = train_features["weight"].values.astype(np.float32)
y_val = val_features["target"].values.astype(np.float32)

batch_size = 2048
train_dataset = TensorDataset(
    torch.from_numpy(X_train_cont),
    torch.from_numpy(X_train_cat),
    torch.from_numpy(y_train),
    torch.from_numpy(w_train),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_cont),
    torch.from_numpy(X_val_cat),
    torch.from_numpy(y_val),
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_cont),
    torch.from_numpy(X_test_cat),
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)

# -------------------------------------------------------------------------
# 7. Model Training and Validation Loop
# -------------------------------------------------------------------------
epochs = 12
patience = 4

model = build_model(
    num_cont_features=num_cont_features,
    cat_cardinalities=cat_cardinalities,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    feature_groups=GROUP_DIMS,
).to(device)

criterion = build_loss(
    gamma_neg=2.0,
    gamma_pos=1.0,
    clip=0.05,
    pos_weight=2.5,
    margin=0.5,
    ranking_weight=0.2,
).to(device)

total_steps = epochs * len(train_loader)
optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
scheduler = build_scheduler(
    optimizer, total_steps=total_steps, pct_start=0.1, min_lr_ratio=1e-2
)

best_val_ap = -1.0
best_model_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for xb_cont, xb_cat, yb, wb in train_loader:
        xb_cont = xb_cont.to(device, non_blocking=True)
        xb_cat = xb_cat.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb_cont, xb_cat).squeeze(-1)
        loss = criterion(logits, yb, weights=wb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * len(yb)
        total_samples += len(yb)

    train_loss = running_loss / total_samples

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for xb_cont, xb_cat, _ in val_loader:
            xb_cont = xb_cont.to(device, non_blocking=True)
            xb_cat = xb_cat.to(device, non_blocking=True)
            logits = model(xb_cont, xb_cat).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds_list.append(probs)

    val_preds = np.concatenate(val_preds_list)
    val_ap = average_precision_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val AP: {val_ap:.4f} - Val AUC: {val_auc:.4f}"
    )

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# -------------------------------------------------------------------------
# 8. Final Hold-Out Evaluation and Submission Generation
# -------------------------------------------------------------------------
model.load_state_dict(best_model_state)
model.eval()

final_val_preds_list = []
with torch.no_grad():
    for xb_cont, xb_cat, _ in val_loader:
        xb_cont = xb_cont.to(device, non_blocking=True)
        xb_cat = xb_cat.to(device, non_blocking=True)
        logits = model(xb_cont, xb_cat).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        final_val_preds_list.append(probs)

final_val_preds = np.concatenate(final_val_preds_list)
final_val_ap = average_precision_score(y_val, final_val_preds)

# Full test inference pass
test_preds_list = []
with torch.no_grad():
    for xb_cont, xb_cat in test_loader:
        xb_cont = xb_cont.to(device, non_blocking=True)
        xb_cat = xb_cat.to(device, non_blocking=True)
        logits = model(xb_cont, xb_cat).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds_list.append(probs)

test_preds = np.concatenate(test_preds_list)

submission_df = pd.DataFrame(
    {
        "bbl": test_features["bbl"].astype(str).str.zfill(10),
        "score": test_preds.astype(float),
    }
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file not found!"
assert len(submission_df) == len(
    test_features
), f"Expected {len(test_features)} rows, got {len(submission_df)}"
assert submission_df["bbl"].str.len().eq(10).all(), "All BBLs must be 10 digits"
assert submission_df["score"].notna().all(), "Scores must not contain NaN or nulls"

print(f"Final Validation Score: {final_val_ap}")

"""Label-free observables per domain, for the shift analyses (explorations 2+).

Same construction as data_exploration_2.py section (A), as an importable
function so later shift/audit scripts rebuild it identically. Computed in
memory from input/ on every call; nothing is written to disk.

Every label-derived observable uses OTHER domains' labels only (no self loops in
the link graph; the same-site count subtracts the row itself), so it is equally
defined for tracked corpus rows and for target rows.
"""
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from common import WS_ROOT

IN = WS_ROOT / "input"
SECOND_LEVEL = ["co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "go", "gv", "nic", "ltd",
                "plc", "sch", "info", "biz"]


def build():
    dom = pl.read_parquet(IN / "domains.parquet")
    N = dom.height
    tg = pl.read_parquet(IN / "tracking_graph_train.parquet", columns=["domain_id", "tracker_id"])
    tgt_ids = pl.read_csv(IN / "target.tsv", separator="\t")["domain_id"].to_numpy()
    d_arr = tg["domain_id"].to_numpy(); t_arr = tg["tracker_id"].to_numpy()
    count = np.bincount(d_arr, minlength=N).astype(np.float32)
    tracked = (count > 0).astype(np.float32)
    has = {k: np.zeros(N, np.float32) for k in ("ga", "ads", "fb")}
    for k, tid in (("ga", 129), ("ads", 131), ("fb", 104)):
        has[k][d_arr[t_arr == tid]] = 1
    is_target = np.zeros(N, bool); is_target[tgt_ids] = True

    lg = pl.read_parquet(IN / "link-graph.parquet")
    src = lg["source_domain_id"].to_numpy(); dst = lg["target_domain_id"].to_numpy()
    A = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))
    AT = A.T.tocsr()
    del lg, src, dst

    feat = {"out_deg": np.diff(A.indptr).astype(np.float32), "in_deg": np.diff(AT.indptr).astype(np.float32)}
    for direction, M in (("out", A), ("in", AT)):
        lab = M @ tracked
        deg = feat[f"{direction}_deg"]
        feat[f"{direction}_lab"] = lab
        feat[f"{direction}_lab_frac"] = np.where(deg > 0, lab / np.maximum(deg, 1), np.nan)
        feat[f"{direction}_nbr_cnt_mean"] = np.where(lab > 0, (M @ count) / np.maximum(lab, 1), np.nan)
        for k in has:
            feat[f"{direction}_nbr_{k}"] = np.where(lab > 0, (M @ has[k]) / np.maximum(lab, 1), np.nan)

    host = dom.sort("domain_id")["domain"]
    parts = pl.DataFrame({"h": host}).select(
        pl.col("h").str.len_chars().alias("len"),
        pl.col("h").str.count_matches(r"\.").alias("dots"),
        pl.col("h").str.count_matches(r"[0-9]").alias("digits"),
        pl.col("h").str.count_matches("-").alias("hyph"),
        pl.col("h").str.split(".").alias("lbl"),
    ).with_columns(
        pl.col("lbl").list.last().alias("tld"),
        pl.col("lbl").list.get(-2, null_on_oob=True).alias("sld"),
    )
    parts = parts.with_columns(
        pl.when(pl.col("sld").is_in(SECOND_LEVEL) & (pl.col("tld").str.len_chars() == 2) & (pl.col("lbl").list.len() >= 3))
        .then(pl.col("lbl").list.slice(-3, 3).list.join("."))
        .otherwise(pl.col("lbl").list.slice(-2, 2).list.join(".")).alias("site"),
    )
    parts = parts.with_columns(
        (pl.col("dots") - pl.col("site").str.count_matches(r"\.")).alias("sub_depth"),
        pl.Series("tracked", tracked),
    )
    site_agg = parts.group_by("site").agg(pl.len().alias("site_n"), pl.col("tracked").sum().alias("site_tracked"))
    parts = parts.join(site_agg, on="site", how="left", maintain_order="left")
    for c in ("len", "dots", "digits", "hyph", "sub_depth"):
        feat[c] = parts[c].to_numpy().astype(np.float32)
    feat["site_n_others"] = parts["site_n"].to_numpy().astype(np.float32) - 1
    feat["site_tracked_others"] = (parts["site_tracked"].to_numpy() - tracked).astype(np.float32)
    tld_s = parts["tld"].to_pandas()
    top_tld = tld_s.value_counts().index[:150]
    feat_tld = pd.Categorical(tld_s.where(tld_s.isin(top_tld), "other"))

    urlc = pl.read_csv(IN / "url-classification.csv")
    uh = urlc.with_columns(pl.col("url").str.extract(r"^[a-z]+://([^/:]+)", 1).str.to_lowercase()
                           .str.replace(r"^www\.", "").alias("h")).unique("h")
    m = pl.DataFrame({"h": host.str.replace(r"^www\.", "")}).join(
        uh.select("h", "category"), on="h", how="left", maintain_order="left")
    feat_urlcat = pd.Categorical(m["category"].fill_null("none").to_pandas())

    def frame(idx):
        df = pd.DataFrame({k: v[idx] for k, v in feat.items()})
        df["tld"] = feat_tld[idx]
        df["urlcat"] = feat_urlcat[idx]
        return df

    return dict(N=N, count=count, tracked=tracked, is_target=is_target, tgt_ids=tgt_ids,
                frame=frame, tld=feat_tld, d_arr=d_arr, t_arr=t_arr)

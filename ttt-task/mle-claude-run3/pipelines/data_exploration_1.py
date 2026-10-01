"""Exploration 1 — profile the raw pile: sizes, id spaces, overlaps, label mix.

Read-only: prints a report, writes nothing.
"""
import time

import numpy as np
import pandas as pd
import polars as pl

from common import WS_ROOT

IN = WS_ROOT / "input"
t0 = time.time()

tg = pl.read_parquet(IN / "tracking_graph_train.parquet")
dom = pl.read_parquet(IN / "domains.parquet")
tgt = pl.read_csv(IN / "target.tsv", separator="\t")
trk = pl.read_csv(IN / "trackers.tsv", separator="\t")
urlc = pl.read_csv(IN / "url-classification.csv")
fop = pl.read_csv(IN / "freedom-of-the-press.csv")
print(f"loaded small tables in {time.time()-t0:.0f}s")

print("\n== domains.parquet", dom.shape, "id range", dom["domain_id"].min(), dom["domain_id"].max(),
      "unique ids", dom["domain_id"].n_unique())
print(dom.head(5))
print("\n== trackers.tsv", trk.shape)
print(trk.head(8))
print(trk["category"].value_counts().sort("count", descending=True))
print("\n== url-classification", urlc.shape)
print(urlc.head(5))
print(urlc["category"].value_counts().sort("count", descending=True).head(15))
print("\n== freedom-of-the-press", fop.shape)
print(fop.head(5))

print("\n== tracking graph", tg.shape)
per_dom = tg.group_by("domain_id").len()
print("train domains", per_dom.height, "id range", tg["domain_id"].min(), tg["domain_id"].max())
print("tracker ids in graph", tg["tracker_id"].n_unique(), "; tracking_domain_id==tracker map 1:1?",
      tg.select(pl.col("tracking_domain_id").n_unique()).item())
dup = tg.height - tg.unique(["domain_id", "tracker_id"]).height
print("duplicate (domain,tracker) pairs:", dup)
pop = tg.group_by("tracker_id").len().sort("len", descending=True)
pop = pop.with_columns((pl.col("len") / per_dom.height).alias("frac_domains"))
print("tracker popularity (top 15):")
print(pop.join(trk.select("tracker_id", "domain", "company", "category"), on="tracker_id").head(15))
cum = np.cumsum(pop["len"].to_numpy()) / tg.height
print("top-10 trackers cover", round(cum[9], 4), "of all pairs; top-30", round(cum[29], 4))

# global-popularity recall@10 on the train corpus (trivial baseline level)
top10 = set(pop["tracker_id"].head(10).to_list())
hit = tg.with_columns(pl.col("tracker_id").is_in(list(top10)).alias("hit")).group_by("domain_id").agg(
    pl.col("hit").mean().alias("r"))
print("in-corpus recall@10 of global top-10:", round(hit["r"].mean(), 4))

print("\n== target", tgt.shape, "id range", tgt["domain_id"].min(), tgt["domain_id"].max())
tids = tgt["domain_id"]
print("target in domains.parquet:", tids.is_in(dom["domain_id"]).sum())
print("target in train:", tids.is_in(per_dom["domain_id"]).sum())

# where do the three id populations sit in id space?
def q(s):
    return np.quantile(s.to_numpy(), [0, .1, .25, .5, .75, .9, 1]).astype(int)
print("id quantiles  all domains:", q(dom["domain_id"]))
print("id quantiles  train      :", q(per_dom["domain_id"]))
print("id quantiles  target     :", q(tids))
untracked = dom.filter(~pl.col("domain_id").is_in(per_dom["domain_id"]) & ~pl.col("domain_id").is_in(tids))
print("untracked (not train, not target):", untracked.height, "id quantiles", q(untracked["domain_id"]))

# hostnames of target vs train samples
tgt_h = tgt.join(dom, on="domain_id", how="left")
print("\ntarget hostnames sample:\n", tgt_h.sample(15, seed=0)["domain"].to_list())
print("train hostnames sample:\n", per_dom.sample(15, seed=0).join(dom, on="domain_id")["domain"].to_list())
print("untracked hostnames sample:\n", untracked.sample(15, seed=0)["domain"].to_list())

# link graph
t1 = time.time()
lg = pl.read_parquet(IN / "link-graph.parquet")
print(f"\n== link graph {lg.shape} loaded in {time.time()-t1:.0f}s")
print("self loops:", lg.filter(pl.col("source_domain_id") == pl.col("target_domain_id")).height)
outd = lg.group_by("source_domain_id").len()
ind = lg.group_by("target_domain_id").len()
print("domains with out-links", outd.height, "with in-links", ind.height)
for name, ids in [("train", per_dom["domain_id"]), ("target", tids),
                  ("untracked", untracked["domain_id"].sample(200_000, seed=0))]:
    o = pl.DataFrame({"domain_id": ids.cast(pl.Int64)}).join(
        outd.rename({"source_domain_id": "domain_id", "len": "out"}).with_columns(pl.col("domain_id").cast(pl.Int64)),
        on="domain_id", how="left").join(
        ind.rename({"target_domain_id": "domain_id", "len": "in"}).with_columns(pl.col("domain_id").cast(pl.Int64)),
        on="domain_id", how="left").fill_null(0)
    print(f"{name:10s} n={o.height:>9}  has_out={np.mean(o['out'].to_numpy()>0):.3f} "
          f"has_in={np.mean(o['in'].to_numpy()>0):.3f}  median out={np.median(o['out'].to_numpy())} "
          f"in={np.median(o['in'].to_numpy())}  mean out={o['out'].mean():.1f} in={o['in'].mean():.1f}")
print(f"done in {time.time()-t0:.0f}s")

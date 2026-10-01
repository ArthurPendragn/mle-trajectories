"""Exploration 9 — residual analysis: what does the ranker miss, and is it fixable?

Model = pipeline_11's single-seed config (softmax, 16 ep, 2x2048, dropout 0.3)
on pipeline_06's blocks, fit on 600k S rows, evaluated on 200k other S rows.
For every true (domain, tracker) pair NOT in the top 10:
  - its rank under the model (11-20 = near miss, ranking fixable; >50 = hopeless)
  - whether ANY block carries evidence for that tracker on that row
  - the tracker's popularity rank and category
Read-only: prints, writes nothing.
"""
import numpy as np
import pandas as pd

import common
from rankers import TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")
tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
hosts = common.read_hosts(common.IN / "domains.parquet")
n = len(hosts)
graph = common.read_graph(common.IN / "link-graph.parquet", n)
tdom = common.read_tracker_domains(common.IN / "trackers.tsv")
trk = pd.read_csv(common.IN / "trackers.tsv", sep="\t").sort_values("tracker_id")
S = common.sample_domains(tg); E = common.shadow_domains(tg)
pool = common.build_pool(tg, S, E, n)
rng = np.random.default_rng(0)
ids = np.sort(rng.choice(S, 800_000, replace=False))
rows = common.build_rows(tg, ids); Y = rows[common.Y_COLS].to_numpy(bool)
X = rows[["domain_id"]]
parts = []
for b in BLOCKS:
    if b in ("out", "in"):
        parts.append(common.nbr_block(X, graph, pool, b))
    elif b == "host":
        parts.append(common.host_block(X, hosts))
    elif b == "direct":
        parts.append(common.direct_block(X, graph, tdom))
    else:
        f, s = (b[:2], b[2:]) if b.startswith("in") else (b[:3], b[3:])
        parts.append(common.twohop_block(X, graph, pool, f, s))
F = pd.concat(parts, axis=1)
perm = rng.permutation(len(ids)); tr, te = perm[:600_000], perm[600_000:]
m = TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, device="cuda").fit(F.iloc[tr], Y[tr])
Sc = m.predict(F.iloc[te]); Yte = Y[te]
print("eval recall@10:", round(common.recall_at_10(Yte, Sc), 4))

rank = np.argsort(np.argsort(-Sc, axis=1), axis=1)          # 0 = best
evid = np.zeros_like(Yte)
cols = F.columns
for b in ("out", "in", "inout", "outin", "outout", "inin", "dl"):
    ix = [i for i, c in enumerate(cols) if c.startswith(b + "_t")]
    if len(ix) == 355:
        evid |= F.iloc[te].to_numpy(np.float32)[:, ix] > 0
true = Yte
missed = true & (rank >= 10)
print(f"true pairs {true.sum()}, missed {missed.sum()} ({missed.sum()/true.sum():.3f})")
r = rank[missed]
for lo, hi in ((10, 15), (15, 20), (20, 50), (50, 100), (100, 355)):
    print(f"  missed at rank {lo+1:>3}-{hi:<3}: {np.mean((r >= lo) & (r < hi)):.3f}")
print(f"  missed pairs with NO block evidence: {np.mean(~evid[missed]):.3f}   (hit pairs: {np.mean(~evid[true & (rank < 10)]):.3f})")
# domains where hits were capped by >10 trackers
c = true.sum(1)
print(f"  share of misses on domains with >10 trackers (unavoidable part): {missed[c > 10].sum()/missed.sum():.3f}")
pop_rank = np.argsort(np.argsort(-Y[tr].mean(0)))
mt = missed.sum(0)
df = trk.assign(missed=mt, true=true.sum(0), miss_rate=mt / np.maximum(true.sum(0), 1), pop_rank=pop_rank)
print("\nmisses by tracker popularity rank:")
for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 355)):
    sel = (df.pop_rank >= lo) & (df.pop_rank < hi)
    print(f"  pop rank {lo+1:>3}-{hi:<3}: share of misses {df.missed[sel].sum()/mt.sum():.3f}, miss rate {df.missed[sel].sum()/max(df.true[sel].sum(),1):.3f}")
print("\ntop 12 trackers by missed pairs:")
print(df.sort_values("missed", ascending=False).head(12)[["tracker_id", "domain", "category", "true", "missed", "miss_rate", "pop_rank"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nmisses by category:", df.groupby("category").missed.sum().div(mt.sum()).round(3).to_dict())

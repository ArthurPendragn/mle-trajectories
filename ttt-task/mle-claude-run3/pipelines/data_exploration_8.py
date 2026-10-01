"""Exploration 8 — deployment choice: which label pool should PREDICTION-time features read?

The plans train on S with features from P = corpus - S - E. For the target
(disjoint from the whole corpus) the final model could instead read labels
from the full corpus, i.e. also S and E: ~14% more label mass, but a mild
train/predict parity break (the model was fit on P-coverage features).

Measured on the shadow set E (disjoint from S and from P): one deployed model
fit on all of S (pipeline_06 blocks, P features), then E scored with
  (a) features from P            (parity, what the CV measures)
  (b) features from P + S        (richer pool; E's own labels still excluded)
(b) - (a) estimates what using S labels at predict time buys the target.
Read-only: prints, writes nothing.
"""
import time

import numpy as np
import pandas as pd

import common
from rankers import TorchMLPRanker

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")
tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
hosts = common.read_hosts(common.IN / "domains.parquet")
n = len(hosts)
graph = common.read_graph(common.IN / "link-graph.parquet", n)
tdom = common.read_tracker_domains(common.IN / "trackers.tsv")
S = common.sample_domains(tg); E = common.shadow_domains(tg)
pool_P = common.build_pool(tg, S, E, n)
pool_PS = common.build_pool(tg, np.array([], dtype=np.int64), E, n)     # excludes only E
log("pools built; P pairs", pool_P["L"].nnz, "P+S pairs", pool_PS["L"].nnz)

def feats(ids, pool):
    X = pd.DataFrame({"domain_id": ids})
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
    return pd.concat(parts, axis=1)

rowsS = common.build_rows(tg, S); YS = rowsS[common.Y_COLS].to_numpy(np.float32)
FS = feats(S, pool_P); log("S features", FS.shape)
m = TorchMLPRanker(epochs=8, device="cuda").fit(FS, YS); log("deployed model fit")
del FS
rowsE = common.build_rows(tg, E); YE = rowsE[common.Y_COLS].to_numpy(bool)
cE = YE.sum(1)
for name, pool in (("(a) P    ", pool_P), ("(b) P + S", pool_PS)):
    r = common.per_domain_recall_at_10(YE, m.predict(feats(E, pool)))
    print(f"  {name}: E all {r.mean():.5f} | E count>=2 {r[cE >= 2].mean():.5f} | E count==1 {r[cE == 1].mean():.5f}",
          flush=True)
log("done")

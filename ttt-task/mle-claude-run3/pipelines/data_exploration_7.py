"""Exploration 7 — does the hostname TEXT carry tracker signal beyond the TLD?

Small MLPs on a 400k-domain slice of S (300k fit / 100k eval), label-free inputs:
  A  host block only (shape + top-60 TLD one-hot)          [what the plans use]
  B  hashed char 3-grams of the hostname (512 dims, l2)    [candidate block]
  C  A + B
  D  pipeline-style out+outin blocks        (reference: graph signal)
  E  D + B                                  (marginal value on top of graph)
Recall@10 on the eval slice. Read-only: prints, writes nothing.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

import common
from rankers import TorchMLPRanker

tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
hosts = common.read_hosts(common.IN / "domains.parquet")
n = len(hosts)
graph = common.read_graph(common.IN / "link-graph.parquet", n)
S = common.sample_domains(tg); E = common.shadow_domains(tg)
pool = common.build_pool(tg, S, E, n)
rng = np.random.default_rng(0)
ids = np.sort(rng.choice(S, 400_000, replace=False))
rows = common.build_rows(tg, ids)
Y = rows[common.Y_COLS].to_numpy(np.float32)
X = rows[["domain_id"]]
perm = rng.permutation(len(ids)); tr, te = perm[:300_000], perm[300_000:]

host = common.host_block(X, hosts)
names = hosts.gather(ids).str.replace(r"\.[^.]+$", "").to_list()        # hostname minus TLD
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=512, alternate_sign=False, norm="l2")
ng = pd.DataFrame(hv.transform(names).toarray().astype(np.float32), columns=[f"ng{i}" for i in range(512)])
graphF = pd.concat([common.nbr_block(X, graph, pool, "out"), common.twohop_block(X, graph, pool, "out", "in")], axis=1)
graphF.index = range(len(ids)); host.index = range(len(ids))

def run(name, F):
    F = np.asarray(F, dtype=np.float32)
    m = TorchMLPRanker(epochs=8, device="cuda").fit(F[tr], Y[tr])
    print(f"  {name:28s} dims {F.shape[1]:5d}  recall@10 {common.recall_at_10(Y[te], m.predict(F[te])):.4f}", flush=True)

print("popularity:", round(common.recall_at_10(Y[te], np.tile(Y[tr].mean(0), (len(te), 1))), 4))
run("A host/TLD", host)
run("B char 3-grams", ng)
run("C host/TLD + 3-grams", pd.concat([host, ng], axis=1))
run("D out + outin", graphF)
run("E out + outin + 3-grams", pd.concat([graphF, ng], axis=1))
run("F out + outin + host", pd.concat([graphF, host], axis=1))
run("G out + outin + host + 3-grams", pd.concat([graphF, host, ng], axis=1))

"""Exploration 5 — reusable leakage audit for feature blocks.

    python data_exploration_5.py out in host [site 2hop ...]

For each block (built by the SAME common.py code the plans use, from the same
label pool P):
  (1) standalone score: rank trackers by the block's 355 tracker columns alone
      (no model) on the S rows the block covers -> Recall@10. Near-perfect on
      covered rows = the block carries the answer.
  (2) the same standalone score on shadow E (disjoint rows) -- a leak through
      S's own labels would show up as S >> E at equal count mix, so E is also
      re-weighted to S's count mix for the comparison.
  (3) coverage / mass parity between S, the target and E: share of rows with
      any signal, median mass, and the per-row sum of the tracker columns.
Read-only: prints, writes nothing.
"""
import sys
import time

import numpy as np
import pandas as pd

import common

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

blocks = sys.argv[1:] or ["out", "in", "host"]
tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
hosts = common.read_hosts(common.IN / "domains.parquet")
n = len(hosts)
graph = common.read_graph(common.IN / "link-graph.parquet", n)
S = common.sample_domains(tg); E = common.shadow_domains(tg)
pool = common.build_pool(tg, S, E, n)
tgt = pd.read_csv(common.IN / "target.tsv", sep="\t")["domain_id"].to_numpy()
cnt = np.bincount(tg["domain_id"].to_numpy(), minlength=n)
rng = np.random.default_rng(0)
S_sub = np.sort(rng.choice(S, 300_000, replace=False))
rowsS = common.build_rows(tg, S_sub); YS = rowsS[common.Y_COLS].to_numpy(bool)
rowsE = common.build_rows(tg, E); YE = rowsE[common.Y_COLS].to_numpy(bool)
E2 = cnt[E] >= 2                           # E restricted to S's support (count >= 2)
log("context built; S_sub", len(S_sub), "E", len(E), "E>=2", E2.sum(), "target", len(tgt))

urlcat = common.read_urlcat(common.IN / "url-classification.csv")
BUILDERS = {
    "outc": lambda X: common.nbr_count_block(X, graph, pool, "out"),
    "inc": lambda X: common.nbr_count_block(X, graph, pool, "in"),
    "meta": lambda X: common.meta_block(X, hosts, pool, urlcat),
    "out": lambda X: common.nbr_block(X, graph, pool, "out"),
    "in": lambda X: common.nbr_block(X, graph, pool, "in"),
    "host": lambda X: common.host_block(X, hosts),
}
for b2 in ("inout", "outin", "outout", "inin"):
    f_, s_ = (b2[:2], b2[2:]) if b2.startswith("in") else (b2[:3], b2[3:])
    BUILDERS[b2] = (lambda f_, s_: (lambda X: common.twohop_block(X, graph, pool, f_, s_)))(f_, s_)
    BUILDERS[b2 + "idf"] = (lambda f_, s_: (lambda X: common.twohop_block(X, graph, pool, f_, s_, "idf")))(f_, s_)

def tracker_cols(df, b):
    cols = [c for c in df.columns if c.endswith(tuple(common.Y_COLS)) and c.split("_")[-1] in common.Y_COLS]
    return df[cols].to_numpy(np.float32) if len(cols) == common.N_TRACKERS else None

for b in blocks:
    fS = BUILDERS[b](pd.DataFrame({"domain_id": S_sub}))
    fE = BUILDERS[b](pd.DataFrame({"domain_id": E}))
    fT = BUILDERS[b](pd.DataFrame({"domain_id": tgt}))
    print(f"\n=== block {b}: {fS.shape[1]} columns")
    MS, ME, MT = tracker_cols(fS, b), tracker_cols(fE, b), tracker_cols(fT, b)
    if MS is not None:
        covS, covE, covT = MS.sum(1) > 0, ME.sum(1) > 0, MT.sum(1) > 0
        rS = common.per_domain_recall_at_10(YS[covS], MS[covS])
        rE = common.per_domain_recall_at_10(YE[covE & E2], ME[covE & E2])
        print(f"  (1) standalone Recall@10 on covered S rows      : {rS.mean():.4f}  (n={covS.sum()})")
        print(f"  (2) standalone Recall@10 on covered E rows (>=2): {rE.mean():.4f}  (n={(covE & E2).sum()})"
              f"  -> S-E gap {rS.mean()-rE.mean():+.4f}")
        # E re-weighted to S's count mix (by exact count, capped at 10)
        cS = np.minimum(YS[covS].sum(1), 11); cE = np.minimum(YE[covE & E2].sum(1), 11)
        wS = np.bincount(cS, minlength=12) / len(cS)
        rE_by = np.array([rE[cE == k].mean() if (cE == k).any() else np.nan for k in range(12)])
        print(f"      E re-weighted to S count mix                 : {np.nansum(wS * rE_by):.4f}")
        for nm, M, cov in (("S", MS, covS), ("E", ME, covE), ("target", MT, covT)):
            mass = M[cov].sum(1)
            print(f"  (3) coverage {nm:6s}: {cov.mean():.4f}   mass median {np.median(mass):.3f} "
                  f"mean {mass.mean():.3f}   n distinct trackers/row med {np.median((M[cov] > 0).sum(1)):.0f}")
    num = fS.select_dtypes("number")
    for c in [c for c in num.columns if not (c.split("_")[-1] in common.Y_COLS)][:12]:
        print(f"      {c:22s} S mean {fS[c].mean():9.4f} | E mean {fE[c].mean():9.4f} | target mean {fT[c].mean():9.4f}")
    log(f"block {b} done")

"""FINAL refit/predict artifact for pipeline_16 -- NOT a scored candidate (never run it through ml-score).

Refits pipeline_16's plan (same feature blocks, same 15-MLP ensemble) once on
ALL of the modelled sample S (tracked domains with >= 2 trackers, 5M pairs),
then ranks the 355 trackers for every domain in input/target.tsv and writes
the top 10 per domain to <workspace>/submission.tsv
(columns domain_id, tracking_domain_id).

Unlike the scored plans (rooted on constants), X, y and the LABEL POOL are
overridable skrub.var nodes:
  fit     : X/y = S, pool = P = corpus - S - E   (exactly the scored setup)
  predict : X = target domains, pool = FULL corpus (P + S + E). The target is
            disjoint from the corpus, so no row sees its own labels; this is the
            deliberate, measured parity break of README section 7 /
            data_exploration_8 (+0.0008 on shadow E). The parity-exact
            prediction (pool = P) is also computed and its overlap reported.
"""
import time

import numpy as np
import pandas as pd
import skrub

import common
from rankers import BlendRanker, DropColumns, TorchMLPRanker

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

WS_ROOT = common.WS_ROOT
BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin", "outc", "inc")  # = pipeline_16
NO_COUNTS = ("outc_", "inc_")

# --- data for the variables (same helpers the scored plans use) -------------
tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
n_dom = len(common.read_hosts(common.IN / "domains.parquet"))
S = common.sample_domains(tg)
E = common.shadow_domains(tg)
rows = common.build_rows(tg, S)
pool_P = common.build_pool(tg, S, E, n_dom)                                        # training-time pool
pool_full = common.build_pool(tg, np.array([], np.int64), np.array([], np.int64), n_dom)  # all corpus labels
target = pd.read_csv(common.IN / "target.tsv", sep="\t")[["domain_id"]]
assert not target["domain_id"].isin(tg["domain_id"]).any(), "target overlaps the labelled corpus"
log(f"S {len(S)} domains; pool P {pool_P['L'].nnz} pairs; full pool {pool_full['L'].nnz} pairs; target {len(target)}")

# --- the plan: pipeline_16 with X / y / pool as variables --------------------
with skrub.config_context(eager_data_ops=False):
    X = skrub.var("X", rows[["domain_id"]]).skb.mark_as_X()
    y = skrub.var("y", rows[common.Y_COLS]).skb.mark_as_y()
    hosts = skrub.as_data_op(common.IN / "domains.parquet").skb.apply_func(common.read_hosts)
    ctx = {
        "hosts": hosts,
        "graph": skrub.deferred(common.read_graph)(common.IN / "link-graph.parquet",
                                                   hosts.skb.apply_func(common.n_domains_of)),
        "tdom": skrub.as_data_op(common.IN / "trackers.tsv").skb.apply_func(common.read_tracker_domains),
        "pool": skrub.var("pool", pool_P),
    }
    F = common.features(X, ctx, blocks=BLOCKS)
    members = (
        tuple(DropColumns(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=s), NO_COUNTS)
              for s in range(5))
        + tuple(DropColumns(TorchMLPRanker(epochs=16, hidden=1024, dropout=0.1, random_state=10 + s), NO_COUNTS)
                for s in range(5))
        + tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=20 + s) for s in range(5))
    )
    pred = F.skb.apply(BlendRanker(estimators=members, method="prob"), y=y)

learner = pred.skb.make_learner()
learner.fit({"X": rows[["domain_id"]], "y": rows[common.Y_COLS], "pool": pool_P})
log("fitted on all of S")


def top10(scores):
    idx = np.argpartition(-scores, 10, axis=1)[:, :10]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)                  # best first


scores_full = learner.predict({"X": target, "pool": pool_full})
log("predicted target (full pool)")
scores_P = learner.predict({"X": target, "pool": pool_P})
log("predicted target (pool P, parity check)")

t_full, t_P = top10(scores_full), top10(scores_P)
same = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(t_full, t_P)])
log(f"top-10 overlap full-pool vs P-pool predictions: {same:.4f}")

tdom = common.read_tracker_domains(common.IN / "trackers.tsv")
sub = pd.DataFrame({"domain_id": np.repeat(target["domain_id"].to_numpy(), 10),
                    "tracking_domain_id": tdom[t_full.reshape(-1)]})
out = WS_ROOT / "submission.tsv"
sub.to_csv(out, sep="\t", index=False)
log(f"wrote {out} ({len(sub)} rows)")

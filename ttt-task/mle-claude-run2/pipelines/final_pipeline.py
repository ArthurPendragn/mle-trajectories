"""FINAL refit/predict artifact for pipeline_13 -- NOT a scored candidate (never ml-score it).

Refits pipeline_13's plan (same feature blocks, same
HybridRanker(nn_weight=2/3, n_seeds=8)) once on ALL of the modelled sample
(domain_id % 1301 == 7, 14,335 domains x 355 trackers), then ranks all 355
trackers for every domain in input/target.tsv and writes the top 10 per domain
to <workspace>/submission.tsv (headered TSV: domain_id, tracking_domain_id;
tracking_domain_id is the tracker's own domain id from trackers.tsv, not the
0-354 tracker_id).

How it is built
---------------
The scored plans root on `skrub.as_data_op(INPUT)` with the sample ids baked
into `load_context`, so they cannot be pointed at other rows. This file instead
calls the SAME functions eagerly: `common.load_context` once, `common.build_rows`
for the training rows, and the identical `common.BLOCKS[...]` functions in the
same order `common.features` concatenates them (tracker_id, then the blocks,
then domain_id appended last, exactly as pipeline_13 does). The estimator is
`common.HybridRanker` unchanged. skrub only routes data in pipeline_13, so this
is the same computation without the graph.

Parity choices (target-time features vs what the model saw in training)
-----------------------------------------------------------------------
* LABEL POOL: target rows read the SAME pool as training, i.e.
  POOL = tracking_graph rows whose domain is NOT in the modelled sample. This is
  parity-exact: every training row's statistics excluded the 14,335 sampled
  domains' labels, and target rows' statistics exclude them too. Target domains
  are absent from tracking_graph_train entirely, so no target row can reach its
  own label either way. (Adding the sample's labels back to the pool would
  change neighbour/TLD/token statistics by ~0.08% of pool domains -- a small,
  unmeasured distribution shift; not done.)
* EDGES: `load_context` already restricts the link graph to FOCUS = sample ids
  + target.tsv ids, so target rows get full neighbourhoods; global degree
  (`gdeg`) comes from the full graph and `ntrk` from POOL, as in training.
* PRIORS / token / TLD / content statistics are global POOL statistics,
  identical for training and target rows. `block_host` restricts POOL token
  statistics to the tokens of the domains being featurised, but each kept
  token's statistic is independent of which other domains are present, so
  building target features in domain chunks is exact.
* `n_true` does not exist for target rows; a dummy 0 column is added so X has
  the training schema, and (as in training) it is dropped before modelling.

Hardware
--------
Runs with ONE visible GPU: `available_devices()` returns ["cuda:0"], so
SeedAveragedListRanker trains its 8 seeds sequentially on that device (thread
pool of 1). Still 8 seeds, same seeds (42 + 1000*i), no code change needed.
LightGBM already uses n_jobs=16 (common.N_JOBS); torch and polars are capped
at 16 threads below.
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import time

import numpy as np
import pandas as pd
import polars as pl
import torch

import common
from common import BLOCKS, NON_FEATURE, N_TRACKERS, HybridRanker

torch.set_num_threads(16)

FEATURE_BLOCKS = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w",
                  "meta", "host", "content", "trkcode")      # = pipeline_13
K = 10
CHUNK_DOMAINS = 10_000          # target domains featurised/predicted per chunk

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)


def build_features(ctx, X):
    """Eager equivalent of common.features(...) + pipeline_13's domain_id concat."""
    parts = [BLOCKS[b](ctx, X) for b in FEATURE_BLOCKS]
    base = X.drop(columns=NON_FEATURE)
    feats = pd.concat([base] + parts + [X[["domain_id"]]], axis=1)
    return feats


# --- context (one read of input/; POOL = corpus minus the modelled sample) ---
ctx = common.load_context(common.INPUT)
log(f"context: pool {ctx['pool'].height} pairs over {ctx['n_pool_domains']} domains; "
    f"sample {len(ctx['sample_ids'])}; target {len(ctx['target_ids'])}; "
    f"focused edges {ctx['edges'].height}")
assert not ctx["pool"]["domain_id"].is_in(ctx["sample_ids"].implode()).any()

# --- training rows: all of the modelled sample -------------------------------
rows = common.build_rows(ctx)
y = rows["label"]
X = rows.drop(columns=["label"])
F_train = build_features(ctx, X)
train_cols = list(F_train.columns)
log(f"train features {F_train.shape}")

model = HybridRanker(nn_weight=2 / 3, n_seeds=8)
log(f"devices: {common.available_devices()}")
model.fit(F_train, y)
log(f"fitted HybridRanker on all of the sample (nn train loss {model.nn_.train_loss_:.4f})")
del F_train, rows, X, y

# --- target rows, chunked ----------------------------------------------------
target_ids = (pd.read_csv(common.INPUT / "target.tsv", sep="\t")["domain_id"]
              .drop_duplicates().to_numpy().astype(np.int64))
labelled = ctx["pool"]["domain_id"].unique().to_numpy()
assert not np.isin(target_ids, labelled).any(), "target overlaps the labelled pool"

tops = []
for s in range(0, len(target_ids), CHUNK_DOMAINS):
    ids = np.sort(target_ids[s:s + CHUNK_DOMAINS])
    Xt = pd.DataFrame({
        "domain_id": np.repeat(ids, N_TRACKERS).astype(np.int32),
        "tracker_id": np.tile(np.arange(N_TRACKERS), len(ids)).astype(np.int16),
        "n_true": np.zeros(len(ids) * N_TRACKERS, dtype=np.int16),   # dummy, dropped
    })
    Ft = build_features(ctx, Xt)
    assert list(Ft.columns) == train_cols, "target/train feature columns differ"
    sc = np.asarray(model.predict(Ft), dtype=np.float64).reshape(len(ids), N_TRACKERS)
    # rows are (domain, tracker)-ordered by construction -> column j = tracker j
    top = np.argpartition(-sc, K - 1, axis=1)[:, :K]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, 1), axis=1), 1)
    tops.append(pd.DataFrame({"domain_id": np.repeat(ids, K),
                              "tracker_id": top.reshape(-1)}))
    log(f"predicted target domains {s + len(ids)}/{len(target_ids)}")
    del Xt, Ft, sc

# --- map compact tracker_id -> tracking_domain_id and write ------------------
trk = pd.read_csv(common.INPUT / "trackers.tsv", sep="\t")
t2d = trk.set_index("tracker_id")["tracking_domain_id"]
assert t2d.index.is_unique and len(t2d) == N_TRACKERS
sub = pd.concat(tops, ignore_index=True)
sub["tracking_domain_id"] = t2d.reindex(sub["tracker_id"]).to_numpy()
assert sub["tracking_domain_id"].notna().all()
sub = sub[["domain_id", "tracking_domain_id"]].astype("int64")

out = common.WS_ROOT / "submission.tsv"
sub.to_csv(out, sep="\t", index=False)
log(f"wrote {out} ({len(sub)} rows)")
log(f"  domains {sub['domain_id'].nunique()} / {len(target_ids)}; "
    f"rows/domain {sub.groupby('domain_id').size().unique()}; "
    f"dup pairs {int(sub.duplicated().sum())}")
print(sub.head(12).to_string(index=False))

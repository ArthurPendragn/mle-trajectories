"""Plan-building helpers for the TrackTheTrackers workspace (skrub DataOps).

Task: for each domain, rank the 355 trackers; scored by Recall@10 averaged over
domains. There is no ready-made table, so the design matrix is built here from
the raw pile in input/, INSIDE the plan (guide section 15), never cached.

Row = one domain of the modelled sample S. y = its raw 355-wide tracker
indicator (marked immediately). X = its domain_id (marked immediately, carries
the CV). Every feature is built downstream of the marks, per fold.

Sample S (deterministic code, no file): drawn from the tracked corpus with a
per-domain inclusion probability set by SAMPLE_MIN_COUNT / SAMPLE_EXPONENT, using a hash of the
domain_id as the uniform draw. The design was chosen in data_exploration_2-4:
target.tsv is NOT a uniform draw over tracked domains, but IS consistent with a
uniform draw over tracked domains having >= 2 trackers -- so S is exactly that.

Label provenance (leakage, part 2): neighbour/sibling features read labels
ONLY from the pool P = tracked corpus minus S minus E (E = shadow eval set). S and P are disjoint by
construction, so no feature of any row of S can contain that row's own labels
-- including via return walks (d -> n -> d contributes nothing, d is not in P).
The target domains are not in the corpus at all, and at prediction time they
read the same P, so train/predict parity on label coverage is exact.

The marked y is the ONLY place S's labels enter; the rankers learn from it
per training fold.

Scoring: Recall@10 per domain, averaged (task metric). Declared in the plan via
attach_scoring (a 2-D indicator y and a 355-wide score matrix cannot go through
a scorer string). Run ml-score WITHOUT --scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import skrub
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 3
WS_ROOT = Path(__file__).resolve().parent.parent
IN = WS_ROOT / "input"

N_TRACKERS = 355
Y_COLS = [f"t{i:03d}" for i in range(N_TRACKERS)]
TARGET_PAIRS = 5_000_000        # sample size, in (domain, tracker) pairs (user: ~5M rows)

# Inclusion probability for a tracked domain with c trackers:
#   (c >= SAMPLE_MIN_COUNT) * min(1, f * c**SAMPLE_EXPONENT),
# f solved so the expected number of pairs is TARGET_PAIRS. Set from
# data_exploration_2-4 before pipeline_01 and never changed afterwards.
# Chosen from the shift analyses: target.tsv is indistinguishable from a
# uniform draw over tracked domains with >= 2 trackers (adversarial AUC 0.503
# vs 0.612 for uniform-over-tracked; T3 0.590, size-biased 0.547, 50/50 mix
# 0.545), and the label-shift (BBSE) inversion independently puts ~0-2% of the
# target on single-tracker domains (data_exploration_3/4).
SAMPLE_MIN_COUNT = 2
SAMPLE_EXPONENT = 0.0
SALT = 0x5EED
# Shadow evaluation set E: a UNIFORM draw over all tracked domains, disjoint
# from S (separate hash stream), also removed from the label pool. Never used by
# the harness; only by the stratified report, so recall can be measured on the
# count buckets S does not cover.
SHADOW_FRAC = 0.011            # ~200k domains
SHADOW_SALT = 0xE7A1


def make_cv():
    """The workspace's CV splitter -- the SINGLE place the split is defined.

    Domains are the unit; features only read labels from the disjoint pool P,
    so rows of S are exchangeable given the design -> plain shuffled KFold.
    """
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# --- recorded reads --------------------------------------------------------
def read_tracking(path):
    df = pd.read_parquet(path, columns=["domain_id", "tracker_id"])
    return df.astype({"domain_id": np.int64, "tracker_id": np.int16})


def read_graph(path, n_domains):
    """Link graph as CSR adjacency: out[d] = domains d links to, in[d] = domains linking to d."""
    t = pl.read_parquet(path)
    src = t["source_domain_id"].to_numpy(); dst = t["target_domain_id"].to_numpy()
    out = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n_domains, n_domains))
    return {"out": out, "in": out.T.tocsr()}


def read_hosts(path):
    """Hostnames indexed by domain_id (ids are 0..N-1, dense)."""
    d = pl.read_parquet(path).sort("domain_id")
    assert d["domain_id"][-1] == d.height - 1
    return d["domain"]


def read_tracker_domains(path):
    """tracking_domain_id of each tracker, indexed by tracker_id (0..354)."""
    t = pd.read_csv(path, sep="\t").sort_values("tracker_id")
    assert (t["tracker_id"].to_numpy() == np.arange(N_TRACKERS)).all()
    return t["tracking_domain_id"].to_numpy()


def read_urlcat(path):
    """url-classification.csv -> (hostname without www., category), one row per hostname."""
    u = pl.read_csv(path)
    return (u.with_columns(pl.col("url").str.extract(r"^[a-z]+://([^/:]+)", 1).str.to_lowercase()
                           .str.replace(r"^www\.", "").alias("h"))
             .drop_nulls("h").unique("h", keep="first").select("h", "category"))


def n_domains_of(hosts):
    return len(hosts)


# --- the sample ----------------------------------------------------------------
def _uniform01(ids, salt=SALT):
    """Deterministic per-domain uniform draw (splitmix64 of id ^ salt)."""
    z = (ids.astype(np.uint64) ^ np.uint64(salt)) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def shadow_domains(tg):
    """Domain ids of the shadow evaluation set E (uniform over tracked, disjoint from S)."""
    ids = np.unique(tg["domain_id"].to_numpy())
    return ids[_uniform01(ids, SHADOW_SALT) < SHADOW_FRAC]


def sample_domains(tg, min_count=None, exponent=None):
    """Domain ids of S: Poisson sample, P(include) = (c >= m) * min(1, f * c**a), minus E."""
    m = SAMPLE_MIN_COUNT if min_count is None else min_count
    a = SAMPLE_EXPONENT if exponent is None else exponent
    ids, cnt = np.unique(tg["domain_id"].to_numpy(), return_counts=True)
    elig = (cnt >= m) & (_uniform01(ids, SHADOW_SALT) >= SHADOW_FRAC)
    ids, cnt = ids[elig], cnt[elig]
    w = cnt.astype(np.float64) ** a
    lo, hi = 0.0, 1.0 / w.min()
    for _ in range(100):                      # bisection on f for E[pairs] = TARGET_PAIRS
        f = (lo + hi) / 2
        if (np.minimum(1, f * w) * cnt).sum() > TARGET_PAIRS:
            hi = f
        else:
            lo = f
    keep = _uniform01(ids) < np.minimum(1, f * w)
    return ids[keep]


def build_rows(tg, sel):
    """One row per sampled domain: domain_id + the raw 355-wide tracker indicator."""
    sub = tg[tg["domain_id"].isin(sel)]
    r = np.searchsorted(sel, sub["domain_id"].to_numpy())
    Y = np.zeros((len(sel), N_TRACKERS), np.uint8)
    Y[r, sub["tracker_id"].to_numpy()] = 1
    return pd.concat([pd.DataFrame({"domain_id": sel}), pd.DataFrame(Y, columns=Y_COLS)], axis=1)


def build_pool(tg, sel, shadow, n_domains):
    """Label pool P = corpus minus S minus E, as an N x 355 CSR indicator + 'tracked' vector."""
    sub = tg[~tg["domain_id"].isin(sel) & ~tg["domain_id"].isin(shadow)]
    L = sp.csr_matrix((np.ones(len(sub), np.float32), (sub["domain_id"].to_numpy(), sub["tracker_id"].to_numpy())),
                      shape=(n_domains, N_TRACKERS))
    tracked = (np.diff(L.indptr) > 0).astype(np.float32)
    return {"L": L, "tracked": tracked}


def load_context():
    """The recorded reads + sample + label pool (shared by every pipeline)."""
    tg = skrub.as_data_op(IN / "tracking_graph_train.parquet").skb.apply_func(read_tracking)
    hosts = skrub.as_data_op(IN / "domains.parquet").skb.apply_func(read_hosts)
    n = hosts.skb.apply_func(n_domains_of)
    graph = skrub.deferred(read_graph)(IN / "link-graph.parquet", n)
    sel = tg.skb.apply_func(sample_domains)
    shadow = tg.skb.apply_func(shadow_domains)
    pool = skrub.deferred(build_pool)(tg, sel, shadow, n)
    tdom = skrub.as_data_op(IN / "trackers.tsv").skb.apply_func(read_tracker_domains)
    urlcat = skrub.as_data_op(IN / "url-classification.csv").skb.apply_func(read_urlcat)
    return {"tg": tg, "hosts": hosts, "n": n, "graph": graph, "sel": sel, "shadow": shadow,
            "pool": pool, "tdom": tdom, "urlcat": urlcat}


def load_xy(ctx, subsample=20_000):
    """Rows of S, marked EARLY: y = raw indicator, X = domain_id (+ the CV)."""
    rows = skrub.deferred(build_rows)(ctx["tg"], ctx["sel"])
    if subsample:
        rows = rows.skb.subsample(n=subsample)
    y = rows[Y_COLS].skb.mark_as_y()
    X = rows[["domain_id"]].skb.mark_as_X(cv=make_cv())
    return X, y


# --- feature blocks (each one node, built after the marks) -----------------
def nbr_block(X, graph, pool, direction):
    """Tracker mix of the row's labelled neighbours (labels from P only).

    <dir>_tXXX = share of labelled neighbours carrying tracker XXX; plus
    log-degree, log #labelled neighbours, labelled fraction.
    """
    ids = X["domain_id"].to_numpy()
    M = graph[direction][ids]
    cnt = (M @ pool["L"]).toarray()
    lab = M @ pool["tracked"]
    deg = np.diff(M.indptr).astype(np.float32)
    frac = cnt / np.maximum(lab, 1)[:, None]
    out = pd.DataFrame(frac.astype(np.float32), columns=[f"{direction}_{c}" for c in Y_COLS], index=X.index)
    out[f"{direction}_deg_log"] = np.log1p(deg)
    out[f"{direction}_lab_log"] = np.log1p(lab)
    out[f"{direction}_lab_frac"] = lab / np.maximum(deg, 1)
    return out


def twohop_block(X, graph, pool, first, second, weight="mean"):
    """2-hop label mix, walking d -first-> u -second-> v; labels of v from P only.

    first="in",  second="out": CO-CITATION  (d <- u -> v: sites linked from the same pages)
    first="out", second="in" : COUPLING     (d -> u <- v: sites linking to the same targets)
    Each intermediate u contributes the tracker shares among ITS labelled
    second-step neighbours; the row averages over its u's. A return walk
    (v == d) adds nothing: d is in S, never in P.
    weight="idf": each u is weighted by log(N / (1 + deg_second(u))), so hubs
    (u linked to/from millions) count less than niche shared links.
    """
    ids = X["domain_id"].to_numpy()
    M = graph[first][ids]
    U = np.unique(M.indices)
    MU = graph[second][U]
    lab_u = MU @ pool["tracked"]
    Z = sp.diags((1 / np.maximum(lab_u, 1)).astype(np.float32)) @ (MU @ pool["L"])   # |U| x 355 shares
    has_u = (lab_u > 0).astype(np.float32)
    # remap M's columns to positions in U, then average over the row's informative u's
    Mc = sp.csr_matrix((M.data, np.searchsorted(U, M.indices), M.indptr), shape=(len(ids), len(U)))
    n_u = Mc @ has_u
    tag = f"{first}{second}"
    if weight == "idf":
        idf = np.log(graph[second].shape[0] / (1.0 + np.diff(MU.indptr))).astype(np.float32)
        Mw = Mc @ sp.diags(idf)
        mix = (Mw @ Z).toarray() / np.maximum(Mw @ has_u, 1e-6)[:, None]
        tag += "idf"
    else:
        mix = (Mc @ Z).toarray() / np.maximum(n_u, 1)[:, None]
    out = pd.DataFrame(mix.astype(np.float32), columns=[f"{tag}_{c}" for c in Y_COLS], index=X.index)
    out[f"{tag}_nu_log"] = np.log1p(n_u)
    out[f"{tag}_lab_log"] = np.log1p(Mc @ lab_u)
    return out


def direct_block(X, graph, tdom):
    """d links straight to tracker t's own domain (355 binary; label-free, from the link graph).

    data_exploration_6: precision 0.55 per link (e.g. linkwithin 0.99, histats
    0.98), 14% of rows covered, identical coverage in the target.
    """
    ids = X["domain_id"].to_numpy()
    D = (graph["out"][ids][:, tdom].toarray() > 0).astype(np.float32)
    out = pd.DataFrame(D, columns=[f"dl_{c}" for c in Y_COLS], index=X.index)
    out["dl_n"] = D.sum(1)
    return out


URL_CATS = ["Adult", "Arts", "Business", "Computers", "Games", "Health", "Home", "Kids", "News",
            "Recreation", "Reference", "Science", "Shopping", "Society", "Sports"]
TLD_MIN_SUPPORT = 50


def meta_block(X, hosts, pool, urlcat):
    """Full-TLD label prior + url-classification category.

    tldp_tXXX: share of labelled pool domains with the row's exact TLD carrying
    tracker XXX (labels from P only; TLDs with < TLD_MIN_SUPPORT labelled pool
    domains fall back to the global pool share). Every TLD is covered, not just
    the top-60 one-hots of host_block.
    uc_*: url-classification category one-hot (hostname match, www. stripped).
    """
    tld_all = hosts.str.extract(r"\.([^.]+)$", 1).fill_null("").to_numpy()
    codes, tld_idx = np.unique(tld_all, return_inverse=True)
    lab = pool["tracked"] > 0
    Lc = pool["L"][np.flatnonzero(lab)]
    G = sp.csr_matrix((np.ones(lab.sum(), np.float32), (tld_idx[lab], np.arange(lab.sum()))),
                      shape=(len(codes), lab.sum()))
    cnt_t = np.asarray(G.sum(1)).ravel()
    prior = (G @ Lc).toarray() / np.maximum(cnt_t, 1)[:, None]
    glob = np.asarray(Lc.sum(0)).ravel() / max(lab.sum(), 1)
    prior[cnt_t < TLD_MIN_SUPPORT] = glob
    ids = X["domain_id"].to_numpy()
    out = pd.DataFrame(prior[tld_idx[ids]].astype(np.float32), columns=[f"tldp_{c}" for c in Y_COLS], index=X.index)
    out["tldp_support_log"] = np.log1p(cnt_t[tld_idx[ids]]).astype(np.float32)
    h = pl.DataFrame({"h": hosts.gather(ids).str.replace(r"^www\.", "")})
    cat = h.join(urlcat, on="h", how="left", maintain_order="left")["category"].to_numpy()
    for c in URL_CATS:
        out[f"uc_{c}"] = (cat == c).astype(np.float32)
    return out


def nbr_count_block(X, graph, pool, direction):
    """log1p(# labelled neighbours carrying tracker XXX) -- the magnitude that the
    share columns of nbr_block normalise away (3 of 4 vs 3 of 50). Labels from P only."""
    ids = X["domain_id"].to_numpy()
    cnt = (graph[direction][ids] @ pool["L"]).toarray()
    return pd.DataFrame(np.log1p(cnt).astype(np.float32), columns=[f"{direction}c_{c}" for c in Y_COLS],
                        index=X.index)


TOP_TLDS = 60


def host_block(X, hosts):
    """Hostname shape + TLD one-hot (top TLDs over all 46M domains; label-free)."""
    h = hosts.gather(X["domain_id"].to_numpy())
    df = pl.DataFrame({"h": h})
    tld_all = hosts.str.extract(r"\.([^.]+)$", 1)
    top = tld_all.value_counts(sort=True)["domain"].head(TOP_TLDS).to_list()
    f = df.select(
        pl.col("h").str.len_chars().cast(pl.Float32).alias("h_len"),
        pl.col("h").str.count_matches(r"\.").cast(pl.Float32).alias("h_dots"),
        pl.col("h").str.count_matches(r"[0-9]").cast(pl.Float32).alias("h_digits"),
        pl.col("h").str.count_matches("-").cast(pl.Float32).alias("h_hyph"),
        pl.col("h").str.extract(r"\.([^.]+)$", 1).alias("tld"),
    )
    out = f.drop("tld").to_pandas()
    tld = f["tld"].to_numpy()
    for t in top:
        out[f"tld_{t}"] = (tld == t).astype(np.float32)
    out.index = X.index
    return out


def features(X, ctx, blocks=("out", "in", "host")):
    """Concatenate the requested feature blocks (each its own recorded node)."""
    parts = []
    for b in blocks:
        if b in ("out", "in"):
            parts.append(skrub.deferred(nbr_block)(X, ctx["graph"], ctx["pool"], b))
        elif b.removesuffix("idf") in ("inout", "outin", "outout", "inin"):
            w = "idf" if b.endswith("idf") else "mean"
            b0 = b.removesuffix("idf")
            first, second = (b0[:2], b0[2:]) if b0.startswith("in") else (b0[:3], b0[3:])
            parts.append(skrub.deferred(twohop_block)(X, ctx["graph"], ctx["pool"], first, second, w))
        elif b in ("outc", "inc"):
            parts.append(skrub.deferred(nbr_count_block)(X, ctx["graph"], ctx["pool"], b[:-1]))
        elif b == "meta":
            parts.append(skrub.deferred(meta_block)(X, ctx["hosts"], ctx["pool"], ctx["urlcat"]))
        elif b == "direct":
            parts.append(skrub.deferred(direct_block)(X, ctx["graph"], ctx["tdom"]))
        elif b == "host":
            parts.append(skrub.deferred(host_block)(X, ctx["hosts"]))
        else:
            raise ValueError(b)
    return parts[0].skb.concat(parts[1:], axis=1) if len(parts) > 1 else parts[0]


# --- scoring -------------------------------------------------------------------
def recall_at_10(y_true, y_score):
    """Task metric: per domain |top10 ∩ true| / |true|, averaged over domains."""
    Y = np.asarray(y_true, dtype=bool)
    S = np.asarray(y_score, dtype=np.float32)
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    hits = np.take_along_axis(Y, top, axis=1).sum(1)
    return float(np.mean(hits / Y.sum(1)))


def per_domain_recall_at_10(y_true, y_score):
    Y = np.asarray(y_true, dtype=bool)
    top = np.argpartition(-np.asarray(y_score, dtype=np.float32), 10, axis=1)[:, :10]
    return np.take_along_axis(Y, top, axis=1).sum(1) / Y.sum(1)


SCORER = make_scorer(recall_at_10, response_method="predict")
SCORER_NAME = "recall_at_10"


def attach_scoring(pred, sample_weight=None):
    kwargs = {"sample_weight": sample_weight} if sample_weight is not None else None
    return pred.skb.with_scoring(SCORER, kwargs=kwargs, name=SCORER_NAME)

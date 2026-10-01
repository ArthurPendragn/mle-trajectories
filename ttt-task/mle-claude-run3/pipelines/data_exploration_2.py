"""Exploration 2 — is the target representative of the training corpus?

target.tsv carries no labels, so we cannot observe the property the metric
divides by (true tracker count per domain). Instead:

  (A) build label-FREE observables for every domain: link-graph degrees, how
      many neighbours are tracked (labelled) and what their labels look like
      (other rows' labels only, never the row's own), hostname shape, TLD,
      same-site siblings, url-classification coverage;
  (B) adversarial classifiers: target vs. tracked corpus, and target vs.
      untracked domains (in domains.parquet, no trackers known);
  (C) count model: predict n_trackers from (A) on the tracked corpus, apply it
      to target and to a held-out corpus slice, compare the distributions; and
      invert the calibration (P(true count | predicted bin) on the held-out
      slice) to estimate the target's true-count mix.

Read-only: prints a report, writes nothing.
"""
import time

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import WS_ROOT

IN = WS_ROOT / "input"
RNG = np.random.default_rng(0)
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

dom = pl.read_parquet(IN / "domains.parquet")
N = dom.height
tg = pl.read_parquet(IN / "tracking_graph_train.parquet", columns=["domain_id", "tracker_id"])
tgt_ids = pl.read_csv(IN / "target.tsv", separator="\t")["domain_id"].to_numpy()

d_arr = tg["domain_id"].to_numpy(); t_arr = tg["tracker_id"].to_numpy()
count = np.bincount(d_arr, minlength=N).astype(np.float32)          # true count (0 = not tracked)
tracked = (count > 0).astype(np.float32)
GA, ADS, FB = 129, 131, 104
has = {k: np.zeros(N, np.float32) for k in ("ga", "ads", "fb")}
for k, tid in (("ga", GA), ("ads", ADS), ("fb", FB)):
    has[k][d_arr[t_arr == tid]] = 1
is_target = np.zeros(N, bool); is_target[tgt_ids] = True
log("labels built")

# ---------------- (A) observables -----------------------------------------
lg = pl.read_parquet(IN / "link-graph.parquet")
src = lg["source_domain_id"].to_numpy(); dst = lg["target_domain_id"].to_numpy()
A = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))
AT = A.T.tocsr()
del lg, src, dst
log("adjacency built", A.nnz)

feat = {}
feat["out_deg"] = np.diff(A.indptr).astype(np.float32)
feat["in_deg"] = np.diff(AT.indptr).astype(np.float32)
for direction, M in (("out", A), ("in", AT)):
    lab = M @ tracked                                  # tracked neighbours (never self: no self loops)
    feat[f"{direction}_lab"] = lab
    feat[f"{direction}_lab_frac"] = np.where(feat[f"{direction}_deg"] > 0, lab / np.maximum(feat[f"{direction}_deg"], 1), np.nan)
    feat[f"{direction}_nbr_cnt_mean"] = np.where(lab > 0, (M @ count) / np.maximum(lab, 1), np.nan)
    for k in has:
        feat[f"{direction}_nbr_{k}"] = np.where(lab > 0, (M @ has[k]) / np.maximum(lab, 1), np.nan)
log("graph features built")

host = dom.sort("domain_id")["domain"]
hs = pl.DataFrame({"h": host})
parts = hs.select(
    pl.col("h").str.len_chars().alias("len"),
    pl.col("h").str.count_matches(r"\.").alias("dots"),
    pl.col("h").str.count_matches(r"[0-9]").alias("digits"),
    pl.col("h").str.count_matches("-").alias("hyph"),
    pl.col("h").str.split(".").alias("lbl"),
)
parts = parts.with_columns(
    pl.col("lbl").list.last().alias("tld"),
    pl.col("lbl").list.get(-2, null_on_oob=True).alias("sld"),
)
SECOND_LEVEL = ["co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "go", "gv", "nic", "ltd", "plc", "sch", "info", "biz"]
parts = parts.with_columns(
    pl.when(pl.col("sld").is_in(SECOND_LEVEL) & (pl.col("tld").str.len_chars() == 2) & (pl.col("lbl").list.len() >= 3))
    .then(pl.col("lbl").list.slice(-3, 3).list.join("."))
    .otherwise(pl.col("lbl").list.slice(-2, 2).list.join(".")).alias("site"),
)
parts = parts.with_columns(
    (pl.col("dots") + 1 - pl.col("site").str.count_matches(r"\.") - 1).alias("sub_depth"),
    pl.Series("tracked", tracked), pl.Series("is_target", is_target),
)
site_agg = parts.group_by("site").agg(pl.len().alias("site_n"), pl.col("tracked").sum().alias("site_tracked"))
parts = parts.join(site_agg, on="site", how="left", maintain_order="left")
for c in ("len", "dots", "digits", "hyph", "sub_depth"):
    feat[c] = parts[c].to_numpy().astype(np.float32)
feat["site_n_others"] = parts["site_n"].to_numpy().astype(np.float32) - 1
feat["site_tracked_others"] = (parts["site_tracked"].to_numpy() - tracked).astype(np.float32)   # exclude self
tld_s = parts["tld"].to_pandas()
top_tld = tld_s.value_counts().index[:150]
feat_tld = pd.Categorical(tld_s.where(tld_s.isin(top_tld), "other"))
log("hostname features built")

urlc = pl.read_csv(IN / "url-classification.csv")
uh = urlc.with_columns(pl.col("url").str.extract(r"^[a-z]+://([^/:]+)", 1).str.to_lowercase()
                       .str.replace(r"^www\.", "").alias("h")).unique("h")
hs2 = pl.DataFrame({"h": host.str.replace(r"^www\.", "")}).with_row_index("i")
m = hs2.join(uh.select("h", "category"), on="h", how="left", maintain_order="left")
feat_urlcat = pd.Categorical(m["category"].fill_null("none").to_pandas())
log("url-classification coverage: target", float((m["category"].is_not_null().to_numpy()[is_target]).mean()),
    "tracked", float((m["category"].is_not_null().to_numpy()[tracked > 0]).mean()))

def frame(idx):
    df = pd.DataFrame({k: v[idx] for k, v in feat.items()})
    df["tld"] = feat_tld[idx]
    df["urlcat"] = feat_urlcat[idx]
    return df

tracked_ids = np.flatnonzero(tracked > 0)
untracked_ids = np.flatnonzero((tracked == 0) & ~is_target)

# ---------------- (B) adversarial -----------------------------------------
XGB_CLF = dict(tree_method="hist", device="cuda", n_estimators=400, learning_rate=0.08, max_depth=7,
               subsample=0.8, colsample_bytree=0.8, enable_categorical=True, max_cat_to_onehot=1,
               eval_metric="auc")

def adversarial(neg_ids, name, n_neg=500_000):
    neg = RNG.choice(neg_ids, n_neg, replace=False)
    X = frame(np.concatenate([tgt_ids, neg])); yv = np.r_[np.ones(len(tgt_ids)), np.zeros(n_neg)]
    oof = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, yv):
        mdl = xgb.XGBClassifier(**XGB_CLF).fit(X.iloc[tr], yv[tr])
        oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(yv, oof)
    imp = pd.Series(mdl.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    log(f"ADVERSARIAL target vs {name}: AUC = {auc:.4f}")
    print("   top gain features:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items()))
    # where does it separate? share of target in top-scored 1% vs base rate
    top = oof >= np.quantile(oof, 0.99)
    print(f"   base rate {yv.mean():.4f}; target share in top-1% scored {yv[top].mean():.4f}")
    return auc, oof, yv, neg

auc_tr, oof_tr, y_tr, neg_tr = adversarial(tracked_ids, "tracked corpus")
auc_un, *_ = adversarial(untracked_ids, "untracked domains")

# compare marginal distributions directly (target vs a random tracked slice)
print("\nMarginal comparison target vs tracked (median / mean / share>0):")
cmp_ids = RNG.choice(tracked_ids, 500_000, replace=False)
ft, fc = frame(tgt_ids), frame(cmp_ids)
for c in feat:
    a, b = ft[c].to_numpy(), fc[c].to_numpy()
    ks = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)]).statistic
    print(f"  {c:22s} target med {np.nanmedian(a):8.3f} mean {np.nanmean(a):9.3f} >0 {np.nanmean(a>0):.3f} | "
          f"corpus med {np.nanmedian(b):8.3f} mean {np.nanmean(b):9.3f} >0 {np.nanmean(b>0):.3f} | KS {ks:.4f}")
print("  top TLD shares  target:", ft["tld"].value_counts(normalize=True).head(8).round(3).to_dict())
print("                  corpus:", fc["tld"].value_counts(normalize=True).head(8).round(3).to_dict())

# ---------------- (C) count model ------------------------------------------
perm = RNG.permutation(tracked_ids)
tr_ids, ho_ids = perm[:3_000_000], perm[3_000_000:4_000_000]
Xtr, Xho, Xtg = frame(tr_ids), frame(ho_ids), frame(tgt_ids)
reg = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=800, learning_rate=0.05, max_depth=8,
                       subsample=0.8, colsample_bytree=0.8, enable_categorical=True, max_cat_to_onehot=1,
                       objective="count:poisson")
reg.fit(Xtr, count[tr_ids], eval_set=[(Xho, count[ho_ids])], verbose=False)
p_ho, p_tg = reg.predict(Xho), reg.predict(Xtg)
c_ho = count[ho_ids]
from scipy.stats import spearmanr
log(f"COUNT MODEL holdout: spearman {spearmanr(p_ho, c_ho).statistic:.4f}, "
    f"MAE {np.mean(np.abs(p_ho-c_ho)):.3f} vs const-median MAE {np.mean(np.abs(np.median(c_ho)-c_ho)):.3f}")
imp = pd.Series(reg.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("   top gain features:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items()))
qs = [.05, .1, .25, .5, .75, .9, .95, .99]
print("   predicted count quantiles holdout:", np.round(np.quantile(p_ho, qs), 3))
print("   predicted count quantiles target :", np.round(np.quantile(p_tg, qs), 3))
print(f"   mean predicted: holdout {p_ho.mean():.4f}, target {p_tg.mean():.4f};  KS = {ks_2samp(p_ho, p_tg).statistic:.4f}")

# calibration inversion: bin by predicted count, P(true bucket | bin) on holdout
edges = np.quantile(p_ho, np.linspace(0, 1, 21)); edges[0], edges[-1] = -np.inf, np.inf
b_ho = np.digitize(p_ho, edges[1:-1]); b_tg = np.digitize(p_tg, edges[1:-1])
buckets = np.array(["1", "2", "3", "4-5", "6-10", ">10"])
def bucket(c):
    return np.select([c == 1, c == 2, c == 3, c <= 5, c <= 10], [0, 1, 2, 3, 4], 5)
tb_ho = bucket(c_ho)
P = np.zeros((20, 6))
for b in range(20):
    P[b] = np.bincount(tb_ho[b_ho == b], minlength=6) / max((b_ho == b).sum(), 1)
w_ho = np.bincount(b_ho, minlength=20) / len(b_ho); w_tg = np.bincount(b_tg, minlength=20) / len(b_tg)
print("\n   predicted-bin mass (20 holdout-quantile bins), holdout vs target:")
print("   holdout:", np.round(w_ho, 3)); print("   target :", np.round(w_tg, 3))
true_mix = np.bincount(tb_ho, minlength=6) / len(tb_ho)
est_mix = w_tg @ P
print("   true-count mix, corpus holdout (observed)   :", dict(zip(buckets, np.round(true_mix, 4))))
print("   true-count mix, target (count-model implied):", dict(zip(buckets, np.round(est_mix, 4))))
print(f"   implied mean count: holdout {c_ho.mean():.3f}; target via bin-reweighting "
      f"{(w_tg @ np.array([c_ho[b_ho==b].mean() for b in range(20)])):.3f}")
log("done")

"""Exploration 3 — which sampling mechanism produced target.tsv?

Exploration 2 found target != uniform over tracked domains (adversarial AUC
0.61; count model says target domains carry more trackers). Here each candidate
sampling DESIGN draws a corpus comparison set, and an adversarial classifier
tries to tell target from it. A design that reproduces the target's mechanism
should push the AUC to ~0.5.

  U       uniform over tracked domains
  SB^a    P(domain) proportional to count^a, a in {0.5, 1, 1.5}
          (a=1 == "sample (domain, tracker) pairs uniformly, keep their domains")
  COV     density-ratio reweighting on observables (target-vs-U classifier,
          out-of-fold odds) -- matches observables by construction; tells us
          which true-count mix a pure covariate-shift story implies

Then the label-shift (BBSE) estimate of the target's true-count mix:
C[b,k] = P(predicted bin b | true count bucket k) from a held-out uniform slice;
solve w_target = C q for q >= 0, sum(q)=1 (NNLS). Compared against the mix each
design implies.

Read-only: prints a report, writes nothing.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import shift_observables

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

O = shift_observables.build()
count, frame, tgt_ids = O["count"], O["frame"], O["tgt_ids"]
tracked_ids = np.flatnonzero(count > 0)
c_tr = count[tracked_ids]
log("observables built")

XGB_CLF = dict(tree_method="hist", device="cuda", n_estimators=400, learning_rate=0.08, max_depth=7,
               subsample=0.8, colsample_bytree=0.8, enable_categorical=True, max_cat_to_onehot=1)
N_NEG = 250_000
Xt = frame(tgt_ids)

def adv_auc(neg_ids, seed=0):
    X = pd.concat([Xt, frame(neg_ids)], ignore_index=True)
    yv = np.r_[np.ones(len(tgt_ids)), np.zeros(len(neg_ids))]
    oof = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yv):
        oof[te] = xgb.XGBClassifier(**XGB_CLF).fit(X.iloc[tr], yv[tr]).predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(yv, oof), oof

BUCKETS = ["1", "2", "3", "4-5", "6-10", ">10"]
def bucket(c):
    return np.select([c == 1, c == 2, c == 3, c <= 5, c <= 10], [0, 1, 2, 3, 4], 5)
def mix(c, w=None):
    return np.bincount(bucket(c), weights=w, minlength=6) / (len(c) if w is None else w.sum())
def fmt(m):
    return " ".join(f"{b}:{v:.3f}" for b, v in zip(BUCKETS, m))

rng = np.random.default_rng(1)
results = {}
for name, a in (("U", 0.0), ("SB^0.5", 0.5), ("SB^1", 1.0), ("SB^1.5", 1.5)):
    w = c_tr ** a
    neg = rng.choice(tracked_ids, N_NEG, replace=False, p=w / w.sum())
    auc, _ = adv_auc(neg)
    results[name] = (auc, mix(count[neg]), count[neg].mean())
    log(f"design {name:7s}: adversarial AUC {auc:.4f} | mean count {count[neg].mean():.3f} | {fmt(mix(count[neg]))}")

# COV: density ratio from a target-vs-U classifier fit on a separate uniform pool
pool = rng.choice(tracked_ids, 1_000_000, replace=False)
Xp = frame(pool)
clf = xgb.XGBClassifier(**XGB_CLF).fit(pd.concat([Xt, Xp.iloc[:500_000]], ignore_index=True),
                                        np.r_[np.ones(len(tgt_ids)), np.zeros(500_000)])
# score the other half (not used to fit) -> odds as importance weights
p = clf.predict_proba(Xp.iloc[500_000:])[:, 1]
wcov = p / (1 - p)
cand = pool[500_000:]
neg = rng.choice(cand, N_NEG, replace=True, p=wcov / wcov.sum())
auc, _ = adv_auc(neg)   # (target rows were in the density-ratio fit -> optimistic lower bound)
log(f"design COV    : adversarial AUC {auc:.4f} | mean count {count[neg].mean():.3f} | {fmt(mix(count[neg]))}"
    f"  [ESS {wcov.sum()**2/(wcov**2).sum():.0f} of {len(wcov)}]")
results["COV"] = (auc, mix(count[neg]), count[neg].mean())

# BBSE label-shift estimate
perm = rng.permutation(tracked_ids)
tr_ids, ho_ids = perm[:3_000_000], perm[3_000_000:4_000_000]
reg = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=800, learning_rate=0.05, max_depth=8,
                       subsample=0.8, colsample_bytree=0.8, enable_categorical=True, max_cat_to_onehot=1,
                       objective="count:poisson").fit(frame(tr_ids), count[tr_ids])
p_ho, p_tg = reg.predict(frame(ho_ids)), reg.predict(Xt)
edges = np.quantile(p_ho, np.linspace(0, 1, 21))[1:-1]
b_ho, b_tg = np.digitize(p_ho, edges), np.digitize(p_tg, edges)
k_ho = bucket(count[ho_ids])
C = np.zeros((20, 6))
for k in range(6):
    C[:, k] = np.bincount(b_ho[k_ho == k], minlength=20) / max((k_ho == k).sum(), 1)
w_tg = np.bincount(b_tg, minlength=20) / len(b_tg)
# append sum-to-one row with a large weight, then NNLS
lam = 100.0
q, res = nnls(np.vstack([C, lam * np.ones(6)]), np.r_[w_tg, lam])
q = q / q.sum()
# bootstrap the target side for an uncertainty band
qs = []
for _ in range(200):
    bb = rng.choice(b_tg, len(b_tg), replace=True)
    wb = np.bincount(bb, minlength=20) / len(bb)
    qb, _ = nnls(np.vstack([C, lam * np.ones(6)]), np.r_[wb, lam]); qs.append(qb / qb.sum())
qs = np.array(qs)
reps = np.array([1, 2, 3, 4.4, 7.0, 15.0])      # bucket-mean counts ~ for the implied mean
print()
log("BBSE (label-shift) target count mix :", fmt(q), f"| implied mean ~{q @ reps:.3f}")
print("      bootstrap 5-95%:", " ".join(f"{b}:[{lo:.3f},{hi:.3f}]" for b, lo, hi in
                                     zip(BUCKETS, np.quantile(qs, .05, 0), np.quantile(qs, .95, 0))))
print("      residual of fit ||Cq - w||:", round(float(np.linalg.norm(C @ q - w_tg)), 4),
      "| same for uniform mix:", round(float(np.linalg.norm(C @ mix(count[ho_ids]) - w_tg)), 4),
      "| for SB^1 mix:", round(float(np.linalg.norm(C @ results['SB^1'][1] - w_tg)), 4))
print("\nSummary (AUC closer to 0.5 = design closer to the target's mechanism):")
for k, (auc, m, mc) in results.items():
    print(f"  {k:7s} AUC {auc:.4f}  mean count {mc:.3f}  {fmt(m)}")
log("done")

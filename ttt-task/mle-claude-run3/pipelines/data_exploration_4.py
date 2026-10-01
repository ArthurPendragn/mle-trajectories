"""Exploration 4 — test the truncation hypothesis suggested by BBSE.

Exploration 3: no count-power (size-biased) design explains the target
(best AUC 0.547), but the label-shift (BBSE) inversion put ~0% of the target
on single-tracker domains and a >=2 mix close to the corpus's own >=2 mix.
Hypothesis T2: target = uniform draw over tracked domains with >= 2 trackers.

Designs compared by adversarial AUC (target vs design draw), 2 seeds each for a
noise level:
  U        uniform over tracked            (reference, expect ~0.61)
  T2       uniform over count >= 2
  T3       uniform over count >= 3
  T2+SB    count >= 2, proportional to count (does extra size bias help?)
  MIX50    50% U + 50% T2                  (partial truncation)
BBSE is repeated with finer buckets (1,2,3,4,5,6-10,>10) to see whether the
>=2 shape matches the corpus's truncated shape bucket by bucket.

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

def adv_auc(neg_ids, seed):
    X = pd.concat([Xt, frame(neg_ids)], ignore_index=True)
    yv = np.r_[np.ones(len(tgt_ids)), np.zeros(len(neg_ids))]
    oof = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yv):
        oof[te] = xgb.XGBClassifier(**XGB_CLF, random_state=seed).fit(X.iloc[tr], yv[tr]).predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(yv, oof)

B = ["1", "2", "3", "4", "5", "6-10", ">10"]
def bucket(c):
    return np.select([c == 1, c == 2, c == 3, c == 4, c == 5, c <= 10], [0, 1, 2, 3, 4, 5], 6)
def mix(c):
    return np.bincount(bucket(c), minlength=7) / len(c)
def fmt(m):
    return " ".join(f"{b}:{v:.3f}" for b, v in zip(B, m))

def draw(name, rng):
    if name == "U":
        return rng.choice(tracked_ids, N_NEG, replace=False)
    if name == "T2":
        return rng.choice(tracked_ids[c_tr >= 2], N_NEG, replace=False)
    if name == "T3":
        return rng.choice(tracked_ids[c_tr >= 3], N_NEG, replace=False)
    if name == "T2+SB":
        ids, w = tracked_ids[c_tr >= 2], c_tr[c_tr >= 2]
        return rng.choice(ids, N_NEG, replace=False, p=w / w.sum())
    if name == "MIX50":
        return np.r_[rng.choice(tracked_ids, N_NEG // 2, replace=False),
                     rng.choice(tracked_ids[c_tr >= 2], N_NEG // 2, replace=False)]

summary = {}
for name in ("U", "T2", "T3", "T2+SB", "MIX50"):
    aucs = []
    for seed in (0, 1):
        neg = draw(name, np.random.default_rng(100 + seed))
        aucs.append(adv_auc(neg, seed))
    summary[name] = (aucs, mix(count[neg]))
    log(f"design {name:6s}: AUC {np.mean(aucs):.4f} (seeds {aucs[0]:.4f}, {aucs[1]:.4f}) | {fmt(mix(count[neg]))}")

# BBSE with finer buckets
rng = np.random.default_rng(7)
perm = rng.permutation(tracked_ids)
tr_ids, ho_ids = perm[:3_000_000], perm[3_000_000:4_500_000]
reg = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=800, learning_rate=0.05, max_depth=8,
                       subsample=0.8, colsample_bytree=0.8, enable_categorical=True, max_cat_to_onehot=1,
                       objective="count:poisson").fit(frame(tr_ids), count[tr_ids])
p_ho, p_tg = reg.predict(frame(ho_ids)), reg.predict(Xt)
edges = np.quantile(p_ho, np.linspace(0, 1, 31))[1:-1]
b_ho, b_tg = np.digitize(p_ho, edges), np.digitize(p_tg, edges)
k_ho = bucket(count[ho_ids])
C = np.zeros((30, 7))
for k in range(7):
    C[:, k] = np.bincount(b_ho[k_ho == k], minlength=30) / max((k_ho == k).sum(), 1)
lam = 100.0
def bbse(bins):
    w = np.bincount(bins, minlength=30) / len(bins)
    q, _ = nnls(np.vstack([C, lam * np.ones(7)]), np.r_[w, lam])
    return q / q.sum(), w
q, w_tg = bbse(b_tg)
qs = np.array([bbse(rng.choice(b_tg, len(b_tg)))[0] for _ in range(300)])
corpus = mix(count[ho_ids])
trunc2 = corpus.copy(); trunc2[0] = 0; trunc2 /= trunc2.sum()
print()
log("BBSE fine buckets")
print("   target (BBSE)       :", fmt(q))
print("   bootstrap 5%        :", fmt(np.quantile(qs, .05, 0)))
print("   bootstrap 95%       :", fmt(np.quantile(qs, .95, 0)))
print("   corpus (uniform)    :", fmt(corpus))
print("   corpus truncated >=2:", fmt(trunc2))
for nm, m in (("uniform", corpus), ("trunc>=2", trunc2), ("BBSE", q)):
    print(f"   fit residual ||C m - w_target|| for {nm:9s}: {np.linalg.norm(C @ m - w_tg):.4f}")
# sanity: BBSE recovers a KNOWN truncated mix when applied to a truncated holdout slice
t2_ho = np.flatnonzero(count[ho_ids] >= 2)
q_chk, _ = bbse(rng.choice(b_ho[t2_ho], 50_000))
print("   sanity: BBSE on a 50k truncated-holdout draw:", fmt(q_chk))
q_chk, _ = bbse(rng.choice(b_ho, 50_000))
print("   sanity: BBSE on a 50k uniform-holdout draw  :", fmt(q_chk))
log("done")

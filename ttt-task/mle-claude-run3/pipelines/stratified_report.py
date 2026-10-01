"""Stratified Recall@10 of one pipeline + the headline under named target mixes.

    python stratified_report.py pipeline_XX [--no-shadow]

Analysis only (reads input/, prints; writes nothing, never touches
results.json). Two measurement routes, both leakage-free:

  CV on S   the harness's exact folds (common.make_cv on the sample rows):
            per-domain Recall@10 of the fold model on its held-out fold.
            Covers only the count buckets S contains.
  shadow E  a uniform draw over ALL tracked domains, disjoint from S and removed
            from the label pool (common.shadow_domains). The pipeline is fit
            once on all of S -- the deployed model -- and scored on E, which
            covers every bucket, including the single-tracker domains S excludes.

Per-bucket recall r_k (buckets 1,2,3,4,5,6-10,>10) is then combined with
each named assumption about the target's count mix q_k: headline = sum_k q_k r_k.
That transfer assumes recall | count bucket is the same in target as in E.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

BUCKETS = ["1", "2", "3", "4", "5", "6-10", ">10"]

def bucket(c):
    c = np.asarray(c)
    return np.select([c == 1, c == 2, c == 3, c == 4, c == 5, c <= 10], [0, 1, 2, 3, 4, 5], 6)

# Named assumptions about the target's true-count mix, over BUCKETS.
# Provenance: data_exploration_3 (U, SB^1, COV, coarse BBSE) and
# data_exploration_4 (T2 AUC 0.503; fine-bucket BBSE). U/T2 are recomputed
# exactly from the corpus below; the others are the printed estimates.
FIXED_MIXES = {
    # covariate-shift story: density-ratio reweighting on observables (expl. 3, COV);
    # 4 and 5 split from "4-5" (0.107) in the corpus 4:5 ratio (0.052:0.029)
    "COV (covariate shift, expl.3)": [0.447, 0.253, 0.140, 0.0685, 0.0385, 0.047, 0.005],
    # size-biased (domains drawn proportional to count), expl. 3 SB^1, same split of 4-5
    "SB^1 (size-biased, expl.3)":   [0.277, 0.237, 0.177, 0.114, 0.064, 0.108, 0.022],
}


def load_pipeline(name):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--bbse", default=None,
                    help="comma-separated BBSE mix over BUCKETS (from data_exploration_4)")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

    mod = load_pipeline(args.pipeline)
    pred = mod.pred
    from skrub._data_ops._estimator import _compute_X_y_and_cv
    data = _compute_X_y_and_cv(pred, {})
    X, y = data["X"], data["y"]
    xy = pred.skb.make_learner().__skrub_to_Xy_pipeline__({})
    Y = np.asarray(y, dtype=bool)
    cnt = Y.sum(1)
    log(f"S: {len(X)} domains, {cnt.sum()} pairs, count buckets {np.bincount(bucket(cnt), minlength=7)}")

    # --- route 1: the harness folds -------------------------------------
    rec = np.full(len(X), np.nan)
    fold_scores = []
    for k, (tr, te) in enumerate(common.make_cv().split(X, y)):
        m = clone(xy).fit(X.iloc[tr], y.iloc[tr])
        r = common.per_domain_recall_at_10(Y[te], m.predict(X.iloc[te]))
        rec[te] = r
        fold_scores.append(r.mean())
        log(f"fold {k}: recall@10 {r.mean():.5f}")
    print(f"CV mean {np.mean(fold_scores):.5f} (folds {np.round(fold_scores, 5)})  "
          "-- should match the harness score")
    b = bucket(cnt)
    cv_tab = pd.DataFrame({"bucket": BUCKETS,
                           "n": np.bincount(b, minlength=7),
                           "recall_cv": [rec[b == i].mean() if (b == i).any() else np.nan for i in range(7)]})

    # --- route 2: shadow set E, deployed model -----------------------------
    if not args.no_shadow:
        from common import read_tracking, shadow_domains, build_rows
        tg = read_tracking(common.IN / "tracking_graph_train.parquet")
        E = shadow_domains(tg)
        rowsE = build_rows(tg, E)
        YE = rowsE[common.Y_COLS].to_numpy(bool)
        m = clone(xy).fit(X, y)
        rE = common.per_domain_recall_at_10(YE, m.predict(rowsE[["domain_id"]]))
        bE = bucket(YE.sum(1))
        cv_tab["n_shadow"] = np.bincount(bE, minlength=7)
        cv_tab["recall_shadow"] = [rE[bE == i].mean() for i in range(7)]
        # binomial-ish standard error of each bucket mean
        cv_tab["se_shadow"] = [rE[bE == i].std() / np.sqrt(max((bE == i).sum(), 1)) for i in range(7)]
        corpus_mix = np.bincount(bucket(np.bincount(tg["domain_id"].to_numpy())[np.unique(tg["domain_id"])]),
                                 minlength=7) / tg["domain_id"].nunique()
        log(f"shadow E: {len(E)} domains, overall recall {rE.mean():.5f}")
    print("\nRecall@10 stratified by the number of true trackers per domain:")
    print(cv_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.no_shadow:
        return
    r_k = cv_tab["recall_shadow"].to_numpy()
    se_k = cv_tab["se_shadow"].to_numpy()
    mixes = {"U (uniform over tracked corpus)": corpus_mix}
    t2 = corpus_mix.copy(); t2[0] = 0; t2 /= t2.sum()
    mixes["T2 (uniform over count>=2; expl.4 AUC 0.503)"] = t2
    if args.bbse:
        mixes["BBSE point (label shift, expl.4)"] = np.array([float(v) for v in args.bbse.split(",")])
    mixes.update({k: np.array(v) / np.sum(v) for k, v in FIXED_MIXES.items()})
    s_mix = np.bincount(b, minlength=7) / len(b)
    mixes["S (the sample's own mix)"] = s_mix
    print("\nHeadline Recall@10 under named assumptions about the target's count mix")
    print("(per-bucket recall from shadow E, deployed model; +-2 SE from bucket sampling only):")
    rows = []
    for name, q in mixes.items():
        h = float(q @ r_k); se = float(np.sqrt(((q * se_k) ** 2).sum()))
        rows.append((name, h, se, q[0]))
        print(f"  {name:48s} {h:.4f}  +-{2*se:.4f}   (share of 1-tracker domains {q[0]:.3f})")
    hs = [r[1] for r in rows]
    print(f"\nRange across assumptions: {min(hs):.4f} .. {max(hs):.4f}")
    log("done")


if __name__ == "__main__":
    main()

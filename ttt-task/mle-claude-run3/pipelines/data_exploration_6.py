"""Exploration 6 — direct hyperlinks to tracker domains (and to their companies).

Does "d links to tracker t's own domain" (d -> tracking_domain_id) predict that
t tracks d? And at company level (d links to any tracker domain of company C
-> C's trackers)? Measured on S (labels = S's own, used ONLY to evaluate here,
never as a feature), with coverage parity against the target.
Read-only: prints, writes nothing.
"""
import numpy as np
import pandas as pd

import common

tg = common.read_tracking(common.IN / "tracking_graph_train.parquet")
hosts = common.read_hosts(common.IN / "domains.parquet")
n = len(hosts)
graph = common.read_graph(common.IN / "link-graph.parquet", n)
trk = pd.read_csv(common.IN / "trackers.tsv", sep="\t")
S = common.sample_domains(tg)
tgt = pd.read_csv(common.IN / "target.tsv", sep="\t")["domain_id"].to_numpy()
rows = common.build_rows(tg, S); Y = rows[common.Y_COLS].to_numpy(bool)

tdom = trk.sort_values("tracker_id")["tracking_domain_id"].to_numpy()
M = graph["out"][S][:, tdom].toarray() > 0          # S x 355: d links to tracker t's domain
MT = graph["out"][tgt][:, tdom].toarray() > 0
print("share of S rows linking to >=1 tracker domain:", M.any(1).mean().round(4),
      "| target:", MT.any(1).mean().round(4))
print("mean #tracker domains linked, S:", M.sum(1).mean().round(3), "target:", MT.sum(1).mean().round(3))
nl = M.sum(0); hit = (M & Y).sum(0); base = Y.mean(0)
df = trk.sort_values("tracker_id")[["tracker_id", "domain", "company", "category"]].assign(
    n_link=nl, p_if_link=np.where(nl > 0, hit / np.maximum(nl, 1), np.nan), base=base,
    lift=np.where(nl > 0, hit / np.maximum(nl, 1), np.nan) / np.maximum(base, 1e-9))
print("\ntop linked tracker domains (P(tracker | link) vs base rate):")
print(df.sort_values("n_link", ascending=False).head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# company level
comp = trk.sort_values("tracker_id")["company"].fillna("?").to_numpy()
C = pd.get_dummies(comp).to_numpy(np.float32)       # 355 x n_company
link_c = (M.astype(np.float32) @ C) > 0               # S x companies
same_c = link_c @ C.T > 0                             # S x 355: links to a domain of t's company
nl2 = same_c.sum(0); hit2 = (same_c & Y).sum(0)
print("\ncompany-level link -> tracker: pairs with signal", int(same_c.sum()),
      "precision", (hit2.sum() / max(nl2.sum(), 1)).round(4),
      "| tracker-level: pairs", int(M.sum()), "precision", (hit.sum() / max(M.sum(), 1)).round(4))
print("recall of true pairs covered by a tracker-level link:", ((M & Y).sum() / Y.sum()).round(4),
      "company-level:", ((same_c & Y).sum() / Y.sum()).round(4))

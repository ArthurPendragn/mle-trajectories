"""First learned ranker: MLP over the labelled-neighbour tracker mix.

Features (all built after the marks, labels from the disjoint pool P only):
  out_*  share of the row's labelled OUT-neighbours carrying each tracker
  in_*   same for IN-neighbours (domains linking to the row)
  host   hostname shape + top-60 TLD one-hot
Model: 2x1024 GELU MLP, listwise softmax loss (each domain weighs 1, like the
per-domain-averaged metric).
"""
import common
from rankers import TorchMLPRanker

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=("out", "in", "host"))
pred = common.attach_scoring(F.skb.apply(TorchMLPRanker(epochs=8), y=y))

DESCRIPTION = "MLP (2x1024, softmax loss) on out/in labelled-neighbour tracker shares + host/TLD"
PARENT = "pipeline_01"

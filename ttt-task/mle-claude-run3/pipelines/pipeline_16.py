"""FINAL: the combination of everything that measured positive.

Features: every block that survived ablation 1 (1-hop out/in shares, all four
2-hop walks, direct links, host/TLD) + the 1-hop log-count channels of
pipeline_14 (flat alone, but a different view of the same neighbourhoods).
Model: softmax-averaged ensemble of 15 MLPs (softmax listwise loss, 16 epochs):
  - 5x 2048/dropout 0.3 and 5x 1024/dropout 0.1 on pipeline_11's blocks (= pipeline_15)
  - 5x 2048/dropout 0.3 on pipeline_14's blocks (+ outc/inc)  -> feature-set diversity
Measured and rejected on the way (see README): pairwise GBM (07) and its blend
(08), capacity push (12), wide path (13), hostname n-grams / hub-damped 2-hop /
full-TLD prior + url category (explorations 5 and 7).
"""
import common
from rankers import BlendRanker, DropColumns, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin", "outc", "inc")
NO_COUNTS = ("outc_", "inc_")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
members = (
    tuple(DropColumns(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=s), NO_COUNTS)
          for s in range(5))
    + tuple(DropColumns(TorchMLPRanker(epochs=16, hidden=1024, dropout=0.1, random_state=10 + s), NO_COUNTS)
            for s in range(5))
    + tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=20 + s) for s in range(5))
)
pred = common.attach_scoring(F.skb.apply(BlendRanker(estimators=members, method="prob"), y=y))

DESCRIPTION = "FINAL: 15-MLP ensemble (pipeline_15's 10 members + 5 on the +log-count feature set)"
PARENT = "pipeline_15"

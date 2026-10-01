"""pipeline_11 + per-tracker log-count channels for the 1-hop neighbourhoods.

Residual analysis (data_exploration_9): 48% of misses are near misses (rank
11-20), concentrated on mid-tail trackers (popularity rank 11-100 = 62% of
misses). The 1-hop blocks only give shares, which hide magnitude (3 of 4
neighbours vs 3 of 50); outc/inc expose log1p(count) per tracker. Audited in
data_exploration_5 (S == E standalone, target parity).
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin", "outc", "inc")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=s)
                                     for s in range(5)), method="prob")
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "pipeline_11 + 1-hop per-tracker log-count channels (outc, inc)"
PARENT = "pipeline_11"

"""Push past ablation-2's grid edge: longer, wider, more dropout (still 5 seeds).

pipeline_10's best cell sat at the corner of the grid (most epochs, widest,
most dropout), so the optimum may lie beyond it: 24 epochs, 2x3072, dropout 0.4.
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=tuple(TorchMLPRanker(epochs=24, hidden=3072, dropout=0.4, random_state=s)
                                     for s in range(5)), method="prob")
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "capacity push: 5-seed MLP ensemble, 24 ep, 2x3072, dropout 0.4"
PARENT = "pipeline_11"

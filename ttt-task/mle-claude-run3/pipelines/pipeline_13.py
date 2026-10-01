"""Wide & deep: pipeline_11 + a linear input -> 355-logit path beside the MLP.

Direct per-tracker evidence (a direct link to t's domain, a neighbour share
for a rare t) maps one feature to one output; the wide path makes that a
single weight instead of something the dropout-regularised deep path must learn.
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, skip=True, random_state=s)
                                     for s in range(5)), method="prob")
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "pipeline_11 + wide linear skip path (wide & deep), 5 seeds"
PARENT = "pipeline_11"

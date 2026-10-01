"""Blend: MLP (pipeline_06) + pairwise GBM (pipeline_07) via reciprocal-rank fusion.

pipeline_07 alone was 0.024 below the MLP; this tests whether its errors are
diverse enough to help as a minority vote (weights 1.0 / 0.5, RRF c=10).
"""
import common
from rankers import BlendRanker, PairGBMRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=(TorchMLPRanker(epochs=8), PairGBMRanker(k=40, n_estimators=600, device="cuda")),
                    weights=(1.0, 0.5), method="rrf", c=10.0)
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "RRF blend MLP (w1.0) + pairwise GBM (w0.5), pipeline_06 feature blocks"
PARENT = "pipeline_06"

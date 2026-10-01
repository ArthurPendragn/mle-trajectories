"""Variance reduction: 5-seed ensemble of the pipeline_06 MLP (mean of softmax).

Evidence it is variance-limited: the deployed model (fit on all of S) beats the
2/3-S fold models by ~+0.003 on every count bucket (stratified_report on 04).
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=tuple(TorchMLPRanker(epochs=8, random_state=s) for s in range(5)), method="prob")
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "5-seed MLP ensemble (softmax average), pipeline_06 blocks"
PARENT = "pipeline_06"

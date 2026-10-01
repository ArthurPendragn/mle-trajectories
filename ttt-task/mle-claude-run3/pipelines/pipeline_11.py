"""Promote ablation-2's winner into the seed ensemble.

pipeline_10 grid: softmax > bce everywhere; 16 epochs > 8; best cell
softmax/16 epochs/2048 wide/dropout 0.3 = 0.88985 (base 8/1024/0.1 = 0.88776).
pipeline_09 showed a 5-seed softmax average adds ~+0.001 on top of a single MLP.
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = BlendRanker(estimators=tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=s)
                                     for s in range(5)), method="prob")
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "5-seed ensemble of the ablation-2 winner MLP (softmax, 16 ep, 2x2048, dropout 0.3)"
PARENT = "pipeline_10"

"""Promote ablation-1's winner: pipeline_04's blocks + out>out and in>in 2-hop walks.

pipeline_05 grid: +outout+inin 0.88776 vs full 0.88734; every drop-one variant
lost score (largest: -outin -0.0116, -inout -0.0068, -host -0.0048).
"""
import common
from rankers import TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
pred = common.attach_scoring(F.skb.apply(TorchMLPRanker(epochs=8), y=y))

DESCRIPTION = "promoted ablation winner: all 1-hop + all four 2-hop walks + direct links + host (MLP 2x1024)"
PARENT = "pipeline_05"

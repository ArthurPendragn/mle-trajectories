"""ABLATION 2 (explorative fused-choice): MLP hyperparameters on pipeline_06's blocks.

2x2x2x2 grid: epochs {8,16} x width {1024,2048} x dropout {0.1,0.3} x loss
{softmax (per-domain listwise), bce (per-pair)}. Single model (no seed
ensemble) so the grid isolates the configuration effect.
"""
import skrub

import common
from rankers import TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
model = TorchMLPRanker(
    epochs=skrub.choose_from([8, 16], name="epochs"),
    hidden=skrub.choose_from([1024, 2048], name="hidden"),
    dropout=skrub.choose_from([0.1, 0.3], name="dropout"),
    loss=skrub.choose_from(["softmax", "bce"], name="loss"),
    device="auto",
)
pred = common.attach_scoring(F.skb.apply(model, y=y))

DESCRIPTION = "ABLATION: MLP epochs x width x dropout x loss (16 variants), pipeline_06 blocks"
PARENT = "pipeline_06"

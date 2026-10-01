"""ABLATION 1 (explorative fused-choice): which feature blocks carry the score?

Base = pipeline_04's blocks. Variants drop one block at a time, plus one that
adds the two remaining 2-hop walks (out>out, in>in). One scored run; the full
per-variant grid lands in results.json -> extra.grid.
"""
import skrub

import common
from rankers import TorchMLPRanker

BASE = ("out", "in", "host", "inout", "outin", "direct")
VARIANTS = {"full": BASE, "+outout+inin": BASE + ("outout", "inin")}
VARIANTS.update({f"-{b}": tuple(x for x in BASE if x != b) for b in BASE})

ctx = common.load_context()
X, y = common.load_xy(ctx)
pred = skrub.choose_from(
    {name: common.features(X, ctx, blocks=bl).skb.apply(TorchMLPRanker(epochs=8, device="auto"), y=y)
     for name, bl in VARIANTS.items()},
    name="featureset").as_data_op()
pred = common.attach_scoring(pred)

DESCRIPTION = "ABLATION: drop-one-block over pipeline_04's feature set, plus +outout+inin"
PARENT = "pipeline_04"

"""+ direct hyperlinks from the domain to each tracker's own domain.

Label-free (link graph only); data_exploration_6: a direct link means the
tracker is present 55% of the time, 14% of rows covered, same in the target.
"""
import common
from rankers import TorchMLPRanker

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=("out", "in", "host", "inout", "outin", "direct"))
pred = common.attach_scoring(F.skb.apply(TorchMLPRanker(epochs=8), y=y))

DESCRIPTION = "pipeline_03 + direct-link block (domain links to tracker t's own domain, 355 binary)"
PARENT = "pipeline_03"

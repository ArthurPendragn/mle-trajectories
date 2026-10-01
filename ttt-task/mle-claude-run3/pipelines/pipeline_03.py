"""+ 2-hop label mixes: co-citation (d <- u -> v) and coupling (d -> u <- v).

Both audited in data_exploration_5 (standalone S == E, target parity), labels
from the disjoint pool P only.
"""
import common
from rankers import TorchMLPRanker

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=("out", "in", "host", "inout", "outin"))
pred = common.attach_scoring(F.skb.apply(TorchMLPRanker(epochs=8), y=y))

DESCRIPTION = "pipeline_02 + 2-hop co-citation (in>out) and coupling (out>in) tracker mixes"
PARENT = "pipeline_02"

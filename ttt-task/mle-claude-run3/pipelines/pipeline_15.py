"""Mixed-config 10-member ensemble on pipeline_11's blocks.

After pipeline_06 every feature/capacity change was flat or negative (12
capacity push -0.0003, 13 wide path -0.0014, 14 log-counts -0.0002); only
training/ensembling paid (09 +0.0010, 11 +0.0026). So: 5 seeds of each of
ablation-2's two best cells -- 2048/dropout 0.3 (0.88985) and 1024/dropout 0.1
(0.88945), both softmax, 16 epochs -- softmax-averaged.
"""
import common
from rankers import BlendRanker, TorchMLPRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
members = tuple(TorchMLPRanker(epochs=16, hidden=2048, dropout=0.3, random_state=s) for s in range(5)) + \
          tuple(TorchMLPRanker(epochs=16, hidden=1024, dropout=0.1, random_state=10 + s) for s in range(5))
pred = common.attach_scoring(F.skb.apply(BlendRanker(estimators=members, method="prob"), y=y))

DESCRIPTION = "10-member mixed-config MLP ensemble (5x 2048/0.3 + 5x 1024/0.1), pipeline_11 blocks"
PARENT = "pipeline_11"

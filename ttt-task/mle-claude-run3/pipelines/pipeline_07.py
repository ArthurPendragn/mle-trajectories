"""New model family on pipeline_06's features: per-(domain, tracker) GBM ranker.

Instead of one 355-way output layer, XGBoost rank:ndcg learns ONE scoring
function over (domain, candidate tracker) pairs: the value of every block at
that tracker + tracker id + training-fold popularity + row scalars. Candidates:
top-40 per row by block-sum + popularity (ceiling = candidate recall).
"""
import common
from rankers import PairGBMRanker

BLOCKS = ("out", "in", "host", "inout", "outin", "direct", "outout", "inin")

ctx = common.load_context()
X, y = common.load_xy(ctx)
F = common.features(X, ctx, blocks=BLOCKS)
pred = common.attach_scoring(F.skb.apply(PairGBMRanker(k=40, n_estimators=600, device="cuda"), y=y))

DESCRIPTION = "pairwise XGBoost rank:ndcg over top-40 candidates, pipeline_06 feature blocks"
PARENT = "pipeline_06"

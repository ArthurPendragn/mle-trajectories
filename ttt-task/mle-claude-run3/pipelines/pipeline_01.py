"""Baseline: the same 10 trackers for every domain (training-fold popularity).

No features at all -- the floor every learned ranker has to beat.
"""
import common
from rankers import PopularityRanker

ctx = common.load_context()
X, y = common.load_xy(ctx)
pred = common.attach_scoring(X.skb.apply(PopularityRanker(), y=y))

DESCRIPTION = "global popularity top-10 (training-fold tracker frequency), no features"
PARENT = None

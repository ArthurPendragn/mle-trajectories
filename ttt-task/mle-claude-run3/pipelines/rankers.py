"""Multi-label tracker rankers shared by the pipelines.

Every ranker is fit on (features, Y) where Y is the n x 355 indicator of the
row's trackers (the marked y, raw), and `predict` returns an n x 355 float32
SCORE matrix; common.recall_at_10 takes the top-10 of it. Rankers inherit
RegressorMixin only so skrub sees a `score` method and routes scoring to the
plan's with_scoring node (the mixin's own R^2 is never used).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class PopularityRanker(RegressorMixin, BaseEstimator):
    """Same list for everyone: trackers ranked by training-fold frequency."""

    def fit(self, X, y):
        self.p_ = np.asarray(y, dtype=np.float32).mean(0)
        return self

    def predict(self, X):
        return np.broadcast_to(self.p_, (len(X), len(self.p_))).copy()


class _WideDeep(nn.Module):
    def __init__(self, deep, wide):
        super().__init__()
        self.deep, self.wide = deep, wide

    def forward(self, x):
        return self.deep(x) + self.wide(x)


def _device(spec):
    """"auto" spreads parallel CV workers (separate processes) over the visible GPUs."""
    if spec == "auto":
        return torch.device(f"cuda:{os.getpid() % torch.cuda.device_count()}")
    return torch.device(spec)


class TorchMLPRanker(RegressorMixin, BaseEstimator):
    """MLP over the dense feature matrix -> 355 logits, trained on GPU.

    loss="softmax": listwise cross-entropy against y/|y| -- every domain gets
    total weight 1, as in the per-domain-averaged Recall@10.
    loss="bce": independent sigmoid per tracker (domains with more trackers
    weigh more).
    Features are standardised with training-fold statistics inside fit().
    skip=True adds a wide (linear, input -> 355 logits) path next to the deep one.
    """
    skip = False    # class default: estimators pickled before `skip` existed stay valid

    def __init__(self, hidden=1024, n_layers=2, dropout=0.1, epochs=8, lr=1e-3,
                 weight_decay=1e-5, batch_size=4096, loss="softmax", device="cuda", random_state=0,
                 skip=False):
        self.skip = skip
        self.hidden, self.n_layers, self.dropout = hidden, n_layers, dropout
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.loss, self.device, self.random_state = batch_size, loss, device, random_state

    def _net(self, d_in, d_out):
        layers, d = [], d_in
        for _ in range(self.n_layers):
            layers += [nn.Linear(d, self.hidden), nn.GELU(), nn.Dropout(self.dropout)]
            d = self.hidden
        layers.append(nn.Linear(d, d_out))
        deep = nn.Sequential(*layers)
        return _WideDeep(deep, nn.Linear(d_in, d_out)) if self.skip else deep

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        Xn = np.asarray(X, dtype=np.float32)
        self.mu_ = Xn.mean(0)
        self.sd_ = Xn.std(0) + 1e-6
        dev = _device(self.device)
        self.dev_ = dev
        Xt = torch.from_numpy((Xn - self.mu_) / self.sd_).to(dev)
        Yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
        self.n_out_ = Yt.shape[1]
        self.net_ = self._net(Xt.shape[1], Yt.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = Xt.shape[0]
        steps = self.epochs * ((n + self.batch_size - 1) // self.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps,
                                                    pct_start=min(0.5, max(0.1, 3.0 / steps)))
        g = torch.Generator(device=dev).manual_seed(self.random_state)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.net_(Xt[idx])
                yb = Yt[idx]
                if self.loss == "softmax":
                    tgt = yb / yb.sum(1, keepdim=True).clamp_min(1)
                    loss = -(tgt * torch.log_softmax(logits, 1)).sum(1).mean()
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        self.net_.eval()
        del Xt, Yt
        torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = self.dev_
        Xn = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        out = np.empty((Xn.shape[0], self.n_out_), np.float32)
        for i in range(0, Xn.shape[0], 65536):
            out[i:i + 65536] = self.net_(torch.from_numpy(Xn[i:i + 65536]).to(dev)).float().cpu().numpy()
        return out


class PairGBMRanker(RegressorMixin, BaseEstimator):
    """Per-(domain, candidate tracker) gradient-boosted ranker (XGBoost, GPU).

    The dense feature frame is read by column name: every block that has one
    column per tracker (`<block>_tXXX`) becomes an n x 355 matrix; all other
    columns are row-level scalars. Candidates per row = top-`k` trackers by a
    heuristic (sum of the block matrices + training-fold popularity). Each
    (row, candidate) pair gets: the value of every block at that tracker, the
    tracker id (categorical, so per-tracker biases are learnable), its training
    fold popularity, and the row scalars. Objective rank:ndcg grouped by row;
    non-candidates are scored -inf, so the recall ceiling is the candidate recall.
    """

    def __init__(self, k=40, n_estimators=600, learning_rate=0.08, max_depth=8,
                 objective="rank:ndcg", device="auto", random_state=0):
        self.k, self.n_estimators, self.learning_rate = k, n_estimators, learning_rate
        self.max_depth, self.objective, self.device, self.random_state = max_depth, objective, device, random_state

    def _split(self, X):
        cols = list(X.columns)
        blocks = {}
        for i, c in enumerate(cols):
            head, _, tail = c.rpartition("_")
            if len(tail) == 4 and tail[0] == "t" and tail[1:].isdigit():
                blocks.setdefault(head, []).append(i)
        blocks = {b: ix for b, ix in blocks.items() if len(ix) == 355}
        used = {i for ix in blocks.values() for i in ix}
        scal = [i for i in range(len(cols)) if i not in used]
        return blocks, scal

    def _pairs(self, X):
        A = np.asarray(X, dtype=np.float32)
        mats = [A[:, ix] for ix in self.blocks_.values()]
        h = sum(mats) + self.prior_[None, :]
        cand = np.argpartition(-h, self.k, axis=1)[:, :self.k]
        n = A.shape[0]
        feats = [np.take_along_axis(m, cand, 1).reshape(-1) for m in mats]
        feats.append(self.prior_[cand].reshape(-1))
        feats.append(cand.reshape(-1).astype(np.float32))           # tracker id
        S = A[:, self.scal_]
        rows = np.repeat(np.arange(n), self.k)
        F = np.column_stack(feats + [S[rows]]) if S.shape[1] else np.column_stack(feats)
        return F, cand, rows

    def fit(self, X, y):
        import xgboost as xgb
        Y = np.asarray(y, dtype=np.float32)
        self.blocks_, self.scal_ = self._split(X)
        self.prior_ = Y.mean(0)
        F, cand, rows = self._pairs(X)
        lab = np.take_along_axis(Y, cand, 1).reshape(-1)
        self.cand_recall_ = float((lab.reshape(cand.shape).sum(1) / np.maximum(Y.sum(1), 1)).mean())
        n_blk = len(self.blocks_)
        ft = ["q"] * (n_blk + 1) + ["c"] + ["q"] * (F.shape[1] - n_blk - 2)
        dev = str(_device(self.device))
        self.model_ = xgb.XGBRanker(tree_method="hist", device=dev, n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate, max_depth=self.max_depth,
                                    objective=self.objective, lambdarank_pair_method="topk",
                                    lambdarank_num_pair_per_sample=10, eval_metric="ndcg@10",
                                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                                    feature_types=ft, enable_categorical=True, max_cat_to_onehot=1)
        self.model_.fit(F, lab, qid=rows)
        return self

    def predict(self, X):
        F, cand, rows = self._pairs(X)
        s = self.model_.predict(F).reshape(cand.shape)
        out = np.full((cand.shape[0], 355), -1e9, np.float32)
        np.put_along_axis(out, cand, s.astype(np.float32), 1)
        return out


class BlendRanker(RegressorMixin, BaseEstimator):
    """Fit several rankers on the same features; fuse their per-row rankings.

    method="rrf": reciprocal-rank fusion, sum_m w_m / (c + rank_m(t)) -- scale-free,
    so a softmax MLP and a margin-scored GBM combine without calibration.
    method="prob": weighted mean of per-row softmax(score / T_m) (T=1 for all).
    """

    def __init__(self, estimators=(), weights=None, method="rrf", c=10.0):
        self.estimators, self.weights, self.method, self.c = estimators, weights, method, c

    def fit(self, X, y):
        from sklearn.base import clone
        self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]
        return self

    def predict(self, X):
        w = self.weights or [1.0] * len(self.fitted_)
        out = None
        for wi, m in zip(w, self.fitted_):
            s = m.predict(X)
            if self.method == "rrf":
                rank = np.argsort(np.argsort(-s, axis=1), axis=1).astype(np.float32)
                part = wi / (self.c + rank)
            else:
                z = s - s.max(1, keepdims=True)
                e = np.exp(z)
                part = wi * e / e.sum(1, keepdims=True)
            out = part if out is None else out + part
        return out.astype(np.float32)


class DropColumns(RegressorMixin, BaseEstimator):
    """Fit/predict `estimator` on the frame minus columns starting with `prefixes`
    (lets ensemble members inside one BlendRanker see different feature sets)."""

    def __init__(self, estimator=None, prefixes=()):
        self.estimator, self.prefixes = estimator, prefixes

    def _cols(self, X):
        return [c for c in X.columns if not c.startswith(tuple(self.prefixes))]

    def fit(self, X, y):
        from sklearn.base import clone
        self.cols_ = self._cols(X)
        self.est_ = clone(self.estimator).fit(X[self.cols_], y)
        return self

    def predict(self, X):
        return self.est_.predict(X[self.cols_])

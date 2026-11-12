# -*- coding: utf-8 -*-
"""Gradient-boosted regression trees, hand-written NumPy.
Primary author: Li Baixi (histogram GBT, early stopping, pooled training).


This is the project's primary forecasting model (course rule: the main
implementation must be ours; scikit-learn appears only as a benchmark).

Design (Friedman's stochastic gradient boosting + LightGBM-style histogram
splits):

  F_0(x)                     = mean(y)
  F_m(x)                     = F_{m-1}(x) + lr * tree_m(x)
  tree_m                     fits the current L2 residuals r = y - F_{m-1}

Each tree is a CART grown greedily on quantile-binned features ("histogram"
method): every feature is pre-discretised into n_bins buckets, so a node only
needs a bincount of its rows per candidate feature to evaluate ALL split
points of that feature at once:

  gain(feat, split) = S_L^2 / N_L + S_R^2 / N_R      (S = sum of residuals)

which is exactly the L2 (variance-reduction) objective. Rows are subsampled
and features subsampled per tree (stochastic boosting).

Multi-step forecasting is handled by the caller (recursive strategy): the
model predicts one day ahead and predicted values are fed back as lag
features (see models/features.py + build_docs.py).
"""
import numpy as np


class _Node:
    __slots__ = ("feat", "bin_thr", "left", "right", "value")

    def __init__(self):
        self.feat = None      # None => leaf
        self.value = 0.0


class HistGBR:
    def __init__(self, n_estimators=500, lr=0.05, max_depth=6, min_leaf=40,
                 colsample=0.8, subsample=0.8, n_bins=32,
                 eval_set=None, patience=60, seed=7005):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.colsample = colsample
        self.subsample = subsample
        self.n_bins = n_bins
        self.eval_set = eval_set          # (X_val, y_val) for early stopping
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.trees_ = []
        self.importance_ = None
        self.best_m_ = None               # trees kept after early stopping

    # ---------- binning ----------
    def _make_bins(self, X):
        """Per-feature quantile edges; repeated cuts removed via np.unique."""
        self.edges_ = []
        for j in range(X.shape[1]):
            qs = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.edges_.append(np.unique(qs))

    def _bin(self, X):
        """Continuous X -> uint8 bin codes (0..n_bins-1)."""
        out = np.empty(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            out[:, j] = np.searchsorted(self.edges_[j], X[:, j], side="right")
        return out

    # ---------- training ----------
    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, f = X.shape
        self._make_bins(X)
        Xb = self._bin(X)
        self.importance_ = np.zeros(f)
        self.f0_ = float(y.mean())
        F = np.full(n, self.f0_)

        rng = self.rng
        feat_cols = max(1, int(f * self.colsample))
        row_n = max(self.min_leaf, int(n * self.subsample))

        # early-stopping state (running validation prediction, kept incremental
        # so each round costs one tree-traversal instead of re-scoring all trees)
        Xvb = self._bin(self.eval_set[0]) if self.eval_set is not None else None
        Fv = np.full(len(Xvb), self.f0_) if Xvb is not None else None
        best_val, best_m, since_best = -np.inf, 0, 0

        for m in range(self.n_estimators):
            r = y - F                                   # L2 gradient
            rows = rng.choice(n, row_n, replace=False)
            cols = rng.choice(f, feat_cols, replace=False)
            tree, gain = self._grow(Xb[rows], r[rows], cols)
            self.trees_.append(tree)
            self.importance_[cols] += gain[cols]
            F += self.lr * self._predict_tree(tree, Xb)

            if Xvb is not None:
                Fv += self.lr * self._predict_tree(tree, Xvb)
                val = -float(np.mean((self.eval_set[1] - Fv) ** 2))
                if val > best_val + 1e-12:
                    best_val, best_m, since_best = val, m + 1, 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break

        self.best_m_ = best_m if Xvb is not None else len(self.trees_)
        if self.best_m_ < len(self.trees_):             # roll back overfit tail
            self.trees_ = self.trees_[: self.best_m_]
        self.sigma_ = float(np.std(y - self._pred_raw(Xb)))   # residual spread
        return self

    def _grow(self, Xb, r, cols):
        """Grow one tree on (row-subsampled, binned) data. Returns (root, gain per feature)."""
        root = _Node()
        gain = np.zeros(Xb.shape[1])
        stack = [(root, np.arange(len(Xb)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
                node.value = r[idx].mean()
                continue
            best = None                                # (gain, feat, bin_thr)
            for j in cols:
                codes = Xb[idx, j]
                cnt = np.bincount(codes, minlength=self.n_bins).astype(float)
                ssum = np.bincount(codes, weights=r[idx], minlength=self.n_bins)
                cl = np.cumsum(cnt[:-1]); sl = np.cumsum(ssum[:-1])
                cr = cnt.sum() - cl; sr = ssum.sum() - sl
                ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
                if not ok.any():
                    continue
                g = np.where(ok, sl ** 2 / np.maximum(cl, 1) + sr ** 2 / np.maximum(cr, 1), -np.inf)
                b = int(np.argmax(g))
                if best is None or g[b] > best[0]:
                    best = (g[b], j, b)
            if best is None or best[0] <= 1e-12:
                node.value = r[idx].mean()
                continue
            _, j, b = best
            gain[j] += max(best[0], 0.0)
            mask = Xb[idx, j] <= b
            node.feat, node.bin_thr = j, b
            node.left, node.right = _Node(), _Node()
            stack.append((node.left, idx[mask], depth + 1))
            stack.append((node.right, idx[~mask], depth + 1))
        return root, gain

    # ---------- prediction ----------
    def _predict_tree(self, node, Xb):
        """Vectorised traversal: recurse by node, partition row indices with
        boolean masks — O(rows) numpy work per tree level, no Python row loop."""
        out = np.empty(len(Xb))
        stack = [(node, np.arange(len(Xb)))]
        while stack:
            nd, idx = stack.pop()
            if nd.feat is None:
                out[idx] = nd.value
                continue
            mask = Xb[idx, nd.feat] <= nd.bin_thr
            stack.append((nd.left, idx[mask]))
            stack.append((nd.right, idx[~mask]))
        return out

    def _pred_raw(self, Xb):
        F = np.full(len(Xb), self.f0_)
        for t in self.trees_:
            F += self.lr * self._predict_tree(t, Xb)
        return F

    def predict(self, X):
        Xb = self._bin(np.asarray(X, dtype=float))
        return self._pred_raw(Xb)

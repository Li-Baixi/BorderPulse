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

【中文说明 —— 梯度提升回归树（项目主预测模型，纯 NumPy 手写）】
作用：预测每个口岸每天的客流量。课程要求主模型必须自己实现，
scikit-learn 只作为基准对照。
思路（Friedman 随机梯度提升 + LightGBM 式直方图分裂）：
  1) 初始预测 F0 = 训练集均值；
  2) 每一轮：算当前残差 r = y - F，种一棵小树去拟合 r，
     然后 F ← F + 学习率 × 树输出。几百轮后 F 逼近 y。
  3) 每棵树用"直方图"法生长：先把每个特征离散成 32 个分位数桶，
     节点分裂只需对桶做 bincount 累加，就能一次性评完该特征所有
     分裂点 —— 增益 = 左残差和²/左行数 + 右残差和²/右行数
     （正是 L2 方差减少目标）。
  4) 每棵树随机抽 80% 行、80% 列（随机梯度提升，抗过拟合）。
  5) 训练时留出验证集做早停：连续 60 轮验证集无改进就停，
     并把之后长出来的"过拟合尾巴"剪掉。
多步预测由调用方用递归策略完成：模型只预测下一天，预测值再当作
滞后特征喂回去（见 features.py + build_docs.py）。
"""
import numpy as np


class _Node:
    # 中文：树节点。feat=None 表示叶子；内部节点记录分裂特征和分桶阈值。
    # __slots__ 省内存：几万行 × 几百棵树的节点数不小。
    __slots__ = ("feat", "bin_thr", "left", "right", "value")

    def __init__(self):
        self.feat = None      # None => leaf
        self.value = 0.0


class HistGBR:
    # 中文：直方图梯度提升回归器。默认 500 棵树（实际约 400 棵因早停）、
    # 学习率 0.05、深度 6、叶至少 40 行、行/列抽样各 0.8、32 个分桶。
    # eval_set 传 (X_val, y_val) 启用早停；seed 固定保证可复现。
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
        # 中文：给每个特征造分位数分桶边界（去掉重复边界——独热特征
        # 只有两个值，重复分位会切出空桶）。直方图法的关键预处理。
        self.edges_ = []
        for j in range(X.shape[1]):
            qs = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.edges_.append(np.unique(qs))

    def _bin(self, X):
        """Continuous X -> uint8 bin codes (0..n_bins-1)."""
        # 中文：把连续值映射成桶编号（uint8 足够 32 个桶），
        # 之后所有分裂都在桶编号上做，不再碰原始浮点值。
        out = np.empty(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            out[:, j] = np.searchsorted(self.edges_[j], X[:, j], side="right")
        return out

    # ---------- training ----------
    def fit(self, X, y):
        # 中文：主训练循环。分桶 → 从均值起步 → 逐轮"种树拟合残差、更新 F"，
        # 早停生效时回滚到验证集最优的那一轮。
        X = np.ascontiguousarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, f = X.shape
        self._make_bins(X)
        Xb = self._bin(X)                               # 中文：训练集桶编号矩阵
        self.importance_ = np.zeros(f)
        self.f0_ = float(y.mean())                      # 中文：初始预测 = 均值
        F = np.full(n, self.f0_)                        # 中文：当前累积预测

        rng = self.rng
        feat_cols = max(1, int(f * self.colsample))     # 中文：每棵树抽 80% 列
        row_n = max(self.min_leaf, int(n * self.subsample))  # 中文：每棵树抽 80% 行

        # early-stopping state (running validation prediction, kept incremental
        # so each round costs one tree-traversal instead of re-scoring all trees)
        # 中文：早停状态。验证集预测 Fv 增量更新——每轮只需走一遍新树，
        # 不必每轮把所有树重新打分（那是 O(轮数²) 的浪费）。
        Xvb = self._bin(self.eval_set[0]) if self.eval_set is not None else None
        Fv = np.full(len(Xvb), self.f0_) if Xvb is not None else None
        best_val, best_m, since_best = -np.inf, 0, 0

        for m in range(self.n_estimators):
            r = y - F                                   # L2 gradient
            # 中文：L2 损失的负梯度就是残差；随机抽行抽列后种树拟合残差
            rows = rng.choice(n, row_n, replace=False)
            cols = rng.choice(f, feat_cols, replace=False)
            tree, gain = self._grow(Xb[rows], r[rows], cols)
            self.trees_.append(tree)
            self.importance_[cols] += gain[cols]        # 中文：累加特征重要性
            F += self.lr * self._predict_tree(tree, Xb)  # 中文：小步更新（lr=0.05）

            if Xvb is not None:
                # 中文：验证集同样增量更新；val = 负 MSE（越大越好）。
                # 连续 patience 轮无改进 → 早停。
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
            # 中文：剪掉验证集最优轮之后的"过拟合尾巴"——实际保留约 400 棵
            self.trees_ = self.trees_[: self.best_m_]
        self.sigma_ = float(np.std(y - self._pred_raw(Xb)))   # residual spread
        # 中文：训练残差标准差 → 前端不确定性区间和排队模拟的 σ 都用它
        return self

    def _grow(self, Xb, r, cols):
        """Grow one tree on (row-subsampled, binned) data. Returns (root, gain per feature)."""
        # 中文：种一棵树（用栈代替递归）。对每个候选特征做 bincount
        # 累加 → 一次算完所有分裂点的增益 → 取全局最优分裂。
        root = _Node()
        gain = np.zeros(Xb.shape[1])
        stack = [(root, np.arange(len(Xb)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
                node.value = r[idx].mean()             # 中文：太深/太少行 → 叶子（输出=残差均值）
                continue
            best = None                                # (gain, feat, bin_thr)
            for j in cols:
                # 中文：直方图分裂——对桶编号 bincount：每桶行数 cnt、
                # 每桶残差和 ssum；前缀和展开成"左/右子树"的 N 与 S，
                # 即可向量化评完该特征所有分桶阈值。
                codes = Xb[idx, j]
                cnt = np.bincount(codes, minlength=self.n_bins).astype(float)
                ssum = np.bincount(codes, weights=r[idx], minlength=self.n_bins)
                cl = np.cumsum(cnt[:-1]); sl = np.cumsum(ssum[:-1])
                cr = cnt.sum() - cl; sr = ssum.sum() - sl
                ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)  # 中文：两侧都够 min_leaf
                if not ok.any():
                    continue
                # 中文：L2 增益 = S_L²/N_L + S_R²/N_R；不合法位置填 -inf
                g = np.where(ok, sl ** 2 / np.maximum(cl, 1) + sr ** 2 / np.maximum(cr, 1), -np.inf)
                b = int(np.argmax(g))                   # 中文：该特征最优分桶阈值
                if best is None or g[b] > best[0]:
                    best = (g[b], j, b)
            if best is None or best[0] <= 1e-12:
                node.value = r[idx].mean()             # 中文：找不到有效分裂 → 叶子
                continue
            _, j, b = best
            gain[j] += max(best[0], 0.0)               # 中文：按特征累计增益 → 重要性
            mask = Xb[idx, j] <= b
            node.feat, node.bin_thr = j, b             # 中文：记录分裂规则，子节点入栈
            node.left, node.right = _Node(), _Node()
            stack.append((node.left, idx[mask], depth + 1))
            stack.append((node.right, idx[~mask], depth + 1))
        return root, gain

    # ---------- prediction ----------
    def _predict_tree(self, node, Xb):
        """Vectorised traversal: recurse by node, partition row indices with
        boolean masks — O(rows) numpy work per tree level, no Python row loop."""
        # 中文：向量化走树——按节点切分行索引（布尔掩码），
        # 每层只做 O(行数) 的 NumPy 运算，避免逐行 Python 循环。
        out = np.empty(len(Xb))
        stack = [(node, np.arange(len(Xb)))]
        while stack:
            nd, idx = stack.pop()
            if nd.feat is None:
                out[idx] = nd.value                     # 中文：到叶子 → 填叶值
                continue
            mask = Xb[idx, nd.feat] <= nd.bin_thr
            stack.append((nd.left, idx[mask]))          # 中文：≤ 阈值走左，否则走右
            stack.append((nd.right, idx[~mask]))
        return out

    def _pred_raw(self, Xb):
        # 中文：整模型预测 = 均值起步 + 每棵树的小步更新求和
        F = np.full(len(Xb), self.f0_)
        for t in self.trees_:
            F += self.lr * self._predict_tree(t, Xb)
        return F

    def predict(self, X):
        # 中文：对外接口——先分桶再累加所有树
        Xb = self._bin(np.asarray(X, dtype=float))
        return self._pred_raw(Xb)

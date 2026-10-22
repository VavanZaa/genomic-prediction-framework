"""Empirical training support: distance to the k-th nearest training animal.

Min–max bounds call any point "inside" if it lies between the two most extreme training animals, so
a single isolated animal widens the whole region and a gap between clusters counts as supported.
Here a point is supported only when it lies as close to its k-th nearest training animal as typical
training animals do.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree


class KNNSupport:
    """Leave-one-out k-nearest-neighbour support region of a training sample.

    Features are standardised with training means and SDs. d_k(x) is the distance to the k-th nearest
    training point; a training point's own d_k leaves it out. The support radius r is the `level`
    quantile of the training d_k, so a fraction 1 - level of training animals lie just outside by
    construction. A query is supported when d_k(x) <= r. k defaults to 1 % of the training sample
    (at least 5).
    """

    def __init__(self, k: Optional[int] = None, level: float = 0.95):
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        self.k, self.level = k, level

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[:, None] if X.ndim == 1 else X

    def fit(self, X) -> "KNNSupport":
        X = self._as_2d(X)
        n = len(X)
        if n < 3:
            raise ValueError("Support needs at least three training points")
        if not np.isfinite(X).all():
            raise ValueError("Support features must be finite")
        self.k_ = min(int(self.k) if self.k is not None else max(5, int(round(0.01 * n))), n - 1)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mean_) / self.scale_
        self.tree_ = cKDTree(Z)
        d, _ = self.tree_.query(Z, k=self.k_ + 1)      # the first neighbour of a training point is itself
        self.train_distance_ = d[:, -1]
        self.radius_ = float(np.quantile(self.train_distance_, self.level))
        return self

    def _scaled(self, X) -> np.ndarray:
        X = self._as_2d(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("Feature count differs from the training sample")
        return (X - self.mean_) / self.scale_

    def distance(self, X) -> np.ndarray:
        """Distance to the k-th nearest training point, in training-SD units."""
        d, _ = self.tree_.query(self._scaled(X), k=self.k_)
        return d if self.k_ == 1 else d[:, -1]

    def ratio(self, X) -> np.ndarray:
        """d_k(x) / r: at most 1 inside the support."""
        return self.distance(X) / max(self.radius_, 1e-12)

    def excess(self, X) -> np.ndarray:
        """Training-SD distance beyond the support radius (0 inside the support)."""
        return np.maximum(0.0, self.distance(X) - self.radius_)

    def supported(self, X) -> np.ndarray:
        return self.distance(X) <= self.radius_ + 1e-12

    def local_count(self, X) -> np.ndarray:
        """Number of training points within the support radius of each query."""
        return np.asarray(self.tree_.query_ball_point(self._scaled(X), self.radius_, return_length=True))

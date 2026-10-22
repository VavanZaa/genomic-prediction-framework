"""Phenotypic prediction models: Ancestry-only, Additive GBLUP, Additive+Dominance, and Non-linear RKHS.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from ..real_data.genotype_preprocessing import GenotypeImputer


@dataclass
class EvaluationMetrics:
    model_name: str
    trait_name: str
    pearson_r: float
    spearman_rho: float
    mse: float
    rmse: float


class AncestryPredictor:
    """Model 1: Ancestry-only baseline. Predicts phenotype purely from pedigree/genomic breed proportions."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.breed_names: List[str] = []

    def fit(self, breed_fractions: np.ndarray, y: np.ndarray, breed_names: Optional[List[str]] = None) -> AncestryPredictor:
        self.breed_names = breed_names or [f"B_{i}" for i in range(breed_fractions.shape[1])]
        self.model.fit(breed_fractions, y)
        return self

    def predict(self, breed_fractions: np.ndarray) -> np.ndarray:
        return self.model.predict(breed_fractions)


class AdditiveGBLUP:
    """Model 2: Additive GBLUP / Ridge Regression (VanRaden Genomic Relationship Matrix G)."""

    def __init__(self, alpha: Optional[float] = None):
        self.alpha = alpha
        self.model: Optional[Ridge] = None
        self.allele_frequencies: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0

    def fit(self, marker_matrix: np.ndarray, y: np.ndarray) -> AdditiveGBLUP:
        raw_marker_matrix = np.asarray(marker_matrix)
        self.imputer_ = GenotypeImputer()
        marker_matrix = self.imputer_.fit_transform(marker_matrix)
        N, P = marker_matrix.shape
        # Compute allele frequencies strictly from training data
        p = np.clip(np.mean(marker_matrix, axis=0) / 2.0, 1e-4, 1.0 - 1e-4)
        self.allele_frequencies = p

        # Center markers: Z = M - 2p
        Z = marker_matrix - 2.0 * p
        # VanRaden scaling: 2 * sum(p * (1 - p))
        self.scale_factor = np.sqrt(2.0 * np.sum(p * (1.0 - p)) + 1e-8)
        Z_scaled = Z / self.scale_factor

        if self.alpha is None:
            # Cross-validated ridge parameter
            alphas = np.logspace(-2, 4, 20)
            ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True)
            ridge_cv.fit(Z_scaled, y)
            self.model = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
            self.model.fit(Z_scaled, y)
        else:
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(Z_scaled, y)

        return self

    def predict(self, marker_matrix: np.ndarray) -> np.ndarray:
        if self.model is None or self.allele_frequencies is None:
            raise RuntimeError("Model must be fitted before prediction.")
        marker_matrix = self.imputer_.transform(marker_matrix)
        Z = marker_matrix - 2.0 * self.allele_frequencies
        Z_scaled = Z / self.scale_factor
        return self.model.predict(Z_scaled)

    def linear_coefficients(self) -> Tuple[float, np.ndarray]:
        """(c0, beta) with predict(M) == c0 + M @ beta.

        Because the model is linear in allele counts, an offspring's prediction splits into one
        term per parental gamete, which lets pair-level offspring distributions be sampled per parent.
        """
        if self.model is None or self.allele_frequencies is None:
            raise RuntimeError("Model must be fitted before prediction.")
        beta = self.model.coef_ / self.scale_factor
        c0 = float(self.model.intercept_ - np.dot(2.0 * self.allele_frequencies, beta))
        return c0, beta


class AdditiveDominanceGBLUP:
    """Model 3: Additive + Dominance GBLUP (Su et al. 2012 / Vitezica et al. 2013)."""

    def __init__(self, alpha: Optional[float] = None):
        self.alpha = alpha
        self.model: Optional[Ridge] = None
        self.allele_frequencies: Optional[np.ndarray] = None
        self.scale_a: float = 1.0
        self.scale_d: float = 1.0

    def _compute_dominance_matrix(self, M: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Dominance coding: W_ik = -2p^2 for 0; 2p(1-p) for 1; -2(1-p)^2 for 2."""
        N, P = M.shape
        W = np.zeros_like(M, dtype=np.float64)
        # M == 0
        zero_mask = (M == 0)
        W[zero_mask] = -2.0 * (p**2)[np.newaxis, :].repeat(N, axis=0)[zero_mask]
        # M == 1
        het_mask = (M == 1)
        W[het_mask] = 2.0 * (p * (1.0 - p))[np.newaxis, :].repeat(N, axis=0)[het_mask]
        # M == 2
        two_mask = (M == 2)
        W[two_mask] = -2.0 * ((1.0 - p)**2)[np.newaxis, :].repeat(N, axis=0)[two_mask]
        return W

    def fit(self, marker_matrix: np.ndarray, y: np.ndarray) -> AdditiveDominanceGBLUP:
        raw_marker_matrix = np.asarray(marker_matrix)
        self.imputer_ = GenotypeImputer()
        marker_matrix = self.imputer_.fit_transform(marker_matrix)
        N, P = marker_matrix.shape
        p = np.clip(np.mean(marker_matrix, axis=0) / 2.0, 1e-4, 1.0 - 1e-4)
        self.allele_frequencies = p

        Z = marker_matrix - 2.0 * p
        W = self._compute_dominance_matrix(marker_matrix, p)

        self.scale_a = np.sqrt(2.0 * np.sum(p * (1.0 - p)) + 1e-8)
        self.scale_d = np.sqrt(np.sum((2.0 * p * (1.0 - p))**2) + 1e-8)

        Z_scaled = Z / self.scale_a
        W_scaled = W / self.scale_d

        # Concatenate design matrix: [Z_scaled, W_scaled]
        X = np.hstack([Z_scaled, W_scaled])

        if self.alpha is None:
            alphas = np.logspace(-2, 4, 20)
            ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True)
            ridge_cv.fit(X, y)
            self.model = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
            self.model.fit(X, y)
        else:
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(X, y)

        return self

    def predict(self, marker_matrix: np.ndarray) -> np.ndarray:
        if self.model is None or self.allele_frequencies is None:
            raise RuntimeError("Model must be fitted before prediction.")
        marker_matrix = self.imputer_.transform(marker_matrix)
        p = self.allele_frequencies
        Z = marker_matrix - 2.0 * p
        W = self._compute_dominance_matrix(marker_matrix, p)

        Z_scaled = Z / self.scale_a
        W_scaled = W / self.scale_d
        X = np.hstack([Z_scaled, W_scaled])
        return self.model.predict(X)


class RkhsPredictor:
    """Model 4: Reproducing Kernel Hilbert Space (RKHS) / Interaction-capable Kernel Ridge Regression.

    Uses Gaussian RBF kernel on standardized markers to capture multi-locus epistasis and interactions.
    Any of gamma / alpha left as None is tuned by K-fold cross-validation on the training data only
    (gamma over multiples of the median-distance heuristic).
    """

    def __init__(
        self,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        gamma_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
        n_folds: int = 5,
        random_state: int = 0
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.gamma_scales = gamma_scales
        self.n_folds = n_folds
        self.random_state = random_state
        self.gamma_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.X_train: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.mean_markers_: Optional[np.ndarray] = None
        self.std_markers_: Optional[np.ndarray] = None

    def _pairwise_sq_dist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1 x2^T
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
        sq_dist = X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T)
        return np.maximum(sq_dist, 0.0)

    def fit(self, marker_matrix: np.ndarray, y: np.ndarray) -> RkhsPredictor:
        raw_marker_matrix = np.asarray(marker_matrix)
        self.imputer_ = GenotypeImputer()
        marker_matrix = self.imputer_.fit_transform(marker_matrix)
        N, P = marker_matrix.shape
        self.mean_markers_ = np.mean(marker_matrix, axis=0)
        self.std_markers_ = np.std(marker_matrix, axis=0) + 1e-6
        X_norm = (marker_matrix - self.mean_markers_) / self.std_markers_

        self.X_train = X_norm
        sq_dist = self._pairwise_sq_dist(X_norm, X_norm)

        if self.gamma is not None:
            gammas = [self.gamma]
        else:
            # Median heuristic for RBF bandwidth, then scaled candidates
            positive = sq_dist[sq_dist > 0]
            med = float(np.median(positive)) if positive.size else 1.0
            gammas = [s / (med + 1e-8) for s in self.gamma_scales]
        alphas = [self.alpha] if self.alpha is not None else list(self.alpha_grid)

        if len(gammas) * len(alphas) > 1 and N >= 2 * self.n_folds:
            best_error=np.inf
            raw=raw_marker_matrix
            folds=np.array_split(np.random.default_rng(self.random_state).permutation(N),self.n_folds)
            for gi,scale in enumerate(([None] if self.gamma is not None else self.gamma_scales)):
                for alpha in alphas:
                    error=0.
                    for va in folds:
                        tr=np.setdiff1d(np.arange(N),va)
                        # A one-candidate model uses statistics from its own fit rows.
                        fold=RkhsPredictor(gamma=self.gamma,alpha=alpha,gamma_scales=(scale,) if scale is not None else (1.,))
                        fold.fit(raw[tr],np.asarray(y)[tr])
                        error+=float(np.sum((np.asarray(y)[va]-fold.predict(raw[va]))**2))
                    if error<best_error:
                        best_error=error; self.gamma_,self.alpha_=gammas[gi],alpha
        else:
            self.gamma_, self.alpha_ = gammas[0], alphas[0]

        K = np.exp(-self.gamma_ * sq_dist)
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_

        # Dual solve: (K + alpha * I) alpha_vec = y_centered
        self.dual_coef_ = np.linalg.solve(K + self.alpha_ * np.eye(N), y_centered)
        return self

    def _select_by_cv(self, sq_dist: np.ndarray, y: np.ndarray, gammas: List[float], alphas: List[float]) -> Tuple[float, float]:
        """Pick (gamma, alpha) with the lowest K-fold out-of-fold MSE; uses training rows only."""
        n = len(y)
        folds = np.array_split(np.random.default_rng(self.random_state).permutation(n), self.n_folds)
        best, best_mse = (gammas[0], alphas[0]), np.inf
        for g in gammas:
            K_full = np.exp(-g * sq_dist)
            for a in alphas:
                sq_err = 0.0
                for val in folds:
                    tr = np.setdiff1d(np.arange(n), val)
                    mu = float(np.mean(y[tr]))
                    coef = np.linalg.solve(K_full[np.ix_(tr, tr)] + a * np.eye(len(tr)), y[tr] - mu)
                    sq_err += float(np.sum((y[val] - mu - K_full[np.ix_(val, tr)] @ coef) ** 2))
                if sq_err < best_mse:
                    best, best_mse = (g, a), sq_err
        return best

    def predict(self, marker_matrix: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.dual_coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        marker_matrix = self.imputer_.transform(marker_matrix)
        X_norm = (marker_matrix - self.mean_markers_) / self.std_markers_
        sq_dist = self._pairwise_sq_dist(X_norm, self.X_train)
        K_test = np.exp(-self.gamma_ * sq_dist)
        return self.intercept_ + np.dot(K_test, self.dual_coef_)


class PredictionBenchmark:
    """Orchestrates comparison of the 4 predictors on training and withheld test cohorts."""

    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, trait_name: str) -> EvaluationMetrics:
        r, _ = stats.pearsonr(y_true, y_pred)
        rho, _ = stats.spearmanr(y_true, y_pred)
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        return EvaluationMetrics(
            model_name=model_name,
            trait_name=trait_name,
            pearson_r=float(r),
            spearman_rho=float(rho),
            mse=mse,
            rmse=rmse
        )

    @classmethod
    def run_comparison(
        cls,
        train_ancestry: np.ndarray,
        train_markers: np.ndarray,
        train_y: np.ndarray,
        test_ancestry: np.ndarray,
        test_markers: np.ndarray,
        test_y: np.ndarray,
        trait_name: str
    ) -> Dict[str, EvaluationMetrics]:
        models = {
            "AncestryOnly": AncestryPredictor(),
            "AdditiveGBLUP": AdditiveGBLUP(),
            "AdditiveDominance": AdditiveDominanceGBLUP(),
            "NonlinearRKHS": RkhsPredictor()
        }

        results = {}

        # 1. Ancestry Only
        m1 = models["AncestryOnly"]
        m1.fit(train_ancestry, train_y)
        p1 = m1.predict(test_ancestry)
        results["AncestryOnly"] = cls.evaluate_predictions(test_y, p1, "AncestryOnly", trait_name)

        # 2. Additive GBLUP
        m2 = models["AdditiveGBLUP"]
        m2.fit(train_markers, train_y)
        p2 = m2.predict(test_markers)
        results["AdditiveGBLUP"] = cls.evaluate_predictions(test_y, p2, "AdditiveGBLUP", trait_name)

        # 3. Additive + Dominance GBLUP
        m3 = models["AdditiveDominance"]
        m3.fit(train_markers, train_y)
        p3 = m3.predict(test_markers)
        results["AdditiveDominance"] = cls.evaluate_predictions(test_y, p3, "AdditiveDominance", trait_name)

        # 4. RKHS
        m4 = models["NonlinearRKHS"]
        m4.fit(train_markers, train_y)
        p4 = m4.predict(test_markers)
        results["NonlinearRKHS"] = cls.evaluate_predictions(test_y, p4, "NonlinearRKHS", trait_name)

        return results

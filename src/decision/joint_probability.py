"""Joint probability estimation P(trait_1 >= target_1 AND trait_2 >= target_2 | parents, data).
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy import stats


class JointProbabilityEstimator:
    """Estimates bivariate joint probability of exceeding production thresholds, preserving covariance."""

    @staticmethod
    def estimate_from_samples(
        trait1_samples: np.ndarray,
        trait2_samples: np.ndarray,
        target_1: float,
        target_2: float,
        method: str = "empirical"
    ) -> float:
        """Estimate P(Y1 >= target_1 and Y2 >= target_2).

        Methods:
        - 'empirical': proportion of sampled virtual offspring satisfying both criteria.
        - 'bivariate_normal': fits bivariate normal (mu, Sigma) and integrates upper quadrant.
        """
        assert len(trait1_samples) == len(trait2_samples)
        N = len(trait1_samples)
        if N == 0:
            return 0.0

        if method == "empirical":
            success = (trait1_samples >= target_1) & (trait2_samples >= target_2)
            return float(np.mean(success))

        elif method == "bivariate_normal":
            # Parametric approximation preserving covariance
            mu = np.array([np.mean(trait1_samples), np.mean(trait2_samples)])
            cov = np.cov(trait1_samples, trait2_samples)
            if np.linalg.matrix_rank(cov) < 2 or np.isnan(cov).any():
                # Fallback to empirical if covariance is degenerate
                return float(np.mean((trait1_samples >= target_1) & (trait2_samples >= target_2)))

            # P(Y1 >= t1, Y2 >= t2) = 1 - P(Y1 < t1) - P(Y2 < t2) + P(Y1 < t1, Y2 < t2)
            mvn = stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)
            p_joint_lower = mvn.cdf([target_1, target_2])
            p_t1_lower = stats.norm.cdf(target_1, loc=mu[0], scale=np.sqrt(cov[0, 0] + 1e-8))
            p_t2_lower = stats.norm.cdf(target_2, loc=mu[1], scale=np.sqrt(cov[1, 1] + 1e-8))

            p_upper = 1.0 - p_t1_lower - p_t2_lower + p_joint_lower
            return float(np.clip(p_upper, 0.0, 1.0))

        else:
            raise ValueError(f"Unknown estimation method: {method}")

    @staticmethod
    def estimate_with_residuals(
        genetic1: np.ndarray,
        genetic2: np.ndarray,
        target_1: float,
        target_2: float,
        residual_noise: np.ndarray
    ) -> np.ndarray:
        """P(g1 + e1 >= t1 and g2 + e2 >= t2), averaged over genetic samples and residual draws.

        genetic1/genetic2: (..., K) predicted genetic values of K virtual offspring (the leading
        axes index pairs). residual_noise: (R, 2) draws of the residual (environment + prediction
        error), shared by all pairs so pairs are compared on common random numbers.
        Returns an array with the leading shape of the inputs.
        """
        e1 = residual_noise[:, 0]
        e2 = residual_noise[:, 1]
        ok = (genetic1[..., None] + e1 >= target_1) & (genetic2[..., None] + e2 >= target_2)
        return ok.mean(axis=(-2, -1))

    @staticmethod
    def out_of_fold_residual_covariance(
        markers: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        model_factory: Callable[[], object],
        n_folds: int = 5,
        seed: int = 0
    ) -> np.ndarray:
        """2x2 covariance of out-of-fold prediction residuals (y - y_hat) for two traits.

        Uses training data only. The residual carries environmental noise plus prediction error, so
        adding it to predicted genetic values gives a predictive distribution for phenotypes that
        keeps the residual correlation between the two traits.
        """
        n = len(y1)
        folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
        residuals = np.zeros((n, 2))
        for val in folds:
            tr = np.setdiff1d(np.arange(n), val)
            m1 = model_factory().fit(markers[tr], y1[tr])
            m2 = model_factory().fit(markers[tr], y2[tr])
            residuals[val, 0] = y1[val] - m1.predict(markers[val])
            residuals[val, 1] = y2[val] - m2.predict(markers[val])
        return np.cov(residuals, rowvar=False)

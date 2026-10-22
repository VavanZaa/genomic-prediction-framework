"""The vectorized bivariate normal CDF agrees with SciPy's numerical integration."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from src.decision.bivariate_normal import bivariate_normal_cdf


@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.1, 0.0, 0.02, 0.3, 0.7, 0.9])
def test_matches_scipy(rho):
    grid = np.array([-2.5, -1.0, -0.2, 0.0, 0.4, 1.3, 2.8])
    h, k = np.meshgrid(grid, grid)
    ours = bivariate_normal_cdf(h, k, rho)
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(np.column_stack([h.ravel(), k.ravel()])).reshape(h.shape)
    np.testing.assert_allclose(ours, ref, atol=2e-5)


def test_broadcasts_over_pairwise_correlations():
    rho = np.array([[0.1, -0.4], [0.6, 0.0]])
    p = bivariate_normal_cdf(0.5, -0.3, rho)
    assert p.shape == (2, 2)
    assert p[1, 1] == pytest.approx(0.6914624612740131 * 0.3820885778110474, abs=1e-9)

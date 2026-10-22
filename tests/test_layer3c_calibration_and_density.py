"""Unit and integration tests for Layer 3C probability calibration and Gate A density metrics."""
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.prediction.calibration import ProbabilityCalibrationEngine, CalibrationMetrics


def test_brier_decomposition_identity():
    """Verify that Brier score equals Reliability - Resolution + Uncertainty."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.clip(rng.uniform(0.1, 0.9, size=500), 0.01, 0.99)

    brier, rel, res, unc = ProbabilityCalibrationEngine.compute_brier_decomposition(y_true, y_prob, n_bins=10)

    # Decomposed identity: BS approx REL - RES + UNC
    reconstructed_brier = rel - res + unc
    assert abs(brier - reconstructed_brier) < 0.05, f"Brier decomposition mismatch: {brier} vs {reconstructed_brier}"


def test_logistic_calibration_perfect_model():
    """Verify that a well-calibrated probability generator yields alpha approx 0 and beta approx 1."""
    rng = np.random.default_rng(123)
    n = 2000
    # True latent log-odds
    latent = rng.normal(0, 1, size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y_obs = (rng.uniform(0, 1, size=n) < true_prob).astype(int)

    metrics = ProbabilityCalibrationEngine.evaluate_calibration(y_obs, true_prob, n_bins=10)

    # In a well-calibrated sample, alpha should be near 0 and beta near 1
    assert abs(metrics.calibration_intercept_alpha) < 0.25, f"Alpha too far from 0: {metrics.calibration_intercept_alpha}"
    assert abs(metrics.calibration_slope_beta - 1.0) < 0.25, f"Beta too far from 1: {metrics.calibration_slope_beta}"
    assert metrics.expected_calibration_error < 0.05, f"ECE too high: {metrics.expected_calibration_error}"


def test_contemporary_group_shift_impact():
    """Verify that an unseen positive shift in test phenotypes systematically inflates predicted attainment."""
    rng = np.random.default_rng(42)
    n = 500

    # Training distribution: N(0, 1)
    # Threshold for attainment: top 50% (tau = 0.0, favorable is <= 0.0)
    tau = 0.0

    # Predicted probabilities assuming historical N(0, 1)
    pred_vals = rng.normal(0, 0.2, size=n)
    p_pred = norm.cdf((tau - pred_vals) / 1.0)

    # Actual test cohort shifted by +1.5 standard deviations (like hair length in Project Wes)
    actual_test_vals = pred_vals + rng.normal(1.5, 1.0, size=n)
    actual_attainment = (actual_test_vals <= tau).astype(int)

    metrics = ProbabilityCalibrationEngine.evaluate_calibration(actual_attainment, p_pred)

    # Model predicted ~50% attainment
    assert abs(metrics.mean_predicted_prob - 0.50) < 0.05
    # Realized attainment should be much lower (< 20%) due to +1.5 shift
    assert metrics.observed_attainment_rate < 0.20
    # ECE should be large (> 20%)
    assert metrics.expected_calibration_error > 0.20
    # Intercept alpha should be strongly negative
    assert metrics.calibration_intercept_alpha < -1.0


def test_marginal_vs_joint_diagnosis_structure():
    """Verify diagnose_marginal_vs_joint returns metrics for both marginals and joint target."""
    n = 200
    y_test_l = np.random.normal(30, 8, size=n)
    y_test_h = np.random.normal(7, 2, size=n)
    p_l = np.random.normal(30, 2, size=n)
    p_h = np.random.normal(7, 1, size=n)

    cov_res = np.array([[64.0, 0.0], [0.0, 4.0]])

    diag = ProbabilityCalibrationEngine.diagnose_marginal_vs_joint(
        y_test_load=y_test_l,
        y_test_hair=y_test_h,
        pred_load=p_l,
        pred_hair=p_h,
        tau_load=30.0,
        tau_hair=7.0,
        cov_residual=cov_res,
        model_name="TestModel"
    )

    assert "LoadSlope" in diag
    assert "AvgShortLength" in diag
    assert "JointTarget" in diag
    assert isinstance(diag["JointTarget"], CalibrationMetrics)
    assert 0.0 <= diag["JointTarget"].expected_calibration_error <= 1.0

"""Probability calibration and diagnostic evaluation engine for crossbreeding predictions.

Evaluates:
1. Logistic calibration parameters: logit(P(Y=1)) = alpha + beta * logit(p_hat)
   where alpha = 0 (calibration-in-the-large / intercept) and beta = 1 (calibration slope)
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across binned predictions
3. Brier score and Brier score decomposition (Reliability, Resolution, Uncertainty)
4. Marginal vs Joint target calibration diagnosis under contemporary-group / environmental shifts
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class CalibrationMetrics:
    """Quantitative calibration scorecard for binary target predictions."""
    model_name: str
    target_name: str
    num_samples: int
    mean_predicted_prob: float
    observed_attainment_rate: float
    calibration_intercept_alpha: float
    calibration_slope_beta: float
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    roc_auc: float
    binned_reliability_table: pd.DataFrame


class ProbabilityCalibrationEngine:
    """Evaluates and recalibrates predicted target attainment probabilities."""

    @staticmethod
    def compute_brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, float, float]:
        """Decomposes Brier score into: BS = Reliability - Resolution + Uncertainty.
        
        - Reliability (REL): Weighted squared deviation between predicted probabilities and observed frequencies. Ideal = 0.
        - Resolution (RES): How much bin event rates differ from the base rate. Higher is better.
        - Uncertainty (UNC): Base rate variance p_bar * (1 - p_bar). Inherent to data.
        """
        N = len(y_true)
        base_rate = float(np.mean(y_true))
        uncertainty = base_rate * (1.0 - base_rate)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        reliability = 0.0
        resolution = 0.0

        for b in range(n_bins):
            mask = (bin_indices == b)
            n_k = np.sum(mask)
            if n_k > 0:
                p_k_bar = float(np.mean(y_prob[mask]))
                o_k_bar = float(np.mean(y_true[mask]))
                weight = n_k / N
                reliability += weight * (p_k_bar - o_k_bar) ** 2
                resolution += weight * (o_k_bar - base_rate) ** 2

        brier = float(np.mean((y_prob - y_true) ** 2))
        return brier, reliability, resolution, uncertainty

    @classmethod
    def evaluate_calibration(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        target_name: str = "JointTarget",
        n_bins: int = 10,
        eps: float = 1e-6
    ) -> CalibrationMetrics:
        """Computes logistic calibration parameters, ECE, MCE, and reliability table."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        N = len(y_true)

        # 1. Base rates
        mean_pred = float(np.mean(y_prob))
        obs_rate = float(np.mean(y_true))

        # 2. Logistic calibration regression: logit(P(Y=1)) = alpha + beta * logit(p_hat)
        p_clipped = np.clip(y_prob, eps, 1.0 - eps)
        logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        # Handle degenerate cases (all 0s or all 1s in y_true)
        if len(np.unique(y_true)) < 2:
            alpha = -999.0 if obs_rate == 0.0 else 999.0
            beta = 0.0
        else:
            # Strictly unregularized maximum-likelihood logistic regression (C=np.inf)
            lr = LogisticRegression(C=np.inf, solver="lbfgs")
            lr.fit(logit_p, y_true)
            alpha = float(lr.intercept_[0])
            beta = float(lr.coef_[0][0])

        # 3. Binned reliability curve & ECE
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_rows = []
        ece = 0.0
        mce = 0.0

        for b in range(n_bins):
            mask = (bin_indices == b)
            n_in_bin = int(np.sum(mask))
            if n_in_bin > 0:
                p_bin_mean = float(np.mean(y_prob[mask]))
                o_bin_mean = float(np.mean(y_true[mask]))
                bin_err = abs(p_bin_mean - o_bin_mean)
                weight = n_in_bin / N
                ece += weight * bin_err
                if bin_err > mce:
                    mce = bin_err

                bin_rows.append({
                    "bin": b,
                    "range_low": round(float(bin_edges[b]), 2),
                    "range_high": round(float(bin_edges[b + 1]), 2),
                    "count": n_in_bin,
                    "mean_pred": round(p_bin_mean, 4),
                    "observed_freq": round(o_bin_mean, 4),
                    "abs_error": round(bin_err, 4)
                })

        df_rel = pd.DataFrame(bin_rows)

        # 4. Brier decomposition
        brier, rel, res, unc = cls.compute_brier_decomposition(y_true, y_prob, n_bins=n_bins)

        # 5. ROC-AUC
        try:
            auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return CalibrationMetrics(
            model_name=model_name,
            target_name=target_name,
            num_samples=N,
            mean_predicted_prob=mean_pred,
            observed_attainment_rate=obs_rate,
            calibration_intercept_alpha=alpha,
            calibration_slope_beta=beta,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            brier_reliability=rel,
            brier_resolution=res,
            brier_uncertainty=unc,
            roc_auc=auc,
            binned_reliability_table=df_rel
        )

    @classmethod
    def diagnose_marginal_vs_joint(
        cls,
        y_test_load: np.ndarray,
        y_test_hair: np.ndarray,
        pred_load: np.ndarray,
        pred_hair: np.ndarray,
        tau_load: float,
        tau_hair: float,
        cov_residual: np.ndarray,
        model_name: str = "GBLUP"
    ) -> Dict[str, CalibrationMetrics]:
        """Diagnoses why joint probability diverges by separating marginal and joint calibration."""
        # Marginal predictions and outcomes
        std_load_res = np.sqrt(cov_residual[0, 0])
        std_hair_res = np.sqrt(cov_residual[1, 1])

        p_marg_load = norm.cdf((tau_load - pred_load) / std_load_res)
        p_marg_hair = norm.cdf((tau_hair - pred_hair) / std_hair_res)

        y_true_load = (y_test_load <= tau_load).astype(int)
        y_true_hair = (y_test_hair <= tau_hair).astype(int)

        # Joint predictions via bivariate normal CDF
        joint_probs = []
        for i in range(len(pred_load)):
            mvn = multivariate_normal(mean=[pred_load[i], pred_hair[i]], cov=cov_residual, allow_singular=True)
            prob = mvn.cdf([tau_load, tau_hair])
            joint_probs.append(prob)
        joint_probs = np.array(joint_probs)

        y_true_joint = (y_true_load & y_true_hair).astype(int)

        cal_load = cls.evaluate_calibration(y_true_load, p_marg_load, model_name=model_name, target_name="Marginal_LoadSlope")
        cal_hair = cls.evaluate_calibration(y_true_hair, p_marg_hair, model_name=model_name, target_name="Marginal_AvgShortLength")
        cal_joint = cls.evaluate_calibration(y_true_joint, joint_probs, model_name=model_name, target_name="Joint_Load_and_Hair")

        return {
            "LoadSlope": cal_load,
            "AvgShortLength": cal_hair,
            "JointTarget": cal_joint
        }

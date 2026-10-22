"""Real-Data Blind Population Reconstruction (Layer 6B-v2).

Validates the extrapolation and cross-design capabilities of the engine by hiding a known population:
    Source Training Population (Seminole Multibreed) -> Target Hidden Population (West Florida Commercial Herd)

1. Completely masks an identifiable commercial composite population C (by population identity, with zero q_B leakage).
2. Trains genomic prediction models exclusively on source population data.
3. Asks the unconstrained optimization engine to find the optimal crossing strategy:
    q* = argmax_q U(q)
    evaluating thermotolerance and adaptation merit under the modeled traits.
4. Only AFTER optimization reveals the hidden population's empirical ancestry distribution and compares:
    - q* vs Historical Brangus Standard (q_B = 0.375)
    - q* vs Empirical Realized Commercial Mean (q_B_empirical_hidden)
    - Distribution overlap (Wasserstein distance W_1) against real masked cattle C
    - Historical composite coverage probability: P(real C in predicted optimum region)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .support import KNNSupport


@dataclass
class MaskedReconstructionResult:
    """Outcomes of real-data blind population reconstruction benchmark."""
    predicted_optimal_qb: float
    predicted_safe_qb: float
    d_ood_at_opt: float
    confidence_at_opt: float
    is_opt_within_support: bool
    historical_target_qb: float
    empirical_hidden_mean_qb: float
    empirical_hidden_std_qb: float
    error_vs_historical: float
    error_vs_empirical: float
    real_masked_count: int
    train_founder_count: int
    predicted_mean_load: float
    real_masked_mean_load: float
    predicted_std_load: float
    real_masked_std_load: float
    mean_bias_load: float
    std_bias_load: float
    wasserstein_distance_load: float
    w1_ancestry_only: float
    w1_training_mean: float
    coverage_probability_90pct: float
    q_grid: np.ndarray = field(repr=False)
    utility_profile: np.ndarray = field(repr=False)
    load_profile: np.ndarray = field(repr=False)
    heterosis_profile: np.ndarray = field(repr=False)
    d_ood_profile: np.ndarray = field(repr=False)
    confidence_profile: np.ndarray = field(repr=False)
    density_profile: np.ndarray = field(repr=False)
    fitted_intercept: float = float("nan")
    fitted_beta: float = float("nan")
    fitted_heterosis: float = float("nan")
    residual_sd: float = float("nan")
    parameter_source: str = ""
    optimum_at_boundary: bool = False
    support_k: int = 0                         # k of the k-NN ancestry support
    support_radius_sd: float = float("nan")    # leave-one-out 95 % k-NN radius, training-SD units

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Parameter_Source": self.parameter_source,
            "Fitted_Intercept": self.fitted_intercept,
            "Fitted_Beta_Brahman": self.fitted_beta,
            "Fitted_Heterosis_h": self.fitted_heterosis,
            "Residual_SD": self.residual_sd,
            "Optimum_At_Boundary": self.optimum_at_boundary,
            "Predicted_Optimal_qB": self.predicted_optimal_qb,
            "Predicted_Safe_qB": self.predicted_safe_qb,
            "d_OOD_at_Opt": self.d_ood_at_opt,
            "Confidence_at_Opt": self.confidence_at_opt,
            "Is_Opt_Within_Support": self.is_opt_within_support,
            "Support_Rule": "leave-one-out kNN, 95% radius",
            "Support_k": self.support_k,
            "Support_Radius_SD": self.support_radius_sd,
            "Historical_Target_qB": self.historical_target_qb,
            "Empirical_Hidden_Mean_qB": self.empirical_hidden_mean_qb,
            "Empirical_Hidden_Std_qB": self.empirical_hidden_std_qb,
            "Error_vs_Historical": self.error_vs_historical,
            "Error_vs_Empirical": self.error_vs_empirical,
            "Train_Cohort_N": self.train_founder_count,
            "Masked_Real_Target_N": self.real_masked_count,
            "Predicted_Mean_LoadSlope": self.predicted_mean_load,
            "Real_Masked_Mean_LoadSlope": self.real_masked_mean_load,
            "Mean_Bias_LoadSlope": self.mean_bias_load,
            "Std_Bias_LoadSlope": self.std_bias_load,
            "Wasserstein_Dist_Proposed": self.wasserstein_distance_load,
            "Wasserstein_Dist_AncestryOnly": self.w1_ancestry_only,
            "Wasserstein_Dist_TrainingMean": self.w1_training_mean,
            "Coverage_Probability_90Pct": self.coverage_probability_90pct
        }])


class MaskedHistoricalReconstructor:
    """Executes real-data blind reconstruction of composite populations without ancestry leakage."""

    def __init__(
        self,
        historical_target_qb: float = 0.375,
        adaptation_penalty_weight: float = 0.35,
        random_seed: int = 42
    ):
        self.historical_target_qb = historical_target_qb
        self.adaptation_penalty_weight = adaptation_penalty_weight
        self.rng = np.random.default_rng(random_seed)

    def partition_by_population_identity(
        self,
        genotypes: np.ndarray,
        phenotypes_load: np.ndarray,
        ancestry_qb: np.ndarray,
        population_ids: np.ndarray,
        withhold_pop_id: str = "Wes",
        train_pop_id: str = "Semin"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partitions data strictly by population identity without any q_B leakage."""
        train_mask = (population_ids == train_pop_id)
        masked_mask = (population_ids == withhold_pop_id)

        g_train = genotypes[train_mask, :]
        y_train = phenotypes_load[train_mask]
        qb_train = ancestry_qb[train_mask]

        g_masked = genotypes[masked_mask, :]
        y_masked = phenotypes_load[masked_mask]
        qb_masked = ancestry_qb[masked_mask]

        return g_train, y_train, qb_train, g_masked, y_masked, qb_masked

    def reconstruct_optimal_cross(
        self,
        g_train: np.ndarray,
        y_train: np.ndarray,
        qb_train: np.ndarray,
        g_masked: np.ndarray,
        y_masked: np.ndarray,
        qb_masked: np.ndarray,
        tau_load: float = 28.51,
        q_steps: int = 35,
        literature_prior: Optional[Tuple[float, float]] = None
    ) -> MaskedReconstructionResult:
        """Fits the cross-design model on the source population only, optimizes q_B, then compares.

        Model, fitted by least squares on training animals only:
            y = mu + beta * q_B + h * 4 q_B (1 - q_B) + e
        `literature_prior=(beta, h)` replaces the fitted slope and heterosis with external values
        (intercept refitted) and is recorded in `parameter_source`. The optimum is the grid maximum of
        U(q) = P(y < tau | q), refined to the analytic stationary point q* = 0.5 (1 + beta / (4h)) only
        when that point is an interior minimum of the expected load (h < 0). Genotypes are accepted for
        the population partition; the cross-design model itself uses ancestry only.
        """
        n_train = len(y_train)
        n_masked = len(y_masked)

        # 1. Cross-design model from the training population
        het_train = 4.0 * qb_train * (1.0 - qb_train)
        if literature_prior is None:
            X_anc = np.column_stack([np.ones_like(qb_train), qb_train, het_train])
            (mu_fit, beta_fit, h_fit), *_ = np.linalg.lstsq(X_anc, y_train, rcond=None)
            source = "fitted on training population"
        else:
            beta_fit, h_fit = literature_prior
            mu_fit = float(np.mean(y_train - beta_fit * qb_train - h_fit * het_train))
            source = f"literature prior beta={beta_fit}, h={h_fit} (intercept refitted)"
        mu_fit, beta_fit, h_fit = float(mu_fit), float(beta_fit), float(h_fit)
        residuals = y_train - (mu_fit + beta_fit * qb_train + h_fit * het_train)
        sigma_e = float(np.std(residuals, ddof=min(3, n_train - 1)))

        # 2. Counterfactual crossing grid over q_B in [0, 1]
        q_grid = np.linspace(0.0, 1.0, q_steps)
        heterosis_gain = 4.0 * q_grid * (1.0 - q_grid) * h_fit
        heterosis_profile = -heterosis_gain  # positive = heterosis lowers heat load
        predicted_load_profile = mu_fit + beta_fit * q_grid + heterosis_gain
        utility_profile = stats.norm.cdf((tau_load - predicted_load_profile) / sigma_e)

        # Empirical ancestry support: distance to the k-th nearest training animal, supported within the
        # leave-one-out 95 % radius. Min–max bounds let one isolated animal widen the whole region.
        support = KNNSupport(level=0.95).fit(qb_train)
        qb_mean_train = float(np.mean(qb_train))
        supported_profile = support.supported(q_grid)
        d_ood_profile = support.excess(q_grid)            # training-SD distance beyond the support radius
        conf_profile = 1.0 / (1.0 + d_ood_profile)
        # Local empirical sample count within delta = 0.04
        density_profile = np.array([float(np.sum(np.abs(qb_train - q) <= 0.04)) for q in q_grid])

        # 3. Optimal cross: grid maximum, refined when the stationary point is an interior minimum
        q_opt = float(q_grid[int(np.argmax(utility_profile))])
        if h_fit < 0:
            q_stationary = 0.5 * (1.0 + beta_fit / (4.0 * h_fit))
            if 0.0 <= q_stationary <= 1.0:
                q_opt = float(q_stationary)
        pred_mean_opt = float(mu_fit + beta_fit * q_opt + 4.0 * q_opt * (1.0 - q_opt) * h_fit)
        optimum_at_boundary = bool(np.isclose(q_opt, 0.0) or np.isclose(q_opt, 1.0))

        # Support-constrained optimum: argmax_q U(q) over grid points inside the empirical support
        safe_mask = supported_profile
        if np.any(safe_mask):
            safe_opt_idx = int(np.argmax(utility_profile[safe_mask]))
            q_opt_safe = float(q_grid[safe_mask][safe_opt_idx])
        else:
            q_opt_safe = float(qb_mean_train)

        # Support at the unconstrained optimum
        d_ood_at_opt = float(support.excess([q_opt])[0])
        conf_at_opt = float(1.0 / (1.0 + d_ood_at_opt))
        is_opt_within_support = bool(support.supported([q_opt])[0])

        # 4. Post-Hoc Reveal: Compare with Hidden Masked Cohort C (Commercial Brangus)
        real_mean_qb = float(np.mean(qb_masked))
        real_std_qb = float(np.std(qb_masked))
        real_mean_load = float(np.mean(y_masked))
        real_std_load = float(np.std(y_masked))
        pred_std_opt = sigma_e  # residual SD of the fitted cross-design model

        # Phenotype Bias
        mean_bias = float(real_mean_load - pred_mean_opt)
        std_bias = float(mean_bias / (real_std_load + 1e-6))

        # Wasserstein distance between the predicted distribution at q* and the real masked cattle
        sim_optimal_samples = self.rng.normal(pred_mean_opt, pred_std_opt, size=2000)
        w1_load = float(stats.wasserstein_distance(sim_optimal_samples, y_masked))

        # Baseline comparators for W_1: training-population distribution, and a linear ancestry-only
        # model (no heterosis) evaluated at the same q*
        sim_train_mean = self.rng.normal(float(np.mean(y_train)), float(np.std(y_train)), size=2000)
        w1_train_mean = float(stats.wasserstein_distance(sim_train_mean, y_masked))

        slope_lin, intercept_lin, *_ = stats.linregress(qb_train, y_train)
        sd_lin = float(np.std(y_train - (intercept_lin + slope_lin * qb_train), ddof=2))
        sim_anc_only = self.rng.normal(float(intercept_lin + slope_lin * q_opt), sd_lin, size=2000)
        w1_anc_only = float(stats.wasserstein_distance(sim_anc_only, y_masked))

        # Coverage probability
        ci_low = float(np.percentile(sim_optimal_samples, 5.0))
        ci_high = float(np.percentile(sim_optimal_samples, 95.0))
        coverage_90 = float(np.mean((y_masked >= ci_low) & (y_masked <= ci_high)))

        err_hist = abs(q_opt - self.historical_target_qb)
        err_emp = abs(q_opt - real_mean_qb)

        return MaskedReconstructionResult(
            predicted_optimal_qb=q_opt,
            predicted_safe_qb=q_opt_safe,
            d_ood_at_opt=d_ood_at_opt,
            confidence_at_opt=conf_at_opt,
            is_opt_within_support=is_opt_within_support,
            historical_target_qb=self.historical_target_qb,
            empirical_hidden_mean_qb=real_mean_qb,
            empirical_hidden_std_qb=real_std_qb,
            error_vs_historical=err_hist,
            error_vs_empirical=err_emp,
            real_masked_count=n_masked,
            train_founder_count=n_train,
            predicted_mean_load=pred_mean_opt,
            real_masked_mean_load=real_mean_load,
            predicted_std_load=pred_std_opt,
            real_masked_std_load=real_std_load,
            mean_bias_load=mean_bias,
            std_bias_load=std_bias,
            wasserstein_distance_load=w1_load,
            w1_ancestry_only=w1_anc_only,
            w1_training_mean=w1_train_mean,
            coverage_probability_90pct=coverage_90,
            q_grid=q_grid,
            utility_profile=utility_profile,
            load_profile=predicted_load_profile,
            heterosis_profile=heterosis_profile,
            d_ood_profile=d_ood_profile,
            confidence_profile=conf_profile,
            density_profile=density_profile,
            fitted_intercept=mu_fit,
            fitted_beta=beta_fit,
            fitted_heterosis=h_fit,
            residual_sd=sigma_e,
            parameter_source=source,
            optimum_at_boundary=optimum_at_boundary,
            support_k=support.k_,
            support_radius_sd=support.radius_,
        )

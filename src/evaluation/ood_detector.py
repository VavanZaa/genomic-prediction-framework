"""Applicability Domain & Calibrated Selective Prediction Engine (Layer 6A-v2).

Implements rigorous genomic out-of-distribution detection and selective prediction to prevent
false certainty on unknown populations, alien breeds, ancestry extrapolation, or uncalibrated herd shifts.

Key Capabilities:
1. Multi-factor Genomic Manifold Characterization:
   - PCA Mahalanobis distance d_PCA(x, D_train)
   - Kinship leverage G_max = max_j G_xj
   - Relative RKHS leverage deficiency in [0, 1]
   - Allele frequency divergence D_AF(x)
   - Empirical ancestry support (leave-one-out k-NN radius) and the distance d_ancestry beyond it
   - Genomic support d_support: k-NN distance in whitened PC space over the training 95 % radius
   - Macro-environmental / herd domain match indicator
2. Calibrated Selective Prediction:
   - Evaluates Risk(kappa) = E[L(y, y_hat) | confidence >= c_kappa] across coverage fractions kappa
   - Evaluates False-Safe Rate: P(HIGH confidence | prediction failure)
   - Enforces explicit refusal for severe extrapolation or unsupported populations.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import linalg, stats

from .support import KNNSupport


class ConfidenceTier(str, Enum):
    HIGH = "HIGH (Numeric Prediction with Empirically Calibrated 95% Interval)"
    MODERATE = "MODERATE (Exploratory Genomic Similarity Only; Absolute Phenotype & Selection Ranking Suppressed)"
    LOW = "LOW (Full Refusal; All Quantitative Predictions Suppressed)"


@dataclass
class OODAssessment:
    """Individual or cohort OOD evaluation results with operational action semantics."""
    animal_id: str
    d_pca: float
    g_max: float
    d_kernel: float
    d_af: float
    d_ancestry: float
    d_ood_composite: float
    selective_confidence: float
    confidence_tier: ConfidenceTier
    uncertainty_inflation_factor: float
    is_refused: bool
    allow_numeric_prediction: bool = True
    allow_ranking_only: bool = False
    refusal_reason: Optional[str] = None
    env_mismatch: bool = False
    c_absolute: float = 1.0
    c_ranking: float = 1.0
    d_support: float = 0.0            # k-NN distance / training 95 % radius in PC space (<= 1 inside)
    ancestry_supported: bool = True
    support_score: float = 0.0        # max(genomic, ancestry) k-NN ratio used by the calibrated mode

    def to_dict(self) -> Dict[str, Union[str, float, bool]]:
        return {
            "animal_id": self.animal_id,
            "d_pca": self.d_pca,
            "g_max": self.g_max,
            "d_kernel": self.d_kernel,
            "d_af": self.d_af,
            "d_ancestry": self.d_ancestry,
            "ancestry_supported": self.ancestry_supported,
            "d_support": self.d_support,
            "d_ood_composite": self.d_ood_composite,
            "selective_confidence": self.selective_confidence,
            "c_absolute": self.c_absolute,
            "c_ranking": self.c_ranking,
            "confidence_tier": self.confidence_tier.value,
            "uncertainty_inflation_factor": self.uncertainty_inflation_factor,
            "is_refused": self.is_refused,
            "allow_numeric_prediction": self.allow_numeric_prediction,
            "allow_ranking_only": self.allow_ranking_only,
            "refusal_reason": self.refusal_reason or "N/A",
            "env_mismatch": self.env_mismatch
        }


class GenomicOODDetector:
    """Learns the genomic support manifold of a reference training population.

    The tier thresholds (PCA distance, G_max, kernel deficiency, allele-frequency divergence,
    ancestry extrapolation) and the composite-score weights are heuristic choices, not values
    calibrated on data; tune them on held-out populations before relying on the tiers.

    Support is empirical, not min–max: d_ancestry is the training-SD distance beyond the leave-one-out
    k-NN radius of the training ancestries (0 inside it), so the default moderate threshold of 0 marks
    every animal outside the empirical ancestry support. d_support does the same in whitened PC space.
    """

    def __init__(
        self,
        n_components: int = 15,
        lambda_ridge: float = 1e-4,
        uncertainty_lambda: float = 0.5,
        pca_threshold_mod: float = 2.2,
        pca_threshold_low: float = 4.0,
        g_max_threshold_high: float = 0.08,
        g_max_threshold_low: float = 0.02,
        kernel_threshold_low: float = 0.90,
        ancestry_extrap_threshold_mod: float = 0.0,
        ancestry_extrap_threshold_low: float = 1.20,
        support_level: float = 0.95,
        support_threshold_low: float = 2.0,
        n_support_components: int = 10,
        support_k: Optional[int] = None,
        calibrated_thresholds: Optional[Dict[str, Optional[float]]] = None,
    ):
        """`calibrated_thresholds` ({"ranking": c, "absolute": c}, from load_frozen_thresholds) switches
        the tiers to the risk-calibrated rule: ranking is allowed when the support score is within the
        ranking threshold, numeric (absolute) prediction when it is also within the absolute threshold
        and the herd was seen in training. A None threshold refuses that kind of prediction. The
        heuristic components are still computed and reported, but do not decide the tier."""
        self.n_components = n_components
        self.lambda_ridge = lambda_ridge
        self.uncertainty_lambda = uncertainty_lambda
        self.pca_threshold_mod = pca_threshold_mod
        self.pca_threshold_low = pca_threshold_low
        self.g_max_threshold_high = g_max_threshold_high
        self.g_max_threshold_low = g_max_threshold_low
        self.kernel_threshold_low = kernel_threshold_low
        self.ancestry_extrap_threshold_mod = ancestry_extrap_threshold_mod
        self.ancestry_extrap_threshold_low = ancestry_extrap_threshold_low
        self.support_level = support_level
        self.support_threshold_low = support_threshold_low
        self.n_support_components = n_support_components
        # None keeps KNNSupport's own default (1 % of the training sample, at least 5); an explicit
        # value exists so the support region's sensitivity to k can be mapped without editing it.
        self.support_k = support_k
        self.genomic_support: Optional[KNNSupport] = None
        self.ancestry_support: Optional[KNNSupport] = None
        self.calibrated_thresholds = calibrated_thresholds

        self.is_fitted = False
        self.ref_p: Optional[np.ndarray] = None
        self.ref_std: Optional[np.ndarray] = None
        self.z_train: Optional[np.ndarray] = None
        self.v_pca: Optional[np.ndarray] = None
        self.eigenvals_pca: Optional[np.ndarray] = None
        self.k_inv: Optional[np.ndarray] = None
        self.train_grm: Optional[np.ndarray] = None
        self.mean_train_d_pca: float = 1.0
        self.mean_train_d_kernel: float = 1.0
        self.mean_train_d_af: float = 1.0
        self.std_train_d_af: float = 1.0

        # Domain metadata
        self.qb_min: Optional[float] = None
        self.qb_max: Optional[float] = None
        self.qb_mean: Optional[float] = None
        self.qb_std: Optional[float] = None
        self.env_set: Optional[set] = None

    def fit(
        self,
        g_train: np.ndarray,
        ancestry_qb_train: Optional[np.ndarray] = None,
        env_ids_train: Optional[Union[List[str], np.ndarray]] = None
    ) -> GenomicOODDetector:
        """Fits the genomic manifold representation on training genotypes G (N x M) and metadata."""
        n, m = g_train.shape
        self.ref_p = np.mean(g_train, axis=0) / 2.0
        p_safe = np.clip(self.ref_p, 0.005, 0.995)
        self.ref_std = np.sqrt(2.0 * p_safe * (1.0 - p_safe))
        
        self.z_train = (g_train - 2.0 * self.ref_p) / self.ref_std

        # GRM & Ridge Inverse
        self.train_grm = (self.z_train @ self.z_train.T) / m
        k_ridge = self.train_grm + self.lambda_ridge * np.eye(n)
        self.k_inv = linalg.pinvh(k_ridge)

        # PCA via SVD
        k_pca = min(self.n_components, n - 1, m - 1)
        u, s, vt = linalg.svd(self.z_train, full_matrices=False)
        self.v_pca = vt[:k_pca, :].T  # M x K
        self.eigenvals_pca = (s[:k_pca] ** 2) / (n - 1)

        # Internal calibration on training data
        z_v = self.z_train @ self.v_pca  # N x K
        d_pca_train = np.sqrt(np.mean((z_v ** 2) / (self.eigenvals_pca + 1e-6), axis=1))
        self.mean_train_d_pca = float(np.mean(d_pca_train))
        n_sc = min(self.n_support_components, k_pca)
        self.genomic_support = KNNSupport(k=self.support_k, level=self.support_level).fit(
            z_v[:, :n_sc] / np.sqrt(self.eigenvals_pca[:n_sc] + 1e-12))

        diag_k = np.diag(self.train_grm)
        quad = np.sum((self.train_grm @ self.k_inv) * self.train_grm, axis=1)
        d_k_train = np.maximum(diag_k - quad, 0.0)
        self.mean_train_d_kernel = float(np.mean(d_k_train)) + 1e-6

        d_af_train = np.mean(self.z_train ** 2, axis=1)
        self.mean_train_d_af = float(np.mean(d_af_train))
        self.std_train_d_af = float(np.std(d_af_train)) + 1e-6

        # Ancestry support bounds
        if ancestry_qb_train is not None:
            self.qb_min = float(np.min(ancestry_qb_train))
            self.qb_max = float(np.max(ancestry_qb_train))
            self.qb_mean = float(np.mean(ancestry_qb_train))
            self.qb_std = float(np.std(ancestry_qb_train)) + 1e-6
            self.ancestry_support = KNNSupport(level=self.support_level).fit(np.asarray(ancestry_qb_train, dtype=float))

        # Environmental support set
        if env_ids_train is not None:
            self.env_set = set(str(x) for x in env_ids_train)

        self.is_fitted = True
        return self

    def support_ratios(self, g_cohort: np.ndarray, ancestry_qb=None) -> Tuple[np.ndarray, np.ndarray]:
        """Batch k-NN support ratios (<= 1 inside the training 95 % radius): genomic PC space and ancestry."""
        if not self.is_fitted:
            raise RuntimeError("GenomicOODDetector must be fitted first.")
        z = (np.asarray(g_cohort, dtype=float) - 2.0 * self.ref_p) / self.ref_std
        n_sc = self.genomic_support.tree_.m
        scores = (z @ self.v_pca)[:, :n_sc] / np.sqrt(self.eigenvals_pca[:n_sc] + 1e-12)
        genomic = self.genomic_support.ratio(scores)
        if ancestry_qb is not None and self.ancestry_support is not None:
            ancestry = self.ancestry_support.ratio(np.asarray(ancestry_qb, dtype=float))
        else:
            ancestry = np.zeros(len(genomic))
        return genomic, ancestry

    def assess_animal(
        self,
        g_animal: np.ndarray,
        ancestry_qb: Optional[float] = None,
        env_id: Optional[str] = None,
        animal_id: str = "query"
    ) -> OODAssessment:
        """Evaluates OOD distance and applicability tier for a single query animal."""
        if not self.is_fitted:
            raise RuntimeError("GenomicOODDetector must be fitted before assessing animals.")

        m = len(g_animal)
        z_x = (g_animal - 2.0 * self.ref_p) / self.ref_std

        # 1. PCA Mahalanobis Distance
        v_x = z_x @ self.v_pca
        d_pca_sq = np.sum((v_x ** 2) / (self.eigenvals_pca + 1e-6)) / len(self.eigenvals_pca)
        d_pca = float(np.sqrt(d_pca_sq))

        # 2. Maximum Genomic Relationship
        k_x = (self.z_train @ z_x) / m
        g_max = float(np.max(k_x))

        # 3. Kernel Distance / Leverage Deficiency in RKHS
        k_xx = float((z_x @ z_x) / m)
        leverage = float(k_x @ self.k_inv @ k_x)
        d_kernel = float(max(k_xx - leverage, 0.0))
        rel_deficiency = float(np.clip(d_kernel / (k_xx + 1e-6), 0.0, 1.0))

        # 4. Allele Frequency Divergence
        d_af = float(np.mean((g_animal - 2.0 * self.ref_p) ** 2 / (self.ref_std ** 2 + 1e-6)))
        z_af = float((d_af - self.mean_train_d_af) / self.std_train_d_af)

        # 5. Ancestry support: training-SD distance beyond the empirical k-NN support radius. Unlike
        # min–max bounds, a gap between clusters or a lone extreme animal does not count as support.
        d_ancestry, ancestry_supported = 0.0, True
        if ancestry_qb is not None and self.ancestry_support is not None:
            d_ancestry = float(self.ancestry_support.excess([ancestry_qb])[0])
            ancestry_supported = bool(self.ancestry_support.supported([ancestry_qb])[0])

        # 5b. Genomic support: k-NN distance in whitened PC space over the training 95 % radius
        n_sc = self.genomic_support.tree_.m
        d_support = float(self.genomic_support.ratio(
            (v_x[:n_sc] / np.sqrt(self.eigenvals_pca[:n_sc] + 1e-12))[None, :])[0])

        # 6. Environmental Domain Check
        env_mismatch = False
        if env_id is not None and self.env_set is not None:
            env_mismatch = (str(env_id) not in self.env_set)

        # 7. Composite OOD Score
        norm_pca = d_pca / (self.mean_train_d_pca + 1e-6)
        norm_af = max(0.0, (d_af - self.mean_train_d_af) / (self.mean_train_d_af + 1e-6))
        g_penalty = max(0.0, (self.g_max_threshold_high - g_max) / self.g_max_threshold_high)
        env_penalty = 0.8 if env_mismatch else 0.0

        d_ood = float(
            0.25 * norm_pca +
            0.20 * rel_deficiency +
            0.20 * norm_af +
            0.15 * g_penalty +
            0.20 * d_ancestry +
            0.20 * max(0.0, d_support - 1.0) +
            env_penalty
        )

        # Selective Confidence Score in [0, 1]: 1 = optimal interpolation, 0 = pure extrapolation
        selective_conf = float(1.0 / (1.0 + d_ood))

        # Uncertainty Inflation
        inflation = float(1.0 + self.uncertainty_lambda * d_ood)

        # Dual Confidence Axes:
        # C_absolute evaluates trustworthiness of calibrated numeric phenotypes (suppressed by env mismatch)
        c_absolute = float(selective_conf * (0.10 if env_mismatch else 1.0))
        
        # C_ranking evaluates trustworthiness of relative candidate selection ranking
        c_ranking = float(1.0 / (1.0 + 0.6 * d_ood + (0.80 if env_mismatch else 0.0)))

        # 8a. Risk-calibrated tiers (docs/OOD_CALIBRATION.md) replace the heuristic ones when supplied
        if self.calibrated_thresholds is not None:
            anc_ratio = (float(self.ancestry_support.ratio([ancestry_qb])[0])
                         if ancestry_qb is not None and self.ancestry_support is not None else 0.0)
            support_score = max(d_support, anc_ratio)
            c_rank = self.calibrated_thresholds.get("ranking")
            c_abs = self.calibrated_thresholds.get("absolute")
            compatible = z_af <= 4.0          # gross genotype incompatibility is not calibratable
            ranking_ok = compatible and c_rank is not None and support_score <= c_rank
            numeric_ok = ranking_ok and c_abs is not None and support_score <= c_abs and not env_mismatch
            if numeric_ok:
                tier, reason = ConfidenceTier.HIGH, None
            elif ranking_ok:
                tier = ConfidenceTier.MODERATE
                reason = ("Ranking only: herd/environment not seen in training" if env_mismatch else
                          f"Ranking only: support score {support_score:.2f} beyond the absolute-risk threshold {c_abs}")
            else:
                tier = ConfidenceTier.LOW
                reason = (f"Refused: support score {support_score:.2f} beyond the ranking-risk threshold {c_rank}"
                          if compatible else f"Refused: genotype incompatible with the reference (z_AF = {z_af:.1f})")
            return OODAssessment(
                animal_id=animal_id, d_pca=d_pca, g_max=g_max, d_kernel=d_kernel, d_af=d_af,
                d_ancestry=d_ancestry, d_ood_composite=d_ood, selective_confidence=selective_conf,
                confidence_tier=tier, uncertainty_inflation_factor=inflation, is_refused=not ranking_ok,
                allow_numeric_prediction=numeric_ok, allow_ranking_only=ranking_ok, refusal_reason=reason,
                env_mismatch=env_mismatch, c_absolute=c_absolute, c_ranking=c_ranking, d_support=d_support,
                ancestry_supported=ancestry_supported, support_score=support_score,
            )

        # 8. Classification and Refusal Trigger
        is_severe_ood = (
            d_pca > self.pca_threshold_low or
            g_max < self.g_max_threshold_low or
            rel_deficiency > self.kernel_threshold_low or
            d_af > 1.35 or
            z_af > 4.0 or
            d_ancestry > self.ancestry_extrap_threshold_low or
            d_support > self.support_threshold_low
        )

        is_refused = False
        refusal_reason = None

        if is_severe_ood:
            tier = ConfidenceTier.LOW
            is_refused = True
            allow_numeric = False
            allow_ranking = False
            reasons = []
            if d_pca > self.pca_threshold_low:
                reasons.append(f"Subspace divergence d_PCA={d_pca:.2f} > {self.pca_threshold_low}")
            if g_max < self.g_max_threshold_low:
                reasons.append(f"Kinship leverage G_max={g_max:.3f} < {self.g_max_threshold_low}")
            if rel_deficiency > self.kernel_threshold_low:
                reasons.append(f"Kernel leverage deficiency={rel_deficiency:.3f} > {self.kernel_threshold_low}")
            if d_af > 1.35 or z_af > 4.0:
                reasons.append(f"Allele divergence D_AF={d_af:.2f} (z={z_af:.1f})")
            if d_ancestry > self.ancestry_extrap_threshold_low:
                reasons.append(f"Ancestry extrapolation d_ancestry={d_ancestry:.2f} > {self.ancestry_extrap_threshold_low}")
            if d_support > self.support_threshold_low:
                reasons.append(f"Outside genomic training support d_support={d_support:.2f} > {self.support_threshold_low}")
            refusal_reason = "Genomic support insufficient: " + "; ".join(reasons) + ". Quantitative prediction suppressed."
        elif (
            d_pca > self.pca_threshold_mod or
            g_max < self.g_max_threshold_high or
            rel_deficiency > 0.78 or
            d_af > 1.15 or
            d_ancestry > self.ancestry_extrap_threshold_mod or
            d_support > 1.0 or
            env_mismatch
        ):
            tier = ConfidenceTier.MODERATE
            is_refused = False
            allow_numeric = False  # Absolute numeric prediction suppressed!
            allow_ranking = False  # Selection ranking also suppressed due to lack of rank concordance on uncalibrated shifts!
            mod_reasons = []
            if env_mismatch:
                mod_reasons.append("Macro-environmental/herd shift uncalibrated")
            if d_ancestry > self.ancestry_extrap_threshold_mod:
                mod_reasons.append("Ancestry outside empirical training support")
            if d_support > 1.0:
                mod_reasons.append("Beyond the 95 % genomic training support")
            refusal_reason = "Operational Notice: Absolute numeric phenotype suppressed & breeding ranking suppressed (" + "; ".join(mod_reasons or ["Mild manifold extrapolation"]) + "). Exploratory genomic similarity only."
        else:
            tier = ConfidenceTier.HIGH
            is_refused = False
            allow_numeric = True
            allow_ranking = True

        return OODAssessment(
            animal_id=animal_id,
            d_pca=d_pca,
            g_max=g_max,
            d_kernel=d_kernel,
            d_af=d_af,
            d_ancestry=d_ancestry,
            d_ood_composite=d_ood,
            selective_confidence=selective_conf,
            confidence_tier=tier,
            uncertainty_inflation_factor=inflation,
            is_refused=is_refused,
            allow_numeric_prediction=allow_numeric,
            allow_ranking_only=allow_ranking,
            refusal_reason=refusal_reason,
            env_mismatch=env_mismatch,
            c_absolute=c_absolute,
            c_ranking=c_ranking,
            d_support=d_support,
            ancestry_supported=ancestry_supported,
        )

    def assess_cohort(
        self,
        g_cohort: np.ndarray,
        ancestry_qb_cohort: Optional[np.ndarray] = None,
        env_ids_cohort: Optional[Union[List[str], np.ndarray]] = None,
        animal_ids: Optional[List[str]] = None
    ) -> List[OODAssessment]:
        """Assesses an entire cohort of query animals."""
        n = g_cohort.shape[0]
        ids = animal_ids or [f"query_{i}" for i in range(n)]
        qb_vals = ancestry_qb_cohort if ancestry_qb_cohort is not None else [None] * n
        env_vals = env_ids_cohort if env_ids_cohort is not None else [None] * n

        return [
            self.assess_animal(
                g_cohort[i, :],
                ancestry_qb=qb_vals[i],
                env_id=env_vals[i],
                animal_id=ids[i]
            )
            for i in range(n)
        ]

    @staticmethod
    def compute_selective_risk(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        coverage_grid: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Computes empirical selective risk (MSE) and correlation across coverage fractions."""
        if coverage_grid is None:
            coverage_grid = np.linspace(0.1, 1.0, 19)

        n = len(y_true)
        sorted_indices = np.argsort(confidences)[::-1]  # Highest confidence first

        rows = []
        for cov in coverage_grid:
            k = max(1, int(np.round(cov * n)))
            selected_idx = sorted_indices[:k]
            y_t_sel = y_true[selected_idx]
            y_p_sel = y_pred[selected_idx]

            mse = float(np.mean((y_t_sel - y_p_sel) ** 2))
            rmse = float(np.sqrt(mse))
            r = float(stats.pearsonr(y_t_sel, y_p_sel)[0]) if len(selected_idx) > 2 and np.std(y_p_sel) > 1e-6 else 0.0

            rows.append({
                "Coverage": cov,
                "Selected_N": k,
                "Risk_MSE": mse,
                "Risk_RMSE": rmse,
                "Pearson_r": r,
                "Min_Confidence_Threshold": float(confidences[selected_idx[-1]])
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compute_false_safe_rate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        assessments: List[OODAssessment],
        residual_threshold_sd: float = 1.5
    ) -> Dict[str, float]:
        """Computes False-Safe Rate: P(HIGH confidence | Prediction Failure)."""
        residuals = np.abs(y_true - y_pred)
        sigma_res = float(np.std(residuals)) + 1e-6
        is_failure = (residuals > residual_threshold_sd * sigma_res)

        n_failures = int(np.sum(is_failure))
        if n_failures == 0:
            return {"false_safe_rate": 0.0, "failure_count": 0, "false_safe_count": 0}

        high_mask = np.array([a.confidence_tier == ConfidenceTier.HIGH for a in assessments])
        false_safe_count = int(np.sum(is_failure & high_mask))
        fsr = float(false_safe_count / n_failures)

        return {
            "false_safe_rate": fsr,
            "failure_count": n_failures,
            "false_safe_count": false_safe_count
        }

    @staticmethod
    def compute_false_refusal_rate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        assessments: List[OODAssessment],
        residual_threshold_sd: float = 1.5
    ) -> Dict[str, float]:
        """Computes False-Refusal Rate: P(Refused / LOW | Prediction was actually acceptable)."""
        residuals = np.abs(y_true - y_pred)
        sigma_res = float(np.std(residuals)) + 1e-6
        is_acceptable = (residuals <= residual_threshold_sd * sigma_res)

        n_acceptable = int(np.sum(is_acceptable))
        if n_acceptable == 0:
            return {"false_refusal_rate": 0.0, "acceptable_count": 0, "false_refused_count": 0}

        refused_mask = np.array([a.confidence_tier == ConfidenceTier.LOW for a in assessments])
        false_refused_count = int(np.sum(is_acceptable & refused_mask))
        frr = float(false_refused_count / n_acceptable)

        return {
            "false_refusal_rate": frr,
            "acceptable_count": n_acceptable,
            "false_refused_count": false_refused_count
        }

    @staticmethod
    def compute_aurc(risk_df: pd.DataFrame, risk_col: str = "Risk_MSE") -> float:
        """Computes Area Under the Risk-Coverage curve (AURC) via trapezoidal integration."""
        covs = risk_df["Coverage"].values
        risks = risk_df[risk_col].values
        idx = np.argsort(covs)
        covs_s = covs[idx]
        risks_s = risks[idx]
        # np.trapezoid in newer numpy or np.trapz in older
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(trapz_fn(risks_s, covs_s))

    @staticmethod
    def compute_coverage_at_risk(risk_df: pd.DataFrame, max_risk: float, risk_col: str = "Risk_MSE") -> float:
        """Computes maximum coverage achieved while maintaining risk <= max_risk."""
        valid = risk_df[risk_df[risk_col] <= max_risk]
        if len(valid) == 0:
            return 0.0
        return float(valid["Coverage"].max())

    @staticmethod
    def compute_ranking_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k_pct: float = 0.10
    ) -> Dict[str, Union[float, int, bool]]:
        """Evaluates relative selection ranking capacity on an evaluation cohort.
        
        Calculates:
        1. Spearman rank correlation rho
        2. Pairwise rank concordance: P((y_i - y_j)(y_hat_i - y_hat_j) > 0)
        3. Top-k candidate selection enrichment (lower phenotype = superior thermotolerance)
        """
        n = len(y_true)
        if n < 3:
            return {
                "pearson_r": 0.0,
                "spearman_rho": 0.0,
                "pairwise_concordance": 0.5000,
                "top_k_enrichment": 1.0,
                "top_k_overlap": 0,
                "top_k_count": 0,
                "is_ranking_valid": False
            }

        r_p = float(stats.pearsonr(y_true, y_pred)[0])
        r_s = float(stats.spearmanr(y_true, y_pred)[0])

        diff_y = y_true[:, None] - y_true[None, :]
        diff_pred = y_pred[:, None] - y_pred[None, :]
        pairs = (diff_y * diff_pred) > 0
        concordance = float(np.sum(pairs) / (n * (n - 1)))

        k = max(2, int(np.round(top_k_pct * n)))
        top_true = set(np.argsort(y_true)[:k])
        top_pred = set(np.argsort(y_pred)[:k])
        overlap = len(top_true.intersection(top_pred))
        enrichment = float(overlap / (k * top_k_pct))

        # Operational ranking validity requires positive rank correlation and concordance > 0.60
        is_valid = bool(r_s >= 0.20 and concordance >= 0.60)

        return {
            "pearson_r": r_p,
            "spearman_rho": r_s,
            "pairwise_concordance": concordance,
            "top_k_enrichment": enrichment,
            "top_k_overlap": overlap,
            "top_k_count": k,
            "is_ranking_valid": is_valid
        }

"""Out-of-sample prediction and joint target probability evaluation on withheld real cattle cohorts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from ..prediction.uncertainty import residual_covariance
# prediction.models imports real_data.genotype_preprocessing, so the models are imported inside
# run_benchmark to keep `import src.real_data` free of an import cycle.


@dataclass
class EmpiricalEvaluationResult:
    model_name: str
    growth_rmse: float
    growth_r: float
    growth_rho: float
    marbling_rmse: float
    marbling_r: float
    marbling_rho: float
    brier_score_joint: float
    auc_joint: float
    predicted_attainment_mean: float
    observed_attainment_mean: float


class EmpiricalBenchmark:
    """Evaluates predictive accuracy and joint target probability calibration on withheld real calves."""

    @staticmethod
    def run_benchmark(
        train_ancestry: np.ndarray,
        train_markers: np.ndarray,
        train_growth: np.ndarray,
        train_marbling: np.ndarray,
        test_ancestry: np.ndarray,
        test_markers: np.ndarray,
        test_growth: np.ndarray,
        test_marbling: np.ndarray,
        target_growth: float,
        target_marbling: float,
        train_groups: Optional[np.ndarray] = None
    ) -> Dict[str, EmpiricalEvaluationResult]:
        from ..prediction.models import AdditiveGBLUP, AncestryPredictor, RkhsPredictor

        # Observed joint binary outcome on withheld real calves
        y_joint_true = ((test_growth >= target_growth) & (test_marbling >= target_marbling)).astype(int)

        models = {
            "AncestryOnly": (AncestryPredictor(), AncestryPredictor()),
            "AdditiveGBLUP": (AdditiveGBLUP(), AdditiveGBLUP()),
            "NonlinearRKHS": (RkhsPredictor(), RkhsPredictor()),
        }

        results: Dict[str, EmpiricalEvaluationResult] = {}

        for m_name, (m_growth, m_marb) in models.items():
            if m_name == "AncestryOnly":
                m_growth.fit(train_ancestry, train_growth)
                m_marb.fit(train_ancestry, train_marbling)
                p_growth = m_growth.predict(test_ancestry)
                p_marb = m_marb.predict(test_ancestry)
            else:
                m_growth.fit(train_markers, train_growth)
                m_marb.fit(train_markers, train_marbling)
                p_growth = m_growth.predict(test_markers)
                p_marb = m_marb.predict(test_markers)
            X = train_ancestry if m_name == "AncestryOnly" else train_markers
            res_cov = residual_covariance(X,train_growth,train_marbling,type(m_growth),train_groups)

            # Compute trait accuracy metrics
            g_rmse = float(np.sqrt(mean_squared_error(test_growth, p_growth)))
            g_r, _ = stats.pearsonr(test_growth, p_growth)
            g_rho, _ = stats.spearmanr(test_growth, p_growth)

            m_rmse = float(np.sqrt(mean_squared_error(test_marbling, p_marb)))
            m_r, _ = stats.pearsonr(test_marbling, p_marb)
            m_rho, _ = stats.spearmanr(test_marbling, p_marb)

            # Bivariate probability of meeting joint target preserving covariance
            pred_probs = []
            mvn_scale = np.maximum(np.diag(res_cov), 1e-4)
            for pg, pm in zip(p_growth, p_marb):
                mu = np.array([pg, pm])
                # P(Growth >= t1, Marb >= t2)
                mvn = stats.multivariate_normal(mean=mu, cov=res_cov, allow_singular=True)
                p_ll = mvn.cdf([target_growth, target_marbling])
                p_l1 = stats.norm.cdf(target_growth, loc=pg, scale=np.sqrt(mvn_scale[0]))
                p_l2 = stats.norm.cdf(target_marbling, loc=pm, scale=np.sqrt(mvn_scale[1]))
                prob = float(np.clip(1.0 - p_l1 - p_l2 + p_ll, 0.0, 1.0))
                pred_probs.append(prob)

            pred_probs_arr = np.array(pred_probs)

            # Probability calibration metrics
            brier = float(brier_score_loss(y_joint_true, pred_probs_arr))
            try:
                auc = float(roc_auc_score(y_joint_true, pred_probs_arr)) if len(np.unique(y_joint_true)) > 1 else 0.50
            except Exception:
                auc = 0.50

            results[m_name] = EmpiricalEvaluationResult(
                model_name=m_name,
                growth_rmse=g_rmse,
                growth_r=float(g_r),
                growth_rho=float(g_rho),
                marbling_rmse=m_rmse,
                marbling_r=float(m_r),
                marbling_rho=float(m_rho),
                brier_score_joint=brier,
                auc_joint=auc,
                predicted_attainment_mean=float(np.mean(pred_probs_arr)),
                observed_attainment_mean=float(np.mean(y_joint_true))
            )

        return results

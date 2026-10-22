"""Zero-Leakage Within-Training Cross-Fitted Probability Calibrator for Layer 4A.

Ensures that the prospective test cohort (658 animals) remains genuinely untouched:
1. Splits the historical training cohort (1,754 animals) into K cross-fitting folds.
2. For each fold k:
   - Fits the GBLUP model on folds != k
   - Predicts out-of-fold phenotypes and probabilities on fold k
3. Fits unpenalized Platt logistic regression strictly on out-of-fold training predictions:
   logit(P(y < tau)) = alpha_cv + beta_cv * logit(p_raw)
4. Prospective evaluations and Layer 4 calibrations apply (alpha_cv, beta_cv) without using
   ANY information from prospective test phenotypes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from ..prediction.kernel_cv import predict_kernel, nested_kernel_oof


@dataclass
class CrossFittedCalibrationResult:
    trait_name: str
    tau: float
    alpha_cv: float
    beta_cv: float
    oof_brier: float
    oof_ece: float
    oof_auc: float
    diagnostic_scope: str = "Calibration-fit diagnostics on OOF base predictions; validate the mapping on a separate cohort"


class CrossFittedCalibrator:
    """Learns probability calibration parameters strictly within training data via K-fold cross-fitting."""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.alpha_: Optional[float] = None
        self.beta_: Optional[float] = None
        self.tau_: Optional[float] = None
        self.trait_name_: Optional[str] = None

    def fit(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
        tau: float,
        trait_name: str = "Trait",
        lam: float = 10.0,
    ) -> CrossFittedCalibrationResult:
        """Fits calibration mapping using out-of-fold training predictions."""
        self.tau_ = tau
        self.trait_name_ = trait_name

        n = len(y_train)
        fold_size = n // self.n_folds
        indices = np.arange(n)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)

        oof_preds = np.zeros(n)
        oof_sigma = np.zeros(n)

        # 1. Generate out-of-fold predictions
        for k in range(self.n_folds):
            val_idx = indices[k * fold_size : (k + 1) * fold_size] if k < self.n_folds - 1 else indices[k * fold_size :]
            tr_idx = np.setdiff1d(indices, val_idx)

            K_tr_sub = K_train[np.ix_(tr_idx, tr_idx)]
            K_val_sub = K_train[np.ix_(val_idx, tr_idx)]

            y_tr_sub = y_train[tr_idx]
            mu_sub = float(np.mean(y_tr_sub))
            y_tr_cent = y_tr_sub - mu_sub

            oof_preds[val_idx] = predict_kernel(K_tr_sub,K_val_sub,y_tr_sub,lam)
            inner_pred=nested_kernel_oof([K_tr_sub],y_tr_sub,n_folds=3,seed=self.random_state,fixed_lambda=lam)
            oof_sigma[val_idx] = np.std(y_tr_sub-inner_pred,ddof=1)

        # 2. Compute uncalibrated out-of-fold probabilities: P(y < tau)
        oof_raw_prob = norm.cdf(tau, loc=oof_preds, scale=np.clip(oof_sigma, 1e-4, None))
        oof_raw_prob = np.clip(oof_raw_prob, 1e-5, 1.0 - 1e-5)

        # 3. True binary targets in training data
        y_binary = (y_train < tau).astype(int)

        # 4. Unpenalized logistic regression: logit(p_true) = alpha + beta * logit(p_raw)
        logits = logit(oof_raw_prob).reshape(-1, 1)
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        lr.fit(logits, y_binary)

        self.alpha_ = float(lr.intercept_[0])
        self.beta_ = float(lr.coef_[0][0])

        # 5. Fit diagnostics: base predictions are OOF, the mapping is fitted here.
        calibrated_probs = expit(self.alpha_ + self.beta_ * logits.ravel())
        brier = float(np.mean((calibrated_probs - y_binary) ** 2))

        # ECE on out-of-fold training data (10 bins)
        bins = np.linspace(0.0, 1.0, 11)
        bin_indices = np.digitize(calibrated_probs, bins) - 1
        ece = 0.0
        for b in range(10):
            mask = (bin_indices == b)
            if np.sum(mask) > 0:
                conf = np.mean(calibrated_probs[mask])
                acc = np.mean(y_binary[mask])
                ece += (np.sum(mask) / n) * np.abs(acc - conf)

        # ROC-AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_binary, calibrated_probs))
        except Exception:
            auc = 0.5

        return CrossFittedCalibrationResult(
            trait_name=trait_name,
            tau=tau,
            alpha_cv=self.alpha_,
            beta_cv=self.beta_,
            oof_brier=brier,
            oof_ece=float(ece * 100.0),
            oof_auc=auc,
        )

    def calibrate(self, raw_probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Applies learned (alpha_cv, beta_cv) to new probabilities without data leakage."""
        if self.alpha_ is None or self.beta_ is None:
            raise RuntimeError("CrossFittedCalibrator must be fitted before calibrate() is called.")
        is_scalar = np.isscalar(raw_probs)
        p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        log_odds = logit(p)
        calibrated = expit(self.alpha_ + self.beta_ * log_odds)
        if is_scalar:
            return float(calibrated)
        return calibrated

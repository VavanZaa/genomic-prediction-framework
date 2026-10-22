"""Breed-of-Origin Alleles (BOA) Multi-Kernel Genomic Prediction Models.

Implements the 5 models evaluated in the UF Multibreed beef cattle architecture:
- M0: Global Ancestry Baseline (y ~ q_Angus)
- M1: Standard SNP GBLUP (y ~ G)
- M2: Pure Breed-of-Origin GBLUP (y ~ B)
- M3: Combined SNP + BOA GBLUP (y ~ w_G * G + w_B * B)
- M4: SNP + BOA + Interaction (y ~ w_G * G + w_B * B + w_Omega * Omega)

Supports both:
1. Fixed diagnostic weights (e.g. 50/50 for M3, 1/3-1/3-1/3 for M4).
2. Fully data-driven kernel weight optimization via K-fold cross-validation strictly on training data.
3. Paired bootstrap resampling reporting both one-sided and two-sided significance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from ..real_data.genotype_preprocessing import GenotypeImputer


@dataclass
class BootstrapComparisonResult:
    r_model: float
    r_baseline: float
    delta_r: float
    ci_lower: float
    ci_upper: float
    p_value_one_sided: float  # Empirical P(delta_r <= 0)
    p_value_two_sided: float  # 2 * min(P(delta_r <= 0), P(delta_r >= 0))
    p_value: float            # Default alias to two-sided p-value
    is_significant: bool      # True iff 0 is strictly outside 95% CI (two-sided alpha = 0.05)


class BOAValidationStats:
    """Rigorous statistical testing between competing genomic prediction architectures."""

    @staticmethod
    def paired_bootstrap_test(
        y_true: np.ndarray,
        y_pred_model: np.ndarray,
        y_pred_baseline: np.ndarray,
        n_bootstraps: int = 1000,
        random_seed: int = 42
    ) -> BootstrapComparisonResult:
        """Evaluates whether model significantly outperforms baseline using paired bootstrap resampling."""
        rng = np.random.default_rng(random_seed)
        n = len(y_true)

        # Point estimates
        r_mod, _ = stats.pearsonr(y_true, y_pred_model)
        r_base, _ = stats.pearsonr(y_true, y_pred_baseline)
        delta_point = r_mod - r_base

        delta_dist = np.zeros(n_bootstraps)
        for b in range(n_bootstraps):
            boot_idx = rng.integers(0, n, size=n)
            y_b = y_true[boot_idx]
            p_mod_b = y_pred_model[boot_idx]
            p_base_b = y_pred_baseline[boot_idx]

            # Protect against zero variance in bootstrap sample
            if np.std(p_mod_b) < 1e-8 or np.std(p_base_b) < 1e-8 or np.std(y_b) < 1e-8:
                delta_dist[b] = delta_point
            else:
                rb_mod, _ = stats.pearsonr(y_b, p_mod_b)
                rb_base, _ = stats.pearsonr(y_b, p_base_b)
                delta_dist[b] = rb_mod - rb_base

        ci_lower = float(np.percentile(delta_dist, 2.5))
        ci_upper = float(np.percentile(delta_dist, 97.5))
        
        # Empirical tail probabilities
        p_one_sided = float(np.mean(delta_dist <= 0.0))
        p_two_sided = float(min(1.0, 2.0 * min(p_one_sided, np.mean(delta_dist >= 0.0))))
        
        # Two-sided significance at alpha = 0.05 requires 0 to fall outside [ci_lower, ci_upper]
        is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)

        return BootstrapComparisonResult(
            r_model=float(r_mod),
            r_baseline=float(r_base),
            delta_r=float(delta_point),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value_one_sided=p_one_sided,
            p_value_two_sided=p_two_sided,
            p_value=p_two_sided,
            is_significant=is_sig
        )


class BOAComparativePredictor:
    """Multi-kernel GBLUP predictor supporting SNP (G), BOA (B), and Interaction (Omega) kernels.

    ridge_param is lambda = sigma_e^2 / sigma_g^2 on the kernel scale (lambda 0.4 implies h2 ~ 0.71).
    With ridge_param=None it is chosen by K-fold cross-validation on the training data only.
    """

    RIDGE_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

    def __init__(self, ridge_param: Optional[float] = 0.50):
        self.ridge_param = ridge_param
        self.ridge_param_: Optional[float] = ridge_param
        self.y_train_mean: float = 0.0
        self.weights_train: Optional[np.ndarray] = None
        self.X_train_norm: Optional[np.ndarray] = None
        self.O_train_norm: Optional[np.ndarray] = None
        self.OX_train_norm: Optional[np.ndarray] = None
        self.model_type: str = "M1_SNP"
        self.kernel_weights: Tuple[float, ...] = (1.0,)
        self.ancestry_slope: float = 0.0
        self.ancestry_intercept: float = 0.0

    @staticmethod
    def standardize_matrix(M: np.ndarray) -> np.ndarray:
        """Standardize genotype or BOA matrix by its own column mean and standard deviation."""
        mu, sd = BOAComparativePredictor._column_stats(M)
        return (M - mu) / sd

    @staticmethod
    def _column_stats(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(M, axis=0)
        sd = np.std(M, axis=0)
        sd[sd < 1e-8] = 1.0
        return mu, sd

    def _transform(self, X: np.ndarray, O: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize with the TRAINING column statistics, so predictions do not depend on the test batch."""
        X = self.x_imputer_.transform(X)
        O = self.o_imputer_.transform(O)
        X_n = (X.astype(float) - self.x_mu_) / self.x_sd_
        O_n = (O.astype(float) - self.o_mu_) / self.o_sd_
        OX_n = (X_n * O_n - self.ox_mu_) / self.ox_sd_
        return X_n, O_n, OX_n

    def fit(
        self,
        X_train: np.ndarray,  # (N_tr, M) SNP genotypes
        O_train: np.ndarray,  # (N_tr, M) BOA allele counts (0 = Angus/Angus, 2 = Brahman/Brahman)
        q_train: np.ndarray,  # (N_tr,) Angus breed fraction
        y_train: np.ndarray,  # (N_tr,) adjusted phenotype
        model_type: str = "M3_SNP_BOA",
        optimize_weights_cv: bool = False,
        fixed_weights: Optional[Tuple[float, ...]] = None
    ) -> BOAComparativePredictor:
        self.model_type = model_type
        self.y_train_mean = float(np.mean(y_train))
        y_cent = y_train - self.y_train_mean
        n_tr = len(y_train)

        if model_type == "M0_Ancestry":
            # Simple linear regression on Angus breed fraction
            slope, intercept, _, _, _ = stats.linregress(q_train, y_train)
            self.ancestry_slope = slope
            self.ancestry_intercept = intercept
            return self

        raw_X, raw_O = np.asarray(X_train), np.asarray(O_train)
        if optimize_weights_cv or self.ridge_param is None:
            from sklearn.model_selection import KFold
            if model_type == "M3_SNP_BOA" and optimize_weights_cv:
                grid=[(w,1-w) for w in np.linspace(0,1,6)]
            elif model_type == "M4_SNP_BOA_INTER" and optimize_weights_cv:
                grid=[(.5,.5,0.),(1/3,1/3,1/3),(.4,.4,.2),(.2,.6,.2),(.6,.2,.2)]
            else:
                default={"M1_SNP":(1.,),"M2_BOA":(1.,),"M3_SNP_BOA":(.5,.5),"M4_SNP_BOA_INTER":(1/3,1/3,1/3)}
                grid=[tuple(fixed_weights) if fixed_weights is not None else default[model_type]]
            penalties=[self.ridge_param] if self.ridge_param is not None else self.RIDGE_GRID
            errors=np.zeros((len(grid),len(penalties)))
            folds=KFold(n_splits=min(5,n_tr),shuffle=True,random_state=42).split(raw_X)
            for tr,va in folds:
                fold=BOAComparativePredictor(ridge_param=1.).fit(raw_X[tr],raw_O[tr],q_train[tr],y_train[tr],model_type=model_type,fixed_weights=grid[0])
                train=(fold.X_train_norm,fold.O_train_norm,fold.OX_train_norm)
                valid=fold._transform(raw_X[va],raw_O[va])
                selected={"M1_SNP":[0],"M2_BOA":[1],"M3_SNP_BOA":[0,1],"M4_SNP_BOA_INTER":[0,1,2]}[model_type]
                kt=[train[i]@train[i].T/train[i].shape[1] for i in selected]
                kv=[valid[i]@train[i].T/train[i].shape[1] for i in selected]
                mu=float(np.mean(y_train[tr]))
                for wi,weights in enumerate(grid):
                    K=sum(w*k for w,k in zip(weights,kt))
                    V=sum(w*k for w,k in zip(weights,kv))
                    eig,Q=np.linalg.eigh(K)
                    proj=Q.T@(y_train[tr]-mu)
                    cross=V@Q
                    for li,penalty in enumerate(penalties):
                        pred=mu+cross@(proj/(np.maximum(eig,0)+penalty))
                        errors[wi,li]+=np.sum((y_train[va]-pred)**2)
            wi,li=np.unravel_index(np.argmin(errors),errors.shape)
            self.cv_weights_,self.cv_penalty_=grid[wi],penalties[li]
            fixed_weights=self.cv_weights_
            optimize_weights_cv=False
        self.x_imputer_=GenotypeImputer().fit(raw_X)
        self.o_imputer_=GenotypeImputer().fit(raw_O)
        X_train=self.x_imputer_.transform(raw_X)
        O_train=self.o_imputer_.transform(raw_O)
        # Standardize matrices with training statistics (stored for prediction)
        self.x_mu_, self.x_sd_ = self._column_stats(X_train.astype(float))
        self.o_mu_, self.o_sd_ = self._column_stats(O_train.astype(float))
        x_n = (X_train.astype(float) - self.x_mu_) / self.x_sd_
        o_n = (O_train.astype(float) - self.o_mu_) / self.o_sd_
        self.ox_mu_, self.ox_sd_ = self._column_stats(x_n * o_n)
        self.X_train_norm, self.O_train_norm, self.OX_train_norm = self._transform(X_train, O_train)

        m = self.X_train_norm.shape[1]

        # Realized relationship kernels
        G = np.dot(self.X_train_norm, self.X_train_norm.T) / m
        B = np.dot(self.O_train_norm, self.O_train_norm.T) / m
        Omega = np.dot(self.OX_train_norm, self.OX_train_norm.T) / m
        # Kernel weights are chosen at the fixed penalty if given, else at a provisional lambda = 10
        weight_lam = self.ridge_param if self.ridge_param is not None else 10.0

        if model_type == "M1_SNP":
            K_train = G
            self.kernel_weights = (1.0,)
        elif model_type == "M2_BOA":
            K_train = B
            self.kernel_weights = (1.0,)
        elif model_type == "M3_SNP_BOA":
            if fixed_weights is not None:
                self.kernel_weights = fixed_weights
            else:
                self.kernel_weights = (0.5, 0.5)
            w_g, w_b = self.kernel_weights
            K_train = w_g * G + w_b * B
        elif model_type == "M4_SNP_BOA_INTER":
            if fixed_weights is not None:
                self.kernel_weights = fixed_weights
            else:
                self.kernel_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            w_g, w_b, w_om = self.kernel_weights
            K_train = w_g * G + w_b * B + w_om * Omega
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Regularized solve: alpha = (K + lambda * I)^(-1) y_cent
        self.ridge_param_ = self.ridge_param if self.ridge_param is not None else self.cv_penalty_
        K_reg = K_train + self.ridge_param_ * np.eye(n_tr)
        self.weights_train = np.linalg.solve(K_reg, y_cent)

        return self

    def predict(
        self,
        X_test: np.ndarray,
        O_test: np.ndarray,
        q_test: np.ndarray
    ) -> np.ndarray:
        if self.model_type == "M0_Ancestry":
            return self.ancestry_intercept + self.ancestry_slope * q_test

        X_test_norm, O_test_norm, OX_test_norm = self._transform(X_test, O_test)

        m = X_test_norm.shape[1]

        G_test = np.dot(X_test_norm, self.X_train_norm.T) / m
        B_test = np.dot(O_test_norm, self.O_train_norm.T) / m
        Omega_test = np.dot(OX_test_norm, self.OX_train_norm.T) / m

        if self.model_type == "M1_SNP":
            K_test = G_test
        elif self.model_type == "M2_BOA":
            K_test = B_test
        elif self.model_type == "M3_SNP_BOA":
            w_g, w_b = self.kernel_weights
            K_test = w_g * G_test + w_b * B_test
        elif self.model_type == "M4_SNP_BOA_INTER":
            w_g, w_b, w_om = self.kernel_weights
            K_test = w_g * G_test + w_b * B_test + w_om * Omega_test
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pred_cent = np.dot(K_test, self.weights_train)
        return self.y_train_mean + pred_cent

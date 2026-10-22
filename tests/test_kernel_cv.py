"""Training-only selection of GBLUP penalties and kernel weights."""
import numpy as np
from src.prediction.kernel_cv import DEFAULT_LAMBDAS, TWO_KERNEL_WEIGHTS, cv_kernel_ridge, tune_snp_boa_models


def _kernels(seed=0, n=160, m=300):
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n, m))
    X2 = rng.standard_normal((n, m))
    return rng, X1, X2, X1 @ X1.T / m, X2 @ X2.T / m


def test_noise_selects_more_shrinkage_than_signal():
    rng, X1, _, K1, _ = _kernels()
    signal = X1 @ rng.normal(0, 0.15, size=X1.shape[1]) + 0.5 * rng.standard_normal(len(X1))
    noise = rng.standard_normal(len(X1))
    lam_signal = cv_kernel_ridge([K1], signal)["lambda"]
    lam_noise = cv_kernel_ridge([K1], noise)["lambda"]
    assert lam_signal in DEFAULT_LAMBDAS and lam_noise in DEFAULT_LAMBDAS
    assert lam_noise > lam_signal


def test_weight_goes_to_the_informative_kernel():
    rng, _, X2, K1, K2 = _kernels(seed=1)
    y = X2 @ rng.normal(0, 0.15, size=X2.shape[1]) + 0.5 * rng.standard_normal(len(X2))
    best = cv_kernel_ridge([K1, K2], y, TWO_KERNEL_WEIGHTS)
    assert best["weights"][1] >= 0.6
    assert len(best["table"]) == len(TWO_KERNEL_WEIGHTS) * len(DEFAULT_LAMBDAS)


def test_tune_snp_boa_models_covers_every_model_and_trait():
    rng, X1, X2, K1, K2 = _kernels(seed=2, n=100)
    traits = {"load": rng.standard_normal(100), "hair": X1 @ rng.normal(0, 0.1, size=X1.shape[1])}
    best, table = tune_snp_boa_models(K1, K2, traits)
    assert set(best) == {(m, t) for m in ("SNP", "BOA", "Combined") for t in traits}
    assert set(table["model"]) == {"SNP", "BOA", "Combined"}

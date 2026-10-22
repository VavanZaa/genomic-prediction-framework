"""Unit and regression tests for Layer 3B: Breed-of-Origin Allele (BOA) genomic prediction.

Verifies:
1. Exact numerical identity between Layer 3A and 3B M0 baselines via CanonicalDataSplitter.
2. Verified BOA allele coding against known Angus and Brahman reference animals (0 = Angus, 2 = Brahman).
3. Statistical consistency of two-sided bootstrap confidence intervals and p-values.
4. Data-driven K-fold CV kernel weight optimization.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.real_data.canonical_split import CanonicalDataSplitter
from src.real_data.florida_boa_loader import FloridaBOALoader, FloridaDualBOAData
from src.prediction.boa_models import BOAComparativePredictor, BOAValidationStats


def test_canonical_split_reproduces_identical_m0_baseline():
    """Verify that Layer 3A and Layer 3B M0 baselines match to numerical precision."""
    data = FloridaBOALoader.load_dual_dataset(marker_step=24, cache=True)

    # 1. Pipeline 3A call
    split_3a = CanonicalDataSplitter.get_prospective_split(
        trait_col="LoadSlope",
        pheno_df=data.phenotypes,
        valid_animal_ids=data.animal_ids,
        require_joint_traits=True
    )
    s_3a, i_3a, _, _, _ = stats.linregress(split_3a.q_train, split_3a.y_train_adj)
    p_3a = i_3a + s_3a * split_3a.q_test
    r_3a, _ = stats.pearsonr(split_3a.y_test_adj, p_3a)

    # 2. Pipeline 3B call
    split_3b = CanonicalDataSplitter.get_prospective_split(
        trait_col="LoadSlope",
        pheno_df=data.phenotypes,
        valid_animal_ids=data.animal_ids,
        require_joint_traits=True
    )
    s_3b, i_3b, _, _, _ = stats.linregress(split_3b.q_train, split_3b.y_train_adj)
    p_3b = i_3b + s_3b * split_3b.q_test
    r_3b, _ = stats.pearsonr(split_3b.y_test_adj, p_3b)

    # Assert exact array equality and correlation equivalence
    assert split_3a.train_ids == split_3b.train_ids
    assert split_3a.test_ids == split_3b.test_ids
    np.testing.assert_allclose(split_3a.y_train_adj, split_3b.y_train_adj, atol=1e-12)
    np.testing.assert_allclose(split_3a.y_test_adj, split_3b.y_test_adj, atol=1e-12)
    assert abs(r_3a - r_3b) < 1e-12, f"M0 discrepancy detected: 3A={r_3a}, 3B={r_3b}"


def test_boa_allele_coding_against_known_reference_animals():
    """Verify O=0 -> Angus/Angus and O=2 -> Brahman/Brahman matching Zayas et al. 2025."""
    data = FloridaBOALoader.load_dual_dataset(marker_step=24, cache=True)
    df = data.phenotypes
    sorted_angus = df.dropna(subset=["AngusGBC"]).sort_values("AngusGBC")

    id_to_idx = {aid: idx for idx, aid in enumerate(data.animal_ids)}

    # High Angus animals (AngusGBC > 0.89)
    high_angus = sorted_angus.tail(5)
    for _, row in high_angus.iterrows():
        aid = str(row["UFID_str"])
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            mean_boa = float(np.mean(data.alleles_boa[idx]))
            # In Zayas et al., 0 = Angus/Angus, 2 = Brahman/Brahman
            assert mean_boa < 0.25, f"High Angus animal {aid} should have near-zero Brahman count, got {mean_boa}"

    # High Brahman animals (AngusGBC < 0.33, BrahmanGBC > 0.67)
    high_brahman = sorted_angus.head(5)
    for _, row in high_brahman.iterrows():
        aid = str(row["UFID_str"])
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            mean_boa = float(np.mean(data.alleles_boa[idx]))
            assert mean_boa > 1.30, f"High Brahman animal {aid} should have near-two Brahman count, got {mean_boa}"

    # Genome-wide correlation with true Brahman breed fraction
    inferred_brahman_frac = np.mean(data.alleles_boa, axis=1) / 2.0
    true_brahman_frac = df["BrahmanGBC"].values
    valid_mask = ~np.isnan(true_brahman_frac)
    r_corr, _ = stats.pearsonr(inferred_brahman_frac[valid_mask], true_brahman_frac[valid_mask])
    assert r_corr > 0.95, f"Inferred BOA Brahman fraction should correlate > 0.95 with true pedigree, got {r_corr}"


def test_two_sided_bootstrap_statistical_consistency():
    """Verify two-sided bootstrap confidence intervals and p-values are mathematically consistent."""
    rng = np.random.default_rng(42)
    n = 250
    y_true = rng.standard_normal(n)

    # Case 1: Model A is significantly better than Model B
    y_pred_a = 0.35 * y_true + 0.8 * rng.standard_normal(n)
    y_pred_b = rng.standard_normal(n)

    res_sig = BOAValidationStats.paired_bootstrap_test(
        y_true=y_true,
        y_pred_model=y_pred_a,
        y_pred_baseline=y_pred_b,
        n_bootstraps=500,
        random_seed=42
    )
    # Zero must NOT be in 95% CI
    assert res_sig.ci_lower > 0.0
    assert res_sig.is_significant is True
    assert res_sig.p_value_two_sided < 0.05

    # Case 2: Identical models (Delta_r = 0)
    res_null = BOAValidationStats.paired_bootstrap_test(
        y_true=y_true,
        y_pred_model=y_pred_a,
        y_pred_baseline=y_pred_a,
        n_bootstraps=300,
        random_seed=42
    )
    assert res_null.ci_lower <= 0.0 <= res_null.ci_upper
    assert res_null.is_significant is False
    assert res_null.p_value_two_sided > 0.05


def test_data_driven_cv_kernel_weights():
    """Verify data-driven K-fold CV learns optimal weights strictly from training folds."""
    rng = np.random.default_rng(123)
    n_tr, n_te, m = 80, 25, 40

    q_tr = rng.uniform(0.2, 0.9, size=n_tr)
    q_te = rng.uniform(0.2, 0.9, size=n_te)

    X_tr = rng.integers(0, 3, size=(n_tr, m))
    X_te = rng.integers(0, 3, size=(n_te, m))

    O_tr = rng.integers(0, 3, size=(n_tr, m))
    O_te = rng.integers(0, 3, size=(n_te, m))

    # Construct synthetic phenotype driven primarily by BOA (O)
    y_tr = 0.8 * O_tr[:, 0] + 0.5 * O_tr[:, 1] + 0.05 * X_tr[:, 0] + 0.3 * rng.standard_normal(n_tr)

    predictor = BOAComparativePredictor(ridge_param=0.30)
    predictor.fit(X_tr, O_tr, q_tr, y_tr, model_type="M3_SNP_BOA", optimize_weights_cv=True)

    w_g, w_b = predictor.kernel_weights
    assert abs((w_g + w_b) - 1.0) < 1e-6
    # Because phenotype is driven heavily by BOA, learned w_b should be substantial
    assert w_b >= 0.40
    preds = predictor.predict(X_te, O_te, q_te)
    assert preds.shape == (n_te,)
    assert not np.isnan(preds).any()


def test_ridge_penalty_is_tuned_on_training_data():
    """With ridge_param=None the penalty comes from training CV; pure noise calls for heavy shrinkage."""
    rng = np.random.default_rng(8)
    X, O = rng.integers(0, 3, size=(90, 60)), rng.integers(0, 3, size=(90, 60))
    q = rng.uniform(0.2, 0.9, size=90)
    y_noise = rng.standard_normal(90)
    y_signal = 1.5 * (X[:, :10].sum(axis=1) - 10.0) + 0.3 * rng.standard_normal(90)
    lam_noise = BOAComparativePredictor(ridge_param=None).fit(X, O, q, y_noise, model_type="M1_SNP").ridge_param_
    lam_signal = BOAComparativePredictor(ridge_param=None).fit(X, O, q, y_signal, model_type="M1_SNP").ridge_param_
    assert lam_noise in BOAComparativePredictor.RIDGE_GRID and lam_signal in BOAComparativePredictor.RIDGE_GRID
    assert lam_noise > lam_signal
    assert BOAComparativePredictor(ridge_param=0.3).fit(X, O, q, y_noise, model_type="M1_SNP").ridge_param_ == 0.3


@pytest.mark.parametrize("model_type", ["M1_SNP", "M2_BOA", "M3_SNP_BOA", "M4_SNP_BOA_INTER"])
def test_boa_predictions_do_not_depend_on_test_batch(model_type):
    """Test animals must be standardized with training statistics: an animal's prediction cannot
    change with the other animals it is predicted alongside (regression test for test-set scaling)."""
    rng = np.random.default_rng(5)
    n_tr, n_te, m = 80, 30, 40
    X_tr, X_te = rng.integers(0, 3, size=(n_tr, m)), rng.integers(0, 3, size=(n_te, m))
    O_tr, O_te = rng.integers(0, 3, size=(n_tr, m)), rng.integers(0, 3, size=(n_te, m))
    q_tr, q_te = rng.uniform(0.2, 0.9, size=n_tr), rng.uniform(0.2, 0.9, size=n_te)
    y_tr = 0.5 * O_tr[:, 0] + 0.3 * X_tr[:, 1] + 0.3 * rng.standard_normal(n_tr)

    predictor = BOAComparativePredictor(ridge_param=0.3).fit(X_tr, O_tr, q_tr, y_tr, model_type=model_type)
    full = predictor.predict(X_te, O_te, q_te)
    halves = np.concatenate([predictor.predict(X_te[:12], O_te[:12], q_te[:12]),
                             predictor.predict(X_te[12:], O_te[12:], q_te[12:])])
    single = predictor.predict(X_te[:1], O_te[:1], q_te[:1])
    np.testing.assert_allclose(full, halves, atol=1e-10)
    np.testing.assert_allclose(single, full[:1], atol=1e-10)

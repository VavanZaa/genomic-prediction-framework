"""Unit tests for Genomic Out-of-Distribution Detector & Selective Prediction Engine."""
import numpy as np
import pytest
from src.evaluation.ood_detector import GenomicOODDetector, ConfidenceTier, OODAssessment


@pytest.fixture
def synthetic_training_population():
    rng = np.random.default_rng(42)
    n_train = 100
    n_snps = 300

    # Base frequencies
    p = rng.uniform(0.2, 0.8, size=n_snps)
    g_train = rng.binomial(2, p, size=(n_train, n_snps)).astype(np.int8)
    qb_train = rng.uniform(0.10, 0.30, size=n_train)

    # In-distribution animal (sampled from same distribution)
    g_in_dist = rng.binomial(2, p, size=n_snps).astype(np.int8)

    # Mildly out-of-distribution (shifted frequencies)
    p_mild = np.clip(p + 0.15, 0.05, 0.95)
    g_mild = rng.binomial(2, p_mild, size=n_snps).astype(np.int8)

    # Strongly out-of-distribution (inverted frequencies / orthogonal)
    p_extreme = 1.0 - p
    g_extreme = rng.binomial(2, p_extreme, size=n_snps).astype(np.int8)

    return {
        "g_train": g_train,
        "qb_train": qb_train,
        "g_in_dist": g_in_dist,
        "g_mild": g_mild,
        "g_extreme": g_extreme,
    }


def test_ood_detector_fit_and_assessment(synthetic_training_population):
    data = synthetic_training_population
    detector = GenomicOODDetector(n_components=10)
    detector.fit(data["g_train"], ancestry_qb_train=data["qb_train"])
    assert detector.is_fitted
    assert detector.qb_min == pytest.approx(data["qb_train"].min())
    assert detector.qb_max == pytest.approx(data["qb_train"].max())

    # In-distribution test
    assess_in = detector.assess_animal(data["g_in_dist"], ancestry_qb=0.20, animal_id="in_dist_01")
    assert isinstance(assess_in, OODAssessment)
    assert assess_in.confidence_tier in [ConfidenceTier.HIGH, ConfidenceTier.MODERATE]
    assert not assess_in.is_refused
    assert assess_in.allow_numeric_prediction is True or assess_in.allow_ranking_only is True
    assert assess_in.uncertainty_inflation_factor >= 1.0

    # Extreme OOD test
    assess_ext = detector.assess_animal(data["g_extreme"], ancestry_qb=0.20, animal_id="extreme_01")
    assert assess_ext.d_pca > assess_in.d_pca
    assert assess_ext.d_ood_composite > assess_in.d_ood_composite
    assert assess_ext.uncertainty_inflation_factor > assess_in.uncertainty_inflation_factor
    assert assess_ext.confidence_tier == ConfidenceTier.LOW
    assert assess_ext.is_refused
    assert assess_ext.allow_numeric_prediction is False
    assert "Genomic support insufficient" in assess_ext.refusal_reason


def test_ancestry_domain_extrapolation(synthetic_training_population):
    data = synthetic_training_population
    detector = GenomicOODDetector(n_components=10)
    detector.fit(data["g_train"], ancestry_qb_train=data["qb_train"])

    # Query with q_B = 0.50 (well outside [0.10, 0.30])
    assess_extrap = detector.assess_animal(data["g_in_dist"], ancestry_qb=0.50, animal_id="extrap_qb")
    assert assess_extrap.d_ancestry > 1.0
    # Must NOT be HIGH confidence due to ancestry extrapolation
    assert assess_extrap.confidence_tier in [ConfidenceTier.MODERATE, ConfidenceTier.LOW]


def test_selective_risk_and_false_safe_rate():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.normal(30, 5, size=n)
    # Predictors with varying noise correlated with confidence
    confidences = rng.uniform(0.1, 0.9, size=n)
    noise = rng.normal(0, 1.0 / confidences, size=n)
    y_pred = y_true + noise

    risk_df = GenomicOODDetector.compute_selective_risk(y_true, y_pred, confidences)
    assert len(risk_df) > 0
    assert "Coverage" in risk_df.columns
    assert "Risk_MSE" in risk_df.columns

    # Mock assessments
    assessments = [
        OODAssessment(
            animal_id=f"id_{i}",
            d_pca=0.5,
            g_max=0.1,
            d_kernel=0.2,
            d_af=1.0,
            d_ancestry=0.0,
            d_ood_composite=0.3,
            selective_confidence=confidences[i],
            confidence_tier=ConfidenceTier.HIGH if confidences[i] > 0.5 else ConfidenceTier.MODERATE,
            uncertainty_inflation_factor=1.1,
            is_refused=False
        )
        for i in range(n)
    ]

    fsr_res = GenomicOODDetector.compute_false_safe_rate(y_true, y_pred, assessments)
    assert "false_safe_rate" in fsr_res
    assert 0.0 <= fsr_res["false_safe_rate"] <= 1.0

    # AURC and Coverage at Risk
    aurc = GenomicOODDetector.compute_aurc(risk_df)
    assert aurc > 0.0

    cov_target = GenomicOODDetector.compute_coverage_at_risk(risk_df, max_risk=float(risk_df["Risk_MSE"].max()))
    assert 0.0 <= cov_target <= 1.0

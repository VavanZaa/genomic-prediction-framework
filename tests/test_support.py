"""Empirical k-NN support replaces min–max ancestry bounds (Layers 4B, 6A, 6B)."""
import numpy as np
import pytest

from src.evaluation.masked_reconstruction import MaskedHistoricalReconstructor
from src.evaluation.ood_detector import ConfidenceTier, GenomicOODDetector
from src.evaluation.support import KNNSupport


def _clusters(rng):
    """Two ancestry clusters and one isolated animal: min–max spans 0.10–0.95."""
    return np.r_[rng.uniform(0.10, 0.20, 300), rng.uniform(0.50, 0.60, 300), [0.95]]


def test_gap_and_isolated_animal_are_not_support():
    q = _clusters(np.random.default_rng(0))
    s = KNNSupport().fit(q)
    assert q.min() < 0.35 < q.max() and q.min() < 0.90 < q.max()      # both inside min–max
    assert s.supported([0.15, 0.55]).all()
    assert not s.supported([0.35]).any()                                # gap between clusters
    assert not s.supported([0.90]).any()                                # next to one lone animal
    assert s.excess([0.35])[0] > 0 and s.excess([0.15])[0] == 0
    assert np.mean(s.train_distance_ <= s.radius_) == pytest.approx(0.95, abs=0.01)


def test_ood_detector_uses_empirical_ancestry_support():
    rng = np.random.default_rng(1)
    q = _clusters(rng)
    p = rng.uniform(0.2, 0.8, 200)
    detector = GenomicOODDetector(n_components=10).fit(rng.binomial(2, p, size=(len(q), 200)), ancestry_qb_train=q)
    g_new = rng.binomial(2, p, size=200)
    inside = detector.assess_animal(g_new, ancestry_qb=0.15)
    gap = detector.assess_animal(g_new, ancestry_qb=0.35)
    assert inside.d_ancestry == 0.0 and inside.ancestry_supported
    assert gap.d_ancestry > 0.0 and not gap.ancestry_supported
    assert gap.confidence_tier != ConfidenceTier.HIGH
    assert np.isfinite(inside.d_support) and inside.d_support > 0


def test_safe_optimum_ignores_an_isolated_extreme_animal():
    rng = np.random.default_rng(2)
    qb = np.r_[rng.uniform(0.05, 0.45, 300), [0.95]]
    y = 32.0 - 10.0 * qb + rng.normal(0, 1.0, len(qb))
    res = MaskedHistoricalReconstructor(random_seed=0).reconstruct_optimal_cross(
        np.zeros((len(qb), 3)), y, qb, np.zeros((20, 3)), y[:20], qb[:20], tau_load=28.5, q_steps=101)
    assert res.predicted_optimal_qb == pytest.approx(1.0)
    assert not res.is_opt_within_support
    # Min–max bounds reached 0.95, and their 0.70-confidence rule went further; support stops at the data
    assert res.predicted_safe_qb <= 0.46

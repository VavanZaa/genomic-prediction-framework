"""Risk-coverage threshold selection and the calibrated OOD mode (docs/OOD_CALIBRATION.md)."""
import json

import numpy as np
import pytest

from src.evaluation.ood_detector import ConfidenceTier, GenomicOODDetector
from src.evaluation.risk_calibration import (load_frozen_thresholds, risk_coverage, select_threshold,
                                             sha256_file)

GRID = np.round(np.arange(0.5, 4.01, 0.1), 2)


def test_threshold_stops_where_risk_exceeds_the_target():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 4, 4000)
    loss = rng.exponential(1.0 + np.maximum(0, s - 2.0), size=4000)     # risk rises beyond s = 2
    curve = risk_coverage(s, loss, GRID, n_boot=300)
    c_ucb = select_threshold(curve, risk_max=1.10)
    assert c_ucb is not None and 1.6 <= c_ucb <= 2.8
    point = curve[(curve["n_accepted"] >= 100) & (curve["risk"] <= 1.10)]["threshold"].max()
    assert c_ucb <= point                                                # upper bound is conservative
    assert (curve["risk_ucb"] >= curve["risk"] - 1e-12).all()


def test_no_threshold_when_every_region_is_too_risky_and_all_when_none_is():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 4, 2000)
    high = risk_coverage(s, rng.exponential(2.0, 2000), GRID, n_boot=200)
    low = risk_coverage(s, rng.exponential(0.5, 2000), GRID, n_boot=200)
    assert select_threshold(high, risk_max=1.10) is None
    assert select_threshold(low, risk_max=1.10) == pytest.approx(GRID.max())


def test_bootstrap_resamples_units_not_records():
    s = np.repeat(np.linspace(0, 3, 50), 4)                  # 50 animals, 4 records each
    loss = np.repeat(np.linspace(0.5, 1.5, 50), 4)
    units = np.repeat(np.arange(50), 4)
    by_unit = risk_coverage(s, loss, [3.0], units=units, n_boot=500)["risk_ucb"].iloc[0]
    by_record = risk_coverage(s, loss, [3.0], n_boot=500)["risk_ucb"].iloc[0]
    assert by_unit > by_record                               # correlated records widen the bound


def test_frozen_thresholds_refuse_a_changed_file(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"c_star": {"ranking": 1.5, "absolute": None}}), encoding="utf-8")
    frozen = sha256_file(path)
    assert load_frozen_thresholds(path, frozen)["ranking"] == 1.5
    path.write_text(json.dumps({"c_star": {"ranking": 2.5, "absolute": None}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_frozen_thresholds(path, frozen)


def test_calibrated_detector_tiers():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.2, 0.8, 200)
    g_train = rng.binomial(2, p, size=(400, 200))
    q = rng.uniform(0.1, 0.3, 400)
    det = GenomicOODDetector(n_components=10, calibrated_thresholds={"ranking": 1.5, "absolute": 1.2})
    det.fit(g_train, ancestry_qb_train=q, env_ids_train=["herdA"] * 400)
    g_in = rng.binomial(2, p, size=200)
    inside = det.assess_animal(g_in, ancestry_qb=0.2, env_id="herdA")
    new_herd = det.assess_animal(g_in, ancestry_qb=0.2, env_id="herdB")
    far = det.assess_animal(g_in, ancestry_qb=0.9, env_id="herdA")
    assert inside.support_score <= 1.2 and inside.confidence_tier == ConfidenceTier.HIGH
    assert new_herd.confidence_tier == ConfidenceTier.MODERATE and new_herd.allow_ranking_only
    assert not new_herd.allow_numeric_prediction
    assert far.confidence_tier == ConfidenceTier.LOW and far.is_refused
    genomic, ancestry = det.support_ratios(g_train[:5], q[:5])
    assert genomic.shape == (5,) and ancestry.shape == (5,)
    closed = GenomicOODDetector(n_components=10, calibrated_thresholds={"ranking": None, "absolute": None})
    closed.fit(g_train, ancestry_qb_train=q)
    assert closed.assess_animal(g_in, ancestry_qb=0.2).is_refused

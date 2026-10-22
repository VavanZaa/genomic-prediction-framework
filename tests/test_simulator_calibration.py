"""Unit tests for Layer 5A Simulator Calibration Suite."""
import numpy as np
import pandas as pd
import pytest
from src.evaluation.simulator_calibration import SimulatorCalibrator, CalibrationMetrics


@pytest.fixture
def dummy_genomic_data():
    rng = np.random.default_rng(42)
    n_animals = 50
    n_snps = 200

    # Founder freqs
    p = rng.uniform(0.1, 0.9, size=n_snps)
    g_real = rng.binomial(2, p, size=(n_animals, n_snps)).astype(np.int8)

    # Simulated with slightly shifted frequencies
    g_sim_close = g_real.copy()
    g_sim_dist = rng.binomial(2, np.roll(p, 10), size=(n_animals, n_snps)).astype(np.int8)

    # Marker map across 2 autosomes
    chrom = np.array([1] * 100 + [2] * 100)
    pos = np.concatenate([np.linspace(1000, 1_000_000, 100), np.linspace(1000, 1_000_000, 100)])
    marker_map = pd.DataFrame({"chrom": chrom, "pos": pos})

    # BOA dummy
    boa_real = rng.choice([0, 1, 2], size=(n_animals, n_snps), p=[0.5, 0.3, 0.2]).astype(np.int8)
    boa_sim = boa_real.copy()

    # Phenotypes
    y_real = rng.normal(0, 1, size=(n_animals, 2))
    y_sim = y_real.copy()

    return {
        "g_real": g_real,
        "g_sim_close": g_sim_close,
        "g_sim_dist": g_sim_dist,
        "marker_map": marker_map,
        "boa_real": boa_real,
        "boa_sim": boa_sim,
        "y_real": y_real,
        "y_sim": y_sim,
    }


def test_identical_populations_zero_discrepancy(dummy_genomic_data):
    d = dummy_genomic_data
    calib = SimulatorCalibrator.evaluate_simulation(
        g_real=d["g_real"],
        g_sim=d["g_sim_close"],
        marker_map=d["marker_map"],
        boa_real=d["boa_real"],
        boa_sim=d["boa_sim"],
        y_real=d["y_real"],
        y_sim=d["y_sim"],
        thresholds=[0.0, 0.0]
    )
    assert isinstance(calib, CalibrationMetrics)
    assert calib.d_afs == pytest.approx(0.0, abs=1e-5)
    assert calib.d_het == pytest.approx(0.0, abs=1e-5)
    assert calib.d_ld == pytest.approx(0.0, abs=1e-5)
    assert calib.d_tract == pytest.approx(0.0, abs=1e-5)
    assert calib.d_spec == pytest.approx(0.0, abs=1e-5)
    assert calib.d_pheno == pytest.approx(0.0, abs=1e-5)
    assert calib.d_tail == pytest.approx(0.0, abs=1e-5)
    assert calib.d_composite == pytest.approx(0.0, abs=1e-5)


def test_divergent_populations_positive_discrepancy(dummy_genomic_data):
    d = dummy_genomic_data
    calib = SimulatorCalibrator.evaluate_simulation(
        g_real=d["g_real"],
        g_sim=d["g_sim_dist"],
        marker_map=d["marker_map"]
    )
    assert calib.d_afs > 0.01
    assert calib.d_composite > 0.01

    df = calib.to_dataframe()
    assert len(df) == 8
    assert "COMPOSITE CALIBRATION (D_sim-real)" in df["Metric"].values
    assert len(calib.raw_vector) == 7


def test_dirichlet_weight_robustness(dummy_genomic_data):
    d = dummy_genomic_data
    calib_close = SimulatorCalibrator.evaluate_simulation(
        g_real=d["g_real"],
        g_sim=d["g_sim_close"],
        marker_map=d["marker_map"]
    )
    calib_dist = SimulatorCalibrator.evaluate_simulation(
        g_real=d["g_real"],
        g_sim=d["g_sim_dist"],
        marker_map=d["marker_map"]
    )

    # calib_close should unequivocally beat calib_dist under Dirichlet weights
    res = SimulatorCalibrator.evaluate_dirichlet_robustness(
        calib_a=calib_dist,
        calib_b=calib_close,
        n_samples=500
    )
    assert res["prob_model_b_superior"] >= 0.99
    assert res["mean_discrepancy_advantage"] > 0.0
    assert len(res["ci95_advantage"]) == 2

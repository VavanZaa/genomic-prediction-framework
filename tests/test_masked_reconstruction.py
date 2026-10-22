"""Unit tests for Real-Data Masked Population Reconstruction (Layer 6B-v2)."""
import numpy as np
import pytest
from src.evaluation.masked_reconstruction import MaskedHistoricalReconstructor, MaskedReconstructionResult


def _genotypes(rng, qb, m=200):
    p_angus = rng.uniform(0.1, 0.5, size=m)
    p_brahman = rng.uniform(0.5, 0.9, size=m)
    return np.vstack([rng.binomial(2, (1.0 - q) * p_angus + q * p_brahman) for q in qb]).astype(np.int8)


@pytest.fixture
def synthetic_multibreed_data():
    rng = np.random.default_rng(42)
    # Population IDs: 200 Seminole (source), 100 Williamson/Wes (masked target)
    pop_ids = np.array(["Semin"] * 200 + ["Wes"] * 100)
    qb = np.concatenate([rng.uniform(0.05, 0.70, size=200), rng.normal(0.35, 0.05, size=100)])
    # Purely additive Brahman effect on heat load: no heterosis
    y_load = 32.0 - 10.0 * qb + rng.normal(0, 1.5, size=300)
    return {
        "genotypes": _genotypes(rng, qb),
        "phenotypes_load": y_load,
        "ancestry_qb": qb,
        "population_ids": pop_ids,
    }


def _reconstruct(d, **kw):
    rec = MaskedHistoricalReconstructor(historical_target_qb=0.375, random_seed=42)
    parts = rec.partition_by_population_identity(
        genotypes=d["genotypes"], phenotypes_load=d["phenotypes_load"], ancestry_qb=d["ancestry_qb"],
        population_ids=d["population_ids"], withhold_pop_id="Wes", train_pop_id="Semin"
    )
    g_train, y_train, qb_train, g_masked, y_masked, qb_masked = parts
    res = rec.reconstruct_optimal_cross(g_train, y_train, qb_train, g_masked, y_masked, qb_masked, **kw)
    return parts, res


def test_partition_by_population_identity_and_reconstruction(synthetic_multibreed_data):
    (g_train, y_train, qb_train, g_masked, y_masked, qb_masked), res = _reconstruct(
        synthetic_multibreed_data, tau_load=28.5, q_steps=25
    )
    # Strict population-identity partition (no q_B filtering)
    assert g_train.shape[0] == 200 and y_train.shape[0] == 200
    assert g_masked.shape[0] == 100 and y_masked.shape[0] == 100

    assert isinstance(res, MaskedReconstructionResult)
    assert res.parameter_source == "fitted on training population"
    # The model is fitted from the training data: slope ~ -10, no heterosis
    assert res.fitted_beta == pytest.approx(-10.0, abs=1.5)
    assert res.fitted_heterosis == pytest.approx(0.0, abs=1.5)

    # A monotone additive effect puts the heat-load optimum at pure Brahman, outside the
    # training support; the support-constrained optimum stays near the edge of the data.
    assert res.predicted_optimal_qb == pytest.approx(1.0)
    assert res.optimum_at_boundary
    assert not res.is_opt_within_support
    assert 0.6 <= res.predicted_safe_qb <= float(np.max(qb_train)) + 0.5 * float(np.std(qb_train))
    assert res.wasserstein_distance_load >= 0.0

    df = res.to_dataframe()
    assert len(df) == 1
    for col in ("Predicted_Optimal_qB", "Historical_Target_qB", "Empirical_Hidden_Mean_qB",
                "Error_vs_Historical", "Error_vs_Empirical", "Parameter_Source", "Fitted_Beta_Brahman"):
        assert col in df.columns


def test_interior_optimum_is_recovered_from_heterosis():
    """With heterosis lowering heat load, the fitted model finds the analytic interior optimum."""
    rng = np.random.default_rng(7)
    qb = np.concatenate([rng.uniform(0.02, 0.98, size=400), rng.normal(0.4, 0.05, size=100)])
    beta, h = -2.0, -3.0
    y = 30.0 + beta * qb + h * 4.0 * qb * (1.0 - qb) + rng.normal(0, 1.0, size=len(qb))
    data = {"genotypes": _genotypes(rng, qb, m=50), "phenotypes_load": y, "ancestry_qb": qb,
            "population_ids": np.array(["Semin"] * 400 + ["Wes"] * 100)}
    _, res = _reconstruct(data, tau_load=float(np.percentile(y[:400], 35)), q_steps=51)

    expected = 0.5 * (1.0 + beta / (4.0 * h))
    assert res.fitted_heterosis == pytest.approx(h, abs=0.5)
    assert res.predicted_optimal_qb == pytest.approx(expected, abs=0.06)
    assert not res.optimum_at_boundary
    assert res.is_opt_within_support


def test_literature_prior_is_explicit(synthetic_multibreed_data):
    _, res = _reconstruct(synthetic_multibreed_data, tau_load=28.5, literature_prior=(-2.71, -1.18))
    assert res.parameter_source.startswith("literature prior")
    assert res.predicted_optimal_qb == pytest.approx(0.5 * (1 + -2.71 / (4 * -1.18)))

"""Layer 5B-T recovery metrics on synthetic trios stored the way BEAGLE stores phased trios."""
import numpy as np
import pytest
from src.evaluation.trio_recovery import (
    CHROM_LENGTH_MB, TrioPanel, analyze_panel, midparent_accuracy, segregation_table, simulate_gametes,
)


def _synthetic_panel(n_trios=30, markers_per_chrom=400, chroms=(1, 2, 3), seed=0):
    """Parents drawn from breed allele frequencies, calf by true meiosis; each parent is then stored as
    (transmitted, untransmitted) haplotypes, exactly as BEAGLE trio phasing writes them."""
    rng = np.random.default_rng(seed)
    chrom = np.repeat(np.array(chroms, dtype=np.int16), markers_per_chrom)
    m = len(chrom)
    bounds = {c: (i * markers_per_chrom, (i + 1) * markers_per_chrom) for i, c in enumerate(chroms)}
    lengths = {c: CHROM_LENGTH_MB[c] / 100.0 for c in chroms}
    p = rng.uniform(0.1, 0.9, size=m)
    cols = []
    for _ in range(n_trios):
        a1, b1, a2, b2 = [(rng.random(m) < p).astype(np.int8) for _ in range(4)]
        t1 = simulate_gametes(a1, b1, bounds, lengths, 1, rng)[0]
        t2 = simulate_gametes(a2, b2, bounds, lengths, 1, rng)[0]
        cols += [t1, a1 + b1 - t1, t2, a2 + b2 - t2]
    return TrioPanel("Synthetic", np.column_stack(cols).astype(np.int8), chrom)


@pytest.fixture(scope="module")
def results():
    panel = _synthetic_panel()
    return panel, analyze_panel(panel, k=80, n_traits=10, n_qtl=200, seed=1)


def test_parents_predict_the_calf_far_better_than_breed_frequencies(results):
    _, df = results
    # Mid-parent explains half the calf's genotype deviation: r ~ sqrt(0.5)
    assert df["deviation_r_parents"].mean() == pytest.approx(np.sqrt(0.5), abs=0.05)
    assert (df["loglik_per_locus_parents"] > df["loglik_per_locus_breed"]).all()
    assert (df["loglik_per_locus_parents"] > df["loglik_per_locus_wrong_parents"]).all()
    assert (df["call_accuracy_parents"] > df["call_accuracy_breed"]).all()


def test_simulated_offspring_distribution_is_calibrated(results):
    _, df = results
    assert 0.82 <= df["gv_cover90_sim"].mean() <= 0.97
    assert df["gv_z_sim_sd"].mean() == pytest.approx(1.0, abs=0.2)
    assert 0.8 <= df["gv_cover90_const"].mean() <= 0.98
    assert midparent_accuracy(df)["r_midparent_child"].mean() > 0.5


def test_segregation_matches_mendel(results):
    panel, _ = results
    seg = segregation_table(panel).set_index("mating_type")
    assert seg.loc["1x1", "obs_1"] == pytest.approx(0.5, abs=0.02)
    assert seg.loc["0x1", "obs_0"] == pytest.approx(0.5, abs=0.02)
    assert seg.loc["0x2", "obs_1"] == pytest.approx(1.0, abs=1e-9)

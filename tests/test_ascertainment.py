"""SNP-array ascertainment of simulated genomes, and correct breed pools in the coalescent engine."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.simulation.ascertainment import ArrayAscertainment
from src.simulation.coalescent_engine import BovineCoalescentSimulator


def _real_panel(rng, n=200, m=1500):
    p = rng.beta(0.9, 0.9, m)
    G = rng.binomial(2, p, (n, m)).astype(float)
    G[rng.random(G.shape) < 0.01] = np.nan
    marker_map = pd.DataFrame({"chrom": 1, "snp_id": [f"s{i}" for i in range(m)],
                               "pos_bp": np.sort(rng.choice(50_000_000, m, replace=False))})
    return G, marker_map


def test_ascertainment_matches_real_maf_spectrum_qc_and_missingness():
    rng = np.random.default_rng(0)
    G, marker_map = _real_panel(rng)
    asc = ArrayAscertainment(min_maf=0.03).fit(G, marker_map)
    assert asc.missing_rate_ == pytest.approx(0.01, abs=0.002)

    freq = rng.beta(0.2, 1.5, 20000)                      # coalescent-like: mostly rare sites
    pos = np.sort(rng.choice(15_000_000, 20000, replace=False))
    maf = np.minimum(freq, 1 - freq)
    sel = asc.select(freq, pos, 600)
    assert len(sel) == 600 and len(np.unique(sel)) == 600
    assert (maf[sel] >= 0.03).all()
    assert stats.ks_2samp(maf[sel], asc.target_maf_).statistic < 0.08     # array-like spectrum
    assert stats.ks_2samp(maf, asc.target_maf_).statistic > 0.4           # raw sites are not
    assert np.diff(np.sort(pos[sel])).max() < 0.05 * 15_000_000           # spread along the chromosome

    X = asc.add_missing_calls(np.ones((300, 400)), np.random.default_rng(1))
    assert np.isnan(X).mean() == pytest.approx(asc.missing_rate_, abs=0.003)


def test_coalescent_haplotype_pools_hold_their_own_breed():
    c = BovineCoalescentSimulator().simulate_chromosome(
        length_bp=2_000_000, num_angus_haplotypes=30, num_brahman_haplotypes=30, random_seed=3)
    assert c.angus_haplotypes.shape[1] == 30 and c.brahman_haplotypes.shape[1] == 30
    assert set(np.unique(c.angus_haplotypes)) <= {0, 1} and set(np.unique(c.brahman_haplotypes)) <= {0, 1}
    pa, pb = c.angus_haplotypes.mean(axis=1), c.brahman_haplotypes.mean(axis=1)
    assert np.mean(np.abs(pa - pb)) > 0.5      # long-separated breeds differ at most sites

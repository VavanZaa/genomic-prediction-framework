"""Unit and Integration Tests for Advanced Simulation Engines.

Tests:
1. BovineCoalescentSimulator (a null / stress-test generator) produces common-variant LD above a permutation null.
2. RealHaplotypeResampler reproduces empirical cattle heterozygosity (Ho in [0.15, 0.35]) and ancestry.
3. Out-of-sample genomic prediction achieves positive predictive correlation on realistic LD templates.
"""
from pathlib import Path
import numpy as np
import pytest
from scipy import stats

from src.simulation.coalescent_engine import BovineCoalescentSimulator
from src.simulation.real_haplotype_engine import RealHaplotypeResampler
from src.prediction.models import AdditiveGBLUP


def test_coalescent_simulation_generates_ld():
    """Verify that msprime historical coalescence creates realistic Linkage Disequilibrium."""
    sim = BovineCoalescentSimulator()
    chrom = sim.simulate_chromosome(
        chrom_id=1,
        length_bp=5_000_000,
        num_angus_haplotypes=40,
        num_brahman_haplotypes=40,
        random_seed=42
    )

    assert chrom.num_sites > 500
    assert len(chrom.positions_bp) == chrom.num_sites

    # Adjacent-site r^2 among common variants (MAF >= 0.05, as on SNP arrays) in the Angus pool, against
    # the same sites with haplotypes permuted independently per site (no LD). Including rare sites made
    # the statistic swing from 0.07 to 0.35 across seeds; common-variant r^2 is 0.17-0.52 vs a null of 0.02-0.04.
    angus_pool = chrom.angus_haplotypes.astype(float)  # (M, 40)
    p = angus_pool.mean(axis=1)
    common = np.where(np.minimum(p, 1 - p) >= 0.05)[0][:50]
    assert len(common) >= 20

    def adjacent_r2(pool):
        return float(np.nanmean(np.diag(np.corrcoef(pool), k=1) ** 2))

    rng = np.random.default_rng(0)
    linked = adjacent_r2(angus_pool[common])
    unlinked = adjacent_r2(np.array([rng.permutation(row) for row in angus_pool[common]]))
    assert linked > 3.0 * unlinked


def test_real_haplotype_resampling_fidelity():
    """Verify that resampled progeny match the SNP heterozygosity of the real Florida herd."""
    resampler = RealHaplotypeResampler(marker_step=48)
    real = resampler.data.genotypes
    called = np.isfinite(real)
    ho_real = np.sum(real == 1, axis=1) / np.sum(called, axis=1)
    pop = resampler.simulate_multibreed_population(
        num_animals=150,
        angus_fraction_mean=0.71,
        angus_fraction_std=0.07,
        random_seed=42
    )

    assert pop.num_animals == 150
    assert pop.num_snps > 2000
    assert pop.genotypes.shape == (150, pop.num_snps)

    # Values strictly in {0, 1, 2}
    assert set(np.unique(pop.genotypes)).issubset({0, 1, 2})

    # SNP heterozygosity matches the real herd: mean within 0.03 and between-animal spread within 2x.
    # (The old "sd > 0.03" bound described BOA ancestry heterozygosity, not SNP heterozygosity.)
    ho = np.mean(pop.genotypes == 1, axis=1)
    assert 0.15 <= np.mean(ho) <= 0.45
    assert abs(np.mean(ho) - np.mean(ho_real)) < 0.03
    assert 0.5 < np.std(ho) / np.std(ho_real) < 2.0

    # Ancestry proportions valid
    assert np.all(pop.ancestry_fractions >= 0.0)
    assert np.all(pop.ancestry_fractions <= 1.0)
    assert abs(np.mean(pop.ancestry_fractions[:, 0]) - 0.71) < 0.05


def test_methods_prediction_convergence():
    """Verify GBLUP prediction on synthetic genomes with realistic LD.

    One 60-animal test set gives r anywhere from -0.04 to 0.30, so accuracy against the true
    genetic value is averaged over three independent populations of 600 animals.
    """
    resampler = RealHaplotypeResampler(marker_step=48)
    accuracies = []
    for seed in range(3):
        pop = resampler.simulate_multibreed_population(num_animals=600, random_seed=seed)
        rng = np.random.default_rng(seed)
        # 20 causal QTLs with heritability h2 = 0.30
        qtl_idx = rng.choice(pop.num_snps, size=20, replace=False)
        g_true = pop.genotypes[:, qtl_idx] @ rng.normal(0, 1.0, size=20)
        y = g_true + rng.normal(0, np.sqrt(np.var(g_true) * 2.33), size=len(g_true))

        train_n = 450
        pred = AdditiveGBLUP().fit(pop.genotypes[:train_n], y[:train_n]).predict(pop.genotypes[train_n:])
        accuracies.append(stats.pearsonr(g_true[train_n:], pred)[0])

    assert np.mean(accuracies) > 0.10

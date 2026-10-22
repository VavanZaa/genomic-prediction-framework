"""Tests for Stage B: Inheritance Engine (Chromosome maps, recombination, founder ancestry tracking).
"""
import numpy as np
import pytest
from scipy import stats
from src.ancestry.pedigree import BreedFraction
from src.inheritance.chromosome import ChromosomeSpec, GenomeSpec, Haplotype, Segment
from src.inheritance.meiosis import MeiosisSimulator, Population, SimulatedIndividual


def test_recombination_poisson_distribution():
    """Verify that crossover count follows Poisson distribution with lambda = chromosome length."""
    rng = np.random.default_rng(42)
    spec = ChromosomeSpec(chrom_id=1, length_m=1.20, num_markers=50, num_qtls=5)

    pat_hap = Haplotype.founder(spec, "F1_pat", "Angus", np.zeros(50, dtype=np.int8), np.zeros(5, dtype=np.int8))
    mat_hap = Haplotype.founder(spec, "F1_mat", "Brahman", np.ones(50, dtype=np.int8), np.ones(5, dtype=np.int8))

    n_samples = 2000
    crossover_counts = []

    for _ in range(n_samples):
        # We can count segments: number of segments = crossovers + 1
        gamete = MeiosisSimulator.recombine_chromosome(pat_hap, mat_hap, spec, rng)
        crossover_counts.append(len(gamete.segments) - 1)

    mean_k = np.mean(crossover_counts)
    var_k = np.var(crossover_counts)

    # For Poisson(1.2), E[X] = 1.20 and Var[X] = 1.20
    assert pytest.approx(mean_k, abs=0.05) == 1.20
    assert pytest.approx(var_k, abs=0.08) == 1.20

    # Chi-square goodness-of-fit test against Poisson(1.2)
    # Clip to 0..4 and 5+ for tail
    clipped = np.clip(crossover_counts, 0, 5)
    obs_freq = np.bincount(clipped, minlength=6)
    expected_prob = [stats.poisson.pmf(i, 1.20) for i in range(5)]
    expected_prob.append(1.0 - stats.poisson.cdf(4, 1.20))
    expected_freq = np.array(expected_prob) * n_samples

    # Normalize expected to exactly match total observed count
    expected_freq = expected_freq * (np.sum(obs_freq) / np.sum(expected_freq))

    chi2, p_val = stats.chisquare(obs_freq, expected_freq)
    assert p_val > 0.01, f"Crossover count failed Poisson goodness-of-fit test (p={p_val})"


def test_segment_continuity_no_gaps():
    """Verify that recombined gametes form continuous, non-overlapping coverage of [0, L]."""
    rng = np.random.default_rng(123)
    spec = ChromosomeSpec(chrom_id=1, length_m=1.0, num_markers=20, num_qtls=2)
    pat = Haplotype.founder(spec, "P", "BreedA", np.zeros(20, dtype=np.int8), np.zeros(2, dtype=np.int8))
    mat = Haplotype.founder(spec, "M", "BreedB", np.ones(20, dtype=np.int8), np.ones(2, dtype=np.int8))

    for _ in range(100):
        gamete = MeiosisSimulator.recombine_chromosome(pat, mat, spec, rng)
        assert len(gamete.segments) >= 1
        assert pytest.approx(gamete.segments[0].start_m) == 0.0
        assert pytest.approx(gamete.segments[-1].end_m) == 1.0

        for i in range(len(gamete.segments) - 1):
            assert pytest.approx(gamete.segments[i].end_m) == gamete.segments[i + 1].start_m


def test_ancestry_expectations_match_pedigree():
    """Milestone B Completion Criterion:
    Offspring inherit correctly; ancestry expectations match the pedigree.
    """
    rng = np.random.default_rng(2026)
    # Representative bovine genome (10 autosomes for fast test, each 1.0 Morgan)
    genome = GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=20, qtls_per_chrom=4)

    # 1. Founders: Angus (pure) and Brahman (pure)
    pop = Population(genome)
    angus_sire = pop.create_founder("ANGUS_SIRE_1", sex="M", breed="Angus", rng=rng)
    brahman_dam = pop.create_founder("BRAHMAN_DAM_1", sex="F", breed="Brahman", rng=rng)

    assert pytest.approx(angus_sire.realized_genomic_ancestry("Angus", genome)) == 1.0
    assert pytest.approx(brahman_dam.realized_genomic_ancestry("Angus", genome)) == 0.0

    # 2. F1 Cross: Exactly 50% Angus, 50% Brahman
    f1_female = MeiosisSimulator.mate(
        sire=angus_sire,
        dam=brahman_dam,
        child_id="F1_COW",
        sex="F",
        genome_spec=genome,
        rng=rng
    )
    # F1 has one full paternal Angus genome and one maternal Brahman genome
    assert pytest.approx(f1_female.realized_genomic_ancestry("Angus", genome)) == 0.50
    assert pytest.approx(f1_female.realized_genomic_ancestry("Brahman", genome)) == 0.50
    assert f1_female.pedigree_composition.get("Angus") == 0.5

    # 3. Backcross F1 female to an unrelated Brahman bull: expected 25% Angus
    brahman_sire2 = pop.create_founder("BRAHMAN_SIRE_2", sex="M", breed="Brahman", rng=rng)

    n_reps = 300
    bc_angus_fractions = []
    for i in range(n_reps):
        bc_calf = MeiosisSimulator.mate(
            sire=brahman_sire2,
            dam=f1_female,
            child_id=f"BC_{i}",
            sex="F",
            genome_spec=genome,
            rng=rng
        )
        assert bc_calf.pedigree_composition.get("Angus") == 0.25
        bc_angus_fractions.append(bc_calf.realized_genomic_ancestry("Angus", genome))

    bc_arr = np.array(bc_angus_fractions)
    # Offspring inherit correctly: sample mean must match pedigree expectation 0.25
    mean_bc = np.mean(bc_arr)
    t_stat, p_val = stats.ttest_1samp(bc_arr, 0.25)
    assert p_val > 0.05, f"Empirical ancestry mean {mean_bc:.4f} differs from expected 0.25 (p={p_val})"
    # Realized variance must be strictly positive due to meiotic segregation
    assert np.var(bc_arr) > 0.0005, "Variance in realized ancestry should be positive!"

    # 4. Brangus cross: Cross 25% Angus cow with an unrelated pure Angus sire: expected 62.5% Angus
    angus_sire2 = pop.create_founder("ANGUS_SIRE_2", sex="M", breed="Angus", rng=rng)
    # Take a 25% cow
    bc_cow = MeiosisSimulator.mate(
        sire=brahman_sire2,
        dam=f1_female,
        child_id="BC_COW_SELECTED",
        sex="F",
        genome_spec=genome,
        rng=rng
    )

    brangus_angus_fractions = []
    for i in range(n_reps):
        brangus_calf = MeiosisSimulator.mate(
            sire=angus_sire2,
            dam=bc_cow,
            child_id=f"BRANGUS_{i}",
            sex="F",
            genome_spec=genome,
            rng=rng
        )
        assert brangus_calf.pedigree_composition.get("Angus") == 0.625
        brangus_angus_fractions.append(brangus_calf.realized_genomic_ancestry("Angus", genome))

    br_arr = np.array(brangus_angus_fractions)
    mean_br = np.mean(br_arr)
    t_stat, p_val = stats.ttest_1samp(br_arr, 0.625)
    assert p_val > 0.05, f"Empirical Brangus ancestry mean {mean_br:.4f} differs from 0.625 (p={p_val})"


def test_marker_alleles_follow_segment_ancestry_at_both_ends():
    """Markers at 0 and at the chromosome end must come from the strand whose segment covers them.

    Regression test: the end marker used to be copied from the opposite strand after a crossover.
    """
    rng = np.random.default_rng(7)
    spec = ChromosomeSpec(chrom_id=1, length_m=1.5, num_markers=30, num_qtls=3)
    assert spec.marker_positions[-1] == pytest.approx(spec.length_m)
    pat = Haplotype.founder(spec, "P", "Angus", np.zeros(30, dtype=np.int8), np.zeros(3, dtype=np.int8))
    mat = Haplotype.founder(spec, "M", "Brahman", np.ones(30, dtype=np.int8), np.ones(3, dtype=np.int8))

    n_with_crossover = 0
    for _ in range(500):
        g = MeiosisSimulator.recombine_chromosome(pat, mat, spec, rng)
        n_with_crossover += len(g.segments) > 1
        expected_first = 0 if g.segments[0].breed == "Angus" else 1
        expected_last = 0 if g.segments[-1].breed == "Angus" else 1
        assert g.markers[0] == expected_first
        assert g.markers[-1] == expected_last
        # every marker agrees with the segment containing it
        for pos, allele in zip(spec.marker_positions, g.markers):
            seg = next(s for s in g.segments if s.start_m <= pos <= s.end_m)
            assert allele == (0 if seg.breed == "Angus" else 1)
    assert n_with_crossover > 300

"""Tests for Stage C: Trait Simulation (Founder differences, dominance, epistasis, phenotype noise).
"""
import numpy as np
import pytest
from scipy import stats
from src.inheritance.chromosome import GenomeSpec
from src.inheritance.meiosis import MeiosisSimulator, Population
from src.traits.architecture import TraitDefinition
from src.traits.simulator import TraitSimulator


def test_brangus_founder_breed_differences():
    """Verify explicit founder breed divergence for Brangus case."""
    rng = np.random.default_rng(42)
    genome = GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=20, qtls_per_chrom=6)

    simulator, founder_freqs = TraitSimulator.create_brangus_architecture(genome, seed=42)

    # Generate 50 Angus and 50 Brahman founders
    pop = Population(genome)
    angus_inds = [
        pop.create_founder(f"A_{i}", sex="M" if i % 2 == 0 else "F", breed="Angus", rng=rng,
                           founder_allele_frequencies=founder_freqs)
        for i in range(50)
    ]
    brahman_inds = [
        pop.create_founder(f"B_{i}", sex="M" if i % 2 == 0 else "F", breed="Brahman", rng=rng,
                           founder_allele_frequencies=founder_freqs)
        for i in range(50)
    ]

    df_angus = simulator.simulate_records(angus_inds, genome, rng)
    df_brahman = simulator.simulate_records(brahman_inds, genome, rng)

    # 1. Production: Angus must be significantly higher than Brahman
    angus_prod = df_angus["true_g_Production"].values
    brahman_prod = df_brahman["true_g_Production"].values
    t_stat, p_val = stats.ttest_ind(angus_prod, brahman_prod)
    assert np.mean(angus_prod) > np.mean(brahman_prod)
    assert p_val < 1e-5, f"Expected strong founder divergence in Production (p={p_val})"

    # 2. EnvironmentalResponse: Brahman must be significantly higher than Angus
    angus_env = df_angus["true_g_EnvironmentalResponse"].values
    brahman_env = df_brahman["true_g_EnvironmentalResponse"].values
    t_stat, p_val = stats.ttest_ind(brahman_env, angus_env)
    assert np.mean(brahman_env) > np.mean(angus_env)
    assert p_val < 1e-5, f"Expected strong founder divergence in EnvironmentalResponse (p={p_val})"


def test_wagyu_angus_founder_divergence():
    """Verify explicit founder breed divergence for Wagyu x Angus case."""
    rng = np.random.default_rng(2026)
    genome = GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=20, qtls_per_chrom=6)

    simulator, founder_freqs = TraitSimulator.create_wagyu_angus_architecture(genome, seed=123)

    pop = Population(genome)
    wagyu_inds = [
        pop.create_founder(f"W_{i}", sex="M" if i % 2 == 0 else "F", breed="Wagyu", rng=rng,
                           founder_allele_frequencies=founder_freqs)
        for i in range(50)
    ]
    angus_inds = [
        pop.create_founder(f"A_{i}", sex="M" if i % 2 == 0 else "F", breed="Angus", rng=rng,
                           founder_allele_frequencies=founder_freqs)
        for i in range(50)
    ]

    df_wagyu = simulator.simulate_records(wagyu_inds, genome, rng)
    df_angus = simulator.simulate_records(angus_inds, genome, rng)

    # Wagyu has superior Marbling
    assert np.mean(df_wagyu["true_g_Marbling"]) > np.mean(df_angus["true_g_Marbling"])
    # Angus has superior Growth
    assert np.mean(df_angus["true_g_Growth"]) > np.mean(df_wagyu["true_g_Growth"])


def test_directional_dominance_heterosis_in_f1():
    """Verify that dominance architecture produces heterosis (F1 > mid-parent average)."""
    rng = np.random.default_rng(777)
    genome = GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=20, qtls_per_chrom=6)
    simulator, founder_freqs = TraitSimulator.create_brangus_architecture(genome, seed=999)

    pop = Population(genome)
    sires = [
        pop.create_founder(f"A_sire_{i}", sex="M", breed="Angus", rng=rng, founder_allele_frequencies=founder_freqs)
        for i in range(20)
    ]
    dams = [
        pop.create_founder(f"B_dam_{i}", sex="F", breed="Brahman", rng=rng, founder_allele_frequencies=founder_freqs)
        for i in range(20)
    ]

    # Generate F1 cross offspring
    f1_inds = []
    for i in range(20):
        f1 = MeiosisSimulator.mate(
            sire=sires[i],
            dam=dams[i],
            child_id=f"F1_{i}",
            sex="F",
            genome_spec=genome,
            rng=rng
        )
        f1_inds.append(f1)

    df_parents = simulator.simulate_records(sires + dams, genome, rng)
    df_f1 = simulator.simulate_records(f1_inds, genome, rng)

    # In F1 crosses, dominance deviations should be substantially positive due to directional dominance
    mean_d_f1 = np.mean(df_f1["true_d_Production"])
    mean_d_parents = np.mean(df_parents["true_d_Production"])
    assert mean_d_f1 > mean_d_parents, "F1 should have higher positive dominance value (heterosis) than parents"


def test_phenotype_noise_and_variance():
    """Verify that realized phenotypic variance conforms to Var(G) + Var(E)."""
    rng = np.random.default_rng(555)
    genome = GenomeSpec.default_bovine(num_chromosomes=5, markers_per_chrom=15, qtls_per_chrom=5)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=555)

    pop = Population(genome)
    inds = [
        pop.create_founder(f"Ind_{i}", sex="M" if i % 2 == 0 else "F", breed="Angus", rng=rng,
                           founder_allele_frequencies=freqs)
        for i in range(150)
    ]

    df = simulator.simulate_records(inds, genome, rng)
    var_pheno = np.var(df["phenotype_Production"])
    var_g = np.var(df["true_g_Production"])
    var_e = np.var(df["phenotype_Production"] - df["true_g_Production"])

    # Var(Y) = Var(G) + Var(E) within sampling bounds
    assert pytest.approx(var_pheno, rel=0.15) == var_g + var_e


def test_residual_calibration_hits_target_heritability():
    """Stage C completion: realized architecture matches the configured h2 in the stated reference."""
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=10, qtls_per_chrom=5)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=11)

    def founders(seed, n):
        rng = np.random.default_rng(seed)
        pop = Population(genome)
        return [
            pop.create_founder(f"{b}_{i}", sex="M" if i % 2 else "F", breed=b, rng=rng, founder_allele_frequencies=freqs)
            for b in ("Angus", "Brahman") for i in range(n)
        ]

    reference = founders(1, 150)
    report = simulator.calibrate_residuals(reference, genome, reference_label="pooled founders")
    assert set(report["trait"]) == {"Production", "EnvironmentalResponse"}
    for rec in simulator.calibration.values():
        assert rec["feasible"]
        assert rec["h2_achieved"] == pytest.approx(rec["h2_target"], abs=1e-9)

    # An independent sample of the same reference population realizes h2 close to the target,
    # both from the variance components and from the simulated phenotypes.
    fresh = founders(2, 300)
    comps = simulator.variance_components(fresh, genome).set_index("trait")
    df = simulator.simulate_records(fresh, genome, np.random.default_rng(5))
    for trait in ("Production", "EnvironmentalResponse"):
        assert comps.loc[trait, "h2_realized"] == pytest.approx(comps.loc[trait, "h2_target"], abs=0.08)
        h2_pheno = np.var(df[f"true_a_{trait}"]) / np.var(df[f"phenotype_{trait}"])
        assert h2_pheno == pytest.approx(comps.loc[trait, "h2_target"], abs=0.1)


def test_within_breed_calibration():
    """Within-breed calibration gives the target h2 inside each breed, not in the pooled founders."""
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=10, qtls_per_chrom=5)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=12)
    rng = np.random.default_rng(3)
    pop = Population(genome)
    ref = {b: [pop.create_founder(f"{b}_{i}", "M" if i % 2 else "F", b, rng, freqs) for i in range(300)]
           for b in ("Angus", "Brahman")}
    simulator.calibrate_residuals(ref["Angus"] + ref["Brahman"], genome, group_by_breed=True)
    assert all(rec["within_breed"] and rec["feasible"] for rec in simulator.calibration.values())

    per_breed = [simulator.variance_components(ref[b], genome).set_index("trait") for b in ref]
    pooled = simulator.variance_components(ref["Angus"] + ref["Brahman"], genome).set_index("trait")["h2_realized"]
    for trait, target in (("Production", 0.35), ("EnvironmentalResponse", 0.40)):
        var_a = np.mean([c.loc[trait, "var_A"] for c in per_breed])
        var_g = np.mean([c.loc[trait, "var_G"] for c in per_breed])
        var_e = simulator.architecture.trait_map[trait].residual_variance
        # Exact by construction on the averaged within-breed components ...
        assert var_a / (var_g + var_e) == pytest.approx(target, abs=1e-9)
        # ... and each breed's own h2 is close to it
        assert np.mean([c.loc[trait, "h2_realized"] for c in per_breed]) == pytest.approx(target, abs=0.03)
        assert pooled[trait] > target  # between-breed differences add additive variance when pooled

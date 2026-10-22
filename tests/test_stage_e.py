"""Tests for Stage E: Decision Search (Virtual offspring sampling, joint probability, and policy comparison).
"""
import numpy as np
import pytest
from src.ancestry.pedigree import Pedigree
from src.decision.joint_probability import JointProbabilityEstimator
from src.decision.mating_allocator import BreedingPolicy, MatingAllocator
from src.decision.offspring_sampler import VirtualOffspringSampler
from src.inheritance.chromosome import GenomeSpec
from src.inheritance.meiosis import MeiosisSimulator, Population
from src.prediction.models import AdditiveGBLUP
from src.traits.simulator import TraitSimulator


def _training_setup(seed: int, n_per_breed: int = 30):
    rng = np.random.default_rng(seed)
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=20, qtls_per_chrom=4)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=seed)
    pop = Population(genome)
    inds = [
        pop.create_founder(f"{b}_{i}", sex="M" if i % 2 == 0 else "F", breed=b, rng=rng, founder_allele_frequencies=freqs)
        for b in ("Angus", "Brahman") for i in range(n_per_breed)
    ]
    df = simulator.simulate_records(inds, genome, rng)
    markers = np.vstack([ind.get_marker_genotypes(genome) for ind in inds])
    y1 = df["phenotype_Production"].values
    y2 = df["phenotype_EnvironmentalResponse"].values
    p1 = AdditiveGBLUP().fit(markers, y1)
    p2 = AdditiveGBLUP().fit(markers, y2)
    residual_cov = JointProbabilityEstimator.out_of_fold_residual_covariance(markers, y1, y2, AdditiveGBLUP)
    return genome, simulator, freqs, pop, inds, df, p1, p2, residual_cov, rng


def test_virtual_offspring_sampling_preserves_covariance():
    rng = np.random.default_rng(42)
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=20, qtls_per_chrom=4)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=42)

    pop = Population(genome)
    sire = pop.create_founder("S1", sex="M", breed="Angus", rng=rng, founder_allele_frequencies=freqs)
    dam = pop.create_founder("D1", sex="F", breed="Brahman", rng=rng, founder_allele_frequencies=freqs)

    sampler = VirtualOffspringSampler(genome)
    v_markers, v_qtls = sampler.sample_pair_gametes(sire, dam, num_samples=100, rng=rng)

    assert v_markers.shape == (100, genome.total_markers)
    assert v_qtls.shape == (100, genome.total_qtls)

    # Evaluate traits on virtual offspring
    eval_dict = simulator.architecture.evaluate(v_qtls, rng)
    prod = eval_dict["Production"]["phenotype"]
    env = eval_dict["EnvironmentalResponse"]["phenotype"]

    p_joint = JointProbabilityEstimator.estimate_from_samples(
        prod, env, target_1=100.0, target_2=50.0, method="empirical"
    )
    assert 0.0 <= p_joint <= 1.0


def test_compare_breeding_outcomes_under_equal_resources():
    """Milestone E Completion Criterion:
    Compare breeding outcomes under equal resources (dam capacity, sire quotas).
    """
    rng = np.random.default_rng(101)
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=20, qtls_per_chrom=4)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=101)

    pop = Population(genome)

    # Pool of 8 sires (4 Angus, 4 Brahman) and 30 dams (15 Angus, 15 Brahman)
    sires = [
        pop.create_founder(f"Sire_{i}", sex="M", breed="Angus" if i < 4 else "Brahman", rng=rng,
                           founder_allele_frequencies=freqs)
        for i in range(8)
    ]
    dams = [
        pop.create_founder(f"Dam_{j}", sex="F", breed="Angus" if j < 15 else "Brahman", rng=rng,
                           founder_allele_frequencies=freqs)
        for j in range(30)
    ]

    all_inds = sires + dams
    df_train = simulator.simulate_records(all_inds, genome, rng)
    marker_mat = np.vstack([ind.get_marker_genotypes(genome) for ind in all_inds])
    y_prod = df_train["phenotype_Production"].values
    y_env = df_train["phenotype_EnvironmentalResponse"].values

    # Fit predictors and the out-of-fold residual covariance on the same training data
    pred_prod = AdditiveGBLUP().fit(marker_mat, y_prod)
    pred_env = AdditiveGBLUP().fit(marker_mat, y_env)
    residual_cov = JointProbabilityEstimator.out_of_fold_residual_covariance(marker_mat, y_prod, y_env, AdditiveGBLUP)

    allocator = MatingAllocator(genome_spec=genome, max_sire_quota=6)

    target_prod = float(np.median(y_prod))
    target_env = float(np.median(y_env))
    common = dict(sires=sires, dams=dams, target_1=target_prod, target_2=target_env)

    plan_rand = allocator.build_plan(policy=BreedingPolicy.RANDOM, rng=rng, **common)
    plan_add = allocator.build_plan(
        policy=BreedingPolicy.ADDITIVE_SELECTION, predictor_trait1=pred_prod, predictor_trait2=pred_env, rng=rng, **common
    )
    plan_opt = allocator.build_plan(
        policy=BreedingPolicy.OFFSPRING_AWARE, predictor_trait1=pred_prod, predictor_trait2=pred_env,
        residual_cov=residual_cov, rng=rng, **common
    )

    # Equal resource validation: every plan uses every dam exactly once under the same sire quota
    for plan in (plan_rand, plan_add, plan_opt):
        assert len(plan.pairs) == len(dams)
        assert plan.unassigned_dam_slots == 0
        assert len({p.dam_id for p in plan.pairs}) == len(dams)
        counts = {}
        for p in plan.pairs:
            counts[p.sire_id] = counts.get(p.sire_id, 0) + 1
        assert max(counts.values()) <= 6
    assert 0.0 < plan_opt.expected_attainment < 1.0

    # Re-evaluate all policies with fresh independent simulation draws. 20 draws per pair (600 calves
    # per plan) keeps the sampling SD of attainment near 0.02, well below the policy differences.
    sire_map = {s.id: s for s in sires}
    dam_map = {d.id: d for d in dams}
    evals = {
        name: allocator.evaluate_plan_fresh_draws(
            plan, sire_map, dam_map, simulator, target_prod, target_env,
            "Production", "EnvironmentalResponse", n_draws_per_pair=20, seed=999
        )
        for name, plan in (("random", plan_rand), ("additive", plan_add), ("offspring_aware", plan_opt))
    }

    for res in evals.values():
        lo, hi = res.fresh_draw_ci95
        assert lo <= res.fresh_draw_attainment <= hi
    assert evals["offspring_aware"].fresh_draw_attainment >= evals["random"].fresh_draw_attainment, (
        f"Optimization ({evals['offspring_aware'].fresh_draw_attainment:.3f}) should perform as well or better "
        f"than Random ({evals['random'].fresh_draw_attainment:.3f})"
    )


def test_offspring_aware_requires_both_predictors_and_residuals():
    genome, simulator, freqs, pop, inds, df, p1, p2, residual_cov, rng = _training_setup(5, n_per_breed=10)
    sires = [i for i in inds if i.sex == "M"][:3]
    dams = [i for i in inds if i.sex == "F"][:5]
    allocator = MatingAllocator(genome)
    with pytest.raises(ValueError):
        allocator.build_plan(BreedingPolicy.OFFSPRING_AWARE, sires, dams, 100.0, 50.0, predictor_trait1=p1, rng=rng)
    with pytest.raises(ValueError):
        allocator.build_plan(BreedingPolicy.OFFSPRING_AWARE, sires, dams, 100.0, 50.0, predictor_trait1=p1,
                             predictor_trait2=p2, rng=rng)


def test_allocation_respects_quota_capacity_and_pedigree():
    rng = np.random.default_rng(9)
    genome = GenomeSpec.default_bovine(num_chromosomes=3, markers_per_chrom=8, qtls_per_chrom=2)
    pop = Population(genome)
    ped = Pedigree()

    def founder(name, sex):
        ind = pop.create_founder(name, sex=sex, breed="Angus", rng=rng)
        ped.add_founder(name, sex, "Angus")
        return ind

    s0, d0 = founder("S0", "M"), founder("D0", "F")
    son = MeiosisSimulator.mate(s0, d0, "S1", "M", genome, rng)
    daughter = MeiosisSimulator.mate(s0, d0, "D1", "F", genome, rng)
    for child in (son, daughter):
        ped.add_cross(child.id, "S0", "D0", sex=child.sex)
    d2, d3 = founder("D2", "F"), founder("D3", "F")

    allocator = MatingAllocator(genome, pedigree=ped, max_sire_quota=3, dam_capacity=2)
    plan = allocator.build_plan(BreedingPolicy.RANDOM, [s0, son], [daughter, d2, d3], 0.0, 0.0, rng=rng)

    # D1 is the daughter of S0 and full sister of S1: both matings give F = 0.25 > 0.125
    assert all(p.dam_id != "D1" for p in plan.pairs)
    assert plan.unassigned_dam_slots == 2
    assert len(plan.pairs) == 4
    for sire_id in ("S0", "S1"):
        assert sum(p.sire_id == sire_id for p in plan.pairs) <= 3
    for dam_id in ("D2", "D3"):
        assert sum(p.dam_id == dam_id for p in plan.pairs) == 2
    assert all(p.expected_inbreeding == 0.0 for p in plan.pairs)


def test_additive_index_is_scale_invariant():
    genome, simulator, freqs, pop, inds, df, p1, p2, residual_cov, rng = _training_setup(21, n_per_breed=15)
    sires = [i for i in inds if i.sex == "M"][:6]
    dams = [i for i in inds if i.sex == "F"][:12]

    class Scaled:
        def __init__(self, base, k):
            self.base, self.k = base, k

        def predict(self, markers):
            return self.base.predict(markers) * self.k

    def sire_usage(plan):
        usage = {}
        for p in plan.pairs:
            usage[p.sire_id] = usage.get(p.sire_id, 0) + 1
        return usage

    # A mid-parent (additive) utility only decides which sires are used and how often; every pairing
    # of those sires with the dams scores the same. So compare sire usage, not individual pairs.
    allocator = MatingAllocator(genome, max_sire_quota=3)
    plan_a = allocator.build_plan(BreedingPolicy.ADDITIVE_SELECTION, sires, dams, 0.0, 0.0,
                                  predictor_trait1=p1, predictor_trait2=p2, rng=np.random.default_rng(1))
    plan_b = allocator.build_plan(BreedingPolicy.ADDITIVE_SELECTION, sires, dams, 0.0, 0.0,
                                  predictor_trait1=p1, predictor_trait2=Scaled(p2, 1000.0), rng=np.random.default_rng(1))
    assert sire_usage(plan_a) == sire_usage(plan_b)
    assert sorted(sire_usage(plan_a).values()) == [3, 3, 3, 3]


def test_linear_fast_path_matches_pair_sampling():
    genome, simulator, freqs, pop, inds, df, p1, p2, residual_cov, rng = _training_setup(33, n_per_breed=20)
    sires = [i for i in inds if i.sex == "M"][:2]
    dams = [i for i in inds if i.sex == "F"][:3]
    t1 = float(np.median(df["phenotype_Production"]))
    t2 = float(np.median(df["phenotype_EnvironmentalResponse"]))

    class PredictOnly:
        def __init__(self, base):
            self.base = base

        def predict(self, markers):
            return self.base.predict(markers)

    allocator = MatingAllocator(genome)
    fast = allocator.pair_joint_probabilities(sires, dams, p1, p2, t1, t2, residual_cov,
                                              np.random.default_rng(0), num_virtual_offspring=400)
    slow = allocator.pair_joint_probabilities(sires, dams, PredictOnly(p1), PredictOnly(p2), t1, t2, residual_cov,
                                              np.random.default_rng(0), num_virtual_offspring=400)
    assert fast.shape == (2, 3)
    np.testing.assert_allclose(fast, slow, atol=0.06)

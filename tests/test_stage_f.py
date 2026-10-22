"""Tests for Stage F: Two-case evaluation (Case 1 Brangus & Case 2 Wagyu x Angus).
"""
import pytest
from src.evaluation.brangus_case import BrangusCaseStudy
from src.evaluation.wagyu_angus_case import POLICY_ORDER, WagyuAngusCaseStudy
from src.inheritance.chromosome import GenomeSpec
from src.inheritance.route_executor import RouteExecutor


def test_brangus_case_study_end_to_end():
    """Case 1: routes recovered, the reported route is the executed one, persistence tracked."""
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=20, qtls_per_chrom=4)
    res = BrangusCaseStudy(genome_spec=genome, seed=42).run(
        n_founder_families=3, post_formation_generations=2, generation_size=30
    )

    assert len(res.routes_found) > 0
    assert res.formation_pedigree_exact

    # Every formation calf descends from the executed route's two final-cross subtrees
    sire_side, dam_side = RouteExecutor.split_final_cross(res.route_used.root)
    for calf in res.formation_cohort:
        assert res.pedigree.individuals[calf.sire_id].breed_fraction == sire_side.composition
        assert res.pedigree.individuals[calf.dam_id].breed_fraction == dam_side.composition
        assert res.pedigree.inbreeding(calf.id) == 0.0

    assert pytest.approx(res.realized_ancestry_mean, abs=0.03) == 0.625
    assert res.realized_ancestry_std > 0.01  # segregation variance present

    df = res.generation_progression
    assert len(df) == 3  # Gen 0, 1, 2
    for col in ("production_mean", "env_response_mean", "inbreeding_F", "heterozygosity_Ho", "angus_pedigree_mean"):
        assert col in df.columns
    assert (df["angus_pedigree_mean"] == 0.625).all()  # closed 5/8 herd keeps its pedigree composition
    assert 0.55 <= df["angus_genomic_mean"].iloc[-1] <= 0.70

    assert set(res.calibration["trait"]) == {"Production", "EnvironmentalResponse"}
    assert res.calibration["feasible"].all()


def test_wagyu_angus_case_study_end_to_end():
    """Case 2: all policies under the same dam slots, computed (not hardcoded) accounting."""
    genome = GenomeSpec.default_bovine(num_chromosomes=6, markers_per_chrom=20, qtls_per_chrom=4)
    res = WagyuAngusCaseStudy(genome_spec=genome, seed=2026).run(num_dams=40, target_marbling=6.5, target_growth=360.0)

    assert list(res.policies) == POLICY_ORDER
    p = res.policies
    p_f1, p_bc3, p_comp = p["F1_Terminal"], p["Backcrossing_BC3"], p["Closed_Composite"]

    # Pedigree checks from the workplan (exact pedigree fractions)
    expected = {"F1 (Wagyu x Angus)": 0.5, "BC1": 0.75, "BC2": 0.875, "BC3": 0.9375, "F1 x F1": 0.5}
    for label, value in expected.items():
        assert res.pedigree_checks[label] == pytest.approx(value, abs=1e-12)
    assert pytest.approx(p_f1.wagyu_ancestry_mean, abs=0.05) == 0.50
    assert p_bc3.wagyu_ancestry_mean > 0.85

    # Equal resources: same dam slots, every slot filled by the unrelated-parent policies
    assert {s.dam_slots_per_generation for s in p.values()} == {40}
    assert p_f1.calves_evaluated == 80 and p_bc3.calves_evaluated == 80
    assert all(s.calves_evaluated + s.unassigned_dam_slots >= 80 - 1e-9 for s in p.values())

    # Accounting is computed from the herd, not asserted
    assert p_comp.parental_maintenance_burden == 0.0
    assert p_f1.parental_line_dams > 0
    assert p_f1.parental_maintenance_burden == pytest.approx(p_f1.parental_line_dams / (40 + p_f1.parental_line_dams))
    assert p_f1.inbreeding_F == 0.0      # unrelated purebred parents every generation
    assert p_comp.inbreeding_F > 0.0     # closed herd accumulates inbreeding
    assert p_bc3.elapsed_years > p_f1.elapsed_years

    # No candidate parent contributed its own carcass record to the predictor that selected it
    assert (res.replicate_table["training_parent_overlap"] == 0).all()

    for s in p.values():
        assert len(s.failure_cases) > 0
    assert len(res.summary_table) == len(POLICY_ORDER)

    # The narrative's "highest" policy is the one with the highest attainment
    leader = max(p.values(), key=lambda s: s.target_attainment)
    leader_line = next(line for line in res.narrative_report.splitlines() if line.endswith("highest"))
    assert leader.policy_name in leader_line

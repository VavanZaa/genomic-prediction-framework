"""Tests for Stage A: Pedigree and Ancestry Route Recovery Engine.
"""
from fractions import Fraction
import pytest
from src.ancestry.pedigree import BreedFraction, Individual, Pedigree
from src.ancestry.route_finder import RouteFinder, CrossRoute


def test_breed_fraction_exact_arithmetic():
    angus = BreedFraction.pure("Angus")
    brahman = BreedFraction.pure("Brahman")

    assert angus.get("Angus") == Fraction(1, 1)
    assert angus.get("Brahman") == Fraction(0, 1)

    # F1 cross: 50% Angus, 50% Brahman
    f1 = BreedFraction.cross(angus, brahman)
    assert f1.get("Angus") == Fraction(1, 2)
    assert f1.get("Brahman") == Fraction(1, 2)

    # Backcross F1 to Brahman: 25% Angus, 75% Brahman
    bc_brahman = BreedFraction.cross(f1, brahman)
    assert bc_brahman.get("Angus") == Fraction(1, 4)
    assert bc_brahman.get("Brahman") == Fraction(3, 4)

    # Cross 25% Angus with pure Angus: 5/8 (62.5%) Angus, 3/8 (37.5%) Brahman
    brangus_route1 = BreedFraction.cross(bc_brahman, angus)
    assert brangus_route1.get("Angus") == Fraction(5, 8)
    assert brangus_route1.get("Brahman") == Fraction(3, 8)
    assert brangus_route1.matches({"Angus": "5/8", "Brahman": "3/8"})

    # Alternate route: 3/4 Angus crossed with 1/2 Angus (F1)
    bc_angus = BreedFraction.cross(f1, angus)
    assert bc_angus.get("Angus") == Fraction(3, 4)
    brangus_route2 = BreedFraction.cross(bc_angus, f1)
    assert brangus_route2.get("Angus") == Fraction(5, 8)
    assert brangus_route2.get("Brahman") == Fraction(3, 8)


def test_pedigree_construction_and_inbreeding():
    ped = Pedigree()

    # Unrelated founders
    sire_a = ped.add_founder("ANGUS_M1", sex="M", breed="Angus")
    dam_b = ped.add_founder("BRAHMAN_F1", sex="F", breed="Brahman")

    # F1 cross
    f1 = ped.add_cross("F1_F", sire_id="ANGUS_M1", dam_id="BRAHMAN_F1", sex="F")
    assert f1.generation == 1
    assert f1.breed_fraction.get("Angus") == Fraction(1, 2)
    assert ped.inbreeding("F1_F") == 0.0

    # Mate F1 with another unrelated pure Angus bull
    angus_m2 = ped.add_founder("ANGUS_M2", sex="M", breed="Angus")
    bc_cow = ped.add_cross("BC_75_F", sire_id="ANGUS_M2", dam_id="F1_F", sex="F")
    assert bc_cow.breed_fraction.get("Angus") == Fraction(3, 4)
    assert ped.inbreeding("BC_75_F") == 0.0  # Still zero because ANGUS_M2 is unrelated

    # Intentionally test inbreeding: mate father ANGUS_M2 with daughter BC_75_F
    inbred_calf = ped.add_cross("INBRED_1", sire_id="ANGUS_M2", dam_id="BC_75_F", sex="M")
    # Inbreeding of parent-offspring cross: F = 0.25 (when parents are non-inbred)
    assert pytest.approx(ped.inbreeding("INBRED_1"), 1e-6) == 0.25


def test_route_finder_recovers_brangus_routes():
    """Milestone A Completion Criterion:
    Recover valid Brangus routes (62.5% Angus, 37.5% Brahman) without hardcoding them.
    """
    finder = RouteFinder(founder_breeds=["Angus", "Brahman"])
    target = {"Angus": Fraction(5, 8), "Brahman": Fraction(3, 8)}

    routes = finder.find_routes(target_fractions=target, max_depth=3)
    assert len(routes) > 0, "Failed to discover any Brangus crossing routes!"

    # Check for presence of classic 3-step route:
    # 1. Angus x Brahman -> 50%
    # 2. 50% x Brahman -> 25% (or 50% x Angus -> 75%)
    # 3. 25% x Angus -> 62.5% (or 75% x 50% -> 62.5%)
    found_route_1 = False
    found_route_2 = False

    for r in routes:
        steps_comp = [s.offspring_composition for s in r.steps]
        # Route 1: 50% -> 25% -> 62.5%
        has_50 = any(c.get("Angus") == Fraction(1, 2) for c in steps_comp)
        has_25 = any(c.get("Angus") == Fraction(1, 4) for c in steps_comp)
        has_75 = any(c.get("Angus") == Fraction(3, 4) for c in steps_comp)
        has_625 = any(c.get("Angus") == Fraction(5, 8) for c in steps_comp)

        if has_50 and has_25 and has_625:
            found_route_1 = True
        if has_50 and has_75 and has_625:
            found_route_2 = True

    assert found_route_1, "Did not recover the classic 25% × pure Angus -> 62.5% route"
    assert found_route_2, "Did not recover the 75% × 50% -> 62.5% route"


def test_instantiate_route_pedigree_with_unrelated_founders():
    finder = RouteFinder(founder_breeds=["Angus", "Brahman"])
    target = {"Angus": Fraction(5, 8), "Brahman": Fraction(3, 8)}
    routes = finder.find_routes(target_fractions=target, max_depth=3)

    # Instantiate concrete pedigree for first discovered route
    route = routes[0]
    ped = route.instantiate_pedigree(family_prefix="BrangusTest")

    # Verify that all terminal target calves have exact 5/8 Angus, 3/8 Brahman
    terminal_calves = [
        ind for ind in ped.individuals.values()
        if ind.breed_fraction.matches(target)
    ]
    assert len(terminal_calves) > 0
    for calf in terminal_calves:
        assert calf.breed_fraction.get("Angus") == Fraction(5, 8)
        assert calf.breed_fraction.get("Brahman") == Fraction(3, 8)
        # Because we instantiated with distinct founder lines, inbreeding must be zero
        assert ped.inbreeding(calf.id) == 0.0


def test_routes_are_orientation_free_and_ordered():
    finder = RouteFinder(founder_breeds=["Angus", "Brahman"])
    routes = finder.find_routes({"Angus": "5/8", "Brahman": "3/8"}, max_depth=4)

    signatures = [r.signature for r in routes]
    assert len(signatures) == len(set(signatures)), "Reciprocal crosses must not be listed twice"
    keys = [(r.depth, r.num_founders, r.signature) for r in routes]
    assert keys == sorted(keys)
    # The two documented 3-generation routes are both present
    assert "(((Angus x Brahman) x Brahman) x Angus)" in signatures
    assert "(((Angus x Brahman) x Angus) x (Angus x Brahman))" in signatures


def test_route_executor_realizes_target_without_inbreeding():
    import numpy as np
    from src.inheritance.chromosome import GenomeSpec
    from src.inheritance.meiosis import Population
    from src.inheritance.route_executor import RouteExecutor

    genome = GenomeSpec.default_bovine(num_chromosomes=4, markers_per_chrom=10, qtls_per_chrom=2)
    route = RouteFinder(["Angus", "Brahman"]).find_routes({"Angus": "5/8", "Brahman": "3/8"}, max_depth=3)[0]
    ped = Pedigree()
    executor = RouteExecutor(genome, Population(genome), ped)
    rng = np.random.default_rng(3)

    a = executor.realize(route.root, "F", rng, family="f1")
    b = executor.realize(route.root, "M", rng, family="f2")
    for ind in (a, b):
        assert ind.pedigree_composition.get("Angus") == Fraction(5, 8)
        assert ped.individuals[ind.id].breed_fraction.get("Angus") == Fraction(5, 8)
        assert ped.inbreeding(ind.id) == 0.0
    assert ped.coancestry(a.id, b.id) == 0.0

    sire_side, dam_side = RouteExecutor.split_final_cross(route.root)
    assert sire_side.num_founders <= dam_side.num_founders


def test_coancestry_is_half_the_relationship():
    ped = Pedigree()
    ped.add_founder("S", "M", "Angus")
    ped.add_founder("D", "F", "Angus")
    ped.add_cross("C", "S", "D", sex="F")
    assert ped.additive_relationship("S", "C") == pytest.approx(0.5)
    assert ped.coancestry("S", "C") == pytest.approx(0.25)

"""Focused tests for the repaired, developmental Layer 7 H7-v3 path."""
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paint_simulated_do_genotypes_v3 import (  # noqa: E402
    apply_observation_model,
    founder_state_switches,
    idealized_funnel_seeds,
    missing_call_mask,
)
from run_layer7_h7v3 import generation_increment  # noqa: E402
from src.inheritance.chromosome import ChromosomeSpec, GenomeSpec  # noqa: E402


def test_missing_call_mask_recognizes_cross2_missing_values():
    values = np.array(["A", "-", "NA", "H", None, np.nan], dtype=object)
    assert missing_call_mask(values).tolist() == [False, True, True, False, True, True]


def test_observation_model_reuses_the_exact_observed_missing_mask():
    codes = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    observed = np.array(["A", "-", "H", "NA", "B"])
    out, counts = apply_observation_model(codes, observed, np.random.default_rng(1), error_rate=0)
    assert (out == 3).tolist() == [False, True, False, True, False]
    assert counts == {"matched_missing": 2, "errors": 0, "founder_uninformative": 0}


def test_funnel_seed_has_all_eight_founders_at_one_eighth():
    genome = GenomeSpec([ChromosomeSpec(1, 1.0, num_markers=2, num_qtls=1)])
    seed = idealized_funnel_seeds(genome, 1, np.random.default_rng(3))[0]
    assert seed.generation == 3
    assert seed.pedigree_composition.composition == {f: Fraction(1, 8) for f in "ABCDEFGH"}
    assert founder_state_switches(seed, genome) >= 0


def test_increment_cancels_an_additive_initialization_offset():
    simulated = {8: 80.0, 11: 92.0}
    with_seed_mosaic = {8: 180.0, 11: 192.0}
    assert generation_increment(simulated) == 4.0
    assert generation_increment(with_seed_mosaic) == 4.0
    assert with_seed_mosaic[11] / with_seed_mosaic[8] != simulated[11] / simulated[8]

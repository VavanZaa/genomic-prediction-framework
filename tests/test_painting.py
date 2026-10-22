"""Painting simulated mosaics into genotype calls (Layer 7 / H7 v2).

A painting error would silently give the simulated side different information content from the
observed side, recreating the asymmetry v2 exists to remove, so the rules are pinned here.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paint_simulated_do_genotypes import CODES, combine, paint_haplotype  # noqa: E402

# 4 markers x 8 founders; allele 0 = A, 1 = B, 2 = uninformative
ALLELES = np.array([
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [2, 1, 0, 1, 0, 1, 0, 1],
])
POS = np.array([0.0, 0.1, 0.2, 0.3])


def test_single_founder_haplotype_takes_that_founders_alleles():
    got = paint_haplotype([("C", 0.0, 0.3)], POS, ALLELES)
    assert got.tolist() == ALLELES[:, 2].tolist()


def test_marker_on_a_segment_start_belongs_to_the_new_segment():
    got = paint_haplotype([("A", 0.0, 0.2), ("E", 0.2, 0.3)], POS, ALLELES)
    assert got.tolist() == [ALLELES[0, 0], ALLELES[1, 0], ALLELES[2, 4], ALLELES[3, 4]]


def test_genotypes_combine_to_A_B_and_H():
    h1, h2 = np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])
    assert CODES[combine(h1, h2)].tolist() == ["A", "B", "H", "H"]


def test_an_uninformative_founder_call_makes_the_genotype_missing():
    h1, h2 = np.array([2, 0]), np.array([0, 2])
    assert CODES[combine(h1, h2)].tolist() == ["-", "-"]


def test_inbred_homozygote_from_one_founder_is_never_heterozygous():
    h = paint_haplotype([("G", 0.0, 0.3)], POS, ALLELES)
    codes = CODES[combine(h, h)]
    assert "H" not in codes.tolist()

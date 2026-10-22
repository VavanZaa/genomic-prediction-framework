"""Deterministic toy-genome checks of gamete formation (complements the random C1 regression test).

A paternal strand of all 0 (AAAA...) and a maternal strand of all 1 (BBBB...) are recombined at
forced crossover positions. The gamete must be exactly the alternating blocks A|B|A|..., with the
strand after the last crossover running to the chromosome end and no extra terminal flip.
"""
import numpy as np
import pytest

from src.inheritance.chromosome import ChromosomeSpec, Haplotype
from src.inheritance.meiosis import MeiosisSimulator


class _ForcedCrossovers:
    """Stands in for numpy's Generator: fixed crossover positions and starting strand."""
    def __init__(self, breakpoints, start_paternal=True):
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        self.start = 0.1 if start_paternal else 0.9

    def poisson(self, lam):
        return len(self.breakpoints)

    def uniform(self, low, high, size):
        assert size == len(self.breakpoints)
        return self.breakpoints.copy()

    def random(self):
        return self.start


def _toy():
    spec = ChromosomeSpec(chrom_id=1, length_m=1.5, num_markers=30, num_qtls=3)
    pat = Haplotype.founder(spec, "P", "Angus", np.zeros(30, dtype=np.int8), np.zeros(3, dtype=np.int8))
    mat = Haplotype.founder(spec, "M", "Brahman", np.ones(30, dtype=np.int8), np.ones(3, dtype=np.int8))
    return spec, pat, mat


def _expected_blocks(positions, breakpoints, first_strand):
    """Strand index (0 paternal, 1 maternal) at each position: flips once per crossover passed."""
    passed = np.searchsorted(np.asarray(breakpoints), positions, side="right")
    return (first_strand + passed) % 2


@pytest.mark.parametrize("breakpoints", [[0.40], [0.40, 1.10], [0.30, 0.80, 1.20], [0.05, 0.50, 0.95, 1.45]])
@pytest.mark.parametrize("start_paternal", [True, False])
def test_gamete_is_exact_alternating_blocks(breakpoints, start_paternal):
    spec, pat, mat = _toy()
    gamete = MeiosisSimulator.recombine_chromosome(pat, mat, spec, _ForcedCrossovers(breakpoints, start_paternal))
    first = 0 if start_paternal else 1
    expected = _expected_blocks(spec.marker_positions, breakpoints, first)
    np.testing.assert_array_equal(gamete.markers, expected)
    # The end marker comes from the strand after the last crossover: parity of the crossover count
    assert gamete.markers[-1] == (first + len(breakpoints)) % 2
    # Segments alternate breed exactly at the forced positions
    assert len(gamete.segments) == len(breakpoints) + 1
    np.testing.assert_allclose([s.end_m for s in gamete.segments[:-1]], breakpoints)
    breeds = [s.breed for s in gamete.segments]
    assert all(a != b for a, b in zip(breeds, breeds[1:]))
    assert breeds[0] == ("Angus" if start_paternal else "Brahman")

"""Chromosome definitions, genetic maps, segments, and haplotype models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Segment:
    """A contiguous genomic segment originating from a specific founder allele."""
    start_m: float  # Start position in Morgans [0, length_m)
    end_m: float    # End position in Morgans
    founder_id: str
    breed: str

    @property
    def length(self) -> float:
        return self.end_m - self.start_m


@dataclass
class ChromosomeSpec:
    chrom_id: int
    length_m: float
    num_markers: int = 100
    num_qtls: int = 10
    marker_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    qtl_positions: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.marker_positions) == 0 and self.num_markers > 0:
            # Equidistant markers across chromosome
            self.marker_positions = np.linspace(0.0, self.length_m, self.num_markers)
        if len(self.qtl_positions) == 0 and self.num_qtls > 0:
            # Interspersed QTL positions
            self.qtl_positions = np.linspace(self.length_m / (self.num_qtls + 1),
                                             self.length_m * self.num_qtls / (self.num_qtls + 1),
                                             self.num_qtls)


@dataclass
class GenomeSpec:
    """Bovine genome model (default: 29 autosomes, total genetic length ~28-30 Morgans)."""
    chromosomes: List[ChromosomeSpec] = field(default_factory=list)

    @classmethod
    def default_bovine(cls, num_chromosomes: int = 29, markers_per_chrom: int = 50, qtls_per_chrom: int = 5) -> GenomeSpec:
        """Create representative cattle genome (Bos taurus / Bos indicus: 29 autosomes, ~1 Morgan each)."""
        # Chromosome lengths in Morgans based on bovine linkage maps (approx 0.8 - 1.4 M)
        lengths = [
            1.40, 1.25, 1.15, 1.12, 1.10, 1.08, 1.05, 1.02, 0.98, 0.96,
            0.95, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
            0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58
        ][:num_chromosomes]

        chroms = [
            ChromosomeSpec(
                chrom_id=i + 1,
                length_m=l,
                num_markers=markers_per_chrom,
                num_qtls=qtls_per_chrom
            )
            for i, l in enumerate(lengths)
        ]
        return cls(chromosomes=chroms)

    @property
    def total_length_m(self) -> float:
        return sum(c.length_m for c in self.chromosomes)

    @property
    def total_markers(self) -> int:
        return sum(c.num_markers for c in self.chromosomes)

    @property
    def total_qtls(self) -> int:
        return sum(c.num_qtls for c in self.chromosomes)


@dataclass
class Haplotype:
    """A single gametic chromosome strand containing ancestry segments and discrete marker/QTL alleles."""
    chrom_id: int
    segments: List[Segment]
    markers: np.ndarray  # Shape: (num_markers,), values in {0, 1}
    qtl_alleles: np.ndarray  # Shape: (num_qtls,), values in {0, 1}

    @classmethod
    def founder(
        cls,
        spec: ChromosomeSpec,
        founder_id: str,
        breed: str,
        marker_alleles: np.ndarray,
        qtl_alleles: np.ndarray
    ) -> Haplotype:
        seg = Segment(start_m=0.0, end_m=spec.length_m, founder_id=founder_id, breed=breed)
        return cls(
            chrom_id=spec.chrom_id,
            segments=[seg],
            markers=np.array(marker_alleles, dtype=np.int8),
            qtl_alleles=np.array(qtl_alleles, dtype=np.int8)
        )

    def breed_fraction(self, breed: str, total_len: float) -> float:
        b_len = sum(s.length for s in self.segments if s.breed == breed)
        return b_len / total_len

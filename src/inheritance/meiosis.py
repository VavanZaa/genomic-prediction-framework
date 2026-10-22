"""Meiotic recombination engine and simulated individual/population structures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..ancestry.pedigree import BreedFraction, Individual
from .chromosome import ChromosomeSpec, GenomeSpec, Haplotype, Segment


@dataclass
class SimulatedIndividual:
    id: str
    sex: str
    sire_id: Optional[str]
    dam_id: Optional[str]
    generation: int
    pedigree_composition: BreedFraction
    paternal_haplotypes: Dict[int, Haplotype]  # chrom_id -> Haplotype
    maternal_haplotypes: Dict[int, Haplotype]  # chrom_id -> Haplotype
    meta: Dict = field(default_factory=dict)

    @property
    def is_founder(self) -> bool:
        return self.sire_id is None and self.dam_id is None

    def get_marker_genotypes(self, genome_spec: GenomeSpec) -> np.ndarray:
        """Returns 1D array of marker allele counts (0, 1, 2) across all chromosomes."""
        genotypes = []
        for c in genome_spec.chromosomes:
            pat = self.paternal_haplotypes[c.chrom_id].markers
            mat = self.maternal_haplotypes[c.chrom_id].markers
            genotypes.append(pat + mat)
        return np.concatenate(genotypes)

    def get_qtl_genotypes(self, genome_spec: GenomeSpec) -> np.ndarray:
        """Returns 1D array of QTL allele counts (0, 1, 2) across all chromosomes."""
        genotypes = []
        for c in genome_spec.chromosomes:
            pat = self.paternal_haplotypes[c.chrom_id].qtl_alleles
            mat = self.maternal_haplotypes[c.chrom_id].qtl_alleles
            genotypes.append(pat + mat)
        return np.concatenate(genotypes)

    def realized_genomic_ancestry(self, breed: str, genome_spec: GenomeSpec) -> float:
        """Compute the length-weighted realized genomic fraction for a specific breed."""
        total_len = 2.0 * genome_spec.total_length_m
        breed_len = 0.0
        for c in genome_spec.chromosomes:
            pat = self.paternal_haplotypes[c.chrom_id]
            mat = self.maternal_haplotypes[c.chrom_id]
            breed_len += sum(s.length for s in pat.segments if s.breed == breed)
            breed_len += sum(s.length for s in mat.segments if s.breed == breed)
        return breed_len / total_len

    def all_realized_ancestry(self, genome_spec: GenomeSpec) -> Dict[str, float]:
        """Compute realized genomic ancestry proportions for all present breeds."""
        total_len = 2.0 * genome_spec.total_length_m
        breed_lens: Dict[str, float] = {}
        for c in genome_spec.chromosomes:
            for hap in (self.paternal_haplotypes[c.chrom_id], self.maternal_haplotypes[c.chrom_id]):
                for s in hap.segments:
                    breed_lens[s.breed] = breed_lens.get(s.breed, 0.0) + s.length
        return {b: l / total_len for b, l in breed_lens.items()}

    def heterozygosity(self, genome_spec: GenomeSpec) -> float:
        """Genomic heterozygosity: fraction of markers that are heterozygous (genotype == 1)."""
        geno = self.get_marker_genotypes(genome_spec)
        return float(np.mean(geno == 1))


class MeiosisSimulator:
    """Simulates gamete formation through crossover recombination under Haldane's Poisson model."""

    @staticmethod
    def recombine_chromosome(
        pat_hap: Haplotype,
        mat_hap: Haplotype,
        spec: ChromosomeSpec,
        rng: np.random.Generator
    ) -> Haplotype:
        length_m = spec.length_m

        # Number of crossovers on this chromosome: K ~ Poisson(L)
        k = rng.poisson(lam=length_m)

        if k == 0:
            # No crossovers: inherit either paternal or maternal intact strand
            chosen = pat_hap if (rng.random() < 0.5) else mat_hap
            return Haplotype(
                chrom_id=spec.chrom_id,
                segments=list(chosen.segments),
                markers=chosen.markers.copy(),
                qtl_alleles=chosen.qtl_alleles.copy()
            )

        # Crossover locations: uniform in (0, length_m)
        breakpoints = np.sort(rng.uniform(0.0, length_m, size=k))
        # Add boundaries: 0.0 and length_m
        bounds = np.concatenate([[0.0], breakpoints, [length_m]])

        # Initial strand: 0 for paternal, 1 for maternal
        current_strand = 0 if (rng.random() < 0.5) else 1

        new_segments: List[Segment] = []
        new_markers = np.empty_like(pat_hap.markers)
        new_qtls = np.empty_like(pat_hap.qtl_alleles)

        marker_pos = spec.marker_positions
        qtl_pos = spec.qtl_positions

        last_strand = current_strand
        for i in range(len(bounds) - 1):
            seg_start = bounds[i]
            seg_end = bounds[i + 1]
            active_hap = pat_hap if current_strand == 0 else mat_hap

            # Extract segments overlapping [seg_start, seg_end)
            for s in active_hap.segments:
                overlap_s = max(s.start_m, seg_start)
                overlap_e = min(s.end_m, seg_end)
                if overlap_e > overlap_s:
                    new_segments.append(Segment(
                        start_m=overlap_s,
                        end_m=overlap_e,
                        founder_id=s.founder_id,
                        breed=s.breed
                    ))

            # Assign markers in [seg_start, seg_end)
            m_mask = (marker_pos >= seg_start) & (marker_pos < seg_end)
            new_markers[m_mask] = active_hap.markers[m_mask]

            # Assign QTLs in [seg_start, seg_end)
            q_mask = (qtl_pos >= seg_start) & (qtl_pos < seg_end)
            new_qtls[q_mask] = active_hap.qtl_alleles[q_mask]

            # Switch strand for next interval
            last_strand = current_strand
            current_strand = 1 - current_strand

        # Markers/QTLs placed exactly at length_m belong to the final interval, i.e. to the strand
        # that was active there (not to the strand the loop flipped to afterwards).
        last_hap = pat_hap if last_strand == 0 else mat_hap
        edge_m = marker_pos >= length_m
        if np.any(edge_m):
            new_markers[edge_m] = last_hap.markers[edge_m]
        edge_q = qtl_pos >= length_m
        if np.any(edge_q):
            new_qtls[edge_q] = last_hap.qtl_alleles[edge_q]

        return Haplotype(
            chrom_id=spec.chrom_id,
            segments=new_segments,
            markers=new_markers,
            qtl_alleles=new_qtls
        )

    @classmethod
    def generate_gamete(
        cls,
        ind: SimulatedIndividual,
        genome_spec: GenomeSpec,
        rng: np.random.Generator
    ) -> Dict[int, Haplotype]:
        gamete: Dict[int, Haplotype] = {}
        for c in genome_spec.chromosomes:
            pat_h = ind.paternal_haplotypes[c.chrom_id]
            mat_h = ind.maternal_haplotypes[c.chrom_id]
            gamete[c.chrom_id] = cls.recombine_chromosome(pat_h, mat_h, c, rng)
        return gamete

    @classmethod
    def mate(
        cls,
        sire: SimulatedIndividual,
        dam: SimulatedIndividual,
        child_id: str,
        sex: str,
        genome_spec: GenomeSpec,
        rng: np.random.Generator,
        meta: Optional[Dict] = None
    ) -> SimulatedIndividual:
        if sire.sex != 'M':
            raise ValueError(f"Sire {sire.id} must be male ('M').")
        if dam.sex != 'F':
            raise ValueError(f"Dam {dam.id} must be female ('F').")

        pat_gamete = cls.generate_gamete(sire, genome_spec, rng)
        mat_gamete = cls.generate_gamete(dam, genome_spec, rng)

        child_comp = BreedFraction.cross(sire.pedigree_composition, dam.pedigree_composition)
        generation = max(sire.generation, dam.generation) + 1

        return SimulatedIndividual(
            id=child_id,
            sex=sex,
            sire_id=sire.id,
            dam_id=dam.id,
            generation=generation,
            pedigree_composition=child_comp,
            paternal_haplotypes=pat_gamete,
            maternal_haplotypes=mat_gamete,
            meta=meta or {}
        )


class Population:
    """Holds a cohort of simulated individuals and provides population-level genomic operations."""

    def __init__(self, genome_spec: GenomeSpec):
        self.genome_spec = genome_spec
        self.individuals: Dict[str, SimulatedIndividual] = {}

    def add(self, ind: SimulatedIndividual) -> None:
        self.individuals[ind.id] = ind

    def create_founder(
        self,
        id: str,
        sex: str,
        breed: str,
        rng: np.random.Generator,
        founder_allele_frequencies: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    ) -> SimulatedIndividual:
        """Create an outbred founder with breed-specific allele frequencies."""
        pat_hap: Dict[int, Haplotype] = {}
        mat_hap: Dict[int, Haplotype] = {}

        for c in self.genome_spec.chromosomes:
            if founder_allele_frequencies and breed in founder_allele_frequencies:
                m_freq, q_freq = founder_allele_frequencies[breed]
                # Slice frequencies for chromosome
                m_pat = (rng.random(c.num_markers) < m_freq[c.chrom_id]).astype(np.int8)
                m_mat = (rng.random(c.num_markers) < m_freq[c.chrom_id]).astype(np.int8)
                q_pat = (rng.random(c.num_qtls) < q_freq[c.chrom_id]).astype(np.int8)
                q_mat = (rng.random(c.num_qtls) < q_freq[c.chrom_id]).astype(np.int8)
            else:
                # Default: fixed or divergent alleles
                # e.g. for pure breeds: marker alleles sampled around intermediate frequency
                m_pat = (rng.random(c.num_markers) < 0.5).astype(np.int8)
                m_mat = (rng.random(c.num_markers) < 0.5).astype(np.int8)
                q_pat = (rng.random(c.num_qtls) < 0.5).astype(np.int8)
                q_mat = (rng.random(c.num_qtls) < 0.5).astype(np.int8)

            pat_hap[c.chrom_id] = Haplotype.founder(c, f"{id}_pat", breed, m_pat, q_pat)
            mat_hap[c.chrom_id] = Haplotype.founder(c, f"{id}_mat", breed, m_mat, q_mat)

        ind = SimulatedIndividual(
            id=id,
            sex=sex,
            sire_id=None,
            dam_id=None,
            generation=0,
            pedigree_composition=BreedFraction.pure(breed),
            paternal_haplotypes=pat_hap,
            maternal_haplotypes=mat_hap
        )
        self.individuals[id] = ind
        return ind

    def get_genotype_matrix(self, ind_ids: Optional[List[str]] = None) -> np.ndarray:
        """Returns (N, M) matrix of marker genotypes {0, 1, 2}."""
        if ind_ids is None:
            ind_ids = list(self.individuals.keys())
        return np.vstack([self.individuals[i].get_marker_genotypes(self.genome_spec) for i in ind_ids])

    def get_qtl_matrix(self, ind_ids: Optional[List[str]] = None) -> np.ndarray:
        """Returns (N, Q) matrix of QTL genotypes {0, 1, 2}."""
        if ind_ids is None:
            ind_ids = list(self.individuals.keys())
        return np.vstack([self.individuals[i].get_qtl_genotypes(self.genome_spec) for i in ind_ids])

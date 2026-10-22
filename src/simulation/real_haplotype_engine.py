"""Real Haplotype Resampling Engine.

Builds founder templates from the most Angus- and most Brahman-like animals of the Florida
multibreed herd and assembles progeny as mosaics of those templates.

What this does and does not preserve:
- Allele frequencies and genotype-level breed structure come from real animals.
- Heterozygous sites are phased at random (there is no phasing step), so haplotype-level LD
  between heterozygous sites is NOT preserved; only LD carried by homozygous stretches survives.
- A "crossover" switches to a freshly drawn template haplotype (a mosaic), not a meiosis between two
  parental haplotypes, and there is no pedigree, so realistic inbreeding is not modelled.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..real_data.florida_loader import FloridaLoader, FloridaMultibreedData


@dataclass
class SimulatedRealHaplotypePopulation:
    genotypes: np.ndarray  # (num_animals, num_snps) in {0, 1, 2}
    marker_map: pd.DataFrame
    ancestry_fractions: np.ndarray  # (num_animals, 2)
    num_animals: int
    num_snps: int


class RealHaplotypeResampler:
    """Uses physical cattle haplotype templates to generate simulated breeding populations."""

    def __init__(self, data: Optional[FloridaMultibreedData] = None, marker_step: int = 24):
        self.data = data or FloridaLoader.load_dataset(marker_step=marker_step, min_maf=0.03)

    def extract_anchor_haplotypes(
        self,
        n_angus_founders: int = 35,
        n_brahman_founders: int = 35,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract phased haploid templates from highest-purity Angus and Brahman cattle."""
        rng = np.random.default_rng(random_seed)

        df_pheno = self.data.phenotypes
        q_angus_series = df_pheno["AngusGBC"]

        # Sort animals by Angus fraction
        sorted_indices = np.argsort(q_angus_series.values)
        brahman_candidate_indices = sorted_indices[:n_brahman_founders]
        angus_candidate_indices = sorted_indices[-n_angus_founders:]

        from ..real_data.genotype_preprocessing import GenotypeImputer
        geno = self.data.genotypes.copy()
        ref=(df_pheno["Year"].to_numpy()<=2018) if "Year" in df_pheno else np.ones(len(geno),dtype=bool)
        imputer=GenotypeImputer().fit(geno[ref])
        missing=~np.isfinite(geno) | (geno<0)
        sampled=rng.binomial(2,imputer.means_/2,size=geno.shape)
        geno=np.where(missing,sampled,geno)

        def phase_individuals(indices: np.ndarray) -> np.ndarray:
            """Phase diploid genotypes into 2 * K haploid templates."""
            haps = []
            for idx in indices:
                g = geno[idx, :]
                h1 = np.zeros_like(g, dtype=np.int8)
                h2 = np.zeros_like(g, dtype=np.int8)

                # Homozygous loci
                h1[g == 0] = 0
                h2[g == 0] = 0
                h1[g == 2] = 1
                h2[g == 2] = 1

                # Heterozygous loci: phase with block continuity
                het_mask = (g == 1)
                het_indices = np.where(het_mask)[0]
                if len(het_indices) > 0:
                    # Alternating / Markovian phase assignment
                    phase_flips = rng.binomial(1, 0.5, size=len(het_indices))
                    h1[het_indices] = phase_flips
                    h2[het_indices] = 1 - phase_flips

                haps.append(h1)
                haps.append(h2)

            return np.array(haps)  # (2 * K, M)

        angus_haps = phase_individuals(angus_candidate_indices)
        brahman_haps = phase_individuals(brahman_candidate_indices)

        return angus_haps, brahman_haps

    def simulate_multibreed_population(
        self,
        num_animals: int = 3194,
        angus_fraction_mean: float = 0.71,
        angus_fraction_std: float = 0.07,
        crossover_rate_per_chrom: float = 1.0,  # ~1 Morgan per autosome
        random_seed: int = 42
    ) -> SimulatedRealHaplotypePopulation:
        """Simulate progeny by recombining physical Angus and Brahman haplotype blocks."""
        rng = np.random.default_rng(random_seed)

        angus_pool, brahman_pool = self.extract_anchor_haplotypes(random_seed=random_seed)
        n_angus_haps, n_snps = angus_pool.shape
        n_brahman_haps, _ = brahman_pool.shape

        map_df = self.data.marker_map
        chroms = map_df["chrom"].unique()

        # Individual Angus fractions matching real population
        q_angus = np.clip(rng.normal(angus_fraction_mean, angus_fraction_std, size=num_animals), 0.0, 1.0)
        q_brahman = 1.0 - q_angus
        ancestry_mat = np.column_stack([q_angus, q_brahman])

        # Precompute chromosome marker slice indices
        chr_slices = []
        for c in chroms:
            c_mask = (map_df["chrom"] == c).values
            c_indices = np.where(c_mask)[0]
            chr_slices.append((c, c_indices))

        progeny_genotypes = np.zeros((num_animals, n_snps), dtype=np.int8)

        for i in range(num_animals):
            q_a = q_angus[i]

            # Generate paternal gamete across all chromosomes
            pat_gamete = np.zeros(n_snps, dtype=np.int8)
            mat_gamete = np.zeros(n_snps, dtype=np.int8)

            for _, idxs in chr_slices:
                m_c = len(idxs)

                def sample_recombinant_gamete() -> np.ndarray:
                    # Choose starting founder haplotype based on ancestry
                    is_angus = rng.random() < q_a
                    pool = angus_pool if is_angus else brahman_pool
                    pool_size = n_angus_haps if is_angus else n_brahman_haps
                    h_curr = pool[rng.integers(0, pool_size), idxs].copy()

                    # Simulate Poisson crossovers on this chromosome
                    n_xo = rng.poisson(crossover_rate_per_chrom)
                    if n_xo > 0 and m_c > 1:
                        xo_points = np.sort(rng.integers(1, m_c, size=n_xo))
                        # Perform crossovers
                        for pt in xo_points:
                            is_angus = rng.random() < q_a
                            pool = angus_pool if is_angus else brahman_pool
                            pool_size = n_angus_haps if is_angus else n_brahman_haps
                            h_other = pool[rng.integers(0, pool_size), idxs]
                            h_curr[pt:] = h_other[pt:]
                    return h_curr

                pat_gamete[idxs] = sample_recombinant_gamete()
                mat_gamete[idxs] = sample_recombinant_gamete()

            progeny_genotypes[i, :] = pat_gamete + mat_gamete

        return SimulatedRealHaplotypePopulation(
            genotypes=progeny_genotypes,
            marker_map=map_df,
            ancestry_fractions=ancestry_mat,
            num_animals=num_animals,
            num_snps=n_snps
        )

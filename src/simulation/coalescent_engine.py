"""Bovine Coalescent Historical Burn-in Simulation Engine using msprime.

Simulates realistic bovine population genetics:
- Ancestral wild cattle population (Ne ~ 5,000 - 10,000).
- Divergence into Bos taurus (Angus) and Bos indicus (Brahman) with historical divergence.
- Post-domestication commercial breed bottlenecks (Ne ~ 100).
- Bovine recombination rate (~1 cM/Mb = 1e-8 / bp) and mutation rate (1.2e-8 / bp).
- Generates coalescent founder haplotypes with LD and site-frequency spectrum from the model.

Limits of the admixed population built by `simulate_multibreed_population`: each gamete copies a
whole chromosome from one breed's founder pool with probability q, so there are no within-chromosome
ancestry tracts; individual q is drawn from a normal distribution, not from a pedigree; and the
demographic parameters (Ne, split time) are assumed values, not fitted to cattle data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import msprime


@dataclass
class BovineDemographyConfig:
    ancestral_ne: int = 5000
    angus_ne: int = 100
    brahman_ne: int = 100
    split_generations: int = 5000
    recombination_rate: float = 1.0e-8  # ~1 cM / Mb
    mutation_rate: float = 1.2e-8


@dataclass
class CoalescentChromosome:
    chrom_id: int
    length_bp: int
    positions_bp: np.ndarray
    angus_haplotypes: np.ndarray  # (num_sites, num_angus_haps) in {0, 1}
    brahman_haplotypes: np.ndarray  # (num_sites, num_brahman_haps) in {0, 1}
    ts: msprime.TreeSequence

    @property
    def num_sites(self) -> int:
        return len(self.positions_bp)


@dataclass
class SimulatedCoalescentPopulation:
    genotypes: np.ndarray  # (num_animals, num_snps) in {0, 1, 2}
    marker_map: pd.DataFrame  # chrom, snp_id, pos_bp
    ancestry_fractions: np.ndarray  # (num_animals, 2): [Angus, Brahman]
    num_animals: int
    num_snps: int


class BovineCoalescentSimulator:
    """Simulates coalescent founder haplotypes and admixed breeding populations.

    STATUS: synthetic stress-test / null generator, NOT a realistic cattle population simulator.
    Measured against the Florida herd (after array ascertainment) it matches mean heterozygosity but not
    the joint genomic structure: LD r^2 ~0.59 at every distance (real 0.074 falling to 0.016) and PC1
    ~20 % (real ~1.8 %), because each gamete copies one whole founder chromosome and 5,000 generations at
    Ne = 100 leave the breeds almost fixed. Use it for controlled experiments with known ground truth;
    use meiosis on genuinely phased real parents (Layer 5B) for biological offspring simulation.
    """
    STATUS = "synthetic stress-test / null generator; not a realistic cattle population simulator"

    def __init__(self, config: Optional[BovineDemographyConfig] = None):
        self.config = config or BovineDemographyConfig()

    def build_demography(self) -> msprime.Demography:
        dem = msprime.Demography()
        dem.add_population(name="ancestral", initial_size=self.config.ancestral_ne)
        dem.add_population(name="Angus", initial_size=self.config.angus_ne)
        dem.add_population(name="Brahman", initial_size=self.config.brahman_ne)
        dem.add_population_split(
            time=self.config.split_generations,
            derived=["Angus", "Brahman"],
            ancestral="ancestral"
        )
        return dem

    def simulate_chromosome(
        self,
        chrom_id: int = 1,
        length_bp: int = 15_000_000,  # 15 Mb for fast accurate coalescent LD
        num_angus_haplotypes: int = 100,
        num_brahman_haplotypes: int = 100,
        random_seed: int = 42
    ) -> CoalescentChromosome:
        """Simulate a single autosome with Bos taurus and Bos indicus divergence."""
        dem = self.build_demography()

        # Sample haplotypes (ploidy 1 per sample) so the column counts are the requested haplotype
        # numbers; the coalescent time scale keeps the default diploid ploidy.
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(num_angus_haplotypes, population="Angus", ploidy=1),
                     msprime.SampleSet(num_brahman_haplotypes, population="Brahman", ploidy=1)],
            demography=dem,
            sequence_length=length_bp,
            recombination_rate=self.config.recombination_rate,
            random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=self.config.mutation_rate, random_seed=random_seed)

        # Derived-allele indicator: recurrent mutation can give allele index 2 at a site
        G_mat = (ts.genotype_matrix() > 0).astype(np.int8)  # (num_sites, total_samples)
        positions = ts.sites_position.astype(int)

        # Columns are sample nodes; take each breed's own nodes. (Slicing by column position used to
        # put Angus haplotypes in the "Brahman" pool, because each diploid sample had two nodes.)
        pop_id = {pop.metadata["name"]: pop.id for pop in ts.populations()}
        angus_haps = G_mat[:, ts.samples(population=pop_id["Angus"])]
        brahman_haps = G_mat[:, ts.samples(population=pop_id["Brahman"])]

        return CoalescentChromosome(
            chrom_id=chrom_id,
            length_bp=length_bp,
            positions_bp=positions,
            angus_haplotypes=angus_haps,
            brahman_haplotypes=brahman_haps,
            ts=ts
        )

    def simulate_multibreed_population(
        self,
        num_animals: int = 3194,
        num_markers: int = 5000,
        chrom_length_bp: int = 15_000_000,
        angus_fraction_mean: float = 0.71,
        angus_fraction_std: float = 0.07,
        num_founder_haps: int = 120,
        random_seed: int = 42,
        ascertainment=None,
        discovery: str = "admixed",
    ) -> SimulatedCoalescentPopulation:
        """Generates an admixed population using coalescent founder haplotypes.

        `ascertainment` (an ArrayAscertainment fitted on a real panel) selects array-like markers; the
        site MAF comes from the `discovery` panel: "admixed" (the population's expected frequency),
        "pooled" (both breeds equally) or "Angus". Without it, founder MAF >= 0.05 sites are thinned
        evenly, which matches no real array.
        """
        rng = np.random.default_rng(random_seed)

        # 1. Simulate coalescent founder pools for 5 chromosomes to match genome length
        num_chroms = 5
        chrom_len = chrom_length_bp
        markers_per_chrom = num_markers // num_chroms

        all_genotypes_list = []
        map_rows = []

        # Generate individual Angus fractions
        q_angus = np.clip(rng.normal(angus_fraction_mean, angus_fraction_std, size=num_animals), 0.0, 1.0)
        q_brahman = 1.0 - q_angus
        ancestry_mat = np.column_stack([q_angus, q_brahman])

        total_snp_idx = 0
        for c in range(1, num_chroms + 1):
            coalescent_chr = self.simulate_chromosome(
                chrom_id=c,
                length_bp=chrom_len,
                num_angus_haplotypes=num_founder_haps,
                num_brahman_haplotypes=num_founder_haps,
                random_seed=random_seed + c * 100
            )

            pa = coalescent_chr.angus_haplotypes.mean(axis=1)
            pb = coalescent_chr.brahman_haplotypes.mean(axis=1)
            if ascertainment is not None:
                discovery_freq = {"admixed": angus_fraction_mean * pa + (1.0 - angus_fraction_mean) * pb,
                                  "pooled": 0.5 * (pa + pb), "Angus": pa}[discovery]
                selected_sites = ascertainment.select(discovery_freq, coalescent_chr.positions_bp, markers_per_chrom)
            else:
                # Legacy selection: founder MAF >= 0.05, evenly thinned
                af = 0.5 * (pa + pb)
                maf = np.minimum(af, 1.0 - af)
                poly_sites = np.where(maf >= 0.05)[0]
                if len(poly_sites) > markers_per_chrom:
                    step = len(poly_sites) // markers_per_chrom
                    selected_sites = poly_sites[::step][:markers_per_chrom]
                elif len(poly_sites) > 0:
                    selected_sites = poly_sites
                else:
                    selected_sites = np.arange(min(markers_per_chrom, coalescent_chr.num_sites))

            sel_pos = coalescent_chr.positions_bp[selected_sites]
            angus_pool = coalescent_chr.angus_haplotypes[selected_sites, :]  # (M_c, H)
            brahman_pool = coalescent_chr.brahman_haplotypes[selected_sites, :]  # (M_c, H)
            m_c = len(selected_sites)

            # Assemble diploid genotypes for each animal via random gamete sampling proportional to q
            chr_geno = np.zeros((num_animals, m_c), dtype=np.int8)
            for i in range(num_animals):
                # Paternal gamete
                if rng.random() < q_angus[i]:
                    pat_hap = angus_pool[:, rng.integers(0, num_founder_haps)]
                else:
                    pat_hap = brahman_pool[:, rng.integers(0, num_founder_haps)]

                # Maternal gamete
                if rng.random() < q_angus[i]:
                    mat_hap = angus_pool[:, rng.integers(0, num_founder_haps)]
                else:
                    mat_hap = brahman_pool[:, rng.integers(0, num_founder_haps)]

                chr_geno[i, :] = pat_hap + mat_hap

            all_genotypes_list.append(chr_geno)

            for p in sel_pos:
                total_snp_idx += 1
                map_rows.append({
                    "chrom": c,
                    "snp_id": f"COAL_rs{total_snp_idx}",
                    "pos_bp": int(p)
                })

        genotypes = np.hstack(all_genotypes_list)
        map_df = pd.DataFrame(map_rows)

        return SimulatedCoalescentPopulation(
            genotypes=genotypes,
            marker_map=map_df,
            ancestry_fractions=ancestry_mat,
            num_animals=num_animals,
            num_snps=genotypes.shape[1]
        )

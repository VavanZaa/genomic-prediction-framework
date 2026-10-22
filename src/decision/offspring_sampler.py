"""Virtual offspring sampling for evaluating candidate mating pair genetic distributions.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import MeiosisSimulator, SimulatedIndividual


class VirtualOffspringSampler:
    """Samples hypothetical offspring gametes to characterize bivariate trait distributions before physical mating."""

    def __init__(self, genome_spec: GenomeSpec):
        self.genome_spec = genome_spec

    def sample_pair_gametes(
        self,
        sire: SimulatedIndividual,
        dam: SimulatedIndividual,
        num_samples: int = 50,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample virtual offspring marker genotypes (S, M) and QTL genotypes (S, Q)."""
        if rng is None:
            rng = np.random.default_rng()

        marker_samples = []
        qtl_samples = []

        for _ in range(num_samples):
            # Form virtual gamete from sire and dam
            sire_gamete = MeiosisSimulator.generate_gamete(sire, self.genome_spec, rng)
            dam_gamete = MeiosisSimulator.generate_gamete(dam, self.genome_spec, rng)

            # Sum alleles across chromosomes
            m_list = []
            q_list = []
            for c in self.genome_spec.chromosomes:
                cid = c.chrom_id
                m_list.append(sire_gamete[cid].markers + dam_gamete[cid].markers)
                q_list.append(sire_gamete[cid].qtl_alleles + dam_gamete[cid].qtl_alleles)

            marker_samples.append(np.concatenate(m_list))
            qtl_samples.append(np.concatenate(q_list))

        return np.vstack(marker_samples), np.vstack(qtl_samples)

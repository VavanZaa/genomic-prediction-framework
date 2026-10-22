"""Executes an ancestry route (RouteNode tree) as physical matings with chromosome inheritance.
"""
from __future__ import annotations
import itertools
from typing import Dict, Optional, Tuple
import numpy as np
from ..ancestry.pedigree import Pedigree
from ..ancestry.route_finder import RouteNode
from .chromosome import GenomeSpec
from .meiosis import MeiosisSimulator, Population, SimulatedIndividual


class RouteExecutor:
    """Realizes every cross of a route tree with freshly sampled, unrelated founders.

    Each `realize` call draws new founders for every leaf, so animals from separate calls share
    no pedigree ancestry. Every animal is registered in both the Population and the Pedigree.
    """

    def __init__(
        self,
        genome_spec: GenomeSpec,
        population: Population,
        pedigree: Pedigree,
        founder_allele_frequencies: Optional[Dict] = None,
        id_prefix: str = "R"
    ):
        self.genome_spec = genome_spec
        self.population = population
        self.pedigree = pedigree
        self.founder_allele_frequencies = founder_allele_frequencies
        self.id_prefix = id_prefix
        self._counter = itertools.count(1)

    def _next_id(self, label: str) -> str:
        return f"{self.id_prefix}_{label}_{next(self._counter)}"

    def realize(self, node: RouteNode, sex: str, rng: np.random.Generator, family: str = "") -> SimulatedIndividual:
        """Build one animal of the node's composition; parent_a is used as sire, parent_b as dam."""
        if node.is_founder:
            breed = node.founder_breed
            ind = self.population.create_founder(
                self._next_id(f"{family}{breed}_{sex}"), sex=sex, breed=breed, rng=rng,
                founder_allele_frequencies=self.founder_allele_frequencies
            )
            self.pedigree.add_founder(ind.id, sex=sex, breed=breed)
            return ind

        sire = self.realize(node.parent_a, "M", rng, family)
        dam = self.realize(node.parent_b, "F", rng, family)
        child = MeiosisSimulator.mate(
            sire, dam, self._next_id(f"{family}X{node.depth}_{sex}"), sex, self.genome_spec, rng,
            meta={"route_depth": node.depth, "family": family}
        )
        self.population.add(child)
        self.pedigree.add_cross(child.id, sire_id=sire.id, dam_id=dam.id, sex=sex)
        return child

    @staticmethod
    def split_final_cross(root: RouteNode, dam_side: str = "auto") -> Tuple[RouteNode, RouteNode]:
        """Return (sire_side, dam_side) subtrees of the route's final cross.

        'auto' puts the subtree needing fewer founders on the sire side: bulls are bought or reused
        across many cows, while each crossbred cow is bred as her own family.
        """
        if root.is_founder:
            raise ValueError("A founder node has no final cross to split.")
        a, b = root.parent_a, root.parent_b
        if dam_side == "a":
            return b, a
        if dam_side == "b":
            return a, b
        if dam_side != "auto":
            raise ValueError("dam_side must be 'auto', 'a' or 'b'")
        if (a.num_founders, a.signature()) <= (b.num_founders, b.signature()):
            return a, b
        return b, a

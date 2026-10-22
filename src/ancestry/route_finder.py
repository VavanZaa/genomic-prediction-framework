"""Route finder for discovering crossbreeding pathways to target ancestry compositions without hardcoding.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Set
from .pedigree import BreedFraction, Pedigree


@dataclass(frozen=True)
class RouteNode:
    composition: BreedFraction
    parent_a: Optional[RouteNode] = None
    parent_b: Optional[RouteNode] = None
    depth: int = 0

    @property
    def is_founder(self) -> bool:
        return self.parent_a is None and self.parent_b is None

    @property
    def founder_breed(self) -> str:
        """Breed of a pure founder node."""
        return next(iter(self.composition.composition))

    @property
    def num_founders(self) -> int:
        """Number of founder animals one realization of this subtree needs."""
        if self.is_founder:
            return 1
        return self.parent_a.num_founders + self.parent_b.num_founders

    def signature(self) -> str:
        """Orientation-free topology signature: A x B and B x A give the same string."""
        if self.is_founder:
            return self.founder_breed
        a, b = sorted((self.parent_a.signature(), self.parent_b.signature()))
        return f"({a} x {b})"

    def describe(self) -> str:
        if self.is_founder:
            return f"Pure {self.founder_breed}"
        return f"({self.parent_a.describe()} × {self.parent_b.describe()}) -> {self.composition}"


@dataclass
class CrossStep:
    step_num: int
    parent_a_desc: str
    parent_b_desc: str
    offspring_composition: Dict[str, Fraction]
    expected_fractions_float: Dict[str, float]


class CrossRoute:
    def __init__(self, root_node: RouteNode):
        self.root = root_node
        self.steps: List[CrossStep] = []
        self._linearize()

    def _linearize(self) -> None:
        nodes: List[RouteNode] = []
        visited: Set[RouteNode] = set()

        def post_order(node: RouteNode):
            if node in visited:
                return
            if not node.is_founder:
                if node.parent_a:
                    post_order(node.parent_a)
                if node.parent_b:
                    post_order(node.parent_b)
            visited.add(node)
            nodes.append(node)

        post_order(self.root)

        step_idx = 1
        for n in nodes:
            if not n.is_founder:
                self.steps.append(CrossStep(
                    step_num=step_idx,
                    parent_a_desc=str(n.parent_a.composition),
                    parent_b_desc=str(n.parent_b.composition),
                    offspring_composition=n.composition.composition,
                    expected_fractions_float=n.composition.to_dict()
                ))
                step_idx += 1

    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def signature(self) -> str:
        return self.root.signature()

    @property
    def num_founders(self) -> int:
        return self.root.num_founders

    def summary(self) -> str:
        lines = [f"Route Depth: {self.depth} generations ({self.signature})"]
        for s in self.steps:
            lines.append(f"Step {s.step_num}: {s.parent_a_desc} × {s.parent_b_desc} => {s.expected_fractions_float}")
        return "\n".join(lines)

    def instantiate_pedigree(self, family_prefix: str = "F") -> Pedigree:
        """Instantiate a concrete, valid Pedigree in which every leaf is a distinct founder.

        No animal is mated to a relative, so the route is achieved with zero inbreeding (F = 0).
        """
        ped = Pedigree()
        founder_counters: Dict[str, int] = {}

        def make_founder(breed: str, sex: str) -> str:
            c = founder_counters.get(breed, 1)
            f_id = f"{family_prefix}_{breed}_{sex}_{c}"
            founder_counters[breed] = c + 1
            ped.add_founder(f_id, sex=sex, breed=breed)
            return f_id

        def build(node: RouteNode, branch_id: int, target_sex: str) -> str:
            if node.is_founder:
                return make_founder(node.founder_breed, target_sex)

            sire_id = build(node.parent_a, branch_id * 2, 'M')
            dam_id = build(node.parent_b, branch_id * 2 + 1, 'F')
            child_id = f"{family_prefix}_Cross_D{node.depth}_B{branch_id}_{target_sex}"
            ped.add_cross(child_id, sire_id=sire_id, dam_id=dam_id, sex=target_sex)
            return child_id

        build(self.root, 1, 'F')
        return ped


class RouteFinder:
    """Discovers breeding routes to reach target breed composition using search."""

    def __init__(self, founder_breeds: List[str]):
        self.founder_breeds = founder_breeds

    def find_routes(
        self,
        target_fractions: Dict[str, Fraction | str | float],
        max_depth: int = 4,
        tolerance: float = 0.0
    ) -> List[CrossRoute]:
        """Enumerate cross topologies up to max_depth and return those matching target_fractions.

        Reciprocal crosses (A x B vs B x A) are one topology; sire/dam orientation is decided when a
        route is executed. Routes are ordered by generations, then founders needed, then signature.
        """
        target_clean = {
            b: Fraction(val) if not isinstance(val, Fraction) else val
            for b, val in target_fractions.items()
        }

        nodes_by_depth: Dict[int, List[RouteNode]] = {0: []}
        seen_signatures: Set[str] = set()

        for b in self.founder_breeds:
            node = RouteNode(composition=BreedFraction.pure(b), depth=0)
            nodes_by_depth[0].append(node)
            seen_signatures.add(node.signature())

        matched: Dict[str, CrossRoute] = {}

        for d in range(1, max_depth + 1):
            nodes_by_depth[d] = []
            # A node at depth d crosses a depth d-1 node with any node from depth <= d-1
            pool_prev = nodes_by_depth[d - 1]
            pool_all_prior = [n for depth_idx in range(d) for n in nodes_by_depth[depth_idx]]

            for n1 in pool_prev:
                for n2 in pool_all_prior:
                    child_bf = BreedFraction.cross(n1.composition, n2.composition)

                    # Avoid trivial uninformative crossings (e.g. pure A x pure A)
                    if child_bf == n1.composition or child_bf == n2.composition:
                        continue

                    new_node = RouteNode(composition=child_bf, parent_a=n1, parent_b=n2, depth=d)
                    sig = new_node.signature()
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    nodes_by_depth[d].append(new_node)

                    if child_bf.matches(target_clean, tolerance=tolerance):
                        matched[sig] = CrossRoute(new_node)

        return sorted(matched.values(), key=lambda r: (r.depth, r.num_founders, r.signature))

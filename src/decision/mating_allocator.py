"""Constrained mating allocation, policy comparison, and fresh-draw validation.

Every policy shares one allocation step: a linear assignment of sire slots (max_sire_quota per sire)
to dam slots (dam_capacity per dam) that maximizes the policy's pair utility and excludes pairs whose
offspring would exceed the pedigree inbreeding threshold. Policies differ only in the utility, so
they are compared under identical resources.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..ancestry.pedigree import BreedFraction, Pedigree
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import MeiosisSimulator, SimulatedIndividual
from ..traits.simulator import TraitSimulator
from .bivariate_normal import bivariate_normal_cdf
from .joint_probability import JointProbabilityEstimator
from .offspring_sampler import VirtualOffspringSampler

_INFEASIBLE_COST = 1e9


class BreedingPolicy(str, Enum):
    RANDOM = "Random"
    PRESCRIBED_ROUTE = "PrescribedRoute"
    ADDITIVE_SELECTION = "AdditiveSelection"
    OFFSPRING_AWARE = "OffspringAwareOptimization"


@dataclass
class MatingPair:
    sire_id: str
    dam_id: str
    predicted_score: float = 0.0
    joint_target_prob: float = 0.0
    expected_inbreeding: float = 0.0


@dataclass
class MatingPlan:
    policy: BreedingPolicy
    pairs: List[MatingPair] = field(default_factory=list)
    expected_attainment: float = 0.0
    fresh_draw_attainment: float = 0.0
    fresh_draw_ci95: Tuple[float, float] = (0.0, 0.0)
    realized_traits_mean: Dict[str, float] = field(default_factory=dict)
    average_inbreeding: float = 0.0
    unassigned_dam_slots: int = 0


def dominant_breed(bf: BreedFraction) -> str:
    """Breed with the largest pedigree fraction (alphabetical on ties)."""
    comp = bf.composition
    return max(sorted(comp), key=lambda b: comp[b])


def gamete_marker_vector(ind: SimulatedIndividual, genome_spec: GenomeSpec, rng: np.random.Generator) -> np.ndarray:
    gamete = MeiosisSimulator.generate_gamete(ind, genome_spec, rng)
    return np.concatenate([gamete[c.chrom_id].markers for c in genome_spec.chromosomes])


class MatingAllocator:
    """Allocates eligible sires to dams under capacity quotas, diversity limits, and distinct policies."""

    def __init__(
        self,
        genome_spec: GenomeSpec,
        pedigree: Optional[Pedigree] = None,
        max_sire_quota: int = 10,
        max_inbreeding_threshold: float = 0.125,
        dam_capacity: int = 1
    ):
        self.genome_spec = genome_spec
        self.pedigree = pedigree or Pedigree()
        self.max_sire_quota = max_sire_quota
        self.max_inbreeding_threshold = max_inbreeding_threshold
        self.dam_capacity = dam_capacity
        self.sampler = VirtualOffspringSampler(genome_spec)

    def expected_offspring_inbreeding(self, sire: SimulatedIndividual, dam: SimulatedIndividual) -> float:
        if sire.id in self.pedigree.individuals and dam.id in self.pedigree.individuals:
            return self.pedigree.coancestry(sire.id, dam.id)
        return 0.0

    def is_eligible_pair(self, sire: SimulatedIndividual, dam: SimulatedIndividual) -> bool:
        if sire.sex != 'M' or dam.sex != 'F':
            return False
        # Offspring inbreeding equals the parents' coancestry; excludes parent-offspring and full-sib
        # matings at the default threshold (0.125 still allows half-sibs).
        return self.expected_offspring_inbreeding(sire, dam) <= self.max_inbreeding_threshold

    def _assign(self, utility: np.ndarray, eligible: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
        """Max-utility assignment of sire slots to dam slots; ineligible pairs are never returned."""
        n_sires, n_dams = utility.shape
        cost = np.where(eligible, -utility, _INFEASIBLE_COST)
        rows = np.repeat(np.arange(n_sires), self.max_sire_quota)
        cols = np.repeat(np.arange(n_dams), self.dam_capacity)
        r_idx, c_idx = linear_sum_assignment(cost[np.ix_(rows, cols)])
        pairs = [(int(rows[r]), int(cols[c])) for r, c in zip(r_idx, c_idx) if eligible[rows[r], cols[c]]]
        pairs.sort(key=lambda sd: (sd[1], sd[0]))
        return pairs, n_dams * self.dam_capacity - len(pairs)

    def pair_joint_probabilities(
        self,
        sires: Sequence[SimulatedIndividual],
        dams: Sequence[SimulatedIndividual],
        predictor_trait1,
        predictor_trait2,
        target_1: float,
        target_2: float,
        residual_cov: np.ndarray,
        rng: np.random.Generator,
        num_virtual_offspring: int = 50,
        num_noise_draws: int = 40
    ) -> np.ndarray:
        """P(y1 >= t1 and y2 >= t2 | sire, dam) for every pair, shape (n_sires, n_dams).

        Virtual offspring genetic values come from parental gametes sampled by meiosis and scored by
        the predictors (both traits on the same gametes, so the genetic correlation is kept). Residual
        deviations follow residual_cov, the out-of-fold residual covariance of the training data
        (environment plus prediction error). Virtual offspring are evaluation draws, not physical calves.

        Linear predictors (with `linear_coefficients`): an offspring's prediction is
        c0 + sire-gamete term + dam-gamete term with independent gametes, so each parent's K sampled
        gametes give the mean and covariance of its contribution. The pair's phenotype is then taken as
        bivariate normal with mean c0 + m_sire + m_dam and covariance C_sire + C_dam + residual_cov (a sum
        over many loci), and P is the exact normal orthant probability. This avoids ranking pairs on
        Monte Carlo noise. Other predictors use per-pair offspring sampling with shared residual draws.
        """
        lin1 = getattr(predictor_trait1, "linear_coefficients", None)
        lin2 = getattr(predictor_trait2, "linear_coefficients", None)
        K = num_virtual_offspring

        if callable(lin1) and callable(lin2):
            c1, b1 = lin1()
            c2, b2 = lin2()

            def gamete_moments(parents):
                means = np.empty((len(parents), 2))
                covs = np.empty((len(parents), 2, 2))
                for i, ind in enumerate(parents):
                    gam = np.vstack([gamete_marker_vector(ind, self.genome_spec, rng) for _ in range(K)])
                    terms = np.column_stack([gam @ b1, gam @ b2])
                    means[i] = terms.mean(axis=0)
                    covs[i] = np.cov(terms, rowvar=False)
                return means, covs

            m_s, c_s = gamete_moments(sires)
            m_d, c_d = gamete_moments(dams)
            mu1 = c1 + m_s[:, None, 0] + m_d[None, :, 0]
            mu2 = c2 + m_s[:, None, 1] + m_d[None, :, 1]
            cov = c_s[:, None] + c_d[None, :] + np.asarray(residual_cov)
            sd1, sd2 = np.sqrt(cov[..., 0, 0]), np.sqrt(cov[..., 1, 1])
            return bivariate_normal_cdf((mu1 - target_1) / sd1, (mu2 - target_2) / sd2, cov[..., 0, 1] / (sd1 * sd2))

        noise = rng.multivariate_normal(np.zeros(2), residual_cov, size=num_noise_draws)
        probs = np.empty((len(sires), len(dams)))
        for s, sire in enumerate(sires):
            for d, dam in enumerate(dams):
                v_markers, _ = self.sampler.sample_pair_gametes(sire, dam, num_samples=K, rng=rng)
                probs[s, d] = JointProbabilityEstimator.estimate_with_residuals(
                    predictor_trait1.predict(v_markers), predictor_trait2.predict(v_markers), target_1, target_2, noise
                )
        return probs

    def build_plan(
        self,
        policy: BreedingPolicy,
        sires: List[SimulatedIndividual],
        dams: List[SimulatedIndividual],
        target_1: float,
        target_2: float,
        predictor_trait1=None,
        predictor_trait2=None,
        prescribed_breed_map: Optional[Dict[str, str]] = None,
        rng: Optional[np.random.Generator] = None,
        residual_cov: Optional[np.ndarray] = None,
        index_weights: Tuple[float, float] = (1.0, 1.0),
        num_virtual_offspring: int = 50,
    ) -> MatingPlan:
        """Create a constrained mating plan (dam_capacity calves per dam, sire quotas, inbreeding limit).

        - RANDOM: random utility, i.e. a uniformly random feasible allocation.
        - PRESCRIBED_ROUTE: prefer sires whose dominant breed is prescribed for the dam's dominant breed.
        - ADDITIVE_SELECTION: mid-parent index of predicted values, each trait standardized across the
          candidate pool so traits on different scales get the stated index_weights.
        - OFFSPRING_AWARE: maximize the expected number of offspring meeting both targets; needs both
          predictors and residual_cov (see pair_joint_probabilities).
        Model-based joint probabilities are reported for any policy when both predictors and
        residual_cov are supplied, so plans can be compared on the same model.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        n_s, n_d = len(sires), len(dams)
        eligible = np.array([[self.is_eligible_pair(s, d) for d in dams] for s in sires], dtype=bool).reshape(n_s, n_d)
        have_model = predictor_trait1 is not None and predictor_trait2 is not None and residual_cov is not None
        probs: Optional[np.ndarray] = None

        if policy == BreedingPolicy.RANDOM:
            utility = rng.random((n_s, n_d))

        elif policy == BreedingPolicy.PRESCRIBED_ROUTE:
            prescribed = prescribed_breed_map or {}
            match = np.array([
                [1.0 if dominant_breed(s.pedigree_composition) == prescribed.get(dominant_breed(d.pedigree_composition)) else 0.0
                 for d in dams]
                for s in sires
            ]).reshape(n_s, n_d)
            utility = match + 1e-3 * rng.random((n_s, n_d))

        elif policy == BreedingPolicy.ADDITIVE_SELECTION:
            predictors = [p for p in (predictor_trait1, predictor_trait2)]
            if all(p is None for p in predictors):
                raise ValueError("ADDITIVE_SELECTION needs at least one trait predictor.")
            markers = np.vstack([ind.get_marker_genotypes(self.genome_spec) for ind in list(sires) + list(dams)])
            index = np.zeros(n_s + n_d)
            for weight, pred in zip(index_weights, predictors):
                if pred is None:
                    continue
                ebv = pred.predict(markers)
                sd = float(np.std(ebv))
                index += weight * ((ebv - np.mean(ebv)) / sd if sd > 0 else 0.0)
            utility = index[:n_s, None] + index[None, n_s:]

        elif policy == BreedingPolicy.OFFSPRING_AWARE:
            if not have_model:
                raise ValueError(
                    "OFFSPRING_AWARE needs predictor_trait1, predictor_trait2 and residual_cov; a missing "
                    "predictor cannot be replaced by zeros without making every joint probability zero."
                )
            probs = self.pair_joint_probabilities(
                sires, dams, predictor_trait1, predictor_trait2, target_1, target_2, residual_cov, rng,
                num_virtual_offspring=num_virtual_offspring
            )
            utility = probs

        else:
            raise ValueError(f"Unknown policy: {policy}")

        if probs is None and have_model:
            probs = self.pair_joint_probabilities(
                sires, dams, predictor_trait1, predictor_trait2, target_1, target_2, residual_cov, rng,
                num_virtual_offspring=num_virtual_offspring
            )

        assignments, unassigned = self._assign(utility, eligible)
        plan_pairs = []
        for s_idx, d_idx in assignments:
            sire, dam = sires[s_idx], dams[d_idx]
            plan_pairs.append(MatingPair(
                sire_id=sire.id,
                dam_id=dam.id,
                predicted_score=float(utility[s_idx, d_idx]),
                joint_target_prob=float(probs[s_idx, d_idx]) if probs is not None else 0.0,
                expected_inbreeding=self.expected_offspring_inbreeding(sire, dam),
            ))

        return MatingPlan(
            policy=policy,
            pairs=plan_pairs,
            expected_attainment=float(np.mean([p.joint_target_prob for p in plan_pairs])) if (plan_pairs and probs is not None) else 0.0,
            average_inbreeding=float(np.mean([p.expected_inbreeding for p in plan_pairs])) if plan_pairs else 0.0,
            unassigned_dam_slots=unassigned,
        )

    def evaluate_plan_fresh_draws(
        self,
        plan: MatingPlan,
        all_sires: Dict[str, SimulatedIndividual],
        all_dams: Dict[str, SimulatedIndividual],
        simulator: TraitSimulator,
        target_1: float,
        target_2: float,
        t1_name: str,
        t2_name: str,
        n_draws_per_pair: int = 1,
        seed: int = 888
    ) -> MatingPlan:
        """Re-evaluate a plan with fresh meiosis and phenotype draws from the hidden simulator.

        Fresh draws avoid selecting on lucky simulation noise. The 95% interval is a normal
        approximation over calves and ignores the correlation among calves of the same sire.
        """
        fresh_rng = np.random.default_rng(seed)
        offspring_inds = []

        for pair in plan.pairs:
            sire = all_sires[pair.sire_id]
            dam = all_dams[pair.dam_id]
            for rep in range(n_draws_per_pair):
                calf = MeiosisSimulator.mate(
                    sire=sire,
                    dam=dam,
                    child_id=f"Eval_{pair.sire_id}x{pair.dam_id}_{rep}",
                    sex="M" if fresh_rng.random() < 0.5 else "F",
                    genome_spec=self.genome_spec,
                    rng=fresh_rng
                )
                offspring_inds.append(calf)

        df_eval = simulator.simulate_records(offspring_inds, self.genome_spec, fresh_rng)
        if len(df_eval) == 0:
            return plan

        succ = (df_eval[f"phenotype_{t1_name}"] >= target_1) & (df_eval[f"phenotype_{t2_name}"] >= target_2)
        realized_attain = float(np.mean(succ))
        half_width = 1.96 * np.sqrt(realized_attain * (1.0 - realized_attain) / len(succ))

        plan.fresh_draw_attainment = realized_attain
        plan.fresh_draw_ci95 = (max(0.0, realized_attain - half_width), min(1.0, realized_attain + half_width))
        plan.realized_traits_mean = {
            t1_name: float(np.mean(df_eval[f"phenotype_{t1_name}"])),
            t2_name: float(np.mean(df_eval[f"phenotype_{t2_name}"]))
        }
        return plan

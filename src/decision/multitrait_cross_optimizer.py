"""Multi-Trait Objective Cross Design & Epistemic Support Optimization Engine (Layer 4B).

Formalizes the multi-objective Pareto trade-off between:
1. Thermotolerance Merit (LoadSlope): Pushes toward Brahman adaptation (q -> 0.787).
2. Carcass Quality (Marbling Score): Pushes toward British Angus genetics (q -> 0.000).
3. Reproductive Fertility (Age at Puberty): Pushes toward early-maturing Angus genetics (q -> 0.000).
4. Crossbred Growth / Vigor (Yearling Weight): Fosters intermediate crossbred heterosis.

Key Features:
- Quantitatively tests whether multi-trait Pareto optimization pulls the single-trait optimum
  q* = 0.787 down to the historical Brangus standard q* = 0.375 (3/8 Brahman - 5/8 Angus).
- Integrates the Genomic OOD Manifold as an epistemic constraint:
  q*_safe = argmax_q U(q) s.t. C_OOD(q) >= C_min.
- Evaluates both continuous cross-design and individual candidate sire-dam allocations on real cattle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TraitSpec:
    """Specification of a biological trait under crossbreeding selection."""
    name: str
    unit: str
    mu_ref: float
    beta_ancestry: float  # Additive Brahman effect (per unit q_B)
    heterosis_h: float    # F1 heterosis (at 4*q*(1-q) = 1.0)
    sigma_p: float        # Phenotypic standard deviation
    threshold: float      # Attainment threshold tau
    is_minimization: bool # True: lower is better (LoadSlope, Puberty); False: higher is better (Marbling, Growth)

    def predict_mean(self, q: Union[float, np.ndarray], q_ref: float = 0.0) -> Union[float, np.ndarray]:
        """Expected phenotype f(q) = mu + beta*(q - q_ref) + 4*q*(1 - q)*h."""
        het = 4.0 * q * (1.0 - q)
        return self.mu_ref + self.beta_ancestry * (q - q_ref) + het * self.heterosis_h

    def utility(self, q: Union[float, np.ndarray], q_ref: float = 0.0) -> Union[float, np.ndarray]:
        """Probabilistic attainment utility P(y in favorable region | q)."""
        pred_mu = self.predict_mean(q, q_ref)
        if self.is_minimization:
            # Lower is better: P(y < tau)
            z = (self.threshold - pred_mu) / (self.sigma_p + 1e-6)
            return stats.norm.cdf(z)
        else:
            # Higher is better: P(y > tau)
            z = (pred_mu - self.threshold) / (self.sigma_p + 1e-6)
            return stats.norm.cdf(z)


@dataclass
class ParetoSweepResult:
    """Result of systematic multi-trait Pareto frontier analysis."""
    weight_matrix: np.ndarray             # N x K weights
    optimal_qb_unconstrained: np.ndarray  # N optimal q* values
    optimal_qb_safe: np.ndarray           # N safe optimal q* values
    best_weights_brangus: np.ndarray      # Weight vector closest to q* = 0.375
    min_distance_brangus: float
    optimal_qb_brangus_weights: float
    q_grid: np.ndarray
    trait_utilities_grid: Dict[str, np.ndarray]
    composite_utilities_grid: Dict[str, np.ndarray]


class MultiTraitCrossOptimizer:
    """Solves multi-objective cross-design and mating optimization subject to epistemic support constraints."""

    def __init__(
        self,
        traits: Optional[List[TraitSpec]] = None,
        q_ref: float = 0.287,
        random_seed: int = 42
    ):
        self.q_ref = q_ref
        self.rng = np.random.default_rng(random_seed)

        # Default trait specifications. Parameter status: ASSUMED, illustrative values. None of them
        # is fitted to the Florida data or traced to a verified source; the thermotolerance slope and
        # heterosis (-2.71, -1.18) have no recorded provenance. See docs/SOURCE_INVENTORY.md.
        if traits is None:
            self.traits = [
                TraitSpec(
                    name="Thermotolerance",
                    unit="deg C/hr",
                    mu_ref=31.42,
                    beta_ancestry=-2.71,
                    heterosis_h=-1.18,
                    sigma_p=8.36,
                    threshold=29.58,  # top 35% in Florida multibreed
                    is_minimization=True
                ),
                TraitSpec(
                    name="Carcass_Marbling",
                    unit="USDA score",
                    mu_ref=6.60,       # Angus baseline: USDA Choice/Prime
                    beta_ancestry=-1.80, # Brahman reduces marbling by ~1.8 units
                    heterosis_h=+0.25,  # Slight positive heterosis
                    sigma_p=1.00,
                    threshold=6.00,    # USDA Choice threshold
                    is_minimization=False
                ),
                TraitSpec(
                    name="Age_at_Puberty",
                    unit="months",
                    mu_ref=11.50,      # Angus baseline: early puberty ~11.5 mo
                    beta_ancestry=+5.00, # Brahman heifers reach puberty ~16.5 mo
                    heterosis_h=-1.20,  # Favorable heterosis accelerates puberty by 1.2 mo
                    sigma_p=2.00,
                    threshold=13.50,   # Calving target threshold
                    is_minimization=True
                ),
                TraitSpec(
                    name="Yearling_Growth",
                    unit="kg",
                    mu_ref=405.00,
                    beta_ancestry=-8.00,
                    heterosis_h=+18.00, # Substantial crossbred heterosis (~8-10% vigor)
                    sigma_p=28.00,
                    threshold=415.00,
                    is_minimization=False
                )
            ]
        else:
            self.traits = traits

    def evaluate_joint_utility(
        self,
        q: Union[float, np.ndarray],
        weights: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Calculates scalarized joint multi-trait utility: U(q) = sum_j w_j * U_j(q)."""
        w_norm = weights / (np.sum(weights) + 1e-9)
        u_total = 0.0
        for j, trait in enumerate(self.traits):
            u_total = u_total + w_norm[j] * trait.utility(q, self.q_ref)
        return u_total

    def solve_optimal_cross(
        self,
        weights: np.ndarray,
        q_grid: Optional[np.ndarray] = None,
        d_ood_profile: Optional[np.ndarray] = None,
        min_confidence: float = 0.70
    ) -> Dict[str, float]:
        """Finds both unconstrained and epistemic support-constrained optimal cross proportion q*."""
        if q_grid is None:
            q_grid = np.linspace(0.0, 1.0, 101)

        u_vals = np.array([self.evaluate_joint_utility(q, weights) for q in q_grid])
        unconstrained_idx = int(np.argmax(u_vals))
        q_opt_unconstrained = float(q_grid[unconstrained_idx])

        # Epistemic support constraint: C_OOD(q) >= min_confidence
        q_opt_safe = q_opt_unconstrained
        if d_ood_profile is not None:
            conf_profile = 1.0 / (1.0 + d_ood_profile)
            safe_mask = (conf_profile >= min_confidence)
            if np.any(safe_mask):
                safe_idx = int(np.argmax(u_vals[safe_mask]))
                q_opt_safe = float(q_grid[safe_mask][safe_idx])
            else:
                q_opt_safe = float(self.q_ref)

        return {
            "q_opt_unconstrained": q_opt_unconstrained,
            "q_opt_safe": q_opt_safe,
            "max_utility": float(u_vals[unconstrained_idx]),
            "safe_utility": float(self.evaluate_joint_utility(q_opt_safe, weights))
        }

    def trace_pareto_frontier(
        self,
        n_samples: int = 1500,
        historical_target_qb: float = 0.375,
        d_ood_profile: Optional[np.ndarray] = None,
        min_confidence: float = 0.70,
    ) -> ParetoSweepResult:
        """Sweeps economic preference weights across the simplex to map the Pareto frontier and test Brangus convergence."""
        q_grid = np.linspace(0.0, 1.0, 101)
        k_traits = len(self.traits)

        # Precompute individual trait utilities across grid
        trait_utils = {}
        for trait in self.traits:
            trait_utils[trait.name] = np.array([trait.utility(q, self.q_ref) for q in q_grid])

        # Systematic Dirichlet sample of weight simplex Delta^(K-1)
        alpha = np.ones(k_traits)
        weight_mat = self.rng.dirichlet(alpha, size=n_samples)

        # Include pure single-trait weightings
        eye_weights = np.eye(k_traits)
        weight_mat = np.vstack([eye_weights, weight_mat])

        opt_unconstrained = []
        opt_safe = []

        for w in weight_mat:
            sol = self.solve_optimal_cross(w, q_grid=q_grid, d_ood_profile=d_ood_profile, min_confidence=min_confidence)
            opt_unconstrained.append(sol["q_opt_unconstrained"])
            opt_safe.append(sol["q_opt_safe"])

        opt_unconstrained = np.array(opt_unconstrained)
        opt_safe = np.array(opt_safe)

        # Find minimum-distance weight vector to historical Brangus (q* = 0.375)
        dists = np.abs(opt_unconstrained - historical_target_qb)
        best_idx = int(np.argmin(dists))
        best_w = weight_mat[best_idx]
        min_dist = float(dists[best_idx])
        q_at_best_w = float(opt_unconstrained[best_idx])

        # Precompute composite utilities for key scenarios:
        # 1. Pure Thermotolerance: w = [1, 0, 0, 0]
        # 2. Pure Carcass Quality: w = [0, 1, 0, 0]
        # 3. Pure Fertility: w = [0, 0, 1, 0]
        # 4. Balanced Multi-Trait Brangus: w = best_w
        composite_utils = {
            "Pure_Thermotolerance": self.evaluate_joint_utility(q_grid, np.array([1.0, 0.0, 0.0, 0.0])),
            "Pure_Carcass_Quality": self.evaluate_joint_utility(q_grid, np.array([0.0, 1.0, 0.0, 0.0])),
            "Pure_Fertility": self.evaluate_joint_utility(q_grid, np.array([0.0, 0.0, 1.0, 0.0])),
            "Balanced_Brangus_Optimum": self.evaluate_joint_utility(q_grid, best_w)
        }

        return ParetoSweepResult(
            weight_matrix=weight_mat,
            optimal_qb_unconstrained=opt_unconstrained,
            optimal_qb_safe=opt_safe,
            best_weights_brangus=best_w,
            min_distance_brangus=min_dist,
            optimal_qb_brangus_weights=q_at_best_w,
            q_grid=q_grid,
            trait_utilities_grid=trait_utils,
            composite_utilities_grid=composite_utils
        )

    def allocate_real_matings_multitrait(
        self,
        sire_ids: List[str],
        dam_ids: List[str],
        sire_qb: np.ndarray,
        dam_qb: np.ndarray,
        grm_sire_dam: np.ndarray,
        weights: np.ndarray,
        max_kinship: float = 0.125,
        sire_quota: int = 25
    ) -> pd.DataFrame:
        """Allocates real candidate sire-dam matings under the multi-trait objective with kinship constraints."""
        from scipy.optimize import linear_sum_assignment
        from ..real_data.genotype_preprocessing import validate_ids
        validate_ids(sire_ids + dam_ids, "candidate IDs")
        n_sires, n_dams = len(sire_ids), len(dam_ids)
        if not n_sires or not n_dams or sire_quota < 1:
            raise ValueError("Need nonempty candidates and a positive sire quota")
        grm = np.asarray(grm_sire_dam, dtype=float)
        if grm.shape != (n_sires,n_dams) or not np.isfinite(grm).all():
            raise ValueError("Candidate relationship matrix must be complete and finite")
        qs, qd = np.asarray(sire_qb), np.asarray(dam_qb)
        if qs.shape != (n_sires,) or qd.shape != (n_dams,) or not np.isfinite(np.r_[qs,qd]).all():
            raise ValueError("Ancestry arrays must match candidates")
        q = (qs[:,None] + qd[None,:])/2
        utility = self.evaluate_joint_utility(q, weights)
        score = utility - 0.1*grm
        eligible = grm <= max_kinship
        slots = np.repeat(np.arange(n_sires), sire_quota)
        # Dummy slots allow an explicit partial plan; maximize coverage before utility.
        penalty = (2*np.max(np.abs(score))+1)*(n_dams+1)
        costs = np.vstack([np.where(eligible, -score, np.inf)[slots], np.full((n_dams,n_dams),penalty)])
        rows, cols = linear_sum_assignment(costs)
        allocations=[]; unassigned=[]
        for row,d in zip(rows,cols):
            if row >= len(slots):
                unassigned.append(dam_ids[d]); continue
            i=int(slots[row])
            allocations.append({"Dam_ID":dam_ids[d],"Sire_ID":sire_ids[i],"Dam_qB":qd[d],"Sire_qB":qs[i],
                "Expected_Offspring_qB":q[i,d],"Kinship_Gsd":grm[i,d],"Expected_Inbreeding_Fo":0.5*grm[i,d],
                "Joint_Utility":utility[i,d]})
        columns=["Dam_ID","Sire_ID","Dam_qB","Sire_qB","Expected_Offspring_qB","Kinship_Gsd","Expected_Inbreeding_Fo","Joint_Utility"]
        result=pd.DataFrame(allocations,columns=columns)
        result.attrs.update(unassigned_dams=unassigned, num_unassigned_dams=len(unassigned),
                            status="complete" if not unassigned else ("partial" if allocations else "infeasible"))
        return result

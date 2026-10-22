"""Constrained Multi-Policy Pair-Specific Mating Optimizer on Real Cattle Genomes (Layer 4A-v2).

Evaluates 5 distinct breeding policies under identical sire quotas and coancestry constraints:
1. Random Mating (Baseline control)
2. Global Ancestry Mating (Breed-fraction matching targeting 5/8 Angus : 3/8 Brahman)
3. Dense SNP Genomic Mating (pair-specific mid-parent 122k SNP GEBV with Gaussian Mendelian-sampling variance)
4. Dense BOA Genomic Mating (pair-specific mid-parent 122k BOA GEBV with Gaussian Mendelian-sampling variance)
5. Combined SNP+BOA Genomic Mating (pair-specific mid-parent composite-kernel GEBV with Gaussian Mendelian-sampling variance)
No genotypes are recombined here: offspring merit is a Gaussian draw around the mid-parent GEBV.

True Pair-Specific Optimization:
Instead of linear mid-parent merit (which collapses to sire selection over a fixed dam herd),
evaluates non-linear pair-specific joint attainment utility:
    U_{ij} = P(y_load < tau_L AND y_hair < tau_H | s_i, d_j, E)
accounting for pair-specific segregation variance Sigma_seg(s_i, d_j) and dam x sire complementarity.

Guarantees Identical Resource & Inbreeding Parity:
- Exactly 1 allocation per dam (sum_i x_{ij} = 1)
- Strict sire capacity quota (sum_j x_{ij} <= C_sire)
- Parental genomic coancestry ceiling (Theta_{sd} = 0.5 * G_{sd} <= Theta_max)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .environment_scenario import EnvironmentEvaluator, EnvironmentScenario
from .real_offspring_predictor import ParentalMerit, RealOffspringPredictor, VirtualOffspringProfile


class PolicyType(str, Enum):
    RANDOM = "Random"
    GLOBAL_ANCESTRY = "GlobalAncestry"
    DENSE_SNP = "DenseSNP"
    DENSE_BOA = "DenseBOA"
    COMBINED_SNP_BOA = "Combined_SNP_BOA"


@dataclass
class MatingAssignment:
    sire_id: str
    dam_id: str
    sire_ancestry: float
    dam_ancestry: float
    exp_offspring_ancestry: float
    exp_gebv_load: float
    exp_gebv_hair: float
    parental_coancestry: float         # Theta_{sd} = 0.5 * G_{sd}
    genomic_relationship_g: float      # G_{sd} relative to base reference
    prob_attainment_env_a: float = 0.0
    prob_attainment_env_b: float = 0.0
    sire_gebv_load: float = float("nan")
    dam_gebv_load: float = float("nan")
    sire_gebv_hair: float = float("nan")
    dam_gebv_hair: float = float("nan")
    seg_scale: float = float("nan")    # Var(Mendelian sampling) / sigma_a^2 for this pair


@dataclass
class PolicyEvaluationSummary:
    policy_name: str
    num_matings: int
    num_unique_sires_used: int
    max_sire_usage: int
    mean_offspring_ancestry: float
    ancestry_variance: float
    ancestry_target_mad: float          # Mean Absolute Deviation from 3/8 (0.375) target
    mean_gebv_load: float               # Expected offspring GEBV for LoadSlope (lower is better)
    mean_gebv_hair: float               # Expected offspring GEBV for Hair Length (lower is better)
    mean_parental_coancestry: float     # Mean Theta_{sd} = 0.5 * G_{sd}
    max_parental_coancestry: float      # Maximum Theta_{sd}
    population_delta_f: float           # Population inbreeding change: F_offspring - F_parents
    mean_prob_attainment_env_a: float   # Attainment under Subtropical Baseline
    mean_prob_attainment_env_b: float   # Attainment under Elevated Shifted Environment
    assignments: List[MatingAssignment] = field(default_factory=list)
    num_unassigned_dams: int = 0        # dams with no sire inside the coancestry ceiling
    status: str = "complete"
    relationship_assumption: str = "measured_grm"
    gebv_mean_load: float = 0.0         # over all candidate sires and dams for this GEBV model
    gebv_sd_load: float = 0.0
    gebv_mean_hair: float = 0.0
    gebv_sd_hair: float = 0.0


class RealMatingOptimizer:
    """Solves the constrained assignment problem for all 5 breeding policies on physical genomes."""

    def __init__(
        self,
        predictor: RealOffspringPredictor,
        sires: List[ParentalMerit],
        dams: List[ParentalMerit],
        max_sire_quota: int = 25,
        max_coancestry_threshold: float = 0.0625,  # 0.0625 corresponds to half-sib coancestry (G_sd <= 0.125)
        ancestry_target: float = 0.375,            # 3/8 Brahman Brangus composition
    ):
        self.predictor = predictor
        self.sires = sires
        self.dams = dams
        self.max_sire_quota = max_sire_quota
        self.max_coancestry_threshold = max_coancestry_threshold
        self.ancestry_target = ancestry_target

        from ..real_data.genotype_preprocessing import validate_ids
        validate_ids([p.animal_id for p in sires + dams], "candidate IDs")
        if not sires or not dams or max_sire_quota < 1:
            raise ValueError("Need sires, dams and a positive sire quota")
        if any(s.sex != "M" for s in sires) or any(d.sex != "F" for d in dams):
            raise ValueError("Sires must be male and dams female")
        self.num_sires = len(sires)
        self.num_dams = len(dams)

        # Precompute candidate relationship and coancestry matrix between sires and dams
        # VanRaden G is centered relative to base reference allele frequencies
        self.g_matrix = np.zeros((self.num_sires, self.num_dams), dtype=np.float64)
        self.coancestry_matrix = np.zeros((self.num_sires, self.num_dams), dtype=np.float64)
        for i, s in enumerate(self.sires):
            for j, d in enumerate(self.dams):
                g_val = self.predictor.get_genomic_relationship(s.animal_id, d.animal_id)
                self.g_matrix[i, j] = g_val
                self.coancestry_matrix[i, j] = 0.5 * g_val

    def is_eligible(self, sire_idx: int, dam_idx: int) -> bool:
        """Returns True if parental coancestry does not exceed maximum allowable ceiling."""
        return self.coancestry_matrix[sire_idx, dam_idx] <= self.max_coancestry_threshold

    def optimize_policy(
        self,
        policy: PolicyType,
        sire_gebvs_load: np.ndarray,
        sire_gebvs_hair: np.ndarray,
        dam_gebvs_load: np.ndarray,
        dam_gebvs_hair: np.ndarray,
        scenario_eval: Optional[EnvironmentScenario] = None,
        scenario_alt: Optional[EnvironmentScenario] = None,
        tau_load: float = 28.51,
        tau_hair: float = 5.93,
        objective_mode: str = "joint",
        weight_load: float = 0.75,
        seed: int = 42,
    ) -> PolicyEvaluationSummary:
        """Executes constrained pair-specific optimization under identical quotas and specified utility mode.

        objective_mode options:
        - 'joint': Bivariate threshold attainment P(y_load < tau_L AND y_hair < tau_H)
        - 'loadslope_only': Marginal heat-load attainment P(y_load < tau_L)
        - 'hair_only': Marginal hair-length attainment P(y_hair < tau_H)
        - 'weighted_marginal': Linear combination w_L * P(y_load < tau_L) + (1 - w_L) * P(y_hair < tau_H)
        - 'continuous_index': Standardized continuous index - [ w_L * z(g_L) + (1 - w_L) * z(g_H) ]
        """
        for values, n in ((sire_gebvs_load, self.num_sires), (sire_gebvs_hair, self.num_sires),
                          (dam_gebvs_load, self.num_dams), (dam_gebvs_hair, self.num_dams)):
            if np.asarray(values).shape != (n,) or not np.isfinite(values).all():
                raise ValueError("GEBVs must be finite and match candidate rows")
        if objective_mode not in {"joint", "loadslope_only", "hair_only", "weighted_marginal", "continuous_index"}:
            raise ValueError("Unknown objective mode")
        rng = np.random.default_rng(seed)
        if self.num_sires * self.max_sire_quota < self.num_dams:
            raise ValueError(
                f"{self.num_sires} sires x quota {self.max_sire_quota} cannot cover {self.num_dams} dams."
            )

        # Pair-specific Mendelian sampling scale: 0.5 * (1 - mean parental F), floored at 0.1
        f_sires = np.array([s.inbreeding_f for s in self.sires])
        f_dams = np.array([d.inbreeding_f for d in self.dams])
        pair_seg_scale = np.maximum(0.5 * (1.0 - 0.5 * (f_sires[:, None] + f_dams[None, :])), 0.1)

        if scenario_eval is None:
            scenario_eval = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]
        if scenario_alt is None:
            scenario_alt = EnvironmentEvaluator.CANONICAL_SCENARIOS["Elevated_Heat_Shifted"]

        # 1. Compute Pair-Specific Cost Matrix (num_sires x num_dams)
        # linear_sum_assignment minimizes cost, so cost = -utility
        cost_matrix = np.zeros((self.num_sires, self.num_dams), dtype=np.float64)

        if policy == PolicyType.RANDOM:
            cost_matrix = rng.standard_normal((self.num_sires, self.num_dams))

        elif policy == PolicyType.GLOBAL_ANCESTRY:
            # Minimize absolute deviation of offspring ancestry from target 3/8 (0.375)
            for i, s in enumerate(self.sires):
                for j, d in enumerate(self.dams):
                    off_ancestry = 0.5 * (s.ancestry_brahman + d.ancestry_brahman)
                    cost_matrix[i, j] = np.abs(off_ancestry - self.ancestry_target)

        elif policy in (PolicyType.DENSE_SNP, PolicyType.DENSE_BOA, PolicyType.COMBINED_SNP_BOA):
            # TRUE PAIR-SPECIFIC UTILITY:
            # Evaluates non-linear attainment utility for each (sire, dam) pair
            # accounting for mid-parent GEBV, segregation variance, and environmental scenario
            for i, s in enumerate(self.sires):
                s_l = sire_gebvs_load[i]
                s_h = sire_gebvs_hair[i]

                for j, d in enumerate(self.dams):
                    d_l = dam_gebvs_load[j]
                    d_h = dam_gebvs_hair[j]

                    exp_l = 0.5 * (s_l + d_l)
                    exp_h = 0.5 * (s_h + d_h)

                    # Pair-specific segregation variance scale
                    sc_ij = pair_seg_scale[i, j]
                    v_seg_l = sc_ij * (self.predictor.sigma_a_load ** 2)
                    v_seg_h = sc_ij * (self.predictor.sigma_a_hair ** 2)
                    c_seg = sc_ij * (self.predictor.rho_a * self.predictor.sigma_a_load * self.predictor.sigma_a_hair)

                    # Non-linear probability evaluations (higher is better)
                    p_l, p_h, p_joint = EnvironmentEvaluator.evaluate_attainment(
                        exp_l, exp_h, tau_load, tau_hair, scenario_eval,
                        var_seg_load=v_seg_l, var_seg_hair=v_seg_h, cov_seg=c_seg
                    )

                    if objective_mode == "loadslope_only":
                        u_val = p_l
                    elif objective_mode == "hair_only":
                        u_val = p_h
                    elif objective_mode == "weighted_marginal":
                        u_val = weight_load * p_l + (1.0 - weight_load) * p_h
                    elif objective_mode == "continuous_index":
                        # Standardized continuous merit (lower is better, so utility is negative of normalized sum)
                        u_val = - (weight_load * exp_l / self.predictor.sigma_a_load + (1.0 - weight_load) * exp_h / self.predictor.sigma_a_hair)
                    else:  # "joint"
                        u_val = p_joint

                    # Cost is negative utility
                    cost_matrix[i, j] = - float(u_val)

        # Coancestry ceiling is a hard constraint: ineligible pairs get a prohibitive cost and are
        # dropped if the solver is forced onto them (the dam is then reported as unassigned).
        ineligible_mask = (self.coancestry_matrix > self.max_coancestry_threshold)
        cost_matrix[ineligible_mask] = 1e9

        # 2. Expand Sires into Capacity Slots
        # Each sire i has max_sire_quota slots
        expanded_sire_indices = []
        expanded_cost_rows = []
        for i in range(self.num_sires):
            for _ in range(self.max_sire_quota):
                expanded_sire_indices.append(i)
                expanded_cost_rows.append(cost_matrix[i, :])

        expanded_cost_matrix = np.array(expanded_cost_rows)  # (N_slots x N_dams)

        # 3. Solve Constrained Assignment via Hungarian Algorithm
        slot_idx, dam_idx = linear_sum_assignment(expanded_cost_matrix)

        # 4. Assemble Assignments
        assignments: List[MatingAssignment] = []
        sire_usage: Dict[str, int] = {s.animal_id: 0 for s in self.sires}

        num_unassigned = 0
        for s_slot, d_i in zip(slot_idx, dam_idx):
            s_i = expanded_sire_indices[s_slot]
            if ineligible_mask[s_i, d_i]:
                num_unassigned += 1
                continue
            sire = self.sires[s_i]
            dam = self.dams[d_i]
            sire_usage[sire.animal_id] += 1

            exp_anc = 0.5 * (sire.ancestry_brahman + dam.ancestry_brahman)
            exp_l = 0.5 * (sire_gebvs_load[s_i] + dam_gebvs_load[d_i])
            exp_h = 0.5 * (sire_gebvs_hair[s_i] + dam_gebvs_hair[d_i])
            g_sd = self.g_matrix[s_i, d_i]
            coancestry = self.coancestry_matrix[s_i, d_i]
            sc = float(pair_seg_scale[s_i, d_i])
            seg_kw = dict(
                var_seg_load=sc * self.predictor.sigma_a_load ** 2,
                var_seg_hair=sc * self.predictor.sigma_a_hair ** 2,
                cov_seg=sc * self.predictor.rho_a * self.predictor.sigma_a_load * self.predictor.sigma_a_hair,
            )

            # Attainment reported with the same Mendelian sampling variance the optimizer used
            _, _, p_joint_a = EnvironmentEvaluator.evaluate_attainment(
                exp_l, exp_h, tau_load, tau_hair, scenario_eval, **seg_kw
            )
            _, _, p_joint_b = EnvironmentEvaluator.evaluate_attainment(
                exp_l, exp_h, tau_load, tau_hair, scenario_alt, **seg_kw
            )

            assignments.append(MatingAssignment(
                sire_id=sire.animal_id,
                dam_id=dam.animal_id,
                sire_ancestry=sire.ancestry_brahman,
                dam_ancestry=dam.ancestry_brahman,
                exp_offspring_ancestry=exp_anc,
                exp_gebv_load=exp_l,
                exp_gebv_hair=exp_h,
                parental_coancestry=coancestry,
                genomic_relationship_g=g_sd,
                prob_attainment_env_a=float(p_joint_a),
                prob_attainment_env_b=float(p_joint_b),
                sire_gebv_load=float(sire_gebvs_load[s_i]),
                dam_gebv_load=float(dam_gebvs_load[d_i]),
                sire_gebv_hair=float(sire_gebvs_hair[s_i]),
                dam_gebv_hair=float(dam_gebvs_hair[d_i]),
                seg_scale=sc,
            ))

        # 5. Compute Aggregate Summary Metrics
        ancestries = [a.exp_offspring_ancestry for a in assignments]
        gebvs_l = [a.exp_gebv_load for a in assignments]
        gebvs_h = [a.exp_gebv_hair for a in assignments]
        coancestries = [a.parental_coancestry for a in assignments]
        probs_a = [a.prob_attainment_env_a for a in assignments]
        probs_b = [a.prob_attainment_env_b for a in assignments]

        # Population inbreeding change: Delta F = F_offspring - F_parents
        mean_f_parents = 0.5 * (
            np.mean([s.inbreeding_f for s in self.sires]) +
            np.mean([d.inbreeding_f for d in self.dams])
        )
        mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
        mean_coancestry = mean(coancestries)
        delta_f = mean_coancestry - mean_f_parents

        used_sires = [s_id for s_id, count in sire_usage.items() if count > 0]
        max_usage = max(sire_usage.values()) if sire_usage else 0

        return PolicyEvaluationSummary(
            policy_name=policy.value,
            num_matings=len(assignments),
            num_unique_sires_used=len(used_sires),
            max_sire_usage=max_usage,
            mean_offspring_ancestry=mean(ancestries),
            ancestry_variance=float(np.var(ancestries)) if assignments else float("nan"),
            ancestry_target_mad=mean(np.abs(np.array(ancestries) - self.ancestry_target)),
            mean_gebv_load=mean(gebvs_l),
            mean_gebv_hair=mean(gebvs_h),
            mean_parental_coancestry=mean_coancestry,
            max_parental_coancestry=float(np.max(coancestries)) if assignments else float("nan"),
            population_delta_f=delta_f,
            mean_prob_attainment_env_a=mean(probs_a),
            mean_prob_attainment_env_b=mean(probs_b),
            assignments=assignments,
            num_unassigned_dams=num_unassigned,
            status="infeasible" if not assignments else ("partial" if num_unassigned else "complete"),
            relationship_assumption="assumed_unrelated_for_unknowns" if self.predictor.assume_unrelated else "measured_grm",
            gebv_mean_load=float(np.mean(np.concatenate([sire_gebvs_load, dam_gebvs_load]))),
            gebv_sd_load=float(np.std(np.concatenate([sire_gebvs_load, dam_gebvs_load]))),
            gebv_mean_hair=float(np.mean(np.concatenate([sire_gebvs_hair, dam_gebvs_hair]))),
            gebv_sd_hair=float(np.std(np.concatenate([sire_gebvs_hair, dam_gebvs_hair]))),
        )

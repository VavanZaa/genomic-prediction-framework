"""Counterfactual sensitivity analysis under a common reference truth model.

Every parent has one latent value per replicate, reused for all policies.
No real offspring outcomes are observed here. These are Gaussian Mendelian and
environmental draws, not empirical validation or physical genotype meiosis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from ..decision.environment_scenario import EnvironmentScenario
from ..decision.real_mating_optimizer import PolicyEvaluationSummary
from ..decision.real_offspring_predictor import RealOffspringPredictor

AccuracySpec = Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]]


def accuracy_from_predictive_ability(predictive_ability: float, h2: float) -> float:
    """Accuracy r(gebv, g) implied by predictive ability r(gebv, y) = r(gebv, g) * sqrt(h2), clipped to [0, 1]."""
    return float(np.clip(predictive_ability / np.sqrt(h2), 0.0, 1.0))


def measured_gebv_accuracy(
    gebv_models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_idx: Sequence[int],
    y_test_load: np.ndarray,
    y_test_hair: np.ndarray,
    h2: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Held-out predictive ability of each GEBV model and the implied accuracy per trait.

    gebv_models maps a model name to (GEBV_load, GEBV_hair) over all animals; test_idx selects the
    held-out animals whose adjusted phenotypes are y_test_*. h2 is an assumed heritability.
    """
    rows, accuracy = [], {}
    for name, (g_l, g_h) in gebv_models.items():
        ry_l = float(np.corrcoef(np.asarray(g_l)[test_idx], y_test_load)[0, 1])
        ry_h = float(np.corrcoef(np.asarray(g_h)[test_idx], y_test_hair)[0, 1])
        accuracy[name] = {"load": accuracy_from_predictive_ability(ry_l, h2), "hair": accuracy_from_predictive_ability(ry_h, h2)}
        rows.append({
            "GEBV_Model": name,
            "Predictive_Ability_Load": ry_l,
            "Predictive_Ability_Hair": ry_h,
            "Assumed_h2": h2,
            "Accuracy_Load": accuracy[name]["load"],
            "Accuracy_Hair": accuracy[name]["hair"],
        })
    return pd.DataFrame(rows), accuracy


@dataclass
class PolicyMonteCarloResult:
    policy_name: str
    scenario_name: str
    num_replicates: int
    mean_joint_pct: float
    ci95_joint_pct: Tuple[float, float]
    mean_delta_joint_vs_ancestry_pct: float
    ci95_delta_joint_vs_ancestry_pct: Tuple[float, float]
    prob_joint_superior_to_ancestry: float    # P(Joint_policy > Joint_ancestry)
    mean_load_pct: float
    prob_load_superior_to_ancestry: float
    mean_hair_pct: float
    prob_hair_superior_to_ancestry: float
    replicate_joint_attainments: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    sd_joint_pct: float = 0.0
    truth_model: str = "gebv_as_truth"
    accuracy_load: Optional[float] = None
    accuracy_hair: Optional[float] = None
    truth_source: str = ""

    @property
    def replicate_attainments(self) -> np.ndarray:
        return self.replicate_joint_attainments

    @property
    def mean_attainment_pct(self) -> float:
        return self.mean_joint_pct

    @property
    def sd_attainment_pct(self) -> float:
        return self.sd_joint_pct

    @property
    def ci95_attainment_pct(self) -> Tuple[float, float]:
        return self.ci95_joint_pct

    @property
    def mean_delta_vs_ancestry_pct(self) -> float:
        return self.mean_delta_joint_vs_ancestry_pct

    @property
    def ci95_delta_vs_ancestry_pct(self) -> Tuple[float, float]:
        return self.ci95_delta_joint_vs_ancestry_pct

    @property
    def prob_superior_to_ancestry(self) -> float:
        return self.prob_joint_superior_to_ancestry


class MatingMonteCarloSimulator:
    """Executes R forward breeding replicates for a set of mating plans."""

    def __init__(
        self,
        predictor: RealOffspringPredictor,
        num_replicates: int = 200,
        random_state: int = 42,
        gebv_accuracy: AccuracySpec = None,
        reference_parental_gebvs: Optional[Dict[str, Tuple[float, float]]] = None,
        truth_source: str = "assumed common reference model",
    ):
        """gebv_accuracy: None (legacy GEBV-as-truth), {'load': r, 'hair': r} for every policy, or
        {policy_name: {'load': r, 'hair': r}}."""
        self.predictor = predictor
        self.num_replicates = num_replicates
        self.rng = np.random.default_rng(random_state)
        self.gebv_accuracy = gebv_accuracy
        self.reference_parental_gebvs = reference_parental_gebvs
        self.truth_source = truth_source
        if num_replicates < 1: raise ValueError("num_replicates must be positive")
        if gebv_accuracy is not None and any(isinstance(v,dict) for v in gebv_accuracy.values()):
            raise ValueError("Per-policy accuracy would change the truth between policies; supply one common reference accuracy")

    def run_replicates(
        self,
        policy_summaries: Dict[str, PolicyEvaluationSummary],
        scenario: EnvironmentScenario,
        tau_load: float = 28.51,
        tau_hair: float = 5.93,
    ) -> Dict[str, PolicyMonteCarloResult]:
        """Runs R replicate cohorts for every policy under a shared environment."""
        R = self.num_replicates
        sa_l, sa_h, rho_a = self.predictor.sigma_a_load, self.predictor.sigma_a_hair, self.predictor.rho_a

        if not policy_summaries or any(not s.assignments for s in policy_summaries.values()):
            raise ValueError("Cannot simulate an empty or infeasible mating plan")
        if "GlobalAncestry" not in policy_summaries:
            raise ValueError("GlobalAncestry baseline is required for paired comparisons")
        animal_ids = sorted({aid for s in policy_summaries.values() for a in s.assignments for aid in (a.sire_id,a.dam_id)})
        pos = {aid:i for i,aid in enumerate(animal_ids)}
        reference = self.reference_parental_gebvs
        if reference is None:
            # Backward compatibility only when every appearance of each animal agrees.
            reference={}
            for summary in policy_summaries.values():
                for a in summary.assignments:
                    for aid,value in ((a.sire_id,(a.sire_gebv_load,a.sire_gebv_hair)),(a.dam_id,(a.dam_gebv_load,a.dam_gebv_hair))):
                        if aid in reference and not np.allclose(reference[aid],value):
                            raise ValueError("Policy-dependent GEBVs require an explicit common reference_parental_gebvs")
                        reference[aid]=value
        if not set(animal_ids)<=set(reference):
            raise ValueError("Common truth reference does not cover every selected animal")
        ref=np.array([reference[aid] for aid in sorted(reference)],dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 2 or not np.isfinite(ref).all():
            raise ValueError("Common reference GEBVs must contain two finite traits per animal")
        values=np.array([reference[aid] for aid in animal_ids],dtype=float)
        if self.gebv_accuracy is None:
            g=np.broadcast_to(values,(R,len(animal_ids),2))
            truth,r_l,r_h="common_gebv_as_truth",None,None
        else:
            r=np.array([self.gebv_accuracy["load"],self.gebv_accuracy["hair"]],dtype=float)
            if not np.isfinite(r).all() or np.any((r<0)|(r>1)):
                raise ValueError("Reference accuracy must be finite and in [0,1]")
            sd=ref.std(axis=0); sigma=np.array([sa_l,sa_h])
            slope=np.divide(r*sigma,sd,out=np.zeros(2),where=sd>0)
            z=self.rng.standard_normal((R,len(animal_ids),2))
            z[:,:,1]=rho_a*z[:,:,0]+np.sqrt(1-rho_a**2)*z[:,:,1]
            # Genetic merit is centered on zero; no policy-specific intercept survives r=0.
            g=(values-ref.mean(axis=0))*slope+np.sqrt(1-r**2)*sigma*z
            truth,r_l,r_h="common_reference_sensitivity",float(r[0]),float(r[1])
        self.last_parental_truth_ = {aid:g[:,i,:].copy() for aid,i in pos.items()}
        # Common noise per mating occurrence makes identical plans exactly identical,
        # irrespective of policy labels, input ordering, or duplicated dam slots.
        from collections import Counter
        maxima={}
        for summary in policy_summaries.values():
            for pair,n in Counter((a.sire_id,a.dam_id) for a in summary.assignments).items():
                maxima[pair]=max(maxima.get(pair,0),n)
        keys=[(s,d,k) for (s,d),n in sorted(maxima.items()) for k in range(n)]
        key_pos={k:i for i,k in enumerate(keys)}
        zs_all=self.rng.standard_normal((R,len(keys),2))
        ze_all=self.rng.standard_normal((R,len(keys),2))
        joint,load,hair,meta={},{},{},{}
        for p_name,summary in policy_summaries.items():
            asg=summary.assignments; n_m=len(asg)
            seg=np.array([a.seg_scale if np.isfinite(a.seg_scale) else 0.5 for a in asg])
            si=np.array([pos[a.sire_id] for a in asg]); di=np.array([pos[a.dam_id] for a in asg])
            mid_l=0.5*(g[:,si,0]+g[:,di,0]); mid_h=0.5*(g[:,si,1]+g[:,di,1])
            counts=Counter(); ki=[]
            for a in asg:
                pair=(a.sire_id,a.dam_id); ki.append(key_pos[(*pair,counts[pair])]); counts[pair]+=1
            meta[p_name]=(truth,r_l,r_h)
            # Mendelian sampling (pair-specific scale) and environmental residual, each with its correlation
            zs = zs_all[:,ki,:]
            ze = ze_all[:,ki,:]
            seg_l = np.sqrt(seg) * sa_l * zs[:, :, 0]
            seg_h = np.sqrt(seg) * sa_h * (rho_a * zs[:, :, 0] + np.sqrt(1 - rho_a ** 2) * zs[:, :, 1])
            env_l = scenario.sigma_e_load * ze[:, :, 0]
            env_h = scenario.sigma_e_hair * (scenario.rho_e * ze[:, :, 0] + np.sqrt(1 - scenario.rho_e ** 2) * ze[:, :, 1])
            y_l = scenario.mu_load + mid_l + seg_l + env_l
            y_h = scenario.mu_hair + mid_h + seg_h + env_h

            ok_l, ok_h = y_l < tau_load, y_h < tau_hair
            joint[p_name] = np.mean(ok_l & ok_h, axis=1) * 100.0
            load[p_name] = np.mean(ok_l, axis=1) * 100.0
            hair[p_name] = np.mean(ok_h, axis=1) * 100.0

        anc_joint = joint.get("GlobalAncestry", np.zeros(R))
        anc_load = load.get("GlobalAncestry", np.zeros(R))
        anc_hair = hair.get("GlobalAncestry", np.zeros(R))

        results = {}
        for p_name in policy_summaries:
            reps_j = joint[p_name]
            deltas = reps_j - anc_joint
            truth, r_l, r_h = meta[p_name]
            results[p_name] = PolicyMonteCarloResult(
                policy_name=p_name,
                scenario_name=scenario.name,
                num_replicates=R,
                mean_joint_pct=float(np.mean(reps_j)),
                ci95_joint_pct=(float(np.percentile(reps_j, 2.5)), float(np.percentile(reps_j, 97.5))),
                mean_delta_joint_vs_ancestry_pct=float(np.mean(deltas)),
                ci95_delta_joint_vs_ancestry_pct=(float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
                prob_joint_superior_to_ancestry=float(np.mean(deltas > 0)),
                mean_load_pct=float(np.mean(load[p_name])),
                prob_load_superior_to_ancestry=float(np.mean((load[p_name] - anc_load) > 0)),
                mean_hair_pct=float(np.mean(hair[p_name])),
                prob_hair_superior_to_ancestry=float(np.mean((hair[p_name] - anc_hair) > 0)),
                replicate_joint_attainments=reps_j,
                sd_joint_pct=float(np.std(reps_j, ddof=1)) if R > 1 else 0.0,
                truth_model=truth,
                accuracy_load=r_l,
                accuracy_hair=r_h,
                truth_source=self.truth_source,
            )
        return results

"""Case 2: Wagyu x Angus breeding-strategy comparison under equal resources.

Resource model (every number is an explicit, assumed parameter of `run`):
- Each policy has the same `num_dams` dam slots per generation. Each dam has `calves_per_dam` calves
  per generation interval (discrete-generation approximation), each from a separately allocated
  mating; sires are limited to `sire_quota` calves per generation; matings whose offspring would
  exceed `max_inbreeding` (pedigree F) are not allowed.
- Replacement dams come from the policy's own female calves (young females first, previous dams
  reused if short); purchased females fill any remaining gap and are counted.
- The F1 terminal policy cannot keep its F1 females, so its Angus cows are replaced from a purebred
  parental line whose size is reported as `parental_line_dams`.
- Purebred bulls (Wagyu) are external sire lines, sampled fresh each generation.
- Carcass traits are recorded only on steers (a fixed fraction of male calves) that never become
  parents; genomic predictors used at generation t are trained on steers born before t, so no
  candidate's own carcass record is used to select it.
All policies share the founders and the generation-1 F1 cohort (common random numbers) and are
compared on the calves born in the same final generation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from ..ancestry.pedigree import Pedigree
from ..decision.joint_probability import JointProbabilityEstimator
from ..decision.mating_allocator import BreedingPolicy, MatingAllocator, MatingPlan
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import MeiosisSimulator, Population, SimulatedIndividual
from ..prediction.models import AdditiveGBLUP
from ..traits.simulator import TraitSimulator

POLICY_ORDER = ["F1_Terminal", "Backcrossing_BC3", "Closed_Composite", "Composite_GenomicSelection", "Optimization_ML"]
POLICY_LABELS = {
    "F1_Terminal": "1. Repeated F1 terminal cross",
    "Backcrossing_BC3": "2. Backcrossing to Wagyu (BC1-BC3)",
    "Closed_Composite": "3. Closed composite, random mating",
    "Composite_GenomicSelection": "3b. Composite + genomic selection (reference)",
    "Optimization_ML": "4. Composite + offspring-aware mating optimisation",
}
MIN_TRAINING_STEERS = 12


@dataclass
class PolicySummary:
    policy_name: str
    target_attainment: float
    marbling_mean: float
    marbling_std: float
    growth_mean: float
    growth_std: float
    inbreeding_F: float
    heterozygosity_Ho: float
    wagyu_ancestry_mean: float
    parental_maintenance_burden: float   # parental-line dams / (herd dams + parental-line dams)
    elapsed_years: float                 # years until the policy first produces its target product
    failure_cases: List[str]
    target_attainment_ci95: Tuple[float, float] = (0.0, 0.0)
    pedigree_wagyu_mean: float = 0.0
    parental_line_dams: float = 0.0
    purchased_females: float = 0.0
    unassigned_dam_slots: float = 0.0
    calves_evaluated: float = 0.0
    dam_slots_per_generation: int = 0
    generations_to_first_product: int = 0
    production_calves_per_dam_slot: float = 0.0
    predictor_accuracy: Optional[float] = None
    n_replicates: int = 1


@dataclass
class WagyuAngusResults:
    policies: Dict[str, PolicySummary]
    summary_table: pd.DataFrame
    narrative_report: str
    trajectory: Optional[pd.DataFrame] = None
    replicate_table: Optional[pd.DataFrame] = None
    pedigree_checks: Dict[str, float] = field(default_factory=dict)
    calibration: Optional[pd.DataFrame] = None
    paired_comparisons: Optional[pd.DataFrame] = None
    assumptions: Dict[str, object] = field(default_factory=dict)


@dataclass
class _PolicyState:
    key: str
    latest_calves: List[SimulatedIndividual]
    previous_dams: List[SimulatedIndividual]
    candidate_bulls: List[SimulatedIndividual]
    steer_ids: List[str]
    purchased_females: int = 0
    parental_line_dams: int = 0
    unassigned_dam_slots: int = 0
    retained_last: int = 0
    predictor_accuracy: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    parent_ids: set = field(default_factory=set)


class WagyuAngusCaseStudy:
    """Executes Case 2: Wagyu x Angus policy comparison with explicit resource accounting."""

    def __init__(self, genome_spec: Optional[GenomeSpec] = None, seed: int = 2026):
        self.genome_spec = genome_spec or GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=30, qtls_per_chrom=6)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ public API

    def run(
        self,
        num_dams: int = 60,
        target_marbling: float = 6.5,
        target_growth: float = 360.0,
        generation_interval_years: float = 5.0,
        generations: int = 4,
        calves_per_dam: int = 2,
        sire_quota: int = 12,
        steer_fraction: float = 0.5,
        max_inbreeding: float = 0.125,
        n_replicates: int = 1,
        num_virtual_offspring: int = 40,
        reference_size_per_breed: int = 100,
    ) -> WagyuAngusResults:
        if n_replicates < 1 or num_dams < 2 or calves_per_dam < 1 or sire_quota < 1:
            raise ValueError("Positive replicates and capacities, and at least two dams, are required")
        if not 0 <= steer_fraction < 1:
            raise ValueError("steer_fraction must be in [0, 1)")
        if generations < 2:
            raise ValueError("At least two generations are needed to compare the policies.")
        params = dict(
            num_dams=num_dams, target_marbling=target_marbling, target_growth=target_growth,
            generations=generations, calves_per_dam=calves_per_dam, sire_quota=sire_quota,
            steer_fraction=steer_fraction, max_inbreeding=max_inbreeding,
            num_virtual_offspring=num_virtual_offspring, reference_size_per_breed=reference_size_per_breed,
        )
        replicates = [self._run_replicate(seed=self.seed + 1000 * r, **params) for r in range(n_replicates)]

        rep_rows, traj_rows = [], []
        for r, rep in enumerate(replicates):
            for key, m in rep["metrics"].items():
                rep_rows.append({"replicate": r, "policy": key, **m})
            for row in rep["trajectory"]:
                traj_rows.append({"replicate": r, **row})
        rep_df = pd.DataFrame(rep_rows)
        traj_df = pd.DataFrame(traj_rows)

        policies = {
            key: self._summarize_policy(key, rep_df[rep_df["policy"] == key], params, generation_interval_years)
            for key in POLICY_ORDER
        }
        comparisons = self._paired_comparisons(rep_df)
        self._attach_failure_cases(policies, rep_df, params, comparisons)

        summary_df = pd.DataFrame([self._table_row(p) for p in policies.values()])
        pedigree_checks = {k: float(np.mean([rep["pedigree_checks"][k] for rep in replicates])) for k in replicates[0]["pedigree_checks"]}
        assumptions = {
            "dam_slots_per_generation": num_dams,
            "calves_per_dam_per_generation": calves_per_dam,
            "sire_quota_per_generation": sire_quota,
            "max_pedigree_inbreeding_of_offspring": max_inbreeding,
            "steer_fraction_of_male_calves": steer_fraction,
            "generations_compared": generations,
            "generation_interval_years": generation_interval_years,
            "targets": {"Marbling (BMS)": target_marbling, "Growth (kg)": target_growth},
            "replacement_rule": "composites use own females or existing composite dams only; unavailable dam capacity is reported",
            "final_horizon": "all final-generation calves are available for marketing; no replacements retained beyond the horizon",
            "wagyu_sires": "fresh unrelated purebred bulls each generation (external sire line)",
            "training_data": "steers born before the decision generation (carcass-recorded, never parents)",
            "h2_reference": replicates[0]["calibration"]["reference"].iloc[0],
            "n_replicates": n_replicates,
            "parameter_status": "all resource and trait parameters are assumed (synthetic case), not measured",
        }
        narrative = self._narrative(policies, pedigree_checks, assumptions, comparisons)
        return WagyuAngusResults(
            policies=policies,
            paired_comparisons=comparisons,
            summary_table=summary_df,
            narrative_report=narrative,
            trajectory=traj_df,
            replicate_table=rep_df,
            pedigree_checks=pedigree_checks,
            calibration=replicates[0]["calibration"],
            assumptions=assumptions,
        )

    # ------------------------------------------------------------------ one replicate

    def _run_replicate(
        self, seed: int, num_dams: int, target_marbling: float, target_growth: float, generations: int,
        calves_per_dam: int, sire_quota: int, steer_fraction: float, max_inbreeding: float,
        num_virtual_offspring: int, reference_size_per_breed: int,
    ) -> Dict:
        genome = self.genome_spec
        rng = np.random.default_rng(seed)
        simulator, freqs = TraitSimulator.create_wagyu_angus_architecture(genome, seed=seed)

        ref_pop = Population(genome)
        reference = [
            ref_pop.create_founder(f"REF_{b}_{i}", sex="M" if i % 2 else "F", breed=b, rng=rng, founder_allele_frequencies=freqs)
            for b in ("Wagyu", "Angus") for i in range(reference_size_per_breed)
        ]
        calibration = simulator.calibrate_residuals(
            reference, genome, reference_label=f"within-breed Wagyu and Angus founders (n={len(reference)})",
            group_by_breed=True
        )

        pop, ped = Population(genome), Pedigree()
        records: Dict[str, dict] = {}
        n_sires = math.ceil(num_dams * calves_per_dam / sire_quota)

        def founders(prefix: str, breed: str, sex: str, n: int) -> List[SimulatedIndividual]:
            out = []
            for i in range(n):
                ind = pop.create_founder(f"{prefix}_{i}", sex=sex, breed=breed, rng=rng, founder_allele_frequencies=freqs)
                ped.add_founder(ind.id, sex=sex, breed=breed)
                out.append(ind)
            return out

        def allocator() -> MatingAllocator:
            return MatingAllocator(genome, pedigree=ped, max_sire_quota=sire_quota,
                                   max_inbreeding_threshold=max_inbreeding, dam_capacity=calves_per_dam)

        def breed(tag: str, plan: MatingPlan, sires, dams) -> List[SimulatedIndividual]:
            smap = {s.id: s for s in sires}
            dmap = {d.id: d for d in dams}
            calves = []
            for i, pair in enumerate(plan.pairs):
                sex = "M" if rng.random() < 0.5 else "F"
                calf = MeiosisSimulator.mate(smap[pair.sire_id], dmap[pair.dam_id], f"{tag}_{i}", sex, genome, rng)
                pop.add(calf)
                ped.add_cross(calf.id, pair.sire_id, pair.dam_id, sex=sex)
                calves.append(calf)
            for row in simulator.simulate_records(calves, genome, rng).to_dict("records"):
                records[row["id"]] = row  # each calf's phenotype is simulated exactly once
            return calves

        def split_males(calves):
            males = [c for c in calves if c.sex == "M"]
            order = rng.permutation(len(males))
            n_steer = int(round(steer_fraction * len(males)))
            return [males[i] for i in order[:n_steer]], [males[i] for i in order[n_steer:]]

        # Generation 1 (shared by every policy): Angus cows x Wagyu bulls -> F1
        angus_cows = founders("G0_AngusCow", "Angus", "F", num_dams)
        wagyu_bulls = founders("G1_WagyuBull", "Wagyu", "M", n_sires)
        plan_g1 = allocator().build_plan(BreedingPolicy.RANDOM, wagyu_bulls, angus_cows, 0.0, 0.0, rng=rng)
        g1 = breed("G1_F1", plan_g1, wagyu_bulls, angus_cows)
        g1_steers, g1_bulls = split_males(g1)
        states = {
            key: _PolicyState(key, latest_calves=g1, previous_dams=angus_cows, candidate_bulls=g1_bulls,
                              steer_ids=[s.id for s in g1_steers], unassigned_dam_slots=plan_g1.unassigned_dam_slots)
            for key in POLICY_ORDER
        }
        trajectory = [self._traj_row(key, 1, g1, records, ped, target_marbling, target_growth) for key in POLICY_ORDER]
        bc_by_gen: Dict[int, List[SimulatedIndividual]] = {1: g1}
        comp_g2: List[SimulatedIndividual] = []

        for t in range(2, generations + 1):
            for key in POLICY_ORDER:
                st = states[key]
                tag = f"{key}_G{t}"
                young_f = [c for c in st.latest_calves if c.sex == "F"]
                purchased = 0

                if key == "F1_Terminal":
                    dams = founders(f"{tag}_AngusCow", "Angus", "F", num_dams)
                    sires = founders(f"{tag}_WagyuBull", "Wagyu", "M", n_sires)
                    # Purebred dams needed to breed num_dams replacement heifers per generation
                    st.parental_line_dams = math.ceil(num_dams / (calves_per_dam * 0.5))
                    plan = allocator().build_plan(BreedingPolicy.RANDOM, sires, dams, 0.0, 0.0, rng=rng)

                elif key == "Backcrossing_BC3":
                    grade = lambda d: float(d.pedigree_composition.get("Wagyu"))
                    dams, short = self._choose_dams(young_f, st.previous_dams, num_dams, grade)
                    if short:
                        dams += founders(f"{tag}_BoughtCow", "Angus", "F", short)
                        purchased = short
                    sires = founders(f"{tag}_WagyuBull", "Wagyu", "M", n_sires)
                    plan = allocator().build_plan(BreedingPolicy.RANDOM, sires, dams, 0.0, 0.0, rng=rng)

                else:
                    models = None if key == "Closed_Composite" else self._fit_models(st.steer_ids, records, pop, genome)
                    if key != "Closed_Composite" and models is None:
                        st.notes.append(f"G{t}: fewer than {MIN_TRAINING_STEERS} training steers; used random selection")
                    if models is None:
                        rand_key = {c.id: rng.random() for c in young_f + st.previous_dams + st.candidate_bulls}
                        rank = lambda d: rand_key[d.id]
                    else:
                        p_m, p_g, res_cov = models
                        candidates = young_f + st.previous_dams + st.candidate_bulls
                        index = self._index(candidates, p_m, p_g, genome)
                        rank = lambda d: index[d.id]
                        st.predictor_accuracy = self._accuracy(young_f + st.candidate_bulls, p_m, p_g, records, genome)
                    closed_previous = [d for d in st.previous_dams if float(d.pedigree_composition.get("Wagyu")) == 0.5]
                    dams, short = self._choose_dams(young_f, closed_previous, num_dams, rank)
                    st.unassigned_dam_slots += short * calves_per_dam
                    if short:
                        st.notes.append(f"G{t}: {short} dam slots unavailable; no outside females introduced into closed composite")
                    if not dams:
                        raise RuntimeError(f"{key} generation {t}: no females remain in the closed herd")
                    sires = sorted(st.candidate_bulls, key=rank, reverse=True)[:n_sires]
                    if not sires:
                        raise RuntimeError(f"{key} generation {t}: no candidate bulls left.")
                    if key == "Optimization_ML" and models is not None:
                        plan = allocator().build_plan(
                            BreedingPolicy.OFFSPRING_AWARE, sires, dams, target_marbling, target_growth,
                            predictor_trait1=p_m, predictor_trait2=p_g, residual_cov=res_cov, rng=rng,
                            num_virtual_offspring=num_virtual_offspring,
                        )
                    else:
                        plan = allocator().build_plan(BreedingPolicy.RANDOM, sires, dams, 0.0, 0.0, rng=rng)

                latest_ids = {c.id for c in st.latest_calves}
                used_parents = {p.sire_id for p in plan.pairs} | {p.dam_id for p in plan.pairs}
                st.retained_last = len(latest_ids & used_parents)
                prior_row = next(row for row in trajectory if row["policy"] == key and row["generation"] == t-1)
                prior_row["retained_as_parents"] = st.retained_last
                prior_row["marketed_calves"] = prior_row["births"] - st.retained_last
                st.parent_ids.update(d.id for d in dams)
                st.parent_ids.update(s.id for s in sires)
                calves = breed(tag, plan, sires, dams)
                st.unassigned_dam_slots += plan.unassigned_dam_slots
                st.purchased_females += purchased
                st.previous_dams, st.latest_calves = dams, calves
                steers, bulls = split_males(calves)
                st.steer_ids += [s.id for s in steers]
                st.candidate_bulls = bulls
                trajectory.append(self._traj_row(key, t, calves, records, ped, target_marbling, target_growth))
                if key == "Backcrossing_BC3":
                    bc_by_gen[t] = calves
                if key == "Closed_Composite" and t == 2:
                    comp_g2 = calves

        metrics = {}
        for key in POLICY_ORDER:
            st = states[key]
            m = self._cohort_metrics(st.latest_calves, records, ped, target_marbling, target_growth)
            n_calves = len(st.latest_calves)
            m.update({
                "parental_line_dams": st.parental_line_dams,
                "purchased_females": st.purchased_females,
                "unassigned_dam_slots": st.unassigned_dam_slots,
                "production_calves_per_dam_slot": n_calves / (num_dams + st.parental_line_dams),
                "final_births": n_calves, "final_retained_as_parents": 0, "final_marketed_calves": n_calves,
                "predictor_accuracy": st.predictor_accuracy,
                # carcass-recorded training animals that were also used as parents (must be 0)
                "training_parent_overlap": len(set(st.steer_ids) & st.parent_ids),
                "notes": "; ".join(st.notes),
            })
            metrics[key] = m

        return {
            "metrics": metrics,
            "trajectory": trajectory,
            "pedigree_checks": self._pedigree_checks(bc_by_gen, comp_g2, ped),
            "calibration": calibration,
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _choose_dams(young, previous, n, rank) -> Tuple[List[SimulatedIndividual], int]:
        """Best-ranked young females first, then previous dams; returns (dams, shortfall)."""
        chosen = sorted(young, key=rank, reverse=True)[:n]
        if len(chosen) < n:
            taken = {d.id for d in chosen}
            chosen += sorted((d for d in previous if d.id not in taken), key=rank, reverse=True)[: n - len(chosen)]
        return chosen, n - len(chosen)

    @staticmethod
    def _fit_models(steer_ids, records, pop, genome):
        if len(steer_ids) < MIN_TRAINING_STEERS:
            return None
        markers = np.vstack([pop.individuals[i].get_marker_genotypes(genome) for i in steer_ids])
        y_m = np.array([records[i]["phenotype_Marbling"] for i in steer_ids])
        y_g = np.array([records[i]["phenotype_Growth"] for i in steer_ids])
        p_m = AdditiveGBLUP().fit(markers, y_m)
        p_g = AdditiveGBLUP().fit(markers, y_g)
        res_cov = JointProbabilityEstimator.out_of_fold_residual_covariance(
            markers, y_m, y_g, AdditiveGBLUP, n_folds=min(5, len(steer_ids) // 4)
        )
        return p_m, p_g, res_cov

    @staticmethod
    def _index(individuals, p_m, p_g, genome) -> Dict[str, float]:
        markers = np.vstack([ind.get_marker_genotypes(genome) for ind in individuals])
        total = np.zeros(len(individuals))
        for pred in (p_m.predict(markers), p_g.predict(markers)):
            sd = float(np.std(pred))
            total += (pred - np.mean(pred)) / sd if sd > 0 else 0.0
        return {ind.id: float(v) for ind, v in zip(individuals, total)}

    @staticmethod
    def _accuracy(individuals, p_m, p_g, records, genome) -> Optional[float]:
        """Mean correlation of predicted and true genetic values over the candidates (simulation truth)."""
        if len(individuals) < 3:
            return None
        markers = np.vstack([ind.get_marker_genotypes(genome) for ind in individuals])
        rs = []
        for pred, trait in ((p_m.predict(markers), "Marbling"), (p_g.predict(markers), "Growth")):
            truth = np.array([records[ind.id][f"true_g_{trait}"] for ind in individuals])
            if np.std(pred) > 0 and np.std(truth) > 0:
                rs.append(float(np.corrcoef(pred, truth)[0, 1]))
        return float(np.mean(rs)) if rs else None

    @staticmethod
    def _cohort_metrics(calves, records, ped, t_marb, t_grow) -> Dict[str, float]:
        df = pd.DataFrame([records[c.id] for c in calves])
        succ = (df["phenotype_Marbling"] >= t_marb) & (df["phenotype_Growth"] >= t_grow)
        # Whole-cohort pedigree range: a closed composite must stay at exactly 1/2 Wagyu for every calf
        ped_wagyu = [float(c.pedigree_composition.get("Wagyu")) for c in calves]
        return {
            "target_attainment": float(succ.mean()),
            "calves_evaluated": len(df),
            "marbling_mean": float(df["phenotype_Marbling"].mean()),
            "marbling_std": float(df["phenotype_Marbling"].std(ddof=0)),
            "growth_mean": float(df["phenotype_Growth"].mean()),
            "growth_std": float(df["phenotype_Growth"].std(ddof=0)),
            "inbreeding_F": float(np.mean([ped.inbreeding(c.id) for c in calves])),
            "heterozygosity_Ho": float(df["heterozygosity"].mean()),
            "wagyu_ancestry_mean": float(df.get("genomic_Wagyu", pd.Series(0.0, index=df.index)).fillna(0.0).mean()),
            "pedigree_wagyu_mean": float(df.get("pedigree_Wagyu", pd.Series(0.0, index=df.index)).fillna(0.0).mean()),
            "pedigree_wagyu_min": min(ped_wagyu),
            "pedigree_wagyu_max": max(ped_wagyu),
        }

    def _traj_row(self, key, t, calves, records, ped, t_marb, t_grow) -> Dict:
        m = self._cohort_metrics(calves, records, ped, t_marb, t_grow)
        return {"policy": key, "generation": t, "births": len(calves), "retained_as_parents": 0, "marketed_calves": len(calves), **m}

    @staticmethod
    def _pedigree_checks(bc_by_gen, comp_g2, ped) -> Dict[str, float]:
        """Mean pedigree Wagyu fraction of backcross calves out of dams of the expected grade."""
        checks = {"F1 (Wagyu x Angus)": float(np.mean([float(c.pedigree_composition.get("Wagyu")) for c in bc_by_gen[1]]))}
        for t, label in ((2, "BC1"), (3, "BC2"), (4, "BC3")):
            if t not in bc_by_gen:
                continue
            dam_grade = 1.0 - 0.5 ** (t - 1)
            subset = [c for c in bc_by_gen[t] if abs(float(ped.individuals[c.dam_id].breed_fraction.get("Wagyu")) - dam_grade) < 1e-12]
            if subset:
                checks[label] = float(np.mean([float(c.pedigree_composition.get("Wagyu")) for c in subset]))
        f1xf1 = [c for c in comp_g2
                 if float(ped.individuals[c.sire_id].breed_fraction.get("Wagyu")) == 0.5
                 and float(ped.individuals[c.dam_id].breed_fraction.get("Wagyu")) == 0.5]
        if f1xf1:
            checks["F1 x F1"] = float(np.mean([float(c.pedigree_composition.get("Wagyu")) for c in f1xf1]))
        return checks

    @staticmethod
    def _summarize_policy(key, rows: pd.DataFrame, params, interval) -> PolicySummary:
        n_rep = len(rows)
        att = rows["target_attainment"].to_numpy()
        mean_att = float(att.mean())
        if n_rep >= 2:
            half = float(stats.t.ppf(0.975, n_rep-1)) * float(att.std(ddof=1)) / math.sqrt(n_rep)
        else:
            # One population cannot estimate between-population uncertainty. Report
            # the full bounded interval rather than a falsely precise calf-level CI.
            half = 1.0
        first_product = {"F1_Terminal": 1, "Backcrossing_BC3": min(4, params["generations"])}.get(key, 2)
        parental = float(rows["parental_line_dams"].mean())
        acc = rows["predictor_accuracy"].dropna()
        return PolicySummary(
            policy_name=POLICY_LABELS[key],
            target_attainment=mean_att,
            target_attainment_ci95=(max(0.0, mean_att - half), min(1.0, mean_att + half)),
            marbling_mean=float(rows["marbling_mean"].mean()),
            marbling_std=float(rows["marbling_std"].mean()),
            growth_mean=float(rows["growth_mean"].mean()),
            growth_std=float(rows["growth_std"].mean()),
            inbreeding_F=float(rows["inbreeding_F"].mean()),
            heterozygosity_Ho=float(rows["heterozygosity_Ho"].mean()),
            wagyu_ancestry_mean=float(rows["wagyu_ancestry_mean"].mean()),
            pedigree_wagyu_mean=float(rows["pedigree_wagyu_mean"].mean()),
            parental_line_dams=parental,
            parental_maintenance_burden=parental / (params["num_dams"] + parental),
            purchased_females=float(rows["purchased_females"].mean()),
            unassigned_dam_slots=float(rows["unassigned_dam_slots"].mean()),
            calves_evaluated=float(rows["calves_evaluated"].mean()),
            dam_slots_per_generation=params["num_dams"],
            generations_to_first_product=first_product,
            elapsed_years=first_product * interval,
            production_calves_per_dam_slot=float(rows["production_calves_per_dam_slot"].mean()),
            predictor_accuracy=float(acc.mean()) if len(acc) else None,
            n_replicates=n_rep,
            failure_cases=[],
        )

    @staticmethod
    def _paired_comparisons(rep_df):
        wide=rep_df.pivot(index="replicate",columns="policy",values="target_attainment")
        rows=[]
        for baseline in ("F1_Terminal","Composite_GenomicSelection"):
            for policy in POLICY_ORDER:
                if policy==baseline: continue
                d=(wide[policy]-wide[baseline]).to_numpy(); n=len(d)
                half=float(stats.t.ppf(.975,n-1)*np.std(d,ddof=1)/np.sqrt(n)) if n>1 else 2.0
                rows.append({"policy":policy,"baseline":baseline,"mean_difference":float(d.mean()),
                             "ci95_low":max(-1.,float(d.mean()-half)),"ci95_high":min(1.,float(d.mean()+half)),
                             "replicates":n,"interval_method":"paired t" if n>1 else "unavailable: one population"})
        return pd.DataFrame(rows)

    @staticmethod
    def _attach_failure_cases(policies: Dict[str, PolicySummary], rep_df: pd.DataFrame, params,
                              comparisons: pd.DataFrame) -> None:
        vs_f1 = comparisons[comparisons["baseline"] == "F1_Terminal"].set_index("policy")
        for key, p in policies.items():
            flags = []
            if key != "F1_Terminal":
                row = vs_f1.loc[key]
                if row["replicates"] > 1 and row["ci95_high"] < 0:
                    flags.append(f"Joint attainment below the F1 terminal cross (paired difference "
                                 f"{row['mean_difference']:+.1%}; 95% interval excludes zero).")
            if p.inbreeding_F > 0.0625:
                flags.append(f"Mean pedigree inbreeding {p.inbreeding_F:.3f} exceeds 1/16.")
            if p.unassigned_dam_slots > 0:
                flags.append(f"Pedigree constraint left {p.unassigned_dam_slots:.1f} dam slots unfilled.")
            if p.purchased_females > 0:
                flags.append(f"Needed {p.purchased_females:.1f} purchased females to fill dam slots.")
            if p.parental_line_dams > 0:
                flags.append(f"Needs a purebred Angus parental line of {p.parental_line_dams:.0f} dams per generation.")
            if p.predictor_accuracy is not None and p.predictor_accuracy < 0.3:
                flags.append(f"Genomic predictor accuracy only r = {p.predictor_accuracy:.2f} against true genetic value.")
            if p.growth_mean < params["target_growth"]:
                flags.append(f"Mean growth {p.growth_mean:.1f} kg below the {params['target_growth']:.0f} kg target.")
            if p.marbling_mean < params["target_marbling"]:
                flags.append(f"Mean marbling {p.marbling_mean:.2f} below the {params['target_marbling']} target.")
            notes = rep_df.loc[rep_df["policy"] == key, "notes"]
            flags += sorted({n for n in notes if n})
            p.failure_cases = flags or ["No failure condition triggered at this horizon."]

    @staticmethod
    def _table_row(p: PolicySummary) -> Dict[str, str]:
        lo, hi = p.target_attainment_ci95
        return {
            "Policy": p.policy_name,
            "Target Attainment (%)": f"{p.target_attainment * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]",
            "Marbling (mean ± sd)": f"{p.marbling_mean:.2f} ± {p.marbling_std:.2f}",
            "Growth kg (mean ± sd)": f"{p.growth_mean:.1f} ± {p.growth_std:.1f}",
            "Inbreeding F": f"{p.inbreeding_F:.4f}",
            "Heterozygosity Ho": f"{p.heterozygosity_Ho:.3f}",
            "Wagyu Ancestry (genomic)": f"{p.wagyu_ancestry_mean * 100:.1f}%",
            "Parental-line dams": f"{p.parental_line_dams:.0f}",
            "Parent Herd Burden": f"{p.parental_maintenance_burden * 100:.0f}%",
            "Final marketed calves / total dam slot": f"{p.production_calves_per_dam_slot:.2f}",
            "Years to first product": f"{p.elapsed_years:.0f}",
            "Replicates": str(p.n_replicates),
        }

    @staticmethod
    def _narrative(policies: Dict[str, PolicySummary], pedigree_checks: Dict[str, float], assumptions: Dict, comparisons: pd.DataFrame) -> str:
        ranked = sorted(policies.items(), key=lambda kv: -kv[1].target_attainment)
        lead_key, lead = ranked[0]
        lines = [
            "=== Case 2: Wagyu x Angus Policy Comparison ===",
            f"Horizon: calves of generation {assumptions['generations_compared']}; "
            f"{assumptions['dam_slots_per_generation']} dam slots x {assumptions['calves_per_dam_per_generation']} calves per generation for every policy; "
            f"{assumptions['n_replicates']} replicate(s). All parameters are assumed, not measured.",
            "",
            "Joint target attainment (95% interval):",
        ]
        for key, p in ranked:
            lo, hi = p.target_attainment_ci95
            if key == lead_key:
                rel = "highest"
            elif hi < lead.target_attainment_ci95[0]:
                rel = "lower point estimate"
            else:
                rel = "lower or tied point estimate; see paired comparisons"
            lines.append(f"- {p.policy_name}: {p.target_attainment:.1%} [{lo:.1%}, {hi:.1%}], {rel}")

        alloc = comparisons[(comparisons["policy"] == "Optimization_ML")
                            & (comparisons["baseline"] == "Composite_GenomicSelection")].iloc[0]
        alloc_ci = (f"paired 95% interval [{alloc['ci95_low']:+.1%}, {alloc['ci95_high']:+.1%}]"
                    if alloc["replicates"] > 1 else "no interval from a single replicate population")
        lines += [
            "",
            f"Mating-allocation effect (offspring-aware minus genomic selection with random mating, same parents' "
            f"selection rule): {alloc['mean_difference']:+.1%}, {alloc_ci}.",
            f"Resources: the F1 terminal cross needs {policies['F1_Terminal'].parental_line_dams:.0f} purebred parental-line dams per "
            f"generation on top of the herd; production calves per dam slot: "
            + ", ".join(f"{k.split('_')[0]} {p.production_calves_per_dam_slot:.2f}" for k, p in policies.items()) + ".",
            "Pedigree checks (mean pedigree Wagyu fraction): " + ", ".join(f"{k} {v:.4f}" for k, v in pedigree_checks.items()) + ".",
        ]
        lines += ["", "Paired policy differences (95% t-intervals across replicate populations):",
                  comparisons.to_string(index=False),
                  "Production counts use matching birth cohorts; all final-generation calves are marketed at the terminal horizon."]
        return "\n".join(lines) + "\n"

"""Case 1: Brangus ancestry reconstruction and subsequent trait selection stability.

1. RouteFinder recovers crossing routes to 5/8 Angus : 3/8 Brahman from pure founders (no hardcoding).
2. The selected route is executed as physical matings: every formation cow is bred through the
   route's dam-side subtree from her own unrelated founders, and mated once to a bull from the
   sire-side subtree (bull lines = n_founder_families). The formation cohort therefore has the exact
   target pedigree composition and zero pedigree inbreeding.
3. Realized genomic ancestry around the pedigree expectation is quantified.
4. The cohort is closed and selected on a standardized two-trait phenotype index for further
   generations, with pedigree-constrained mating, to see whether the trait combination persists.
Ancestry-target recovery (steps 1-2) is kept separate from trait selection (step 4).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..ancestry.pedigree import Pedigree
from ..ancestry.route_finder import CrossRoute, RouteFinder
from ..decision.mating_allocator import BreedingPolicy, MatingAllocator
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import MeiosisSimulator, Population, SimulatedIndividual
from ..inheritance.route_executor import RouteExecutor
from ..traits.simulator import TraitSimulator

TARGET = {"Angus": Fraction(5, 8), "Brahman": Fraction(3, 8)}


@dataclass
class BrangusResults:
    routes_found: List[CrossRoute]
    route_used: CrossRoute
    realized_ancestry_mean: float
    realized_ancestry_std: float
    realized_ancestry_quantiles: Dict[str, float]
    generation_progression: pd.DataFrame
    summary_report: str
    formation_pedigree_exact: bool = False
    calibration: Optional[pd.DataFrame] = None
    formation_cohort: List[SimulatedIndividual] = field(default_factory=list)
    population: Optional[Population] = None
    pedigree: Optional[Pedigree] = None


class BrangusCaseStudy:
    """Executes Case 1: Brangus ancestry reconstruction and multi-generation trait selection."""

    def __init__(self, genome_spec: Optional[GenomeSpec] = None, seed: int = 42):
        self.genome_spec = genome_spec or GenomeSpec.default_bovine(num_chromosomes=10, markers_per_chrom=30, qtls_per_chrom=6)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        n_founder_families: int = 4,
        post_formation_generations: int = 3,
        generation_size: int = 40,
        route_index: int = 0,
        dam_side: str = "auto",
        max_route_depth: int = 3,
        calves_per_dam: int = 2,
        male_selection_fraction: float = 0.25,
        max_inbreeding: float = 0.125,
        reference_size_per_breed: int = 100
    ) -> BrangusResults:
        """n_founder_families is the number of independent bull lines used for formation."""
        genome = self.genome_spec
        rng = self.rng

        # Step 1: routes from exact ancestry search
        routes = RouteFinder(founder_breeds=["Angus", "Brahman"]).find_routes(TARGET, max_depth=max_route_depth)
        if not routes:
            raise RuntimeError("RouteFinder failed to recover Brangus 5/8 : 3/8 ancestry routes.")
        selected_route = routes[route_index]

        # Step 2: trait architecture, h2 calibrated on a stated founder reference population
        simulator, founder_freqs = TraitSimulator.create_brangus_architecture(genome, seed=self.seed)
        ref_pop = Population(genome)
        reference = [
            ref_pop.create_founder(f"REF_{b}_{i}", sex="M" if i % 2 else "F", breed=b, rng=rng, founder_allele_frequencies=founder_freqs)
            for b in ("Angus", "Brahman") for i in range(reference_size_per_breed)
        ]
        calibration = simulator.calibrate_residuals(
            reference, genome, reference_label=f"within-breed Angus and Brahman founders (n={len(reference)})",
            group_by_breed=True
        )

        # Step 3: execute the selected route
        pop = Population(genome)
        ped = Pedigree()
        executor = RouteExecutor(genome, pop, ped, founder_allele_frequencies=founder_freqs, id_prefix="BR")
        sire_side, dam_side_node = RouteExecutor.split_final_cross(selected_route.root, dam_side)

        bulls = [executor.realize(sire_side, "M", rng, family=f"S{f}_") for f in range(n_founder_families)]
        brangus_cohort: List[SimulatedIndividual] = []
        for k in range(generation_size):
            cow = executor.realize(dam_side_node, "F", rng, family=f"D{k}_")
            sire = bulls[k % n_founder_families]
            sex = "M" if rng.random() < 0.5 else "F"
            calf = MeiosisSimulator.mate(sire, cow, f"Brangus_G0_{k}", sex, genome, rng)
            pop.add(calf)
            ped.add_cross(calf.id, sire_id=sire.id, dam_id=cow.id, sex=sex)
            brangus_cohort.append(calf)

        formation_exact = all(ped.individuals[c.id].breed_fraction.matches(TARGET) for c in brangus_cohort)

        realized_angus = np.array([ind.realized_genomic_ancestry("Angus", genome) for ind in brangus_cohort])
        anc_mean = float(np.mean(realized_angus))
        anc_std = float(np.std(realized_angus))
        anc_quantiles = {
            "q05": float(np.quantile(realized_angus, 0.05)),
            "q25": float(np.quantile(realized_angus, 0.25)),
            "q50": float(np.median(realized_angus)),
            "q75": float(np.quantile(realized_angus, 0.75)),
            "q95": float(np.quantile(realized_angus, 0.95)),
        }

        # Step 4: closed inter se breeding with selection on a standardized two-trait index
        progression_records = []
        current_cohort = brangus_cohort
        for g in range(post_formation_generations + 1):
            df_curr = simulator.simulate_records(current_cohort, genome, rng)
            angus_fracs = [ind.realized_genomic_ancestry("Angus", genome) for ind in current_cohort]
            progression_records.append({
                "generation": g,
                "cohort_size": len(current_cohort),
                "angus_genomic_mean": float(np.mean(angus_fracs)),
                "angus_genomic_std": float(np.std(angus_fracs)),
                "angus_pedigree_mean": float(np.mean([float(ind.pedigree_composition.get("Angus")) for ind in current_cohort])),
                "production_mean": float(np.mean(df_curr["phenotype_Production"])),
                "production_std": float(np.std(df_curr["phenotype_Production"])),
                "env_response_mean": float(np.mean(df_curr["phenotype_EnvironmentalResponse"])),
                "env_response_std": float(np.std(df_curr["phenotype_EnvironmentalResponse"])),
                "inbreeding_F": float(np.mean([ped.inbreeding(ind.id) for ind in current_cohort])),
                "heterozygosity_Ho": float(np.mean([ind.heterozygosity(genome) for ind in current_cohort])),
            })

            if g == post_formation_generations:
                break

            z_prod = (df_curr["phenotype_Production"] - df_curr["phenotype_Production"].mean()) / df_curr["phenotype_Production"].std()
            z_env = (df_curr["phenotype_EnvironmentalResponse"] - df_curr["phenotype_EnvironmentalResponse"].mean()) / df_curr["phenotype_EnvironmentalResponse"].std()
            index = dict(zip(df_curr["id"], z_prod + z_env))

            males = sorted((i for i in current_cohort if i.sex == "M"), key=lambda i: -index[i.id])
            females = sorted((i for i in current_cohort if i.sex == "F"), key=lambda i: -index[i.id])
            n_dams = min(len(females), math.ceil(generation_size / calves_per_dam))
            n_sires = min(len(males), max(2, round(male_selection_fraction * len(males))))
            if n_dams == 0 or n_sires == 0:
                raise RuntimeError(f"Generation {g} has no selectable {'females' if n_dams == 0 else 'males'}.")
            sel_females, sel_males = females[:n_dams], males[:n_sires]

            # When fewer females were born than needed, each selected dam carries more calves.
            dam_capacity = math.ceil(generation_size / n_dams)
            allocator = MatingAllocator(
                genome, pedigree=ped, max_inbreeding_threshold=max_inbreeding, dam_capacity=dam_capacity,
                max_sire_quota=math.ceil(n_dams * dam_capacity / n_sires)
            )
            plan = allocator.build_plan(BreedingPolicy.RANDOM, sel_males, sel_females, 0.0, 0.0, rng=rng)
            pairs = list(plan.pairs)
            rng.shuffle(pairs)
            if len(pairs) < generation_size:
                raise RuntimeError(
                    f"Generation {g}: pedigree constraint left only {len(pairs)} eligible matings for {generation_size} calves."
                )

            sire_map = {m.id: m for m in sel_males}
            dam_map = {f.id: f for f in sel_females}
            next_cohort = []
            for k, pair in enumerate(pairs[:generation_size]):
                sex = "M" if rng.random() < 0.5 else "F"
                child = MeiosisSimulator.mate(sire_map[pair.sire_id], dam_map[pair.dam_id], f"Brangus_G{g + 1}_{k}", sex, genome, rng)
                pop.add(child)
                ped.add_cross(child.id, sire_id=pair.sire_id, dam_id=pair.dam_id, sex=sex)
                next_cohort.append(child)
            current_cohort = next_cohort

        df_prog = pd.DataFrame(progression_records)
        first, last = df_prog.iloc[0], df_prog.iloc[-1]
        summary = (
            f"=== Brangus Case 1 Evaluation Summary ===\n"
            f"Discovered routes: {len(routes)} valid pathways up to {max_route_depth} generations (no hardcoding).\n"
            f"Executed route [{route_index}]: {selected_route.summary()}\n"
            f"Formation: {n_founder_families} bull lines x {generation_size} single-calf cows; "
            f"all calves exact 5/8 Angus by pedigree: {formation_exact}\n"
            f"Realized Angus ancestry at formation: mean = {anc_mean:.4f} (pedigree expectation 0.6250), sd = {anc_std:.4f}\n"
            f"90% interval: [{anc_quantiles['q05']:.4f}, {anc_quantiles['q95']:.4f}]\n"
            f"Trait persistence over {post_formation_generations} selected generations:\n"
            f"  Gen 0: Production = {first['production_mean']:.2f}, EnvResponse = {first['env_response_mean']:.2f}, Angus = {first['angus_genomic_mean']:.4f}\n"
            f"  Gen {int(last['generation'])}: Production = {last['production_mean']:.2f}, EnvResponse = {last['env_response_mean']:.2f}, Angus = {last['angus_genomic_mean']:.4f}\n"
            f"Pedigree F: {first['inbreeding_F']:.4f} -> {last['inbreeding_F']:.4f}; "
            f"Ho: {first['heterozygosity_Ho']:.3f} -> {last['heterozygosity_Ho']:.3f}\n"
        )

        return BrangusResults(
            routes_found=routes,
            route_used=selected_route,
            realized_ancestry_mean=anc_mean,
            realized_ancestry_std=anc_std,
            realized_ancestry_quantiles=anc_quantiles,
            generation_progression=df_prog,
            summary_report=summary,
            formation_pedigree_exact=formation_exact,
            calibration=calibration,
            formation_cohort=brangus_cohort,
            population=pop,
            pedigree=ped,
        )

    def run_replicates(self, n_replicates: int = 5, **run_kwargs) -> pd.DataFrame:
        """Repeat the case with independent seeds; one row per replicate with formation and final metrics."""
        rows = []
        for r in range(n_replicates):
            res = BrangusCaseStudy(self.genome_spec, seed=self.seed + 1000 * r).run(**run_kwargs)
            first, last = res.generation_progression.iloc[0], res.generation_progression.iloc[-1]
            rows.append({
                "replicate": r,
                "seed": self.seed + 1000 * r,
                "formation_angus_mean": res.realized_ancestry_mean,
                "formation_angus_sd": res.realized_ancestry_std,
                "final_angus_mean": last["angus_genomic_mean"],
                "production_gain": last["production_mean"] - first["production_mean"],
                "env_response_gain": last["env_response_mean"] - first["env_response_mean"],
                "final_inbreeding_F": last["inbreeding_F"],
                "final_heterozygosity": last["heterozygosity_Ho"],
            })
        return pd.DataFrame(rows)

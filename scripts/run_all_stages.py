"""Runs Stages A-F of the crossbreeding workplan end-to-end with explicit completion gates.

Each stage checks its workplan completion criterion and prints PASS or FAIL; the script exits with
status 1 if any gate fails. Tables and plots go to output/, founder haplotypes to output/founders/,
and seeds, configuration, software versions and gate results to output/run_manifest.json.
Everything here is simulation under assumed parameters; nothing is an empirical claim.
"""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ancestry.pedigree import Pedigree
from src.ancestry.route_finder import RouteFinder
from src.decision.joint_probability import JointProbabilityEstimator
from src.decision.mating_allocator import BreedingPolicy, MatingAllocator
from src.evaluation.brangus_case import BrangusCaseStudy
from src.evaluation.reproducibility import save_founders, write_manifest
from src.evaluation.wagyu_angus_case import POLICY_LABELS, POLICY_ORDER, WagyuAngusCaseStudy
from src.inheritance.alphasimr_bridge import AlphaSimRBridge
from src.inheritance.chromosome import GenomeSpec, Haplotype
from src.inheritance.meiosis import MeiosisSimulator, Population
from src.inheritance.route_executor import RouteExecutor
from src.prediction.cohort_benchmark import run_stage_d_benchmark
from src.prediction.models import AdditiveGBLUP
from src.traits.simulator import TraitSimulator

OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
FOUNDERS_DIR = OUTPUT_DIR / "founders"
TARGET = {"Angus": Fraction(5, 8), "Brahman": Fraction(3, 8)}
DOCUMENTED_ROUTES = [
    "(((Angus x Brahman) x Brahman) x Angus)",            # F1 x Brahman -> 25%, then x Angus
    "(((Angus x Brahman) x Angus) x (Angus x Brahman))",  # 75% x 50%
]
CONFIG = {
    "seed": 2026,
    "genome": {"num_chromosomes": 10, "markers_per_chrom": 30, "qtls_per_chrom": 6},
    "stage_b": {"gametes": 3000, "route_realizations": 300},
    "stage_c": {"reference_per_breed": 150, "independent_per_breed": 300, "f1": 100},
    "stage_d": {"seeds": list(range(8))},
    "stage_e": {"seeds": list(range(101, 109)), "n_sires": 8, "n_dams": 30, "sire_quota": 6, "fresh_draws_per_pair": 20},
    "brangus": {"seed": 123, "n_founder_families": 4, "post_formation_generations": 3, "generation_size": 40, "replicates": 5},
    "wagyu": {"seed": 999, "num_dams": 60, "generations": 4, "calves_per_dam": 2, "sire_quota": 12,
              "target_marbling": 6.5, "target_growth": 360.0, "generation_interval_years": 5.0, "n_replicates": 5},
}
GATES: list = []
OUTPUTS: list = []


def gate(stage: str, criterion: str, passed: bool, detail: str) -> None:
    GATES.append({"stage": stage, "criterion": criterion, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {criterion} -- {detail}")


def header(title: str) -> None:
    print("\n" + "=" * 88 + f"\n{title}\n" + "=" * 88)


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=index)
    OUTPUTS.append(str(path.relative_to(BASE_DIR)))


def save_fig(fig, name: str) -> None:
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    OUTPUTS.append(str(path.relative_to(BASE_DIR)))


def founder_set(genome, freqs, rng, n_per_breed, breeds, prefix):
    pop = Population(genome)
    return [pop.create_founder(f"{prefix}_{b}_{i}", sex="M" if i % 2 else "F", breed=b, rng=rng, founder_allele_frequencies=freqs)
            for b in breeds for i in range(n_per_breed)]


# --------------------------------------------------------------------------------------- plots

def plot_routes(routes, name: str) -> None:
    fig, axes = plt.subplots(1, len(routes), figsize=(6.5 * len(routes), 4.8))
    for ax, route in zip(np.atleast_1d(axes), routes):
        counter = [0]

        def draw(node):
            if node.is_founder:
                x = float(counter[0])
                counter[0] += 1
                label = node.founder_breed
            else:
                xa, ya = draw(node.parent_a)
                xb, yb = draw(node.parent_b)
                x = (xa + xb) / 2
                ax.plot([xa, x], [ya, node.depth], color="#888888", lw=1.2, zorder=1)
                ax.plot([xb, x], [yb, node.depth], color="#888888", lw=1.2, zorder=1)
                label = f"{node.composition.get('Angus')} Angus"
            colour = {"Angus": "#dbe9f6", "Brahman": "#fde2c8"}.get(label, "#e8e8e8")
            ax.text(x, node.depth, label, ha="center", va="center", fontsize=9, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.35", fc=colour, ec="#555555", lw=0.8))
            return x, node.depth

        draw(route.root)
        ax.set_title(f"{route.depth} generations, {route.num_founders} founders per animal", fontsize=10)
        ax.set_ylim(-0.6, route.depth + 0.6)
        ax.set_xlim(-0.8, counter[0] - 0.2)
        ax.axis("off")
    fig.suptitle("Stage A: recovered crossing routes to 5/8 Angus (founders at the bottom)", fontsize=11)
    save_fig(fig, name)


def plot_chromosome_ancestry(animals, genome, name: str) -> None:
    colours = {"Angus": "#1f77b4", "Brahman": "#ff7f0e"}
    offsets = np.concatenate([[0.0], np.cumsum([c.length_m for c in genome.chromosomes])])
    fig, ax = plt.subplots(figsize=(12, 0.9 * len(animals) + 1.2))
    for row, ind in enumerate(animals):
        for strand, haps in enumerate((ind.paternal_haplotypes, ind.maternal_haplotypes)):
            y = row * 3 + strand
            for c_idx, c in enumerate(genome.chromosomes):
                for seg in haps[c.chrom_id].segments:
                    ax.add_patch(Rectangle((offsets[c_idx] + seg.start_m, y), seg.length, 0.85,
                                           color=colours.get(seg.breed, "#999999"), lw=0))
        angus = ind.realized_genomic_ancestry("Angus", genome)
        ax.text(-0.3, row * 3 + 0.9, f"{ind.id}\n{angus:.1%} Angus", ha="right", va="center", fontsize=8)
    for off in offsets:
        ax.axvline(off, color="white", lw=2)
    ax.set_xlim(0, offsets[-1])
    ax.set_ylim(-0.5, len(animals) * 3)
    ax.set_yticks([])
    ax.set_xlabel("Genome position (Morgans; white lines separate autosomes). Upper bar paternal, lower maternal.")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=v) for v in colours.values()], labels=list(colours), loc="upper right", fontsize=8)
    ax.set_title("Stage B: chromosome ancestry of 5/8 Angus calves (pedigree expectation 62.5%)", fontsize=11)
    save_fig(fig, name)


# --------------------------------------------------------------------------------------- stages

def stage_a():
    header("STAGE A: ancestry engine -- exact fractions and route discovery")
    routes = RouteFinder(["Angus", "Brahman"]).find_routes(TARGET, max_depth=3)
    for r in routes:
        print("  " + r.summary().replace("\n", "\n    "))
    sigs = [r.signature for r in routes]
    found = [s in sigs for s in DOCUMENTED_ROUTES]
    gate("A", "Recover valid Brangus routes without hardcoding them", all(found),
         f"{len(routes)} distinct routes within 3 generations; documented routes found: {found}")
    ok = True
    for r in routes:
        ped = r.instantiate_pedigree("A")
        terminal = [i for i in ped.individuals.values() if i.breed_fraction.matches(TARGET)]
        ok &= bool(terminal) and all(ped.inbreeding(i.id) == 0.0 for i in terminal)
    gate("A", "Each route instantiates as a pedigree with exact 5/8 Angus and F = 0", ok, f"{len(routes)} routes")
    plot_routes(routes[:2], "stageA_routes.png")
    return routes


def stage_b(genome, routes, rng):
    header("STAGE B: inheritance engine -- recombination and ancestry tracking")
    spec = genome.chromosomes[0]
    pat = Haplotype.founder(spec, "P", "Angus", np.zeros(spec.num_markers, np.int8), np.zeros(spec.num_qtls, np.int8))
    mat = Haplotype.founder(spec, "M", "Brahman", np.ones(spec.num_markers, np.int8), np.ones(spec.num_qtls, np.int8))
    gametes = [MeiosisSimulator.recombine_chromosome(pat, mat, spec, rng) for _ in range(CONFIG["stage_b"]["gametes"])]
    k = np.array([len(g.segments) - 1 for g in gametes])
    se = np.sqrt(spec.length_m / len(k))
    gate("B", "Crossovers per chromosome follow Poisson(length in Morgans)",
         abs(k.mean() - spec.length_m) < 3 * se and abs(k.var() - spec.length_m) < 0.15,
         f"mean {k.mean():.3f}, variance {k.var():.3f}, expected {spec.length_m:.2f}")
    allele = lambda seg: 0 if seg.breed == "Angus" else 1
    phase_ok = all(g.markers[0] == allele(g.segments[0]) and g.markers[-1] == allele(g.segments[-1]) for g in gametes)
    gate("B", "Marker alleles follow their segment's founder at both chromosome ends", phase_ok, f"{len(gametes)} gametes")

    pop, ped = Population(genome), Pedigree()
    executor = RouteExecutor(genome, pop, ped, id_prefix="B")
    calves = [executor.realize(routes[0].root, "F", rng, family=f"f{i}_") for i in range(CONFIG["stage_b"]["route_realizations"])]
    anc = np.array([c.realized_genomic_ancestry("Angus", genome) for c in calves])
    exact = all(ped.individuals[c.id].breed_fraction.matches(TARGET) and ped.inbreeding(c.id) == 0.0 for c in calves)
    p_val = stats.ttest_1samp(anc, 0.625).pvalue
    gate("B", "Offspring inherit correctly; realized ancestry matches the pedigree expectation",
         exact and p_val > 0.01,
         f"{len(calves)} calves via route [0]: pedigree exact {exact}; genomic mean {anc.mean():.4f} "
         f"(expected 0.6250), sd {anc.std():.4f}, t-test p = {p_val:.3f}")
    bridge = AlphaSimRBridge()
    print(f"  [INFO] AlphaSimR cross-check: {'available' if bridge.is_available() else 'not run -- Rscript/AlphaSimR not installed; Python meiosis engine used'}")
    n = save_founders(pop, FOUNDERS_DIR / "stageB_route_founders.npz")
    OUTPUTS.append(f"output/founders/stageB_route_founders.npz ({n} founders)")
    plot_chromosome_ancestry(calves[:6], genome, "stageB_chromosome_ancestry.png")


def stage_c(genome, rng):
    header("STAGE C: trait simulation -- founder differences, dominance, epistasis, noise")
    cfg = CONFIG["stage_c"]
    sim, freqs = TraitSimulator.create_brangus_architecture(genome, seed=CONFIG["seed"])
    reference = founder_set(genome, freqs, rng, cfg["reference_per_breed"], ("Angus", "Brahman"), "REF")
    calib = sim.calibrate_residuals(reference, genome, f"within-breed Angus and Brahman founders (n={len(reference)})",
                                    group_by_breed=True)
    independent = founder_set(genome, freqs, rng, cfg["independent_per_breed"], ("Angus", "Brahman"), "IND")
    angus = [i for i in independent if i.id.startswith("IND_Angus")]
    brahman = [i for i in independent if i.id.startswith("IND_Brahman")]
    f1 = [MeiosisSimulator.mate(next(a for a in angus[2 * j:] if a.sex == "M"), next(b for b in brahman[2 * j:] if b.sex == "F"),
                                f"F1_{j}", "F", genome, rng) for j in range(cfg["f1"])]

    comps = pd.concat([
        sim.variance_components(reference, genome).assign(population="reference founders (calibration)"),
        sim.variance_components(independent, genome).assign(population="independent founder sample"),
        sim.variance_components(angus, genome).assign(population="Angus founders"),
        sim.variance_components(brahman, genome).assign(population="Brahman founders"),
        sim.variance_components(f1, genome).assign(population="F1 Angus x Brahman"),
    ])
    save_csv(comps, "stageC_variance_components.csv")
    print(comps[["population", "trait", "var_A", "var_D", "var_I", "var_E", "h2_realized", "h2_target"]].round(3).to_string(index=False))

    within = comps[comps["population"].isin(["Angus founders", "Brahman founders"])].groupby("trait")[["h2_realized", "h2_target"]].mean()
    gap = (within["h2_realized"] - within["h2_target"]).abs()
    gate("C", "Realized within-breed h2 matches the configured h2 in independent founder samples",
         bool((gap < 0.08).all()), ", ".join(f"{t}: {within.loc[t, 'h2_realized']:.3f} vs {within.loc[t, 'h2_target']:.2f}" for t in within.index))
    pooled = comps[comps["population"] == "independent founder sample"].set_index("trait")["h2_realized"]
    f1_h2 = comps[comps["population"] == "F1 Angus x Brahman"].set_index("trait")["h2_realized"]
    print("  [INFO] h2 in pooled two-breed founders (includes between-breed variance): "
          + ", ".join(f"{t} {v:.3f}" for t, v in pooled.items()) + "; in F1: " + ", ".join(f"{t} {v:.3f}" for t, v in f1_h2.items()))

    df_a, df_b = sim.simulate_records(angus, genome, rng), sim.simulate_records(brahman, genome, rng)
    p_prod = stats.ttest_ind(df_a["true_g_Production"], df_b["true_g_Production"], alternative="greater").pvalue
    p_env = stats.ttest_ind(df_b["true_g_EnvironmentalResponse"], df_a["true_g_EnvironmentalResponse"], alternative="greater").pvalue
    gate("C", "Configured founder differences are realized (Angus > Brahman production, Brahman > Angus adaptation)",
         p_prod < 1e-3 and p_env < 1e-3, f"one-sided p = {p_prod:.1e} and {p_env:.1e}")
    df_f1 = sim.simulate_records(f1, genome, rng)
    d_f1, d_par = df_f1["true_d_Production"].mean(), pd.concat([df_a, df_b])["true_d_Production"].mean()
    gate("C", "Directional dominance produces heterosis in the F1", d_f1 > d_par,
         f"mean dominance deviation F1 {d_f1:.2f} vs purebred founders {d_par:.2f}")
    save_csv(calib, "stageC_residual_calibration.csv")


def stage_d():
    header("STAGE D: prediction -- ancestry-only, GBLUP, additive+dominance, RKHS on held-out data")
    seeds = CONFIG["stage_d"]["seeds"]
    df = run_stage_d_benchmark(seeds)
    save_csv(df, "stageD_prediction_benchmark.csv")
    summary = df.groupby(["split", "model"])[["pearson_r", "spearman_rho", "rmse"]].agg(["mean", "std"]).round(3)
    print(summary.to_string())
    complete = np.isfinite(df[["pearson_r", "spearman_rho", "rmse"]].to_numpy()).all() and df["model"].nunique() == 4
    gate("D", "All four predictors measured on withheld families and a later cohort", complete,
         f"{len(seeds)} seeds; mean test n: " + ", ".join(f"{s} {int(g['n_test'].mean())}" for s, g in df.groupby("split")))
    r = df.groupby(["split", "model"])["pearson_r"].mean()
    gate("D", "Genomic prediction carries signal on withheld families", r[("withheld_family", "AdditiveGBLUP")] > 0.0,
         f"mean r: GBLUP {r[('withheld_family', 'AdditiveGBLUP')]:.3f}, RKHS {r[('withheld_family', 'NonlinearRKHS')]:.3f}, "
         f"ancestry-only {r[('withheld_family', 'AncestryOnly')]:.3f}")
    print(f"  [INFO] Later backcross cohort: mean r GBLUP {r[('later_cohort', 'AdditiveGBLUP')]:.3f}, "
          f"ancestry-only {r[('later_cohort', 'AncestryOnly')]:.3f} -- within-cross prediction is weak at this training size.")


def stage_e_seed(genome, seed: int, cfg) -> list:
    """One Stage E replicate: train on founders, build the four plans, evaluate each on fresh draws."""
    rng = np.random.default_rng(seed)
    sim, freqs = TraitSimulator.create_brangus_architecture(genome, seed=seed)
    sim.calibrate_residuals(founder_set(genome, freqs, rng, 100, ("Angus", "Brahman"), "REF"), genome,
                            "within-breed founders (n=200)", group_by_breed=True)
    pop = Population(genome)
    half_s, half_d = cfg["n_sires"] // 2, cfg["n_dams"] // 2
    sires = [pop.create_founder(f"Sire_{i}", "M", "Angus" if i < half_s else "Brahman", rng, freqs) for i in range(cfg["n_sires"])]
    dams = [pop.create_founder(f"Dam_{j}", "F", "Angus" if j < half_d else "Brahman", rng, freqs) for j in range(cfg["n_dams"])]
    train = sim.simulate_records(sires + dams, genome, rng)
    markers = np.vstack([i.get_marker_genotypes(genome) for i in sires + dams])
    y1, y2 = train["phenotype_Production"].values, train["phenotype_EnvironmentalResponse"].values
    p1, p2 = AdditiveGBLUP().fit(markers, y1), AdditiveGBLUP().fit(markers, y2)
    res_cov = JointProbabilityEstimator.out_of_fold_residual_covariance(markers, y1, y2, AdditiveGBLUP)
    t1, t2 = float(np.median(y1)), float(np.median(y2))

    allocator = MatingAllocator(genome, max_sire_quota=cfg["sire_quota"])
    model = dict(predictor_trait1=p1, predictor_trait2=p2, residual_cov=res_cov)
    policies = {
        "Random mating": (BreedingPolicy.RANDOM, {}),
        "Prescribed route (complementary breeds)": (BreedingPolicy.PRESCRIBED_ROUTE, {"prescribed_breed_map": {"Angus": "Brahman", "Brahman": "Angus"}}),
        "Additive selection": (BreedingPolicy.ADDITIVE_SELECTION, {}),
        "Offspring-aware optimisation": (BreedingPolicy.OFFSPRING_AWARE, {}),
    }
    smap, dmap = {s.id: s for s in sires}, {d.id: d for d in dams}
    rows = []
    for label, (policy, extra) in policies.items():
        plan = allocator.build_plan(policy, sires, dams, t1, t2, rng=rng, **model, **extra)
        # Same fresh-draw seed for every policy (common random numbers within a replicate)
        plan = allocator.evaluate_plan_fresh_draws(plan, smap, dmap, sim, t1, t2, "Production", "EnvironmentalResponse",
                                                   n_draws_per_pair=cfg["fresh_draws_per_pair"], seed=seed + 999)
        usage = pd.Series([p.sire_id for p in plan.pairs]).value_counts()
        rows.append({
            "seed": seed, "policy": label, "matings": len(plan.pairs), "unfilled_dam_slots": plan.unassigned_dam_slots,
            "sires_used": len(usage), "max_sire_use": int(usage.max()),
            "model_expected_attainment": round(plan.expected_attainment, 4),
            "fresh_draw_attainment": round(plan.fresh_draw_attainment, 4),
            "fresh_ci95_low": round(plan.fresh_draw_ci95[0], 4), "fresh_ci95_high": round(plan.fresh_draw_ci95[1], 4),
            "production_mean": round(plan.realized_traits_mean["Production"], 2),
            "env_response_mean": round(plan.realized_traits_mean["EnvironmentalResponse"], 2),
        })
    return rows


def stage_e(genome):
    header("STAGE E: decision search -- policies compared under equal resources")
    cfg = CONFIG["stage_e"]
    table = pd.DataFrame([row for seed in cfg["seeds"] for row in stage_e_seed(genome, seed, cfg)])
    save_csv(table, "stageE_policy_comparison.csv")
    summary = table.groupby("policy", sort=False)[["model_expected_attainment", "fresh_draw_attainment"]].agg(["mean", "std"]).round(4)
    print(f"  Targets: both traits at or above the training medians; {len(cfg['seeds'])} replicate populations")
    print(summary.to_string())
    equal = (table["matings"] == cfg["n_dams"]).all() and (table["max_sire_use"] <= cfg["sire_quota"]).all()
    gate("E", "Equal resources: every plan uses each dam once within the same sire quota", bool(equal),
         f"{cfg['n_dams']} dams, quota {cfg['sire_quota']}, {len(cfg['seeds'])} replicates")

    wide = table.pivot(index="seed", columns="policy", values="fresh_draw_attainment")
    n = len(wide)
    t_crit = stats.t.ppf(0.975, n - 1)

    def paired(policy):
        diff = wide[policy] - wide["Random mating"]
        half = t_crit * diff.std(ddof=1) / np.sqrt(n)
        return diff.mean(), diff.mean() - half, diff.mean() + half

    for policy in ("Prescribed route (complementary breeds)", "Additive selection"):
        m, lo, hi = paired(policy)
        print(f"  [INFO] {policy} minus random: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    m, lo, hi = paired("Offspring-aware optimisation")
    gate("E", "Offspring-aware allocation improves joint attainment over random mating (paired, fresh draws)",
         lo > 0, f"mean difference {m:+.3f}, 95% t-interval [{lo:+.3f}, {hi:+.3f}] over {n} replicates; "
                 "plans are the best found by exact assignment under the stated search budget")


def stage_f(genome):
    header("STAGE F (Case 1): Brangus route execution and trait persistence")
    b = CONFIG["brangus"]
    study = BrangusCaseStudy(genome_spec=genome, seed=b["seed"])
    run_kw = dict(n_founder_families=b["n_founder_families"], post_formation_generations=b["post_formation_generations"],
                  generation_size=b["generation_size"])
    res = study.run(**run_kw)
    print(res.summary_report)
    save_csv(res.generation_progression, "case1_brangus_progression.csv")
    reps = study.run_replicates(b["replicates"], **run_kw)
    save_csv(reps, "case1_brangus_replicates.csv")
    print(reps.round(4).to_string(index=False))
    n = save_founders(res.population, FOUNDERS_DIR / "case1_brangus_founders.npz")
    OUTPUTS.append(f"output/founders/case1_brangus_founders.npz ({n} founders)")
    gate("F", "Case 1 executes the recovered route with exact 5/8 pedigree and zero formation inbreeding",
         res.formation_pedigree_exact, res.route_used.signature)
    gate("F", "Case 1 realized formation ancestry centres on 62.5% across replicates",
         abs(reps["formation_angus_mean"].mean() - 0.625) < 0.02,
         f"mean {reps['formation_angus_mean'].mean():.4f} +/- {reps['formation_angus_mean'].std():.4f} (sd across {len(reps)} replicates)")
    print(f"  [INFO] After {b['post_formation_generations']} selected generations (mean +/- sd over replicates): "
          f"production gain {reps['production_gain'].mean():.2f} +/- {reps['production_gain'].std():.2f}, "
          f"adaptation gain {reps['env_response_gain'].mean():.2f} +/- {reps['env_response_gain'].std():.2f}, "
          f"Angus {reps['final_angus_mean'].mean():.3f}, F {reps['final_inbreeding_F'].mean():.3f}")

    df = res.generation_progression
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(df["generation"], df["production_mean"], yerr=df["production_std"], fmt="o-", capsize=3, label="Production")
    axes[0].errorbar(df["generation"], df["env_response_mean"], yerr=df["env_response_std"], fmt="s-", capsize=3, label="Environmental response")
    axes[0].set(xlabel="Generation after formation", ylabel="Phenotype mean (+/- sd)", title="Brangus trait trajectories under selection")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(df["generation"], df["angus_genomic_mean"] * 100, yerr=df["angus_genomic_std"] * 100, fmt="d-", capsize=3, color="purple", label="Realized genomic Angus %")
    axes[1].axhline(62.5, color="red", ls="--", label="Pedigree 62.5%")
    axes[1].set(xlabel="Generation after formation", ylabel="Angus ancestry (%)", title="Ancestry composition", ylim=(50, 75))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    save_fig(fig, "case1_brangus_trends.png")

    header("STAGE F (Case 2): Wagyu x Angus policies under equal resources")
    w = CONFIG["wagyu"]
    wres = WagyuAngusCaseStudy(genome_spec=genome, seed=w["seed"]).run(
        num_dams=w["num_dams"], target_marbling=w["target_marbling"], target_growth=w["target_growth"],
        generation_interval_years=w["generation_interval_years"], generations=w["generations"],
        calves_per_dam=w["calves_per_dam"], sire_quota=w["sire_quota"], n_replicates=w["n_replicates"],
    )
    print(wres.narrative_report)
    for key, p in wres.policies.items():
        print(f"  {p.policy_name}: " + " | ".join(p.failure_cases))
    save_csv(wres.summary_table, "case2_wagyu_angus_policies.csv")
    save_csv(wres.trajectory, "case2_wagyu_angus_trajectory.csv")
    save_csv(wres.replicate_table, "case2_wagyu_angus_replicates.csv")
    save_csv(wres.paired_comparisons, "case2_wagyu_angus_paired_comparisons.csv")
    expected = {"F1 (Wagyu x Angus)": 0.5, "BC1": 0.75, "BC2": 0.875, "BC3": 0.9375, "F1 x F1": 0.5}
    gate("F", "Case 2 pedigree checks (F1 50%, BC1 75%, BC2 87.5%, BC3 93.75%, F1 x F1 50%)",
         all(abs(wres.pedigree_checks.get(k, -1) - v) < 1e-12 for k, v in expected.items()),
         ", ".join(f"{k} {v:.4f}" for k, v in wres.pedigree_checks.items()))
    slots = {p.dam_slots_per_generation for p in wres.policies.values()}
    gate("F", "Case 2 policies share identical dam slots, calves per dam and sire quota", len(slots) == 1,
         f"{slots.pop()} dams x {w['calves_per_dam']} calves, quota {w['sire_quota']}, {w['n_replicates']} replicates")
    gate("F", "Case 2 predictors never use a candidate parent's own carcass record",
         bool((wres.replicate_table["training_parent_overlap"] == 0).all()), "training steers vs parents overlap = 0")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = [POLICY_LABELS[k] for k in POLICY_ORDER]
    att = np.array([wres.policies[k].target_attainment for k in POLICY_ORDER]) * 100
    lo = np.array([wres.policies[k].target_attainment_ci95[0] for k in POLICY_ORDER]) * 100
    hi = np.array([wres.policies[k].target_attainment_ci95[1] for k in POLICY_ORDER]) * 100
    ax.bar(range(len(labels)), att, yerr=[att - lo, hi - att], capsize=5, color=["#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#C44E52"], width=0.6)
    for i, v in enumerate(att):
        ax.annotate(f"{v:.1f}%", (i, hi[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split(". ", 1)[1] for l in labels], rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(f"Joint attainment, generation {w['generations']} calves (%)")
    ax.set_title(f"Wagyu x Angus: marbling >= {w['target_marbling']} and growth >= {w['target_growth']:.0f} kg "
                 f"(mean and 95% CI over {w['n_replicates']} replicates)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "case2_wagyu_policy_comparison.png")


def main() -> int:
    for d in (OUTPUT_DIR, PLOTS_DIR, FOUNDERS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])
    genome = GenomeSpec.default_bovine(**CONFIG["genome"])
    print(f"Genome: {len(genome.chromosomes)} autosomes, {genome.total_length_m:.2f} M, "
          f"{genome.total_markers} markers, {genome.total_qtls} QTL. All results are simulation under assumed parameters.")

    routes = stage_a()
    stage_b(genome, routes, rng)
    stage_c(genome, rng)
    stage_d()
    stage_e(genome)
    stage_f(genome)

    write_manifest(OUTPUT_DIR / "run_manifest.json", CONFIG, GATES, OUTPUTS)
    failed = [g for g in GATES if not g["passed"]]
    header(f"SUMMARY: {len(GATES) - len(failed)}/{len(GATES)} completion gates passed")
    for g in failed:
        print(f"  FAILED [{g['stage']}] {g['criterion']}: {g['detail']}")
    print(f"  Manifest: {(OUTPUT_DIR / 'run_manifest.json').relative_to(BASE_DIR)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

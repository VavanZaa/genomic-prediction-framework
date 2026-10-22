"""Summarize the developmental H7-v3 G8-to-G11 crossover increment.

The statistic is (mean at G11 - mean at G8) / 3.  Unlike the v1/v2 G11:G8 ratio,
this subtraction cancels an additive founder-mosaic baseline.  Results remain post-hoc and
developmental because the Svenson data were already inspected and qtl2 uses ``ngen`` in its HMM.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/layer7v3"
SEED = 20260913
N_BOOT = 4000
GENERATIONS = (8, 11)


def generation_increment(means, generations=GENERATIONS):
    """Change per generation; subtraction removes a constant initialization offset."""
    g0, g1 = generations
    return (float(means[g1]) - float(means[g0])) / (g1 - g0)


def load_reconstruction(directory):
    summary = pd.read_csv(directory / "crossovers.csv", dtype={"id": str, "chr": str})
    draws = pd.read_csv(directory / "posterior_draws.csv.gz", dtype={"id": str, "chr": str})
    required_summary = {"id", "chr", "ngen", "viterbi_xo", "posterior_mean_xo"}
    required_draws = {"id", "chr", "ngen", "draw", "xo"}
    if not required_summary.issubset(summary) or not required_draws.issubset(draws):
        raise ValueError(f"invalid v3 reconstruction in {directory}")
    if set(summary["ngen"].astype(int)) != set(GENERATIONS):
        raise ValueError(f"{directory}: expected generation labels {GENERATIONS}")
    vit = summary.groupby(["id", "ngen"], as_index=False)["viterbi_xo"].sum()
    post = draws.groupby(["id", "ngen", "draw"], as_index=False)["xo"].sum()
    chromosomes = sorted(summary["chr"].astype(str).unique(), key=lambda x: int(x))
    return vit, post, chromosomes


def _means(frame, value):
    return {int(g): float(x) for g, x in frame.groupby("ngen")[value].mean().items()}


def summarize_reconstruction(directory, rng, n_boot=N_BOOT):
    vit, post, chromosomes = load_reconstruction(directory)
    vit_means = _means(vit, "viterbi_xo")
    draw_means = post.groupby(["draw", "ngen"])["xo"].mean().unstack("ngen")
    posterior_means = {g: float(draw_means[g].mean()) for g in GENERATIONS}

    posterior_by_generation = {
        g: post.loc[post["ngen"] == g].pivot(index="id", columns="draw", values="xo").to_numpy(float)
        for g in GENERATIONS
    }
    vit_by_generation = {
        g: vit.loc[vit["ngen"] == g, "viterbi_xo"].to_numpy(float) for g in GENERATIONS
    }
    n_draws = posterior_by_generation[GENERATIONS[0]].shape[1]
    boot_vit = np.empty(n_boot)
    boot_post = np.empty(n_boot)
    for b in range(n_boot):
        sampled_vit, sampled_post = {}, {}
        draw = int(rng.integers(n_draws))
        for g in GENERATIONS:
            v = vit_by_generation[g]
            p = posterior_by_generation[g]
            idx = rng.integers(len(v), size=len(v))
            sampled_vit[g] = v[idx].mean()
            sampled_post[g] = p[idx, draw].mean()
        boot_vit[b] = generation_increment(sampled_vit)
        boot_post[b] = generation_increment(sampled_post)

    return {
        "chromosomes": chromosomes,
        "n_individuals": {str(g): int(len(vit_by_generation[g])) for g in GENERATIONS},
        "posterior_draws": int(n_draws),
        "viterbi_mean_crossovers": {str(g): round(vit_means[g], 5) for g in GENERATIONS},
        "posterior_mean_crossovers": {str(g): round(posterior_means[g], 5) for g in GENERATIONS},
        "viterbi_increment_per_generation": round(generation_increment(vit_means), 5),
        "posterior_increment_per_generation": round(generation_increment(posterior_means), 5),
        "viterbi_increment_ci95": np.quantile(boot_vit, [0.025, 0.975]).round(5).tolist(),
        "posterior_increment_ci95": np.quantile(boot_post, [0.025, 0.975]).round(5).tolist(),
    }, boot_vit, boot_post


def chromosome_increments(directory, cohort):
    """Development diagnostic showing whether a whole-genome difference is broadly distributed."""
    frame = pd.read_csv(directory / "crossovers.csv", dtype={"id": str, "chr": str})
    rows = []
    for method in ("viterbi_xo", "posterior_mean_xo"):
        means = frame.groupby(["chr", "ngen"])[method].mean().unstack("ngen")
        for chrom, values in means.iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "method": method.removesuffix("_xo"),
                    "chr": str(chrom),
                    "increment_per_generation": (float(values[11]) - float(values[8])) / 3,
                }
            )
    return pd.DataFrame(rows)


def compare(observed, painted, rng, n_boot=N_BOOT):
    obs_summary, obs_vit, obs_post = summarize_reconstruction(observed, rng, n_boot)
    pnt_summary, pnt_vit, pnt_post = summarize_reconstruction(painted, rng, n_boot)
    if obs_summary["chromosomes"] != pnt_summary["chromosomes"]:
        raise ValueError("observed and painted reconstructions cover different chromosomes")
    diff_vit = obs_vit - pnt_vit
    diff_post = obs_post - pnt_post
    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_status": "DEVELOPMENTAL_POST_HOC",
        "H7_status": "UNRESOLVED",
        "estimand": "increase in qtl2-inferred crossovers per outbreeding generation from G8 to G11",
        "why_difference": "subtraction cancels an additive seed-mosaic baseline; the old G11/G8 ratio does not",
        "observed": obs_summary,
        "painted": pnt_summary,
        "observed_minus_painted_increment": {
            "viterbi": round(obs_summary["viterbi_increment_per_generation"] - pnt_summary["viterbi_increment_per_generation"], 5),
            "viterbi_bootstrap_ci95": np.quantile(diff_vit, [0.025, 0.975]).round(5).tolist(),
            "posterior": round(obs_summary["posterior_increment_per_generation"] - pnt_summary["posterior_increment_per_generation"], 5),
            "posterior_bootstrap_ci95": np.quantile(diff_post, [0.025, 0.975]).round(5).tolist(),
        },
        "decision_rule": None,
        "interpretation_limits": [
            "The Svenson genotypes were already examined, so these numbers cannot become a new confirmatory H7 pass/fail test.",
            "The qtl2 DO HMM uses ngen in its transition prior, so this is posterior-predictive compatibility rather than an independent test of inheritance.",
            "The historical 144 partially inbred CC seed mosaics are unavailable; an idealized funnel is only a sensitivity model.",
            "No phenotype is present, so G_genomic and cross-species phenotype validation remain unavailable.",
        ],
    }
    return result, pd.DataFrame({
        "replicate": np.arange(1, n_boot + 1),
        "viterbi_observed_minus_painted": diff_vit,
        "posterior_observed_minus_painted": diff_post,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, default=DEFAULT_OUT / "observed_native")
    parser.add_argument("--painted", type=Path, default=DEFAULT_OUT / "painted_funnel_native")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--painted-truth",
        type=Path,
        default=DEFAULT_OUT / "painted_funnel_cross/simulation_truth.csv",
        help="optional exact founder-state counts saved by the v3 painter",
    )
    parser.add_argument(
        "--direct-truth",
        type=Path,
        default=DEFAULT_OUT / "painted_direct_cross/simulation_truth.csv",
        help="optional direct-founder exact-count sensitivity",
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    rng = np.random.default_rng(SEED)
    result, boot = compare(args.observed.resolve(), args.painted.resolve(), rng, args.bootstrap)
    per_chr = pd.concat(
        [chromosome_increments(args.observed.resolve(), "observed"),
         chromosome_increments(args.painted.resolve(), "painted")],
        ignore_index=True,
    )
    wide = per_chr.pivot(index=["method", "chr"], columns="cohort", values="increment_per_generation")
    wide["observed_minus_painted"] = wide["observed"] - wide["painted"]
    result["chromosome_consistency"] = {
        method: {
            "observed_increment_greater_count": int((sub["observed_minus_painted"] > 0).sum()),
            "chromosomes": int(len(sub)),
            "median_observed_minus_painted": round(float(sub["observed_minus_painted"].median()), 5),
        }
        for method, sub in wide.groupby(level="method")
    }
    if args.painted_truth.exists():
        truth = pd.read_csv(args.painted_truth, dtype={"id": str})
        required = {"id", "ngen", "exact_founder_state_switches"}
        if not required.issubset(truth):
            raise ValueError(f"invalid simulation truth file: {args.painted_truth}")
        means = _means(truth, "exact_founder_state_switches")
        result["painted_simulation_truth"] = {
            "quantity": "exact founder-label switches in both simulated haplotypes",
            "mean": {str(g): round(means[g], 5) for g in GENERATIONS},
            "increment_per_generation": round(generation_increment(means), 5),
            "note": "This counts observable founder-state changes, not crossovers that return to the same founder label.",
        }
        truth_increment = result["painted_simulation_truth"]["increment_per_generation"]
        result["painted_reconstruction_minus_truth_increment"] = {
            "viterbi": round(result["painted"]["viterbi_increment_per_generation"] - truth_increment, 5),
            "posterior": round(result["painted"]["posterior_increment_per_generation"] - truth_increment, 5),
            "purpose": "checks whether the repaired qtl2 measurement recovers the simulated founder-state accumulation",
        }
    if args.direct_truth.exists():
        direct = pd.read_csv(args.direct_truth, dtype={"id": str})
        direct_means = _means(direct, "exact_founder_state_switches")
        direct_increment = generation_increment(direct_means)
        result["initialization_sensitivity"] = {
            "direct_founder_exact_mean": {str(g): round(direct_means[g], 5) for g in GENERATIONS},
            "direct_founder_increment_per_generation": round(direct_increment, 5),
            "idealized_funnel_increment_per_generation": result.get("painted_simulation_truth", {}).get(
                "increment_per_generation"
            ),
            "interpretation": "Initialization changes the absolute mosaic baseline much more than the G8-to-G11 increment.",
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "development_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    boot.to_csv(args.out / "increment_bootstrap.csv", index=False)
    wide.reset_index().sort_values(["method", "chr"], key=lambda s: s.map(int) if s.name == "chr" else s).to_csv(
        args.out / "per_chromosome_increment.csv", index=False
    )
    rows = []
    for cohort in ("observed", "painted"):
        for generation in GENERATIONS:
            rows.append({
                "cohort": cohort,
                "ngen": generation,
                "viterbi_mean_crossovers": result[cohort]["viterbi_mean_crossovers"][str(generation)],
                "posterior_mean_crossovers": result[cohort]["posterior_mean_crossovers"][str(generation)],
            })
    pd.DataFrame(rows).to_csv(args.out / "cohort_summary.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

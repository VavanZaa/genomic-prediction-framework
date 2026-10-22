"""Layer 7 / H7: does the meiosis simulation reproduce a real multiparental population?

Pre-registered in docs/LAYER7_H7_PROTOCOL.md. Two stages:

  --stage simulate  run the unmodified meiosis engine on the Diversity Outbred design (8 founders,
                    G generations, mouse genetic map) and write the simulated mosaic statistics.
                    Touches no observed mouse data beyond map lengths, founder count and G.
  --stage unblind   verify the H7 lock, read the qtl2 reconstruction of the observed mosaics, and
                    evaluate the pre-registered endpoints once.

Nothing here tunes the simulator to mouse data. No cattle threshold, panel, model or result is
touched by this layer, whichever way it falls.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402
from src.inheritance.chromosome import ChromosomeSpec, GenomeSpec  # noqa: E402
from src.inheritance.meiosis import MeiosisSimulator, Population  # noqa: E402

MOUSE = ROOT / "data/real_mice/svenson"
OUT = ROOT / "output/layer7"
LOCK = OUT / "PROTOCOL_LOCK_H7.json"

FOUNDERS = list("ABCDEFGH")
N_PAIRS = 100                 # breeding pairs per generation
GENERATIONS = [8, 11]
N_REPLICATES = 25
N_SAMPLED = 200               # animals kept from each replicate's final generation
SEED = 20260912
MIN_SEGMENT_M = 0.02          # 2 cM: below the reconstruction's resolution, excluded both sides
E3_TOL, E1_TOL, E2_TOL = 0.15, 0.20, 0.20


def mouse_genome() -> GenomeSpec:
    """19 autosomes with Morgan lengths from the MegaMUGA genetic map. X is excluded."""
    chroms = []
    for c in range(1, 20):
        g = pd.read_csv(MOUSE / f"MM/MM_gmap{c}.csv", comment="#")
        pos = g["pos"].astype(float)
        chroms.append(ChromosomeSpec(chrom_id=c, length_m=float((pos.max() - pos.min()) / 100.0),
                                     num_markers=2, num_qtls=1))  # markers unused: segments carry ancestry
    return GenomeSpec(chromosomes=chroms)


def merged_segments(hap):
    """Founder-state runs along one haplotype, merging adjacent segments of the same founder."""
    out = []
    for s in sorted(hap.segments, key=lambda x: x.start_m):
        if out and out[-1][0] == s.breed and abs(out[-1][2] - s.start_m) < 1e-12:
            out[-1][2] = s.end_m
        else:
            out.append([s.breed, s.start_m, s.end_m])
    return out


def simulate_generation(genome, g, rng):
    """One replicate DO-like population: 8 founders, g generations of random outcrossing."""
    pop = Population(genome)
    parents = []
    # Each of the 8 strains is represented by both sexes: assigning a sex per strain would let a
    # strain enter the population through one sex only, which the real DO funnel does not do.
    per_strain = max(2, 2 * (N_PAIRS // len(FOUNDERS)))
    for f in FOUNDERS:
        for j in range(per_strain):
            sex = "M" if j % 2 == 0 else "F"
            parents.append(pop.create_founder(f"F0_{f}_{j}_{sex}", sex, f, rng))
    for gen in range(1, g + 1):
        males = [p for p in parents if p.sex == "M"]
        females = [p for p in parents if p.sex == "F"]
        if not males or not females:
            raise SystemExit(f"generation {gen}: one sex exhausted")
        children = []
        for k in range(N_PAIRS):
            for _ in range(20):                      # resample rather than allow sib or self pairs
                sire = males[rng.integers(len(males))]
                dam = females[rng.integers(len(females))]
                same_sibs = (sire.sire_id is not None and sire.sire_id == dam.sire_id
                             and sire.dam_id == dam.dam_id)
                if sire.id != dam.id and not same_sibs:
                    break
            for child_no in range(2):                # two offspring keeps the sexes balanced
                sex = "M" if child_no == 0 else "F"
                children.append(MeiosisSimulator.mate(
                    sire, dam, f"G{gen}_{k}_{sex}", sex, genome, rng))
        parents = children
    return parents


def statistics_from_haplotypes(records):
    """Switches per haplotype and segment lengths, from a list of merged-segment lists."""
    switches, lengths = [], []
    for segs in records:
        switches.append(max(0, len(segs) - 1))
        lengths.extend(e - s for _, s, e in segs)
    lengths = np.array([x for x in lengths if x >= MIN_SEGMENT_M])
    return np.array(switches), lengths


def simulate():
    genome = mouse_genome()
    print(f"mouse genome: {len(genome.chromosomes)} autosomes, "
          f"{genome.total_length_m:.3f} Morgans")
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "evidence": "SIMULATION", "stage": "simulate",
              "genome": {"autosomes": len(genome.chromosomes),
                         "total_morgans": round(genome.total_length_m, 4)},
              "parameters": {"founders": len(FOUNDERS), "breeding_pairs": N_PAIRS,
                             "replicates": N_REPLICATES, "sampled_per_replicate": N_SAMPLED,
                             "seed": SEED, "min_segment_M": MIN_SEGMENT_M},
              "generations": {}}

    for g in GENERATIONS:
        rep_means, all_switches, all_lengths = [], [], []
        for r in range(N_REPLICATES):
            cohort = simulate_generation(genome, g, rng)
            if len(cohort) > N_SAMPLED:
                pick = rng.choice(len(cohort), N_SAMPLED, replace=False)
                cohort = [cohort[i] for i in pick]
            records = []
            for ind in cohort:
                for haps in (ind.paternal_haplotypes, ind.maternal_haplotypes):
                    # switches are counted per haplotype across the whole autosomal genome
                    per_hap = []
                    for c in genome.chromosomes:
                        per_hap.extend(merged_segments(haps[c.chrom_id]))
                    records.append(per_hap)
            sw, ln = statistics_from_haplotypes(records)
            # one genome-wide haplotype is 19 chromosomes, so subtract the 18 chromosome joins
            sw = sw - (len(genome.chromosomes) - 1)
            rep_means.append(float(sw.mean()))
            all_switches.append(sw)
            all_lengths.append(ln)
            print(f"  G={g} replicate {r+1}/{N_REPLICATES}: mean switches {sw.mean():.2f}", flush=True)
        sw = np.concatenate(all_switches)
        ln = np.concatenate(all_lengths)
        rep_means = np.array(rep_means)
        result["generations"][str(g)] = {
            "mean_switches_per_haplotype": round(float(sw.mean()), 4),
            "replicate_mean_ci95": [round(float(np.percentile(rep_means, 2.5)), 4),
                                    round(float(np.percentile(rep_means, 97.5)), 4)],
            "median_segment_M": round(float(np.median(ln)), 5),
            "p90_segment_M": round(float(np.percentile(ln, 90)), 5),
            "n_haplotypes": int(len(sw)), "n_segments": int(len(ln)),
        }
        print(f"G={g}: mean switches {sw.mean():.2f}, median segment {np.median(ln):.4f} M")
    (OUT / "simulated_mosaics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {(OUT / 'simulated_mosaics.json').relative_to(ROOT)}")


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the H7 protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def observed_statistics():
    """Observed switches per haplotype and segment lengths, from the qtl2 reconstruction."""
    seg = pd.read_csv(OUT / "observed_segments.csv")
    covar = pd.read_csv(MOUSE / "svenson_covar.csv", dtype={"id": str})
    ngen = dict(zip(covar["id"], covar["ngen"].astype(int)))
    seg["ngen"] = seg["id"].astype(str).map(ngen)
    if seg["ngen"].isna().any():
        raise SystemExit("some reconstructed animals have no generation number")
    out = {}
    for g, sub in seg.groupby("ngen"):
        per_hap = sub.groupby(["id", "hap"]).size()          # segments per haplotype, all chromosomes
        chroms = sub.groupby(["id", "hap"])["chr"].nunique()
        switches = (per_hap - chroms).to_numpy(dtype=float)  # subtract chromosome joins
        lengths = sub.loc[sub["length_M"] >= MIN_SEGMENT_M, "length_M"].to_numpy(dtype=float)
        out[int(g)] = {"switches": switches, "lengths": lengths,
                       "n_haplotypes": int(len(switches)), "n_segments": int(len(lengths))}
    return out


def unblind():
    verify_lock()
    sim = json.loads((OUT / "simulated_mosaics.json").read_text(encoding="utf-8"))
    obs = observed_statistics()
    missing = [g for g in GENERATIONS if g not in obs]
    if missing:
        raise SystemExit(f"no reconstructed animals at generation(s) {missing}")

    endpoints, per_gen = {}, {}
    for g in GENERATIONS:
        s, o = sim["generations"][str(g)], obs[g]
        obs_mean = float(o["switches"].mean())
        lo, hi = s["replicate_mean_ci95"]
        rel = abs(obs_mean - s["mean_switches_per_haplotype"]) / s["mean_switches_per_haplotype"]
        obs_med, obs_p90 = float(np.median(o["lengths"])), float(np.percentile(o["lengths"], 90))
        per_gen[str(g)] = {
            "observed_mean_switches": round(obs_mean, 4),
            "simulated_mean_switches": s["mean_switches_per_haplotype"],
            "simulated_ci95": [lo, hi],
            "relative_difference": round(rel, 4),
            "E1_pass": bool(lo <= obs_mean <= hi and rel <= E1_TOL),
            "observed_median_segment_M": round(obs_med, 5),
            "simulated_median_segment_M": s["median_segment_M"],
            "median_relative_difference": round(abs(obs_med - s["median_segment_M"]) / s["median_segment_M"], 4),
            "observed_p90_segment_M": round(obs_p90, 5),
            "simulated_p90_segment_M": s["p90_segment_M"],
            "p90_relative_difference": round(abs(obs_p90 - s["p90_segment_M"]) / s["p90_segment_M"], 4),
            "n_observed_haplotypes": o["n_haplotypes"], "n_observed_segments": o["n_segments"],
        }
        per_gen[str(g)]["E2_pass"] = bool(
            per_gen[str(g)]["median_relative_difference"] <= E2_TOL
            and per_gen[str(g)]["p90_relative_difference"] <= E2_TOL)

    r_obs = obs[11]["switches"].mean() / obs[8]["switches"].mean()
    r_sim = (sim["generations"]["11"]["mean_switches_per_haplotype"]
             / sim["generations"]["8"]["mean_switches_per_haplotype"])
    e3_rel = abs(r_obs / r_sim - 1.0)
    endpoints["E3_generation_scaling"] = {
        "observed_ratio_G11_over_G8": round(float(r_obs), 4),
        "simulated_ratio_G11_over_G8": round(float(r_sim), 4),
        "relative_difference": round(float(e3_rel), 4), "tolerance": E3_TOL,
        "pass": bool(e3_rel <= E3_TOL),
        "why_primary": ("a ratio cancels constant offsets from unmodelled design details (funnel "
                        "structure, sib-mating avoidance, selection history), isolating the "
                        "recombination process H7 is actually about"),
    }
    endpoints["E1_breakpoints_per_genome"] = {str(g): per_gen[str(g)]["E1_pass"] for g in GENERATIONS}
    endpoints["E2_segment_lengths"] = {str(g): per_gen[str(g)]["E2_pass"] for g in GENERATIONS}

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL+SIMULATION",
        "protocol": "docs/LAYER7_H7_PROTOCOL.md", "lock": str(LOCK.relative_to(ROOT)),
        "axis": "G_inheritance (cross-species)",
        "reconstruction": json.loads((OUT / "observed_summary.json").read_text(encoding="utf-8")),
        "per_generation": per_gen, "endpoints": endpoints,
        "primary_pass": endpoints["E3_generation_scaling"]["pass"],
        "evidence_labels": [
            "Cross-species inheritance test (G_inheritance): performed on genotypes only.",
            "Cross-species genomic test (G_genomic): not available - no phenotype in hand.",
            "Same-trait cross-species phenotype validation: not available.",
        ],
        "does_not_claim": ("A pass says the inheritance layer is species-general. It says nothing about "
                           "G_genomic or G_phenotype and must never be summarised as 'the framework "
                           "transfers to mice'."),
        "changes_nothing": "no cattle threshold, panel, model or result is modified by this layer",
    }
    (OUT / "results_h7.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(endpoints, indent=2))
    print(f"\nprimary endpoint E3: {'PASS' if result['primary_pass'] else 'FAIL'}")
    print(f"wrote {(OUT / 'results_h7.json').relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["simulate", "unblind"], required=True)
    args = ap.parse_args()
    simulate() if args.stage == "simulate" else unblind()

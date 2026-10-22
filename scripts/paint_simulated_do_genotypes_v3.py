"""Build the Layer 7 H7-v3 simulated cross with matched observation patterns.

This is a developmental repair of the H7 measurement, not a new confirmatory test.  It differs
from v2 in two consequential ways:

* an idealized eight-founder funnel is available before the requested DO outbreeding generations;
* every simulated animal receives the exact missing-call mask of one observed animal from the same
  generation, rather than independent missing calls drawn from a pooled rate.

The historical 144 partially inbred CC seed mosaics are not in the Svenson archive.  The funnel is
therefore a transparent sensitivity model and must not be described as the actual DO pedigree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paint_simulated_do_genotypes import (  # noqa: E402
    CHROMS,
    CODES,
    FOUNDERS,
    combine,
    founder_table,
    paint_haplotype,
)
from run_layer7_h7 import merged_segments, mouse_genome  # noqa: E402
from src.inheritance.meiosis import MeiosisSimulator, Population  # noqa: E402

MOUSE = ROOT / "data/real_mice/svenson"
DEFAULT_OUT = ROOT / "output/layer7v3/painted_funnel_cross"
SEED = 20260913
N_PAIRS = 175
ERROR_RATE = 0.0


def _mate(sire, dam, child_id, sex, genome, rng):
    return MeiosisSimulator.mate(sire, dam, child_id, sex, genome, rng)


def idealized_funnel_seeds(genome, n_seeds, rng):
    """Create independent balanced eight-way funnel animals.

    Each seed has pedigree contribution 1/8 from A--H.  Fresh founder copies are used for each
    seed so this sensitivity does not invent relationships among the unavailable CC seed lines.
    """
    pop = Population(genome)
    seeds = []
    for i in range(n_seeds):
        order = list(np.asarray(FOUNDERS)[rng.permutation(len(FOUNDERS))])
        f0 = []
        for j, founder in enumerate(order):
            sex = "M" if j % 2 == 0 else "F"
            f0.append(pop.create_founder(f"FUN{i:04d}_F0_{founder}_{sex}", sex, founder, rng))
        f1 = [
            _mate(f0[0], f0[1], f"FUN{i:04d}_F1a", "M", genome, rng),
            _mate(f0[2], f0[3], f"FUN{i:04d}_F1b", "F", genome, rng),
            _mate(f0[4], f0[5], f"FUN{i:04d}_F1c", "M", genome, rng),
            _mate(f0[6], f0[7], f"FUN{i:04d}_F1d", "F", genome, rng),
        ]
        f2a = _mate(f1[0], f1[1], f"FUN{i:04d}_F2a", "M", genome, rng)
        f2b = _mate(f1[2], f1[3], f"FUN{i:04d}_F2b", "F", genome, rng)
        seeds.append(_mate(f2a, f2b, f"FUN{i:04d}_SEED", "M" if i % 2 == 0 else "F", genome, rng))
    return seeds


def direct_founder_seeds(genome, n_seeds, rng):
    """Balanced pure-founder starting population retained as a v1-comparable sensitivity."""
    pop = Population(genome)
    return [
        pop.create_founder(
            f"DIRECT{i:04d}_{FOUNDERS[(i // 2) % len(FOUNDERS)]}",
            "M" if i % 2 == 0 else "F",
            FOUNDERS[(i // 2) % len(FOUNDERS)],
            rng,
        )
        for i in range(n_seeds)
    ]


def advance_outbreeding(parents, genome, generations, n_pairs, rng):
    """Random outbreeding with no self or full-sib matings and two children per pair."""
    for gen in range(1, generations + 1):
        males = [p for p in parents if p.sex == "M"]
        females = [p for p in parents if p.sex == "F"]
        children = []
        for pair in range(n_pairs):
            for _ in range(100):
                sire = males[int(rng.integers(len(males)))]
                dam = females[int(rng.integers(len(females)))]
                full_sibs = (
                    sire.sire_id is not None
                    and sire.sire_id == dam.sire_id
                    and sire.dam_id == dam.dam_id
                )
                if sire.id != dam.id and not full_sibs:
                    break
            else:
                raise RuntimeError(f"generation {gen}: could not find an eligible pair")
            children.extend(
                [
                    _mate(sire, dam, f"O{generations}_G{gen}_{pair:03d}_M", "M", genome, rng),
                    _mate(sire, dam, f"O{generations}_G{gen}_{pair:03d}_F", "F", genome, rng),
                ]
            )
        parents = children
    return parents


def simulate_generation_v3(genome, generation, rng, seed_mode, n_pairs=N_PAIRS):
    """Simulate one cohort after ``generation`` random-outbreeding generations."""
    n_seeds = 2 * n_pairs
    if seed_mode == "idealized_funnel":
        parents = idealized_funnel_seeds(genome, n_seeds, rng)
    elif seed_mode == "direct":
        parents = direct_founder_seeds(genome, n_seeds, rng)
    else:
        raise ValueError(f"unknown seed mode: {seed_mode}")
    return advance_outbreeding(parents, genome, generation, n_pairs, rng)


def founder_state_switches(individual, genome):
    """Exact founder-label changes retained in both simulated haplotypes."""
    total = 0
    for chrom in genome.chromosomes:
        for haplotypes in (individual.paternal_haplotypes, individual.maternal_haplotypes):
            total += max(0, len(merged_segments(haplotypes[chrom.chrom_id])) - 1)
    return total


def missing_call_mask(values):
    """Return the missing-call mask accepted by the Svenson qtl2 cross."""
    arr = np.asarray(values, dtype=object)
    as_text = np.char.upper(np.asarray(["" if x is None else str(x) for x in arr.ravel()]))
    mask = np.isin(as_text, ["", "-", "NA", "NAN", "NONE"])
    return mask.reshape(arr.shape)


def apply_observation_model(codes, observed_calls, rng, error_rate=ERROR_RATE):
    """Apply an observed animal's exact missing mask and optional declared assay error.

    ``codes`` contains indices into ``CODES``. Founder-uninformative simulated calls remain missing.
    The default error rate is zero because qtl2's 0.002 ``error_prob`` is an inference setting, not
    an estimate of the Svenson array's empirical error rate.
    """
    out = np.asarray(codes, dtype=np.int8).copy()
    observed_missing = missing_call_mask(observed_calls)
    informative = out != 3
    eligible = informative & ~observed_missing
    errors = eligible & (rng.random(out.shape) < error_rate)
    if errors.any():
        out[errors] = (out[errors] + rng.integers(1, 3, int(errors.sum()))) % 3
    out[observed_missing] = 3
    return out, {
        "matched_missing": int(observed_missing.sum()),
        "errors": int(errors.sum()),
        "founder_uninformative": int((~observed_missing & ~informative).sum()),
    }


def _observed_genotypes(chrom, ids, markers):
    frame = pd.read_csv(MOUSE / f"svenson_geno{chrom}.csv", comment="#", dtype=str).set_index("marker")
    frame = frame.reindex(index=markers, columns=ids)
    if frame.isna().all(axis=0).any():
        bad = frame.columns[frame.isna().all(axis=0)].tolist()
        raise RuntimeError(f"chr {chrom}: observed donor ids missing from genotype file: {bad[:5]}")
    return frame


def build_cross(out, seed_mode="idealized_funnel", n_pairs=N_PAIRS, error_rate=ERROR_RATE):
    rng = np.random.default_rng(SEED)
    genome = mouse_genome()
    covar = pd.read_csv(MOUSE / "svenson_covar.csv", dtype={"id": str})
    if set(covar["ngen"].astype(int)) != {8, 11}:
        raise RuntimeError("H7 v3 expects exactly the Svenson G8 and G11 cohorts")

    animals = []
    for generation, donors in covar.groupby("ngen", sort=True):
        cohort = simulate_generation_v3(genome, int(generation), rng, seed_mode, n_pairs)
        if len(cohort) < len(donors):
            raise RuntimeError(f"G{generation}: {len(cohort)} simulated animals for {len(donors)} donors")
        selected = rng.choice(len(cohort), len(donors), replace=False)
        for donor, idx in zip(donors.itertuples(index=False), selected):
            animals.append((str(donor.id), cohort[int(idx)], int(generation), str(donor.sex)))
        print(f"G={generation}: simulated {len(cohort)}, matched {len(donors)} observed masks", flush=True)

    # Preserve the source cross's individual order so each id maps transparently to its own mask.
    by_id = {record[0]: record for record in animals}
    animals = [by_id[str(i)] for i in covar["id"]]
    ids = [record[0] for record in animals]
    out.mkdir(parents=True, exist_ok=True)
    (out / "MM").mkdir(exist_ok=True)
    truth = pd.DataFrame(
        {
            "id": ids,
            "ngen": [record[2] for record in animals],
            "exact_founder_state_switches": [founder_state_switches(record[1], genome) for record in animals],
        }
    )
    truth.to_csv(out / "simulation_truth.csv", index=False)
    tables = {chrom: founder_table(chrom) for chrom in CHROMS}

    counters = {"cells": 0, "matched_missing": 0, "errors": 0, "founder_uninformative": 0, "called": 0}
    per_chrom = {}
    for chrom in CHROMS:
        markers, pos, alleles = tables[chrom]
        observed = _observed_genotypes(chrom, ids, markers)
        geno = np.empty((len(markers), len(animals)), dtype="<U1")
        chrom_counts = {key: 0 for key in counters}
        for j, (donor_id, ind, _, _) in enumerate(animals):
            h1 = paint_haplotype(merged_segments(ind.paternal_haplotypes[chrom]), pos, alleles)
            h2 = paint_haplotype(merged_segments(ind.maternal_haplotypes[chrom]), pos, alleles)
            code = combine(h1, h2)
            code, counts = apply_observation_model(code, observed[donor_id].to_numpy(), rng, error_rate)
            geno[:, j] = CODES[code]
            chrom_counts["cells"] += len(code)
            chrom_counts["called"] += int((code != 3).sum())
            for key, value in counts.items():
                chrom_counts[key] += value
        for key in counters:
            counters[key] += chrom_counts[key]
        per_chrom[str(chrom)] = {
            "cells": chrom_counts["cells"],
            "matched_missing_share": chrom_counts["matched_missing"] / chrom_counts["cells"],
            "called_share": chrom_counts["called"] / chrom_counts["cells"],
        }
        with open(out / f"painted_geno{chrom}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# H7 v3 {seed_mode}; observed missing masks matched by animal; chr {chrom}\n")
            pd.DataFrame(geno, index=markers, columns=ids).to_csv(fh, index_label="marker")
        for kind in ("foundergeno", "gmap", "pmap"):
            shutil.copy2(MOUSE / f"MM/MM_{kind}{chrom}.csv", out / f"MM/MM_{kind}{chrom}.csv")
        print(f"  chr {chrom}: {len(markers):,} markers", flush=True)

    covar.to_csv(out / "painted_covar.csv", index=False)
    spec = {
        "description": f"H7 v3 developmental painted simulation ({seed_mode})",
        "crosstype": "do",
        "sep": ",",
        "na.strings": ["-", "NA"],
        "comment.char": "#",
        "geno": [f"painted_geno{c}.csv" for c in CHROMS],
        "founder_geno": [f"MM/MM_foundergeno{c}.csv" for c in CHROMS],
        "gmap": [f"MM/MM_gmap{c}.csv" for c in CHROMS],
        "pmap": [f"MM/MM_pmap{c}.csv" for c in CHROMS],
        "covar": "painted_covar.csv",
        "genotypes": {"A": 1, "H": 2, "B": 3},
        "geno_transposed": True,
        "founder_geno_transposed": True,
        "sex": {"covar": "sex", "F": "Female", "M": "Male"},
        "cross_info": {"covar": "ngen"},
    }
    (out / "painted.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    summary = {
        "evidence": "SIMULATION",
        "analysis_status": "DEVELOPMENTAL",
        "seed": SEED,
        "seed_mode": seed_mode,
        "seed_model_limitation": (
            "The actual 144 partially inbred CC seed mosaics are unavailable; idealized_funnel is a sensitivity model."
            if seed_mode == "idealized_funnel"
            else "Direct pure-founder start retained only as a v1-comparable sensitivity."
        ),
        "breeding_pairs": n_pairs,
        "cohort": {str(int(g)): int(n) for g, n in covar.groupby("ngen").size().items()},
        "observation_model": "exact observed missing-call mask matched by animal within generation",
        "generative_error_rate": error_rate,
        "error_rate_note": "qtl2 error_prob is an inference parameter and is not used as an empirical assay-error estimate",
        "genotype_cells": counters["cells"],
        "matched_missing_share": counters["matched_missing"] / counters["cells"],
        "founder_uninformative_share_outside_mask": counters["founder_uninformative"] / counters["cells"],
        "error_share": counters["errors"] / counters["cells"],
        "called_share": counters["called"] / counters["cells"],
        "exact_founder_state_switch_mean": {
            str(int(g)): float(x)
            for g, x in truth.groupby("ngen")["exact_founder_state_switches"].mean().items()
        },
        "per_chromosome": per_chrom,
    }
    (out.parent / f"painting_{seed_mode}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_chromosome"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed-mode", choices=["idealized_funnel", "direct"], default="idealized_funnel")
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--error-rate", type=float, default=ERROR_RATE)
    args = parser.parse_args()
    if args.pairs < 96:
        raise SystemExit("--pairs must be at least 96 to sample the 192-animal G11 cohort")
    if not 0 <= args.error_rate < 1:
        raise SystemExit("--error-rate must be in [0, 1)")
    build_cross(args.out.resolve(), args.seed_mode, args.pairs, args.error_rate)


if __name__ == "__main__":
    main()

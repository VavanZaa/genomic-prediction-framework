"""Layer 7 / H7 v2, step 1: paint simulated DO mosaics into MegaMUGA genotypes.

Draft protocol: docs/LAYER7_H7v2_PROTOCOL.md. The simulated side must pass through the same qtl2
reconstruction as the observed mice, so simulated mosaics are converted into the genotype calls an
array would have produced:

  1. simulate with the frozen v1 engine and parameters (scripts/run_layer7_h7.py), and sample 99
     animals at G = 8 and 192 at G = 11 to match the observed cohort;
  2. give each marker on each haplotype the allele of the founder strain covering it, taken from the
     real founder genotype files (founders are inbred: one allele per strain and marker);
  3. combine haplotypes into A / B / H, with markers whose founder genotype is '-' left missing;
  4. add genotyping error at 0.2 % and missing calls at 2.17 % (the rate measured in the real
     genotype files), then write a qtl2 cross2 directory the reconstruction reads unchanged.

Reads no observed mouse genotype. Computes no mosaic statistic.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_layer7_h7 import FOUNDERS, SEED, merged_segments, mouse_genome, simulate_generation  # noqa: E402

MOUSE = ROOT / "data/real_mice/svenson"
OUT = ROOT / "output/layer7v2/painted_cross"
COHORT = {8: 99, 11: 192}
MISSING_RATE = 0.0217
ERROR_RATE = 0.002
CHROMS = list(range(1, 20))
CODES = np.array(["A", "B", "H", "-"])


def founder_table(chrom):
    """Marker names, simulator positions (Morgans from the chromosome start) and founder alleles."""
    gmap = pd.read_csv(MOUSE / f"MM/MM_gmap{chrom}.csv", comment="#")
    fg = pd.read_csv(MOUSE / f"MM/MM_foundergeno{chrom}.csv", comment="#").set_index("marker")
    fg = fg.reindex(gmap["marker"])
    if fg.isna().all(axis=1).any():
        raise SystemExit(f"chr {chrom}: markers in the map have no founder genotype row")
    # the simulator places a chromosome on [0, (max - min) / 100]; markers go on the same scale
    pos = (gmap["pos"].to_numpy(float) - gmap["pos"].min()) / 100.0
    # 0 = A allele, 1 = B allele, 2 = uninformative founder call
    alleles = np.select([fg[FOUNDERS].to_numpy() == "A", fg[FOUNDERS].to_numpy() == "B"], [0, 1], 2)
    return gmap["marker"].tolist(), pos, alleles


def paint_haplotype(segments, pos, alleles):
    """Allele at every marker from the founder strain whose segment covers it."""
    starts = np.array([s for _, s, _ in segments])
    strain = np.array([FOUNDERS.index(f) for f, _, _ in segments])
    covering = np.clip(np.searchsorted(starts, pos, side="right") - 1, 0, len(segments) - 1)
    return alleles[np.arange(len(pos)), strain[covering]]


def combine(h1, h2):
    """Two haplotype allele vectors -> genotype code index into CODES."""
    out = np.where(h1 == h2, h1, 2)                 # A/A -> A, B/B -> B, A/B -> H
    return np.where((h1 == 2) | (h2 == 2), 3, out)  # uninformative founder call -> missing


def main():
    rng = np.random.default_rng(SEED)
    genome = mouse_genome()
    tables = {c: founder_table(c) for c in CHROMS}
    for c, spec in zip(CHROMS, genome.chromosomes):
        span = tables[c][1].max()
        if abs(span - spec.length_m) > 1e-9:
            raise SystemExit(f"chr {c}: map span {span} differs from the simulated length {spec.length_m}")

    animals = []
    for g, n in COHORT.items():
        cohort = simulate_generation(genome, g, rng)
        pick = rng.choice(len(cohort), n, replace=False)
        for k, i in enumerate(pick):
            animals.append((f"S{g}_{k:03d}", cohort[i], g))
        print(f"G={g}: simulated {len(cohort)} animals, sampled {n}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MM").mkdir(exist_ok=True)
    ids = [a[0] for a in animals]
    n_called = n_error = n_missing = n_uninformative = n_total = 0
    for c in CHROMS:
        markers, pos, alleles = tables[c]
        geno = np.empty((len(markers), len(animals)), dtype="<U1")
        for j, (_, ind, _) in enumerate(animals):
            h1 = paint_haplotype(merged_segments(ind.paternal_haplotypes[c]), pos, alleles)
            h2 = paint_haplotype(merged_segments(ind.maternal_haplotypes[c]), pos, alleles)
            code = combine(h1, h2)
            n_uninformative += int((code == 3).sum())
            informative = code != 3
            err = informative & (rng.random(len(code)) < ERROR_RATE)
            # an erroneous call becomes one of the two other valid genotypes, chosen at random
            code[err] = (code[err] + rng.integers(1, 3, int(err.sum()))) % 3
            miss = rng.random(len(code)) < MISSING_RATE
            code[miss] = 3
            n_error += int(err.sum())
            n_missing += int((miss & informative).sum())
            n_total += len(code)
            geno[:, j] = CODES[code]
        n_called += int((geno != "-").sum())
        frame = pd.DataFrame(geno, index=markers, columns=ids)
        with open(OUT / f"painted_geno{c}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# painted simulated DO genotypes for chr {c}\n")
            frame.to_csv(fh, index_label="marker")
        for kind in ("foundergeno", "gmap", "pmap"):
            shutil.copy2(MOUSE / f"MM/MM_{kind}{c}.csv", OUT / f"MM/MM_{kind}{c}.csv")
        print(f"  chr {c}: {len(markers):,} markers painted", flush=True)

    pd.DataFrame({"id": ids,
                  "sex": [a[1].sex for a in animals],
                  "ngen": [a[2] for a in animals]}).to_csv(OUT / "painted_covar.csv", index=False)
    spec = {
        "description": "Painted simulated DO genotypes for Layer 7 / H7 v2 (SIMULATION)",
        "crosstype": "do", "sep": ",", "na.strings": ["-", "NA"], "comment.char": "#",
        "geno": [f"painted_geno{c}.csv" for c in CHROMS],
        "founder_geno": [f"MM/MM_foundergeno{c}.csv" for c in CHROMS],
        "gmap": [f"MM/MM_gmap{c}.csv" for c in CHROMS],
        "pmap": [f"MM/MM_pmap{c}.csv" for c in CHROMS],
        "covar": "painted_covar.csv",
        "genotypes": {"A": 1, "H": 2, "B": 3},
        "geno_transposed": True, "founder_geno_transposed": True,
        "sex": {"covar": "sex", "F": "Female", "M": "Male"},
        "cross_info": {"covar": "ngen"},
    }
    (OUT / "painted.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    summary = {
        "evidence": "SIMULATION", "seed": SEED, "cohort": {str(k): v for k, v in COHORT.items()},
        "missing_rate_applied": MISSING_RATE, "error_rate_applied": ERROR_RATE,
        "genotype_cells": n_total,
        "uninformative_founder_share": round(n_uninformative / n_total, 5),
        "error_share": round(n_error / n_total, 5),
        "added_missing_share": round(n_missing / n_total, 5),
        "called_share": round(n_called / n_total, 5),
        "sex_counts": pd.Series([a[1].sex for a in animals]).value_counts().to_dict(),
    }
    (OUT.parent / "painting_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

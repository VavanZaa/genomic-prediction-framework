# Layer 7 / H7 v3: developmental repair after the v1 void and v2 infeasibility

**Status: DEVELOPMENTAL / POST-HOC. This is not a pre-registration and cannot produce an H7 pass or fail on the Svenson cohort.** The v1 and v2 protocols and outputs remain unchanged as audit records.

## Why a third implementation is justified

The v1 measurement counted hard-call gaps as founder switches. The v2 feasibility gate correctly stopped the analysis because even its lowest confidence threshold left too many same-founder gaps and because independent random missingness did not recreate the observed call quality. Those failures identify a measurement problem rather than an endpoint result.

This implementation replaces the broken measurement instead of lowering the gate:

1. `viterbi()` and posterior paths from `sim_geno()` infer a complete qtl2 diplotype path.
2. qtl2's native `count_xo()` counts crossover changes. No custom `guess_phase()` run extraction, hard-call threshold, gap merge, or short-run filter is used.
3. Simulated arrays receive the exact missing-call mask of an observed animal from the same generation. The mask is matched by animal and marker rather than sampled independently from a pooled rate.
4. The developmental estimand is the G8-to-G11 increase per generation:

   `(mean crossovers at G11 - mean crossovers at G8) / 3`

   Subtraction cancels a constant founder-mosaic baseline. The old G11/G8 ratio does not; adding a constant to numerator and denominator changes that ratio.

## Simulation inputs

The primary development run uses 175 breeding pairs, autosomal MegaMUGA maps, random outbreeding without self or full-sib pairs, and an idealized balanced eight-founder funnel before 8 or 11 outbreeding generations. It paints the simulated founder mosaics with the same founder-genotype files used by the Svenson qtl2 cross.

The idealized funnel is only a sensitivity model. The real DO population was founded from 144 partially inbred Collaborative Cross lineages, whose phased founder mosaics and exact pedigrees are not in this archive. The script also retains a `direct` pure-founder start for comparison with v1, but neither start may be labelled the historical DO pedigree.

The painter does not convert qtl2's `error_prob = 0.002` into synthetic errors. That number controls the HMM likelihood and is not an empirical estimate of MegaMUGA assay error. An explicit generative error rate can be supplied for sensitivity analysis.

## Reconstruction and uncertainty

`scripts/reconstruct_do_crossovers_v3.R` processes one chromosome at a time to bound memory. It applies the same settings to observed and painted crosses:

- qtl2 `viterbi()` for a maximum-probability complete path;
- qtl2 `sim_geno()` for posterior path draws;
- qtl2 `count_xo()` for per-animal, per-chromosome crossover counts;
- Haldane map function and `error_prob = 0.002` by default.

Posterior draws are retained, then `scripts/run_layer7_h7v3.py` combines chromosome counts per animal. Its bootstrap resamples animals within G8 and G11 and samples a posterior draw on each iteration. Viterbi and posterior results are reported side by side.

There is no pass threshold. The current data were examined while v1 and v2 were diagnosed, so a threshold written now would be post-hoc. In addition, qtl2's DO HMM uses `ngen` in its transition prior. The result is therefore a posterior-predictive development diagnostic, not an independent validation of the recombination engine.

## Files and commands

```powershell
python scripts/paint_simulated_do_genotypes_v3.py
& "D:\R\app\bin\Rscript.exe" scripts/reconstruct_do_crossovers_v3.R --cross=data/real_mice/svenson/svenson.json --out=output/layer7v3/observed_native --cores=2 --draws=16
& "D:\R\app\bin\Rscript.exe" scripts/reconstruct_do_crossovers_v3.R --cross=output/layer7v3/painted_funnel_cross/painted.json --out=output/layer7v3/painted_funnel_native --cores=2 --draws=16
python scripts/run_layer7_h7v3.py
```

## What would resolve H7

A confirmatory test needs a frozen protocol and data that were not used to design it. At least one of the following is also needed to separate inheritance from HMM prior assumptions:

- phased genomes for the actual CC lines that seeded the DO population, followed by simulation through the recorded breeding design;
- parent-offspring trios or multigeneration pedigrees, with crossovers inferred from transmission rather than an `ngen`-conditioned population HMM;
- an independently generated multiparental cohort with known founder mosaics and generation labels.

The confirmatory primary statistic should remain an increment or fitted generation slope, with pedigree initialization specified before data access. It should include sex-specific maps and crossover interference if the claim is intended to extend beyond the framework's current Haldane-Poisson engine.

## Completed development run

The full 19-autosome run is reported in `reports/2026-09-13_layer7_h7v3_repair.md`. The repaired posterior measurement recovers the simulated exact increment closely (23.02 reconstructed versus 24.16 exact switches per generation), but the observed posterior increment is 45.87. This leaves H7 unresolved and identifies the next work as independent data and model specification, not further threshold adjustment on the Svenson cohort.

## Evidence labels

> Cross-species inheritance test (G_inheritance): unresolved; v1 void, v2 infeasible, v3 developmental only.  
> Cross-species genomic test (G_genomic): not available — no phenotype in hand.  
> Same-trait cross-species phenotype validation: not available.

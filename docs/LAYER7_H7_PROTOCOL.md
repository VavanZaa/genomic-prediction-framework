# Layer 7 / H7 protocol: does the meiosis simulation transfer to a second species?

**Status: pre-registered 12 September 2026, before any observed founder-mosaic statistic was computed.** Frozen with its SHA-256 in `output/layer7/PROTOCOL_LOCK_H7.json`. After freezing, any change is a documented deviation reported with the results.

Hypothesis and scope are set in `docs/LAYER7_HYPOTHESIS.md`. This document fixes the endpoints, the tolerances and the reconstruction method.

## The claim under test

**H7.** The framework's meiosis simulation is a model of Mendelian inheritance with recombination, not a model of cattle. Given the Diversity Outbred breeding design (8 founders, G generations of outcrossing) and the mouse genetic map, simulated genomes should reproduce the observed founder-mosaic structure **with no parameter fitted to mouse data**.

A failure would mean the inheritance layer has absorbed cattle-specific assumptions — a defect no amount of cattle validation could reveal. A pass says the inheritance layer is species-general. **It says nothing about G_genomic or G_phenotype, and must never be summarised as "the framework transfers to mice".**

## Data

`data/real_mice/svenson/` (R/qtl2 format, genotypes and covariates only, no phenotype file):

- **291 DO mice**, MegaMUGA genotypes, at two generation levels: **ngen 8 (99 mice) and ngen 11 (192 mice)**.
- 8 founder genotype files, one per autosome plus X.
- Genetic map: **19 autosomes, 14.062 Morgans, 67,271 markers**; X (0.768 M, 2,076 markers) is **excluded** throughout, because its inheritance differs by sex and the simulator does not model sex chromosomes.

Sex is recorded (150 F, 141 M) and is not used: autosomal recombination is modelled without sex-specific maps, which is stated here as a known simplification rather than discovered later.

## What is simulated, and what is not fitted

- Genome specification built **only** from the MegaMUGA genetic map: 19 autosomal `ChromosomeSpec` entries with their Morgan lengths above.
- 8 founders created with distinct labels (A–H), matching the DO's 8 inbred founder strains.
- Random outcrossing for G generations, run separately for **G = 8** and **G = 11**, with cohort sizes matching the observed 99 and 192.
- **No parameter is estimated from the mouse data.** The only mouse inputs are the map lengths, the founder count and the generation numbers — all design facts, none of them outcomes.

Simulation parameters, fixed here because they are choices rather than observations:

| Parameter | Value | Note |
|---|---|---|
| Founders | 8 strains (labels A–H), **each represented by both sexes** | matches the DO's 8 inbred founder strains; a sex assigned per strain would let a strain enter through one sex only, which the real funnel does not do |
| Breeding population | 100 pairs per generation | a plausible DO colony size; **E3 is a ratio and is insensitive to it**, which is part of why E3 is primary |
| Mating | random, selfing and full-sib pairs excluded | the simulator's default outcrossing behaviour |
| Generations | G = 8 and G = 11, counted from the founder generation | matches the observed `ngen` values |
| Replicate populations | 25 per generation, 200 animals sampled from each | gives 5,000 simulated animals per generation for the interval in E1 |
| Seed | 20260912 | |

Crossovers follow the simulator's existing Haldane Poisson model with no mouse-specific interference parameter, which is the point: an unmodified inheritance engine is being asked to reproduce a second species.

## Reconstructing the observed mosaics

Observed founder mosaics are reconstructed with **R/qtl2**, the standard tool for this population, rather than a bespoke HMM:

- `read_cross2` on `svenson.json`; `calc_genoprob` with `error_prob = 0.002` and `map_function = "haldane"`; `maxmarg` with `minprob = 0.95` for the most probable founder-pair assignment; `guess_phase` to obtain two haplotypes per animal.
- A **founder-state switch** is a change in the assigned founder along a haplotype between adjacent informative markers; its position is taken as the midpoint in cM. Intervals where `maxmarg` returns no call (below `minprob`) break a segment and are excluded from segment-length statistics, and the excluded fraction is reported.
- `Rscript` at `D:\R\app\bin\Rscript.exe` (R 4.6.1). **qtl2 is not currently installed and must be installed before the unblind step**; the version used is recorded in the results file.

Identical statistics are computed from the simulated `Segment` lists, which carry founder identity natively, so no reconstruction is needed on the simulated side. **This asymmetry is a known bias source:** reconstruction can miss short segments that the simulation reports exactly. It is handled in E2 below.

## Endpoints

Fixed here, before any observed statistic is computed. All statistics are per-haplotype and autosomal.

### E3 (primary) — generation scaling of recombination

The ratio of mean founder-state switches per genome at **G = 11 versus G = 8**, compared between simulation and observation:

$$\text{pass if } \left| \frac{R_\text{obs}}{R_\text{sim}} - 1 \right| \le 0.15, \quad R = \frac{\text{mean switches at } G{=}11}{\text{mean switches at } G{=}8}$$

**This is the primary endpoint because a ratio cancels constant offsets.** The real DO design has funnel structure, sib-mating avoidance and its own selection history, none of which the simulator models; those would shift both generations' absolute counts in the same direction and largely cancel here. E3 therefore isolates the recombination process itself, which is what H7 is actually about.

### E1 (secondary) — breakpoints per genome

Mean founder-state switches per haplotype, at each generation separately. Pass if the observed mean lies inside the simulated 95 % interval (2,000 simulation replicates) **and** the relative difference is ≤ 0.20.

### E2 (secondary) — segment-length distribution

Median and 90th-percentile founder-segment length (Morgans), at each generation. Pass if each is within ±0.20 of the simulated value relatively. **Segments shorter than 2 cM are excluded from both sides** before comparison, because reconstruction cannot resolve them reliably and including them would compare a resolution limit rather than a biological quantity. The share excluded is reported for both.

### E4 (descriptive only, never an endpoint) — founder contribution balance

Mean genome share per founder, observed and simulated. Real DO founder contributions are known to be unequal for reasons of strain fertility and design history that the simulator does not model, so **a deviation here is not evidence against H7** and is reported without a pass criterion.

### Tolerance rationale

The ±0.15 and ±0.20 bands are chosen as the magnitudes that would distinguish a species-general inheritance model from one implicitly tuned to cattle. A 20 % error in mean segment length is a substantial mis-specification; deviations well below that are expected from the unmodelled design details listed above. The bands are set now, with no observed statistic in hand, precisely so they cannot be widened to fit a result.

## If H7 fails

A failure is reported as a failure. Then, and **only as a separately labelled diagnostic, never as a pass**:

- re-run with sib-mating avoidance and a funnel-structured founder generation, to test whether the gap is an unmodelled *design* detail rather than a defect in the meiosis code;
- report both outcomes side by side.

This path is written down in advance because the alternative — discovering it after a bad result and presenting the fixed version as the finding — is exactly the mistake made in the 6C₂ feasibility gate.

## What cannot change

- No threshold, marker panel, prediction model or cattle result is touched by this layer, whichever way it falls.
- The simulator is not tuned to the mouse data. If a modification is needed to run an 8-founder design at all, it is a code change recorded before the lock, not a fitted parameter.
- One run per endpoint. The unblind step verifies every hash in the lock before reading a reconstruction.

## Evidence label carried on any result

> Cross-species inheritance test (G_inheritance): performed on genotypes only.
> Cross-species genomic test (G_genomic): not available — no phenotype in hand.
> Same-trait cross-species phenotype validation: **not available.**

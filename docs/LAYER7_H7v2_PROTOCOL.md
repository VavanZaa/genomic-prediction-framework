# Layer 7 / H7 v2 protocol: cross-species inheritance, with symmetric measurement

**Status: DRAFT, not frozen.** To be frozen with its SHA-256 in `output/layer7v2/PROTOCOL_LOCK_H7v2.json` before any statistic is computed from the observed reconstruction. Items marked **[implement]** are code that must exist and be hashed before the lock.

This supersedes nothing: `docs/LAYER7_H7_PROTOCOL.md` and its result stand as reported, and the v1 run remains **VOID**. This is a new pre-registration because the thing that was wrong was the measurement operator, which an amendment cannot repair.

## What v1 got wrong, stated once

v1 counted founder-state switches from hard `maxmarg` calls at `minprob = 0.95`, as (segments per haplotype) − (chromosomes). That left 13.2 % of markers uncalled, and every uncalled stretch split a segment and was then counted as a switch. Of 164,914 transitions, only **8.2 %** were clean state changes, **44.6 %** were one segment split by a gap with an *identical* founder state on both sides, and 47.2 % were unknowable. The reported 247.5 switches per haplotype at G = 8 exceeded even a valid upper bound of 140.2.

Two defects, and the second is the one that matters:

1. Gaps were counted as recombinations.
2. **The observed side passed through a lossy reconstruction while the simulated side was exact.** Even with (1) fixed, the two sides would not have been measuring the same quantity.

## The central change: both sides go through the same pipeline

v2 does not compare reconstructed mosaics against exact ones. Simulated genomes are **painted into MegaMUGA genotypes and passed through the identical qtl2 reconstruction**, so both sides suffer identical information loss, and any residual difference is about recombination rather than about reconstruction.

This is possible because the founder genotypes are on disk: `MM/MM_foundergeno*.csv` holds all 8 founder strains (labelled A–H) at the same markers, coded `A`/`B`/`-`.

### Painting procedure **[implement: `scripts/paint_simulated_do_genotypes.py`]**

1. Simulate mosaics with the frozen v1 engine and parameters (8 strains both sexes, 100 breeding pairs, random outcrossing, seed 20260912), producing **99 animals at G = 8 and 192 at G = 11** — matching the observed cohort exactly.
2. For each simulated haplotype, assign every MegaMUGA marker the allele of the founder strain whose segment covers that marker's cM position. Founders are inbred, so a strain contributes one allele per marker.
3. Combine the two haplotypes into the observed coding: `A`/`A` → `A`, `B`/`B` → `B`, mixed → `H`.
4. Apply, to match the real data's information content:
   - **missing calls at 2.17 %**, the rate measured in `svenson_geno*.csv`, applied at random;
   - **genotyping error at 0.2 %**, matching the `error_prob` used in the reconstruction, applied by flipping a call to another valid code;
   - markers where the founder genotype itself is `-` are set missing, since they carry no information in either dataset.
5. Write a qtl2 cross2 directory (`crosstype: "do"`, the same `gmap`/`pmap`/founder files, an `ngen` covariate of 8 or 11) so the reconstruction script runs on it unchanged.

Missingness and error rates are properties of the assay, not outcomes of the mice, and are declared here before any endpoint exists.

### Reconstruction, applied identically to both **[implement: reuse `scripts/reconstruct_do_mosaics.R` with a `--cross` argument]**

`calc_genoprob` (error_prob 0.002, Haldane) → `maxmarg` at the minprob selected by the gate below → `guess_phase` → run-length segment extraction (already vectorised and equivalence-checked).

**Short-run filter, pre-registered:** after extraction, any founder run shorter than **2 cM** is absorbed into its longer neighbour before switches are counted, on both sides identically. Forced calls at low-confidence markers produce single-marker state flips that are genotyping noise, not recombination; the filter removes them symmetrically, and the share absorbed is reported for each side.

## The feasibility gate — the discipline that was missing

**No endpoint is computed until this passes.** It uses only properties of the reconstructions, never a comparison between them, so it cannot be steered by the result.

Evaluate a pre-declared ladder of `minprob` ∈ **{0.95, 0.90, 0.80, 0.70, 0.60, 0.50}** and select the **largest** value at which all three criteria hold **on both the observed and the simulated-through-pipeline data**:

| Criterion | Bound |
|---|---|
| Uncalled marker share | ≤ 2 % |
| Transitions that are a gap with identical flanking state | ≤ 5 % of all transitions |
| Ratio of upper to lower switch-count bound (ambiguous gaps included vs abutting only) | ≤ 1.25 |

**If no value on the ladder satisfies all three, H7 v2 is declared INFEASIBLE, no endpoint is computed, and that is the reported result.** The ladder and the rule are fixed here precisely so that the choice cannot become a search for a number that produces a pass. No value outside the ladder may be tried.

The v1 configuration would have failed this gate on all three criteria (13.2 % uncalled, 44.6 % gap-identical transitions, bound ratio 6.2). That is the point of having it.

## Endpoints

All statistics are per haplotype, autosomal, after the short-run filter, computed identically on both sides.

### E1′ (primary) — mean founder-state switches per haplotype

At each generation separately. **Pass** if the observed mean lies inside the simulated 95 % interval **and** the relative difference is ≤ **0.20**.

E1′ is primary in v2, where E3 was primary in v1. With symmetric processing an absolute comparison is finally meaningful, and it is the quantity H7 is actually about. v1 needed a ratio to cancel an asymmetry that v2 removes.

### E2′ (secondary) — segment-length distribution

Median and 90th-percentile founder-segment length at each generation, within ±**0.20** relatively.

### E3′ (secondary) — generation scaling

Ratio of mean switches at G = 11 to G = 8, within ±**0.15**. Retained from v1 as a robustness check: it should agree with E1′, and disagreement between them is itself informative.

### E4 (descriptive, never an endpoint) — founder contribution balance

Real DO founder contributions are unequal for design and fertility reasons the simulator does not model, so deviation here is not evidence against H7.

### Intervals

The simulated 95 % interval combines two components, reported separately: animal-level bootstrap within the painted cohort, and the replicate-to-replicate (drift) spread from the frozen v1 exact simulation across its 25 replicates. Only one painted cohort passes through the pipeline, because each reconstruction costs roughly 45 minutes of CPU.

## What cannot change

- No cattle threshold, panel, model or result is touched by this layer.
- No parameter is fitted to mouse data. The mouse inputs remain map lengths, founder count, generation numbers, founder genotypes and assay error/missingness rates — design and assay facts, not outcomes.
- One run per endpoint, after the gate passes. The unblind step verifies every hash in the lock first.
- The v1 result is not restated, revised or rescued by this protocol.

## If H7 v2 fails

Reported as a failure. Then, and only as a separately labelled diagnostic:

- re-run with sib-mating avoidance and a funnel-structured founder generation, to test whether the gap is unmodelled *design* rather than a defect in the meiosis code;
- report both side by side, never the second as the result.

## Implementation clarifications (written during step 1, before the gate ran)

Choices the draft left open, fixed while writing the code and before any gate or endpoint output existed:

- **Reconstruction script.** `scripts/reconstruct_do_mosaics_v2.R` rather than a `--cross` argument on the v1 script, which is hashed in the (void) v1 lock and is left untouched so that lock still verifies. The run-length extraction is copied verbatim from v1, where its equivalence was verified on 3,593 haplotypes.
- **One chromosome at a time.** For each chromosome, `calc_genoprob` runs once (it does not depend on `minprob`), then `maxmarg` → `guess_phase` → extraction repeats at each rung before the next chromosome is loaded. qtl2's HMM treats chromosomes independently, so this yields the same genotype probabilities as a whole-genome call. The core count changes speed, not results. Two whole-genome attempts were stopped during `calc_genoprob` before writing any output: with 8 cores each parallel worker held its own copy of the data and commit headroom fell to 2.9 GB of 76 GB; with 2 cores it was still falling by about 6.5 GB every 10 minutes. On this machine commit exhaustion can crash unrelated processes, so the per-chromosome structure is a safety requirement, not a preference.
- **Phasing is seeded per chromosome** (`set.seed(20260912 + chromosome)` before each `guess_phase`), because `guess_phase` breaks ties at random; per-chromosome seeding keeps every chromosome and rung reproducible in isolation. v1 did not seed it, a minor reproducibility gap recorded here.
- **The gate is computed on the raw extraction, before the short-run filter.** The gate is about call quality; the filter belongs to the endpoints.
- **Switch counting.** Between consecutive called runs on a haplotype: runs of the same founder across an uncalled gap are one segment (no switch); runs of different founders are one switch whether they abut or are separated by a gap, since at least one change occurred. This is the minimum number of switches consistent with the calls, and it is the correction of v1's defect.
- **Short-run filter algorithm.** Per haplotype and chromosome: merge same-founder neighbours; absorb the shortest run under 2 cM into its longer neighbour; merge again; repeat until no run is under 2 cM. Absorbed counts are reported per cohort.
- **E1′ interval.** Painted-cohort mean ± 1.96 × √(bootstrap variance + drift variance), where the drift standard deviation is the v1 exact simulation's replicate spread expressed as a coefficient of variation ((upper − lower)/3.92 ÷ mean) and scaled to the painted mean. This is the construction the review package flags as questionable; it is fixed here so it cannot be changed after the result.
- **Reproduction check, for free.** The observed cohort's called share at minprob 0.95 is 0.8677 under per-chromosome processing — identical to v1's whole-genome run — which confirms the restructuring does not change the reconstruction.
- **Blinding slip, recorded before the gate ran.** The reconstruction log prints the raw, pre-filter run count at each rung. For the painted cohort those counts were seen on screen. They are pooled over both generations, include gap-induced breaks, and precede the short-run filter, so they are not an endpoint, but a run count is a coarse proxy for E1′. The observed cohort's counts were deliberately not displayed; its v1 count at minprob 0.95 (175,972) was already known from the void v1 run. Every endpoint, tolerance, the filter and the interval construction were written into this document before either reconstruction ran, and nothing was changed after the counts appeared. The log line should omit run counts in any future run.
- **Painting as implemented:** 19,575,861 genotype cells; 0.42 % missing because the covering founder's genotype is `-`; error 0.200 %; added missingness 2.163 %; 97.4 % called.

## Gate outcome (13 September 2026): INFEASIBLE

Steps 1–2 were run; the gate was applied once by `scripts/run_layer7_h7v2.py --stage gate` (`output/layer7v2/gate_result.json`). **No rung met all three criteria on both cohorts, so H7 v2 is INFEASIBLE, no lock was written and no endpoint was computed.**

| minprob | Observed: uncalled / gap-identical / bound ratio | Painted: uncalled / gap-identical / bound ratio |
|---|---|---|
| 0.95 | 0.132 / 0.446 / 6.79 | 0.069 / 0.443 / 6.00 |
| 0.90 | 0.100 / 0.432 / 5.36 | 0.046 / 0.425 / 4.50 |
| 0.80 | 0.073 / 0.420 / 4.38 | 0.030 / 0.409 / 3.64 |
| 0.70 | 0.056 / 0.402 / 3.49 | 0.022 / 0.386 / 2.92 |
| 0.60 | 0.038 / 0.354 / 2.40 | 0.014 / 0.327 / 2.05 |
| 0.50 | 0.020 / 0.179 / 1.32 | 0.007 / 0.119 / 1.18 |
| **bound** | **≤ 0.02 / ≤ 0.05 / ≤ 1.25** | **≤ 0.02 / ≤ 0.05 / ≤ 1.25** |

**The binding criterion is the gap-identical share, and it fails on both cohorts at every rung.** Even at minprob 0.50, where the painted cohort leaves only 0.7 % of cells uncalled, 11.9 % of its transitions are uncalled gaps flanked by the *same* founder. Uncalled cells are therefore not scattered: they concentrate in short stretches that resolve back to the founder they left. The likely cause — stated as a hypothesis, not tested here — is local haplotype sharing among the DO founder strains, several of which descend from related classical laboratory lines: where two founders are identical by state, the diplotype is ambiguous whatever the call threshold. If so, **a hard-call transition statistic cannot reach this precision on DO data at any minprob**, and lowering the threshold further would only trade gaps for noise.

**Painting is not symmetric in call quality.** At every rung the observed cohort leaves roughly twice the uncalled share of the painted one (0.132 vs 0.069 at 0.95; 0.020 vs 0.007 at 0.50). Simulated genotypes with independent 2.17 % missingness and 0.2 % error are cleaner than real MegaMUGA data. This confirms, with data, the concern raised in the protocol review package: independent random missingness and error do not reproduce the real information loss, so the asymmetry v2 set out to remove partly survives painting.

These are properties of the measurement, compared across cohorts only in call quality — they are not endpoints and say nothing about H7.

**What the gate did.** It stopped a third invalid measurement before any endpoint existed, which is its entire purpose. v1's configuration fails it by the largest margins in the table.

**What this does not license.** Designing a v3 immediately after an infeasible gate is the path by which analyses get redesigned until they run. Any further version needs a justification that stands without this result — a posterior-based switch statistic that uses no hard calls, and painting that resamples real per-marker missingness and error patterns — and should go through external design review before implementation.

## Order of work

1. **[implement]** the painting script, the `--cross` argument for the reconstruction script, and the v2 runner.
2. Run the feasibility gate. If it fails, stop and report INFEASIBLE.
3. Freeze the lock, including the painted cohort's hash and the selected `minprob`.
4. Unblind once.

Steps 1–2 happen before the lock because the gate must not be able to see an endpoint; step 3 records the gate's output so the selection is auditable.

## Evidence labels

> Cross-species inheritance test (G_inheritance): v1 void; v2 status to be reported here.
> Cross-species genomic test (G_genomic): not available — no phenotype in hand.
> Same-trait cross-species phenotype validation: **not available.**

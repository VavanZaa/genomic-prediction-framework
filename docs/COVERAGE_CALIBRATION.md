# How many shared markers does the support rule need?

**Status: rule fixed 16 September 2026, before the experiment was run.** Result recorded at the end of this document.

## Why this exists, and the disclosure that comes with it

The 6C₃ clause required **at least 5,000 panel markers** to match, and the Decker worldwide panel yields **2,891**. The arm was reported inconclusive on that basis.

That 5,000 was written as a bound on a method that no longer exists. Its sentence reads "…otherwise 6C₃ is reported as inconclusive and the thresholds are not touched. **Missing calls are filled with Florida training means.**" Amendment 4 withdrew mean-filling for structurally absent markers and replaced it with the shared-marker subspace, in which the support region is refitted on training animals restricted to the markers actually shared. **No one has ever established what the subspace method requires**, and it is a different method with a different failure mode: mean-filling degrades by pulling animals toward the training centre, while the subspace degrades by losing resolution.

**Disclosure, stated plainly because it is the obvious objection:** this calibration is being run with full knowledge that Decker yields 2,891 markers, at the request of the project owner, who wants that arm testable. Knowing the number one hopes to clear is a real bias risk. It is handled as follows, and a reader should judge whether that is sufficient:

- the criterion and the decision rule are fixed in this document **before the experiment runs**;
- the experiment uses **Florida data only** — no Decker genotype is read, and no external dataset enters;
- the grid deliberately **includes 2,891** alongside values above and below it, and the **entire curve is reported**, so a reader can see exactly where it falls and judge the rule against the shape;
- whatever the rule returns is what stands. If 2,891 does not clear it, 6C₃ remains inconclusive and this document says so.

This does not make the calibration confirmatory. It is a design decision made after seeing a coverage number, and it is labelled as such wherever it is used.

## What is being calibrated

Not "how many markers give a good result on Decker" — that would be circular. The question is internal and answerable without any external data:

> At how few shared markers does the k-NN support region still separate animals inside the Florida training support from animals outside it?

## Design

Everything is fitted on Florida animals; the out-of-domain group is created by ancestry holdout rather than borrowed from an external population.

- **Support-region training set:** Florida training animals (Year ≤ 2018) with estimated Brahman coordinate **at or below the 70th percentile** of that cohort.
- **Out-of-domain holdout:** animals **at or above the 90th percentile** — genuinely outside the fitted support region, and never seen by it. The 70th–90th percentile band is left empty as a buffer, so the two groups are not adjacent.
- **In-domain holdout:** a random 25 % of the training-pool animals, held out from the support fit, which the rule should accept.

> **Revision, recorded before any criterion value was computed.** The first version of this split used absolute cut-offs (train q ≤ 0.45, out-of-domain q ≥ 0.70). The script's own guard stopped the run: only **10** training animals sit at q ≥ 0.70, far too few for a stable estimate. The Florida training cohort's ancestry coordinate runs from 0.03 to 0.91 with a median of 0.295 and a 99th percentile of 0.612, a distribution already known and printed earlier in this project, so the fault is mine for picking cut-offs without checking counts. Quantile-based thresholds are used instead, which fix the group sizes by construction. No R_out or R_in had been computed when this was changed.
- **Marker grid:** 500, 1000, 2000, **2891**, 4000, 5067, 8000, 12000, 20087 — the full panel at the top, Decker's count and the slick/hairy count (5,067) marked for reference.
- **Draws:** 10 random marker subsets at each grid value, each refitting the support region on the restricted training set exactly as the subspace rule does. Median across draws is used.
- **Detector:** as it actually runs — `GenomicOODDetector` with the frozen revision-1 thresholds, ranking decision.

## Decision rule, fixed before running

At each marker count *m*, with medians across the 10 draws:

- **R_out(m)** = share of out-of-domain holdout animals refused for ranking;
- **R_in(m)** = share of in-domain holdout animals falsely refused.

> **m is sufficient if R_out(m) ≥ 0.90 and R_in(m) ≤ 0.05.**
> **The minimum required coverage is the smallest grid value that is sufficient and for which every larger grid value is also sufficient** (so a single lucky point cannot set the requirement).

The 0.90 / 0.05 pair is the same order of separation the rule already achieves at full coverage, and it is the level at which a refusal endpoint means something: a rule that refuses fewer than nine in ten genuinely out-of-domain animals is not a safety layer, and one that falsely refuses more than one in twenty in-domain animals is not usable.

## Consequences, also fixed in advance

- If the minimum is **≤ 2,891**, the 6C₃ coverage clause is superseded by that number, and Decker becomes testable under a new pre-registration written before its support scores are computed. Its result would carry the label that its coverage requirement was set after its coverage was known.
- If the minimum is **> 2,891**, 6C₃ stays inconclusive and the Decker panel is not analysed. No second calibration is run, and no criterion is adjusted.

## Limits of this calibration

- The out-of-domain group is high-Brahman Florida cattle, not a foreign breed. It is out of the fitted support region, which is what the rule measures, but it is closer to the training animals than a Holstein or a Bali ox would be. The marker count needed to reject a *distant* population is plausibly lower than the count needed here, which makes this requirement conservative rather than permissive.
- Markers are drawn at random from the panel. Decker's 2,891 are not a random subset: they are the SNP50 backbone, which is ascertained for common variation and spread across the genome. Whether that helps or hurts relative to a random draw is not established here.
- This sets a coverage requirement for the support rule only. It says nothing about the coverage a *prediction* score needs, which is a separate question about signal loss.

## Result (16 September 2026)

Support region fitted on 929 animals (q ≤ 0.345, the 70th percentile); 310 in-domain holdout; 186 out-of-domain holdout (q ≥ 0.425, the 90th percentile). Ten random marker subsets per grid value.

| Markers | R_out (out-of-domain refused) | R_in (in-domain falsely refused) | Sufficient? | |
|---|---|---|---|---|
| 500 | 0.723 | 0.058 | no | |
| 1,000 | 0.898 | 0.063 | no | |
| 2,000 | 0.970 | 0.053 | no | fails R_in by 0.003 |
| **2,891** | **0.992** | **0.042** | **yes** | Decker worldwide panel |
| 4,000 | 0.997 | 0.048 | yes | |
| 5,067 | 1.000 | 0.045 | yes | slick/hairy shared subspace |
| 8,000 | 1.000 | 0.045 | yes | |
| 12,000 | 1.000 | 0.039 | yes | |
| 20,087 | 1.000 | 0.048 | yes | full 6C panel |

**By the rule as fixed: the minimum is 2,891 markers, and Decker's coverage is sufficient.**

### The rule's weak point, which the curve exposes

R_in sits between 0.039 and 0.063 at **every** marker count, including 0.048 at the full panel. That is structural, not a coverage effect: `KNNSupport` sets its radius at the 95th percentile of training distances, so about 5 % of in-domain animals fall outside it by construction whatever the coverage. A criterion of R_in ≤ 0.05 is therefore a knife-edge sitting exactly on the method's built-in false-refusal rate, and it is what failed m = 2,000 — by 0.003. **The precise minimum this rule returns is unstable, and it landed exactly on the number of interest, which is a coincidence a reader is right to be suspicious of.** I should have foreseen that the 95th-percentile radius makes R_in nearly uninformative, and chosen a criterion that measures coverage rather than the quantile design.

The informative column is R_out, which coverage genuinely drives: 0.723 → 0.898 → 0.970 → 0.992 → 1.000.

### What is actually supported

Not "the minimum is 2,891". Rather:

> **At 2,891 shared markers the support rule refuses 99.2 % of genuinely out-of-domain animals while falsely refusing 4.2 % of in-domain animals — materially the same behaviour as the full 20,087-marker panel (100 % / 4.8 %).**

That statement holds with margin and does not depend on the knife-edge criterion. Below roughly 2,000 markers the rule degrades visibly (97 %, then 90 %, then 72 %), so the degradation is real and this is not a claim that coverage never matters.

### Consequence

The 6C₃ coverage clause of **5,000 markers is superseded**: it bounded mean-filling, which Amendment 4 withdrew, and the subspace method reaches full-panel behaviour by roughly 3,000 markers. **Decker becomes testable under a new pre-registration**, written and hashed before any Decker support score is computed.

Any 6C₃ result must carry this label:

> Coverage requirement set after the dataset's coverage was known (`docs/COVERAGE_CALIBRATION.md`), on Florida data only, with the rule fixed before the run. The criterion that set the exact minimum is knife-edge; the supporting claim is the 99.2 % / 4.2 % separation at that coverage.

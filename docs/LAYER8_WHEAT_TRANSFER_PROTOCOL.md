# Layer 8: the first genuine transfer test, on the CIMMYT wheat benchmark

**Status: DRAFT, to be frozen in `output/layer8/PROTOCOL_LOCK_L8.json` before anything is fitted.**

## Why this arm exists

The project has **no external predictive transfer result in any species**. C1 was withdrawn, C4 and C10 are void, C9 is unavailable — and the University of Florida has now declined the request that was C9's only realistic route in cattle. Mizzou cannot substitute: it records no farm identifier, and an attempt to infer one from day length produced impossible latitudes, so a herd-held-out test there would rest on a broken unit — the same defect that voided 6C₁.

The CIMMYT wheat benchmark supports the test directly: **599 lines, 1,279 DArT markers, grain yield in 4 environments**. Training on one set of lines in one environment and predicting **different lines in a different environment** gives new individuals *and* a new environment, which is the structure of "a herd you have never trained on".

**The decisive property is that this dataset carries its own controls.** The phenotypic correlations between environments are:

| | 1 | 2 | 4 | 5 |
|---|---|---|---|---|
| **1** | 1.000 | −0.020 | **−0.193** | −0.123 |
| **2** | −0.020 | 1.000 | **+0.661** | +0.411 |
| **4** | −0.193 | +0.661 | 1.000 | +0.388 |
| **5** | −0.123 | +0.411 | +0.388 | 1.000 |

Environments 2 and 4 agree strongly; environment 1 is uncorrelated or mildly opposed to the rest. So the data contain a case where transfer **should** work and a case where it **should fail**. A method that succeeds everywhere here is not transferring — it is finding structure that is not there.

## What this can and cannot test

- **Can:** genomic prediction and the applicability layer, transferring to unseen individuals in an unseen environment.
- **Cannot:** the inheritance layer. These are inbred homozygous lines with no Mendelian sampling, so C12, C13 and H7 are untouched.
- **Cannot:** anything about cattle accuracy. This is a test of the method, in a species whose biology is irrelevant to the claim.

## Design

- **Markers:** 1,279 DArT, 0/1. VanRaden-style genomic relationship matrix, scaled to mean diagonal 1.
- **Line split:** lines are divided at random into disjoint train and test halves. **Test lines never appear in training in any environment**, so the model has never seen them.
- **Environment grid:** all 16 ordered pairs (E_train, E_test) over environments 1, 2, 4, 5. Same-environment pairs are the internal reference; the 12 cross-environment pairs are the transfer test.
- **Repeats:** 20 random line splits; median and a 95 % interval across repeats.
- **Models:** `GBLUP` (kernel ridge on the GRM, λ by training-only 5-fold CV — the same `cv_kernel_ridge` used for cattle and mice) and `StructureOnly` (the top 5 genomic principal components as fixed effects), the plant analogue of the cattle ancestry baseline.
- **Metric:** Spearman correlation between prediction and observed yield among held-out lines in the test environment.

## Endpoints, fixed in advance

**T1 (positive control).** Transfer between environments 2 and 4, whose phenotypes correlate +0.661, must work:
> Pass if ρ(2→4) **and** ρ(4→2) are both positive with 95 % intervals excluding zero.

**T2 (negative control — the endpoint that makes this test worth running).** Transfer between environments 1 and 4, whose phenotypes correlate −0.193, must **not** appear to work:
> Pass if neither ρ(1→4) nor ρ(4→1) has a 95 % interval lying entirely above zero.

A failure of T2 is the informative outcome: it would mean the pipeline reports successful transfer where the trait does not transfer, which would cast doubt on every accepted prediction the applicability layer has ever allowed.

**T3 (ordering, secondary).** Across the 12 cross-environment pairs, transfer accuracy should track how strongly the two environments agree:
> Pass if the Spearman correlation between the environment-pair phenotypic correlation and the transfer accuracy is **≥ 0.5**.

**T4 (applicability, secondary).** Held-out lines come from the same population as the training lines, so the support rule should accept them:
> Pass if at least **90 %** are accepted for ranking.

T4 repeats the check that **failed in mice** (H7b accepted only 88.5 % of same-population held-out families). A second failure would confirm that the support rule mistakes ordinary relatedness structure for being out of domain.

**Overall verdict** rests on T1 and T2 together. Passing T1 while failing T2 is not a partial success; it is evidence the method cannot tell transfer from its absence.

## Implementation check, reported but not an endpoint

This is a widely published benchmark, so within-environment accuracy on held-out lines is a check on our implementation against the literature. Published analyses of this dataset typically report correlations in the region of 0.4–0.5. A large deviation is a warning about our code rather than a finding about wheat, and is reported as such. It is deliberately **not** a pass criterion, because the exact figure depends on the split and the model, and turning a remembered range into a threshold would be false precision.

## What cannot change

- No cattle or mouse threshold, panel, model or result is touched.
- Nothing is tuned on the test environment or the test lines: λ comes from training-only cross-validation, and the support region is fitted on training lines.
- One run. The unblind stage verifies every hash in the lock first.
- All 16 environment pairs are reported, whatever they show.

## Evidence labels

> External transfer test (new individuals, new environment): performed, in wheat.
> Cross-species inheritance (H7): untouched — void.
> Same-trait external phenotype validation in cattle: **not available**; the UF request was declined.

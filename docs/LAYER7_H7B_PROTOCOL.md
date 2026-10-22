# Layer 7 / H7b: does the prediction machinery work in a second species?

**Status: DRAFT, to be frozen in `output/layer7b/PROTOCOL_LOCK_H7B.json` before anything is fitted.**

## The claim

From `docs/LAYER7_HYPOTHESIS.md`:

> **H7b (genomic).** Given any heritable trait measured in the same animals, a GBLUP fitted and evaluated entirely within the mouse population should show the same qualitative behaviour as in cattle — a positive prospective rank correlation, an applicability layer that refuses animals outside its training support, and no pathological behaviour from the baseline comparison. This tests the machinery, not transfer, and needs no biological comparability at all.

H7b was blocked because the Diversity Outbred archive carries no phenotypes. The Heterogeneous Stock mice do: **1,940 genotyped mice, 85 families, 12,545 markers, 8 inbred founder strains**, with weight, biochemistry, haematology and immunology measured on the same animals.

**What a pass would mean:** the genomic prediction code is species-general software, not a cattle-specific artefact. **What it would not mean:** nothing about cattle accuracy, nothing about cross-species transfer, and nothing about H7 (inheritance), which remains void.

## Data and preparation

- Genotypes: `data/real_mice/hs_mice/genotypes/chr{1..19,X}.build37.data`, ped format, alleles as letters with `NA` for missing. Converted to counted-allele dosage per marker, missing filled with the marker mean, then a VanRaden GRM — the same construction used for cattle.
- Phenotypes: the four files with the most genotyped mice — `Weight.txt`, `Biochemistry.txt`, `Haematology.txt`, `Immunology.txt`.
- **X chromosome markers are excluded**, as in every other layer, because inheritance differs by sex and the pipeline does not model it.

### Trait selection rule, fixed before any model is fitted

From each of the four files, take the **two numeric traits with the most non-missing values among genotyped mice whose column name carries that file's biological prefix** (`Weight.`/`Start.`/`End.` for weight; `Biochem.`; `Haem.`; `Imm.`). Technical columns — dates, cage density, litter, study day, ages — are covariates, never traits.

That yields **8 traits**, and **all 8 are reported**. No trait is dropped after seeing its result. The rule is deliberately mechanical so that traits cannot be chosen for being likely to succeed.

### Covariates

Sex, age at measurement, and study year where the file records them, identically in every model. A trait's covariates are the same for GBLUP and for the baseline, so the comparison isolates the genomic term.

## Design

- **Split: held-out families.** Whole families are assigned to the test set — the mouse analogue of the cattle `withheld_family` split, and the harder of the two. 10 repeats, each holding out a random 20 % of the 85 families.
- **Models:** `GBLUP` (covariates + genomic effect, λ by training-only 5-fold CV, the same `cv_kernel_ridge` used for cattle) and `CovariatesOnly` (the same fixed effects, no genomic term).
- **Metric:** Spearman correlation between prediction and observed phenotype in the held-out families.

## Endpoints, fixed in advance

**E1 (primary) — does genomic prediction work at all in this species?**
> Pass if GBLUP's median held-out Spearman ρ is positive with a 95 % interval excluding zero in **at least 4 of the 8 traits**.

**E2 (secondary) — does the genomic term add over covariates?**
> Pass if GBLUP's median ρ exceeds CovariatesOnly's in **at least 4 of the 8 traits**.

**E3 (secondary) — does the applicability layer behave sanely in a new species?**
> Held-out-family mice are from the same population as the training mice, so the support rule should accept them. Pass if **at least 90 %** are accepted for ranking. A high refusal rate here would mean the rule mistakes ordinary family structure for being out of domain, which would matter for cattle too.

## What this cannot change

- No cattle threshold, panel, model or result is touched.
- Nothing is tuned on the mouse phenotypes: λ comes from training-only cross-validation, and the support region is fitted on training mice.
- One run. The unblind stage verifies every hash in the lock before a phenotype meets a genotype.
- A failure is reported as a failure. If the machinery does not work in a second species, that is informative about the software, and the cattle results would need re-examination.

## Known limitations, stated before the result

- Heterogeneous Stock mice are far more related to one another than a cattle population is; held-out families still share the same 8 founders, so accuracies are not comparable in magnitude with cattle numbers.
- 12,545 markers is sparse by modern standards, though dense relative to the genome size and the 8-founder haplotype structure.
- The source states no licence; results stay internal pending the authors' permission, like the Florida data (open item H8).

## Evidence labels

> Cross-species genomic machinery test (H7b): performed.
> Cross-species inheritance test (H7): v1 void, v2 infeasible — unchanged by this arm.
> Same-trait cross-species phenotype validation: **not available** — and this arm is not one.

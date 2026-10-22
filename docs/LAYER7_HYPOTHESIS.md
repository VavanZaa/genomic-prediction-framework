# Layer 7: cross-species hypothesis, defined before any mouse data is fitted

**Status: hypothesis pre-registered 12 September 2026. Nothing has been fitted. One axis is testable with the data on disk; two are blocked for want of a phenotype.**

The rule this document exists to enforce: the hypothesis, the axes and the endpoints are written before a mouse genotype meets a mouse phenotype, so that whatever the data say cannot be re-described afterwards as what we expected.

## Why mice at all

Layer 6C asks whether the framework transfers to another cattle population. Layer 7 asks something different and stronger: **which parts of the framework are about breeding, and which parts are about cattle.** A multiparental outbred mouse population (Diversity Outbred, Collaborative Cross) is the sharpest available test, because it is a designed crossbreeding system with known founders, known generation counts and dense genotypes — the same structure the framework models, in a species with none of cattle's biology.

The three generalisation axes from Layer 6C carry over, and they are **not** equally testable here.

| Axis | Cross-species question | Needs a phenotype? | Status |
|---|---|---|---|
| **G_inheritance** | Does the meiosis and inheritance simulation reproduce a real multiparental population's genome structure? | **No** — genotypes alone | **Testable with the data on disk** |
| **G_genomic** | Do genomic prediction and ranking machinery behave as designed in a different species? | Yes, any heritable trait | Blocked: no phenotype on disk |
| **G_phenotype** | Do absolute predictions transfer across species? | Yes, a biologically comparable trait | **Not attempted, and not expected to be meaningful** |

G_phenotype across species is not a scientific question this framework can ask. A cattle hair-length model has nothing to say about a mouse's hair length in absolute units, and any number produced by trying would be an artefact. It is listed here only to be explicitly excluded.

## The cross-species hypothesis

**H7 (inheritance):** the framework's meiosis simulation is a model of Mendelian inheritance with recombination, not a model of cattle. Given the DO breeding design (8 founders, known outcrossing generation G), simulated genomes should reproduce the observed population's founder-mosaic structure **without any parameter fitted to mouse data**: the distribution of founder-haplotype segment lengths, the number of recombination breakpoints per genome, and the decay of founder-haplotype sharing with generation number, each within a pre-set tolerance of the observed distribution.

If H7 holds, the inheritance layer is species-general and the cattle results do not rest on it being tuned to cattle. If it fails, the simulation has absorbed cattle-specific assumptions — a defect worth knowing about, and one that no amount of cattle validation would reveal.

**H7 is a falsifiable claim about our own code, tested against a real population that our code has never seen.** That is its whole value; it is not a claim that mouse results transfer to cattle.

**H7b (genomic, blocked):** given any heritable mouse trait measured in the same animals, a GBLUP fitted and evaluated entirely within the mouse population should show the same qualitative behaviour as in cattle — a positive prospective rank correlation, an applicability layer that refuses animals outside its training support, and no advantage for the ancestry-only baseline. This tests the machinery, not transfer, and needs no biological comparability at all. It is blocked only because the genotype set on disk carries no phenotypes.

## Dataset requirements, fixed in advance

For a confirmatory **H7**: founder-haplotype or founder-probability assignments for a multiparental population, per-animal generation number, a genetic map, and either the founder mosaics/pedigree that initialized the population or transmission data that identify crossovers without a generation-conditioned HMM. No phenotype is needed for the inheritance axis itself.

For **H7b**: the above, plus any heritable quantitative trait measured on the same animals, with enough animals (≥ 300) for a prospective split.

For a genuine **same-trait** cross-species comparison, which is what the "biologically comparable phenotype" requirement means: a measured coat or thermoregulation trait — hair length, coat density, shedding, or body-temperature response to heat load — in an outbred mouse population with dense genotypes.

## What is actually on disk, and what it cannot do

`data/real_mice/svenson/` — Svenson et al. Diversity Outbred sample data in R/qtl2 format: **291 DO mice, MegaMUGA genotypes across 20 chromosome files, 8 founder genotype files, genetic and physical maps, and a covariate file holding `id`, `sex` and `ngen` (outcrossing generation).**

**There is no phenotype file.** Not a coat trait, not any trait. The archive contains genotypes and covariates only.

Consequently:
- **H7 is not confirmatorily testable with these files alone.** They support a development diagnostic, but they do not contain the phased mosaics of the 144 partially inbred CC lineages that seeded the DO population. The available qtl2 reconstruction also uses generation number in its transition prior, so it is not independent evidence for generation scaling.
- **H7b is blocked** until a phenotype file for these or comparable animals is obtained.
- **A same-trait cross-species test is not available.** A targeted search of the public DO/CC resources (Mouse Phenome Database, QTLViewer, the JAX DO reference data) did not surface a public DO or CC dataset with a measured hair-length, coat-density or thermoregulation phenotype. Body weight, cholesterol, bone density and behavioural traits are what those populations are phenotyped for.

**Evidence label, to be carried on every Layer 7 result:**

> Cross-species inheritance test: unresolved — v1 void, v2 infeasible, v3 developmental only.
> Cross-species genomic test: not available (no phenotype in hand).
> Same-trait cross-species phenotype validation: **not available.**

No proxy is substituted for the third line. A mouse body-weight model is not a mouse coat model, and calling it one would be the exact failure this document exists to prevent.

## Order of work

1. Preserve the v1 void and v2 infeasible results as audit records.
2. Use H7 v3 only to develop and diagnose the repaired measurement.
3. Freeze a new protocol before collecting or opening an independent cohort, with the initialization pedigree and crossover estimand specified.
4. Run H7b only if a phenotype for these animals or a suitable independent cohort is obtained.
5. Report each axis separately. A future H7 pass would address only the inheritance layer; it would say nothing about G_genomic or G_phenotype.

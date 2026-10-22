# Layer 6C protocol: external cattle validation

**Status: DRAFT, not frozen.** The protocol is frozen, with its SHA-256 recorded in `output/layer6c/PROTOCOL_LOCK.json`, before any external phenotype file is opened. Items marked **[decide]** must be filled first. After freezing, any change is a documented deviation, reported with the results.

## Questions

1. **Can v1 recognise when an external population is predictable?** This tests the applicability detector.
2. **When it accepts animals, are its predictions right?** This tests transportability.

These are scored separately for three kinds of generalisation, which are never merged into one claim:

| Axis | Question | Confidence output |
|---|---|---|
| G_inheritance | Does Mendelian / meiosis simulation transfer to this population? | C_inheritance |
| G_genomic | Do marker effects and rankings transfer? | C_ranking |
| G_phenotype | Do absolute phenotype levels transfer across environments? | C_absolute |

Passing G_inheritance and G_genomic while failing G_phenotype is a valid, informative outcome: it points to environmental recalibration, not a genomic failure.

## Populations

| Arm | Purpose | Requirement |
|---|---|---|
| **Near-domain** | Transportability | Independent herd, not in the Florida files, of Angus × Brahman-type ancestry within the Florida training range; same traits measured with a comparable protocol; SNP genotypes that can be harmonised |
| **Far-domain** | Refusal | Cattle clearly outside Florida support (e.g. a purebred taurine or indicine breed, or another composite) |

**[decide]** Dataset identity for each arm: source, accession or DOI, number of animals, genotype platform and assembly, traits and measurement protocol. Genotype and pedigree files may be inspected and hashed before freezing (harmonisation needs them). **Phenotype files are hashed without being opened.** Their hashes are recorded in the lock file, and the analysis checks them.

Candidates (`reports/2026-09-12_6c_dataset_shortlist.md`, not yet chosen):
- the UF multibreed herd's 2017/2018/2023/2024 heifers (same traits; phenotypes on request);
- the Mizzou hair-shedding project (Dryad ngf1vhhz4): Brangus near-domain, Angus and Hereford far-domain;
- EVA PRJEB75981 (Brangus, genotypes only);
- the Decker 2014 worldwide panel (far-domain, genotypes only).

- **Duplicate / overlap rule:** before phenotypes are opened, any external animal whose genomic self-relationship with a Florida animal exceeds 0.95 is excluded and counted.
- **Same-ranch applicability check (pipeline QA, not a transfer claim):** the 1,052 Seminole steers (EVA PRJEB75981; genotypes only, obtained 12 September 2026) come from the ranch of the training heifers. After harmonisation they are scored by the detector. Expected: at least 80 % accepted for ranking. A much lower rate means a harmonisation or detector fault, which must be fixed before any external phenotype is opened. These steers are also relatives of training animals, so they never count toward any G-axis.
- **Arm assignment for the Mizzou data (the public CSV has no breed column):** each animal's Brahman fraction is estimated from its genotypes by maximum likelihood, using Angus- and Brahman-specific allele frequencies derived from the Florida SNP and breed-of-origin files. The arms are fixed before the hair scores are opened:
  - near-domain: estimated q_B within the Florida training support (support ratio ≤ 1);
  - far-domain: estimated q_B below 0.05 (taurine);
  - all other animals are reported descriptively only.
- **Blinded columns:** in the Mizzou CSV only `hair_score` is a phenotype. The remaining columns (`id`, `sex`, `coat_color`, `age_class`, `age_group`, `calving_season`, `toxic_fescue`, `score_group`, `date_score_recorded`, `mean_apparent_high`, `mean_day_length`) may be read before freezing. `coat_color` is recorded but not used to assign arms.
- **Related-trait arm (only if chosen):** when the external trait differs from the Florida trait (e.g. a 1–5 hair-shedding score against short hair length), the endpoint is the rank correlation between the Florida hair prediction and the external score, among animals accepted for ranking. It is reported as a related-trait G_genomic result, never as transfer of the Florida trait itself. The expected sign is fixed in advance: longer predicted hair goes with more winter coat retained.

## Frozen model and rules

- **Model:** the Florida SNP GBLUP trained on the canonical cohort (Year ≤ 2018, 1,754 animals), with the v1 hyperparameters (λ chosen by training-only CV). There is no refit on external data.
- **Primary analysis:** no recalibration of any kind.
- **Pre-specified secondary analysis:** mean-only recalibration, subtracting the external contemporary-group mean of the residual. It is reported separately and bears only on G_phenotype.
- **Marker harmonisation:**
  - match by SNP name, then by position on the same assembly (lifting coordinates if the assemblies differ);
  - align alleles by letter, and drop strand-ambiguous A/T and C/G SNPs;
  - require that at least **[decide, proposed 80 %]** of the model's markers are present; otherwise the arm is inconclusive;
  - fill missing calls with Florida reference allele frequencies;
  - build the external-to-Florida cross-GRM with the Florida reference frequencies stored in the v3 GRM file.
- **Applicability:** `GenomicOODDetector` in calibrated mode, loaded from `output/ood_calibration/thresholds_rev1.json` through `load_frozen_thresholds` with the hash recorded here: **[SHA-256 of thresholds_rev1.json]**.
  - **Ranking** is allowed when the support score (max of genomic and ancestry k-NN ratios) is at most 1.0. This is an epistemic guard, not a risk-derived value: the pre-specified risk calibration found risk flat in the score (`docs/OOD_CALIBRATION.md`).
  - **Absolute prediction** also requires the animal's contemporary group / herd to have been seen in training. External herds therefore get ranking at most, by design.
  - No threshold, hyperparameter or marker filter changes after unblinding.
- **Inheritance axis:** where parent–offspring trios with genotypes exist, the Layer 5B-T recovery analysis is run unchanged. C_inheritance = accepted if the Mendelian error rate is ≤ 1 % and marker overlap meets the harmonisation rule. Without trios, G_inheritance is "not tested".

## Endpoints and pass criteria (fixed before unblinding)

Minimum sample: 50 animals accepted on an axis; below that the axis is **inconclusive**. Intervals are 95 % bootstrap over animals, 2,000 replicates.

**Near-domain arm**

| Axis | Primary endpoint | Pass if |
|---|---|---|
| G_genomic | Pearson r between prediction and phenotype among animals accepted for ranking | lower 95 % bound > 0 |
| G_phenotype | Calibration-in-the-large (mean residual) and calibration slope among animals accepted for ranking. The detector refuses absolute predictions for unseen herds, so this measures whether that refusal was needed | Pass (the refusal was conservative) if \|mean residual\| ≤ 0.5 phenotypic SD **and** the slope interval includes 1; otherwise the refusal is confirmed |
| Detector | False-safe rate (as defined in 6A) among accepted animals | ≤ 10 % |

Reported, not pass criteria: Spearman ρ, RMSE, risk–coverage curve and AURC, false-refusal rate, and paired Δr versus an ancestry-only model (the "genomic beats ancestry" hypothesis).

**Far-domain arm**

| Endpoint | Pass if |
|---|---|
| Refusal rate for ranking | ≥ 90 % of animals refused |
| Consistency | Where phenotypes allow, r among refused animals is reported; refused animals should not show better accuracy than near-domain accepted animals (descriptive) |

## Frozen specification for the public data (Mizzou hair-shedding, Seminole steers)

Chosen by Vavan on 12 September 2026: public data first. This section overrides the general text above wherever they differ. It is implemented in `scripts/run_layer6c_external_validation.py`.

- **Data and hashes:**
  - Mizzou genotypes and phenotype/metadata file: Dryad ngf1vhhz4, SHA-256 verified against Dryad.
  - Seminole steers: ENA ERZ24663392, MD5 verified.
  - The UF multibreed herd and the Decker panel are not used in this run.
- **Marker panel:** `data/external/panel_6c_markers.csv`, built by `scripts/build_6c_marker_panel.py`. Markers must be oriented, strand-unambiguous and position- and letter-matched to Mizzou, with Minimac R² ≥ 0.80, on autosomes, thinned evenly to about 20,000.
- **Allele coding:** the Florida A/B → nucleotide map from the Seminole steers (`data/external/allele_lookup_f250.csv`; frequency agreement r = 0.993). Missing external calls are filled with Florida training means.
- **Model:** SNP GBLUP on the panel, trained on the canonical cohort (Year ≤ 2018, joint traits, n = 1,754) with training-only contemporary-group adjustment. The kernel is VanRaden with Florida training frequencies, and λ comes from 5-fold training CV. There is no refit on external data.
- **Brahman fraction:** one supervised maximum-likelihood estimator (`src/real_data/ancestry_estimation.py`) for Florida and external animals alike. Breed-specific frequencies come from the Florida SNP and breed-of-origin files. Its agreement with Florida's `BrahmanGBC` is reported.
- **Applicability:** revision 1 (`thresholds_rev1.json`). The environment unit is the herd: Florida training = "Semin", the steers = "Semin" (seen), Mizzou = unseen. Mizzou animals can therefore get ranking at most.
- **Duplicates:** an external animal is excluded if its genomic correlation with any Florida animal exceeds 0.95.
- **Arms (Mizzou), fixed before hair scores are read:**
  - far: estimated q_B < 0.05;
  - near: ancestry support ratio ≤ 1 against the Florida training ancestries;
  - other: everyone else, reported descriptively only.
- **Animal phenotype:** `hair_score` minus its `score_group` mean (groups with ≥ 5 records), averaged over the animal's records.
- **Primary endpoint, G_genomic (related trait):**
  - statistic: Spearman ρ between the Florida hair-length prediction and the animal phenotype, among near-domain animals accepted for ranking;
  - expected sign: positive, since longer predicted hair should go with more winter coat retained;
  - pass: lower 95 % bootstrap bound (2,000 replicates) > 0;
  - inconclusive: fewer than 50 accepted animals.
- **Far-domain endpoint:** pass if at least 90 % of far-domain animals are refused for ranking.
- **Same-ranch QA gate (steers):** at least 80 % accepted for ranking. If this fails, unblinding stops until the cause is found.
- **Not tested with these data:**
  - G_phenotype (different trait scale);
  - G_inheritance (no trios);
  - LoadSlope transfer (no external LoadSlope).
- **Reported, not pass criteria:** Pearson r, ρ among all and among refused animals in each arm, and the "other" arm.

## Amendment 1 — 6C₁ closing test (12 September 2026, after unblinding, at Vavan's request)

**Status: post-hoc by construction.** The primary endpoints were already unblinded and are unchanged by this amendment; nothing here can revise them. The amendment is written, with its statistic and decision rule, before the analysis is run.

Question: in the near-domain arm, do the animals the applicability rule accepted show stronger rank transfer than the animals it refused, once the repeated-measures structure is respected? The unblinded run reported ρ = +0.167 among accepted and +0.000 among refused, but animals share `score_group` contemporary groups, so those correlations are not independent observations.

- **Unit and clustering:** one row per animal (group-centred mean hair score, as in the frozen specification). Each animal is assigned to the `score_group` holding most of its records; ties take the first. All resampling and standard errors cluster on that group.
- **Primary statistic:** Δρ = ρ(accepted) − ρ(refused), Spearman, within the near-domain arm. Interval: 2,000 cluster bootstrap replicates resampling `score_group` clusters with replacement; replicates leaving fewer than 10 accepted or 10 refused animals are discarded and counted.
- **Secondary statistic:** rank-transform prediction and phenotype within the arm, then fit phenotype_rank ~ prediction_rank + accepted + prediction_rank × accepted by OLS with CR1 cluster-robust standard errors on `score_group`; report the interaction coefficient, its cluster-robust t statistic and two-sided p.
- **Decision rule:** "accepted transfer significantly stronger" requires the Δρ bootstrap interval to exclude zero **and** the interaction p < 0.05. Either one alone is reported as inconclusive.
- **Interpretation limit:** acceptance is decided by genotype-based support, not randomised, so a difference shows that the rule separates animals where the model ranks from where it does not; it is not a causal effect of acceptance.

## Amendment 2 — the contemporary unit was wrong (12 September 2026, post-hoc)

Amendment 1 clustered on `score_group`, assuming it identified a contemporary group. It does not: the column takes only **16** distinct values (1–16) because it is a 5-day scoring window *within a herd*, so the same number recurs across farms. Consequences for the frozen specification and Amendment 1:

- the animal phenotype was centred across different farms that share a window number;
- 2,400 of the 2,877 near-domain animals fell into one pooled "cluster", and only 2 pooled groups held both accepted and refused animals, so Amendment 1's cluster-robust statistics are not within-contemporary-group comparisons and are not relied on.

`date_score_recorded` + `mean_apparent_high` + `mean_day_length` identifies a **scoring event** (one herd scored on one day: the weather covariates are location-specific): **433 events**, median 54 animals, with `score_group` nested inside them (event × score_group also gives 433).

- **Corrected contemporary unit:** the scoring event, for both centring and clustering. An animal's phenotype is its mean event-centred score; its cluster is the event holding most of its records.
- **Statistics, all reported for both units side by side:** ρ among accepted and among refused; Δρ with 2,000 event-clustered bootstrap replicates; the CR1 interaction; and the interaction with event fixed effects, which compares accepted and refused animals *inside* the same event.
- **Robustness check of the primary endpoint:** ρ among accepted animals under event centring, with an event-clustered interval, reported beside the frozen result (ρ = +0.167 [+0.103, +0.231]). The frozen endpoint stands as published; this is not a replacement.
- **Rule for closing 6C₁:** the event-clustered Δρ interval must exclude zero **and** the within-event interaction must have p < 0.05 with at least 10 events holding ≥ 10 accepted and ≥ 10 refused animals. Otherwise 6C₁ stays open and the accepted-vs-refused contrast is reported as unresolved.

## 6C₃ — worldwide refusal stress test (pre-registered 12 September 2026, before the data were obtained)

Purpose: show that the applicability boundary is biologically structured, not tuned to one external dataset. Genotypes only; no phenotype, so no transfer endpoint.

- **Data:** Decker et al. 2014, Dryad `10.5061/dryad.th092` (CC0): 1,543 animals, 134 breeds, Illumina BovineSNP50 (43,043 markers), with breed labels in the sample file.
- **Harmonisation:** markers matched to the 6C panel by rsID or Illumina name first; by position only if the assembly is confirmed (the SNP50 map may be UMD3.1, the Florida panel ARS-UCD). Strand-ambiguous pairs stay excluded. **At least 5,000 panel markers must match**, otherwise 6C₃ is reported as inconclusive and the thresholds are not touched. Missing calls are filled with Florida training means.
- **Ancestry groups, assigned from the published breed labels before any support score is computed:** (1) indicine-admixed American composites (e.g. Brangus, Beefmaster, Santa Gertrudis); (2) Asian/American *Bos indicus* (e.g. Brahman, Nelore, Gir); (3) European taurine (e.g. Angus, Hereford, Holstein, Charolais); (4) African taurine (e.g. N'Dama, Baoulé); (5) other/distant bovids (e.g. Bali cattle/*Bos javanicus*, yak-type).
- **Pre-registered expectation:** acceptance for ranking is highest in group 1, lower in groups 2 and 3, and near zero in groups 4 and 5. Formally, with expected-distance ranks 1 < 2 ≈ 3 < 4 < 5, Spearman ρ between a breed's expected-distance rank and its acceptance rate should be **negative**.
- **Endpoints:**
  - pass if that Spearman ρ ≤ −0.5 with a breed-level bootstrap interval excluding zero (2,000 replicates resampling breeds);
  - and if groups 4–5 are refused at ≥ 95 %;
  - reported alongside: acceptance rate and median support score per breed, and the estimated Brahman fraction per breed (which should itself order the groups sensibly).
- **No phenotype, no numeric prediction:** every animal is outside every Florida herd, so absolute prediction is refused by the rule regardless of support.
- **Nothing here may change the thresholds, the panel or the model.** A failure is reported as a property of the applicability layer.

## 6C₃ — feasibility outcome (12 September 2026, genotype side only)

The pre-registered harmonisation clause above decides this arm, and it decides against running it.

The Decker map states UMD3.1; the 6C panel is ARS-UCD1.2, which the conversion confirms (99.6 % of the 43,043 markers convert). Marker matching gives: 0 by Illumina name (the Dryad map carries only rsIDs), 142 by rsID, **2,891 by converted position**. The clause reads "at least 5,000 panel markers must match, otherwise 6C₃ is reported as inconclusive and the thresholds are not touched."

**6C₃ is therefore reported as inconclusive on coverage grounds. No support score, acceptance rate or breed ordering was computed, and nothing in the model, panel or thresholds was touched.** The limit is real and not a processing failure: the BovineSNP50 and the GGP-F250 used in Florida are deliberately complementary array designs, so their shared marker content is small. The arm becomes available if the panel is ever rebuilt on an array that overlaps the SNP50 backbone; that would be a new panel, a new lock and a new pre-registration.

## 6C₂ — slick / hairy specificity probe (pre-registered 12 September 2026, before any score–phenotype association was computed)

Data: Littlejohn et al. 2014, Dryad `10.5061/dryad.nh6v4` (CC0), four files, all four MD5s verified against the published record. What the files contain was established first, because it changes what can be asked:

| File | Contents | Usable here |
|---|---|---|
| `gen_phen_data_for_paper.zip` | 99 animals, 712,123 HD markers (UMD3.1), binary hairy-syndrome status in the `.fam` | yes, partially |
| `phased_gen_phen_data_slick.zip` | 82 slick crossbreds, **26 markers**, chromosome 20 (PRLR region) only | no |
| `physiological_phenos.zip` | 12–24 animals per file (sweating, histometry, lactation, TRH challenge), no genotypes | no |
| `Hairy_bull_progeny_TaqMan_results.txt` | 2,205 progeny, single-variant TaqMan calls, no genotypes | no |

- **The slick arm is infeasible and is not run.** The slick crossbreds carry 26 PRLR-region markers, so the Florida polygenic hair score cannot be computed for them at all. The originally proposed question — "does the Florida genomic hair score recognise biologically slick cattle?" — **cannot be answered with this dataset**, and is recorded as not tested rather than replaced by a substitute.
- **Coverage for the hairy arm:** after conversion, 5,178 panel markers match by position and 5,067 survive allele checking (5,009 direct, 58 complement, 111 monomorphic excluded, 0 strand-ambiguous). Allele agreement among matched non-monomorphic markers is 5,067/5,067, which is the evidence that the conversion is real. That is **25.2 % of the 20,087-marker panel**; the other 74.8 % are filled with Florida training means.
- **What that coverage supports, measured on Florida's own held-out animals before the external data were scored** (658 test animals, panel masked to the same 5,067 markers): predictions keep their rank order (Spearman ρ = 0.81 for AvgShortLength, 0.85 for LoadSlope) but lose most of their spread (sd ratio 0.46 and 0.44); the Brahman-fraction estimate is nearly unchanged (r = 0.939, mean |Δ| = 0.019); the k-NN support score is substantially distorted (r = 0.555).
- **Consequences, fixed here in advance:**
  - numeric prediction is not reported for this arm under any outcome (the score is compressed by a factor of about two);
  - the **primary endpoint is ancestry-driven refusal**, which the ancestry robustness check above licenses; the k-NN support score is reported but carries no endpoint, because at this coverage it is not reliable;
  - the score–phenotype contrast is **secondary and descriptive**.
- **Primary endpoint (G_genomic, far-domain refusal):** every one of the 99 animals is refused for numeric prediction (no Florida herd), and ≥ 95 % are refused for ranking. These are New Zealand dairy cattle; if the layer accepts them for ranking, that is a reportable failure of the applicability rule.
- **Secondary, descriptive only:** the difference in predicted AvgShortLength between affected and unaffected animals, as a standardised mean difference with a 2,000-replicate bootstrap interval. **Pre-registered expectation: no detection.** Hairy syndrome is a near-monogenic PRLR-pathway defect, and a polygenic Angus–Brahman hair-length score is not expected to detect it; a null is the specificity-confirming outcome. A non-null would not be reported as successful transfer, because these 99 animals are close relatives (progeny of a small number of sires), so family structure, not the trait, is the leading explanation.
- **This arm can change nothing.** No threshold, marker panel, model or training set is touched by its outcome, whichever way it falls. It does not count as same-trait external validation, and it does not reopen 6C₁.
- **Blinding:** the case/control counts in the `.fam` (62 unaffected, 37 affected) were read during the feasibility assessment above and are stated here; no association between any score and that column was computed before this section was written and hashed into the lock.

## Amendment 3 — the 6C₂ refusal endpoint measured the imputation (12 September 2026, post-hoc)

The frozen 6C₂ run refused 100 % of the 99 animals for numeric prediction and **0 %** for ranking, against a pre-registered ≥ 95 %. Taken at face value that is a failure of the applicability rule on New Zealand dairy cattle. It is not: the endpoint was void by construction, through a defect in my own feasibility gate.

The harmonisation clause fills unobserved markers with Florida training means. At 25.2 % coverage that sets 74.8 % of every external animal's genotype vector to the training centroid — in exactly the space the k-NN support region measures. The feasibility check that licensed the endpoint (ancestry robust at r = 0.939 under masking) was run **only on Florida held-out animals**, who are in-domain, so mean-filling moves them somewhere they already belong. It could not detect the bias it would cause for an out-of-domain population, and I drew a conclusion from it that it did not support.

`scripts/run_layer6c2_imputation_diagnosis.py` separates the two explanations by changing only the handling of unobserved markers, with the support region and breed frequencies recomputed on Florida **training** animals restricted to the same 5,067 markers (never on external data):

| | n | ranking refusal | median support score |
|---|---|---|---|
| A — external animals, mean-filled over the full panel (the frozen run) | 99 | 0.000 | 0.605 |
| B — the same animals in the 5,067-marker shared subspace | 99 | **1.000** | 0.690 |
| C — Florida held-out animals in that same subspace (control) | 658 | 0.015 | 0.577 |

B refuses everything A accepted while C keeps accepting, so the frozen run measured the imputation, not the population.

- **The 6C₂ primary endpoint is recorded as VOID, not failed.** No threshold, model, panel or training set was changed by it.
- **The harmonisation clause is wrong as written and is not reused.** Mean-filling is acceptable for scattered missing calls; it is not acceptable when a whole array's worth of markers is absent. Any future arm below **80 % panel coverage** must score, estimate ancestry and assess support in the shared-marker subspace, with the support region refitted on training animals restricted to those markers. *(Superseded by Amendment 4: the coverage threshold was the wrong rule and has been withdrawn.)*
- **A feasibility check for an out-of-domain claim must be run on out-of-domain animals.** The in-domain-only check above is the specific mistake to avoid repeating.

**The one substantive result of 6C₂ is about the applicability rule itself.** The supervised two-way ancestry estimator places these Holstein-Friesian animals at q_B = 0.18, which is *inside* the Florida training range (min 0.03, max 0.91; 11.8 % of training animals lie in [0.15, 0.21]) — an out-of-reference breed projected onto an Angus–Brahman axis lands in the middle, which is a known limit of two-way supervised admixture. The **min–max ancestry bounds used before revision 1 would therefore have accepted these cattle as in-domain. The k-NN support region refuses all 99.** This is the first external evidence that replacing ancestry bounds with an empirical support region catches a population the old rule would have let through, and it is a property of the rule, observed on genotypes, independent of the void phenotype endpoint.

## Amendment 5 — the 6C₃ coverage clause is superseded (16 September 2026)

The 6C₃ clause required at least 5,000 matched panel markers, and it was written for a method that no longer exists: its own sentence continues "Missing calls are filled with Florida training means", which Amendment 4 withdrew for structurally absent markers in favour of the shared-marker subspace. The subspace method's coverage requirement had never been established.

`docs/COVERAGE_CALIBRATION.md` establishes it on Florida data only, with the rule fixed before the run: fitting the support region on the lower 70 % of the training ancestry coordinate and testing separation against a held-out upper 10 %. Out-of-domain refusal reaches 0.723 at 500 markers, 0.898 at 1,000, 0.970 at 2,000, **0.992 at 2,891** and 1.000 from 5,067 up, while in-domain false refusal stays at 0.039–0.063 throughout — the latter being structural, since the support radius is the 95th percentile of training distances.

- **The 5,000-marker clause is superseded.** Under the subspace method, coverage at or above roughly 3,000 markers reproduces full-panel behaviour.
- **6C₃ (Decker, 2,891 markers) becomes testable**, under a new pre-registration written and hashed before any Decker support score is computed. The arm's earlier INCONCLUSIVE stands as the record of what was decided under the old clause.
- **Disclosure carried on any 6C₃ result:** the coverage requirement was re-derived after this dataset's coverage was known, at the project owner's request. The exact minimum rests on a knife-edge criterion and should not be quoted; the supporting claim is the 99.2 % / 4.2 % separation at 2,891 markers.

## Amendment 4 — the 80 % coverage threshold is withdrawn (12 September 2026, after external review)

Amendment 3 answered a 25 % failure with an 80 % threshold. An external reviewer pointed out that this rescues the wrong variable, and they are right: **coverage percentage is not the quantity that decides whether mean-filling is valid.**

Substituting a training mean for an unobserved marker moves a genome toward the training centroid, and a distance-based support region measures exactly that displacement. The bias is therefore *directional at any coverage* — small at 10 %, catastrophic at 75 %, but never absent, and with no privileged discontinuity at 80 % or anywhere else. A calibrated threshold would only put a number on an arbitrary cut.

What actually differs is the **kind** of missingness:

- **Sporadic missing calls** — markers present on the platform that failed for this animal. Missingness is roughly independent of the animal's position in genotype space, so filling with a training mean adds noise rather than direction. Mean-filling remains acceptable.
- **Structural platform missingness** — markers never assayed because the arrays differ. Every animal in the cohort is missing the same markers, so the fill is a systematic shift shared by the whole cohort, straight toward the training centre.

**The rule, replacing the 80 % threshold:**

> Where markers are structurally absent because platforms differ, ancestry estimation and support assessment are performed in the shared-marker subspace, with the support region refitted on training animals restricted to those markers — **at any coverage, 95 % or 25 %**. Mean-filling is confined to sporadically missing calls among markers the platform actually carries.

Coverage keeps one legitimate and separate role: deciding whether the **prediction score** is still usable, which is a question about signal loss (at 25 % coverage the score retained rank order, ρ = 0.81, but lost over half its spread) and not about the geometry of the support region. Those two decisions are now made independently, and no coverage number gates the support rule at all.

This removes a heuristic rather than adding one, which is the reason to prefer it.

## Execution

- `scripts/run_layer6c_external_validation.py` is written and hashed into the lock file before phenotypes are opened.
- One run per arm; every run is logged, and failed runs are reported, not repeated silently.
- Results are compared against `output/baseline_v1/` and frozen with `scripts/freeze_baseline.py --tag v2`.

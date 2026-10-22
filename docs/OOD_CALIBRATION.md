# OOD support threshold: risk-based calibration (pre-specified)

Written on 12 September 2026 **before** `scripts/calibrate_ood_thresholds.py` was first run. Any change after that run is listed in the change log at the end, with its reason.

## Why

v1 calls a point supported when its k-nearest-neighbour distance is within the 95th percentile of the training animals' own distances. The 95 % is arbitrary. The calibrated rule replaces it with: **the largest region in which prediction error stays below a level fixed in advance.** For candidate thresholds c on a support score s,

- coverage C(c) = P(s ≤ c), and
- risk R(c) = E[loss | s ≤ c];

c\* = the largest c whose risk meets the target.

## Data: internal Florida holdouts only

- **Calibration pool:** Seminole animals with LoadSlope, breed composition and SNP genotypes, with q_B ≤ 0.32. These are Layer 6A's training and development cohorts.
- **Never used:** 6A's blind cohorts (Seminole q_B > 0.32; every Williamson animal) and every Layer 6C population. 6A therefore stays an untouched internal check of c\*, and 6C an untouched external one.
- Genotypes: the step-24 v3 SNP panel, missing calls imputed from the Year ≤ 2018 reference (as in 6A).

## Holdout schemes

Each scheme partitions the whole pool; every animal is held out once per scheme.

| Scheme | Folds | Shift it represents |
|---|---|---|
| Year | leave one birth year out (2016–2019) | temporal |
| Contemporary group | leave one `Set` out; Sets with < 30 pool animals form one fold | management (Set is nested in Project; the pool has one Project) |
| Ancestry band | leave one q_B quintile of the pool out | ancestry |
| Relatedness cluster | k-means (K = 8, seed 0) on the pool's top 10 genomic PCs; leave one cluster out | family / relatedness |
| Random (reference) | 5-fold random CV, seed 0 | none: the in-domain risk R_ref |

In every fold, everything is fitted on the training part only: contemporary-group effects (a held-out record whose Set was not seen is left unadjusted), SNP GBLUP, and the OOD detector.

## Score and losses

- **Support score:** s = max(genomic ratio, ancestry ratio). Each ratio is the k-NN distance divided by the training fold's 95 % leave-one-out radius (the v1 rule is s ≤ 1). The genomic ratio uses the detector's whitened principal-component space; the ancestry ratio uses q_B.
- **Absolute loss:** (y − ŷ)². This is the G_phenotype / C_absolute axis.
- **Ranking loss:** ((y − ȳ_fold) − (ŷ − ŷ̄_fold))². It is squared error after removing the held-out fold's level, so a pure level shift costs nothing. This is the G_genomic / C_ranking axis.
- **Trait:** LoadSlope, the trait 6A evaluates. AvgShortLength is reported at the selected thresholds as a sensitivity check and does not choose them.

## Selection rule

- **Grid:** c ∈ {0.5, 0.6, …, 4.0} and c = ∞ (accept everything).
- **Risk:** pooled over the four shift schemes (not the random reference).
- **Risk target:** R_max = 1.10 × R_ref for the same axis, i.e. at most 10 % above in-domain risk.
- **Conservative estimate:** the one-sided 95 % bootstrap upper bound of R(c), resampling animals (1,000 replicates, seed 0).
- **Minimum accepted:** 100 held-out records.
- **Choice:** c\* = the largest c meeting the target on its upper bound. If no c qualifies, that axis gets no threshold, and the detector refuses that kind of prediction.
- **Reported alongside:** the v1 rule (c = 1.0).

## Output and freezing

The script writes `output/ood_calibration/`: `thresholds.json` (c\*_ranking, c\*_absolute, R_ref, R_max, coverage and risk at c\*, SHA-256 of the script, the design document and the inputs), `risk_coverage.csv`, `records.csv` and a figure. It refuses to overwrite an existing `thresholds.json`. The SHA-256 of `thresholds.json` is copied into `docs/LAYER6C_PROTOCOL.md` when that protocol is frozen, and the 6C analysis must load the file through `load_frozen_thresholds`, which checks that hash. **Layer 6C results are never used to change c\*.**

## Result of the pre-specified run (12 September 2026)

`output/ood_calibration/thresholds.json`, SHA-256 `29618e059b2adca38a0b1688827ad37c8ba332136a7d34809fb7c7f774105cc0`. Pool: 1,734 animals.

- **LoadSlope (selects the thresholds):** no threshold meets the target on either axis, so c\*_ranking = none and c\*_absolute = none. At the v1 rule (c = 1) coverage is 0.72; ranking risk is 1.16× in-domain and absolute risk 1.37×.
- **AvgShortLength (check only):** ranking c\* = 1.6 (coverage 0.77); absolute: none.

Risk ÷ in-domain risk, LoadSlope:

| Scheme | Ranking at c = 1 | Ranking at c = ∞ | Absolute at c = 1 | Absolute at c = ∞ | Coverage at c = 1 |
|---|---|---|---|---|---|
| Year | 1.415 | 1.400 | 1.527 | 1.525 | 0.94 |
| Contemporary group | 1.012 | 0.995 | 1.536 | 1.540 | 0.93 |
| Relatedness cluster | 1.020 | 1.011 | 1.034 | 1.024 | 0.94 |
| Ancestry band | 1.368 | 1.017 | 1.356 | 1.015 | 0.09 |

Findings:

1. **Risk is flat in the support score.** Within the relatedness-cluster holdouts, Spearman(s, loss) is −0.02 on every axis. Holding out a whole ancestry quintile raises risk by only 2 %. At this predictive ability (r ≈ 0.1–0.15), loss is dominated by environmental and residual variance, not by genomic distance.
2. **The excess risk is environmental:** unseen birth years (both axes) and unseen contemporary groups (absolute axis). No genomic support threshold can detect it.
3. **The pre-specified ranking loss is confounded.** It removes the held-out fold's mean, but a held-out year holds several unseen contemporary groups, whose differences stay in the loss. Folds containing a single contemporary group have ranking risk 1.00.
4. Under the frozen thresholds, the calibrated detector refuses 100 % of every 6A blind cohort (`output/real_layer6/layer6_ood_validation_calibrated.csv`).

No rule has been changed. A revision would be recorded below, with its reason, before any re-run.

## Change log

**Revision 1 — 12 September 2026, decided by Vavan after the pre-specified run and before any Layer 6C data was obtained.**

- **What changes:** the support threshold and the environment requirement take separate roles.
  - **Support is an epistemic guard, not a risk-derived value.** Ranking is allowed when s = max(genomic, ancestry) k-NN ratio ≤ 1.0, the v1 leave-one-out 95 % radius. The 95 % is a stated policy: it keeps predictions inside the data the model has seen.
  - **Absolute (numeric) prediction additionally requires the animal's contemporary group / herd to have been seen in training**, because that is where the measured risk lives (1.53× for unseen years and contemporary groups).
- **Why:** finding 1 above. Risk is flat in the support score, so no support threshold can be risk-calibrated on these data. Finding 2: the excess risk is environmental.
- **Record kept:** `thresholds.json` (SHA-256 `29618e05…`) is unchanged. The active rule is `thresholds_rev1.json`, which names the file it supersedes. Layer 6C loads the active file through `load_frozen_thresholds` with the hash recorded in its protocol.

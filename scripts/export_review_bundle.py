"""Build a self-contained review bundle for an external reviewer (human or AI agent).

The bundle answers one question: what does this project currently claim, and on what evidence?
It is standalone on purpose — every document and result file it cites is embedded verbatim, so a
reviewer can import one JSON file and need nothing else from the repository.

    python scripts/export_review_bundle.py
    python scripts/export_review_bundle.py --dest "C:/path/to/share"

Writes output/review_bundle/review_bundle.json (machine-readable, everything embedded) and
output/review_bundle/REVIEW.md (the same material as prose). Optionally copies both to --dest.

This is a manual step. No experiment, test or pipeline script calls it: scientific result
generation must not depend on the tool that launched it (tests/test_pipeline_hygiene.py).

It deliberately embeds **no genotype or phenotype data**. The Florida source repository states no
licence (open item H8), so the bundle carries results and metadata only, and says so.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402

OUT = ROOT / "output" / "review_bundle"

DOCS = [
    "docs/EVIDENCE_POLICY.md",
    "docs/OOD_CALIBRATION.md",
    "docs/LAYER6C_PROTOCOL.md",
    "docs/LAYER6C4_DECKER_PROTOCOL.md",
    "docs/COVERAGE_CALIBRATION.md",
    "docs/LAYER7_HYPOTHESIS.md",
    "docs/LAYER7_H7_PROTOCOL.md",
    "docs/LAYER7_H7v2_PROTOCOL.md",
    "docs/LAYER7_H7B_PROTOCOL.md",
]
REPORTS = [
    "reports/2026-09-17_h7b_and_scale.md",
    "reports/2026-09-16_layer6c4_results.md",
    "reports/2026-09-12_layer7_h7_results.md",
    "reports/2026-09-12_review_round1_response.md",
    "reports/2026-09-12_layer6c_results.md",
    "reports/2026-09-12_layer6c2_6c3_layer7.md",
    "reports/2026-09-12_v1_cattle_baseline.md",
    "reports/2026-09-12_uf_phenotype_request.md",
]
# Code a reviewer needs to judge a claim rather than take it on trust. An external reviewer found
# that the first bundle promised self-containment while omitting the modules its own review agenda
# asked them to inspect, so the support, split, ancestry and harmonisation paths are embedded here.
CODE = [
    "src/evaluation/support.py",
    "src/evaluation/ood_detector.py",
    "src/evaluation/risk_calibration.py",
    "src/real_data/canonical_split.py",
    "src/real_data/ancestry_estimation.py",
    "src/real_data/harmonisation.py",
    "src/real_data/liftover.py",
    "scripts/run_layer6c_external_validation.py",
    "scripts/run_layer6c2_hairy_probe.py",
    "scripts/run_layer6c2_imputation_diagnosis.py",
    "scripts/run_layer6c2_c3_robustness.py",
    "scripts/run_layer6c2_c5_family_clustered.py",
    "scripts/run_layer6c1_closing_test.py",
    "scripts/run_layer7_h7.py",
    "scripts/run_layer7_h7_diagnosis.py",
    "scripts/reconstruct_do_mosaics.R",
    "src/inheritance/meiosis.py",
    "src/inheritance/chromosome.py",
]
TESTS = [
    "tests/test_support.py",
    "tests/test_ood_detector.py",
    "tests/test_liftover.py",
    "tests/test_ancestry_estimation.py",
    "tests/test_harmonisation.py",
    "tests/test_real_data_ingestion.py",
]
RESULTS = [
    "output/layer6c/results_mizzou.json",
    "output/layer6c/results_6c1_closing.json",
    "output/layer6c2/results_6c2.json",
    "output/layer6c2/results_6c2_imputation_diagnosis.json",
    "output/layer6c2/results_6c2_c3_robustness.json",
    "output/layer6c2/results_6c2_c5_family_clustered.json",
    "output/layer6c2/prepare_summary_6c2.json",
    "output/layer6c3/results_6c3.json",
    "output/layer6c4/results_6c4.json",
    "output/layer6c4/prepare_summary_6c4.json",
    "output/layer7/results_h7.json",
    "output/layer7/results_h7_diagnosis.json",
    "output/layer7/simulated_mosaics.json",
    "output/layer7v2/gate_result.json",
    "output/layer7v2/painting_summary.json",
    "output/ood_calibration/thresholds_rev1.json",
    "output/layer7b/results_h7b.json",
    "output/mizzou_scale/scale_curve.json",
    "output/mizzou_scale/scale_law_fit.json",
    "output/heritability/genomic_heritability.json",
    "output/baseline_v1/stageD_prediction_benchmark.csv",
    "output/baseline_v1/stageE_policy_comparison.csv",
    "output/baseline_v1/EVIDENCE.csv",
    "output/audit_post_fix/real_layer5/layer5b_trio_recovery_summary.csv",
    "output/audit_post_fix/real_layer5/layer5b_trio_recovery_segregation.csv",
]
LOCKS = [
    "output/layer6c/PROTOCOL_LOCK.json",
    "output/layer6c2/PROTOCOL_LOCK_6C2.json",
    "output/layer6c4/PROTOCOL_LOCK_6C4.json",
    "output/layer7/PROTOCOL_LOCK_H7.json",
    "output/layer7/PROTOCOL_LOCK_H7_v1_superseded.json",
]

STATUS = {
    "ESTABLISHED": "supported by evidence that survived its own pre-registered test",
    "POST_HOC_SUPPORTED": ("compelling diagnostic evidence generated after unblinding; robustness may have "
                           "been mapped, but independent confirmation under a rule fixed in advance is "
                           "still required. Never cite as confirmed."),
    "CONSISTENT_WITH_NULL": ("no association detected, in a design that excludes large effects far better "
                             "than modest ones; not evidence of absence"),
    "WITHDRAWN": "was reported, then failed a corrected re-analysis; must not be cited",
    "VOID": "the test could not measure what it claimed to measure; not a result either way",
    "INCONCLUSIVE": "stopped by a pre-registered rule before any endpoint was computed",
    "NOT_TESTED": "the data cannot answer the question; no substitute was used",
    "NOT_AVAILABLE": "no dataset exists in hand for this axis",
}


def load(rel):
    """Parsed result file, or None when it has not been produced yet."""
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def claims():
    """Every current claim, its status, and where a reviewer can check it."""
    rob = load("output/layer6c2/results_6c2_c3_robustness.json") or {}
    fam = load("output/layer6c2/results_6c2_c5_family_clustered.json") or {}
    return [
        {"id": "C1", "axis": "G_genomic (near-domain, related trait)", "status": "WITHDRAWN",
         "claim": "The Florida hair model's ranking transfers to the Mizzou hair-shedding population.",
         "numbers": {"frozen_spec_rho": 0.167, "frozen_ci": [0.103, 0.231],
                     "corrected_rho": 0.055, "corrected_ci": [-0.049, 0.146],
                     "delta_rho": 0.060, "delta_ci": [-0.045, 0.170], "p": 0.29},
         "why": ("The frozen phenotype centred on `score_group`, which takes only 16 values: a 5-day "
                 "window within a herd, not a contemporary group, so the same number recurs across "
                 "farms. The corrected unit is the scoring event (433 events)."),
         "confound": ("Acceptance is nearly collinear with herd: only 1 of 144 events holds both >=10 "
                      "accepted and >=10 refused animals, so no statistic can separate the rule's "
                      "selection from herd effects in this dataset."),
         "audit_status": (
             "An external reviewer asked whether defects of the same kind remain in the Florida "
             "contemporary-group handling, and could not check it because the first bundle omitted the "
             "source. `src/real_data/canonical_split.py` is embedded here so that the production "
             "adjustment path can be read directly. No new defect has been demonstrated; that is not the "
             "same as cleared."),
         "evidence": ["output/layer6c/results_6c1_closing.json", "reports/2026-09-12_layer6c_results.md",
                      "src/real_data/canonical_split.py"]},

        {"id": "C2", "axis": "G_genomic (far-domain refusal)", "status": "ESTABLISHED",
         "scope": "rule behaviour, narrow and mechanical - not predictive validity",
         "claim": ("The applicability layer refuses taurine animals outside the Florida support region, "
                   "as its policy specifies."),
         "numbers": {"animals": 4240, "refusal_rate": 1.0},
         "why": "No centring is involved, so the C1 defect does not touch it.",
         "caveat": ("This verifies that the refusal policy is implemented and fires where intended. It is "
                    "not evidence that predictions for accepted animals are any good, and it should not "
                    "be cited as validation of the model."),
         "evidence": ["output/layer6c/results_mizzou.json"]},

        {"id": "C3", "axis": "applicability rule design", "status": "ESTABLISHED",
         "scope": "the applicability rule's refusal behaviour; says nothing about predictive accuracy",
         "confirmed_by": {
             "arm": "6C4, Decker worldwide panel (1,543 animals, 134 breeds, never previously scored)",
             "protocol": "docs/LAYER6C4_DECKER_PROTOCOL.md, locked over 22 files before any Decker "
                         "ancestry coordinate or support score existed",
             "E1_primary": {"animals": 1183, "qualifying_breeds": 106, "required_breeds": 10,
                            "knn_refusal_rate": 0.9755, "breed_bootstrap_ci95": [0.951, 0.9947],
                            "threshold": 0.90, "pass": True},
             "E3_generality": {"share_of_out_of_reference_breeds_inside_florida_range": 0.803,
                               "reading": "the Holstein case that produced C3 was typical, not "
                                          "exceptional: a two-reference estimator places most foreign "
                                          "breeds mid-axis, where min-max ancestry bounds accept them"},
             "false_refusal_control": "4.2% of in-domain Florida holdouts at the same coverage "
                                      "(docs/COVERAGE_CALIBRATION.md), so the rule discriminates "
                                      "rather than refusing everything",
             "internal_consistency": ("Angus (median q 0.007) and Brahman (1.000) are both refused "
                                      "100% and were excluded from endpoints, since a purebred sits "
                                      "outside a composite herd's support region. The only breeds with "
                                      "accepted animals are indicine x taurine composites near the "
                                      "middle of the axis - Beefmaster, Santa Gertrudis, Canchim - "
                                      "which is the behaviour a working rule should show."),
             "disclosures": [
                 "the coverage requirement enabling this arm was re-derived after this dataset's "
                 "coverage was known (Amendment 5), on Florida data only with the rule fixed first",
                 "the A/B consistency gate was respecified after seeing that the data pass it "
                 "(6C4 Amendment 1). The original gate measured sampling noise rather than coding "
                 "agreement and was unreachable by any correctly coded dataset of this shape; the "
                 "replacement flip test is categorical (0 of 2,891 markers differ by more than 0.5, "
                 "0 of 399 informative markers flipped) and no endpoint existed when it was changed. "
                 "A reviewer who finds this disqualifying should treat E1 as unconfirmed.",
             ]},
         "claim": ("The k-NN empirical support region refuses a population that the min-max ancestry "
                   "bounds it replaced would have accepted."),
         "numbers": {"external_animals": 99, "q_AB_axis_coordinate": 0.18,
                     "florida_training_q_min": 0.03, "florida_training_q_max": 0.91,
                     "training_share_in_0.15_0.21": 0.118,
                     "knn_refusal_shared_subspace": 1.0,
                     "sensitivity_grid": rob.get("summary"),
                     "ancestry_comparator": rob.get("ancestry_comparator"),
                     "projection_coordinate_stability": rob.get("projection_coordinate_stability")},
         "why": ("An out-of-reference breed (Holstein-Friesian) projected onto a two-way Angus-Brahman "
                 "axis lands mid-range, so ancestry bounds accept it. The support region does not."),
         "history": (
             "C3 was generated by a post-hoc diagnosis after the frozen 6C2 endpoint returned 0% "
             "refusal, and was classed POST_HOC_SUPPORTED on external review. A sensitivity map over "
             "k, PCA dimension and radius percentile then showed it was not a configuration artefact "
             "(separation held in all 80 configurations). It has since been confirmed on an "
             "independent panel under a pre-registered rule - see confirmed_by."),
         "terminology_warning": (
             "0.18 is a coordinate on the Angus<->Brahman axis under a two-reference model, NOT a Brahman "
             "ancestry fraction. A two-reference estimator cannot express 'not an Angus-Brahman animal', "
             "so it projects outsiders into the middle of the range. Subsampling stability shows the "
             "coordinate is reproducible; it says nothing about real ancestry."),
         "caveat": "Single external population, n=99, 25% panel coverage, close relatives.",
         "evidence": ["output/layer6c2/results_6c2_imputation_diagnosis.json",
                      "output/layer6c2/results_6c2_c3_robustness.json",
                      "docs/LAYER6C4_DECKER_PROTOCOL.md", "output/layer6c4/results_6c4.json",
                      "output/layer6c4/prepare_summary_6c4.json",
                      "reports/2026-09-16_layer6c4_results.md"]},

        {"id": "C4", "axis": "G_genomic (6C2 refusal endpoint)", "status": "VOID",
         "claim": "Pre-registered: >=95% of the hairy-syndrome animals are refused for ranking.",
         "numbers": {"as_run_ranking_refusal": 0.0, "pre_registered_target": 0.95,
                     "shared_subspace_refusal": 1.0, "florida_control_refusal_same_subspace": 0.0152,
                     "panel_coverage": 0.2523},
         "why": ("The frozen harmonisation clause fills unobserved markers with Florida training means. "
                 "At 25% coverage that places 74.8% of every external animal's genotype at the training "
                 "centroid, inside the space the support region measures. The run measured the "
                 "imputation, not the population."),
         "my_error": ("The feasibility check that licensed this endpoint (ancestry robust at r=0.939 "
                      "under masking) was run only on in-domain Florida animals, where mean-filling is "
                      "benign. A feasibility check for an out-of-domain claim must use out-of-domain "
                      "animals."),
         "rule_now_binding": (
             "Where markers are structurally absent because platforms differ, ancestry estimation and "
             "support assessment use the shared-marker subspace at ANY coverage (Amendment 4). The 80% "
             "coverage threshold introduced in Amendment 3 has been withdrawn: mean-filling is "
             "directionally biased for a distance-based support region at every coverage, so coverage "
             "was the wrong variable to gate on. Mean-filling is now confined to sporadically missing "
             "calls among markers the platform actually carries."),
         "what_void_does_not_mean": (
             "VOID says the frozen endpoint could not measure its estimand. It does NOT mean the post-hoc "
             "100% refusal is a confirmed pass. The frozen run measured 'is 25% Holstein + 75% Florida "
             "centroid outside Florida support?', which is a different estimand from 'is Holstein outside "
             "Florida support?'. Confirming the latter needs another external population under a rule "
             "fixed before its scores are seen - the same requirement as C3."),
         "evidence": ["output/layer6c2/results_6c2.json",
                      "output/layer6c2/results_6c2_imputation_diagnosis.json"]},

        {"id": "C5", "axis": "specificity", "status": "CONSISTENT_WITH_NULL",
         "claim": ("No detectable association between the Florida polygenic scores and hairy-syndrome "
                   "status. The design excludes large effects far better than modest ones."),
         "numbers": {"n_affected": 37, "n_unaffected": 62,
                     "smd_AvgShortLength": -0.031, "naive_ci": [-0.416, 0.372],
                     "detectable_smd_at_80pct_power": (fam.get("power") or {}).get(
                         "detectable_smd_at_80pct_power"),
                     "family_clustered": fam.get("traits"),
                     "family_structure": fam.get("family_structure")},
         "why_not_established": (
             "The frozen interval resampled affected and unaffected animals independently, but these 99 "
             "animals are progeny of three sires, so it was too narrow. Re-run as a sire-cluster "
             "bootstrap the AvgShortLength interval is 3.6x wider. Separately, the experiment could only "
             "detect |d| of about 0.58 at 80% power assuming independence, which is optimistic here, so "
             "the null is uninformative about modest effects."),
         "what_the_reanalysis_changed": (
             "Holding sire fixed moves the AvgShortLength point estimate from -0.03 to +0.32 (within-sire "
             "permutation p = 0.09). The near-zero naive estimate was partly sire confounding masking a "
             "within-family trend this design cannot resolve. That weakens the specificity reading rather "
             "than supporting it."),
         "caveat": ("Three sire clusters only; dam unrecorded for 77 of 99; coat status unevenly "
                    "distributed across sires; 25% coverage compresses the score toward zero."),
         "evidence": ["output/layer6c2/results_6c2.json",
                      "output/layer6c2/results_6c2_c5_family_clustered.json"]},

        {"id": "C6", "axis": "6C2 slick arm", "status": "NOT_TESTED",
         "claim": "Does the Florida genomic hair score recognise biologically slick cattle?",
         "numbers": {"animals": 82, "markers_available": 26, "panel_markers_required": 20087},
         "why": ("The slick crossbreds carry 26 chromosome-20 (PRLR) markers, so the polygenic score "
                 "cannot be computed for them at all. No substitute population was used."),
         "evidence": ["output/layer6c2/prepare_summary_6c2.json"]},

        {"id": "C7", "axis": "6C3 worldwide refusal gradient", "status": "INCONCLUSIVE",
         "claim": "Acceptance falls monotonically with ancestry distance across 134 world breeds.",
         "numbers": {"matched_by_name": 0, "matched_by_rsid": 142, "matched_by_lifted_position": 2891,
                     "required_minimum": 5000},
         "why": ("The pre-registered clause stops the arm below 5,000 matched panel markers. Nothing was "
                 "computed and no threshold was touched. The BovineSNP50 and the GGP-F250 used at "
                 "Florida are deliberately complementary designs, so the ceiling is real."),
         "evidence": ["output/layer6c3/results_6c3.json"]},

        {"id": "C8", "axis": "cross-assembly harmonisation", "status": "ESTABLISHED",
         "claim": "UMD3.1 datasets can be matched to the ARS-UCD1.2 panel by chain conversion.",
         "numbers": {"markers_lifted_share": 0.997, "matched_panel_markers": 5178,
                     "usable_after_allele_check": 5067, "allele_agreement": "5067/5067"},
         "why": ("Complete allele agreement among matched non-monomorphic markers is the evidence that "
                 "the conversion identifies the same variants rather than coincidental positions."),
         "evidence": ["output/layer6c2/prepare_summary_6c2.json", "tests/test_liftover.py"]},

        {"id": "C9", "axis": "G_phenotype (same trait, external)", "status": "NOT_AVAILABLE",
         "claim": "Absolute phenotype predictions transfer to an external herd.",
         "why": ("No external cohort with the same traits and protocol is in hand. The UF multibreed "
                 "request is drafted and unsent; the requester cannot obtain the data in reasonable time. "
                 "No proxy has been substituted."),
         "evidence": ["reports/2026-09-12_uf_phenotype_request.md"]},

        {"id": "C10", "axis": "G_inheritance (cross-species)", "status": "VOID",
         "claim": ("The meiosis simulation reproduces a real multiparental population's founder-mosaic "
                   "structure with no parameter fitted to that species."),
         "numbers": {"observed_mice": 291, "generations": [8, 11],
                     "E3_as_run": {"observed_ratio": 1.2197, "simulated_ratio": 1.4196,
                                   "relative_difference": 0.1408, "tolerance": 0.15,
                                   "pass_as_run": True, "withdrawn": True},
                     "switch_bounds_G8": [22.5, 140.15], "simulated_G8": 85.3768,
                     "switch_bounds_G11": [23.45, 165.71], "simulated_G11": 121.1971,
                     "transitions_that_are_gap_artefacts": 0.4459,
                     "transitions_that_are_clean_state_changes": 0.0816},
         "why": ("Run once under a lock of 67 files. The observed statistic counted switches as "
                 "(segments per haplotype) - (chromosomes); with minprob 0.95 leaving 13.2% of markers "
                 "uncalled, 91.8% of transitions involve a gap rather than an observed state change, "
                 "and 44.6% are one segment split by a gap with an IDENTICAL state on both sides. The "
                 "reported 247.5 switches per haplotype at G=8 exceeds even a valid upper bound of "
                 "140.2."),
         "my_error": ("minprob = 0.95 was chosen in the protocol without checking what share of markers "
                      "it would leave uncalled, or what an uncalled gap does to a segment-counting "
                      "statistic. Same class as the 6C2 feasibility gate: a measurement operator "
                      "pre-registered without checking that it measures the estimand under the "
                      "conditions it will meet."),
         "what_void_does_not_mean": (
             "Not evidence against the meiosis model. The simulated values fall inside the honest "
             "bounds at both generations - but those bounds span six-fold, so the test discriminates "
             "nothing either way. The E3 pass is withdrawn; under strict counting E3 fails at 0.266."),
         "caveat": ("Phase error is cleared as a cause (0.31% coincident switches). The inheritance "
                    "engine did run an 8-founder design unmodified, which is the one premise confirmed."),
         "v2_attempt": (
             "A new pre-registration (docs/LAYER7_H7v2_PROTOCOL.md) passed simulated genomes through the "
             "same reconstruction as the real mice and required a feasibility gate before any endpoint. "
             "The gate returned INFEASIBLE: no minprob on the pre-declared ladder met all three "
             "call-quality criteria on both cohorts. The binding criterion, gaps flanked by the same "
             "founder, stays at 11.9-44.6% on both cohorts at every rung, which suggests a structural "
             "limit of hard-call transition statistics on DO data (hypothesis: founder haplotype sharing). "
             "The gate also showed painting is not symmetric: real genotypes leave about twice the "
             "uncalled share of painted ones. v2 was never locked and computed no endpoint. C10 stays "
             "VOID for v1; for v2 the status is INFEASIBLE, which is a finding about the measurement, "
             "not about the meiosis model."),
         "evidence": ["docs/LAYER7_HYPOTHESIS.md", "docs/LAYER7_H7_PROTOCOL.md",
                      "output/layer7/results_h7.json", "output/layer7/results_h7_diagnosis.json",
                      "reports/2026-09-12_layer7_h7_results.md",
                      "docs/LAYER7_H7v2_PROTOCOL.md", "output/layer7v2/gate_result.json"]},

        {"id": "C12", "axis": "G_inheritance (within cattle)", "status": "ESTABLISHED",
         "scope": "predicting an offspring's GENOTYPE from its parents, not its phenotype",
         "claim": ("Offspring genotypes are recovered from the parents' genotypes better than from "
                   "breed allele frequencies, on real trios."),
         "numbers": {"trios": 171, "breeds": 7, "loci_per_trio": 199526,
                     "call_accuracy_from_parents": 0.7585, "call_accuracy_from_breed_frequencies": 0.6487,
                     "breeds_where_parents_win": "7 of 7",
                     "loglik_per_locus_parents": -0.3789, "loglik_per_locus_breed": -0.7255,
                     "loglik_per_locus_WRONG_parents": -1.9476,
                     "mid_parent_deviation_r": 0.7285, "midparent_child_r": 0.7121,
                     "heterozygosity_90pct_interval_coverage": 0.8480},
         "why": ("External Spanish-breed trios, EMPIRICAL. The falsification control is the "
                 "wrong-parents likelihood: substituting incorrect parents drops the per-locus "
                 "log-likelihood from -0.379 to -1.948, so the model is using real parental "
                 "transmission rather than population structure."),
         "caveat": ("This is genotype recovery, not phenotype prediction. It supports predicting what "
                    "a cross inherits - ancestry composition, heterozygosity, allele transmission - "
                    "and says nothing about the resulting trait value."),
         "evidence": ["output/audit_post_fix/real_layer5/layer5b_trio_recovery_summary.csv",
                      "reports/2026-09-12_v1_cattle_baseline.md"]},

        {"id": "C13", "axis": "G_inheritance (Mendelian correctness)", "status": "ESTABLISHED",
         "claim": "The meiosis engine segregates alleles at Mendelian ratios.",
         "numbers": {"heterozygote_x_heterozygote_locus_trios": 4321052,
                     "observed": [0.2493, 0.5008, 0.2499], "expected": [0.25, 0.50, 0.25]},
         "why": ("Measured on real trios rather than asserted. Forced-crossover toy genomes also give "
                 "exact A|B|A blocks and exact pedigree fractions."),
         "caveat": ("Correct segregation in cattle is necessary, not sufficient, for the inheritance "
                    "layer to be species-general: the cross-species test (C10) remains void."),
         "evidence": ["output/audit_post_fix/real_layer5/layer5b_trio_recovery_segregation.csv",
                      "reports/2026-09-12_v1_cattle_baseline.md"]},

        {"id": "C14", "axis": "G_genomic (internal, within Florida)", "status": "ESTABLISHED",
         "scope": "a NEGATIVE result, and the most decision-relevant one in the project",
         "claim": ("Genomic prediction does not beat breed composition for these traits. No model "
                   "exceeds an ancestry-only baseline with a paired interval clear of zero."),
         "numbers": {"snp_gblup_r_LoadSlope": 0.127, "snp_gblup_r_hair": 0.151, "markers": 122000,
                     "withheld_family_AncestryOnly_r": 0.325, "withheld_family_AdditiveGBLUP_r": 0.217,
                     "later_cohort_AncestryOnly_r": 0.177, "later_cohort_AdditiveGBLUP_r": 0.172,
                     "adding_BOA_to_SNP_hair_delta": -0.040, "adding_BOA_ci95": [-0.073, -0.011]},
         "why": ("Prospective and family-withheld splits, averaged across seeds. Ancestry-only matches "
                 "or beats every genomic model, decisively on withheld families. Adding breed-of-origin "
                 "to SNPs made hair accuracy significantly worse."),
         "caveat": ("This is a within-population result on one herd and two thermotolerance traits. It "
                    "does not generalise to all traits or all populations - but for THIS programme it "
                    "means breed percentages are currently doing the work."),
         "evidence": ["output/baseline_v1/stageD_prediction_benchmark.csv",
                      "reports/2026-09-12_v1_cattle_baseline.md"]},

        {"id": "C15", "axis": "mating allocation", "status": "CONSISTENT_WITH_NULL",
         "claim": "Genomic mating allocation outperforms ancestry-based mating.",
         "numbers": {"attainment_gain_points": [1.2, 1.4], "intervals_span_zero": True,
                     "P_genomic_better_than_ancestry": [0.65, 0.69]},
         "why": ("Every interval spans zero and the probability of superiority is about two in three, "
                 "which is a hint rather than a result. Consistent with C14: if genomic prediction does "
                 "not beat ancestry, genomic mating should not either."),
         "caveat": "EMPIRICAL breeding values, SIMULATED offspring: the outcome side is not observed.",
         "evidence": ["output/baseline_v1/stageE_policy_comparison.csv",
                      "reports/2026-09-12_v1_cattle_baseline.md"]},

        {"id": "C16", "axis": "G_genomic (machinery, second species)", "status": "ESTABLISHED",
         "scope": "the software works in another species; NOT cattle accuracy, NOT transfer",
         "claim": ("A GBLUP fitted and evaluated entirely within a second species predicts held-out "
                   "families, so the prediction code is species-general rather than a cattle artefact."),
         "numbers": {"mice": 1940, "families": 85, "markers": 10996, "traits": 8,
                     "E1_traits_with_interval_excluding_zero": 7, "E1_required": 4,
                     "E2_traits_beating_covariates": 5, "E2_required": 4,
                     "best_traits": {"Haem.MCV": 0.472, "Imm.CD4XGeoMean": 0.462, "Haem.WBC": 0.280},
                     "E3_support_accept_rate": 0.8853, "E3_required": 0.90, "E3_pass": False},
         "why": ("Pre-registered, locked over 34 files, unblinded once, on held-out FAMILIES. Traits "
                 "were chosen by a mechanical completeness rule so they could not be picked for being "
                 "likely to succeed. The genomic term adds most where heritability is known to be high "
                 "(red-cell volume, CD4) and nothing for homeostatic blood chemistry - an ordering a "
                 "broken implementation would not reproduce."),
         "caveat": ("E3 FAILED: the support rule refused 11.5% of held-out-family mice who are from the "
                    "SAME population, so it treats ordinary family structure as out-of-domain. This "
                    "echoes the Layer 6A over-refusal and is now an open defect seen in two species. "
                    "Also, Weight.GrowthRanIntercept (rho 0.72, delta 0.000) is an artefact - that "
                    "column IS a random-effect estimate from the source authors' growth model - and is "
                    "reported rather than dropped. Mouse accuracies are not comparable in magnitude "
                    "with cattle: held-out families still share the same 8 founders."),
         "evidence": ["docs/LAYER7_H7B_PROTOCOL.md", "output/layer7b/results_h7b.json",
                      "reports/2026-09-17_h7b_and_scale.md"]},

        {"id": "C17", "axis": "G_genomic (sample size)", "status": "ESTABLISHED",
         "scope": "methodological, NOT blind: these phenotypes were opened during 6C1",
         "claim": ("Genomic prediction accuracy is limited by training-set size, not by the "
                   "implementation. It scales as theory predicts and has not plateaued at 8,700 animals."),
         "numbers": {"animals_available": 10225, "markers": 20087,
                     "random_split_rho": {"250": 0.087, "2000": 0.239, "8000": 0.361, "8700": 0.381},
                     "held_out_herd_day_rho": {"250": 0.082, "2000": 0.198, "8000": 0.300},
                     "projected_animals_for_rho_0.35": 11327, "for_0.40": 15454, "for_0.50": 27045,
                     "florida_within_ancestry_h2_hair": [0.160, 0.100, 0.230],
                     "florida_within_ancestry_h2_loadslope": [0.133, 0.090, 0.183]},
         "why": ("Three strands agree: the code predicts in a second species (C16); the same code "
                 "scales properly on thousands of cattle; and Florida retains genomic heritability of "
                 "0.160 for hair length after herd and ancestry are fitted. So C14 is a statement "
                 "about 1,754 animals and these traits, not about the software."),
         "caveat": ("In Mizzou the ancestry baseline scores about zero throughout, because the "
                    "Angus-Brahman axis barely varies there (IQR 0.010-0.135 against Florida's "
                    "0.246-0.329). So this does NOT show genomic prediction would beat ancestry in "
                    "Florida at large n, where ancestry is a genuine competitor by design. Projections "
                    "beyond 8,700 animals are extrapolation, and the Florida heritabilities are upper "
                    "bounds because population structure inflates them in a crossbred herd."),
         "evidence": ["reports/2026-09-17_h7b_and_scale.md", "output/mizzou_scale/scale_curve.json",
                      "output/mizzou_scale/scale_law_fit.json",
                      "output/heritability/genomic_heritability.json"]},

        {"id": "C11", "axis": "G_genomic (cross-species)", "status": "NOT_AVAILABLE",
         "claim": "The prediction machinery behaves as designed in a second species.",
         "why": ("The Diversity Outbred archive on disk contains genotypes, founder genotypes, maps and "
                 "covariates, and no phenotype file of any kind. No public DO/CC dataset with a measured "
                 "coat or thermoregulation phenotype was found."),
         "evidence": ["docs/LAYER7_HYPOTHESIS.md"]},
    ]


def prior_reviews():
    """What earlier external reviews changed, so a new reviewer starts from the current state."""
    return [{
        "round": 1, "date": "2026-09-12", "reviewer": "external AI reviewer (bundle v1)",
        "accepted_and_acted_on": [
            {"finding": "C3 was labelled ESTABLISHED although it came from a post-hoc diagnosis after "
                        "the frozen endpoint failed, contradicting this bundle's own legend.",
             "action": "New status POST_HOC_SUPPORTED created; C3 moved to it. A sensitivity map over "
                       "k x PCA dimension x radius percentile was run and is embedded "
                       "(results_6c2_c3_robustness.json)."},
            {"finding": "C5 was labelled ESTABLISHED although the bootstrap resampled affected and "
                        "unaffected animals independently despite the animals being relatives, and the "
                        "design could only detect |d| ~ 0.58.",
             "action": "New status CONSISTENT_WITH_NULL created; C5 moved to it. Re-run with a "
                       "sire-cluster bootstrap and a within-sire permutation test "
                       "(results_6c2_c5_family_clustered.json). Holding sire fixed moved the estimate "
                       "from -0.03 to +0.32, p = 0.09, which weakens the specificity reading."},
            {"finding": "The 80% coverage threshold from Amendment 3 was uncalibrated and gated the "
                        "wrong variable: mean-filling biases a distance-based support region "
                        "directionally at any coverage, with no privileged cut at 80%.",
             "action": "Amendment 4 withdraws the threshold entirely. The rule is now by KIND of "
                       "missingness - structural platform mismatch always uses the shared-marker "
                       "subspace; mean-filling is confined to sporadic missing calls. A heuristic was "
                       "removed rather than calibrated."},
            {"finding": "The bundle claimed self-containment but omitted support.py, ood_detector.py, "
                        "canonical_split.py and ancestry_estimation.py, which its own review agenda "
                        "asked reviewers to inspect.",
             "action": "Those four modules plus harmonisation.py, the 6C runners and six test files are "
                       "now embedded. The build fails if any claim cites unembedded evidence."},
            {"finding": "q_B = 0.18 was described as a Brahman fraction for a population that is "
                        "neither Angus nor Brahman.",
             "action": "Renamed q_AB_axis throughout, with an explicit terminology warning, and its "
                       "stability under 50-80% marker subsampling measured."},
            {"finding": "C2 verifies policy implementation rather than predictive validity.",
             "action": "C2 keeps ESTABLISHED but now carries an explicit scope field: narrow and "
                       "mechanical, not validation of the model."},
        ],
        "agreed_without_change": [
            "C1 remains WITHDRAWN.",
            "C4 remains VOID; the reviewer judged the classification defensible and not self-serving, "
            "while noting that VOID does not make the post-hoc 100% refusal a confirmed pass. That "
            "distinction is now recorded in C4 itself.",
            "C8 remains ESTABLISHED on the 5,067/5,067 allele agreement.",
            "Layer 7 (H7) is deferred until the items above are done - the reviewer's recommended order, "
            "which matches ours.",
        ],
        "open_after_this_round": [
            "C3 and C4 both need a second out-of-reference population under a rule fixed in advance. No "
            "such dataset is currently in hand; the Decker panel cannot serve (2,891 of 5,000 markers).",
            "C1 is not independently auditable without reading canonical_split.py, now embedded; no new "
            "defect has been demonstrated, which is not the same as cleared.",
        ],
    }]


def review_agenda():
    """What an external reviewer should attack first, including what we already doubt."""
    return {
        "how_to_use_this_bundle": (
            "Every claim in `claims` carries a status, the numbers, and the embedded file that produced "
            "them (see `embedded_files`). Check the numbers against the embedded evidence rather than "
            "against the prose. Where a claim's `my_error` or `caveat` field is populated, the weakness "
            "is already known and stated - confirming it is not a finding; showing it is worse than "
            "stated, or that a claim without such a field has one, is."),
        "highest_value_targets": [
            {"target": "C3, still the only positive external finding",
             "ask": ("The sensitivity map over k x PCA dimension x radius percentile is embedded - read "
                     "`summary` in results_6c2_c3_robustness.json before anything else. If the separation "
                     "R_external >> R_florida holds across the grid, the finding is not a configuration "
                     "artefact, but it remains POST_HOC_SUPPORTED: it was generated after the frozen "
                     "endpoint failed. Is there any way to obtain confirmatory evidence from data already "
                     "public, given that the Decker panel fails the marker-overlap requirement?")},
            {"target": "Amendment 4, the replacement for the withdrawn 80% rule",
             "ask": ("The new rule distinguishes structural platform missingness from sporadic missing "
                     "calls, and applies the shared-subspace treatment at any coverage. Is the "
                     "structural/sporadic distinction always decidable in practice? What happens with "
                     "imputed genotypes, which are neither cleanly?")},
            {"target": "C5 after the family-clustered re-analysis",
             "ask": ("Holding sire fixed moved the estimate from -0.03 to +0.32 with permutation p = 0.09. "
                     "Is CONSISTENT_WITH_NULL still the right label, or does a within-family trend at "
                     "p = 0.09 on 99 animals deserve its own status? Note the sire-cluster bootstrap has "
                     "only three clusters and its interval should not be trusted.")},
            {"target": "C1, the withdrawn result",
             "ask": ("`src/real_data/canonical_split.py` is now embedded. Does the production adjustment "
                     "path implement the contemporary-group rule the documents describe - Set nested "
                     "within Project, calibration pool of one Project, unseen held-out Sets left "
                     "unadjusted? A defect of the same kind as the Mizzou one would be the highest-impact "
                     "finding available in this bundle.")},
            {"target": "the claim statuses themselves",
             "ask": ("Two statuses were created in response to the first review round. Are they being "
                     "used consistently, or is POST_HOC_SUPPORTED absorbing claims that should be "
                     "INCONCLUSIVE?")},
        ],
        "known_open_items": [
            "C3 and C4 both need confirmation on a second out-of-reference population under a rule fixed "
            "before its support scores are seen. No suitable dataset is in hand.",
            "The sire-cluster bootstrap for C5 rests on three clusters; its interval is reported but "
            "should not be relied on.",
            "H8: the Florida source repository states no licence; permission is needed before publication.",
            "E14: the coalescent engine is a labelled synthetic null/stress-test generator, not a realistic "
            "cattle population simulator, and its biology has not been validated.",
            "The heuristic OOD thresholds outside the risk-calibrated pair remain untuned.",
            "Risk calibration of the support threshold was impossible on Florida data: risk is flat in the "
            "support score and all excess risk is environmental.",
        ],
        "standing_rules_this_project_follows": [
            "No result survives merely because the test suite passes: software correctness, statistical "
            "validity and evidence provenance are three separate gates.",
            "Pre-register and hash before unblinding; one run per arm; deviations are documented, not hidden.",
            "Never tune thresholds, panels or models on external data.",
            "Label unavailable validations as unavailable; never substitute a proxy and call it equivalent.",
            "Frozen baselines are never edited; corrections are recorded as superseding files.",
        ],
        "reproduction": [
            "python -m pytest -q",
            "python scripts/run_layer6c2_hairy_probe.py --stage prepare",
            "python scripts/run_layer6c2_imputation_diagnosis.py",
            "python scripts/run_layer6c2_c3_robustness.py",
            "python scripts/run_layer6c2_c5_family_clustered.py",
            "python scripts/run_layer6c1_closing_test.py",
            "Note: the 6C2 unblind stage will refuse to re-run, because Amendment 3 changed the protocol "
            "document its lock hashes. That is intended: the lock records the specification as it stood "
            "at unblinding.",
        ],
        "what_is_not_in_this_bundle": (
            "No genotype or phenotype data. The Florida source data carries no licence (H8) and the "
            "external datasets are redistributable only from their own repositories. File hashes are "
            "included so a reviewer can verify identical inputs."),
    }


def embed(paths):
    """path -> {sha256, bytes, text} for each file that exists."""
    out = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            out[rel] = {"present": False}
            continue
        text = p.read_text(encoding="utf-8")
        out[rel] = {"present": True, "sha256": sha256_file(p), "bytes": p.stat().st_size,
                    "content": json.loads(text) if p.suffix == ".json" else text}
    return out


def build():
    bundle = {
        "bundle": "CrossBreedin Algorithm Framework - external review package",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/export_review_bundle.py",
        "purpose": ("Self-contained record of what this project currently claims and on what evidence, "
                    "for independent review. Everything cited is embedded; no repository access needed."),
        "reading_order": ["status_legend", "summary", "claims", "review_agenda", "embedded_files"],
        "status_legend": STATUS,
        "summary": {
            "project": ("Crossbreeding prediction framework: genomic prediction (GBLUP on a VanRaden GRM), "
                        "mating allocation, and an applicability-domain layer that refuses predictions "
                        "outside its empirical support region."),
            "training_data": ("Florida multibreed Angus-Brahman herd, 3,194 genotyped animals; canonical "
                              "prospective split trains on 1,754 animals from 2016-2018 and tests on 658 "
                              "from 2021-2022."),
            "headline": (
                "No external predictive transfer result is established in any species. What IS "
                "established splits in two: the applicability layer refuses out-of-domain animals "
                "correctly (C3, confirmed on an independent world panel), and offspring GENOTYPES are "
                "recovered from parents better than from breed frequencies on real trios (C12). The "
                "most decision-relevant finding is negative: genomic prediction does not beat breed "
                "composition for these traits (C14), and genomic mating shows no established advantage "
                "over ancestry-based mating (C15)."),
            "revision": ("Bundle v2, revised after an external review round. See prior_review_rounds for "
                         "what changed and why; two claim statuses were downgraded and one protocol "
                         "heuristic was withdrawn."),
            "external_datasets_used": [
                "Mizzou hair shedding (Dryad 10.5061/dryad.ngf1vhhz4, CC0)",
                "Littlejohn et al. 2014 slick/hairy (Dryad 10.5061/dryad.nh6v4, CC0)",
                "Decker et al. 2014 worldwide (Dryad 10.5061/dryad.th092, CC0)",
                "Seminole steers (ENA ERZ24663392)",
                "Svenson Diversity Outbred mice (R/qtl2 sample data; genotypes only)",
            ],
        },
        "claims": claims(),
        "prior_review_rounds": prior_reviews(),
        "review_agenda": review_agenda(),
        "embedded_files": {
            "documents": embed(DOCS),
            "reports": embed(REPORTS),
            "results": embed(RESULTS),
            "protocol_locks": embed(LOCKS),
            "code": embed(CODE),
            "tests": embed(TESTS),
        },
    }
    return bundle


def to_markdown(b):
    L = [f"# {b['bundle']}", "",
         f"Generated {b['generated_utc']} by `{b['generated_by']}`.", "",
         b["purpose"], "",
         "## Summary", "",
         b["summary"]["project"], "",
         f"**Training data.** {b['summary']['training_data']}", "",
         f"**Headline.** {b['summary']['headline']}", "",
         f"**Revision.** {b['summary']['revision']}", "",
         "## Status legend", ""]
    L += [f"- **{k}** — {v}" for k, v in b["status_legend"].items()]
    L += ["", "## Claims", "",
          "| ID | Axis | Status | Claim |", "|---|---|---|---|"]
    for c in b["claims"]:
        L.append(f"| {c['id']} | {c['axis']} | **{c['status']}** | {c['claim']} |")
    L += ["", "### Detail", ""]
    for c in b["claims"]:
        L += [f"#### {c['id']} — {c['status']}", "", f"*{c['claim']}*", ""]
        if c.get("numbers"):
            L += ["```json", json.dumps(c["numbers"], indent=2), "```", ""]
        for key, title in (("scope", "Scope"), ("history", "History"),
                           ("confirmed_by", "Confirmed by"), ("why", "Why"), ("confound", "Confound"),
                           ("why_not_established", "Why not ESTABLISHED"),
                           ("what_the_reanalysis_changed", "What the re-analysis changed"),
                           ("my_error", "Our error"),
                           ("what_void_does_not_mean", "What VOID does not mean"),
                           ("v2_attempt", "Second attempt (v2)"),
                           ("terminology_warning", "Terminology warning"),
                           ("confirmation_requires", "Confirmation requires"),
                           ("confirmation_in_progress", "Confirmation in progress"),
                           ("audit_status", "Audit status"),
                           ("caveat", "Caveat"), ("rule_now_binding", "Rule now binding")):
            if c.get(key):
                L += [f"**{title}.** {c[key]}", ""]
        L += ["Evidence: " + ", ".join(f"`{e}`" for e in c["evidence"]), ""]
    for r in b["prior_review_rounds"]:
        L += [f"## Prior review round {r['round']} ({r['date']}, {r['reviewer']})", "",
              "Changes made in response:", ""]
        for item in r["accepted_and_acted_on"]:
            L += [f"- **Finding.** {item['finding']}", f"  - **Action.** {item['action']}"]
        L += ["", "Agreed without change:", ""] + [f"- {x}" for x in r["agreed_without_change"]]
        L += ["", "Still open after that round:", ""] + [f"- {x}" for x in r["open_after_this_round"]] + [""]
    a = b["review_agenda"]
    L += ["## How to review this", "", a["how_to_use_this_bundle"], "", "### Highest-value targets", ""]
    for t in a["highest_value_targets"]:
        L += [f"- **{t['target']}** — {t['ask']}"]
    L += ["", "### Known open items", ""] + [f"- {x}" for x in a["known_open_items"]]
    L += ["", "### Standing rules this project follows", ""] + [f"- {x}" for x in a["standing_rules_this_project_follows"]]
    L += ["", "### Reproduction", ""] + [f"- `{x}`" if not x.startswith("Note:") else f"- {x}"
                                         for x in a["reproduction"]]
    L += ["", "### What is not in this bundle", "", a["what_is_not_in_this_bundle"], "",
          "## Embedded files", "",
          "The JSON bundle embeds each of these in full, with its SHA-256.", "",
          "| File | SHA-256 | Bytes |", "|---|---|---|"]
    for group in b["embedded_files"].values():
        for rel, meta in group.items():
            if meta.get("present"):
                L.append(f"| `{rel}` | `{meta['sha256'][:16]}…` | {meta['bytes']:,} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, help="Also copy the bundle to this folder")
    args = ap.parse_args()

    bundle = build()
    OUT.mkdir(parents=True, exist_ok=True)
    js, md = OUT / "review_bundle.json", OUT / "REVIEW.md"
    js.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    md.write_text(to_markdown(bundle), encoding="utf-8")

    # Self-containment is the bundle's whole promise, so it is checked rather than assumed. A file
    # that is listed but absent does not count: that is exactly how the first bundle shipped claims
    # whose evidence a reviewer could not read.
    embedded = {rel for group in bundle["embedded_files"].values()
                for rel, meta in group.items() if meta.get("present")}
    dangling = sorted({e for c in bundle["claims"] for e in c["evidence"] if e not in embedded})
    if dangling:
        raise SystemExit(f"claims cite evidence that is not embedded: {dangling}")

    missing = [rel for group in bundle["embedded_files"].values()
               for rel, meta in group.items() if not meta.get("present")]
    n_embedded = sum(1 for group in bundle["embedded_files"].values()
                     for meta in group.values() if meta.get("present"))
    print(f"claims: {len(bundle['claims'])}   embedded files: {n_embedded}   "
          f"missing: {missing if missing else 'none'}")
    print(f"wrote {js.relative_to(ROOT)} ({js.stat().st_size:,} bytes)")
    print(f"wrote {md.relative_to(ROOT)} ({md.stat().st_size:,} bytes)")
    if args.dest:
        args.dest.mkdir(parents=True, exist_ok=True)
        for p in (js, md):
            shutil.copy2(p, args.dest / p.name)
            print(f"copied {p.name} -> {args.dest / p.name}")


if __name__ == "__main__":
    main()

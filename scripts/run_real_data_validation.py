"""Real-Data-Compatible Genomic Pipeline — Synthetic Benchmark Population Validation (Mini Cattle Genomic Benchmark).

Evaluates the real-data pipeline architecture on a controlled, synthetic cattle population:
1. Individual-level genotype ingestion from PLINK .bed/.bim/.fam with BGD/AnimalQTLdb prior mapping & QC filtering.
2. Synthetic Mendelian transmission validation: verifying that the validator recovers the simulator's
   Mendelian error rate, transition probability matrix calibration, and autosome crossover distribution.
3. Phenotype modeling with synthetic management covariates: regression test verifying recovery of known
   covariates (age, sex, feeding regimen, contemporary groups).
4. Out-of-sample predictive benchmark on withheld synthetic offspring:
   - Ancestry-only baseline
   - Additive GBLUP (Genomic Relationship Matrix)
   - Non-linear RKHS
   - Overall cohort, Case 1 (Brangus crosses), and Case 2 (Wagyu x Angus crosses).
5. Joint probability target attainment and Brier score calibration.
6. Export of CSV summaries and diagnostic figures.

NOTE: This benchmark validates software pipeline correctness and algorithmic integration under known ground truth.
It does NOT constitute biological empirical proof until evaluated on genuine physical cattle datasets.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.bgd_qtl_loader import BGDQtlLoader
from src.real_data.empirical_benchmark import EmpiricalBenchmark, EmpiricalEvaluationResult
from src.real_data.environmental_model import EnvironmentalCovariateModel, EnvironmentalFitResult
from src.real_data.phenotype_loader import PhenotypeLoader
from src.real_data.plink_parser import PlinkData, PlinkParser, PlinkQC
from src.real_data.transmission_validator import TransmissionValidator, TrioValidationReport


def run_empirical_validation():
    print("=" * 80)
    print("STAGE 0/1: REAL-DATA-COMPATIBLE PIPELINE — SYNTHETIC POPULATION BENCHMARK")
    print("  (Mini Cattle Genomic Benchmark: 300 cattle, 500 SNPs, ARS-UCD1.2 Coordinates)")
    print("=" * 80)

    data_dir = BASE_DIR / "data" / "synthetic_cattle"
    output_dir = BASE_DIR / "output" / "empirical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Genotype Ingestion, QC Filtering, and BGD QTL Mapping
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Loading Individual-Level Genotypes (PLINK format) & BGD Annotation...")
    plink_prefix = data_dir / "synthetic_cattle_50k_population"
    raw_plink = PlinkParser.read_plink(plink_prefix)
    from src.real_data.provenance import evidence_label
    print(f"  Evidence source: {evidence_label(raw_plink.provenance)} (docs/EVIDENCE_POLICY.md)")
    print(f"  Loaded raw dataset: {raw_plink.num_individuals} cattle, {raw_plink.num_snps} SNPs across 29 autosomes.")

    # Apply Quality Control (MAF > 0.03, Call Rate > 0.95)
    qc_plink, qc_report = PlinkQC.filter(raw_plink, min_maf=0.03, max_missing_snp=0.05, max_missing_ind=0.05)
    ind_removed = qc_report['initial_individuals'] - qc_report['passed_individuals']
    snp_removed = qc_report['initial_snps'] - qc_report['passed_snps']
    print(f"  QC passed: {qc_report['passed_individuals']} individuals ({ind_removed} removed), "
          f"{qc_report['passed_snps']} SNPs ({snp_removed} removed).")

    # Map to Bovine Genome Database (BGD) and AnimalQTLdb verified intervals
    bgd = BGDQtlLoader()
    weights_growth = bgd.map_snps_to_qtls(qc_plink.bim, trait_filter="Carcass")  # was "CarcassWeight", which matched nothing
    weights_marb = bgd.map_snps_to_qtls(qc_plink.bim, trait_filter="Marbling")
    print(f"  BGD QTL Annotation: {np.sum(weights_growth > 1.0)} SNPs in Carcass Weight QTLs (LCORL, PLAG1, MSTN), "
          f"{np.sum(weights_marb > 1.0)} SNPs in Marbling QTLs (DGAT1, SCD, TG5).")

    # -------------------------------------------------------------------------
    # 2. Parent-Offspring Trio Transmission & Recombination Validation
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Validating Parent-Offspring Transmission & Crossover Distribution...")
    validator = TransmissionValidator(qc_plink)
    trio_report = validator.validate_transmission()

    print(f"  Trios evaluated: {trio_report.num_trios_evaluated}")
    print(f"  Total locus evaluations: {trio_report.total_locus_evaluations:,}")
    print(f"  Mendelian errors detected: {trio_report.mendelian_error_count} "
          f"({trio_report.mendelian_error_rate * 100:.3f}% error rate)")
    print(f"  Transition probability discrepancy RMSE: {trio_report.theoretical_discrepancy_rmse:.4f}")
    print(f"  Calibration status: {'PASSED (Calibrated)' if trio_report.is_calibrated else 'FAILED'}")

    # Export trio transmission matrix
    trans_rows = []
    for (s_val, d_val), emp_probs in sorted(trio_report.transition_probabilities.items()):
        theo = TransmissionValidator.THEORETICAL_MENDELIAN[(s_val, d_val)]
        trans_rows.append({
            "sire_genotype": s_val,
            "dam_genotype": d_val,
            "theo_P_0": theo[0],
            "theo_P_1": theo[1],
            "theo_P_2": theo[2],
            "emp_P_0": round(float(emp_probs[0]), 4),
            "emp_P_1": round(float(emp_probs[1]), 4),
            "emp_P_2": round(float(emp_probs[2]), 4),
        })
    df_trans = pd.DataFrame(trans_rows)
    df_trans.to_csv(output_dir / "trio_transition_calibration.csv", index=False)
    trio_report.crossover_summary_by_chrom.to_csv(output_dir / "recombination_by_chrom.csv", index=False)

    # -------------------------------------------------------------------------
    # 3. Phenotypic Modeling & Environmental Adjustment
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Loading Matched Carcass Phenotypes & Fitting Environmental Model...")
    pheno_file = data_dir / "synthetic_cattle_carcass_phenotypes.csv"
    dataset = PhenotypeLoader.load_from_csv(pheno_file)
    matched_pheno, matched_geno = dataset.align_with_plink(qc_plink)
    print(f"  Matched records: {len(matched_pheno)} cattle with complete genotypes, phenotypes, and covariates.")

    train_mask = (matched_pheno["cohort"] == "Cohort1").values
    test_mask = (matched_pheno["cohort"] == "WithheldCohort2").values
    env_model = EnvironmentalCovariateModel(
        continuous_covariates=["age_days"],
        categorical_covariates=["sex", "contemporary_group", "feed_regimen"]
    )

    fit_growth = env_model.fit_and_adjust(matched_pheno.loc[train_mask], "growth")
    fit_marb = env_model.fit_and_adjust(matched_pheno.loc[train_mask], "marbling")

    print(f"  [Growth/Carcass Weight] Env R2: {fit_growth.r2_environment:.3f} | "
          f"Var(Raw): {fit_growth.variance_raw:.1f} -> Var(Adj): {fit_growth.variance_adjusted:.1f} "
          f"({(1 - fit_growth.variance_adjusted / fit_growth.variance_raw) * 100:.1f}% variance removed)")
    print(f"  [Marbling Score]       Env R2: {fit_marb.r2_environment:.3f} | "
          f"Var(Raw): {fit_marb.variance_raw:.3f} -> Var(Adj): {fit_marb.variance_adjusted:.3f} "
          f"({(1 - fit_marb.variance_adjusted / fit_marb.variance_raw) * 100:.1f}% variance removed)")

    env_summary = pd.DataFrame([
        {
            "trait": "CarcassWeight_kg",
            "r2_environment": round(fit_growth.r2_environment, 4),
            "variance_raw": round(fit_growth.variance_raw, 2),
            "variance_adjusted": round(fit_growth.variance_adjusted, 2),
            "intercept": round(fit_growth.intercept, 2),
            "age_effect_kg_per_day": round(fit_growth.coefficients.get("age_days", 0.0), 3),
            "sex_steer_effect_kg": round(fit_growth.coefficients.get("sex_steer", 0.0), 2),
            "grain_feed_effect_kg": round(fit_growth.coefficients.get("feed_regimen_pasture", 0.0), 2)
        },
        {
            "trait": "Marbling_BMS",
            "r2_environment": round(fit_marb.r2_environment, 4),
            "variance_raw": round(fit_marb.variance_raw, 4),
            "variance_adjusted": round(fit_marb.variance_adjusted, 4),
            "intercept": round(fit_marb.intercept, 2),
            "age_effect_kg_per_day": round(fit_marb.coefficients.get("age_days", 0.0), 4),
            "sex_steer_effect_kg": round(fit_marb.coefficients.get("sex_steer", 0.0), 3),
            "grain_feed_effect_kg": round(fit_marb.coefficients.get("feed_regimen_pasture", 0.0), 3)
        }
    ])
    env_summary.to_csv(output_dir / "environmental_fit_summary.csv", index=False)

    matched_pheno["growth_adj"] = env_model.transform(matched_pheno, "growth")
    matched_pheno["marbling_adj"] = env_model.transform(matched_pheno, "marbling")

    # -------------------------------------------------------------------------
    # 4. Out-of-Sample Prediction on Withheld Cohort 2
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Out-of-Sample Prediction on Withheld Real Offspring...")

    # Training set = Cohort 1; Test set = Withheld Cohort 2


    ancestry_cols = ["breed_Wagyu", "breed_Angus", "breed_Brahman"]
    ancestry_all = matched_pheno[ancestry_cols].values

    train_ancestry = ancestry_all[train_mask]
    test_ancestry = ancestry_all[test_mask]

    train_geno = matched_geno[train_mask]
    test_geno = matched_geno[test_mask]

    train_growth = matched_pheno.loc[train_mask, "growth_adj"].values
    test_growth = matched_pheno.loc[test_mask, "growth_adj"].values

    train_marb = matched_pheno.loc[train_mask, "marbling_adj"].values
    test_marb = matched_pheno.loc[test_mask, "marbling_adj"].values

    # Joint targets: training-cohort medians, so withheld outcomes cannot move the thresholds
    target_growth_overall = float(np.median(train_growth))
    target_marb_overall = float(np.median(train_marb))

    results_overall = EmpiricalBenchmark.run_benchmark(
        train_ancestry=train_ancestry,
        train_markers=train_geno,
        train_growth=train_growth,
        train_marbling=train_marb,
        test_ancestry=test_ancestry,
        test_markers=test_geno,
        test_growth=test_growth,
        test_marbling=test_marb,
        target_growth=target_growth_overall,
        target_marbling=target_marb_overall
    )

    # -------------------------------------------------------------------------
    # 5. Case-Specific Sub-Evaluations: Case 1 (Brangus) & Case 2 (Wagyu x Angus)
    # -------------------------------------------------------------------------
    print("\n[Stage 5] Evaluating Within-Case Predictions (Brangus & Wagyu x Angus)...")

    # Filter Case 1: Brangus crosses (Angus + Brahman ancestry, no Wagyu)
    c1_test_mask = test_mask & (matched_pheno["breed_Wagyu"] == 0.0).values
    c1_train_mask = train_mask & (matched_pheno["breed_Wagyu"] == 0.0).values

    # Commercial Brangus joint target: Carcass Weight >= 385 kg, Marbling >= 5.0
    results_case1 = EmpiricalBenchmark.run_benchmark(
        train_ancestry=ancestry_all[c1_train_mask],
        train_markers=matched_geno[c1_train_mask],
        train_growth=matched_pheno.loc[c1_train_mask, "growth_adj"].values,
        train_marbling=matched_pheno.loc[c1_train_mask, "marbling_adj"].values,
        test_ancestry=ancestry_all[c1_test_mask],
        test_markers=matched_geno[c1_test_mask],
        test_growth=matched_pheno.loc[c1_test_mask, "growth_adj"].values,
        test_marbling=matched_pheno.loc[c1_test_mask, "marbling_adj"].values,
        target_growth=385.0,
        target_marbling=5.0
    )

    # Filter Case 2: Wagyu x Angus crosses (Wagyu + Angus ancestry, no Brahman)
    c2_test_mask = test_mask & (matched_pheno["breed_Brahman"] == 0.0).values
    c2_train_mask = train_mask & (matched_pheno["breed_Brahman"] == 0.0).values

    # Premium Wagyu x Angus target: Carcass Weight >= 380 kg, Marbling >= 6.0
    results_case2 = EmpiricalBenchmark.run_benchmark(
        train_ancestry=ancestry_all[c2_train_mask],
        train_markers=matched_geno[c2_train_mask],
        train_growth=matched_pheno.loc[c2_train_mask, "growth_adj"].values,
        train_marbling=matched_pheno.loc[c2_train_mask, "marbling_adj"].values,
        test_ancestry=ancestry_all[c2_test_mask],
        test_markers=matched_geno[c2_test_mask],
        test_growth=matched_pheno.loc[c2_test_mask, "growth_adj"].values,
        test_marbling=matched_pheno.loc[c2_test_mask, "marbling_adj"].values,
        target_growth=380.0,
        target_marbling=6.0
    )

    # Build comparison summary table
    def results_to_df(res_dict: dict, label: str) -> pd.DataFrame:
        rows = []
        for m_name, res in res_dict.items():
            rows.append({
                "subset": label,
                "model": m_name,
                "growth_rmse": round(res.growth_rmse, 2),
                "growth_r": round(res.growth_r, 3),
                "growth_rho": round(res.growth_rho, 3),
                "marbling_rmse": round(res.marbling_rmse, 3),
                "marbling_r": round(res.marbling_r, 3),
                "marbling_rho": round(res.marbling_rho, 3),
                "brier_score_joint": round(res.brier_score_joint, 4),
                "auc_joint": round(res.auc_joint, 3),
                "pred_attainment_pct": round(res.predicted_attainment_mean * 100, 1),
                "obs_attainment_pct": round(res.observed_attainment_mean * 100, 1)
            })
        return pd.DataFrame(rows)

    df_overall = results_to_df(results_overall, "Overall Population")
    df_case1 = results_to_df(results_case1, "Case 1: Brangus")
    df_case2 = results_to_df(results_case2, "Case 2: Wagyu x Angus")

    combined_df = pd.concat([df_overall, df_case1, df_case2], ignore_index=True)
    combined_df.to_csv(output_dir / "empirical_model_benchmark_all.csv", index=False)

    print("\n--- Summary: Prediction on Withheld Calves ---")
    print(combined_df[["subset", "model", "growth_r", "marbling_r", "brier_score_joint", "auc_joint"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Generate Publication Figures
    # -------------------------------------------------------------------------
    print("\n[Stage 6] Generating Validation Visualizations...")

    # Figure 1: Trio Transmission Calibration
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Transition probability comparison
    pair_labels = [f"{s}x{d}" for (s, d) in sorted(trio_report.transition_probabilities.keys())]
    emp_het_probs = [trio_report.transition_probabilities[(s, d)][1] for (s, d) in sorted(trio_report.transition_probabilities.keys())]
    theo_het_probs = [TransmissionValidator.THEORETICAL_MENDELIAN[(s, d)][1] for (s, d) in sorted(trio_report.transition_probabilities.keys())]

    x = np.arange(len(pair_labels))
    width = 0.35
    axes[0].bar(x - width/2, theo_het_probs, width, label="Theoretical Mendelian", color="#3b82f6", alpha=0.85)
    axes[0].bar(x + width/2, emp_het_probs, width, label="Simulated Observed", color="#10b981", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("P(Offspring = Heterozygote / 1)")
    axes[0].set_title(f"Allele Transmission Calibration (RMSE = {trio_report.theoretical_discrepancy_rmse:.4f})")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Opposite-homozygote jumps by chromosome (not a crossover count; see TransmissionValidator)
    chrom_df = trio_report.crossover_summary_by_chrom
    axes[1].bar(chrom_df["chrom"].astype(int), chrom_df["mean_opposite_homozygote_jumps_per_calf"], color="#8b5cf6", alpha=0.8)
    axes[1].axhline(1.0, color="#ef4444", linestyle="--", label="Expected Bovine Rate (~1.0 CO/chr)")
    axes[1].set_xlabel("Bovine Chromosome (BTA 1 - 29)")
    axes[1].set_ylabel("Opposite-homozygote jumps per calf")
    axes[1].set_title("Synthetic data: opposite-homozygote jumps per autosome (not crossovers)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig1_path = output_dir / "synthetic_transmission_calibration.png"
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.savefig(output_dir / "empirical_transmission_calibration.png")
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Environmental Adjustment vs Raw Variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].hist(matched_pheno["growth"], bins=20, alpha=0.55, color="#ef4444", label=f"Raw Growth (Var={fit_growth.variance_raw:.1f})")
    axes[0].hist(matched_pheno["growth_adj"], bins=20, alpha=0.65, color="#2563eb", label=f"Adjusted Growth (Var={fit_growth.variance_adjusted:.1f})")
    axes[0].set_xlabel("Carcass Weight (kg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Growth Environmental Pre-adjustment (R²_env = {fit_growth.r2_environment:.3f})")
    axes[0].legend()
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].hist(matched_pheno["marbling"], bins=20, alpha=0.55, color="#f59e0b", label=f"Raw Marbling (Var={fit_marb.variance_raw:.2f})")
    axes[1].hist(matched_pheno["marbling_adj"], bins=20, alpha=0.65, color="#059669", label=f"Adjusted Marbling (Var={fit_marb.variance_adjusted:.2f})")
    axes[1].set_xlabel("Marbling Score (BMS)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Marbling Environmental Pre-adjustment (R²_env = {fit_marb.r2_environment:.3f})")
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    fig2_path = output_dir / "synthetic_environmental_adjustment.png"
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.savefig(output_dir / "empirical_environmental_adjustment.png")
    plt.close()
    print(f"  Saved: {fig2_path}")

    # Figure 3: Model Prediction Comparison on Withheld Offspring
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    models = ["AncestryOnly", "AdditiveGBLUP", "NonlinearRKHS"]
    growth_corrs = [results_overall[m].growth_r for m in models]
    marb_corrs = [results_overall[m].marbling_r for m in models]
    brier_scores = [results_overall[m].brier_score_joint for m in models]

    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w/2, growth_corrs, w, label="Growth Accuracy (r)", color="#3b82f6")
    axes[0].bar(x + w/2, marb_corrs, w, label="Marbling Accuracy (r)", color="#10b981")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Pearson Correlation (r) on Withheld Calves")
    axes[0].set_title("Genomic vs Ancestry-Only Out-of-Sample Accuracy")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(models, brier_scores, color=["#94a3b8", "#6366f1", "#8b5cf6"], width=0.5)
    axes[1].set_ylabel("Joint Target Brier Score (Lower is Better)")
    axes[1].set_title("Probability Calibration on Meeting Joint Target")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for idx, b in enumerate(brier_scores):
        axes[1].text(idx, b + 0.005, f"{b:.4f}", ha="center", fontweight="bold")

    fig3_path = output_dir / "synthetic_model_benchmark.png"
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.savefig(output_dir / "empirical_model_benchmark.png")
    plt.close()
    print(f"  Saved: {fig3_path}")


    print("\n" + "=" * 80)
    print("SYNTHETIC POPULATION BENCHMARK COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_empirical_validation()

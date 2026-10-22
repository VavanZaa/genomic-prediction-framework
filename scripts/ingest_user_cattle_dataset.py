"""Drop-in ingestion runner for user-supplied real cattle datasets (Layer 3).

Accepts any genuine cattle dataset placed into data/real_cattle/ or custom directory:
- Genotypes: PLINK binary (.bed, .bim, .fam) or text (.ped, .map)
- Phenotypes: CSV with animal ID, growth/carcass trait, marbling trait, and optional covariates
- Runs QC filtering, BGD/AnimalQTLdb prior mapping, environmental adjustment, and GBLUP/RKHS prediction.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.bgd_qtl_loader import BGDQtlLoader
from src.real_data.empirical_benchmark import EmpiricalBenchmark
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.real_data.phenotype_loader import PhenotypeLoader
from src.real_data.plink_parser import PlinkData, PlinkParser, PlinkQC
from src.real_data.transmission_validator import TransmissionValidator


def ingest_and_evaluate(
    plink_prefix: Path,
    phenotype_csv: Path,
    output_dir: Path,
    growth_col: str = "growth",
    marbling_col: str = "marbling",
    id_col: str = "id",
    min_maf: float = 0.03,
    train_fraction: float = 0.60
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("CATTLE DATASET INGESTION & EVALUATION (default example data are synthetic)")
    print("=" * 80)
    print(f"Genotypes: {plink_prefix}.bed/.bim/.fam")
    print(f"Phenotypes: {phenotype_csv}")
    print(f"Output: {output_dir}")

    # 1. Load Genotypes
    print("\n[Step 1] Parsing Genotype Files...")
    raw_data = PlinkParser.read_plink(plink_prefix)
    print(f"  Loaded: {raw_data.num_individuals} animals, {raw_data.num_snps} markers.")

    # 2. Quality Control
    print("\n[Step 2] Applying Genomic Quality Control...")
    qc_data, qc_report = PlinkQC.filter(raw_data, min_maf=min_maf)
    print(f"  Passed: {qc_data.num_individuals} animals, {qc_data.num_snps} markers.")

    # 3. Transmission Validation (if pedigree contains parent-offspring links)
    print("\n[Step 3] Evaluating Pedigree Transmission...")
    validator = TransmissionValidator(qc_data)
    trios = validator.find_trios()
    print(f"  Identified {len(trios)} trios in dataset pedigree.")
    if len(trios) >= 5:
        report = validator.validate_transmission()
        print(f"  Mendelian Error Rate: {report.mendelian_error_rate * 100:.3f}%")
        print(f"  Transmission discrepancy RMSE: {report.theoretical_discrepancy_rmse:.4f}")
        report.crossover_summary_by_chrom.to_csv(output_dir / "trio_crossovers.csv", index=False)
    else:
        print("  Insufficient trios for full transmission verification (requires >= 5).")

    # 4. Phenotype Alignment
    print("\n[Step 4] Aligning Phenotypes with Genotypes...")
    pheno_dataset = PhenotypeLoader.load_from_csv(
        phenotype_csv, id_col=id_col, growth_col=growth_col, marbling_col=marbling_col
    )
    matched_pheno, matched_geno = pheno_dataset.align_with_plink(qc_data)
    print(f"  Aligned {len(matched_pheno)} animals with complete genotypes and records.")

    N = len(matched_pheno)
    n_train = int(N * train_fraction)
    perm = np.random.default_rng(2026).permutation(N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    if not 0 < train_fraction < 1 or min(len(train_idx), len(test_idx)) < 5:
        raise ValueError("Need at least five animals in each split")

    # 5. Environmental Adjustment
    print("\n[Step 5] Fitting Environmental Model...")
    env_model = EnvironmentalCovariateModel()
    fit_growth = env_model.fit_and_adjust(matched_pheno.iloc[train_idx], growth_col)
    fit_marb = env_model.fit_and_adjust(matched_pheno.iloc[train_idx], marbling_col)
    print(f"  Growth R2_env: {fit_growth.r2_environment:.3f} | Var: {fit_growth.variance_raw:.1f} -> {fit_growth.variance_adjusted:.1f}")
    print(f"  Marbling R2_env: {fit_marb.r2_environment:.3f} | Var: {fit_marb.variance_raw:.3f} -> {fit_marb.variance_adjusted:.3f}")

    matched_pheno["growth_adj"] = env_model.transform(matched_pheno, growth_col)
    matched_pheno["marbling_adj"] = env_model.transform(matched_pheno, marbling_col)

    # 6. Out-of-Sample Prediction Benchmark
    print("\n[Step 6] Running Out-of-Sample Prediction...")
    # Breed composition (fallback to uniform if not present)
    breed_cols = [c for c in matched_pheno.columns if c.startswith("breed_")]
    if not breed_cols:
        matched_pheno["breed_comp"] = 1.0
        breed_cols = ["breed_comp"]
    ancestry_mat = matched_pheno[breed_cols].values

    results = EmpiricalBenchmark.run_benchmark(
        train_ancestry=ancestry_mat[train_idx],
        train_markers=matched_geno[train_idx],
        train_growth=matched_pheno.loc[train_idx, "growth_adj"].values,
        train_marbling=matched_pheno.loc[train_idx, "marbling_adj"].values,
        test_ancestry=ancestry_mat[test_idx],
        test_markers=matched_geno[test_idx],
        test_growth=matched_pheno.loc[test_idx, "growth_adj"].values,
        test_marbling=matched_pheno.loc[test_idx, "marbling_adj"].values,
        target_growth=float(np.median(matched_pheno.loc[train_idx, "growth_adj"])),
        target_marbling=float(np.median(matched_pheno.loc[train_idx, "marbling_adj"]))
    )

    rows = []
    for m_name, res in results.items():
        rows.append({
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
    df_res = pd.DataFrame(rows)
    df_res.to_csv(output_dir / "user_dataset_prediction_benchmark.csv", index=False)
    print("\n--- Out-of-Sample Benchmark Summary ---")
    print(df_res.to_string(index=False))
    print(f"\nSaved benchmark results to {output_dir / 'user_dataset_prediction_benchmark.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and evaluate user-supplied cattle dataset.")
    parser.add_argument("--plink", type=str, default=str(BASE_DIR / "data" / "synthetic_cattle" / "synthetic_cattle_50k_population"),
                        help="Path prefix to PLINK binary files (.bed/.bim/.fam)")
    parser.add_argument("--pheno", type=str, default=str(BASE_DIR / "data" / "synthetic_cattle" / "synthetic_cattle_carcass_phenotypes.csv"),
                        help="Path to phenotype CSV file")
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "output" / "user_evaluation"),
                        help="Output directory")
    args = parser.parse_args()

    ingest_and_evaluate(Path(args.plink), Path(args.pheno), Path(args.out))

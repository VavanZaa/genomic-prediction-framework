"""Unit tests for out-of-sample prediction and joint probability attainment on withheld real calves.
"""
from pathlib import Path
import numpy as np
import pytest
from src.real_data.empirical_benchmark import EmpiricalBenchmark, EmpiricalEvaluationResult
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.real_data.phenotype_loader import PhenotypeLoader
from src.real_data.plink_parser import PlinkParser

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_cattle"


def test_empirical_prediction_benchmark():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    pheno_path = DATA_DIR / "synthetic_cattle_carcass_phenotypes.csv"

    plink_data = PlinkParser.read_plink(plink_prefix)
    dataset = PhenotypeLoader.load_from_csv(pheno_path)

    matched_pheno, matched_geno = dataset.align_with_plink(plink_data)

    # Environmental adjustment
    env_model = EnvironmentalCovariateModel()
    adj_growth = env_model.fit_and_adjust(matched_pheno, "growth").adjusted_phenotypes
    adj_marbling = env_model.fit_and_adjust(matched_pheno, "marbling").adjusted_phenotypes

    # Split into training cohort (Cohort1) and withheld testing cohort (WithheldCohort2)
    cohort_mask = (matched_pheno["cohort"] == "Cohort1").values
    test_mask = (matched_pheno["cohort"] == "WithheldCohort2").values

    assert np.sum(cohort_mask) > 50
    assert np.sum(test_mask) > 50

    ancestry_cols = ["breed_Wagyu", "breed_Angus", "breed_Brahman"]
    ancestry_matrix = matched_pheno[ancestry_cols].values

    train_ancestry = ancestry_matrix[cohort_mask]
    test_ancestry = ancestry_matrix[test_mask]

    train_geno = matched_geno[cohort_mask]
    test_geno = matched_geno[test_mask]

    train_growth = adj_growth[cohort_mask]
    test_growth = adj_growth[test_mask]

    train_marb = adj_marbling[cohort_mask]
    test_marb = adj_marbling[test_mask]

    target_g = float(np.median(adj_growth))
    target_m = float(np.median(adj_marbling))

    results = EmpiricalBenchmark.run_benchmark(
        train_ancestry=train_ancestry,
        train_markers=train_geno,
        train_growth=train_growth,
        train_marbling=train_marb,
        test_ancestry=test_ancestry,
        test_markers=test_geno,
        test_growth=test_growth,
        test_marbling=test_marb,
        target_growth=target_g,
        target_marbling=target_m
    )

    assert set(results.keys()) == {"AncestryOnly", "AdditiveGBLUP", "NonlinearRKHS"}

    for name, res in results.items():
        assert isinstance(res, EmpiricalEvaluationResult)
        assert res.growth_rmse > 0.0
        assert -1.0 <= res.growth_r <= 1.0
        assert -1.0 <= res.marbling_r <= 1.0
        assert 0.0 <= res.brier_score_joint <= 1.0
        assert 0.0 <= res.auc_joint <= 1.0
        assert 0.0 <= res.predicted_attainment_mean <= 1.0

    # Genomic models should achieve positive Pearson correlation on withheld real calves
    assert results["AdditiveGBLUP"].growth_r > 0.10
    assert results["AdditiveGBLUP"].marbling_r > 0.10

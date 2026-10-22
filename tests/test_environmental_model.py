"""Unit tests for environmental covariate model and phenotype pre-adjustment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.real_data.environmental_model import EnvironmentalCovariateModel, EnvironmentalFitResult
from src.real_data.phenotype_loader import PhenotypeLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_cattle"


def test_environmental_model_fitting():
    pheno_path = DATA_DIR / "synthetic_cattle_carcass_phenotypes.csv"
    dataset = PhenotypeLoader.load_from_csv(pheno_path)
    df = dataset.records

    env_model = EnvironmentalCovariateModel(
        continuous_covariates=["age_days"],
        categorical_covariates=["sex", "contemporary_group", "feed_regimen"]
    )

    # 1. Fit growth (carcass weight)
    res_growth = env_model.fit_and_adjust(df, "growth")
    assert isinstance(res_growth, EnvironmentalFitResult)
    assert res_growth.r2_environment > 0.15, f"Expected substantial environmental variance for growth, got R2={res_growth.r2_environment}"
    assert res_growth.variance_adjusted < res_growth.variance_raw, "Environmental adjustment should reduce raw phenotypic variance"
    assert len(res_growth.adjusted_phenotypes) == len(df)

    # 2. Check coefficient directions
    # Slaughter age should have positive coefficient on carcass weight
    assert res_growth.coefficients.get("age_days", 0.0) > 0.0

    # 3. Fit marbling
    res_marb = env_model.fit_and_adjust(df, "marbling")
    assert isinstance(res_marb, EnvironmentalFitResult)
    assert res_marb.r2_environment > 0.05
    assert res_marb.variance_adjusted < res_marb.variance_raw
    assert len(res_marb.adjusted_phenotypes) == len(df)


def test_environmental_adjustment_centering():
    pheno_path = DATA_DIR / "synthetic_cattle_carcass_phenotypes.csv"
    dataset = PhenotypeLoader.load_from_csv(pheno_path)
    df = dataset.records

    env_model = EnvironmentalCovariateModel()
    res = env_model.fit_and_adjust(df, "growth")

    # The adjusted phenotype should preserve the original scale and mean
    mean_raw = float(df["growth"].mean())
    mean_adj = float(np.mean(res.adjusted_phenotypes))
    assert np.isclose(mean_raw, mean_adj, atol=1e-5), "Adjusted phenotype mean should match raw mean"

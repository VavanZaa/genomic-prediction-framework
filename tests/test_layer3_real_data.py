"""Unit tests for Layer 3: Real Genomic Prediction on Florida Multibreed Cattle.
"""
from pathlib import Path
import numpy as np
import pytest
from scipy import stats

from src.real_data.florida_loader import FloridaLoader
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.prediction.models import AdditiveGBLUP, AncestryPredictor


def test_florida_real_dataset_ingestion():
    """Verify loading, ID alignment, and marker dimensions of the real Florida multibreed dataset."""
    data = FloridaLoader.load_dataset(marker_step=48, min_maf=0.03)  # step=48 for fast test execution

    # 1. Animals and markers
    assert data.num_animals == 3194
    assert data.num_markers > 2000
    assert data.genotypes.shape == (3194, data.num_markers)

    # 2. Called genotypes in {0, 1, 2}; missing calls stay NaN until a training-only imputer fills them
    called = data.genotypes[np.isfinite(data.genotypes)]
    assert set(np.unique(called)).issubset({0, 1, 2})
    assert np.mean(~np.isfinite(data.genotypes)) < 0.02

    # 3. Autosomes represented (BTA 1 to 29)
    chroms = set(data.marker_map["chrom"].unique())
    assert len(chroms) == 29
    assert min(chroms) == 1 and max(chroms) == 29

    # 4. Breed composition in [0, 1]
    assert np.all(data.ancestry_matrix >= 0.0)
    assert np.all(data.ancestry_matrix <= 1.0)
    # Angus + Brahman ~ 1.0
    sum_gbc = np.sum(data.ancestry_matrix, axis=1)
    assert np.all(np.abs(sum_gbc - 1.0) < 0.01)


def test_florida_environmental_deconfounding():
    """Verify fixed-effect covariate adjustment on real continuous cattle phenotypes."""
    data = FloridaLoader.load_dataset(marker_step=48, min_maf=0.03)
    df = data.phenotypes.dropna(subset=["LoadSlope", "AvgShortLength"]).reset_index(drop=True)

    env_model = EnvironmentalCovariateModel(continuous_covariates=["Year"], categorical_covariates=["Set", "Project"])
    res = env_model.fit_and_adjust(df, "LoadSlope")

    # Fixed effects should explain significant variance in real field data (> 20%)
    assert res.r2_environment > 0.20
    assert res.variance_adjusted < res.variance_raw
    assert len(res.adjusted_phenotypes) == len(df)


def test_florida_out_of_sample_genomic_prediction():
    """Verify out-of-sample prospective prediction on withheld cohort calves."""
    data = FloridaLoader.load_dataset(marker_step=48, min_maf=0.03)
    valid_mask = data.phenotypes["LoadSlope"].notna() & data.phenotypes["AvgShortLength"].notna()
    df = data.phenotypes[valid_mask].reset_index(drop=True)
    geno = data.genotypes[valid_mask.values, :]
    anc = data.ancestry_matrix[valid_mask.values, :]

    # Temporal split: Historical (<= 2018) vs Prospective (>= 2019)
    train_mask = (df["Year"] <= 2018).values
    test_mask = (df["Year"] >= 2019).values

    # Environmental effects are estimated on historical records only and applied to the prospective cohort
    env_model = EnvironmentalCovariateModel(continuous_covariates=["Year"], categorical_covariates=["Set", "Project"])
    y_train = env_model.fit_and_adjust(df[train_mask], "LoadSlope").adjusted_phenotypes
    y_test = env_model.transform(df[test_mask], "LoadSlope")

    # Fit GBLUP on real markers
    gblup = AdditiveGBLUP().fit(geno[train_mask], y_train)
    pred_gblup = gblup.predict(geno[test_mask])

    # Fit Ancestry-only
    anc_model = AncestryPredictor().fit(anc[train_mask], y_train)
    pred_anc = anc_model.predict(anc[test_mask])

    r_gblup, _ = stats.pearsonr(y_test, pred_gblup)
    r_anc, _ = stats.pearsonr(y_test, pred_anc)

    # Both models achieve positive out-of-sample prediction on real cattle. Whether SNP GBLUP beats
    # ancestry is an empirical question answered with paired bootstrap intervals by Layer 3B and
    # Gate A; on 2.3k markers it does not, so it is not asserted here.
    assert r_gblup > 0.05
    assert r_anc > 0.05

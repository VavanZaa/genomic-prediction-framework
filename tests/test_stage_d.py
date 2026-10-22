"""Tests for Stage D: Prediction Engine (Ancestry-only, GBLUP, Additive+Dominance, RKHS on withheld data).

Accuracy on a ~30-60 animal test set has a sampling SD of about 0.15, so single-seed thresholds are
unreliable; the checks below average over a fixed set of seeds.
"""
import numpy as np
import pytest
from src.prediction.cohort_benchmark import run_stage_d_benchmark

SEEDS = range(8)


@pytest.fixture(scope="module")
def benchmark():
    return run_stage_d_benchmark(SEEDS)


def test_prediction_on_withheld_families(benchmark):
    """Milestone D Completion Criterion (part 1): Measure performance on withheld families."""
    sub = benchmark[benchmark["split"] == "withheld_family"]
    assert set(sub["model"]) == {"AncestryOnly", "AdditiveGBLUP", "AdditiveDominance", "NonlinearRKHS"}
    assert np.isfinite(sub[["pearson_r", "spearman_rho", "rmse"]].to_numpy()).all()
    assert (sub["n_test"] > 0).all() and (sub["rmse"] > 0).all()

    mean_r = sub.groupby("model")["pearson_r"].mean()
    # Genomic models carry signal on families they never saw
    assert mean_r["AdditiveGBLUP"] > 0.15
    assert mean_r["NonlinearRKHS"] > 0.15


def test_prediction_on_later_cohorts(benchmark):
    """Milestone D Completion Criterion (part 2): Measure performance on a later cohort (generation 2).

    The later cohort is a backcross: its variation is within-cross Mendelian sampling, which models
    trained on purebreds and F1s predict weakly (mean r ~ 0.1 here). The check is that the
    measurement is complete and the genomic models are better than chance on average.
    """
    sub = benchmark[benchmark["split"] == "later_cohort"]
    assert np.isfinite(sub[["pearson_r", "spearman_rho", "rmse"]].to_numpy()).all()
    mean_r = sub.groupby("model")["pearson_r"].mean()
    for model in ("AdditiveGBLUP", "AdditiveDominance", "NonlinearRKHS"):
        assert mean_r[model] > 0.0

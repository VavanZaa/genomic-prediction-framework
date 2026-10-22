"""Unit tests for parent-offspring transmission, Mendelian consistency, and recombination tracking.
"""
from pathlib import Path
import numpy as np
import pytest
from src.real_data.plink_parser import PlinkParser
from src.real_data.transmission_validator import TransmissionValidator, TrioValidationReport

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_cattle"


def test_trio_detection():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    data = PlinkParser.read_plink(plink_prefix)
    validator = TransmissionValidator(data)

    trios = validator.find_trios()
    assert len(trios) > 0, "Should detect parent-offspring trios in the population"

    # Ensure all three animals in each trio exist in the dataset
    fam_ids = set(data.fam["iid"])
    for s_id, d_id, o_id in trios:
        assert s_id in fam_ids
        assert d_id in fam_ids
        assert o_id in fam_ids


def test_transmission_calibration():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    data = PlinkParser.read_plink(plink_prefix)
    validator = TransmissionValidator(data)

    report = validator.validate_transmission()

    assert isinstance(report, TrioValidationReport)
    assert report.num_trios_evaluated >= 100
    assert report.total_locus_evaluations > 10000

    # Mendelian error rate should be low (< 2%)
    assert report.mendelian_error_rate < 0.02, f"Mendelian error rate too high: {report.mendelian_error_rate}"

    # Discrepancy between empirical transition frequencies and Mendelian probabilities should be small
    assert report.theoretical_discrepancy_rmse < 0.05, f"RMSE too high: {report.theoretical_discrepancy_rmse}"

    # Key Mendelian rules:
    # 0 x 0 parents should produce almost exclusively 0 offspring
    p_00 = report.transition_probabilities[(0, 0)]
    assert p_00[0] > 0.95
    assert p_00[2] < 0.02

    # 2 x 2 parents should produce almost exclusively 2 offspring
    p_22 = report.transition_probabilities[(2, 2)]
    assert p_22[2] > 0.95
    assert p_22[0] < 0.02

    # Overall calibration flag
    assert report.is_calibrated is True


def test_recombination_by_chromosome():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    data = PlinkParser.read_plink(plink_prefix)
    validator = TransmissionValidator(data)

    report = validator.validate_transmission()
    df_chrom = report.crossover_summary_by_chrom

    assert not df_chrom.empty
    assert "chrom" in df_chrom.columns
    # The per-chromosome statistic counts opposite-homozygote jumps; it is not a crossover count
    # (crossovers need phased parents), and this synthetic data has no linkage at all.
    assert "mean_crossovers_per_calf" not in df_chrom.columns
    assert np.all(df_chrom["mean_opposite_homozygote_jumps_per_calf"] >= 0.0)

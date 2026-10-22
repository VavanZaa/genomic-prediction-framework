"""Tests for real-data ingestion: PLINK binary parser, QC filters, phenotype alignment, and BGD QTL mapping.
"""
from pathlib import Path
import numpy as np
import pytest
from src.real_data.bgd_qtl_loader import BGDQtlLoader
from src.real_data.phenotype_loader import PhenotypeLoader
from src.real_data.plink_parser import PlinkData, PlinkParser, PlinkQC

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_cattle"


def test_plink_binary_reading_and_roundtrip(tmp_path):
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    data = PlinkParser.read_plink(plink_prefix)

    assert data.num_individuals == 300
    assert data.num_snps > 400
    assert data.genotypes.shape == (300, data.num_snps)
    assert set(np.unique(data.genotypes)).issubset({-1, 0, 1, 2})

    # Test round-trip write and re-read
    out_prefix = tmp_path / "test_roundtrip"
    PlinkParser.write_plink(data, out_prefix)
    reloaded = PlinkParser.read_plink(out_prefix)

    assert reloaded.num_individuals == data.num_individuals
    assert reloaded.num_snps == data.num_snps
    np.testing.assert_array_equal(reloaded.genotypes, data.genotypes)


def test_plink_qc_filtering():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    data = PlinkParser.read_plink(plink_prefix)

    filtered, report = PlinkQC.filter(data, min_maf=0.05, max_missing_snp=0.05, max_missing_ind=0.05)

    assert report["initial_individuals"] == 300
    assert report["passed_individuals"] <= 300
    assert report["passed_snps"] <= report["initial_snps"]
    assert filtered.genotypes.shape == (report["passed_individuals"], report["passed_snps"])


def test_phenotype_loading_and_plink_alignment():
    plink_prefix = DATA_DIR / "synthetic_cattle_50k_population"
    pheno_path = DATA_DIR / "synthetic_cattle_carcass_phenotypes.csv"

    plink_data = PlinkParser.read_plink(plink_prefix)
    dataset = PhenotypeLoader.load_from_csv(pheno_path, growth_col="growth", marbling_col="marbling")

    assert dataset.num_records == 220
    matched_pheno, matched_geno = dataset.align_with_plink(plink_data)

    assert len(matched_pheno) == 220
    assert matched_geno.shape[0] == 220
    assert matched_geno.shape[1] == plink_data.num_snps

    # Verify ID alignment row by row
    fam_ids = plink_data.fam["iid"].values
    for idx, row in matched_pheno.iterrows():
        fam_idx = np.where(fam_ids == row["id"])[0][0]
        np.testing.assert_array_equal(matched_geno[idx], plink_data.genotypes[fam_idx])


def test_bgd_qtl_mapping():
    """SNPs inside a catalogue window get weight > 1; others keep 1. Uses a hand-made map so the test
    does not depend on where random synthetic SNPs happen to fall."""
    import pandas as pd
    bim = pd.DataFrame({
        "chrom": ["14", "6", "1", "14"],
        "pos_bp": [8_300_000, 37_450_000, 50_000_000, 30_000_000],  # TG, LCORL, nothing, nothing
    })
    bgd = BGDQtlLoader()
    weights_marb = bgd.map_snps_to_qtls(bim, trait_filter="Marbling")
    weights_growth = bgd.map_snps_to_qtls(bim, trait_filter="Carcass")
    np.testing.assert_array_equal(weights_marb > 1.0, [True, False, False, False])
    np.testing.assert_array_equal(weights_growth > 1.0, [False, True, False, False])


def test_curated_catalogue_is_verified():
    loader = BGDQtlLoader()
    df = loader.to_dataframe()

    assert len(df) >= 10
    for col in ("pubmed_id", "breed", "p_value", "candidate_gene", "assembly", "study_reference"):
        assert col in df.columns
    # Every built-in record carries a verified PubMed ID and assembly, and no untraceable numbers
    assert df["pubmed_id"].str.fullmatch(r"\d+").all()
    assert df["assembly"].str.startswith("ARS-UCD2.0").all()
    assert df["p_value"].isna().all() and df["effect_size"].isna().all()
    # Chromosome assignments checked against NCBI Gene
    genes = df.set_index("candidate_gene")["chrom"]
    assert genes["TG"] == "14" and genes["CAST"] == "7" and genes["IGF1"] == "5"

    japanese = loader.filter(breed_filter="Japanese Black")
    assert [r.candidate_gene for r in japanese.records] == ["SCD"]

    # Without p-values, evidence weighting falls back to the binary weight
    import pandas as pd
    bim = pd.DataFrame({"chrom": ["14"], "pos_bp": [8_300_000]})
    assert loader.map_snps_to_qtls(bim, trait_filter="Marbling", weighting_mode="evidence")[0] == pytest.approx(2.5)


"""Every dataset reports whether it holds measured or generated animals (docs/EVIDENCE_POLICY.md)."""
from pathlib import Path

import pytest

from src.real_data.florida_boa_loader import FloridaBOALoader
from src.real_data.phenotype_loader import PhenotypeLoader
from src.real_data.plink_parser import PlinkParser, PlinkQC
from src.real_data.provenance import evidence_label, read_provenance

DATA = Path(__file__).resolve().parents[1] / "data"


def test_synthetic_benchmark_reports_synthetic_through_qc():
    plink = PlinkParser.read_plink(DATA / "synthetic_cattle" / "synthetic_cattle_50k_population")
    assert plink.provenance == "synthetic"
    qc, _ = PlinkQC.filter(plink)
    assert qc.provenance == "synthetic"
    pheno = PhenotypeLoader.load_from_csv(DATA / "synthetic_cattle" / "synthetic_cattle_carcass_phenotypes.csv")
    assert evidence_label(pheno.provenance) == "SYNTHETIC"


def test_real_cattle_folder_holds_only_empirical_data():
    real = DATA / "real_cattle"
    assert not list(real.glob("cattle_50k_population.*")) and not (real / "cattle_carcass_phenotypes.csv").exists()
    for sub in ("florida_multibreed", "spanish_trios"):
        assert read_provenance(real / sub) == "empirical"


def test_florida_loader_reports_empirical():
    assert FloridaBOALoader.load_dual_dataset(marker_step=48).provenance == "empirical"


def test_provenance_file_is_validated(tmp_path):
    assert read_provenance(tmp_path) == "unknown"
    (tmp_path / "PROVENANCE").write_text("real-ish\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empirical"):
        read_provenance(tmp_path)

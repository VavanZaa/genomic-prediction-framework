"""Download and cache the University of Florida Angus-Brahman Multibreed Herd dataset.

Data source:
Repository: gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information
Files:
- BOA.map (122,486 markers mapped to ARS-UCD1.2 bovine autosomes)
- BOA.raw.gz (3,194 animals with raw dosage genotypes across all markers)
- Thermo.Phenotypes.csv (3,594 animals with matched phenotypes: LoadSlope, AvgShortLength, AngusGBC, BrahmanGBC, Year, Set, Project)
"""
from __future__ import annotations
import gzip
from pathlib import Path
import shutil
import urllib.request
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "real_cattle" / "florida_multibreed"

URLS = {
    "Thermo.Phenotypes.csv": "https://raw.githubusercontent.com/gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information/main/data/Example/Thermo.Phenotypes.csv",
    "BOA.map": "https://raw.githubusercontent.com/gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information/main/data/Example/BOA.map",
    "BOA.raw.gz": "https://raw.githubusercontent.com/gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information/main/data/Example/BOA.raw.gz",
}


def download_file(url: str, dest_path: Path) -> None:
    print(f"Downloading {dest_path.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  Successfully saved {dest_path.name} ({size_mb:.2f} MB)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("DOWNLOADING REAL CATTLE DATASET: UF ANGUS-BRAHMAN MULTIBREED HERD")
    print(f"Destination: {DATA_DIR}")
    print("=" * 80)

    for filename, url in URLS.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"File {filename} already exists ({size_mb:.2f} MB). Skipping download.")
        else:
            download_file(url, dest)

    # Verify files
    pheno_file = DATA_DIR / "Thermo.Phenotypes.csv"
    map_file = DATA_DIR / "BOA.map"
    raw_file = DATA_DIR / "BOA.raw.gz"

    assert pheno_file.exists(), "Phenotypes file missing!"
    assert map_file.exists(), "Map file missing!"
    assert raw_file.exists(), "Genotypes file missing!"

    print("\n[Verification Summary]")
    with open(pheno_file, "r") as f:
        n_pheno = sum(1 for _ in f) - 1
    print(f"  Phenotype records: {n_pheno} animals")

    with open(map_file, "r") as f:
        n_map = sum(1 for _ in f) - 1
    print(f"  Genotype map:      {n_map} SNP markers")

    with gzip.open(raw_file, "rt") as f:
        header = f.readline().split("\t")
        n_cols = len(header)
        n_raw = sum(1 for _ in f)
    print(f"  Raw genotypes:     {n_raw} animals x {n_cols - 1} markers")
    print("\nDataset download and verification completed successfully!")


if __name__ == "__main__":
    main()

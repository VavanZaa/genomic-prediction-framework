"""Download script for physical Florida Multibreed SNP genotypes (SNP.raw.gz & SNP.map).

Uses GitHub's Git LFS media endpoint:
- SNP.raw.gz (~172 MB uncompressed raw genotypes from Git LFS)
- SNP.map (~7.6 MB marker map)
Stores files in: data/real_cattle/florida_multibreed/
"""
from pathlib import Path
import sys
import time
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "real_cattle" / "florida_multibreed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNP_RAW_URL = "https://media.githubusercontent.com/media/gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information/main/data/Example/SNP.raw.gz"
SNP_MAP_URL = "https://raw.githubusercontent.com/gzayasPR/Genomic-Prediction-Pipeline-with-BOA-Information/main/data/Example/SNP.map"


def download_file(url: str, destination: Path, expected_min_bytes: int = 100_000):
    if destination.exists() and destination.stat().st_size >= expected_min_bytes:
        print(f"[CACHE] {destination.name} already exists ({destination.stat().st_size / 1e6:.2f} MB). Skipping.")
        return

    print(f"[DOWNLOAD] Fetching {destination.name} from {url}...")
    start = time.time()

    headers = {"User-Agent": "Mozilla/5.0 (CrossBreedingResearch/1.0)"}
    req = urllib.request.Request(url, headers=headers)

    temp_dest = destination.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=120) as response, open(temp_dest, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  Progress: {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)

    print()
    temp_dest.replace(destination)
    elapsed = time.time() - start
    print(f"[SUCCESS] Downloaded {destination.name} ({destination.stat().st_size / 1e6:.2f} MB in {elapsed:.1f}s)")


def main():
    print("=" * 80)
    print("DOWNLOADING PHYSICAL CATTLE SNP GENOTYPES (UF ANGUS-BRAHMAN MULTIBREED)")
    print("=" * 80)
    
    # 1. Download SNP.map
    download_file(SNP_MAP_URL, DATA_DIR / "SNP.map", expected_min_bytes=5_000_000)

    # 2. Download SNP.raw.gz
    download_file(SNP_RAW_URL, DATA_DIR / "SNP.raw.gz", expected_min_bytes=100_000_000)

    print("\nPhysical cattle SNP datasets successfully verified in:")
    print(f"  {DATA_DIR}")


if __name__ == "__main__":
    main()

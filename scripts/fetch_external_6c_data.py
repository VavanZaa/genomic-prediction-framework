"""Download the public Layer 6C candidate datasets and verify their published digests.

Datasets:
- mizzou_hair_shedding: Durbin et al., Dryad doi:10.5061/dryad.ngf1vhhz4 (CC0). The phenotype/metadata CSV
  is fetched from the authors' code repository and accepted only if it matches the Dryad SHA-256. The
  genotype VCF is served by Dryad only behind a browser check: download it by hand, then --verify-only.
- decker2014_worldwide: Decker et al. 2014, Dryad doi:10.5061/dryad.th092 (CC0); manual download.
- seminole_steers_eva: EVA/ENA PRJEB75981 (analysis ERZ24663392), 1,052 Seminole steers, genotypes only.

Blinding (docs/LAYER6C_PROTOCOL.md): phenotype files are downloaded and hashed but must not be opened
before the protocol is frozen; the log marks them "blinded".

Usage:
    python scripts/fetch_external_6c_data.py --dataset seminole_steers_eva
    python scripts/fetch_external_6c_data.py --dataset mizzou_hair_shedding
    python scripts/fetch_external_6c_data.py --dataset mizzou_hair_shedding --verify-only   # after a manual download
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
DRYAD = "https://doi.org/{}"

DATASETS = {
    "mizzou_hair_shedding": {
        "doi": "10.5061/dryad.ngf1vhhz4", "license": "CC0-1.0",
        "citation": "Durbin et al., Dryad, Genomic loci involved in sensing environmental cues and metabolism "
                    "affect seasonal coat shedding in Bos taurus and Bos indicus cattle (G3 2024)",
        "files": [
            {"name": "Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv", "blinded": True, "size": 2944572,
             "url": "https://raw.githubusercontent.com/harlydurbin/mizzou_hairshed_public/HEAD/data/derived_data/data_submission/anonymized.csv",
             "sha256": "4a60fe0347d5a6f1725b7e4dcaba8f9588de119cd80abec131c1ccb90098e03c"},
            {"name": "Durbin_etal_2021_MizzouHairShedding_genotypes.vcf.gz.tbi", "manual": True, "size": 2206584,
             "sha256": "ac266d06f43e31fa3c031681b4a3ec3c56da73e4c90c68296763cc4f59e43221"},
            {"name": "Durbin_etal_2021_MizzouHairShedding_genotypes.vcf.gz", "manual": True, "size": 3176980547,
             "sha256": "14dce56596d1434dbcc43af7fe1ef7b5815f5bbf32a23bae9b0c5116978cf1bf"},
        ],
    },
    "decker2014_worldwide": {
        "doi": "10.5061/dryad.th092", "license": "CC0-1.0",
        "citation": "Decker et al. 2014, PLoS Genetics 10:e1004254",
        "files": [
            {"name": "README_for_WorldWideCattleDryad.txt", "manual": True, "size": 919, "md5": "cf0534b3bef89aed315ad6c1997b12a6"},
            {"name": "WorldWideCattleDryad.zip", "manual": True, "size": 24187379, "md5": "0377a330590e4e8af07c1ebe416c975a"},
        ],
    },
    "slick_hairy_prlr": {
        "doi": "10.5061/dryad.nh6v4", "license": "CC0-1.0",
        "citation": "Littlejohn et al. 2014, Nature Communications 5:5861, 'Functionally reciprocal mutations "
                    "of the prolactin signalling pathway define hairy and slick cattle' (Dryad doi:10.5061/dryad.nh6v4)",
        "files": [
            {"name": "gen_phen_data_for_paper.zip", "manual": True, "size": 11247962, "md5": "2bfc903f8c119d004b7237bdd92cfa6e"},
            {"name": "phased_gen_phen_data_slick.zip", "manual": True, "size": 2013, "md5": "8c015e277188c38a06eb3880438b7d69"},
            {"name": "physiological_phenos.zip", "manual": True, "size": 7536, "md5": "7b51483531c00542d1a7026b482421a6"},
            {"name": "Hairy_bull_progeny_TaqMan_results.txt", "manual": True, "size": 106668, "md5": "127ecb91ccdc83c1dc32cf0e96a1c19a"},
        ],
    },
    "seminole_steers_eva": {
        "doi": None, "license": "EMBL-EBI archive (ENA/EVA): public, no use restrictions stated",
        "citation": "University of Florida, EVA study PRJEB75981 (analysis ERZ24663392), 'QTL for beef cattle "
                    "carcass traits and marbling': 1,052 Seminole steers genotyped (VCF, May 2024)",
        "files": [
            {"name": "Seminole.Steers.May.2024.vcf.gz", "size": 42803885, "md5": "031ffb2ab9f23efd8956caf075d51411",
             "url": "https://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ246/ERZ24663392/Seminole.Steers.May.2024.vcf.gz"},
            {"name": "Seminole.Steers.May.2024.vcf.csi", "size": 581581, "md5": "b9ffebbc74d3e2e1610858493935082d",
             "url": "https://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ246/ERZ24663392/Seminole.Steers.May.2024.vcf.csi"},
        ],
    },
}


def digest_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, algo: str) -> str:
    """Streams a file to disk and returns its digest."""
    h = hashlib.new(algo)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "CrossBreedin-6C-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        for block in iter(lambda: resp.read(1 << 22), b""):
            out.write(block)
            h.update(block)
    tmp.replace(dest)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--verify-only", action="store_true",
                        help="Do not download; verify files already placed in data/external/<dataset>/")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    folder = EXTERNAL / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "PROVENANCE").write_text(
        f"empirical\n{spec['citation']}; {('doi:' + spec['doi'] + '; ') if spec['doi'] else ''}licence: {spec['license']}. "
        "External Layer 6C candidate.\n", encoding="utf-8")
    log_path = folder / "DOWNLOAD_LOG.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"files": {}}
    missing = []
    for f in spec["files"]:
        dest = folder / f["name"]
        algo = "sha256" if "sha256" in f else "md5"
        if log["files"].get(f["name"], {}).get("verified") and dest.exists():
            print(f"  {f['name']}: already verified")
            continue
        if args.verify_only or f.get("manual"):
            if not dest.exists():
                missing.append(f["name"])
                continue
            digest, source = digest_file(dest, algo), "placed by hand"
        else:
            print(f"  downloading {f['name']} ({f['size'] / 1e6:.1f} MB)", flush=True)
            digest, source = download(f["url"], dest, algo), f["url"]
        ok = digest == f[algo]
        log["files"][f["name"]] = {"source": source, algo: digest, "expected": f[algo], "verified": ok,
                                   "bytes": dest.stat().st_size, "blinded": bool(f.get("blinded")),
                                   "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        if not ok:
            raise SystemExit(f"{f['name']}: {algo} mismatch ({digest} vs published {f[algo]})")
        print(f"  {f['name']}: {algo} verified" + (" (BLINDED: do not open before the 6C protocol is frozen)" if f.get("blinded") else ""))
    if missing:
        where = DRYAD.format(spec["doi"]) if spec["doi"] else "the source archive"
        print(f"\n  Download by hand from {where} into {folder}:\n    " + "\n    ".join(missing)
              + f"\n  then run: python scripts/fetch_external_6c_data.py --dataset {args.dataset} --verify-only")


if __name__ == "__main__":
    sys.exit(main())

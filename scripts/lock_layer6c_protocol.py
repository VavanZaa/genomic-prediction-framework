"""Freeze the Layer 6C protocol: hash every input that decides the analysis, before phenotypes are opened.

Writes output/layer6c/PROTOCOL_LOCK.json and refuses to overwrite it. The unblind stage of
scripts/run_layer6c_external_validation.py refuses to run unless every file listed here still matches.
The Mizzou phenotype file is hashed without being parsed (its hash must also equal the Dryad digest).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer6c" / "PROTOCOL_LOCK.json"
DRYAD_PHENO_SHA256 = "4a60fe0347d5a6f1725b7e4dcaba8f9588de119cd80abec131c1ccb90098e03c"
FILES = [
    "docs/LAYER6C_PROTOCOL.md", "docs/OOD_CALIBRATION.md",
    "output/ood_calibration/thresholds_rev1.json", "output/ood_calibration/thresholds.json",
    "data/external/allele_lookup_f250.csv", "data/external/panel_6c_markers.csv", "data/external/florida_panel_6c.npz",
    "scripts/run_layer6c_external_validation.py", "scripts/build_6c_marker_panel.py", "scripts/build_allele_lookup.py",
    "src/real_data/harmonisation.py", "src/real_data/ancestry_estimation.py", "src/evaluation/ood_detector.py",
    "src/evaluation/support.py", "src/evaluation/risk_calibration.py",
    "output/layer6c/prepared_steers.csv", "output/layer6c/prepared_mizzou.csv", "output/layer6c/prepare_summary.json",
    "data/external/mizzou_hair_shedding/Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv",
]


def main():
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the protocol is already frozen")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    pheno = "data/external/mizzou_hair_shedding/Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv"
    if hashes[pheno] != DRYAD_PHENO_SHA256:
        raise SystemExit("Mizzou phenotype file does not match the Dryad digest")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

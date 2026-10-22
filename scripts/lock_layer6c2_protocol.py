"""Freeze the Layer 6C2 protocol before the hairy-syndrome phenotype column is read.

Writes output/layer6c2/PROTOCOL_LOCK_6C2.json and refuses to overwrite it. The unblind stage of
scripts/run_layer6c2_hairy_probe.py refuses to run unless every file here still matches.

The phenotype lives in the .fam file, which also holds the sample identifiers the prepare stage
needs, so it cannot be withheld the way the Mizzou CSV was. It is hashed against the published
Dryad archive instead, and the blinding is procedural: the protocol section and this lock are
written before any association between a score and that column is computed.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer6c2" / "PROTOCOL_LOCK_6C2.json"
FILES = [
    "docs/LAYER6C_PROTOCOL.md", "docs/OOD_CALIBRATION.md", "docs/EVIDENCE_POLICY.md",
    "output/ood_calibration/thresholds_rev1.json",
    "data/external/allele_lookup_f250.csv", "data/external/panel_6c_markers.csv",
    "data/external/florida_panel_6c.npz",
    "data/external/ucsc_chain/bosTau6ToBosTau9.over.chain.gz",
    "data/external/slick_hairy_prlr/extracted/gen_phen_data_for_paper/hairy_syndrome_gens_phen.bed",
    "data/external/slick_hairy_prlr/extracted/gen_phen_data_for_paper/hairy_syndrome_gens_phen.bim",
    "data/external/slick_hairy_prlr/extracted/gen_phen_data_for_paper/hairy_syndrome_gens_phen.fam",
    "scripts/run_layer6c2_hairy_probe.py", "scripts/run_layer6c_external_validation.py",
    "src/real_data/liftover.py", "src/real_data/harmonisation.py", "src/real_data/ancestry_estimation.py",
    "src/evaluation/ood_detector.py", "src/evaluation/support.py", "src/evaluation/risk_calibration.py",
    "output/layer6c2/prepared_hairy.csv", "output/layer6c2/prepare_summary_6c2.json",
]


def main():
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the 6C2 protocol is already frozen")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "arm": "6C2 slick/hairy specificity probe",
                                "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

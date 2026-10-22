"""Freeze the Layer 6C4 protocol before any Decker support score or ancestry coordinate exists.

Writes output/layer6c4/PROTOCOL_LOCK_6C4.json and refuses to overwrite it. The unblind stage of
scripts/run_layer6c4_decker.py refuses to run unless every file here still matches.

Decker carries no phenotype, so nothing is withheld the way the Mizzou hair scores were. The
blinding here is that the endpoint quantities - ancestry coordinates, support scores, acceptance
decisions - have never been computed for these animals: the 6C3 coverage stop preserved them, and
the prepare stage was written to compute harmonisation and gates only.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer6c4" / "PROTOCOL_LOCK_6C4.json"
DECKER = "data/external/decker2014_worldwide"
FILES = [
    "docs/LAYER6C4_DECKER_PROTOCOL.md", "docs/COVERAGE_CALIBRATION.md", "docs/LAYER6C_PROTOCOL.md",
    "docs/EVIDENCE_POLICY.md",
    "output/ood_calibration/thresholds_rev1.json",
    "output/coverage_calibration/coverage_calibration.json",
    "data/external/allele_lookup_f250.csv", "data/external/panel_6c_markers.csv",
    "data/external/florida_panel_6c.npz",
    "data/external/ucsc_chain/bosTau6ToBosTau9.over.chain.gz",
    f"{DECKER}/WorldWideCattleDryad.zip",
    f"{DECKER}/extracted/ww_cattle_dryad_20Sept2013.map",
    f"{DECKER}/extracted/ww_cattle_dryad_20Sept2013.pedind",
    "scripts/run_layer6c4_decker.py", "scripts/run_layer6c_external_validation.py",
    "src/real_data/liftover.py", "src/real_data/ancestry_estimation.py",
    "src/evaluation/ood_detector.py", "src/evaluation/support.py",
    "src/evaluation/risk_calibration.py",
    "output/layer6c4/prepared_decker.npz", "output/layer6c4/prepare_summary_6c4.json",
]


def main():
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the 6C4 protocol is already frozen")
    results = ROOT / "output/layer6c4/results_6c4.json"
    if results.exists():
        raise SystemExit(f"{results} exists: endpoints were computed before the lock")
    summary = json.loads((ROOT / "output/layer6c4/prepare_summary_6c4.json").read_text(encoding="utf-8"))
    if not summary["ab_consistency"]["pass"]:
        raise SystemExit("the A/B consistency gate did not pass; there is nothing to lock")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "6C4 Decker worldwide panel, confirmation test for C3",
        "usable_markers": summary["harmonisation"]["usable_markers"],
        "ab_gate": summary["ab_consistency"],
        "disclosures": [
            "the coverage requirement enabling this arm was re-derived after this dataset's coverage "
            "was known (docs/COVERAGE_CALIBRATION.md, Amendment 5)",
            "the A/B gate was respecified after seeing that the data pass it (Amendment 1 of the 6C4 "
            "protocol); no endpoint quantity existed when that change was made",
        ],
        "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

"""Freeze the Layer 8 protocol before any wheat model is fitted.

Writes output/layer8/PROTOCOL_LOCK_L8.json and refuses to overwrite it. The unblind stage of
scripts/run_layer8_wheat_transfer.py refuses to run unless every file here still matches.

The phenotypes are public and already on disk, so the blinding is procedural: the endpoints, the
positive and negative controls and the tolerances are fixed, and no model has been fitted, when
this lock is written.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer8" / "PROTOCOL_LOCK_L8.json"
FILES = [
    "docs/LAYER8_WHEAT_TRANSFER_PROTOCOL.md", "docs/EVIDENCE_POLICY.md",
    "scripts/run_layer8_wheat_transfer.py",
    "src/prediction/kernel_cv.py", "src/evaluation/support.py", "src/evaluation/risk_calibration.py",
    "data/real_plants/cimmyt_wheat/wheat_X_markers.csv",
    "data/real_plants/cimmyt_wheat/wheat_Y_yield.csv",
    "data/real_plants/cimmyt_wheat/wheat_A_pedigree.csv",
    "data/real_plants/cimmyt_wheat/PROVENANCE",
    "output/layer8/wheat_grm.npz", "output/layer8/prepare_summary_l8.json",
]


def main():
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the Layer 8 protocol is already frozen")
    results = ROOT / "output/layer8/results_l8.json"
    if results.exists():
        raise SystemExit(f"{results} exists: models were fitted before the lock")
    summary = json.loads((ROOT / "output/layer8/prepare_summary_l8.json").read_text(encoding="utf-8"))
    if not summary["alignment_check"]["pass"]:
        raise SystemExit("the alignment gate did not pass; there is nothing to lock")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "Layer 8: transfer to unseen lines in an unseen environment (CIMMYT wheat)",
        "lines": summary["lines"], "markers_used": summary["markers_used"],
        "environments": summary["environments"],
        "alignment_check": summary["alignment_check"],
        "controls": {"positive": "2<->4 (phenotypes correlate +0.661) must transfer",
                     "negative": "1<->4 (phenotypes correlate -0.193) must NOT appear to transfer"},
        "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

"""Freeze the Layer 7 / H7b protocol before any mouse phenotype meets a genotype in a model.

Writes output/layer7b/PROTOCOL_LOCK_H7B.json and refuses to overwrite it. The unblind stage of
scripts/run_layer7_h7b.py refuses to run unless every file here still matches.

The prepare stage has already read phenotype columns, to count completeness and apply the trait
selection rule; it fitted nothing and computed no accuracy. The blinding is that no model has been
fitted and no correlation computed when this lock is written.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer7b" / "PROTOCOL_LOCK_H7B.json"
HS = "data/real_mice/hs_mice"
FILES = [
    "docs/LAYER7_H7B_PROTOCOL.md", "docs/LAYER7_HYPOTHESIS.md", "docs/EVIDENCE_POLICY.md",
    "scripts/run_layer7_h7b.py",
    "src/prediction/kernel_cv.py", "src/evaluation/support.py", "src/evaluation/risk_calibration.py",
    f"{HS}/PROVENANCE", f"{HS}/DOWNLOAD_LOG.json",
    "output/layer7b/mouse_grm.npz", "output/layer7b/prepare_summary_h7b.json",
]
FILES += [f"{HS}/genotypes/chr{c}.build37.data" for c in list(range(1, 20))]
FILES += [f"{HS}/phenotypes/{n}" for n in
          ("Weight.txt", "Biochemistry.txt", "Haematology.txt", "Immunology.txt")]


def supersede(reason, evidence):
    """Re-freeze after a documented, results-neutral change, archiving the previous lock.

    Legitimate only while no result exists: the lock's claim is that the specification was fixed
    before anything was fitted, and re-freezing before any model has run preserves that claim.
    """
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} does not exist; nothing to supersede")
    results = ROOT / "output/layer7b/results_h7b.json"
    if results.exists():
        raise SystemExit(f"{results} exists: a result has been produced, so the lock cannot be superseded")
    old = json.loads(LOCK.read_text(encoding="utf-8"))
    old_sha = sha256_file(LOCK)
    new_hashes = {f: sha256_file(ROOT / f) for f in FILES}
    changed = {f: {"was": old["sha256"].get(f), "now": h}
               for f, h in new_hashes.items() if old["sha256"].get(f) != h}
    version = int(old.get("version", 1))
    archive = LOCK.with_name(f"PROTOCOL_LOCK_H7B_v{version}_superseded.json")
    archive.write_text(json.dumps(old, indent=2), encoding="utf-8")
    payload = dict(old)
    payload.update({
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version + 1,
        "supersedes": {"file": archive.name, "lock_sha256": old_sha,
                       "locked_utc": old.get("locked_utc"), "changed_files": changed,
                       "reason": reason, "evidence": evidence, "result_existed": False},
        "sha256": new_hashes})
    LOCK.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Superseded v{version} -> v{version + 1}; archived {archive.name}")
    print(f"Changed: {list(changed)}")
    print(f"New lock SHA-256 {sha256_file(LOCK)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--reason")
    ap.add_argument("--evidence")
    args = ap.parse_args()
    if args.supersede:
        if not args.reason or not args.evidence:
            raise SystemExit("--supersede requires --reason and --evidence")
        return supersede(args.reason, args.evidence)
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the H7b protocol is already frozen")
    results = ROOT / "output/layer7b/results_h7b.json"
    if results.exists():
        raise SystemExit(f"{results} exists: models were fitted before the lock")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    print(f"hashing {len(FILES)} files ...")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    summary = json.loads((ROOT / "output/layer7b/prepare_summary_h7b.json").read_text(encoding="utf-8"))
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "H7b: genomic prediction machinery in a second species",
        "mice": summary["mice"], "families": summary["families"],
        "markers_used": summary["markers_used"],
        "selected_traits": [t["trait"] for t in summary["selected_traits"]],
        "note": ("traits were selected by the mechanical completeness rule in the protocol, before "
                 "any model was fitted; all selected traits are reported whatever they show"),
        "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

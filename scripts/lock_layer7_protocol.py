"""Freeze the Layer 7 / H7 protocol before any observed founder mosaic is reconstructed.

Writes output/layer7/PROTOCOL_LOCK_H7.json and refuses to overwrite it. The unblind stage of
scripts/run_layer7_h7.py refuses to run unless every file here still matches.

Both halves of the comparison are hashed: the protocol and the simulation code (so the simulated
side cannot be adjusted after seeing the observed side), and the mouse genotype, founder-genotype
and map files (so the observed side is the data that was pre-registered). The mouse archive holds
no phenotype file, so there is nothing to withhold - the blinding here is that no observed mosaic
statistic exists when this lock is written.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file

LOCK = ROOT / "output" / "layer7" / "PROTOCOL_LOCK_H7.json"
MOUSE = "data/real_mice/svenson"
FILES = [
    "docs/LAYER7_HYPOTHESIS.md", "docs/LAYER7_H7_PROTOCOL.md", "docs/EVIDENCE_POLICY.md",
    "scripts/run_layer7_h7.py", "scripts/reconstruct_do_mosaics.R",
    "src/inheritance/meiosis.py", "src/inheritance/chromosome.py",
    f"{MOUSE}/svenson.json", f"{MOUSE}/svenson_covar.csv", f"{MOUSE}/PROVENANCE",
]
FILES += [f"{MOUSE}/svenson_geno{c}.csv" for c in list(range(1, 20))]
FILES += [f"{MOUSE}/MM/MM_foundergeno{c}.csv" for c in list(range(1, 20))]
FILES += [f"{MOUSE}/MM/MM_gmap{c}.csv" for c in list(range(1, 20))]


def supersede(reason, evidence):
    """Re-freeze after a documented, results-neutral change, archiving the previous lock.

    Only legitimate while no observed statistic exists: the lock's claim is that the specification
    was fixed before any result was seen, and re-freezing before any result exists preserves that
    claim exactly. The previous lock is archived rather than deleted, and the changed files are
    named with their old and new hashes so the deviation is auditable.
    """
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} does not exist; nothing to supersede")
    observed = ROOT / "output/layer7/observed_segments.csv"
    if observed.exists():
        raise SystemExit(f"{observed} exists: a result has been produced, so the lock cannot be "
                         "superseded. Report a deviation against the existing lock instead.")
    old = json.loads(LOCK.read_text(encoding="utf-8"))
    old_lock_sha = sha256_file(LOCK)
    new_hashes = {f: sha256_file(ROOT / f) for f in FILES}
    changed = {f: {"was": old["sha256"].get(f), "now": h}
               for f, h in new_hashes.items() if old["sha256"].get(f) != h}
    version = int(old.get("version", 1))
    archive = LOCK.with_name(f"PROTOCOL_LOCK_H7_v{version}_superseded.json")
    archive.write_text(json.dumps(old, indent=2), encoding="utf-8")
    LOCK.write_text(json.dumps({
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layer": "7", "hypothesis": "H7 cross-species inheritance", "version": version + 1,
        "supersedes": {"file": archive.name, "lock_sha256": old_lock_sha,
                       "locked_utc": old.get("locked_utc"), "changed_files": changed,
                       "reason": reason, "equivalence_evidence": evidence,
                       "observed_result_existed": False},
        "sha256": new_hashes}, indent=2), encoding="utf-8")
    print(f"Superseded v{version} -> v{version + 1}; archived {archive.name}")
    print(f"Changed files: {list(changed)}")
    print(f"New lock SHA-256 {sha256_file(LOCK)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--supersede", action="store_true",
                    help="re-freeze after a documented, results-neutral change")
    ap.add_argument("--reason", help="why the change was made (required with --supersede)")
    ap.add_argument("--evidence", help="how it was shown not to change results")
    args = ap.parse_args()
    if args.supersede:
        if not args.reason or not args.evidence:
            raise SystemExit("--supersede requires --reason and --evidence")
        return supersede(args.reason, args.evidence)
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists; the H7 protocol is already frozen")
    observed = ROOT / "output/layer7/observed_segments.csv"
    if observed.exists():
        raise SystemExit(f"{observed} already exists: the observed mosaics were reconstructed "
                         "before the lock, so this run cannot be pre-registered")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Cannot lock; missing: {missing}")
    hashes = {f: sha256_file(ROOT / f) for f in FILES}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "layer": "7", "hypothesis": "H7 cross-species inheritance",
                                "sha256": hashes}, indent=2), encoding="utf-8")
    print(f"Locked {len(hashes)} files -> {LOCK.relative_to(ROOT)}; lock SHA-256 {sha256_file(LOCK)}")


if __name__ == "__main__":
    main()

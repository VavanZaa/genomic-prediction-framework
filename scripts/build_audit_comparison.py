"""Snapshot result CSVs from before and after the 11 September 2026 audit, labelled by evidence source.

output/audit_pre_fix/   copies of the pre-fix CSVs saved in .audit_backup/2026-09-11_pre_fix/output_csv
output/audit_post_fix/  copies of the current CSVs in output/ (regenerated on corrected code)
Both folders get an EVIDENCE.csv (see docs/EVIDENCE_POLICY.md). The interpretation of each change is
in reports/2026-09-11_before_after.md.
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PRE_SOURCE = ROOT / ".audit_backup" / "2026-09-11_pre_fix" / "output_csv"
PRE, POST = OUT / "audit_pre_fix", OUT / "audit_post_fix"


def evidence(rel: str) -> str:
    """Evidence label for a result file, by the data its producing script reads."""
    if rel.startswith(("empirical/", "user_evaluation/")):
        return "SYNTHETIC"
    if rel.startswith("real_layer3/simulat"):
        return "EMPIRICAL vs SIMULATION"
    if rel.startswith(("real_layer4/layer4a_monte", "real_layer4/layer4a_utility")):
        return "EMPIRICAL GEBVs, SIMULATED offspring"
    if rel.startswith("real_layer4/layer4b"):
        return "SIMULATION (assumed trait constants) on EMPIRICAL candidates"
    if rel.startswith("real_layer"):
        return "EMPIRICAL"
    return "SIMULATION"


def snapshot(files, source_root: Path, dest: Path, other: set) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    rows = []
    for f in sorted(files):
        rel = f.relative_to(source_root).as_posix()
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
        rows.append({"file": rel, "evidence_source": evidence(rel), "in_other_snapshot": rel in other})
    pd.DataFrame(rows).to_csv(dest / "EVIDENCE.csv", index=False)
    print(f"{dest.relative_to(ROOT)}: {len(rows)} files")


def main():
    pre_files = list(PRE_SOURCE.rglob("*.csv"))
    post_files = [f for f in OUT.rglob("*.csv") if not f.relative_to(OUT).parts[0].startswith("audit_")]
    pre_rel = {f.relative_to(PRE_SOURCE).as_posix() for f in pre_files}
    post_rel = {f.relative_to(OUT).as_posix() for f in post_files}
    snapshot(pre_files, PRE_SOURCE, PRE, post_rel)
    snapshot(post_files, OUT, POST, pre_rel)
    print(f"New since the audit: {len(post_rel - pre_rel)} files; dropped: {sorted(pre_rel - post_rel)}")


if __name__ == "__main__":
    main()

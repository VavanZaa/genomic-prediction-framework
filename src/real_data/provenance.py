"""Dataset provenance: every loaded dataset states whether it holds measured or generated animals.

A dataset directory carries a `PROVENANCE` file whose first word is `empirical` or `synthetic`.
Loaders expose it as `.provenance`, so results can be labelled by evidence source
(see docs/EVIDENCE_POLICY.md) instead of by folder name.
"""
from pathlib import Path

EMPIRICAL, SYNTHETIC, UNKNOWN = "empirical", "synthetic", "unknown"


def read_provenance(directory) -> str:
    """First word of `<directory>/PROVENANCE`; 'unknown' when the file is absent."""
    path = Path(directory) / "PROVENANCE"
    if not path.exists():
        return UNKNOWN
    # utf-8-sig: editors and PowerShell's Out-File write a byte-order mark that is otherwise
    # read as part of the first word.
    words = path.read_text(encoding="utf-8-sig").split()
    word = words[0].lower() if words else ""
    if word not in (EMPIRICAL, SYNTHETIC):
        raise ValueError(f"{path}: first word must be 'empirical' or 'synthetic', not {word!r}")
    return word


def evidence_label(provenance: str) -> str:
    """Upper-case label printed with results: EMPIRICAL, SYNTHETIC or UNKNOWN."""
    return provenance.upper() if provenance in (EMPIRICAL, SYNTHETIC) else "UNKNOWN"

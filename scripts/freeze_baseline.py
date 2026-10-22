"""Freeze the current results as a named, hashed baseline (e.g. the v1 validated cattle baseline).

Writes output/baseline_<tag>/ with
- a copy of every result CSV and figure under output/ (audit and earlier baseline folders excluded),
- EVIDENCE.csv: the evidence source of every result file (docs/EVIDENCE_POLICY.md),
- MANIFEST.json: SHA-256 of every source, script, test, input data file and result file; Python and
  package versions; the test-suite outcome; and the run_all_stages gate results.
Run it after regenerating every result on the code being frozen. It never modifies results.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT))
from scripts.build_audit_comparison import evidence  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "msprime", "tskit"]
DATA_FILES = ["data/real_cattle/florida_multibreed/SNP.raw.gz", "data/real_cattle/florida_multibreed/BOA.raw.gz",
              "data/real_cattle/florida_multibreed/BOA.map", "data/real_cattle/florida_multibreed/Thermo.Phenotypes.csv",
              "data/real_cattle/florida_multibreed/florida_grm_122k_v3.npz"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def run_tests() -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", tail)}
    return {"exit_code": proc.returncode, "summary": tail, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v1", help="Baseline name (folder output/baseline_<tag>)")
    parser.add_argument("--label", default="v1 validated cattle baseline")
    args = parser.parse_args()
    dest = OUTPUT / f"baseline_{args.tag}"
    if dest.exists():
        raise SystemExit(f"{dest} exists; baselines are frozen, choose a new --tag")

    results = [p for p in OUTPUT.rglob("*") if p.is_file() and p.suffix in (".csv", ".png", ".json")
               and not p.relative_to(OUTPUT).parts[0].startswith(("audit_", "baseline_"))]
    tests = run_tests()
    if tests["exit_code"] != 0:
        raise SystemExit(f"Test suite is not green ({tests['summary']}); refusing to freeze")
    gates = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8")) if (OUTPUT / "run_manifest.json").exists() else {}

    dest.mkdir(parents=True)
    rows = ["file,evidence_source"]
    for p in sorted(results):
        rel = p.relative_to(OUTPUT).as_posix()
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)
        rows.append(f"{rel},{evidence(rel)}")
    (dest / "EVIDENCE.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = {
        "baseline": args.label,
        "tag": args.tag,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: md.version(p) for p in PACKAGES if _installed(p)},
        "tests": tests,
        "all_gates_passed": gates.get("all_gates_passed"),
        "gates": gates.get("gates", []),
        "code_sha256": hashes([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
                               *ROOT.joinpath("tests").glob("*.py")]),
        "input_sha256": hashes([ROOT / f for f in DATA_FILES if (ROOT / f).exists()]),
        "result_sha256": hashes(results),
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Froze {len(results)} result files into {dest.relative_to(ROOT)}; tests: {tests['summary']}")


def _installed(name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()

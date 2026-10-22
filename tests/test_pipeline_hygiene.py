"""Scientific result generation must not depend on the IDE or tool that launched it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
PERSONAL_PATH = re.compile(r"\.gemini|antigravity|[A-Za-z]:[\\/]+Users[\\/]", re.I)
EXPORTERS = ("export_figures_for_review", "export_review_bundle", "export_protocol_review")


def test_no_pipeline_code_writes_to_personal_or_ide_folders():
    offenders = [f"{p.relative_to(ROOT)}:{i}" for p in CODE
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if PERSONAL_PATH.search(line)]
    assert not offenders, "Personal/IDE paths in pipeline code: " + ", ".join(offenders)


def test_review_exports_are_never_called_by_the_pipeline():
    """Exporters may be *mentioned* (they cross-reference each other in docstrings); they may not be
    imported or spawned, which is what would make them part of a pipeline run."""
    for exporter in EXPORTERS:
        invocation = re.compile(
            rf"^\s*(?:import|from)\s+\S*{re.escape(exporter)}"          # imported as a module
            rf"|{re.escape(exporter)}\s*\.\s*main\s*\("                  # called directly
            rf"|(?:subprocess|Popen|check_call|check_output|run|system)\s*\([^)]*{re.escape(exporter)}",
            re.M)
        callers = [p.relative_to(ROOT) for p in CODE if p.stem != exporter
                   and invocation.search(p.read_text(encoding="utf-8"))]
        assert not callers, f"{exporter} must stay a manual step; invoked from {callers}"

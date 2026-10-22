"""Copy result figures to a review folder of your choice (optional, manual post-processing).

The pipeline writes every figure to one canonical place inside the project (`output/...`) and never
copies anything elsewhere: scientific result generation must not depend on the IDE or tool that
launched it. Run this script by hand when you want the figures in another viewer, for example:

    python scripts/export_figures_for_review.py --dest "C:/path/to/review/folder"
    python scripts/export_figures_for_review.py --dest ../review --pattern "real_layer4/*.png"

No experiment, test or pipeline script calls it (tests/test_pipeline_hygiene.py checks this).
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def export(dest: Path, pattern: str = "**/*.png", dry_run: bool = False) -> list:
    figures = sorted(p for p in OUTPUT.glob(pattern)
                     if p.is_file() and not p.relative_to(OUTPUT).parts[0].startswith("audit_"))
    if not figures:
        raise SystemExit(f"No figures match {pattern!r} under {OUTPUT}")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    for fig in figures:
        target = dest / fig.relative_to(OUTPUT).as_posix().replace("/", "__")
        print(f"{'would copy' if dry_run else 'copied'} {fig.relative_to(ROOT)} -> {target}")
        if not dry_run:
            shutil.copy2(fig, target)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", required=True, type=Path, help="Folder to copy the figures into")
    parser.add_argument("--pattern", default="**/*.png", help="Glob under output/ (default: every PNG)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be copied")
    args = parser.parse_args()
    export(args.dest, args.pattern, args.dry_run)


if __name__ == "__main__":
    main()

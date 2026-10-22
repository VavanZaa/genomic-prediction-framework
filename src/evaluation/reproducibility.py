"""Run manifests and founder archives for reproducible runs (workplan deliverable).
"""
from __future__ import annotations
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
from ..inheritance.meiosis import Population

DEFAULT_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "msprime")


def software_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> Dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def save_founders(population: Population, path: Path) -> int:
    """Save every founder's phased marker and QTL haplotypes, sex and breed to a compressed .npz."""
    genome = population.genome_spec
    founders = [ind for ind in population.individuals.values() if ind.is_founder]

    def stack(side: str, field: str) -> np.ndarray:
        return np.vstack([
            np.concatenate([getattr(getattr(ind, side)[c.chrom_id], field) for c in genome.chromosomes])
            for ind in founders
        ])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array([f.id for f in founders]),
        sexes=np.array([f.sex for f in founders]),
        breeds=np.array([next(iter(f.pedigree_composition.composition)) for f in founders]),
        paternal_markers=stack("paternal_haplotypes", "markers"),
        maternal_markers=stack("maternal_haplotypes", "markers"),
        paternal_qtl=stack("paternal_haplotypes", "qtl_alleles"),
        maternal_qtl=stack("maternal_haplotypes", "qtl_alleles"),
        chromosome_lengths_m=np.array([c.length_m for c in genome.chromosomes]),
        markers_per_chromosome=np.array([c.num_markers for c in genome.chromosomes]),
    )
    return len(founders)


def write_manifest(path: Path, config: Dict, gates: List[Dict], outputs: List[str]) -> None:
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "config": config,
        "software": software_versions(),
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates),
        "outputs": sorted(outputs),
    }
    Path(path).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

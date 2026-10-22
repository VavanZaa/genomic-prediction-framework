"""Bridge to R / AlphaSimR for cattle inheritance cross-verification.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class AlphaSimRBridge:
    """Manages execution of AlphaSimR R scripts via subprocess for cattle simulations."""

    def __init__(self, rscript_path: Optional[str] = None):
        self.rscript_path = rscript_path or self._find_rscript()

    @staticmethod
    def _find_rscript() -> Optional[str]:
        # Check standard PATH
        found = shutil.which("Rscript")
        if found:
            return found

        # Check standard Windows installation paths
        candidates = [
            r"C:\Program Files\R",
            r"C:\Program Files (x86)\R",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\R"),
            r"D:\R"
        ]
        for base in candidates:
            p = Path(base)
            if p.exists():
                rscripts = list(p.glob("**/Rscript.exe"))
                if rscripts:
                    return str(rscripts[0])
        return None

    def is_available(self) -> bool:
        """Returns True if Rscript exists and AlphaSimR can be loaded."""
        if not self.rscript_path or not os.path.exists(self.rscript_path):
            return False
        try:
            res = subprocess.run(
                [self.rscript_path, "-e", "if (!requireNamespace('AlphaSimR', quietly=TRUE)) quit(status=1)"],
                capture_output=True,
                text=True,
                timeout=15
            )
            return res.returncode == 0
        except Exception:
            return False

    def run_alphasimr_script(self, script_content: str, timeout: int = 120) -> Tuple[int, str, str]:
        if not self.rscript_path:
            raise RuntimeError("Rscript executable not found on system.")

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "run_sim.R")
            with open(script_path, "w") as f:
                f.write(script_content)

            proc = subprocess.run(
                [self.rscript_path, script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return proc.returncode, proc.stdout, proc.stderr

    def generate_cattle_simulation_script(
        self,
        output_dir: str,
        n_angus: int = 10,
        n_brahman: int = 10,
        n_chrom: int = 29,
        n_snps_per_chrom: int = 100,
        n_qtl_per_chrom: int = 10,
        seed: int = 42,
        split_generations: int = 1000,
        rel_aa: float = 0.1
    ) -> str:
        """Generates an AlphaSimR R script with explicit sexes, dominance, epistasis and noise.

        The two breeds come from a runMacs population split `split_generations` ago (assumed value),
        so they differ in allele frequencies; `rel_aa` is the relative additive-by-additive epistatic
        variance. SNP-chip and QTL sites are kept disjoint. Untested here: R/AlphaSimR is not installed
        on the development machine, so verify the realized architecture after the first real run.
        """
        if n_angus != n_brahman:
            raise ValueError("runMacs(split=...) returns equal-sized subpopulations; use n_angus == n_brahman.")
        out_dir_clean = output_dir.replace("\\", "/")
        return f"""# Auto-generated AlphaSimR Cattle Simulation Script
suppressMessages(library(AlphaSimR))

set.seed({seed})
out_dir <- "{out_dir_clean}"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Founder genomes: cattle demography with a population split so the two breeds diverge
founderGenomes <- runMacs(
    nInd = {n_angus + n_brahman},
    nChr = {n_chrom},
    segSites = {n_snps_per_chrom + n_qtl_per_chrom},
    species = "CATTLE",
    split = {split_generations}
)

# 2. Simulation parameters with explicit sexes, dominance, epistasis and noise
SP <- SimParam$new(founderGenomes)
SP$setSexes("yes_rand")
SP$restrSegSites(minQtlPerChr = {n_qtl_per_chrom}, minSnpPerChr = {n_snps_per_chrom}, overlap = FALSE)

SP$addTraitADE(
    nQtlPerChr = {n_qtl_per_chrom},
    mean = 100,
    var = 25,
    meanDD = 0.2, # Directional dominance
    varDD = 0.1,
    relAA = {rel_aa} # Additive-by-additive epistasis
)
SP$setVarE(h2 = 0.40)

# Add SNP chip
SP$addSnpChip(nSnpPerChr = {n_snps_per_chrom})

# 3. Create founder populations: Angus and Brahman
angusFounders <- newPop(founderGenomes[1:{n_angus}])
brahmanFounders <- newPop(founderGenomes[({n_angus}+1):({n_angus}+{n_brahman})])

# Phenotype founders
angusFounders <- setPheno(angusFounders)
brahmanFounders <- setPheno(brahmanFounders)

# 4. Perform F1 cross (Angus males x Brahman females)
angus_males <- angusFounders[angusFounders@sex == "M"]
brahman_females <- brahmanFounders[brahmanFounders@sex == "F"]

if (length(angus_males@id) > 0 && length(brahman_females@id) > 0) {{
    f1Pop <- randCross2(females = brahman_females, males = angus_males, nCrosses = 20)
    f1Pop <- setPheno(f1Pop)
    
    # Export pedigree & phenotypes
    ped_df <- data.frame(
        id = c(angusFounders@id, brahmanFounders@id, f1Pop@id),
        sire = c(angusFounders@father, brahmanFounders@father, f1Pop@father),
        dam = c(angusFounders@mother, brahmanFounders@mother, f1Pop@mother),
        sex = c(angusFounders@sex, brahmanFounders@sex, f1Pop@sex),
        pheno = c(angusFounders@pheno[,1], brahmanFounders@pheno[,1], f1Pop@pheno[,1])
    )
    write.csv(ped_df, file.path(out_dir, "pedigree_phenotypes.csv"), row.names = FALSE)
    
    # Export SNP genotypes for F1
    snps <- pullSnpGeno(f1Pop)
    write.csv(snps, file.path(out_dir, "f1_snp_genotypes.csv"))
    cat("AlphaSimR simulation completed successfully.\\n")
}} else {{
    cat("Insufficient male/female founders.\\n")
}}
"""

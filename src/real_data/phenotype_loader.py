"""Loader and alignment pipeline for matched individual phenotypic records and environmental covariates.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .plink_parser import PlinkData
from .genotype_preprocessing import validate_ids
from .provenance import read_provenance


@dataclass
class PhenotypeRecord:
    id: str
    growth: float           # e.g., Carcass Weight or Yearling Weight in kg
    marbling: float         # e.g., BMS score 1-12 or Intramuscular Fat (IMF) %
    age_days: float         # Age at slaughter/measurement
    sex: str                # 'steer', 'heifer', or 'bull'
    contemporary_group: str # Herd-year-season / cohort code
    feed_regimen: str       # 'grain', 'pasture', 'feedlot'


@dataclass
class PhenotypeDataset:
    data: pd.DataFrame
    growth_col: str
    marbling_col: str
    covariate_cols: List[str]
    provenance: str = "unknown"  # "empirical" or "synthetic", from the directory's PROVENANCE file

    @property
    def num_records(self) -> int:
        return len(self.data)

    @property
    def records(self) -> pd.DataFrame:
        return self.data

    def align_with_plink(self, plink_data: PlinkData) -> Tuple[pd.DataFrame, np.ndarray]:
        """Align phenotype records with PLINK genotype rows by individual ID (iid).

        Returns:
            aligned_pheno: pd.DataFrame matching rows of aligned_geno
            aligned_geno: np.ndarray of shape (N_matched, P)
        """
        fam_ids = validate_ids(plink_data.fam["iid"], "PLINK IDs")
        pheno_ids = validate_ids(self.data["id"], "phenotype IDs")
        if len(fam_ids) != plink_data.genotypes.shape[0]:
            raise ValueError("PLINK metadata/genotype row count mismatch")

        common_ids = [i for i in fam_ids if i in set(pheno_ids)]
        if len(common_ids) == 0:
            raise ValueError("Zero matching individual IDs found between PLINK .fam and phenotype table.")

        # Find row indices in PLINK data
        id_to_geno_idx = {iid: idx for idx, iid in enumerate(fam_ids)}
        geno_indices = [id_to_geno_idx[cid] for cid in common_ids]

        # Extract matching genotypes
        matched_geno = plink_data.genotypes[geno_indices, :]

        # Extract matching phenotypes in exact identical order
        self_indexed = self.data.assign(id=pheno_ids).set_index("id")
        matched_pheno = self_indexed.loc[common_ids].reset_index()

        return matched_pheno, matched_geno


class PhenotypeLoader:
    """Loads and validates cattle phenotypic records from CSV / TSV files."""

    @classmethod
    def load_from_csv(
        cls,
        filepath: Union[str, Path],
        id_col: str = "id",
        growth_col: str = "growth",
        marbling_col: str = "marbling",
        covariate_cols: Optional[List[str]] = None
    ) -> PhenotypeDataset:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Phenotype file not found: {p}")

        df = pd.read_csv(p, dtype={id_col: str})
        df[id_col] = validate_ids(df[id_col], "phenotype IDs")

        if growth_col not in df.columns or marbling_col not in df.columns:
            raise ValueError(f"Required trait columns ('{growth_col}', '{marbling_col}') missing from {df.columns.tolist()}")

        default_covs = ["age_days", "sex", "contemporary_group"]
        covs = covariate_cols or [c for c in default_covs if c in df.columns]

        # Standardize ID column name to 'id'
        if id_col != "id":
            df = df.rename(columns={id_col: "id"})

        # Drop rows with missing values in target traits
        df = df.dropna(subset=["id", growth_col, marbling_col]).reset_index(drop=True)

        return PhenotypeDataset(
            data=df,
            growth_col=growth_col,
            marbling_col=marbling_col,
            covariate_cols=covs,
            provenance=read_provenance(p.parent),
        )

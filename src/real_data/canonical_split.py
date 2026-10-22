"""Canonical Prospective Split & Zero-Leakage Environmental De-confounding.

Guarantees exact identity of training/testing cohorts and adjusted phenotypes
across all prediction pipelines (Layer 3A, Layer 3B, and future Layer 4A).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class CanonicalSplitData:
    """Holds aligned training and testing cohorts with strictly zero-leakage adjustments."""
    trait_name: str
    train_ids: List[str]
    test_ids: List[str]
    y_train_adj: np.ndarray
    y_test_adj: np.ndarray
    q_train: np.ndarray  # Angus genomic breed composition
    q_test: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    df_aligned: pd.DataFrame
    mu_train: float
    r2_train_env: float
    nuisance_effects: Dict[str, Dict[str, float]]


class CanonicalDataSplitter:
    """Single source of truth for prospective temporal splitting and nuisance effect removal."""

    DEFAULT_PHENO_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "real_cattle"
        / "florida_multibreed"
        / "Thermo.Phenotypes.csv"
    )

    @classmethod
    def get_prospective_split(
        cls,
        trait_col: str,
        pheno_df: Optional[pd.DataFrame] = None,
        valid_animal_ids: Optional[Union[List[str], Set[str]]] = None,
        require_joint_traits: bool = False
    ) -> CanonicalSplitData:
        """Constructs prospective train/test splits with zero leakage.
        
        Parameters:
        -----------
        trait_col: str
            Phenotype to adjust ('LoadSlope' or 'AvgShortLength').
        pheno_df: Optional[pd.DataFrame]
            If None, loads Thermo.Phenotypes.csv from default location.
        valid_animal_ids: Optional[Union[List[str], Set[str]]]
            Subset of animal IDs to restrict to (e.g. genotyped animals).
        require_joint_traits: bool
            If True, only retains animals having complete measurements for BOTH
            'LoadSlope' and 'AvgShortLength'. If False, filters for trait_col alone.
            
        Returns:
        --------
        CanonicalSplitData containing aligned arrays and training-only nuisance fits.
        """
        if pheno_df is None:
            df_raw = pd.read_csv(cls.DEFAULT_PHENO_PATH)
        else:
            df_raw = pheno_df.copy()

        df_raw["UFID_str"] = df_raw["UFID"].astype(str)

        # 1. Base eligibility mask
        mask = df_raw[trait_col].notna() & df_raw["AngusGBC"].notna() & df_raw["Year"].notna()
        if require_joint_traits:
            mask = mask & df_raw["LoadSlope"].notna() & df_raw["AvgShortLength"].notna()

        if valid_animal_ids is not None:
            id_set = set(str(x) for x in valid_animal_ids)
            mask = mask & df_raw["UFID_str"].isin(id_set)

        df_sub = df_raw[mask].copy().reset_index(drop=True)

        # 2. Strict Prospective Split:
        # Historical Training: Year <= 2018 (2016, 2017, 2018)
        # Prospective Testing: Year >= 2019 (2019, 2021, 2022 - no 2020 in data)
        train_mask = (df_sub["Year"] <= 2018).values
        test_mask = (df_sub["Year"] >= 2019).values

        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError("Prospective split resulted in empty training or test cohort.")

        # 3. Train-Only Nuisance Model Fitting
        df_train = df_sub.iloc[train_mask]
        mu_train = float(df_train[trait_col].mean())
        
        # Contemporary-group nuisance effects, estimated on training records only.
        # Set is nested within Project, so the two are not added: a record takes its Set deviation if
        # the Set was seen in training, otherwise its Project deviation, otherwise none. Contemporary
        # groups that first appear in the test years cannot be estimated without test phenotypes and
        # stay unadjusted; accuracy there mixes genetic and contemporary-group differences.
        set_eff = df_train.groupby("Set")[trait_col].mean().to_dict()
        proj_eff = df_train.groupby("Project")[trait_col].mean().to_dict()

        def apply_adjustment(sub_records: pd.DataFrame) -> np.ndarray:
            adjs = np.zeros(len(sub_records), dtype=np.float64)
            for idx, (_, row) in enumerate(sub_records.iterrows()):
                if row.get("Set") in set_eff:
                    adjs[idx] = set_eff[row.get("Set")] - mu_train
                elif row.get("Project") in proj_eff:
                    adjs[idx] = proj_eff[row.get("Project")] - mu_train
            return sub_records[trait_col].values.astype(np.float64) - adjs

        y_train_adj = apply_adjustment(df_sub.iloc[train_mask])
        y_test_adj = apply_adjustment(df_sub.iloc[test_mask])

        y_train_raw = df_sub.loc[train_mask, trait_col].values.astype(np.float64)
        y_test_raw = df_sub.loc[test_mask, trait_col].values.astype(np.float64)

        var_raw_tr = float(np.var(y_train_raw))
        var_adj_tr = float(np.var(y_train_adj))
        r2_tr_env = 1.0 - (var_adj_tr / var_raw_tr) if var_raw_tr > 1e-8 else 0.0

        q_train = df_sub.loc[train_mask, "AngusGBC"].values.astype(np.float64)
        q_test = df_sub.loc[test_mask, "AngusGBC"].values.astype(np.float64)

        train_ids = df_sub.loc[train_mask, "UFID_str"].tolist()
        test_ids = df_sub.loc[test_mask, "UFID_str"].tolist()

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        return CanonicalSplitData(
            trait_name=trait_col,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train_adj=y_train_adj,
            y_test_adj=y_test_adj,
            q_train=q_train,
            q_test=q_test,
            y_train_raw=y_train_raw,
            y_test_raw=y_test_raw,
            train_indices=train_idx,
            test_indices=test_idx,
            df_aligned=df_sub,
            mu_train=mu_train,
            r2_train_env=r2_tr_env,
            nuisance_effects={"Set": set_eff, "Project": proj_eff}
        )

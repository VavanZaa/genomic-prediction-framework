"""Loader and processor for the University of Florida Angus-Brahman Multibreed Herd dataset.

Manages 3,194 real cattle animals with complete Illumina BovineHD/SNP50 genotypes
and matched subtropical thermal-adaptation phenotypes (LoadSlope, AvgShortLength)
and genomic breed compositions (AngusGBC, BrahmanGBC).
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .plink_parser import PlinkData, PlinkParser


@dataclass
class FloridaMultibreedData:
    phenotypes: pd.DataFrame
    genotypes: np.ndarray  # Shape (N, M), values in {0, 1, 2}
    marker_map: pd.DataFrame  # Columns: ["chrom", "snp", "cM", "pos", "a1", "a2"]
    ancestry_matrix: np.ndarray  # Shape (N, 2): [AngusGBC, BrahmanGBC]
    animal_ids: List[str]
    num_animals: int
    num_markers: int
    provenance: str = "unknown"  # carried over from the dual SNP/BOA loader


class FloridaLoader:
    """SNP loader; BOA ancestry counts are a separate matrix in the dual loader."""
    DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data/real_cattle/florida_multibreed"

    @classmethod
    def load_dataset(cls, data_dir=None, marker_step=24, min_maf=0.03, save_plink_cache=True):
        from .florida_boa_loader import FloridaBOALoader
        dual=FloridaBOALoader.load_dual_dataset(data_dir,marker_step,cache=True)
        valid=dual.phenotypes["UFID"].notna().to_numpy()
        df=dual.phenotypes.loc[valid].reset_index(drop=True).rename(columns={"UFID_str":"iid"})
        X=dual.genotypes_snp[valid]
        # Marker inclusion depends on historical genotypes only, never future animals.
        ref=(df["Year"].to_numpy()<=2018) if "Year" in df else np.ones(len(df),dtype=bool)
        if not ref.any(): raise ValueError("No historical reference animals for marker QC")
        count=np.sum(np.isfinite(X[ref]),axis=0)
        freq=np.divide(np.nansum(X[ref],axis=0),2*count,out=np.zeros(X.shape[1]),where=count>0)
        keep=(count>0)&(np.minimum(freq,1-freq)>=min_maf)
        X=X[:,keep]; marker_map=dual.marker_map.loc[keep].reset_index(drop=True)
        marker_map=marker_map.assign(pos_m=0.0,a1="counted",a2="other")
        marker_map=marker_map[["chrom","snp_id","pos_m","pos_bp","a1","a2"]]
        # The old BOA-as-SNP PLINK caches are intentionally never read or overwritten.
        return FloridaMultibreedData(df,X,marker_map,df[["AngusGBC","BrahmanGBC"]].to_numpy(),
                                      df.iid.tolist(),len(df),X.shape[1],dual.provenance)

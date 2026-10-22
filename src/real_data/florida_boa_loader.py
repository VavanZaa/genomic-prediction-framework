"""Dual-Matrix Loader for Physical Cattle: SNP Genotypes (X) and Breed-of-Origin Alleles (O).

Loads and aligns:
1. Physical SNP genotypes (X) from SNP.raw.gz (missing calls retained as NaN until training-only preprocessing).
2. Breed-of-Origin Alleles (O) from BOA.raw.gz (Brahman-origin allele counts: 0, 1, 2).
3. Matched animal IDs and phenotypes from Thermo.Phenotypes.csv.
4. Caches aligned arrays in .npz for fast sub-second loading.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
import json
from .genotype_preprocessing import validate_ids, source_hashes
from .provenance import read_provenance
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class FloridaDualBOAData:
    genotypes_snp: np.ndarray  # (N, M) in [0, 2]
    alleles_boa: np.ndarray    # (N, M) in {0, 1, 2}
    marker_map: pd.DataFrame   # chrom, snp_id, pos_bp
    phenotypes: pd.DataFrame   # aligned phenotype records
    animal_ids: List[str]
    num_animals: int
    num_markers: int
    provenance: str = "unknown"  # from the data directory's PROVENANCE file


class FloridaBOALoader:
    """Read raw SNP and BOA counts, joined by unique animal and marker IDs.

    Version 3 caches preserve missing calls and include source hashes. Version 2
    caches cannot be reused because their imputation used all animals.
    """
    DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data/real_cattle/florida_multibreed"

    @classmethod
    def load_dual_dataset(cls, data_dir=None, marker_step=6, cache=True, force_reload=False):
        if not isinstance(marker_step, int) or marker_step < 1:
            raise ValueError("marker_step must be a positive integer")
        d = Path(data_dir) if data_dir is not None else cls.DEFAULT_DIR
        paths = [d/n for n in ("Thermo.Phenotypes.csv", "BOA.map", "BOA.raw.gz", "SNP.raw.gz")]
        hashes = source_hashes(paths)
        tag = f"step{marker_step}_v3"
        cf, mf, pf = d/f"florida_dual_snp_boa_{tag}.npz", d/f"florida_dual_map_{tag}.csv", d/f"florida_dual_pheno_{tag}.csv"
        df_pheno = pd.read_csv(paths[0], dtype={"UFID": str})
        df_pheno["UFID_str"] = validate_ids(df_pheno["UFID"], "phenotype IDs")
        if cache and not force_reload and all(p.exists() for p in (cf,mf,pf)):
            with np.load(cf, allow_pickle=False) as z:
                if json.loads(str(z["source_hashes"])) == hashes:
                    X, O, ids = z["genotypes_snp"], z["alleles_boa"], validate_ids(z["animal_ids"])
                    marker_map = pd.read_csv(mf)
                    pheno = df_pheno.set_index("UFID_str", drop=False).reindex(ids).reset_index(drop=True)
                    pheno["UFID_str"] = ids
                    if X.shape != O.shape or X.shape != (len(ids),len(marker_map)):
                        raise ValueError("Invalid cached matrix dimensions")
                    return FloridaDualBOAData(X,O,marker_map,pheno,ids,len(ids),X.shape[1],read_provenance(d))
        bmap = pd.read_csv(paths[1], sep=r"\s+")
        validate_ids(bmap["SNP"], "map marker IDs")
        selected = bmap.iloc[::marker_step]
        def header(path):
            with gzip.open(path, "rt") as f: cols = f.readline().split()
            names = [c.rsplit("_",1)[0] for c in cols[1:]]
            validate_ids(names,"raw marker IDs")
            return cols, {n:i+1 for i,n in enumerate(names)}
        bc,bcols=header(paths[2]); _,scols=header(paths[3])
        names=[n for n in selected.SNP if n in bcols and n in scols]
        if not names: raise ValueError("No common SNP/BOA markers")
        marker_map=selected.set_index("SNP").loc[names].reset_index().rename(columns={"CHR":"chrom","SNP":"snp_id","POS":"pos_bp"})[["chrom","snp_id","pos_bp"]]
        def read_rows(path, indices):
            rows={}
            with gzip.open(path,"rt") as f:
                f.readline()
                for line in f:
                    parts=line.split()
                    if not parts: continue
                    aid=parts[0]
                    if aid in rows: raise ValueError(f"Duplicate animal ID {aid} in {path.name}")
                    vals=np.array([np.nan if parts[i] in ("NA","-9","-1") else float(parts[i]) for i in indices], dtype=np.float32)
                    if np.isinf(vals).any() or np.any((vals<0)|(vals>2)):
                        raise ValueError(f"Invalid genotype for {aid} in {path.name}")
                    rows[aid]=vals
            validate_ids(rows)
            return rows
        br=read_rows(paths[2],[bcols[n] for n in names])
        sr=read_rows(paths[3],[scols[n] for n in names])
        # Join on animal ID; BOA defines the animal set (as in build_full_density_122k_grm.py).
        # Every BOA animal needs SNP calls. SNP-only animals (four in the public release: 160018,
        # 160038, 600007, 600053) are dropped and reported, never paired by row position.
        missing_snp=sorted(set(br)-set(sr))
        if missing_snp:
            raise ValueError(f"{len(missing_snp)} BOA animals have no SNP record: {missing_snp[:10]}")
        ids=list(br)
        snp_only=sorted(set(sr)-set(br))
        if snp_only:
            print(f"  Dropped {len(snp_only)} SNP-only animals without BOA calls: {snp_only}")
        X=np.stack([sr[i] for i in ids]); O=np.stack([br[i] for i in ids])
        alleles=[bc[bcols[n]].rsplit("_",1)[-1].upper() for n in names]
        if not set(alleles)<= {"A","B"}: raise ValueError("BOA counted allele must be A or B")
        invert=np.array([a=="A" for a in alleles]);O[:,invert]=2-O[:,invert]
        pheno=df_pheno.set_index("UFID_str",drop=False).reindex(ids).reset_index(drop=True)
        pheno["UFID_str"]=ids
        if cache:
            marker_map.to_csv(mf,index=False); pheno.to_csv(pf,index=False)
            np.savez_compressed(cf,genotypes_snp=X,alleles_boa=O,animal_ids=np.array(ids),source_hashes=json.dumps(hashes,sort_keys=True))
        return FloridaDualBOAData(X,O,marker_map,pheno,ids,len(ids),len(names),read_provenance(d))

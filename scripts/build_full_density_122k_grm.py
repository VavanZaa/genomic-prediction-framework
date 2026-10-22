"""Builds full-density (122,486 matched markers) Genomic Relationship Matrices for Florida Multibreed cattle.

Streams SNP.raw.gz and BOA.raw.gz in memory-efficient marker blocks:
1. Standardizes SNP markers: Z_SNP = X - 2*p
2. Harmonizes BOA pseudo-genotypes: O in {0, 1, 2} Brahman count, Z_BOA = O - 2*q
3. Computes VanRaden normalized GRMs:
   G_122k = (Z_SNP @ Z_SNP.T) / (2 * sum(p * (1 - p)))
   B_122k = (Z_BOA @ Z_BOA.T) / (2 * sum(q * (1 - q)))
4. Saves to data/real_cattle/florida_multibreed/florida_grm_122k_v3.npz
"""
import gzip
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE_DIR))
from src.real_data.genotype_preprocessing import validate_ids, source_hashes

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_cattle" / "florida_multibreed"
BOA_RAW = DATA_DIR / "BOA.raw.gz"
SNP_RAW = DATA_DIR / "SNP.raw.gz"
OUT_NPZ = DATA_DIR / "florida_grm_122k_v3.npz"


def build_full_density_grm():
    print("=" * 80)
    print("GATE A: BUILDING FULL-DENSITY (122,486 MARKER) GENOMIC RELATIONSHIP MATRICES")
    print("=" * 80)

    t0 = time.time()

    # 1. Header parsing and marker alignment
    print("\n[1. Parsing headers and aligning 122,486 markers...]")
    with gzip.open(BOA_RAW, "rt") as f_boa:
        boa_header = f_boa.readline().split()
    with gzip.open(SNP_RAW, "rt") as f_snp:
        snp_header = f_snp.readline().split()

    boa_col_names = [col.rsplit("_", 1)[0] for col in boa_header[1:]]
    boa_counted_alleles = [col.rsplit("_", 1)[1] if "_" in col else "A" for col in boa_header[1:]]
    # In BOA.raw.gz, if counted allele is 'A' (Angus), then O_brahman = 2 - O_raw
    boa_is_angus_counted = np.array([allele.upper() == "A" for allele in boa_counted_alleles], dtype=bool)

    snp_col_dict = {col.rsplit("_", 1)[0]: idx for idx, col in enumerate(snp_header[1:], start=1)}

    matched_boa_indices = []
    matched_snp_indices = []
    for b_idx, name in enumerate(boa_col_names):
        if name in snp_col_dict:
            matched_boa_indices.append(b_idx + 1)
            matched_snp_indices.append(snp_col_dict[name])

    n_markers = len(matched_boa_indices)
    print(f"  Matched common markers: {n_markers} / {len(boa_col_names)} ({n_markers/len(boa_col_names)*100:.1f}%)")
    print(f"  Angus-counted BOA markers to invert (2 - O): {np.sum(boa_is_angus_counted)}")

    # 2. Extract animal IDs
    print("\n[2. Extracting animal IDs from BOA.raw.gz...]")
    animal_ids = []
    with gzip.open(BOA_RAW, "rt") as f_boa:
        f_boa.readline()
        for line in f_boa:
            parts = line.split(maxsplit=1)
            animal_ids.append(parts[0].strip())

    validate_ids(animal_ids)
    pheno_path=DATA_DIR / "Thermo.Phenotypes.csv"
    pheno=pd.read_csv(pheno_path,dtype={"UFID":str})
    validate_ids(pheno.UFID,"phenotype IDs")
    # A fixed historical reference independent of availability of future phenotypes.
    ref_ids=pheno.loc[pheno.Year<=2018,"UFID"].tolist()
    ref_idx=np.array([i for i,aid in enumerate(animal_ids) if aid in set(ref_ids)])
    if not len(ref_idx): raise ValueError("No historical genotype reference animals")
    hashes=source_hashes([SNP_RAW,BOA_RAW,pheno_path])
    n_animals = len(animal_ids)
    print(f"  Total animals: {n_animals}")

    # 3. Stream all markers into compact int8 arrays (each takes ~391 MB)
    print(f"\n[3. Streaming all {n_markers} markers for {n_animals} cattle into memory...]")
    t_read = time.time()
    
    # Load BOA
    print("  Streaming BOA pseudo-genotypes...")
    boa_mat = np.empty((n_animals, n_markers), dtype=np.int8)
    with gzip.open(BOA_RAW, "rt") as f_boa:
        f_boa.readline()
        for row_idx, line in enumerate(f_boa):
            parts = line.split()
            boa_mat[row_idx, :] = [int(parts[c]) for c in matched_boa_indices]
            if (row_idx + 1) % 1000 == 0:
                print(f"    Loaded {row_idx + 1}/{n_animals} cattle BOA...")

    # Vectorized harmonization: O strictly = Brahman allele count
    print("  Vectorizing BOA harmonization to Brahman count (0=Angus, 2=Brahman)...")
    angus_mask = boa_is_angus_counted[np.array(matched_boa_indices)-1]
    boa_mat[:, angus_mask] = 2 - boa_mat[:, angus_mask]

    # Load SNP
    print("  Streaming SNP genotypes...")
    snp_mat = np.empty((n_animals, n_markers), dtype=np.int8)
    id_to_idx = {aid: i for i, aid in enumerate(animal_ids)}
    loaded_snp_count = 0
    seen_ids=set()
    with gzip.open(SNP_RAW, "rt") as f_snp:
        f_snp.readline()
        for line in f_snp:
            parts = line.split()
            iid = parts[0].strip()
            if iid in seen_ids: raise ValueError(f"Duplicate SNP animal ID {iid}")
            seen_ids.add(iid)
            if iid in id_to_idx:
                row_idx = id_to_idx[iid]
                row_vals = []
                for c in matched_snp_indices:
                    val = parts[c]
                    row_vals.append(int(val) if val not in ("NA", "-9") else -1)
                snp_mat[row_idx, :] = row_vals
                loaded_snp_count += 1
                if loaded_snp_count % 1000 == 0:
                    print(f"    Loaded {loaded_snp_count}/{n_animals} cattle SNP...")
    # Rows are placed by ID. Every BOA animal must have SNP calls; SNP-only animals (four in the
    # public release) have no BOA record and are left out.
    missing_snp = sorted(set(animal_ids) - seen_ids)
    if missing_snp: raise ValueError(f"{len(missing_snp)} BOA animals have no SNP record: {missing_snp[:10]}")
    snp_only = sorted(seen_ids - set(animal_ids))
    if snp_only: print(f"  Ignored {len(snp_only)} SNP-only animals without BOA calls: {snp_only}")
    print(f"  Matched and loaded SNP genotypes for {loaded_snp_count} cattle.")

    print(f"  Streaming completed in {time.time() - t_read:.2f}s.")

    # Accumulate in marker blocks; every statistic is fitted on historical rows.
    print("\n[4. Computing reference-only imputation and GRMs in marker blocks...]")
    t_grm=time.time()
    def build(matrix):
        product=np.zeros((n_animals,n_animals),dtype=np.float64)
        denominator=0.; frequencies=[]
        for start in range(0,n_markers,2048):
            X=matrix[:,start:start+2048].astype(np.float32)
            X[X<0]=np.nan
            ref=X[ref_idx]
            counts=np.sum(np.isfinite(ref),axis=0)
            if np.any(counts==0): raise ValueError("Entirely missing reference marker; repair/filter input")
            means=np.nansum(ref,axis=0)/counts
            X=np.where(np.isnan(X),means,X).astype(np.float32)
            p=means/2.; frequencies.extend(p.tolist())
            denominator+=float(np.sum(2*p*(1-p)))
            Z=X-means
            product += Z @ Z.T
        if denominator<=0: raise ValueError("No polymorphic reference markers")
        return product/denominator,denominator,np.array(frequencies)
    G_122k,denom_snp,p_snp=build(snp_mat)
    B_122k,denom_boa,q_boa=build(boa_mat)

    print(f"  G_122k computed: shape {G_122k.shape}, diagonal mean = {np.mean(np.diag(G_122k)):.4f}, off-diag mean = {np.mean(G_122k[np.triu_indices(n_animals, 1)]):.4f}")
    print(f"  B_122k computed: shape {B_122k.shape}, diagonal mean = {np.mean(np.diag(B_122k)):.4f}, off-diag mean = {np.mean(B_122k[np.triu_indices(n_animals, 1)]):.4f}")
    print(f"  GRM calculation took {time.time() - t_grm:.2f}s.")

    # 6. Save compressed NPZ
    print(f"\n[6. Saving full-density GRM cache to {OUT_NPZ}...]")
    np.savez_compressed(
        OUT_NPZ,
        preprocessing_version="train_reference_v3",
        source_hashes=json.dumps(hashes,sort_keys=True),
        reference_ids=np.array([animal_ids[i] for i in ref_idx]),
        reference_year_max=2018,
        p_snp=p_snp, p_boa=q_boa,
        G_122k=G_122k.astype(np.float32),
        B_122k=B_122k.astype(np.float32),
        animal_ids=np.array(animal_ids),
        num_markers=n_markers,
        denom_snp=float(denom_snp),
        denom_boa=float(denom_boa)
    )
    print(f"  Saved {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / (1024*1024):.2f} MB). Total elapsed: {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    build_full_density_grm()

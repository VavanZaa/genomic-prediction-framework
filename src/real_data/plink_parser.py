"""PLINK binary and text parser with standard quality control filters for cattle genomic data.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PlinkData:
    bim: pd.DataFrame       # Columns: chrom, snp_id, pos_m, pos_bp, a1, a2
    fam: pd.DataFrame       # Columns: fid, iid, sire, dam, sex, pheno
    genotypes: np.ndarray   # Shape (N, P), values in {0, 1, 2}, or -1 for missing
    provenance: str = "unknown"  # "empirical" or "synthetic", from the directory's PROVENANCE file

    @property
    def num_individuals(self) -> int:
        return len(self.fam)

    @property
    def num_snps(self) -> int:
        return len(self.bim)

    def get_individual_index(self, iid: str) -> Optional[int]:
        matches = np.where(self.fam["iid"] == iid)[0]
        return int(matches[0]) if len(matches) > 0 else None

    def to_raw(self, output_prefix: Union[str, Path]) -> None:
        """Export as PLINK .raw additive genotype table."""
        out_path = Path(output_prefix).with_suffix(".raw")
        cols = list(self.fam.columns)
        df_out = self.fam.copy()
        snp_cols = [f"{snp}_{a1}" for snp, a1 in zip(self.bim["snp_id"], self.bim["a1"])]
        geno_df = pd.DataFrame(self.genotypes, columns=snp_cols)
        full_df = pd.concat([df_out.reset_index(drop=True), geno_df.reset_index(drop=True)], axis=1)
        full_df.to_csv(out_path, sep=" ", index=False)


class PlinkParser:
    """Reads and writes PLINK binary (.bed/.bim/.fam) and text formats."""

    @staticmethod
    def read_bim(bim_path: Union[str, Path]) -> pd.DataFrame:
        cols = ["chrom", "snp_id", "pos_m", "pos_bp", "a1", "a2"]
        df = pd.read_csv(bim_path, sep=r"\s+", header=None, names=cols, dtype={"chrom": str, "snp_id": str})
        return df

    @staticmethod
    def read_fam(fam_path: Union[str, Path]) -> pd.DataFrame:
        cols = ["fid", "iid", "sire", "dam", "sex", "pheno"]
        df = pd.read_csv(fam_path, sep=r"\s+", header=None, names=cols, dtype=str)
        # Convert numeric sex and pheno where applicable
        df["sex"] = df["sex"].replace({"1": "M", "2": "F", "0": "U"})
        return df

    @classmethod
    def read_bed(cls, bed_path: Union[str, Path], n_ind: int, n_snp: int) -> np.ndarray:
        """Parses standard binary PLINK .bed file in SNP-major mode."""
        with open(bed_path, "rb") as f:
            magic = f.read(3)
            if len(magic) < 3 or magic[0] != 0x6C or magic[1] != 0x1B:
                raise ValueError("Invalid PLINK .bed file signature (must start with 0x6C, 0x1B).")
            mode = magic[2]
            if mode != 0x01:
                raise NotImplementedError("Only SNP-major mode (0x01) is supported.")

            bytes_per_snp = (n_ind + 3) // 4
            raw_bytes = f.read(bytes_per_snp * n_snp)

        geno_matrix = np.empty((n_ind, n_snp), dtype=np.int8)
        # Mapping 2-bit PLINK codes:
        # 00 (0) -> 2 (homozygous A1)
        # 01 (1) -> -1 (missing)
        # 10 (2) -> 1 (heterozygous)
        # 11 (3) -> 0 (homozygous A2)
        lut = np.array([2, -1, 1, 0], dtype=np.int8)

        for s in range(n_snp):
            snp_bytes = raw_bytes[s * bytes_per_snp : (s + 1) * bytes_per_snp]
            b_arr = np.frombuffer(snp_bytes, dtype=np.uint8)

            # Unpack 4 individuals per byte
            i0 = b_arr & 0x03
            i1 = (b_arr >> 2) & 0x03
            i2 = (b_arr >> 4) & 0x03
            i3 = (b_arr >> 6) & 0x03

            unpacked = np.column_stack([i0, i1, i2, i3]).ravel()[:n_ind]
            geno_matrix[:, s] = lut[unpacked]

        return geno_matrix

    @classmethod
    def read_plink(cls, prefix: Union[str, Path]) -> PlinkData:
        p = Path(prefix)
        bim_path = p.with_suffix(".bim")
        fam_path = p.with_suffix(".fam")
        bed_path = p.with_suffix(".bed")

        if not bim_path.exists() or not fam_path.exists() or not bed_path.exists():
            raise FileNotFoundError(f"Missing PLINK files for prefix {prefix} (.bed, .bim, or .fam)")

        bim = cls.read_bim(bim_path)
        fam = cls.read_fam(fam_path)
        geno = cls.read_bed(bed_path, len(fam), len(bim))
        from .provenance import read_provenance
        return PlinkData(bim=bim, fam=fam, genotypes=geno, provenance=read_provenance(p.parent))

    @staticmethod
    def write_plink(data: PlinkData, output_prefix: Union[str, Path]) -> None:
        """Writes PlinkData to standard .bed/.bim/.fam binary trio."""
        p = Path(output_prefix)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write .bim
        bim_df = data.bim.copy()
        bim_df.to_csv(p.with_suffix(".bim"), sep="\t", header=False, index=False)

        # Write .fam
        fam_df = data.fam.copy()
        fam_df_out = pd.DataFrame({
            "fid": fam_df["fid"],
            "iid": fam_df["iid"],
            "sire": fam_df["sire"].replace({"": "0", None: "0"}),
            "dam": fam_df["dam"].replace({"": "0", None: "0"}),
            "sex": fam_df["sex"].map({"M": "1", "F": "2", "U": "0"}).fillna("0"),
            "pheno": fam_df["pheno"].replace({"": "-9", None: "-9"}) if "pheno" in fam_df else "-9"
        })
        fam_df_out.to_csv(p.with_suffix(".fam"), sep="\t", header=False, index=False)

        # Write .bed
        n_ind, n_snp = data.genotypes.shape
        bytes_per_snp = (n_ind + 3) // 4

        # Inverse mapping from {2, -1, 1, 0} to {0, 1, 2, 3}
        reverse_lut = {2: 0, -1: 1, 1: 2, 0: 3}

        with open(p.with_suffix(".bed"), "wb") as f:
            # Header
            f.write(bytes([0x6C, 0x1B, 0x01]))
            for s in range(n_snp):
                col = data.genotypes[:, s]
                snp_bytes = bytearray(bytes_per_snp)
                for b_idx in range(bytes_per_snp):
                    byte_val = 0
                    for bit_pos in range(4):
                        ind_idx = b_idx * 4 + bit_pos
                        if ind_idx < n_ind:
                            g = col[ind_idx]
                            val = reverse_lut.get(int(g), 1)
                        else:
                            val = 0
                        byte_val |= (val << (bit_pos * 2))
                    snp_bytes[b_idx] = byte_val
                f.write(snp_bytes)


class PlinkQC:
    """Standard quality control filters for genomic data."""

    @staticmethod
    def filter(
        data: PlinkData,
        min_maf: float = 0.02,
        max_missing_snp: float = 0.10,
        max_missing_ind: float = 0.10,
        hwe_pval: float = 1e-6
    ) -> Tuple[PlinkData, Dict[str, int]]:
        M = data.genotypes.copy()
        N, P = M.shape

        # 1. Filter individuals by missing rate
        ind_missing = np.mean(M == -1, axis=1)
        valid_inds = ind_missing <= max_missing_ind

        M_filtered = M[valid_inds, :]
        fam_filtered = data.fam.iloc[valid_inds].reset_index(drop=True)

        # 2. Filter SNPs by missing rate
        snp_missing = np.mean(M_filtered == -1, axis=0)
        valid_missing_snp = snp_missing <= max_missing_snp

        # 3. Filter SNPs by MAF and HWE on non-missing
        valid_maf = np.zeros(P, dtype=bool)
        valid_hwe = np.zeros(P, dtype=bool)

        for s in range(P):
            if not valid_missing_snp[s]:
                continue
            col = M_filtered[:, s]
            non_miss = col[col != -1]
            if len(non_miss) == 0:
                continue

            # Minor allele frequency
            af = np.mean(non_miss) / 2.0
            maf = min(af, 1.0 - af)
            if maf >= min_maf:
                valid_maf[s] = True

            # Hardy-Weinberg equilibrium test
            n_aa = np.sum(non_miss == 0)
            n_ab = np.sum(non_miss == 1)
            n_bb = np.sum(non_miss == 2)
            n_tot = len(non_miss)

            p_a = (2.0 * n_aa + n_ab) / (2.0 * n_tot)
            p_b = 1.0 - p_a

            exp_aa = (p_a**2) * n_tot
            exp_ab = (2.0 * p_a * p_b) * n_tot
            exp_bb = (p_b**2) * n_tot

            if min(exp_aa, exp_ab, exp_bb) > 1.0:
                chi2 = ((n_aa - exp_aa)**2 / exp_aa +
                        (n_ab - exp_ab)**2 / exp_ab +
                        (n_bb - exp_bb)**2 / exp_bb)
                pval = float(1.0 - stats.chi2.cdf(chi2, df=1))
                if pval >= hwe_pval:
                    valid_hwe[s] = True
            else:
                valid_hwe[s] = True

        final_snps = valid_missing_snp & valid_maf & valid_hwe

        qc_report = {
            "initial_individuals": N,
            "passed_individuals": int(np.sum(valid_inds)),
            "initial_snps": P,
            "passed_snps": int(np.sum(final_snps)),
            "failed_maf": int(np.sum(~valid_maf & valid_missing_snp)),
            "failed_hwe": int(np.sum(~valid_hwe & valid_maf & valid_missing_snp))
        }

        filtered_data = PlinkData(
            bim=data.bim.iloc[final_snps].reset_index(drop=True),
            fam=fam_filtered,
            genotypes=M_filtered[:, final_snps],
            provenance=data.provenance,
        )
        return filtered_data, qc_report

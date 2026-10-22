"""Parent-offspring trio inheritance validation, Mendelian error checking, and transmission probability calibration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from .plink_parser import PlinkData


@dataclass
class TrioValidationReport:
    num_trios_evaluated: int
    num_duos_evaluated: int
    total_locus_evaluations: int
    mendelian_error_count: int
    mendelian_error_rate: float
    transition_probabilities: Dict[Tuple[int, int], np.ndarray] # (sire, dam) -> [P(O=0), P(O=1), P(O=2)]
    theoretical_discrepancy_rmse: float
    crossover_summary_by_chrom: pd.DataFrame  # opposite-homozygote jump counts; not crossovers (see validate_transmission)
    is_calibrated: bool


class TransmissionValidator:
    """Tests allele transmission and recombination distributions against real parent-offspring trios."""

    # Theoretical Mendelian transition probabilities: (Sire, Dam) -> [P(0), P(1), P(2)]
    THEORETICAL_MENDELIAN: Dict[Tuple[int, int], np.ndarray] = {
        (0, 0): np.array([1.0, 0.0, 0.0]),
        (0, 1): np.array([0.5, 0.5, 0.0]),
        (1, 0): np.array([0.5, 0.5, 0.0]),
        (0, 2): np.array([0.0, 1.0, 0.0]),
        (2, 0): np.array([0.0, 1.0, 0.0]),
        (1, 1): np.array([0.25, 0.50, 0.25]),
        (1, 2): np.array([0.0, 0.5, 0.5]),
        (2, 1): np.array([0.0, 0.5, 0.5]),
        (2, 2): np.array([0.0, 0.0, 1.0]),
    }

    def __init__(self, plink_data: PlinkData):
        self.data = plink_data
        self.id_to_idx = {str(iid): idx for idx, iid in enumerate(self.data.fam["iid"])}

    def find_trios(self) -> List[Tuple[str, str, str]]:
        """Identify (sire, dam, offspring) trios where all 3 animals have genotypes."""
        trios = []
        fam = self.data.fam
        for _, row in fam.iterrows():
            off_id = str(row["iid"])
            sire_id = str(row["sire"])
            dam_id = str(row["dam"])
            if sire_id in self.id_to_idx and dam_id in self.id_to_idx:
                trios.append((sire_id, dam_id, off_id))
        return trios

    def find_duos(self) -> List[Tuple[str, str, str]]:
        """Identify parent-offspring duos where exactly one parent is genotyped."""
        duos = []
        fam = self.data.fam
        for _, row in fam.iterrows():
            off_id = str(row["iid"])
            sire_id = str(row["sire"])
            dam_id = str(row["dam"])
            has_sire = sire_id in self.id_to_idx
            has_dam = dam_id in self.id_to_idx
            if has_sire and not has_dam:
                duos.append((sire_id, "Unknown", off_id))
            elif has_dam and not has_sire:
                duos.append(("Unknown", dam_id, off_id))
        return duos

    def validate_transmission(self) -> TrioValidationReport:
        trios = self.find_trios()
        duos = self.find_duos()
        num_snps = self.data.num_snps

        if not trios and not duos:
            raise ValueError("No parent-offspring trios or duos found in PLINK pedigree.")

        # Transition counts: (sire, dam) -> count of offspring with 0, 1, 2
        trans_counts: Dict[Tuple[int, int], np.ndarray] = {
            k: np.zeros(3, dtype=np.int64) for k in self.THEORETICAL_MENDELIAN
        }

        total_evals = 0
        mendelian_errors = 0

        # Evaluate trios
        for sire_id, dam_id, off_id in trios:
            s_idx = self.id_to_idx[sire_id]
            d_idx = self.id_to_idx[dam_id]
            o_idx = self.id_to_idx[off_id]

            s_geno = self.data.genotypes[s_idx]
            d_geno = self.data.genotypes[d_idx]
            o_geno = self.data.genotypes[o_idx]

            # Filter valid loci where none are missing
            valid = (s_geno != -1) & (d_geno != -1) & (o_geno != -1)
            total_evals += int(np.sum(valid))

            for s_val, d_val, o_val in zip(s_geno[valid], d_geno[valid], o_geno[valid]):
                pair = (int(s_val), int(d_val))
                if pair in trans_counts:
                    trans_counts[pair][int(o_val)] += 1
                    # Check compatibility
                    theo = self.THEORETICAL_MENDELIAN[pair]
                    if theo[int(o_val)] == 0.0:
                        mendelian_errors += 1

        error_rate = mendelian_errors / max(1, total_evals)

        # Compute empirical transition probabilities and RMSE against theoretical
        empirical_probs: Dict[Tuple[int, int], np.ndarray] = {}
        sq_diffs = []

        for pair, counts in trans_counts.items():
            tot = np.sum(counts)
            if tot > 0:
                p_emp = counts / tot
            else:
                p_emp = self.THEORETICAL_MENDELIAN[pair].copy()
            empirical_probs[pair] = p_emp
            sq_diffs.append(np.mean((p_emp - self.THEORETICAL_MENDELIAN[pair])**2))

        rmse = float(np.sqrt(np.mean(sq_diffs)))

        # Per-chromosome count of jumps between opposite homozygous offspring genotypes at consecutive
        # fully genotyped markers. This is NOT a crossover count: crossovers can only be located with
        # phased parents (e.g. three-generation data). It is kept as a crude genotype-consistency signal.
        chrom_records = []
        bim = self.data.bim
        unique_chroms = bim["chrom"].unique()

        for c in unique_chroms:
            c_mask = (bim["chrom"] == c).values
            c_snps = np.sum(c_mask)
            if c_snps < 2:
                continue

            jumps = 0
            for sire_id, dam_id, off_id in trios:
                o_c = self.data.genotypes[self.id_to_idx[off_id], c_mask]
                s_c = self.data.genotypes[self.id_to_idx[sire_id], c_mask]
                d_c = self.data.genotypes[self.id_to_idx[dam_id], c_mask]
                genotyped = (s_c != -1) & (d_c != -1) & (o_c != -1)
                o_gen = o_c[genotyped]
                if len(o_gen) >= 2:
                    jumps += int(np.sum(np.abs(np.diff(o_gen)) == 2))

            chrom_records.append({
                "chrom": str(c),
                "num_snps": int(c_snps),
                "trios_evaluated": len(trios),
                "opposite_homozygote_jumps": int(jumps),
                "mean_opposite_homozygote_jumps_per_calf": float(jumps / max(1, len(trios)))
            })

        df_chrom = pd.DataFrame(chrom_records)

        # Calibration criterion: RMSE < 0.05 and Mendelian error rate < 0.02
        is_calibrated = (rmse < 0.05) and (error_rate < 0.02)

        return TrioValidationReport(
            num_trios_evaluated=len(trios),
            num_duos_evaluated=len(duos),
            total_locus_evaluations=total_evals,
            mendelian_error_count=mendelian_errors,
            mendelian_error_rate=error_rate,
            transition_probabilities=empirical_probs,
            theoretical_discrepancy_rmse=rmse,
            crossover_summary_by_chrom=df_chrom,
            is_calibrated=is_calibrated
        )

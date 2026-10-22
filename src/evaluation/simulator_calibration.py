"""Cohort-Distribution Simulator Calibration Suite (Layer 5A).

Quantifies the empirical fidelity of simulated populations against prospective cohort distributions.
NOTE ON EVIDENCE LEVEL:
In the public Florida multibreed dataset, commercial heifers do not have genotyped parents in the panel.
Layer 5A evaluates Cohort-Distribution Simulator Calibration against the prospective test cohort.
Layer 5B is reserved for true parent-offspring trio validation when genotyped trios are available.

Computes the 7-dimensional raw discrepancy vector:
    D = (D_AFS, D_Het, D_LD, D_Tract, D_Spec, D_Pheno, D_Tail)
And evaluates weight-invariance via 10,000 Dirichlet weight draws:
    P(D_sim_real(Model B) < D_sim_real(Model A)).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import linalg, stats


@dataclass
class CalibrationMetrics:
    """Detailed breakdown of calibration discrepancies between simulated and real populations."""
    d_afs: float
    d_het: float
    d_ld: float
    d_tract: float
    d_spec: float
    d_pheno: float
    d_tail: float
    d_composite: float
    metric_weights: Dict[str, float]
    summary_dict: Dict[str, float] = field(default_factory=dict)
    null_medians: Optional[np.ndarray] = None
    null_means: Optional[np.ndarray] = None
    null_stds: Optional[np.ndarray] = None
    dimensionless_vector: Optional[np.ndarray] = None
    d_composite_dimensionless: Optional[float] = None

    @property
    def raw_vector(self) -> np.ndarray:
        """Returns the 7-dimensional discrepancy vector D."""
        return np.array([
            self.d_afs,
            self.d_het,
            self.d_ld,
            self.d_tract,
            self.d_spec,
            self.d_pheno,
            self.d_tail
        ])

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"Metric": "Allele Frequency Spectrum (D_AFS)", "Discrepancy": self.d_afs, "Weight": self.metric_weights.get("afs", 1.0)},
            {"Metric": "Heterozygosity Distribution (D_Het)", "Discrepancy": self.d_het, "Weight": self.metric_weights.get("het", 1.0)},
            {"Metric": "LD Decay Curve (D_LD)", "Discrepancy": self.d_ld, "Weight": self.metric_weights.get("ld", 1.0)},
            {"Metric": "Ancestry Tract Length (D_Tract)", "Discrepancy": self.d_tract, "Weight": self.metric_weights.get("tract", 1.0)},
            {"Metric": "GRM Eigenvalue Spectrum (D_Spec)", "Discrepancy": self.d_spec, "Weight": self.metric_weights.get("spec", 1.0)},
            {"Metric": "Phenotype / GEBV Dist (D_Pheno)", "Discrepancy": self.d_pheno, "Weight": self.metric_weights.get("pheno", 1.0)},
            {"Metric": "Extreme Tail Attainment (D_Tail)", "Discrepancy": self.d_tail, "Weight": self.metric_weights.get("tail", 1.0)},
            {"Metric": "COMPOSITE CALIBRATION (D_sim-real)", "Discrepancy": self.d_composite, "Weight": 1.0},
        ]
        if self.dimensionless_vector is not None:
            for idx, r in enumerate(rows[:7]):
                r["Dimensionless_Discrepancy"] = float(self.dimensionless_vector[idx])
            rows[7]["Dimensionless_Discrepancy"] = float(self.d_composite_dimensionless or 0.0)
        return pd.DataFrame(rows)


class SimulatorCalibrator:
    """Evaluates multi-dimensional empirical fidelity of simulated genomes against real descendants."""

    DEFAULT_WEIGHTS = {
        "afs": 1.0,
        "het": 1.0,
        "ld": 1.0,
        "tract": 1.0,
        "spec": 1.0,
        "pheno": 1.0,
        "tail": 1.0,
    }

    @staticmethod
    def compute_afs_discrepancy(g_real: np.ndarray, g_sim: np.ndarray) -> float:
        """Computes Wasserstein-1 distance between the allele frequency spectra."""
        p_real = np.mean(g_real, axis=0) / 2.0
        p_sim = np.mean(g_sim, axis=0) / 2.0
        return float(stats.wasserstein_distance(p_real, p_sim))

    @staticmethod
    def compute_heterozygosity_discrepancy(g_real: np.ndarray, g_sim: np.ndarray) -> float:
        """Computes Wasserstein-1 distance between individual observed heterozygosities."""
        het_real = np.mean(g_real == 1, axis=1)
        het_sim = np.mean(g_sim == 1, axis=1)
        return float(stats.wasserstein_distance(het_real, het_sim))

    @staticmethod
    def compute_ld_decay_discrepancy(
        g_real: np.ndarray,
        g_sim: np.ndarray,
        marker_map: pd.DataFrame,
        distance_bins: Optional[List[Tuple[float, float]]] = None,
        max_pairs_per_chrom: int = 1500,
        random_seed: int = 42
    ) -> float:
        """Computes discrepancy in Linkage Disequilibrium (r^2) decay across physical distance bins."""
        if distance_bins is None:
            distance_bins = [
                (0, 50_000),
                (50_000, 200_000),
                (200_000, 500_000),
                (500_000, 1_000_000),
                (1_000_000, 3_000_000),
                (3_000_000, 10_000_000)
            ]

        rng = np.random.default_rng(random_seed)
        chroms = marker_map["chrom"].unique()

        bin_r2_real: Dict[int, List[float]] = {b_idx: [] for b_idx in range(len(distance_bins))}
        bin_r2_sim: Dict[int, List[float]] = {b_idx: [] for b_idx in range(len(distance_bins))}

        for chrom in chroms:
            chr_mask = (marker_map["chrom"] == chrom).values
            chr_indices = np.where(chr_mask)[0]
            m_c = len(chr_indices)
            if m_c < 2:
                continue

            pos_col = "pos" if "pos" in marker_map.columns else "pos_bp"
            pos = marker_map.iloc[chr_indices][pos_col].values
            g_c_real = g_real[:, chr_indices]
            g_c_sim = g_sim[:, chr_indices]

            # Sample pairs
            n_sample = min(max_pairs_per_chrom, (m_c * (m_c - 1)) // 2)
            idx1 = rng.integers(0, m_c, size=n_sample * 2)
            idx2 = rng.integers(0, m_c, size=n_sample * 2)
            valid_mask = idx1 < idx2
            idx1, idx2 = idx1[valid_mask][:n_sample], idx2[valid_mask][:n_sample]

            dist = np.abs(pos[idx2] - pos[idx1])

            for b_idx, (d_low, d_high) in enumerate(distance_bins):
                in_bin = (dist >= d_low) & (dist < d_high)
                b_i1 = idx1[in_bin]
                b_i2 = idx2[in_bin]

                for p1, p2 in zip(b_i1, b_i2):
                    var_r1 = np.var(g_c_real[:, p1])
                    var_r2 = np.var(g_c_real[:, p2])
                    if var_r1 > 1e-6 and var_r2 > 1e-6:
                        corr_r = np.corrcoef(g_c_real[:, p1], g_c_real[:, p2])[0, 1]
                        if not np.isnan(corr_r):
                            bin_r2_real[b_idx].append(corr_r ** 2)

                    var_s1 = np.var(g_c_sim[:, p1])
                    var_s2 = np.var(g_c_sim[:, p2])
                    if var_s1 > 1e-6 and var_s2 > 1e-6:
                        corr_s = np.corrcoef(g_c_sim[:, p1], g_c_sim[:, p2])[0, 1]
                        if not np.isnan(corr_s):
                            bin_r2_sim[b_idx].append(corr_s ** 2)

        bin_diffs = []
        for b_idx in range(len(distance_bins)):
            r2_r = np.mean(bin_r2_real[b_idx]) if len(bin_r2_real[b_idx]) > 0 else 0.0
            r2_s = np.mean(bin_r2_sim[b_idx]) if len(bin_r2_sim[b_idx]) > 0 else 0.0
            bin_diffs.append(abs(r2_r - r2_s))

        return float(np.mean(bin_diffs)) if len(bin_diffs) > 0 else 0.0

    @staticmethod
    def compute_tract_length_discrepancy(
        boa_real: Optional[np.ndarray],
        boa_sim: Optional[np.ndarray],
        marker_map: pd.DataFrame
    ) -> float:
        """Computes discrepancy in local ancestry contiguous tract length distribution."""
        if boa_real is None or boa_sim is None:
            return 0.0

        def extract_tract_lengths(boa_mat: np.ndarray) -> np.ndarray:
            tract_lengths = []
            chroms = marker_map["chrom"].unique()
            for chrom in chroms:
                chr_indices = np.where(marker_map["chrom"] == chrom)[0]
                if len(chr_indices) < 2:
                    continue
                chr_boa = boa_mat[:, chr_indices]
                for i in range(chr_boa.shape[0]):
                    row = chr_boa[i, :]
                    diffs = np.diff(row)
                    transitions = np.where(diffs != 0)[0] + 1
                    starts = np.concatenate([[0], transitions])
                    ends = np.concatenate([transitions, [len(row)]])
                    lengths = ends - starts
                    tract_lengths.extend(lengths.tolist())
            return np.array(tract_lengths, dtype=float)

        tracts_r = extract_tract_lengths(boa_real)
        tracts_s = extract_tract_lengths(boa_sim)

        if len(tracts_r) == 0 or len(tracts_s) == 0:
            return 0.0

        mean_len = float(np.mean(tracts_r))
        w1 = float(stats.wasserstein_distance(tracts_r, tracts_s))
        return w1 / (mean_len + 1e-6)

    @staticmethod
    def compute_spectral_discrepancy(
        g_real: np.ndarray,
        g_sim: np.ndarray,
        n_components: int = 25
    ) -> float:
        """Computes discrepancy between the normalized eigenvalue spectra of the empirical GRMs."""
        n_r = g_real.shape[0]
        n_s = g_sim.shape[0]
        p_r = np.mean(g_real, axis=0) / 2.0
        p_s = np.mean(g_sim, axis=0) / 2.0

        denom_r = 2.0 * np.sum(p_r * (1.0 - p_r)) + 1e-6
        denom_s = 2.0 * np.sum(p_s * (1.0 - p_s)) + 1e-6

        z_r = g_real - 2.0 * p_r
        z_s = g_sim - 2.0 * p_s

        grm_r = (z_r @ z_r.T) / denom_r
        grm_s = (z_s @ z_s.T) / denom_s

        k = min(n_components, n_r - 1, n_s - 1)
        eig_r = np.sort(np.linalg.eigvalsh(grm_r))[::-1][:k]
        eig_s = np.sort(np.linalg.eigvalsh(grm_s))[::-1][:k]

        eig_r = np.maximum(eig_r, 0.0)
        eig_s = np.maximum(eig_s, 0.0)

        # Normalize spectra to probability simplex
        sum_r = np.sum(eig_r) + 1e-9
        sum_s = np.sum(eig_s) + 1e-9

        spec_r = eig_r / sum_r
        spec_s = eig_s / sum_s

        return float(np.sum(np.abs(spec_r - spec_s)))

    @staticmethod
    def compute_phenotype_discrepancy(
        y_real: Optional[np.ndarray],
        y_sim: Optional[np.ndarray]
    ) -> float:
        """Computes bivariate Wasserstein distance between phenotype / GEBV distributions."""
        if y_real is None or y_sim is None:
            return 0.0

        if y_real.ndim == 1:
            y_r = y_real.reshape(-1, 1)
            y_s = y_sim.reshape(-1, 1)
        else:
            y_r = y_real
            y_s = y_sim

        n_traits = y_r.shape[1]
        w_dists = []
        for t in range(n_traits):
            w = float(stats.wasserstein_distance(y_r[:, t], y_s[:, t]))
            std_r = float(np.std(y_r[:, t])) + 1e-6
            w_dists.append(w / std_r)

        return float(np.mean(w_dists))

    @staticmethod
    def compute_tail_discrepancy(
        y_real: Optional[np.ndarray],
        y_sim: Optional[np.ndarray],
        thresholds: Optional[List[float]] = None
    ) -> float:
        """Computes discrepancy in extreme bivariate lower-tail attainment probability."""
        if y_real is None or y_sim is None or thresholds is None:
            return 0.0

        if y_real.ndim == 1 or y_real.shape[1] == 1:
            tau = thresholds[0]
            p_r = np.mean(y_real < tau)
            p_s = np.mean(y_sim < tau)
        else:
            tau_l, tau_h = thresholds[0], thresholds[1]
            p_r = np.mean((y_real[:, 0] < tau_l) & (y_real[:, 1] < tau_h))
            p_s = np.mean((y_sim[:, 0] < tau_l) & (y_sim[:, 1] < tau_h))

        return float(abs(p_r - p_s))

    @classmethod
    def evaluate_simulation(
        cls,
        g_real: np.ndarray,
        g_sim: np.ndarray,
        marker_map: pd.DataFrame,
        boa_real: Optional[np.ndarray] = None,
        boa_sim: Optional[np.ndarray] = None,
        y_real: Optional[np.ndarray] = None,
        y_sim: Optional[np.ndarray] = None,
        thresholds: Optional[List[float]] = None,
        weights: Optional[Dict[str, float]] = None,
        null_medians: Optional[np.ndarray] = None,
        null_means: Optional[np.ndarray] = None,
        null_stds: Optional[np.ndarray] = None
    ) -> CalibrationMetrics:
        """Evaluates all discrepancy metrics and returns composite calibration score."""
        w = cls.DEFAULT_WEIGHTS.copy()
        if weights is not None:
            w.update(weights)

        d_afs = cls.compute_afs_discrepancy(g_real, g_sim)
        d_het = cls.compute_heterozygosity_discrepancy(g_real, g_sim)
        d_ld = cls.compute_ld_decay_discrepancy(g_real, g_sim, marker_map)
        d_tract = cls.compute_tract_length_discrepancy(boa_real, boa_sim, marker_map)
        d_spec = cls.compute_spectral_discrepancy(g_real, g_sim)
        d_pheno = cls.compute_phenotype_discrepancy(y_real, y_sim)
        d_tail = cls.compute_tail_discrepancy(y_real, y_sim, thresholds)

        raw_components = {
            "afs": d_afs,
            "het": d_het,
            "ld": d_ld,
            "tract": d_tract,
            "spec": d_spec,
            "pheno": d_pheno,
            "tail": d_tail,
        }

        # Weighted average
        active_keys = [k for k in raw_components.keys() if raw_components[k] > 0.0 or k in ["afs", "het", "ld", "spec"]]
        total_w = sum(w.get(k, 1.0) for k in active_keys)
        d_composite = sum(w.get(k, 1.0) * raw_components[k] for k in active_keys) / (total_w + 1e-9)

        raw_vec = np.array([d_afs, d_het, d_ld, d_tract, d_spec, d_pheno, d_tail])
        dim_vec = None
        d_comp_dim = None

        if null_medians is not None:
            dim_vec = raw_vec / np.maximum(null_medians, 1e-6)
            d_comp_dim = float(np.mean(dim_vec))

        return CalibrationMetrics(
            d_afs=d_afs,
            d_het=d_het,
            d_ld=d_ld,
            d_tract=d_tract,
            d_spec=d_spec,
            d_pheno=d_pheno,
            d_tail=d_tail,
            d_composite=float(d_composite),
            metric_weights=w,
            summary_dict=raw_components,
            null_medians=null_medians,
            null_means=null_means,
            null_stds=null_stds,
            dimensionless_vector=dim_vec,
            d_composite_dimensionless=d_comp_dim
        )

    @classmethod
    def compute_biological_null_distribution(
        cls,
        g_real: np.ndarray,
        marker_map: pd.DataFrame,
        boa_real: Optional[np.ndarray] = None,
        y_real: Optional[np.ndarray] = None,
        thresholds: Optional[List[float]] = None,
        n_splits: int = 25,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """Computes real-vs-real biological null discrepancy distributions via split-half resampling.
        
        Evaluates D_k,null = D_k(R_1, R_2) on matched bootstrap halves of real cattle to obtain
        the empirical biological noise floor for each dimension.
        """
        rng = np.random.default_rng(random_seed)
        n = g_real.shape[0]
        null_records = []

        for _ in range(n_splits):
            perm = rng.permutation(n)
            half = n // 2
            r1, r2 = perm[:half], perm[half:2 * half]

            d_afs = cls.compute_afs_discrepancy(g_real[r1], g_real[r2])
            d_het = cls.compute_heterozygosity_discrepancy(g_real[r1], g_real[r2])
            d_ld = cls.compute_ld_decay_discrepancy(g_real[r1], g_real[r2], marker_map)
            d_tract = cls.compute_tract_length_discrepancy(boa_real[r1], boa_real[r2], marker_map) if boa_real is not None else 0.0
            d_spec = cls.compute_spectral_discrepancy(g_real[r1], g_real[r2])
            d_pheno = cls.compute_phenotype_discrepancy(y_real[r1], y_real[r2]) if y_real is not None else 0.0
            d_tail = cls.compute_tail_discrepancy(y_real[r1], y_real[r2], thresholds) if (y_real is not None and thresholds is not None) else 0.0

            null_records.append([d_afs, d_het, d_ld, d_tract, d_spec, d_pheno, d_tail])

        arr = np.array(null_records)
        return {
            "medians": np.median(arr, axis=0),
            "means": np.mean(arr, axis=0),
            "stds": np.std(arr, axis=0) + 1e-9,
            "raw_draws": arr
        }

    @staticmethod
    def evaluate_dirichlet_robustness(
        calib_a: CalibrationMetrics,
        calib_b: CalibrationMetrics,
        n_samples: int = 10000,
        random_seed: int = 42
    ) -> Dict[str, Union[float, Tuple[float, float], int]]:
        """Evaluates P(D_B < D_A) across n_samples random Dirichlet weight vectors.
        
        Evaluates both raw and dimensionless discrepancies, returning the central 95% interval
        across weight preference draws (capturing metric preference sensitivity, not biological sampling error).
        """
        rng = np.random.default_rng(random_seed)
        vec_a = calib_a.raw_vector
        vec_b = calib_b.raw_vector

        alpha = np.ones(len(vec_a))
        weights = rng.dirichlet(alpha, size=n_samples)

        d_a_samples = weights @ vec_a
        d_b_samples = weights @ vec_b
        prob_b_wins = float(np.mean(d_b_samples < d_a_samples))
        deltas = d_a_samples - d_b_samples
        mean_delta = float(np.mean(deltas))
        ci_raw_95 = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

        # Dimensionless evaluation if available
        prob_dim_wins = prob_b_wins
        mean_dim_delta = mean_delta
        ci_dim_95 = ci_raw_95

        if calib_a.dimensionless_vector is not None and calib_b.dimensionless_vector is not None:
            dim_a = calib_a.dimensionless_vector
            dim_b = calib_b.dimensionless_vector
            dim_a_samples = weights @ dim_a
            dim_b_samples = weights @ dim_b
            prob_dim_wins = float(np.mean(dim_b_samples < dim_a_samples))
            dim_deltas = dim_a_samples - dim_b_samples
            mean_dim_delta = float(np.mean(dim_deltas))
            ci_dim_95 = (float(np.percentile(dim_deltas, 2.5)), float(np.percentile(dim_deltas, 97.5)))

        return {
            "prob_raw_b_superior": prob_b_wins,
            "prob_dimensionless_b_superior": prob_dim_wins,
            "mean_raw_delta": mean_delta,
            "mean_dimensionless_delta": mean_dim_delta,
            "central_95_raw_weight_interval": ci_raw_95,
            "central_95_dim_weight_interval": ci_dim_95,
            # Backward-compatible aliases
            "prob_model_b_superior": prob_b_wins,
            "mean_discrepancy_advantage": mean_delta,
            "ci95_advantage": ci_raw_95,
            "num_samples": n_samples
        }

    @classmethod
    def bootstrap_biological_sampling_confidence(
        cls,
        g_real: np.ndarray,
        g_sim_a: np.ndarray,
        g_sim_b: np.ndarray,
        marker_map: pd.DataFrame,
        boa_real: Optional[np.ndarray] = None,
        boa_sim_a: Optional[np.ndarray] = None,
        boa_sim_b: Optional[np.ndarray] = None,
        y_real: Optional[np.ndarray] = None,
        y_sim_a: Optional[np.ndarray] = None,
        y_sim_b: Optional[np.ndarray] = None,
        thresholds: Optional[List[float]] = None,
        null_medians: Optional[np.ndarray] = None,
        n_bootstraps: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, Union[float, Tuple[float, float], int]]:
        """Bootstraps real cattle records to compute true biological sampling confidence intervals for Delta D = D_A - D_B.
        
        Evaluates both dimensionless real-vs-real normalized composite discrepancy and raw discrepancy
        across B resamples, reporting replicate distributions: D_tilde_A^(b), D_tilde_B^(b), Delta^(b),
        with medians, 95% bootstrap CIs, and standard continuity-corrected p-values.
        """
        rng = np.random.default_rng(random_seed)
        n = g_real.shape[0]
        n_sim = g_sim_a.shape[0]
        m = g_real.shape[1]

        # Pre-select representative marker pairs for fast vectorized LD decay evaluation
        n_sample_pairs = min(1500, m * (m - 1) // 2)
        idx1 = rng.integers(0, m, size=n_sample_pairs * 2)
        idx2 = rng.integers(0, m, size=n_sample_pairs * 2)
        valid_pairs = idx1 < idx2
        p1_sub = idx1[valid_pairs][:n_sample_pairs]
        p2_sub = idx2[valid_pairs][:n_sample_pairs]

        # Pre-extract animal tract lengths once to eliminate loop overhead in bootstrap
        chrom_indices_list = [np.where(marker_map["chrom"] == c)[0] for c in marker_map["chrom"].unique()]
        def extract_animal_tracts(boa_mat):
            all_animal_tracts = []
            for i in range(boa_mat.shape[0]):
                tracts = []
                for c_idx in chrom_indices_list:
                    if len(c_idx) < 2:
                        continue
                    row = boa_mat[i, c_idx]
                    diffs = np.diff(row)
                    transitions = np.where(diffs != 0)[0] + 1
                    starts = np.concatenate([[0], transitions])
                    ends = np.concatenate([transitions, [len(row)]])
                    tracts.extend((ends - starts).tolist())
                all_animal_tracts.append(np.array(tracts, dtype=np.float32))
            return all_animal_tracts

        pre_tracts_r = extract_animal_tracts(boa_real) if boa_real is not None else None
        pre_tracts_sa = extract_animal_tracts(boa_sim_a) if boa_sim_a is not None else None
        pre_tracts_sb = extract_animal_tracts(boa_sim_b) if boa_sim_b is not None else None

        # Standardize genotypes once for fast spectral discrepancy
        p_ref = np.clip(np.mean(g_real, axis=0) / 2.0, 0.01, 0.99)
        std_ref = np.sqrt(2.0 * p_ref * (1.0 - p_ref))
        m_sub = min(400, m)
        sub_markers = rng.choice(m, size=m_sub, replace=False)
        Z_r_sub = ((g_real[:, sub_markers] - 2.0 * p_ref[sub_markers]) / std_ref[sub_markers]).astype(np.float32)
        Z_sa_sub = ((g_sim_a[:, sub_markers] - 2.0 * p_ref[sub_markers]) / std_ref[sub_markers]).astype(np.float32)
        Z_sb_sub = ((g_sim_b[:, sub_markers] - 2.0 * p_ref[sub_markers]) / std_ref[sub_markers]).astype(np.float32)

        raw_a_list = []
        raw_b_list = []
        dim_a_list = []
        dim_b_list = []

        for b in range(n_bootstraps):
            b_idx = rng.choice(n, size=n, replace=True)
            s_idx = rng.choice(n_sim, size=n, replace=True)

            g_r = g_real[b_idx]
            g_sa = g_sim_a[s_idx]
            g_sb = g_sim_b[s_idx]

            # 1. D_AFS
            p_r = np.mean(g_r, axis=0) / 2.0
            d_afs_a = float(stats.wasserstein_distance(p_r, np.mean(g_sa, axis=0) / 2.0))
            d_afs_b = float(stats.wasserstein_distance(p_r, np.mean(g_sb, axis=0) / 2.0))

            # 2. D_Het
            h_r = np.mean(g_r == 1, axis=1)
            d_het_a = float(stats.wasserstein_distance(h_r, np.mean(g_sa == 1, axis=1)))
            d_het_b = float(stats.wasserstein_distance(h_r, np.mean(g_sb == 1, axis=1)))

            # 3. D_LD (vectorized across pre-selected pairs)
            x_r, y_r = g_r[:, p1_sub].astype(np.float32), g_r[:, p2_sub].astype(np.float32)
            r_r = np.mean((x_r - np.mean(x_r, axis=0)) * (y_r - np.mean(y_r, axis=0)), axis=0) / (np.std(x_r, axis=0) * np.std(y_r, axis=0) + 1e-9)
            r2_r = np.mean(r_r ** 2)

            x_sa, y_sa = g_sa[:, p1_sub].astype(np.float32), g_sa[:, p2_sub].astype(np.float32)
            r_sa = np.mean((x_sa - np.mean(x_sa, axis=0)) * (y_sa - np.mean(y_sa, axis=0)), axis=0) / (np.std(x_sa, axis=0) * np.std(y_sa, axis=0) + 1e-9)
            r2_sa = np.mean(r_sa ** 2)

            x_sb, y_sb = g_sb[:, p1_sub].astype(np.float32), g_sb[:, p2_sub].astype(np.float32)
            r_sb = np.mean((x_sb - np.mean(x_sb, axis=0)) * (y_sb - np.mean(y_sb, axis=0)), axis=0) / (np.std(x_sb, axis=0) * np.std(y_sb, axis=0) + 1e-9)
            r2_sb = np.mean(r_sb ** 2)

            d_ld_a = float(abs(r2_r - r2_sa))
            d_ld_b = float(abs(r2_r - r2_sb))

            # 4. D_Tract (concatenation of pre-extracted animal tracts, normalized by mean real tract length)
            d_tract_a, d_tract_b = 0.0, 0.0
            if pre_tracts_r is not None and pre_tracts_sa is not None and pre_tracts_sb is not None:
                tr_r = np.concatenate([pre_tracts_r[i] for i in b_idx[:100]])
                tr_sa = np.concatenate([pre_tracts_sa[i] for i in s_idx[:100]])
                tr_sb = np.concatenate([pre_tracts_sb[i] for i in s_idx[:100]])
                mean_tr_r = float(np.mean(tr_r)) + 1e-6
                d_tract_a = float(stats.wasserstein_distance(tr_r, tr_sa)) / mean_tr_r
                d_tract_b = float(stats.wasserstein_distance(tr_r, tr_sb)) / mean_tr_r

            # 5. D_Spec (fast SVD on representative standardized marker subset, L1 spectral distance)
            s_r = linalg.svdvals(Z_r_sub[b_idx[:200]])[:10]
            s_r = s_r / (np.sum(s_r) + 1e-9)
            s_sa = linalg.svdvals(Z_sa_sub[s_idx[:200]])[:10]
            s_sa = s_sa / (np.sum(s_sa) + 1e-9)
            s_sb = linalg.svdvals(Z_sb_sub[s_idx[:200]])[:10]
            s_sb = s_sb / (np.sum(s_sb) + 1e-9)
            d_spec_a = float(np.sum(np.abs(s_r - s_sa)))
            d_spec_b = float(np.sum(np.abs(s_r - s_sb)))

            # 6. D_Pheno
            d_pheno_a, d_pheno_b = 0.0, 0.0
            if y_real is not None and y_sim_a is not None and y_sim_b is not None:
                d_pheno_a = cls.compute_phenotype_discrepancy(y_real[b_idx], y_sim_a[s_idx])
                d_pheno_b = cls.compute_phenotype_discrepancy(y_real[b_idx], y_sim_b[s_idx])

            # 7. D_Tail
            d_tail_a, d_tail_b = 0.0, 0.0
            if y_real is not None and thresholds is not None:
                d_tail_a = cls.compute_tail_discrepancy(y_real[b_idx], y_sim_a[s_idx], thresholds)
                d_tail_b = cls.compute_tail_discrepancy(y_real[b_idx], y_sim_b[s_idx], thresholds)

            vec_a = np.array([d_afs_a, d_het_a, d_ld_a, d_tract_a, d_spec_a, d_pheno_a, d_tail_a], dtype=np.float64)
            vec_b = np.array([d_afs_b, d_het_b, d_ld_b, d_tract_b, d_spec_b, d_pheno_b, d_tail_b], dtype=np.float64)

            # Raw mean composite (matches evaluate_simulation d_composite)
            raw_a_list.append(float(np.mean(vec_a)))
            raw_b_list.append(float(np.mean(vec_b)))

            # Dimensionless mean composite (matches evaluate_simulation d_composite_dimensionless)
            if null_medians is not None:
                dim_vec_a = vec_a / np.maximum(null_medians, 1e-6)
                dim_vec_b = vec_b / np.maximum(null_medians, 1e-6)
                dim_a_list.append(float(np.mean(dim_vec_a)))
                dim_b_list.append(float(np.mean(dim_vec_b)))
            else:
                dim_a_list.append(float(np.mean(vec_a)))
                dim_b_list.append(float(np.mean(vec_b)))

        raw_a_arr = np.array(raw_a_list)
        raw_b_arr = np.array(raw_b_list)
        delta_raw_arr = raw_a_arr - raw_b_arr

        dim_a_arr = np.array(dim_a_list)
        dim_b_arr = np.array(dim_b_list)
        delta_dim_arr = dim_a_arr - dim_b_arr

        # Dimensionless statistics
        median_dim_a = float(np.median(dim_a_arr))
        median_dim_b = float(np.median(dim_b_arr))
        median_delta_dim = float(np.median(delta_dim_arr))
        mean_delta_dim = float(np.mean(delta_dim_arr))
        std_delta_dim = float(np.std(delta_dim_arr))
        ci95_dim = (float(np.percentile(delta_dim_arr, 2.5)), float(np.percentile(delta_dim_arr, 97.5)))
        k_le_zero_dim = int(np.sum(delta_dim_arr <= 0.0))
        p_val_dim = float(min(1.0, 2.0 * (k_le_zero_dim + 1) / (n_bootstraps + 1)))

        # Raw statistics
        median_raw_a = float(np.median(raw_a_arr))
        median_raw_b = float(np.median(raw_b_arr))
        median_delta_raw = float(np.median(delta_raw_arr))
        mean_delta_raw = float(np.mean(delta_raw_arr))
        std_delta_raw = float(np.std(delta_raw_arr))
        ci95_raw = (float(np.percentile(delta_raw_arr, 2.5)), float(np.percentile(delta_raw_arr, 97.5)))
        k_le_zero_raw = int(np.sum(delta_raw_arr <= 0.0))
        p_val_raw = float(min(1.0, 2.0 * (k_le_zero_raw + 1) / (n_bootstraps + 1)))

        return {
            # Primary dimensionless metrics (matches headline D_tilde scale)
            "median_dim_a": median_dim_a,
            "median_dim_b": median_dim_b,
            "median_delta_dim": median_delta_dim,
            "mean_delta_dim": mean_delta_dim,
            "std_delta_dim": std_delta_dim,
            "ci95_dim": ci95_dim,
            "p_val_dim": p_val_dim,
            "k_le_zero_dim": k_le_zero_dim,
            # Raw composite metrics
            "median_raw_a": median_raw_a,
            "median_raw_b": median_raw_b,
            "median_delta_raw": median_delta_raw,
            "mean_delta_raw": mean_delta_raw,
            "std_delta_raw": std_delta_raw,
            "ci95_raw": ci95_raw,
            "p_val_raw": p_val_raw,
            "k_le_zero_raw": k_le_zero_raw,
            # Aliases for backward compatibility
            "mean_delta_d": mean_delta_dim,
            "std_delta_d": std_delta_dim,
            "biological_ci95": ci95_dim,
            "p_value_two_sided": p_val_dim,
            "n_bootstraps": n_bootstraps
        }

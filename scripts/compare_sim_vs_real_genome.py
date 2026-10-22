"""Empirical Comparison: Simulated Genome Architecture vs. Real Bovine Genomic Data.

Compares:
1. Minor Allele Frequency (MAF) distribution / Site Frequency Spectrum.
2. Observed Heterozygosity (Ho) distribution across individuals.
3. Linkage Disequilibrium (LD, r^2) decay across physical genomic distance.
4. Genomic Relationship Matrix (GRM) eigenvalue spectrum (PCA / population structure).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.florida_loader import FloridaLoader
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.prediction.models import AdditiveGBLUP


def run_comparison():
    print("=" * 80)
    print("EMPIRICAL EVALUATION: SIMULATED GENOME vs. REAL FLORIDA BOVINE GENOME")
    print("=" * 80)

    out_dir = BASE_DIR / "output" / "real_layer3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Real Florida Dataset
    print("\n[1] Ingesting Real Florida Multibreed Genotypes & Map...")
    real_data = FloridaLoader.load_dataset(marker_step=24, min_maf=0.03)
    # The loader keeps missing calls as NaN (~0.6%); fill them from the historical (Year <= 2018)
    # reference so LD, heterozygosity and the GRM are defined. Imputed means are never counted as
    # heterozygous, so Ho is slightly conservative.
    from src.real_data.genotype_preprocessing import GenotypeImputer
    ref = (real_data.phenotypes["Year"] <= 2018).to_numpy()
    G_real = GenotypeImputer().fit(real_data.genotypes[ref]).transform(real_data.genotypes)  # (3194, M)
    map_real = real_data.marker_map
    n_ind_real, n_snps = G_real.shape
    print(f"  Real data: {n_ind_real} animals x {n_snps} SNPs across {len(map_real['chrom'].unique())} autosomes.")

    # 2. Simulate Synthetic Population of Identical Dimension
    print("\n[2] Generating Synthetic Benchmark Population (Matching N and M)...")
    rng = np.random.default_rng(42)

    # Synthetic breed allele frequencies (Beta model as used in synthetic pipelines)
    # Angus vs Brahman with delta drift
    p_angus = rng.beta(2.0, 2.0, size=n_snps)
    p_brahman = np.clip(p_angus + rng.normal(0, 0.25, size=n_snps), 0.05, 0.95)

    # Synthetic multibreed population with real Angus fractions q ~ N(0.71, 0.07^2)
    q_vec = np.clip(rng.normal(0.71, 0.07, size=n_ind_real), 0.0, 1.0)

    # Independent binomial founder generation (current synthetic model)
    G_sim = np.zeros((n_ind_real, n_snps), dtype=np.int8)
    for i in range(n_ind_real):
        p_i = q_vec[i] * p_angus + (1.0 - q_vec[i]) * p_brahman
        G_sim[i, :] = rng.binomial(2, p_i)

    print(f"  Synthetic data: {n_ind_real} animals x {n_snps} SNPs generated under Beta-Binomial model.")

    # -------------------------------------------------------------------------
    # Metric 1: Minor Allele Frequency (MAF) Spectrum
    # -------------------------------------------------------------------------
    print("\n[3] Computing Site Frequency Spectrum (MAF)...")
    p_real = np.mean(G_real, axis=0) / 2.0
    maf_real = np.minimum(p_real, 1.0 - p_real)

    p_sim = np.mean(G_sim, axis=0) / 2.0
    maf_sim = np.minimum(p_sim, 1.0 - p_sim)

    ks_stat_maf, p_val_maf = stats.ks_2samp(maf_real, maf_sim)
    print(f"  Real MAF: mean={np.mean(maf_real):.4f}, std={np.std(maf_real):.4f}, median={np.median(maf_real):.4f}")
    print(f"  Sim  MAF: mean={np.mean(maf_sim):.4f}, std={np.std(maf_sim):.4f}, median={np.median(maf_sim):.4f}")
    print(f"  Kolmogorov-Smirnov Test (MAF): stat={ks_stat_maf:.4f}, p={p_val_maf:.4e}")

    # -------------------------------------------------------------------------
    # Metric 2: Observed Heterozygosity (Ho) Spectrum across Individuals
    # -------------------------------------------------------------------------
    print("\n[4] Computing Individual Observed Heterozygosity (Ho)...")
    ho_real = np.mean(G_real == 1, axis=1)
    ho_sim = np.mean(G_sim == 1, axis=1)

    ks_stat_ho, p_val_ho = stats.ks_2samp(ho_real, ho_sim)
    print(f"  Real Ho: mean={np.mean(ho_real):.4f}, std={np.std(ho_real):.4f}, range=[{np.min(ho_real):.4f}, {np.max(ho_real):.4f}]")
    print(f"  Sim  Ho: mean={np.mean(ho_sim):.4f}, std={np.std(ho_sim):.4f}, range=[{np.min(ho_sim):.4f}, {np.max(ho_sim):.4f}]")
    print(f"  Kolmogorov-Smirnov Test (Ho): stat={ks_stat_ho:.4f}, p={p_val_ho:.4e}")

    # -------------------------------------------------------------------------
    # Metric 3: Linkage Disequilibrium (LD, r^2) on Bovine Chromosome 1 (BTA 1)
    # -------------------------------------------------------------------------
    print("\n[5] Computing Linkage Disequilibrium (r^2) Decay on BTA 1...")
    bta1_mask = (map_real["chrom"] == 1).values
    bta1_bps = map_real.loc[bta1_mask, "pos_bp"].values
    bta1_real = G_real[:, bta1_mask].astype(float)
    bta1_sim = G_sim[:, bta1_mask].astype(float)
    n_bta1_snps = len(bta1_bps)
    print(f"  Evaluating {n_bta1_snps} SNPs on BTA 1...")

    # Center genotypes
    bta1_real_c = (bta1_real - np.mean(bta1_real, axis=0)) / (np.std(bta1_real, axis=0) + 1e-8)
    bta1_sim_c = (bta1_sim - np.mean(bta1_sim, axis=0)) / (np.std(bta1_sim, axis=0) + 1e-8)

    sample_pairs = []
    for i in range(min(n_bta1_snps, 300)):
        for j in range(i + 1, min(n_bta1_snps, 300)):
            dist_mb = (bta1_bps[j] - bta1_bps[i]) / 1e6
            if dist_mb <= 15.0:
                sample_pairs.append((i, j, dist_mb))

    print(f"  Computing r^2 for {len(sample_pairs):,} marker pairs within 15 Mb...")
    dist_list = []
    r2_real_list = []
    r2_sim_list = []

    for i, j, dist_mb in sample_pairs:
        r_real = np.dot(bta1_real_c[:, i], bta1_real_c[:, j]) / n_ind_real
        r_sim = np.dot(bta1_sim_c[:, i], bta1_sim_c[:, j]) / n_ind_real
        dist_list.append(dist_mb)
        r2_real_list.append(r_real ** 2)
        r2_sim_list.append(r_sim ** 2)

    df_ld = pd.DataFrame({"dist_mb": dist_list, "r2_real": r2_real_list, "r2_sim": r2_sim_list})
    bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    df_ld["bin"] = pd.cut(df_ld["dist_mb"], bins=bins)
    ld_binned = df_ld.groupby("bin", observed=True)[["r2_real", "r2_sim"]].mean()
    print("\n  Linkage Disequilibrium (r^2) by Distance Bin:")
    print(ld_binned.to_string())

    # -------------------------------------------------------------------------
    # Metric 4: Genomic Relationship Matrix (GRM) PCA Spectrum
    # -------------------------------------------------------------------------
    print("\n[6] Computing Genomic Relationship Matrix (VanRaden) Spectrum...")
    sub_idx = rng.choice(n_ind_real, size=min(1000, n_ind_real), replace=False)
    
    # Real GRM
    Z_real = G_real[sub_idx, :].astype(float) - 2.0 * p_real
    denom_real = 2.0 * np.sum(p_real * (1.0 - p_real))
    G_mat_real = np.dot(Z_real, Z_real.T) / denom_real
    eig_real = np.linalg.eigvalsh(G_mat_real)[::-1]
    eig_real_norm = eig_real / np.sum(np.maximum(eig_real, 0))

    # Sim GRM
    Z_sim = G_sim[sub_idx, :].astype(float) - 2.0 * p_sim
    denom_sim = 2.0 * np.sum(p_sim * (1.0 - p_sim))
    G_mat_sim = np.dot(Z_sim, Z_sim.T) / denom_sim
    eig_sim = np.linalg.eigvalsh(G_mat_sim)[::-1]
    eig_sim_norm = eig_sim / np.sum(np.maximum(eig_sim, 0))

    print(f"  Real GRM Top 3 Eigenvalue Variance: PC1={eig_real_norm[0]*100:.2f}%, PC2={eig_real_norm[1]*100:.2f}%, PC3={eig_real_norm[2]*100:.2f}%")
    print(f"  Sim  GRM Top 3 Eigenvalue Variance: PC1={eig_sim_norm[0]*100:.2f}%, PC2={eig_sim_norm[1]*100:.2f}%, PC3={eig_sim_norm[2]*100:.2f}%")

    # -------------------------------------------------------------------------
    # 5. Diagnostic 4-Panel Visualization
    # -------------------------------------------------------------------------
    print("\n[7] Generating Comparison Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel A: MAF Spectrum
    ax = axes[0, 0]
    bins_maf = np.linspace(0.0, 0.5, 25)
    ax.hist(maf_real, bins=bins_maf, alpha=0.6, color="#1f77b4", label=f"Real Florida Chip (mean={np.mean(maf_real):.2f})", density=True)
    ax.hist(maf_sim, bins=bins_maf, alpha=0.6, color="#ff7f0e", label=f"Simulated Beta(2,2) (mean={np.mean(maf_sim):.2f})", density=True)
    ax.set_title("A. Site Frequency Spectrum (MAF)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Minor Allele Frequency (MAF)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Individual Heterozygosity
    ax = axes[0, 1]
    bins_ho = np.linspace(0.15, 0.55, 30)
    ax.hist(ho_real, bins=bins_ho, alpha=0.6, color="#1f77b4", label=f"Real Cattle (mean={np.mean(ho_real):.3f})", density=True)
    ax.hist(ho_sim, bins=bins_ho, alpha=0.6, color="#ff7f0e", label=f"Simulated Cattle (mean={np.mean(ho_sim):.3f})", density=True)
    ax.set_title("B. Individual Observed Heterozygosity ($H_o$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Observed Heterozygosity ($H_o$)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: LD Decay (r^2 vs Distance)
    ax = axes[1, 0]
    bin_centers = [0.25, 0.75, 1.5, 3.5, 7.5, 12.5]
    ax.plot(bin_centers, ld_binned["r2_real"].values, marker="o", lw=2, color="#1f77b4", label="Real Cattle BTA 1 ($r^2$ decay)")
    ax.plot(bin_centers, ld_binned["r2_sim"].values, marker="s", lw=2, color="#ff7f0e", linestyle="--", label="Simulated (Independent Founders)")
    ax.set_title("C. Linkage Disequilibrium ($r^2$) Decay on BTA 1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Genomic Distance (Mb)", fontsize=10)
    ax.set_ylabel("Mean $r^2$", fontsize=10)
    ax.set_ylim(0, max(np.max(ld_binned["r2_real"]) * 1.3, 0.1))
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: GRM Eigenvalue Spectrum (PCA Scree Plot)
    ax = axes[1, 1]
    pcs = np.arange(1, 21)
    ax.plot(pcs, eig_real_norm[:20] * 100, marker="o", lw=2, color="#1f77b4", label=f"Real GRM (PC1={eig_real_norm[0]*100:.1f}%)")
    ax.plot(pcs, eig_sim_norm[:20] * 100, marker="s", lw=2, color="#ff7f0e", linestyle="--", label=f"Simulated GRM (PC1={eig_sim_norm[0]*100:.1f}%)")
    ax.set_title("D. Genomic Relationship Matrix (GRM) PCA Scree", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Variance Explained (%)", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / "simulated_vs_real_genome_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


    # Export metrics table
    # Assessments are computed, not written in advance. (A fixed "prediction accuracy" row with numbers
    # this script never computed was removed; prediction is compared in compare_genome_simulation_methods.py.)
    def ratio(sim, real):
        return f"simulated/real = {sim / real:.2f}" if real else "real value is 0"
    ld_real = float(ld_binned.loc[ld_binned.index[0], "r2_real"])
    ld_sim = float(ld_binned.loc[ld_binned.index[0], "r2_sim"])
    summary_df = pd.DataFrame([
        {"Metric": "MAF Mean ± SD", "Real Cattle": f"{np.mean(maf_real):.4f} ± {np.std(maf_real):.4f}", "Simulated Model": f"{np.mean(maf_sim):.4f} ± {np.std(maf_sim):.4f}",
         "Match Assessment": f"KS statistic {ks_stat_maf:.3f}"},
        {"Metric": "Heterozygosity Ho", "Real Cattle": f"{np.mean(ho_real):.4f} ± {np.std(ho_real):.4f}", "Simulated Model": f"{np.mean(ho_sim):.4f} ± {np.std(ho_sim):.4f}",
         "Match Assessment": f"KS statistic {ks_stat_ho:.3f}; mean {ratio(np.mean(ho_sim), np.mean(ho_real))}"},
        {"Metric": "LD r^2 (< 0.5 Mb)", "Real Cattle": f"{ld_real:.4f}", "Simulated Model": f"{ld_sim:.4f}",
         "Match Assessment": ratio(ld_sim, ld_real)},
        {"Metric": "GRM PC1 Variance", "Real Cattle": f"{eig_real_norm[0]*100:.2f}%", "Simulated Model": f"{eig_sim_norm[0]*100:.2f}%",
         "Match Assessment": ratio(eig_sim_norm[0], eig_real_norm[0])},
    ])
    summary_csv = out_dir / "simulated_vs_real_genome_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved metrics: {summary_csv}")
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_comparison()

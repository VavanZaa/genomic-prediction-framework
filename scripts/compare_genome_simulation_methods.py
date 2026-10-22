"""Benchmark Suite: Comparing Genome Simulation Methods against Physical Cattle Data.

Methods Evaluated:
1. Naive Baseline (Independent Beta-Binomial Founder Sampling)
2. Method 1: Coalescent Historical Burn-in (msprime: Ne=100 bottleneck, divergence, recombination)
3. Method 2: Real Haplotype Resampling (Florida genotype templates, random phasing, mosaic template switches)
3b. Method 1b: the same coalescent population seen through the real array (ArrayAscertainment: real MAF
    spectrum, QC floor, even spacing, real missing-call rate, then reference imputation)
4. Ground Truth: Real Physical Cattle (University of Florida Angus-Brahman Multibreed Herd)

Evaluates:
- Linkage Disequilibrium (r^2) decay across physical genomic distance (0 - 15 Mb)
- Individual Observed Heterozygosity (Ho) spectrum & inbreeding depression
- Minor Allele Frequency (MAF) spectrum & KS distance
- Genomic Relationship Matrix (GRM) PCA scree & PC1-PC3 variance
- Out-of-sample Genomic Prediction Accuracy (GBLUP r, rho, RMSE) under calibrated h2=0.25
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.florida_loader import FloridaLoader
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.simulation.coalescent_engine import BovineCoalescentSimulator
from src.simulation.real_haplotype_engine import RealHaplotypeResampler
from src.prediction.models import AdditiveGBLUP


def run_methods_benchmark():
    print("=" * 85)
    print("BENCHMARK: COALESCENT vs REAL HAPLOTYPE RESAMPLING vs NAIVE SIMULATION vs REAL CATTLE")
    print("=" * 85)

    out_dir = BASE_DIR / "output" / "real_layer3"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------------
    # 1. Ingest Real Physical Florida Cattle (Ground Truth)
    # -------------------------------------------------------------------------
    print("\n[1/4] Ingesting Real Physical Florida Cattle Dataset...")
    real_data = FloridaLoader.load_dataset(marker_step=24, min_maf=0.03)
    # The loader keeps missing calls as NaN (~0.6%); fill them from the historical (Year <= 2018)
    # reference so LD, heterozygosity and the GRM are defined. Imputed means are never counted as
    # heterozygous, so Ho is slightly conservative.
    from src.real_data.genotype_preprocessing import GenotypeImputer
    ref = (real_data.phenotypes["Year"] <= 2018).to_numpy()
    G_real = GenotypeImputer().fit(real_data.genotypes[ref]).transform(real_data.genotypes)  # (3194, M)
    map_real = real_data.marker_map
    n_animals, n_snps = G_real.shape
    print(f"  Real Data: {n_animals} cattle x {n_snps} SNPs across 29 autosomes.")

    # -------------------------------------------------------------------------
    # 2. Generate Method 1: Coalescent Historical Burn-in (msprime)
    # -------------------------------------------------------------------------
    print("\n[2/4] Simulating Method 1: Coalescent Historical Burn-in (msprime)...")
    coal_sim = BovineCoalescentSimulator()
    coal_kw = dict(num_animals=n_animals, num_markers=n_snps, chrom_length_bp=15_000_000,
                   angus_fraction_mean=0.71, angus_fraction_std=0.07, num_founder_haps=120, random_seed=42)
    coal_pop = coal_sim.simulate_multibreed_population(**coal_kw)
    G_coal = coal_pop.genotypes[:, :n_snps]
    map_coal = coal_pop.marker_map.iloc[:n_snps].copy()
    print(f"  Coalescent Data: {G_coal.shape[0]} cattle x {G_coal.shape[1]} SNPs with historical Ne=100 coalescence.")
    print(f"  Status: {BovineCoalescentSimulator.STATUS}")

    # Method 1b: the same simulation through the real array's measurement process
    from src.simulation.ascertainment import ArrayAscertainment
    asc = ArrayAscertainment(min_maf=0.03).fit(real_data.genotypes, real_data.marker_map)
    coal_asc_pop = coal_sim.simulate_multibreed_population(**coal_kw, ascertainment=asc, discovery="admixed")
    G_coal_asc = asc.add_missing_calls(coal_asc_pop.genotypes, np.random.default_rng(7))
    G_coal_asc = GenotypeImputer().fit(G_coal_asc).transform(G_coal_asc)
    map_coal_asc = coal_asc_pop.marker_map
    sim_span_mb = sum((g["pos_bp"].max() - g["pos_bp"].min()) / 1e6 for _, g in map_coal_asc.groupby("chrom"))
    print(f"  Array-ascertained coalescent: {G_coal_asc.shape[1]} markers (last-chromosome shortfall "
          f"{asc.last_shortfall_}); missing-call rate {asc.missing_rate_:.4f}; density "
          f"{G_coal_asc.shape[1] / sim_span_mb:.1f} markers/Mb vs real {asc.markers_per_mb_:.2f} (genome length not matched)")

    # -------------------------------------------------------------------------
    # 3. Generate Method 2: Real Haplotype Resampling
    # -------------------------------------------------------------------------
    print("\n[3/4] Simulating Method 2: Real Haplotype Resampling (Florida Templates)...")
    resampler = RealHaplotypeResampler(data=real_data)
    real_hap_pop = resampler.simulate_multibreed_population(
        num_animals=n_animals,
        angus_fraction_mean=0.71,
        angus_fraction_std=0.07,
        crossover_rate_per_chrom=1.0,
        random_seed=42
    )
    G_realhap = real_hap_pop.genotypes
    map_realhap = map_real
    print(f"  Real Haplotype Data: {G_realhap.shape[0]} cattle x {G_realhap.shape[1]} SNPs with real founder LD.")

    # -------------------------------------------------------------------------
    # 4. Generate Naive Baseline (Independent Beta-Binomial)
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Naive Baseline (Independent Beta-Binomial Sampling)...")
    p_angus = rng.beta(2.0, 2.0, size=n_snps)
    p_brahman = np.clip(p_angus + rng.normal(0, 0.25, size=n_snps), 0.05, 0.95)
    q_vec = np.clip(rng.normal(0.71, 0.07, size=n_animals), 0.0, 1.0)
    G_naive = np.zeros((n_animals, n_snps), dtype=np.int8)
    for i in range(n_animals):
        p_i = q_vec[i] * p_angus + (1.0 - q_vec[i]) * p_brahman
        G_naive[i, :] = rng.binomial(2, p_i)
    print(f"  Naive Baseline Data: {G_naive.shape[0]} cattle x {G_naive.shape[1]} SNPs.")

    # =========================================================================
    # Comparative Evaluation
    # =========================================================================
    models = {
        "Real Physical Cattle": (G_real, map_real),
        "Method 2: Real Haplotype Resampling": (G_realhap, map_realhap),
        "Method 1: Coalescent Burn-in": (G_coal, map_coal),
        "Method 1b: Coalescent + array ascertainment": (G_coal_asc, map_coal_asc),
        "Naive Beta-Binomial Baseline": (G_naive, map_real),
    }

    # --- Metric 1: MAF Spectrum ---
    print("\n--- Computing Minor Allele Frequency (MAF) Spectra ---")
    maf_dict = {}
    for name, (g_mat, _) in models.items():
        p_vec = np.mean(g_mat, axis=0) / 2.0
        maf = np.minimum(p_vec, 1.0 - p_vec)
        maf_dict[name] = maf

    p_real_ref = maf_dict["Real Physical Cattle"]
    ks_results = {}
    for name, maf in maf_dict.items():
        stat, pval = stats.ks_2samp(p_real_ref, maf)
        ks_results[name] = (stat, pval)
        print(f"  {name:36s}: Mean={np.mean(maf):.4f}, Std={np.std(maf):.4f}, KS-stat={stat:.4f} (p={pval:.2e})")

    # --- Metric 2: Observed Heterozygosity (Ho) ---
    print("\n--- Computing Individual Observed Heterozygosity (Ho) ---")
    ho_dict = {}
    for name, (g_mat, _) in models.items():
        ho = np.mean(g_mat == 1, axis=1)
        ho_dict[name] = ho
        print(f"  {name:36s}: Mean={np.mean(ho):.4f}, Std={np.std(ho):.4f}, Range=[{np.min(ho):.4f}, {np.max(ho):.4f}]")

    # --- Metric 3: Linkage Disequilibrium (r^2) on Chromosome 1 ---
    print("\n--- Computing Linkage Disequilibrium (r^2) on Chromosome 1 ---")
    ld_binned_dict = {}
    bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    for name, (g_mat, m_df) in models.items():
        chr1_mask = (m_df["chrom"] == 1).values
        chr1_pos = m_df.loc[chr1_mask, "pos_bp"].values
        chr1_g = g_mat[:, chr1_mask].astype(float)
        n_c1_snps = len(chr1_pos)

        # Normalize markers
        sd = np.std(chr1_g, axis=0)
        valid_idx = np.where(sd > 1e-6)[0]
        chr1_g = chr1_g[:, valid_idx]
        chr1_pos = chr1_pos[valid_idx]
        chr1_norm = (chr1_g - np.mean(chr1_g, axis=0)) / np.std(chr1_g, axis=0)

        # At most 250 markers, spread evenly along the chromosome so every distance bin is populated
        keep = np.unique(np.linspace(0, len(chr1_pos) - 1, min(len(chr1_pos), 250)).astype(int))
        chr1_norm, chr1_pos = chr1_norm[:, keep], chr1_pos[keep]
        pairs = []
        n_eval = len(chr1_pos)
        for i in range(n_eval):
            for j in range(i + 1, n_eval):
                d_mb = abs(chr1_pos[j] - chr1_pos[i]) / 1e6
                if d_mb <= 15.0:
                    pairs.append((i, j, d_mb))

        dist_arr = []
        r2_arr = []
        for i, j, d_mb in pairs:
            r = np.dot(chr1_norm[:, i], chr1_norm[:, j]) / n_animals
            dist_arr.append(d_mb)
            r2_arr.append(r ** 2)

        df_ld = pd.DataFrame({"dist_mb": dist_arr, "r2": r2_arr})
        df_ld["bin"] = pd.cut(df_ld["dist_mb"], bins=bins)
        binned = df_ld.groupby("bin", observed=False)["r2"].mean()   # every bin kept; NaN if empty
        ld_binned_dict[name] = binned
        print(f"  {name:36s}: r^2 (< 0.5 Mb)={binned.iloc[0]:.4f}, (5 - 10 Mb)={binned.iloc[4]:.4f}")

    # --- Metric 4: Genomic Relationship Matrix (GRM) Spectrum ---
    print("\n--- Computing Genomic Relationship Matrix (GRM) PCA Scree ---")
    sub_n = min(800, n_animals)
    sub_idx = rng.choice(n_animals, size=sub_n, replace=False)
    grm_scree_dict = {}

    for name, (g_mat, _) in models.items():
        p_vec = np.mean(g_mat, axis=0) / 2.0
        p_vec = np.clip(p_vec, 0.01, 0.99)
        Z = g_mat[sub_idx, :].astype(float) - 2.0 * p_vec
        denom = 2.0 * np.sum(p_vec * (1.0 - p_vec))
        G = np.dot(Z, Z.T) / denom
        eigvals = np.linalg.eigvalsh(G)[::-1]
        eig_pct = (eigvals / np.sum(np.maximum(eigvals, 0))) * 100.0
        grm_scree_dict[name] = eig_pct
        print(f"  {name:36s}: PC1={eig_pct[0]:.2f}%, PC2={eig_pct[1]:.2f}%, PC3={eig_pct[2]:.2f}%")

    # --- Metric 5: Out-of-Sample Genomic Prediction Accuracy ---
    print("\n--- Out-of-Sample Genomic Prediction Accuracy (Calibrated h2 = 0.25) ---")
    # For synthetic populations, simulate polygenic trait with h2=0.25 and 29% environmental noise
    n_qtl = 50
    qtl_idx = rng.choice(n_snps, size=n_qtl, replace=False)
    beta_qtl = rng.normal(0, 1.0, size=n_qtl)

    pred_results = {}
    train_n = int(0.72 * n_animals)
    train_idx = np.arange(train_n)
    test_idx = np.arange(train_n, n_animals)

    for name, (g_mat, _) in models.items():
        if name == "Real Physical Cattle":
            # Use real Florida thermal stress slope (LoadSlope)
            y_raw = real_data.phenotypes["LoadSlope"].values
            # Environmental adjustment
            env_mod = EnvironmentalCovariateModel(continuous_covariates=["Year"], categorical_covariates=["Set", "Project"])
            df_valid = real_data.phenotypes.dropna(subset=["LoadSlope"]).reset_index(drop=True)
            y_adj = env_mod.fit_and_adjust(df_valid, "LoadSlope").adjusted_phenotypes
            v_mask = real_data.phenotypes["LoadSlope"].notna().values
            g_valid = g_mat[v_mask, :]

            # Historical vs Prospective calendar split
            tr_mask = (df_valid["Year"] <= 2018).values
            te_mask = (df_valid["Year"] >= 2019).values

            gblup = AdditiveGBLUP().fit(g_valid[tr_mask], y_adj[tr_mask])
            pred_y = gblup.predict(g_valid[te_mask])
            true_y = y_adj[te_mask]
        else:
            # Calibrated simulated genetic values with true LD
            g_val = np.dot(g_mat[:, qtl_idx % g_mat.shape[1]], beta_qtl)
            g_var = np.var(g_val)
            # h2 = 0.25, R2_env = 0.29 => e_var = g_var * (1 - 0.25) / 0.25
            e_var = g_var * 3.0
            e_noise = rng.normal(0, np.sqrt(e_var), size=n_animals)
            y_sim = g_val + e_noise

            gblup = AdditiveGBLUP().fit(g_mat[train_idx], y_sim[train_idx])
            pred_y = gblup.predict(g_mat[test_idx])
            true_y = y_sim[test_idx]

        r, _ = stats.pearsonr(true_y, pred_y)
        rho, _ = stats.spearmanr(true_y, pred_y)
        rmse = np.sqrt(np.mean((true_y - pred_y) ** 2))
        pred_results[name] = {"r": r, "rho": rho, "rmse": rmse}
        print(f"  {name:36s}: Pearson r={r:.4f}, Spearman rho={rho:.4f}, RMSE={rmse:.4f}")

    # =========================================================================
    # Visualization: Publication 6-Panel Benchmark Figure
    # =========================================================================
    print("\n--- Generating 6-Panel Diagnostic Comparison Figure ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    colors = {
        "Real Physical Cattle": "#1f77b4",             # Deep Blue
        "Method 2: Real Haplotype Resampling": "#2ca02c",# Green
        "Method 1: Coalescent Burn-in": "#9467bd",     # Purple
        "Method 1b: Coalescent + array ascertainment": "#8c564b",  # Brown
        "Naive Beta-Binomial Baseline": "#ff7f0e",     # Orange
    }
    styles = {
        "Real Physical Cattle": ("-", "o"),
        "Method 2: Real Haplotype Resampling": ("-", "^"),
        "Method 1: Coalescent Burn-in": ("--", "s"),
        "Method 1b: Coalescent + array ascertainment": ("-.", "D"),
        "Naive Beta-Binomial Baseline": (":", "x"),
    }

    # Panel 1: Site Frequency Spectrum (MAF)
    ax = axes[0, 0]
    bins_maf = np.linspace(0.0, 0.5, 25)
    for name, maf in maf_dict.items():
        ax.hist(maf, bins=bins_maf, histtype="step", lw=2, color=colors[name], label=name, density=True)
    ax.set_title("A. Site Frequency Spectrum (MAF)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Minor Allele Frequency (MAF)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Observed Heterozygosity (Ho)
    ax = axes[0, 1]
    bins_ho = np.linspace(0.10, 0.55, 30)
    for name, ho in ho_dict.items():
        ax.hist(ho, bins=bins_ho, histtype="step", lw=2, color=colors[name], label=f"{name} (μ={np.mean(ho):.2f})", density=True)
    ax.set_title("B. Individual Observed Heterozygosity ($H_o$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Observed Heterozygosity ($H_o$)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Linkage Disequilibrium Decay (r^2 vs Distance)
    ax = axes[0, 2]
    bin_midpoints = [0.25, 0.75, 1.5, 3.5, 7.5, 12.5]
    for name, binned in ld_binned_dict.items():
        ls, m = styles[name]
        ax.plot(bin_midpoints, binned.values, linestyle=ls, marker=m, lw=2, color=colors[name], label=name)
    ax.set_title("C. Linkage Disequilibrium ($r^2$) Decay on BTA 1", fontsize=11, fontweight="bold")
    ax.set_xlabel("Genomic Distance (Mb)", fontsize=9)
    ax.set_ylabel("Mean $r^2$", fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: GRM PCA Scree Plot
    ax = axes[1, 0]
    pcs = np.arange(1, 16)
    for name, scree in grm_scree_dict.items():
        ls, m = styles[name]
        ax.plot(pcs, scree[:15], linestyle=ls, marker=m, lw=2, color=colors[name], label=f"{name} (PC1={scree[0]:.1f}%)")
    ax.set_title("D. Genomic Relationship Matrix PCA Scree", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Variance Explained (%)", fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Genomic Prediction Accuracy Comparison (Bar Chart)
    ax = axes[1, 1]
    m_names = list(pred_results.keys())
    short = {"Real Physical Cattle": "Real Cattle\n(Phys)", "Method 2: Real Haplotype Resampling": "Method 2\n(RealHap)",
             "Method 1: Coalescent Burn-in": "Method 1\n(Coalescent)",
             "Method 1b: Coalescent + array ascertainment": "Method 1b\n(Coal+Array)",
             "Naive Beta-Binomial Baseline": "Naive Base\n(BetaBin)"}
    short_names = [short[m] for m in m_names]
    r_vals = [pred_results[m]["r"] for m in m_names]
    bar_cols = [colors[m] for m in m_names]
    bars = ax.bar(short_names, r_vals, color=bar_cols, alpha=0.85, edgecolor="black", width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("E. Genomic Prediction Accuracy (Pearson $r$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample Accuracy ($r$)", fontsize=9)
    ax.set_ylim(0, max(r_vals) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 6: Summary Radar / Scorecard Comparison
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for m in m_names:
        table_data.append([
            m[:18],
            f"{ld_binned_dict[m].iloc[0]:.3f}",
            f"{np.mean(ho_dict[m]):.3f}",
            f"{grm_scree_dict[m][0]:.1f}%",
            f"{pred_results[m]['r']:.3f}"
        ])
    col_labels = ["Method", "LD r² (<0.5M)", "Mean Ho", "GRM PC1 %", "Pred r"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)
    ax.set_title("F. Architecture Fidelity Scorecard", fontsize=11, fontweight="bold", pad=15)

    plot_path = out_dir / "simulation_methods_benchmark.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved benchmark plot: {plot_path}")


    # Export metrics CSV
    rows = []
    for m in m_names:
        rows.append({
            "Method": m,
            "MAF_Mean": round(float(np.mean(maf_dict[m])), 4),
            "MAF_Std": round(float(np.std(maf_dict[m])), 4),
            "MAF_KS_Stat": round(float(ks_results[m][0]), 4),
            "Ho_Mean": round(float(np.mean(ho_dict[m])), 4),
            "Ho_Std": round(float(np.std(ho_dict[m])), 4),
            "LD_r2_lt_0.5Mb": round(float(ld_binned_dict[m].iloc[0]), 4),
            "LD_r2_5_10Mb": round(float(ld_binned_dict[m].iloc[4]), 4),
            "GRM_PC1_Pct": round(float(grm_scree_dict[m][0]), 2),
            "GRM_PC2_Pct": round(float(grm_scree_dict[m][1]), 2),
            "Prediction_Pearson_r": round(float(pred_results[m]["r"]), 4),
            "Prediction_Spearman_rho": round(float(pred_results[m]["rho"]), 4),
            "Prediction_RMSE": round(float(pred_results[m]["rmse"]), 4),
        })
    df_metrics = pd.DataFrame(rows)
    metrics_csv = out_dir / "simulation_methods_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"  Saved metrics CSV: {metrics_csv}")
    print("\nFinal Comparative Metrics Table:")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    run_methods_benchmark()

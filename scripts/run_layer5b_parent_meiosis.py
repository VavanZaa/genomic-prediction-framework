"""Run Layer 5B-P: Real-Parent Meiotic Simulation Benchmark.

Evaluates meiotic gamete simulation and transmission dynamics on authentic 777K BovineHD
phased parental haplotypes from 171 sire-dam breeding pairs across 7 Spanish autochthonous beef breeds:
- Asturiana de los Valles (n = 25 pairs)
- Avileña-Negra Ibérica (n = 24 pairs)
- Bruna dels Pirineus (n = 25 pairs)
- Morucha (n = 25 pairs)
- Pirenaica (n = 24 pairs)
- Retinta (n = 24 pairs)
- Rubia Gallega (n = 24 pairs)

Scope (5B-P vs 5B-T):
- 5B-P simulates gametes from the phased parents and reports expected vs simulated offspring
  heterozygosity, parental relationship and pair-specific segregation variance. The crossover count
  is a model input (1 cM/Mb, assumed), not a measured result.
- Offspring validation (5B-T) is in scripts/run_layer5b_trio_recovery.py. The calves are not missing:
  in BEAGLE trio-phased files the first haplotype of each parent is the transmitted one, so the calf
  is their sum (BEAGLE 3.3 manual, "Phased trio data"). Parent phase here is child-informed.

Outputs:
- output/real_layer5/layer5b_parent_meiosis_scorecard.csv
- output/real_layer5/layer5b_parent_meiosis.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def run_layer5b_parent_benchmark():
    print("=" * 95)
    print("LAYER 5B-P: REAL-PARENT MEIOTIC SIMULATION BENCHMARK (777K BOVINE PAIR BENCHMARK)")
    print("=" * 95)

    out_dir = BASE_DIR / "output" / "real_layer5"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Define Breeds and Chromosomes
    data_dir = BASE_DIR / "data" / "real_cattle" / "spanish_trios"
    breeds_info = [
        ("Asturiana de los Valles", "asturiana", 1, 25),
        ("Avileña-Negra Ibérica", "avilena", 2, 24),
        ("Bruna dels Pirineus", "bruna", 3, 25),
        ("Morucha", "morucha", 4, 25),
        ("Pirenaica", "pirenaica", 5, 24),
        ("Retinta", "retinta", 6, 24),
        ("Rubia Gallega", "rubia", 7, 24)
    ]
    # Representative chromosomes spanning size range: Chr 1 (158 Mb), Chr 5 (121 Mb), Chr 14 (84 Mb), Chr 20 (72 Mb), Chr 29 (51 Mb)
    chromosomes = [1, 5, 14, 20, 29]
    chr_lengths_mb = {1: 158.3, 5: 121.2, 14: 84.4, 20: 72.0, 29: 51.5}

    print("\n[1/4] Ingesting 171 Authentic Phased Bovine Breeding Pairs across 7 Spanish Breeds...")
    total_pairs = sum(n for _, _, _, n in breeds_info)
    print(f"  Target: {total_pairs} sire-dam pairs (342 phased parents) at 777K BovineHD resolution.")
    print(f"  Chromosomes evaluated: {chromosomes} (spanning {sum(chr_lengths_mb.values()):.1f} Mb).")

    # Ingest phased haplotypes per breed and chromosome
    # Structure: breed_data[breed_name][chr] = (sire_h1, sire_h2, dam_h1, dam_h2) for each pair
    breed_records = []
    pair_metrics_list = []

    t0 = time.time()
    for breed_title, b_slug, b_id, expected_pairs in breeds_info:
        gz_path = data_dir / f"{b_slug}.tar.gz"
        print(f"  Loading {breed_title:<25} ({expected_pairs} pairs)...", end="", flush=True)
        
        with tarfile.open(gz_path, "r:gz") as tar:
            chr_data = {}
            for chrom in chromosomes:
                member_name = f"Beagle.{b_id}.cromo{chrom}.pre_phase.bgl.phased"
                f = tar.extractfile(member_name)
                # Skip header lines 0 and 1
                f.readline()
                f.readline()
                
                # Read markers with step=8 for ultra-fast dense processing (~15,000 markers total)
                alleles_rows = []
                for idx, line in enumerate(f):
                    if idx % 8 == 0:
                        parts = line.decode("utf-8").split()
                        # parts[0] = 'M', parts[1] = marker_name, parts[2:] = alleles ('A'/'B' or 'A'/'G' etc.)
                        alleles_rows.append(parts[2:])
                
                # Convert alleles to binary 0/1 indicator (relative to major allele)
                raw_mat = np.array(alleles_rows)  # M x (2 * n_parents)
                m_count, n_cols = raw_mat.shape
                
                # Major allele coding per marker
                bin_mat = np.zeros((m_count, n_cols), dtype=np.int8)
                for m in range(m_count):
                    row = raw_mat[m, :]
                    major = row[0]
                    bin_mat[m, :] = (row != major).astype(np.int8)
                
                chr_data[chrom] = bin_mat

        # Process each sire-dam pair for this breed
        n_pairs_found = n_cols // 4  # 4 haplotypes per pair: 2 sire + 2 dam
        print(f" Ingested {n_pairs_found} pairs, {sum(chr_data[c].shape[0] for c in chromosomes)} markers in {time.time() - t0:.2f}s")
        t0 = time.time()

        for p_idx in range(n_pairs_found):
            # Haplotype indices: sire has cols [4*p, 4*p+1], dam has cols [4*p+2, 4*p+3]
            s_h1_list = [chr_data[c][:, 4 * p_idx] for c in chromosomes]
            s_h2_list = [chr_data[c][:, 4 * p_idx + 1] for c in chromosomes]
            d_h1_list = [chr_data[c][:, 4 * p_idx + 2] for c in chromosomes]
            d_h2_list = [chr_data[c][:, 4 * p_idx + 3] for c in chromosomes]

            # Genome-wide concatenated genotypes
            s_geno = np.concatenate(s_h1_list) + np.concatenate(s_h2_list)
            d_geno = np.concatenate(d_h1_list) + np.concatenate(d_h2_list)
            m_total = len(s_geno)

            # 1. Expected Heterozygosity for this mating pair
            # P(het | s, d) = p_s(1 - p_d) + p_d(1 - p_s) where p_s = (h1 + h2)/2
            p_s = s_geno / 2.0
            p_d = d_geno / 2.0
            expected_het = float(np.mean(p_s * (1.0 - p_d) + p_d * (1.0 - p_s)))

            # 2. Parental Relationship & Expected Offbreeding/Inbreeding
            # Allele frequency in this breed
            all_breed_alleles = np.concatenate([chr_data[c] for c in chromosomes], axis=0)
            p_pop = np.clip(np.mean(all_breed_alleles, axis=1), 0.01, 0.99)
            denom_grm = 2.0 * np.sum(p_pop * (1.0 - p_pop)) + 1e-6
            z_s = s_geno - 2.0 * p_pop
            z_d = d_geno - 2.0 * p_pop
            parental_relationship = float((z_s @ z_d) / denom_grm)
            expected_inbreeding = float(0.5 * parental_relationship)

            # 3. Simulate Meiotic Gametogenesis across K=50 offspring
            k_reps = 50
            sim_offspring = np.zeros((k_reps, m_total), dtype=np.int8)
            crossover_counts = []

            for k in range(k_reps):
                off_chr_list = []
                for c in chromosomes:
                    m_c = chr_data[c].shape[0]
                    # Crossovers per gamete ~ Poisson(genetic length), 1 cM/Mb assumed. (The earlier
                    # 1 + Mb/100 added one crossover per chromosome, confusing chiasmata with gamete crossovers.)
                    exp_xo = chr_lengths_mb[c] / 100.0
                    n_xo_s = rng.poisson(exp_xo)
                    n_xo_d = rng.poisson(exp_xo)
                    crossover_counts.extend([n_xo_s, n_xo_d])

                    # Sire gamete
                    s1 = chr_data[c][:, 4 * p_idx]
                    s2 = chr_data[c][:, 4 * p_idx + 1]
                    s_strand = rng.integers(0, 2)
                    gam_s = (s1 if s_strand == 0 else s2).copy()
                    if n_xo_s > 0 and m_c > 1:
                        for pt in np.sort(rng.integers(1, m_c, size=n_xo_s)):
                            s_strand = 1 - s_strand
                            src = s1 if s_strand == 0 else s2
                            gam_s[pt:] = src[pt:]

                    # Dam gamete
                    d1 = chr_data[c][:, 4 * p_idx + 2]
                    d2 = chr_data[c][:, 4 * p_idx + 3]
                    d_strand = rng.integers(0, 2)
                    gam_d = (d1 if d_strand == 0 else d2).copy()
                    if n_xo_d > 0 and m_c > 1:
                        for pt in np.sort(rng.integers(1, m_c, size=n_xo_d)):
                            d_strand = 1 - d_strand
                            src = d1 if d_strand == 0 else d2
                            gam_d[pt:] = src[pt:]

                    off_chr_list.append(gam_s + gam_d)

                sim_offspring[k, :] = np.concatenate(off_chr_list)

            # Realized Offspring Properties
            realized_het = float(np.mean(sim_offspring == 1))
            # Pair-specific segregation variance: variance across offspring genome scores
            # Using standardized genetic merit vector w
            w_score = (p_pop - 0.5)
            offspring_scores = (sim_offspring - 2.0 * p_pop) @ w_score / np.sqrt(m_total)
            pair_seg_variance = float(np.var(offspring_scores))
            mean_crossovers_per_chr = float(np.mean(crossover_counts))

            pair_metrics_list.append({
                "Breed": breed_title,
                "Pair_ID": f"{b_slug}_pair_{p_idx+1:02d}",
                "Expected_Heterozygosity": expected_het,
                "Simulated_Mean_Heterozygosity": realized_het,
                "Heterozygosity_Discrepancy": realized_het - expected_het,
                "Parental_Kinship_GRM": parental_relationship,
                "Expected_Inbreeding_Fo": expected_inbreeding,
                "Pair_Segregation_Variance": pair_seg_variance,
                "Mean_Crossovers_Per_Chr": mean_crossovers_per_chr
            })

    df_pairs = pd.DataFrame(pair_metrics_list)
    scorecard_path = out_dir / "layer5b_parent_meiosis_scorecard.csv"
    df_pairs.to_csv(scorecard_path, index=False)
    print(f"\n[2/4] Saved 171-pair simulation scorecard to {scorecard_path}")

    # Summary by Breed
    breed_summary = df_pairs.groupby("Breed").agg({
        "Pair_ID": "count",
        "Expected_Heterozygosity": "mean",
        "Simulated_Mean_Heterozygosity": "mean",
        "Heterozygosity_Discrepancy": "mean",
        "Parental_Kinship_GRM": "mean",
        "Expected_Inbreeding_Fo": "mean",
        "Pair_Segregation_Variance": "mean",
        "Mean_Crossovers_Per_Chr": "mean"
    }).reset_index()

    print("\n" + "=" * 125)
    print("LAYER 5B-P: REAL-PARENT MEIOTIC SIMULATION BENCHMARK SCORECARD (777K BOVINE BREEDS)")
    print("=" * 125)
    print(f"{'Breed':<28} | {'Pairs':<5} | {'E[Het]':<8} | {'Sim[Het]':<8} | {'Delta Het':<10} | {'Kinship':<8} | {'E[Fo]':<8} | {'Var(Go|s,d)':<12} | {'XO/Chr':<7}")
    print("-" * 125)
    for _, r in breed_summary.iterrows():
        print(f"{r['Breed']:<28} | {r['Pair_ID']:<5} | {r['Expected_Heterozygosity']:<8.4f} | {r['Simulated_Mean_Heterozygosity']:<8.4f} | {r['Heterozygosity_Discrepancy']:<+10.5f} | {r['Parental_Kinship_GRM']:<+8.4f} | {r['Expected_Inbreeding_Fo']:<+8.4f} | {r['Pair_Segregation_Variance']:<12.4f} | {r['Mean_Crossovers_Per_Chr']:<7.2f}")
    print("-" * 125)
    overall_pairs = len(df_pairs)
    overall_exp_het = df_pairs["Expected_Heterozygosity"].mean()
    overall_sim_het = df_pairs["Simulated_Mean_Heterozygosity"].mean()
    overall_delta = df_pairs["Heterozygosity_Discrepancy"].mean()
    overall_kinship = df_pairs["Parental_Kinship_GRM"].mean()
    overall_inb = df_pairs["Expected_Inbreeding_Fo"].mean()
    overall_var = df_pairs["Pair_Segregation_Variance"].mean()
    overall_xo = df_pairs["Mean_Crossovers_Per_Chr"].mean()
    print(f"{'ALL 7 BOVINE BREEDS (TOTAL)':<28} | {overall_pairs:<5} | {overall_exp_het:<8.4f} | {overall_sim_het:<8.4f} | {overall_delta:<+10.5f} | {overall_kinship:<+8.4f} | {overall_inb:<+8.4f} | {overall_var:<12.4f} | {overall_xo:<7.2f}")
    print("=" * 125)

    het_r2 = float(np.corrcoef(df_pairs["Expected_Heterozygosity"], df_pairs["Simulated_Mean_Heterozygosity"])[0, 1] ** 2)
    print("\nSummary (simulation checks, not empirical validation):")
    print(f"  1. Simulated vs expected offspring heterozygosity: mean difference {overall_delta*100:+.3f} points, R^2 = {het_r2:.4f} across pairs.")
    print(f"  2. Parental genomic relationship: mean G_sd = {overall_kinship:+.4f} (implied offspring F = {overall_inb:+.4f}).")
    print(f"  3. Crossovers per chromosome: {overall_xo:.2f} (a model input at 1 cM/Mb, not a finding).")
    print(f"  4. Pair segregation variance of an arbitrary allele-frequency score: range {df_pairs['Pair_Segregation_Variance'].min():.3f} - {df_pairs['Pair_Segregation_Variance'].max():.3f}.")
    print("  Real-calf validation: scripts/run_layer5b_trio_recovery.py")

    # 4. Publication Plot (4 Panels)
    print("\n[3/4] Generating 4-Panel Publication Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Expected vs Realized Heterozygosity
    ax = axes[0, 0]
    breeds = df_pairs["Breed"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(breeds)))
    for b_name, col in zip(breeds, colors):
        sub = df_pairs[df_pairs["Breed"] == b_name]
        ax.scatter(sub["Expected_Heterozygosity"], sub["Simulated_Mean_Heterozygosity"],
                   color=col, alpha=0.75, s=35, label=b_name.split()[0])
    x_line = np.linspace(df_pairs["Expected_Heterozygosity"].min()*0.95, df_pairs["Expected_Heterozygosity"].max()*1.05, 50)
    ax.plot(x_line, x_line, "k--", lw=1.8, label="Identity line ($y = x$)")
    ax.set_title(f"A. Offspring Heterozygosity: Expected vs Simulated ($R^2 = {het_r2:.4f}$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Expected Pair Heterozygosity ($H_{e,sd}$)", fontsize=10)
    ax.set_ylabel("Simulated Offspring Heterozygosity ($\\bar{H}_o$)", fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel B: Parental Kinship & Expected Inbreeding Distribution
    ax = axes[0, 1]
    ax.hist(df_pairs["Parental_Kinship_GRM"], bins=20, color="#1f77b4", alpha=0.65, edgecolor="black", label="Parental Kinship $G_{s,d}$")
    ax.axvline(overall_kinship, color="blue", linestyle="--", lw=2, label=f"Mean $G_{{s,d}} = {overall_kinship:+.3f}$")
    ax.axvline(0.0, color="gray", linestyle=":", lw=1.5)
    ax.set_title("B. Parental Genomic Relationship Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("VanRaden GRM Relationship Coefficient ($G_{s,d}$)", fontsize=10)
    ax.set_ylabel("Number of Breeding Pairs", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Pair-Specific Segregation Variance by Breed
    ax = axes[1, 0]
    box_data = [df_pairs[df_pairs["Breed"] == b]["Pair_Segregation_Variance"].values for b in breeds]
    short_names = [b.split()[0] for b in breeds]
    bplot = ax.boxplot(box_data, patch_artist=True, tick_labels=short_names)
    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_title("C. Pair-Specific Segregation Variance ${\\rm Var}(G_o \\mid G_s, G_d)$ by Breed", fontsize=11, fontweight="bold")
    ax.set_xlabel("Spanish Autochthonous Beef Breed", fontsize=10)
    ax.set_ylabel("Offspring Segregation Variance", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Crossover Distribution per Chromosome
    ax = axes[1, 1]
    ax.hist(df_pairs["Mean_Crossovers_Per_Chr"], bins=15, color="#2ca02c", alpha=0.65, edgecolor="black", label="Mean Crossovers / Chr")
    ax.axvline(overall_xo, color="darkgreen", linestyle="--", lw=2, label=f"Mean = {overall_xo:.2f} per chromosome")
    ax.set_title("D. Simulated Crossovers per Chromosome (model input, 1 cM/Mb)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Simulated Crossovers per Chromosome per Gamete", fontsize=10)
    ax.set_ylabel("Number of Breeding Pairs", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "layer5b_parent_meiosis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to {fig_path}")


if __name__ == "__main__":
    run_layer5b_parent_benchmark()

"""Layer 3B Master Benchmark: Breed-of-Origin Alleles (BOA) Multi-Kernel Genomic Prediction.

Evaluates 5 nested architectures on the University of Florida Angus-Brahman Multibreed Herd:
- M0: Global Ancestry Baseline (y ~ q_Angus)
- M1: Standard SNP GBLUP (y ~ G)
- M2: Pure Breed-of-Origin GBLUP (y ~ B)
- M3: Combined SNP + BOA GBLUP (y ~ w_G * G + w_B * B, with data-driven CV weights)
- M4: SNP + BOA + Interaction (y ~ w_G * G + w_B * B + w_Omega * Omega, with data-driven CV weights)

Enforces:
1. Canonical zero-leakage prospective split: Train (2016-2018), Test (2019, 2021, 2022 - no 2020).
2. Data-driven kernel weights optimized strictly via K-fold cross-validation on training data.
3. Paired bootstrap significance testing reporting both one-sided and two-sided p-values.
4. Harmonized BOA encoding (0 = Angus/Angus, 2 = Brahman/Brahman).
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.florida_boa_loader import FloridaBOALoader
from src.real_data.canonical_split import CanonicalDataSplitter
from src.prediction.boa_models import BOAComparativePredictor, BOAValidationStats


def run_layer3b_benchmark():
    print("=" * 85)
    print("LAYER 3B: BREED-OF-ORIGIN ALLELE (BOA) PROSPECTIVE GENOMIC BENCHMARK")
    print("  Physical Dataset: Florida Angus-Brahman Multibreed Herd (20,415 Autosomal Markers)")
    print("=" * 85)

    out_dir = BASE_DIR / "output" / "real_layer3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dual Dataset (20,415 common physical markers across BTA 1-29)
    print("\n[1] Loading Physical Dual SNP (X) + BOA (O) Dataset (step=6)...")
    data = FloridaBOALoader.load_dual_dataset(marker_step=6, cache=True)
    print(f"  Animals: {data.num_animals}, Physical Markers: {data.num_markers}")

    df_pheno = data.phenotypes.copy()
    id_to_idx = {aid: idx for idx, aid in enumerate(data.animal_ids)}

    # Empirical non-Gaussian Angus ancestry quantiles
    q_angus_all = df_pheno["AngusGBC"].values
    valid_q = q_angus_all[~np.isnan(q_angus_all)]
    quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    q_vals = np.quantile(valid_q, quantiles)
    print("\n[Empirical Ancestry Distribution (AngusGBC)]:")
    print("  Quantiles:", dict(zip([f"{int(q*100)}%" for q in quantiles], np.round(q_vals, 4))))

    # 2. Setup Traits and Models
    traits = ["LoadSlope", "AvgShortLength"]
    trait_labels = {
        "LoadSlope": "Thermal Stress Slope (LoadSlope)",
        "AvgShortLength": "Hair Coat Length (AvgShortLength, mm)"
    }

    model_keys = ["M0_Ancestry", "M1_SNP", "M2_BOA", "M3_SNP_BOA", "M4_SNP_BOA_INTER"]
    model_display_names = {
        "M0_Ancestry": "M0: Global Ancestry Baseline (y ~ q_Angus)",
        "M1_SNP": "M1: Standard SNP GBLUP (y ~ G)",
        "M2_BOA": "M2: Pure Breed-of-Origin (y ~ B)",
        "M3_SNP_BOA": "M3: Combined SNP + BOA (Data-Driven CV)",
        "M4_SNP_BOA_INTER": "M4: SNP + BOA + Interaction (Data-Driven CV)"
    }

    results_table_rows = []
    bootstrap_comparison_rows = []
    trait_predictions = {}

    for trait in traits:
        print(f"\n" + "=" * 80)
        print(f"EVALUATING TRAIT: {trait_labels[trait]}")
        print("=" * 80)

        # Canonical Zero-Leakage Split (Train <= 2018, Test >= 2019 [2019, 2021, 2022])
        split_data = CanonicalDataSplitter.get_prospective_split(
            trait_col=trait,
            pheno_df=df_pheno,
            valid_animal_ids=data.animal_ids,
            require_joint_traits=True
        )

        tr_idx = [id_to_idx[aid] for aid in split_data.train_ids]
        te_idx = [id_to_idx[aid] for aid in split_data.test_ids]

        X_tr, X_te = data.genotypes_snp[tr_idx], data.genotypes_snp[te_idx]
        O_tr, O_te = data.alleles_boa[tr_idx], data.alleles_boa[te_idx]
        q_tr, q_te = split_data.q_train, split_data.q_test
        y_tr, y_te = split_data.y_train_adj, split_data.y_test_adj

        n_train = len(split_data.train_ids)
        n_test = len(split_data.test_ids)

        print(f"  Training Cohort (Birth Years 2016-2018): {n_train} animals")
        print(f"  Prospective Test Cohort (Birth Years 2019, 2021-2022): {n_test} animals")
        print(f"  Training Environmental R^2: {split_data.r2_train_env:.3f} (fitted strictly on train)")

        # Fit 5 models
        preds = {}
        fitted_weights_info = {}

        for m_key in model_keys:
            use_cv = (m_key in ("M3_SNP_BOA", "M4_SNP_BOA_INTER"))
            # Ridge penalty chosen by training-only CV (the earlier fixed 0.40 implied h2 ~ 0.71 and let
            # predictions for the prospective cohort blow up once test genotypes use training scaling)
            predictor = BOAComparativePredictor(ridge_param=None)
            predictor.fit(
                X_train=X_tr,
                O_train=O_tr,
                q_train=q_tr,
                y_train=y_tr,
                model_type=m_key,
                optimize_weights_cv=use_cv
            )
            y_pred_test = predictor.predict(X_te, O_te, q_te)
            preds[m_key] = y_pred_test
            fitted_weights_info[m_key] = predictor.kernel_weights

            r_val, _ = stats.pearsonr(y_te, y_pred_test)
            rho_val, _ = stats.spearmanr(y_te, y_pred_test)
            rmse_val = np.sqrt(np.mean((y_te - y_pred_test) ** 2))

            weights_str = f" [Weights: {tuple(round(w, 2) for w in predictor.kernel_weights)}]" if use_cv else ""
            if predictor.ridge_param_ is not None:
                weights_str += f" [lambda {predictor.ridge_param_:g}]"
            print(f"  {model_display_names[m_key]:50s}{weights_str} -> r={r_val:.4f}, rho={rho_val:.4f}, RMSE={rmse_val:.3f}")

            results_table_rows.append({
                "Trait": trait,
                "Model_Key": m_key,
                "Model_Name": model_display_names[m_key],
                "Pearson_r": round(float(r_val), 4),
                "Spearman_rho": round(float(rho_val), 4),
                "RMSE": round(float(rmse_val), 4),
                "Kernel_Weights": str(tuple(round(w, 2) for w in predictor.kernel_weights)),
                "Ridge_Lambda_CV": predictor.ridge_param_,
                "N_Train": n_train,
                "N_Test": n_test
            })

        trait_predictions[trait] = (y_te, preds)

        # Rigorous Paired Bootstrap Resampling (B = 1000 draws on test set)
        print("\n  [Paired Bootstrap Statistical Comparisons (B = 1000 iterations)]:")
        pred_m0 = preds["M0_Ancestry"]
        pred_m1 = preds["M1_SNP"]

        # 1. Compare models against M0 (Global Ancestry Baseline)
        for m_key in ["M1_SNP", "M2_BOA", "M3_SNP_BOA", "M4_SNP_BOA_INTER"]:
            boot_res = BOAValidationStats.paired_bootstrap_test(
                y_true=y_te,
                y_pred_model=preds[m_key],
                y_pred_baseline=pred_m0,
                n_bootstraps=1000,
                random_seed=42
            )
            print(f"    {m_key:18s} vs M0: Delta_r = {boot_res.delta_r:+.4f} [95% CI: {boot_res.ci_lower:+.4f}, {boot_res.ci_upper:+.4f}], p_1sided = {boot_res.p_value_one_sided:.4f}, p_2sided = {boot_res.p_value_two_sided:.4f}")
            bootstrap_comparison_rows.append({
                "Trait": trait,
                "Comparison": f"{m_key} vs M0_Ancestry",
                "r_Model": round(boot_res.r_model, 4),
                "r_Baseline": round(boot_res.r_baseline, 4),
                "Delta_r": round(boot_res.delta_r, 4),
                "CI_Lower": round(boot_res.ci_lower, 4),
                "CI_Upper": round(boot_res.ci_upper, 4),
                "p_one_sided": round(boot_res.p_value_one_sided, 4),
                "p_two_sided": round(boot_res.p_value_two_sided, 4),
                "Significant_2sided": boot_res.is_significant
            })

        # 2. Compare M3 (Combined SNP + BOA) vs M1 (Standard SNP GBLUP)
        boot_m3_vs_m1 = BOAValidationStats.paired_bootstrap_test(
            y_true=y_te,
            y_pred_model=preds["M3_SNP_BOA"],
            y_pred_baseline=pred_m1,
            n_bootstraps=1000,
            random_seed=42
        )
        print(f"    {'M3_SNP_BOA vs M1':18s}     : Delta_r = {boot_m3_vs_m1.delta_r:+.4f} [95% CI: {boot_m3_vs_m1.ci_lower:+.4f}, {boot_m3_vs_m1.ci_upper:+.4f}], p_1sided = {boot_m3_vs_m1.p_value_one_sided:.4f}, p_2sided = {boot_m3_vs_m1.p_value_two_sided:.4f}")
        bootstrap_comparison_rows.append({
            "Trait": trait,
            "Comparison": "M3_SNP_BOA vs M1_SNP",
            "r_Model": round(boot_m3_vs_m1.r_model, 4),
            "r_Baseline": round(boot_m3_vs_m1.r_baseline, 4),
            "Delta_r": round(boot_m3_vs_m1.delta_r, 4),
            "CI_Lower": round(boot_m3_vs_m1.ci_lower, 4),
            "CI_Upper": round(boot_m3_vs_m1.ci_upper, 4),
            "p_one_sided": round(boot_m3_vs_m1.p_value_one_sided, 4),
            "p_two_sided": round(boot_m3_vs_m1.p_value_two_sided, 4),
            "Significant_2sided": boot_m3_vs_m1.is_significant
        })

        # 3. Compare M4 (SNP + BOA + Interact) vs M1 (Standard SNP GBLUP)
        boot_m4_vs_m1 = BOAValidationStats.paired_bootstrap_test(
            y_true=y_te,
            y_pred_model=preds["M4_SNP_BOA_INTER"],
            y_pred_baseline=pred_m1,
            n_bootstraps=1000,
            random_seed=42
        )
        print(f"    {'M4_INTER vs M1':18s}     : Delta_r = {boot_m4_vs_m1.delta_r:+.4f} [95% CI: {boot_m4_vs_m1.ci_lower:+.4f}, {boot_m4_vs_m1.ci_upper:+.4f}], p_1sided = {boot_m4_vs_m1.p_value_one_sided:.4f}, p_2sided = {boot_m4_vs_m1.p_value_two_sided:.4f}")
        bootstrap_comparison_rows.append({
            "Trait": trait,
            "Comparison": "M4_INTER vs M1_SNP",
            "r_Model": round(boot_m4_vs_m1.r_model, 4),
            "r_Baseline": round(boot_m4_vs_m1.r_baseline, 4),
            "Delta_r": round(boot_m4_vs_m1.delta_r, 4),
            "CI_Lower": round(boot_m4_vs_m1.ci_lower, 4),
            "CI_Upper": round(boot_m4_vs_m1.ci_upper, 4),
            "p_one_sided": round(boot_m4_vs_m1.p_value_one_sided, 4),
            "p_two_sided": round(boot_m4_vs_m1.p_value_two_sided, 4),
            "Significant_2sided": boot_m4_vs_m1.is_significant
        })

    # Export Results Tables
    df_results = pd.DataFrame(results_table_rows)
    df_boot = pd.DataFrame(bootstrap_comparison_rows)

    benchmark_csv = out_dir / "layer3b_boa_model_benchmark.csv"
    bootstrap_csv = out_dir / "layer3b_boa_bootstrap_significance.csv"
    df_results.to_csv(benchmark_csv, index=False)
    df_boot.to_csv(bootstrap_csv, index=False)
    print(f"\n[SAVED] Benchmark Table: {benchmark_csv}")
    print(f"[SAVED] Bootstrap Table: {bootstrap_csv}")

    # =========================================================================
    # Diagnostic 4-Panel Visualization
    # =========================================================================
    print("\n[3] Generating Layer 3B Diagnostic Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    colors = ["#7f7f7f", "#1f77b4", "#e377c2", "#2ca02c", "#d62728"]
    model_short_names = ["M0 (Ancestry)", "M1 (SNP)", "M2 (BOA)", "M3 (SNP+BOA)", "M4 (Interact)"]

    # Panel A: Thermal Stress Slope (LoadSlope) Model Accuracies
    ax = axes[0, 0]
    df_ls = df_results[df_results["Trait"] == "LoadSlope"]
    bars = ax.bar(model_short_names, df_ls["Pearson_r"], color=colors, alpha=0.85, edgecolor="black", width=0.55)
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.005 if h >= 0 else -0.015
        ax.text(b.get_x() + b.get_width() / 2, h + offset, f"{h:.3f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("A. Thermal Stress Slope (LoadSlope) Accuracy (20k SNPs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample Pearson $r$", fontsize=10)
    y_max = max(0.12, max(df_ls["Pearson_r"]) * 1.3)
    y_min = min(-0.05, min(df_ls["Pearson_r"]) * 1.3)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Hair Coat Length (AvgShortLength) Model Accuracies
    ax = axes[0, 1]
    df_hl = df_results[df_results["Trait"] == "AvgShortLength"]
    bars2 = ax.bar(model_short_names, df_hl["Pearson_r"], color=colors, alpha=0.85, edgecolor="black", width=0.55)
    for b in bars2:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.005 if h >= 0 else -0.015
        ax.text(b.get_x() + b.get_width() / 2, h + offset, f"{h:.3f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("B. Hair Coat Length (AvgShortLength) Accuracy (20k SNPs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample Pearson $r$", fontsize=10)
    y_max2 = max(0.14, max(df_hl["Pearson_r"]) * 1.3)
    y_min2 = min(-0.05, min(df_hl["Pearson_r"]) * 1.3)
    ax.set_ylim(y_min2, y_max2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Paired Bootstrap Delta_r Distributions (LoadSlope)
    ax = axes[1, 0]
    y_te_ls, preds_ls = trait_predictions["LoadSlope"]
    rng = np.random.default_rng(42)
    n = len(y_te_ls)
    b_m3_m1 = np.zeros(1000)
    b_m4_m1 = np.zeros(1000)
    for b in range(1000):
        idx = rng.integers(0, n, size=n)
        r_m1_b, _ = stats.pearsonr(y_te_ls[idx], preds_ls["M1_SNP"][idx])
        r_m3_b, _ = stats.pearsonr(y_te_ls[idx], preds_ls["M3_SNP_BOA"][idx])
        r_m4_b, _ = stats.pearsonr(y_te_ls[idx], preds_ls["M4_SNP_BOA_INTER"][idx])
        b_m3_m1[b] = r_m3_b - r_m1_b
        b_m4_m1[b] = r_m4_b - r_m1_b

    ax.hist(b_m3_m1, bins=35, color="#2ca02c", alpha=0.6, edgecolor="black", label=f"M3 - M1 (Mean={np.mean(b_m3_m1):+.3f})")
    ax.hist(b_m4_m1, bins=35, color="#d62728", alpha=0.5, edgecolor="black", label=f"M4 - M1 (Mean={np.mean(b_m4_m1):+.3f})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label=r"Null ($\Delta r = 0$)")
    ax.set_title(r"C. Paired Bootstrap $\Delta r$ Distribution (vs SNP GBLUP)", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Out-of-Sample $\Delta r$", fontsize=10)
    ax.set_ylabel("Bootstrap Frequency (B=1000)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel D: Empirical Ancestry vs BOA Brahman Count Calibration
    ax = axes[1, 1]
    brahman_gbc = df_pheno["BrahmanGBC"].values
    mean_boa_brahman = np.mean(data.alleles_boa, axis=1) / 2.0  # scale to [0, 1]
    valid_pts = ~np.isnan(brahman_gbc)
    ax.scatter(brahman_gbc[valid_pts], mean_boa_brahman[valid_pts], color="#9467bd", alpha=0.35, s=16, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Identity Line (y = x)")
    r_align, _ = stats.pearsonr(brahman_gbc[valid_pts], mean_boa_brahman[valid_pts])
    ax.set_title(f"D. BOA Allele Count vs BrahmanGBC Calibration (r = {r_align:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("True Brahman Genomic Breed Composition ($q_{Brahman}$)", fontsize=10)
    ax.set_ylabel("Inferred BOA Brahman Fraction ($O_{Brahman} / 2$)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Layer 3B: Breed-of-Origin Alleles (BOA) Multi-Kernel Genomic Prediction\nUniversity of Florida Multibreed Beef Herd (Zero Data Leakage, 20k Autosomal Markers)", fontsize=13, fontweight="bold", y=0.99)

    plot_path = out_dir / "layer3b_boa_prediction_benchmark.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Diagnostic Plot: {plot_path}")


    print("\n" + "=" * 85)
    print("LAYER 3B BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    run_layer3b_benchmark()

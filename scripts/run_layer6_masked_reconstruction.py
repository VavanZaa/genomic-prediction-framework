"""Run Layer 6B: Real-Data Masked Historical Reconstruction Benchmark.

Hides the real intermediate composite population C (Brangus crossbreds) from the algorithm.
Trains solely on anchor founders A (Angus-leaning) and B (Brahman-leaning).
Asks the unconstrained optimization engine to find the optimal crossing strategy argmax_q U(q).
Tests whether the optimizer independently reconstructs:
- Historical Brangus optimum q* ~ 0.375 (3/8 Brahman - 5/8 Angus)
- Phenotypic distribution of the real masked intermediate cattle C
- Coverage probability P(historical composite C in predicted optimum region)

Outputs:
- output/real_layer6/layer6_masked_reconstruction_scorecard.csv
- output/real_layer6/layer6_masked_reconstruction.png
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

from src.evaluation.masked_reconstruction import MaskedHistoricalReconstructor, MaskedReconstructionResult


def run_masked_reconstruction_benchmark():
    print("=" * 85)
    print("LAYER 6B: REAL-DATA MASKED HISTORICAL RECONSTRUCTION BENCHMARK")
    print("=" * 85)

    out_dir = BASE_DIR / "output" / "real_layer6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[1/4] Ingesting Florida Multibreed Genotypes and Phenotypes...")
    npz_path = BASE_DIR / "data" / "real_cattle" / "florida_multibreed" / "florida_dual_snp_boa_step24_v3.npz"
    pheno_path = BASE_DIR / "data" / "real_cattle" / "florida_multibreed" / "Thermo.Phenotypes.csv"

    from src.real_data.florida_boa_loader import FloridaBOALoader
    from src.real_data.genotype_preprocessing import GenotypeImputer
    dual = FloridaBOALoader.load_dual_dataset(marker_step=24)
    npz_data = {"genotypes_snp":dual.genotypes_snp,"alleles_boa":dual.alleles_boa,"animal_ids":dual.animal_ids}
    # Descriptive/genomic operations use a stated historical reference for missing calls.
    reference = (dual.phenotypes["Year"].to_numpy() <= 2018)
    for matrix in ("genotypes_snp", "alleles_boa"):
        npz_data[matrix] = GenotypeImputer().fit(npz_data[matrix][reference]).transform(npz_data[matrix])
    g_all_snp = npz_data["genotypes_snp"]
    animal_ids = np.array([str(x) for x in npz_data["animal_ids"]])

    df_pheno = pd.read_csv(pheno_path)
    df_pheno["UFID_str"] = df_pheno["UFID"].astype(str)
    pheno_map = {row["UFID_str"]: row for _, row in df_pheno.iterrows()}

    aligned_phenos = [pheno_map.get(aid, None) for aid in animal_ids]
    valid_mask = np.array([p is not None and not np.isnan(p["LoadSlope"]) and not np.isnan(p["BrahmanGBC"]) for p in aligned_phenos])

    g_valid = g_all_snp[valid_mask, :]
    ids_valid = animal_ids[valid_mask]
    phenos_valid = pd.DataFrame([aligned_phenos[i] for i in range(len(animal_ids)) if valid_mask[i]]).reset_index(drop=True)
    qb_valid = phenos_valid["BrahmanGBC"].values
    y_load_valid = phenos_valid["LoadSlope"].values

    print(f"  Valid records: {len(ids_valid)} cattle")

    # 2. Partition into Source Population (Seminole) and Masked Target Herd C (Williamson Commercial Brangus)
    print("\n[2/4] Partitioning Data Strictly by Population Identity (Zero q_B Leakage)...")
    pop_ids = phenos_valid["Project"].astype(str).values

    reconstructor = MaskedHistoricalReconstructor(
        historical_target_qb=0.375,
        adaptation_penalty_weight=0.35,
        random_seed=42
    )

    g_train, y_train, qb_train, g_masked, y_masked, qb_masked = reconstructor.partition_by_population_identity(
        genotypes=g_valid,
        phenotypes_load=y_load_valid,
        ancestry_qb=qb_valid,
        population_ids=pop_ids,
        withhold_pop_id="Wes",
        train_pop_id="Semin"
    )

    print(f"  Source Training Population ('Semin'): {g_train.shape[0]} cattle")
    print(f"    - Training BrahmanGBC: mean={np.mean(qb_train):.3f} +/- {np.std(qb_train):.3f} (min={np.min(qb_train):.3f}, max={np.max(qb_train):.3f})")
    print(f"  MASKED Target Population ('Wes' Commercial Brangus): {g_masked.shape[0]} cattle (100% WITHHELD by population identity)")
    print(f"    * Note: Target ancestry q_B is completely hidden from training and cross-optimization.")

    # 3. Autonomous Cross Optimization
    print("\n[3/4] Fitting the cross-design model on the source population and solving argmax_q U(q)...")
    tau_35 = float(np.percentile(y_train, 35.0))
    res: MaskedReconstructionResult = reconstructor.reconstruct_optimal_cross(
        g_train=g_train,
        y_train=y_train,
        qb_train=qb_train,
        g_masked=g_masked,
        y_masked=y_masked,
        qb_masked=qb_masked,
        tau_load=tau_35,
        q_steps=51
    )

    # Sensitivity run: the literature slope/heterosis the earlier version hardcoded, now labelled as a prior
    res_prior: MaskedReconstructionResult = MaskedHistoricalReconstructor(historical_target_qb=0.375, random_seed=42).reconstruct_optimal_cross(
        g_train=g_train, y_train=y_train, qb_train=qb_train, g_masked=g_masked, y_masked=y_masked,
        qb_masked=qb_masked, tau_load=tau_35, q_steps=51, literature_prior=(-2.71, -1.18)
    )

    print("\n" + "=" * 95)
    print("REAL-DATA MASKED POPULATION RECONSTRUCTION (LAYER 6B)")
    print("=" * 95)
    for label, r in (("Primary (fitted)", res), ("Sensitivity (literature prior)", res_prior)):
        print(f"{label}: {r.parameter_source}")
        print(f"  LoadSlope = {r.fitted_intercept:.2f} + ({r.fitted_beta:.2f}) q_B + ({r.fitted_heterosis:.2f}) * 4 q_B (1 - q_B); residual SD {r.residual_sd:.2f}")
        print(f"  Unconstrained optimum q_B* = {r.predicted_optimal_qb:.3f}{' (boundary of [0, 1])' if r.optimum_at_boundary else ''}; "
              f"within empirical support: {r.is_opt_within_support} ({r.d_ood_at_opt:.2f} SD beyond the "
              f"k = {r.support_k} nearest-neighbour radius {r.support_radius_sd:.3f} SD); "
              f"support-constrained optimum = {r.predicted_safe_qb:.3f}")
    print("-" * 95)
    print("POST-HOC COMPARISON WITH THE WITHHELD POPULATION ('Wes'):")
    print(f"  Historical Brangus standard q_B = {res.historical_target_qb:.3f}")
    print(f"  Withheld herd q_B: mean {res.empirical_hidden_mean_qb:.3f} +/- {res.empirical_hidden_std_qb:.3f}, "
          f"range {np.min(qb_masked):.3f} - {np.max(qb_masked):.3f}")
    print(f"  |q* - withheld mean| = {res.error_vs_empirical:.3f}; |q*_safe - withheld mean| = {abs(res.predicted_safe_qb - res.empirical_hidden_mean_qb):.3f}")
    print(f"  LoadSlope at q*: predicted {res.predicted_mean_load:.2f} +/- {res.predicted_std_load:.2f}; "
          f"withheld herd {res.real_masked_mean_load:.2f} +/- {res.real_masked_std_load:.2f}; bias {res.mean_bias_load:+.2f}")
    print(f"  Coverage of withheld phenotypes by the 90% predicted interval: {res.coverage_probability_90pct*100:.1f}%")
    w1 = {"cross-design model at q*": res.wasserstein_distance_load, "linear ancestry-only at q*": res.w1_ancestry_only,
          "training-population distribution": res.w1_training_mean}
    print("  Wasserstein W_1 to withheld phenotypes: " + ", ".join(f"{k} {v:.3f}" for k, v in w1.items())
          + f" -> lowest: {min(w1, key=w1.get)}")
    print("  Note: a single-trait heat-load optimum is not expected to match a commercial multi-trait herd;")
    print("  these numbers describe the heat-load model only.")
    print("=" * 95)

    # 4. Save Scorecard (primary fitted result plus the literature-prior sensitivity row)
    scorecard_df = pd.concat([res.to_dataframe(), res_prior.to_dataframe()], ignore_index=True)
    scorecard_path = out_dir / "layer6_masked_reconstruction_scorecard.csv"
    scorecard_df.to_csv(scorecard_path, index=False)
    print(f"\nSaved scorecard to {scorecard_path}")

    # 5. Publication Figure (4 Panels)
    print("\n[4/4] Generating 4-Panel Publication Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Utility Function Curve & Constituent Decomposition
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()

    line1 = ax1.plot(res.q_grid, res.utility_profile, "b-o", lw=2.2, label="Thermotolerance Utility $U(q)$")
    ax1.axvline(res.predicted_optimal_qb, color="blue", linestyle="--", lw=2,
                label=f"Unconstrained Optimum $q^* = {res.predicted_optimal_qb:.3f}$")
    ax1.axvline(res.predicted_safe_qb, color="purple", linestyle="-", lw=2,
                label=f"Safe Constrained Optimum $q^*_{{\\rm safe}} = {res.predicted_safe_qb:.3f}$")
    ax1.axvline(res.historical_target_qb, color="darkgreen", linestyle=":", lw=2,
                label=f"Historical Standard ($q_B = 3/8 = {res.historical_target_qb:.3f}$)")
    ax1.axvline(res.empirical_hidden_mean_qb, color="darkorange", linestyle="-.", lw=2,
                label=f"Empirical Mean ('Wes', $\\bar{{q}}_B = {res.empirical_hidden_mean_qb:.3f}$)")
    ax1.axvspan(res.empirical_hidden_mean_qb - res.empirical_hidden_std_qb,
                res.empirical_hidden_mean_qb + res.empirical_hidden_std_qb,
                color="orange", alpha=0.15, label="Commercial $\\pm 1\\sigma$ Band")

    line2 = ax2.plot(res.q_grid, res.load_profile, "r--", lw=1.8, label="Predicted LoadSlope (lower is better)")
    line3 = ax2.plot(res.q_grid, res.heterosis_profile * 10, "m:", lw=1.8, label="Retained Heterosis (scaled)")

    ax1.set_title("A. Cross Optimization & Unconstrained vs Safe Optimum", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Brahman Genomic Ancestry Fraction ($q_B$)", fontsize=10)
    ax1.set_ylabel("Expected Thermotolerance Utility $U(q)$", fontsize=10, color="blue")
    ax2.set_ylabel("Predicted Trait Merit / Heterosis", fontsize=10, color="red")

    lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, frameon=True, fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel B: Epistemic Support & OOD Detector Manifold
    ax = axes[0, 1]
    ax_twin = ax.twinx()

    ax.bar(res.q_grid, res.density_profile, width=0.018, color="gray", alpha=0.35, label="Training Sample Density $N(q)$")
    l_ood = ax_twin.plot(res.q_grid, res.d_ood_profile, "r-", lw=2, label="SD beyond empirical k-NN support $d_{\\rm OOD}(q)$")
    l_conf = ax_twin.plot(res.q_grid, res.confidence_profile, "g-.", lw=2, label="Selective Confidence $C(q)$")

    ax_twin.axvline(res.predicted_optimal_qb, color="blue", linestyle="--", lw=1.8, label=f"Unconstrained $q^* = {res.predicted_optimal_qb:.3f}$")
    ax_twin.fill_between(res.q_grid, 0, 1, where=res.d_ood_profile == 0, color="green", alpha=0.08,
                         transform=ax_twin.get_xaxis_transform(), label="Inside empirical support (safe region)")

    ax.set_title("B. Epistemic Genomic Support Manifold Across Ancestry", fontsize=11, fontweight="bold")
    ax.set_xlabel("Brahman Genomic Ancestry Fraction ($q_B$)", fontsize=10)
    ax.set_ylabel("Training Sample Count $N(q)$", fontsize=10)
    ax_twin.set_ylabel("OOD Metric / Confidence Scale", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax_twin.legend(loc="upper left", fontsize=8, frameon=True)

    # Panel C: Predicted Optimum Distribution vs Real Masked Commercial Herd ('Wes')
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    sim_samples = rng.normal(res.predicted_mean_load, res.predicted_std_load, size=2000)

    ax.hist(y_masked, bins=25, alpha=0.55, color="black", density=True,
            label=f"Real Commercial 'Wes' ($N={res.real_masked_count}$, $\\mu={res.real_masked_mean_load:.2f}$)")
    ax.hist(sim_samples, bins=25, alpha=0.50, color="#1f77b4", density=True,
            label=f"Predicted Optimum Distribution ($\\mu={res.predicted_mean_load:.2f}$)")
    ax.axvline(res.predicted_mean_load, color="#1f77b4", linestyle="--", lw=2)
    ax.axvline(res.real_masked_mean_load, color="black", linestyle="-", lw=2)
    ax.set_title(f"C. Post-Hoc Phenotype Comparison (Bias=+{res.mean_bias_load:.2f}, Cov={res.coverage_probability_90pct*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("LoadSlope (Thermotolerance Merit)", fontsize=10)
    ax.set_ylabel("Empirical Density", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Phenotypic Reconstruction Wasserstein Benchmarks (Negative Result)
    ax = axes[1, 1]
    comparators = ["Proposed Full Cross\n(Counterfactual Opt)", "Ancestry-Only\nLinear Regression", "Trivial Training Mean\nBaseline"]
    w1_vals = [res.wasserstein_distance_load, res.w1_ancestry_only, res.w1_training_mean]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(comparators, w1_vals, color=colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, w1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"$W_1 = {val:.3f}$",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_title("D. Wasserstein Discrepancy Against Real Commercial Herd $W_1$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Wasserstein Distance $W_1$ (Lower is Better)", fontsize=10)
    ax.set_ylim(0, max(w1_vals) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = out_dir / "layer6_masked_reconstruction.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to {fig_path}")


if __name__ == "__main__":
    run_masked_reconstruction_benchmark()

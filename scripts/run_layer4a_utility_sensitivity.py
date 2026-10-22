"""Layer 4A Utility Sensitivity Analysis: LoadSlope-Only, Hair-Only, Weighted, and Joint Regimes.

Addresses the core scientific critique:
Is the BOA advantage robust across objective definitions, or is it driven by the unstable hair-threshold component?

Evaluates 5 Policies across 4 Distinct Objective Regimes:
1. Regime A: LoadSlope-Only Utility: U = P(y_load < tau_L)
2. Regime B: Hair-Only Utility:      U = P(y_hair < tau_H)
3. Regime C: Weighted Marginal (75% Load, 25% Hair): U = 0.75 P(Load) + 0.25 P(Hair)
4. Regime D: Joint Bivariate Target: U = P(Load < tau_L AND Hair < tau_H)

Under each regime, runs R = 200 Monte Carlo forward breeding cohorts with Gaussian Mendelian sampling.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.decision.environment_scenario import EnvironmentEvaluator, EnvironmentScenario
from src.decision.real_mating_optimizer import PolicyType, RealMatingOptimizer
from src.decision.real_offspring_predictor import ParentalMerit, RealOffspringPredictor
from src.evaluation.mating_monte_carlo import MatingMonteCarloSimulator, measured_gebv_accuracy
from src.prediction.kernel_cv import tune_snp_boa_models
from src.real_data.canonical_split import CanonicalDataSplitter
from src.real_data.reference_grm import load_reference_grm

ASSUMED_H2 = 0.30  # assumed heritability for converting predictive ability to accuracy (truth model)
OUT_DIR = BASE_DIR / "output" / "real_layer4"
GRM_PATH = BASE_DIR / "data" / "real_cattle" / "florida_multibreed" / "florida_grm_122k_v3.npz"
PHENO_PATH = BASE_DIR / "data" / "real_cattle" / "florida_multibreed" / "Thermo.Phenotypes.csv"


def select_candidate_sires(
    animal_ids: list[str],
    train_ids: list[str],
    pheno_df: pd.DataFrame,
    n_sires: int = 40,
    seed: int = 42
) -> list[str]:
    rng = np.random.default_rng(seed)
    df_train = pheno_df[pheno_df["UFID"].astype(str).isin(train_ids)].copy()

    low_b = df_train[df_train["BrahmanGBC"] < 0.15]["UFID"].astype(str).values
    mid_b = df_train[(df_train["BrahmanGBC"] >= 0.25) & (df_train["BrahmanGBC"] <= 0.45)]["UFID"].astype(str).values
    high_b = df_train[df_train["BrahmanGBC"] > 0.50]["UFID"].astype(str).values

    s_low = rng.choice(low_b, size=min(8, len(low_b)), replace=False)
    s_mid = rng.choice(mid_b, size=min(20, len(mid_b)), replace=False)
    s_high = rng.choice(high_b, size=min(12, len(high_b)), replace=False)

    selected = list(s_low) + list(s_mid) + list(s_high)
    if len(selected) < n_sires:
        remaining = [aid for aid in train_ids if aid not in selected]
        extra = rng.choice(remaining, size=n_sires - len(selected), replace=False)
        selected.extend(extra)

    return selected[:n_sires]


def run_utility_sensitivity():
    print("=" * 90)
    print("LAYER 4A: OBJECTIVE UTILITY SENSITIVITY ANALYSIS (LOAD-ONLY vs HAIR-ONLY vs WEIGHTED vs JOINT)")
    print("=" * 90)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest GRMs & Phenotypes
    print("\n[1. Ingesting Full-Density GRMs and Multibreed Phenotypes...]")
    grm_data = load_reference_grm(GRM_PATH)
    G_122k = grm_data["G_122k"].astype(np.float64)
    B_122k = grm_data["B_122k"].astype(np.float64)
    animal_ids = [str(x) for x in grm_data["animal_ids"]]
    id_to_idx = {aid: i for i, aid in enumerate(animal_ids)}

    pheno_df = pd.read_csv(PHENO_PATH)
    pheno_df["UFID"] = pheno_df["UFID"].astype(str)

    split_load = CanonicalDataSplitter.get_prospective_split(
        trait_col="LoadSlope", valid_animal_ids=animal_ids, require_joint_traits=True
    )
    split_hair = CanonicalDataSplitter.get_prospective_split(
        trait_col="AvgShortLength", valid_animal_ids=animal_ids, require_joint_traits=True
    )

    train_ids = split_load.train_ids
    test_ids = split_load.test_ids

    tr_idx = [id_to_idx[aid] for aid in train_ids]
    te_idx = [id_to_idx[aid] for aid in test_ids]

    y_tr_l = split_load.y_train_adj
    y_tr_h = split_hair.y_train_adj
    mu_tr_l = float(np.mean(y_tr_l))
    mu_tr_h = float(np.mean(y_tr_h))
    y_tr_cent_l = y_tr_l - mu_tr_l
    y_tr_cent_h = y_tr_h - mu_tr_h

    # 2. Fit Models on Training Data
    print("\n[2. Fitting Dense Genomic Models on Training Data...]")
    G_tr = G_122k[np.ix_(tr_idx, tr_idx)]
    B_tr = B_122k[np.ix_(tr_idx, tr_idx)]

    # Same training-only CV tuning as run_layer4a_mating_benchmark.py
    tuned, _ = tune_snp_boa_models(G_tr, B_tr, {"load": y_tr_l, "hair": y_tr_h})
    lam_snp_l, lam_snp_h = tuned[("SNP", "load")]["lambda"], tuned[("SNP", "hair")]["lambda"]
    lam_boa_l, lam_boa_h = tuned[("BOA", "load")]["lambda"], tuned[("BOA", "hair")]["lambda"]
    (w_g_l, w_b_l), lam_comb_l = tuned[("Combined", "load")]["weights"], tuned[("Combined", "load")]["lambda"]
    (w_g_h, w_b_h), lam_comb_h = tuned[("Combined", "hair")]["weights"], tuned[("Combined", "hair")]["lambda"]
    K_comb_tr_l = w_g_l * G_tr + w_b_l * B_tr
    K_comb_tr_h = w_g_h * G_tr + w_b_h * B_tr

    alpha_snp_l = np.linalg.solve(G_tr + lam_snp_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_snp_h = np.linalg.solve(G_tr + lam_snp_h * np.eye(len(tr_idx)), y_tr_cent_h)
    alpha_boa_l = np.linalg.solve(B_tr + lam_boa_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_boa_h = np.linalg.solve(B_tr + lam_boa_h * np.eye(len(tr_idx)), y_tr_cent_h)
    alpha_comb_l = np.linalg.solve(K_comb_tr_l + lam_comb_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_comb_h = np.linalg.solve(K_comb_tr_h + lam_comb_h * np.eye(len(tr_idx)), y_tr_cent_h)

    K_snp_all_tr = G_122k[:, tr_idx]
    K_boa_all_tr = B_122k[:, tr_idx]
    K_comb_all_tr_l = w_g_l * K_snp_all_tr + w_b_l * K_boa_all_tr
    K_comb_all_tr_h = w_g_h * K_snp_all_tr + w_b_h * K_boa_all_tr

    gebv_snp_l = K_snp_all_tr @ alpha_snp_l
    gebv_snp_h = K_snp_all_tr @ alpha_snp_h
    gebv_boa_l = K_boa_all_tr @ alpha_boa_l
    gebv_boa_h = K_boa_all_tr @ alpha_boa_h
    gebv_comb_l = K_comb_all_tr_l @ alpha_comb_l
    gebv_comb_h = K_comb_all_tr_h @ alpha_comb_h

    # 3. Form Candidate Batteries
    dam_ids = list(test_ids)
    sire_ids = select_candidate_sires(animal_ids, train_ids, pheno_df, n_sires=40, seed=42)
    sire_indices = [id_to_idx[aid] for aid in sire_ids]
    dam_indices = [id_to_idx[aid] for aid in dam_ids]

    pheno_map = pheno_df.set_index("UFID")["BrahmanGBC"].to_dict()

    sires_merit = [
        ParentalMerit(
            animal_id=aid,
            sex="M",
            ancestry_brahman=float(pheno_map.get(aid, 0.35)),
            gebv_load=float(gebv_snp_l[id_to_idx[aid]]),
            gebv_hair=float(gebv_snp_h[id_to_idx[aid]]),
            inbreeding_f=float(G_122k[id_to_idx[aid], id_to_idx[aid]] - 1.0),
        )
        for aid in sire_ids
    ]
    dams_merit = [
        ParentalMerit(
            animal_id=aid,
            sex="F",
            ancestry_brahman=float(pheno_map.get(aid, 0.35)),
            gebv_load=float(gebv_snp_l[id_to_idx[aid]]),
            gebv_hair=float(gebv_snp_h[id_to_idx[aid]]),
            inbreeding_f=float(G_122k[id_to_idx[aid], id_to_idx[aid]] - 1.0),
        )
        for aid in dam_ids
    ]

    predictor = RealOffspringPredictor(grm_matrix=G_122k, animal_ids=animal_ids)
    optimizer = RealMatingOptimizer(
        predictor=predictor,
        sires=sires_merit,
        dams=dams_merit,
        max_sire_quota=25,
        max_coancestry_threshold=0.0625,  # half-sib coancestry ceiling
        ancestry_target=0.375,
    )

    _, model_accuracy = measured_gebv_accuracy(
        {"SNP": (gebv_snp_l, gebv_snp_h), "BOA": (gebv_boa_l, gebv_boa_h), "Combined": (gebv_comb_l, gebv_comb_h)},
        te_idx, split_load.y_test_adj, split_hair.y_test_adj, ASSUMED_H2,
    )
    reference_accuracy = model_accuracy["SNP"]


    scen_a = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]
    tau_l = float(np.quantile(y_tr_l, 0.35))   # 28.51
    tau_h = float(np.quantile(y_tr_h, 0.35))   # 5.93 mm

    policy_specs = [
        (PolicyType.RANDOM, gebv_snp_l, gebv_snp_h, "Random Mating"),
        (PolicyType.GLOBAL_ANCESTRY, gebv_snp_l, gebv_snp_h, "Global Ancestry (5/8)"),
        (PolicyType.DENSE_SNP, gebv_snp_l, gebv_snp_h, "Dense SNP (M1)"),
        (PolicyType.DENSE_BOA, gebv_boa_l, gebv_boa_h, "Dense BOA (M2)"),
        (PolicyType.COMBINED_SNP_BOA, gebv_comb_l, gebv_comb_h, "Combined SNP+BOA (M3)"),
    ]

    regimes = [
        ("loadslope_only", 1.0, "Regime 1: LoadSlope-Only Utility (Pure Thermotolerance)"),
        ("hair_only", 0.0, "Regime 2: Hair-Only Utility (Pure Hair Reduction)"),
        ("weighted_marginal", 0.75, "Regime 3: Weighted Utility (75% Load, 25% Hair)"),
        ("joint", 0.50, "Regime 4: Joint Bivariate Target (Load < tau_L AND Hair < tau_H)"),
    ]

    all_sensitivity_rows = []
    reference_ids = sire_ids + dam_ids
    reference_gebvs = {aid: (float(gebv_snp_l[id_to_idx[aid]]), float(gebv_snp_h[id_to_idx[aid]])) for aid in reference_ids}
    print("  Counterfactual sensitivity under one SNP reference truth model; not empirical offspring validation.")
    mc_simulator = MatingMonteCarloSimulator(predictor=predictor, num_replicates=200, random_state=42, gebv_accuracy=reference_accuracy, reference_parental_gebvs=reference_gebvs, truth_source="SNP reference; accuracy from prospective phenotypes with assumed h2; sensitivity only")

    print("\n[3. Evaluating 5 Policies Across 4 Objective Regimes with 200 Monte Carlo Replicates Each...]")

    for mode_key, w_l, regime_desc in regimes:
        print(f"\n--- Running {regime_desc} ---")
        regime_summaries = {}

        for pol_type, g_l_all, g_h_all, label in policy_specs:
            s_g_l = g_l_all[sire_indices]
            s_g_h = g_h_all[sire_indices]
            d_g_l = g_l_all[dam_indices]
            d_g_h = g_h_all[dam_indices]

            summary = optimizer.optimize_policy(
                policy=pol_type,
                sire_gebvs_load=s_g_l,
                sire_gebvs_hair=s_g_h,
                dam_gebvs_load=d_g_l,
                dam_gebvs_hair=d_g_h,
                scenario_eval=scen_a,
                tau_load=tau_l,
                tau_hair=tau_h,
                objective_mode=mode_key,
                weight_load=w_l,
                seed=42,
            )
            regime_summaries[pol_type.value] = summary

        # Run Monte Carlo Forward Simulation (200 Replicates) for this regime
        mc_res = mc_simulator.run_replicates(regime_summaries, scen_a, tau_load=tau_l, tau_hair=tau_h)

        for pol_name, summ in regime_summaries.items():
            mc = mc_res[pol_name]
            all_sensitivity_rows.append({
                "Regime": mode_key,
                "Regime_Description": regime_desc,
                "Policy": pol_name,
                "Truth_Model": mc.truth_model,
                "Accuracy_Load": mc.accuracy_load,
                "Accuracy_Hair": mc.accuracy_hair,
                "Mean_GEBV_Load": summ.mean_gebv_load,
                "Mean_GEBV_Hair": summ.mean_gebv_hair,
                "Mean_Parental_Coancestry": summ.mean_parental_coancestry,
                "MC_Realized_Load_Pct": mc.mean_load_pct,
                "P(Load>Ancestry)": mc.prob_load_superior_to_ancestry,
                "MC_Realized_Hair_Pct": mc.mean_hair_pct,
                "P(Hair>Ancestry)": mc.prob_hair_superior_to_ancestry,
                "MC_Realized_Joint_Pct": mc.mean_joint_pct,
                "P(Joint>Ancestry)": mc.prob_joint_superior_to_ancestry,
                "Delta_Joint_vs_Ancestry_Pct": mc.mean_delta_joint_vs_ancestry_pct,
            })

    sens_df = pd.DataFrame(all_sensitivity_rows)
    csv_path = OUT_DIR / "layer4a_utility_sensitivity_scorecard.csv"
    sens_df.to_csv(csv_path, index=False)
    print(f"\nSaved utility sensitivity scorecard to: {csv_path}")

    # 4. Display Key Comparative Subsets
    for mode_key, w_l, regime_desc in regimes:
        sub = sens_df[sens_df["Regime"] == mode_key]
        print(f"\n==========================================================================================================")
        print(f"{regime_desc}:")
        print(f"==========================================================================================================")
        print(sub[[
            "Policy", "Mean_GEBV_Load", "Mean_GEBV_Hair",
            "MC_Realized_Load_Pct", "P(Load>Ancestry)",
            "MC_Realized_Hair_Pct", "P(Hair>Ancestry)",
            "MC_Realized_Joint_Pct", "P(Joint>Ancestry)"
        ]].to_string(index=False))

    # 5. Generate Multi-Panel Publication Figure
    print("\n[4. Generating Layer 4A Utility Sensitivity Multi-Panel Plot...]")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: LoadSlope-Only Regime - Realized Load Attainment
    ax = axes[0, 0]
    sub1 = sens_df[sens_df["Regime"] == "loadslope_only"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(sub1["Policy"], sub1["MC_Realized_Load_Pct"], color=colors, alpha=0.85)
    ax.set_title("A. Regime 1 (LoadSlope-Only): Realized Load Attainment Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Offspring Load Attainment (%)", fontsize=10)
    ax.set_xticklabels(sub1["Policy"], rotation=20, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Hair-Only Regime - Realized Hair Attainment
    ax = axes[0, 1]
    sub2 = sens_df[sens_df["Regime"] == "hair_only"]
    bars = ax.bar(sub2["Policy"], sub2["MC_Realized_Hair_Pct"], color=colors, alpha=0.85)
    ax.set_title("B. Regime 2 (Hair-Only): Realized Hair Attainment Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Offspring Hair Attainment (%)", fontsize=10)
    ax.set_xticklabels(sub2["Policy"], rotation=20, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel C: Weighted Marginal Regime (75% Load, 25% Hair)
    ax = axes[1, 0]
    sub3 = sens_df[sens_df["Regime"] == "weighted_marginal"]
    x = np.arange(len(sub3))
    w = 0.35
    ax.bar(x - w/2, sub3["MC_Realized_Load_Pct"], w, label="Load Attainment (%)", color="#2ca02c", alpha=0.85)
    ax.bar(x + w/2, sub3["MC_Realized_Hair_Pct"], w, label="Hair Attainment (%)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sub3["Policy"], rotation=20, ha="right", fontsize=9)
    ax.set_title("C. Regime 3 (75% Load, 25% Hair): Trait Decomposition", fontsize=11, fontweight="bold")
    ax.set_ylabel("Realized Attainment Rate (%)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Empirical Probability of Outperforming Global Ancestry across Regimes
    ax = axes[1, 1]
    policies_gen = ["DenseSNP", "DenseBOA", "Combined_SNP_BOA"]
    regime_keys = ["loadslope_only", "hair_only", "weighted_marginal", "joint"]
    regime_labels = ["Load-Only", "Hair-Only", "Weighted (75:25)", "Joint (50:50)"]

    x = np.arange(len(regime_labels))
    w = 0.25
    col_dict = {"DenseSNP": "#2ca02c", "DenseBOA": "#ff7f0e", "Combined_SNP_BOA": "#d62728"}

    for idx, p in enumerate(policies_gen):
        probs = []
        for r_key in regime_keys:
            val = sens_df[(sens_df["Regime"] == r_key) & (sens_df["Policy"] == p)]["P(Joint>Ancestry)" if r_key in ("weighted_marginal", "joint") else ("P(Load>Ancestry)" if r_key == "loadslope_only" else "P(Hair>Ancestry)")].values[0]
            probs.append(val * 100.0)
        ax.bar(x + (idx - 1)*w, probs, w, label=p, color=col_dict[p], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=10, fontweight="bold")
    ax.set_title("D. Empirical Probability P(Genomic > Ancestry) across Utility Regimes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Outperforming Ancestry (%)", fontsize=10)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1.0, label="Neutral (50%)")
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = OUT_DIR / "layer4a_utility_sensitivity.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved sensitivity plot to: {fig_path}")


if __name__ == "__main__":
    run_utility_sensitivity()

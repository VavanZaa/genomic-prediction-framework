"""Layer 4A-v2: Real Genomics- and BOA-Aware Thermotolerance Mating Optimization Benchmark.

Counterfactual Mating Experiment on Real Physical Cattle Genomes:
- Dam Herd: N_d = 658 commercial replacement heifers from prospective test cohort (Williamson cattle).
- Candidate Sire Battery: N_s = 40 computational sire-role candidates stratified across the Brahman ancestry gradient
  (low Brahman/pure Angus, pure Brahman, F1, 3/8-5/8 Brangus) drawn from the historical research population.
- Equal Resource & Inbreeding Parity:
  * Maximum sire quota: C_sire = 25 calves/bull
  * Parental genomic coancestry ceiling: Theta_{sd} = 0.5 * G_{sd} <= 0.0625 (precludes half-sib or closer matings, G_{sd} <= 0.125)
  * Exactly 1 allocation per dam (N_matings = 658)

Compares 5 Distinct Breeding Policies:
1. Random Mating (Baseline Control)
2. Global Ancestry Mating (Pedigree/genomic 5/8 Angus : 3/8 Brahman target)
3. Dense SNP Genomic Mating (pair-specific mid-parent 122k SNP GEBV, Gaussian Mendelian sampling)
4. Dense BOA Genomic Mating (pair-specific mid-parent 122k BOA GEBV, Gaussian Mendelian sampling)
5. Combined SNP+BOA Genomic Mating (pair-specific mid-parent composite-kernel GEBV, Gaussian Mendelian sampling)

Counterfactual Sensitivity Analysis:
- Monte Carlo forward simulation across R = 200 independent cohorts with Gaussian Mendelian sampling for every mating.
- Evaluates realized offspring attainment under decoupled environmental contexts:
  * Scenario A: Subtropical Baseline (Seminole conditions: mu_L=29.58, mu_H=6.80 mm)
  * Scenario B: Elevated Heat & Shifted Pasture (Williamson conditions: mu_L=29.58, mu_H=8.56 mm)
- Reports empirical Delta_bar, 95% empirical confidence intervals, and P(genomic beats ancestry).
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

# Assumed heritability used to turn held-out predictive ability into accuracy for the Monte Carlo
# truth model (labelled "assumed" in the outputs).
ASSUMED_H2 = 0.30
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
    """Selects 40 computational sire-role candidate genomes from historical training cattle spanning ancestry groups."""
    rng = np.random.default_rng(seed)
    df_train = pheno_df[pheno_df["UFID"].astype(str).isin(train_ids)].copy()

    # Strata definitions by Brahman genomic fraction
    # 1. Low Brahman / High Angus (q < 0.15): 8 candidates
    # 2. Intermediate Brangus (0.25 <= q <= 0.45): 20 candidates
    # 3. High Brahman (q > 0.50): 12 candidates
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


def run_layer4a_v2_benchmark():
    print("=" * 85)
    print("LAYER 4A-v2: PAIR-SPECIFIC MATING OPTIMIZATION (GAUSSIAN MENDELIAN SAMPLING) & MONTE CARLO BENCHMARK")
    print("=" * 85)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load GRMs & Phenotypes
    print("\n[1. Ingesting Full-Density GRMs and Multibreed Phenotypes...]")
    grm_data = load_reference_grm(GRM_PATH)
    G_122k = grm_data["G_122k"].astype(np.float64)
    B_122k = grm_data["B_122k"].astype(np.float64)
    animal_ids = [str(x) for x in grm_data["animal_ids"]]
    id_to_idx = {aid: i for i, aid in enumerate(animal_ids)}
    num_markers = int(grm_data["num_markers"])
    print(f"  Loaded G_122k & B_122k for {len(animal_ids)} cattle across {num_markers} markers.")

    pheno_df = pd.read_csv(PHENO_PATH)
    pheno_df["UFID"] = pheno_df["UFID"].astype(str)

    # 2. Canonical Prospective Split
    split_load = CanonicalDataSplitter.get_prospective_split(
        trait_col="LoadSlope", valid_animal_ids=animal_ids, require_joint_traits=True
    )
    split_hair = CanonicalDataSplitter.get_prospective_split(
        trait_col="AvgShortLength", valid_animal_ids=animal_ids, require_joint_traits=True
    )

    train_ids = split_load.train_ids
    test_ids = split_load.test_ids
    print(f"  Historical Training Population (Seminole): N = {len(train_ids)} cattle")
    print(f"  Commercial Heifer Dam Herd (Williamson):   N = {len(test_ids)} heifers")

    tr_idx = [id_to_idx[aid] for aid in train_ids]
    te_idx = [id_to_idx[aid] for aid in test_ids]

    y_tr_l = split_load.y_train_adj
    y_tr_h = split_hair.y_train_adj
    mu_tr_l = float(np.mean(y_tr_l))
    mu_tr_h = float(np.mean(y_tr_h))
    y_tr_cent_l = y_tr_l - mu_tr_l
    y_tr_cent_h = y_tr_h - mu_tr_h

    # 3. Fit Prediction Models Strictly on Training Data
    print("\n[2. Fitting Dense Genomic Models Strictly on Training Data...]")
    G_tr = G_122k[np.ix_(tr_idx, tr_idx)]
    B_tr = B_122k[np.ix_(tr_idx, tr_idx)]

    # Ridge penalties and SNP/BOA kernel weights chosen by 5-fold CV on the training cohort only
    # (these replace hardcoded values whose origin was not recorded)
    tuned, tuning_table = tune_snp_boa_models(G_tr, B_tr, {"load": y_tr_l, "hair": y_tr_h})
    tuning_table.to_csv(OUT_DIR / "layer4a_gebv_tuning.csv", index=False)
    lam_snp_l, lam_snp_h = tuned[("SNP", "load")]["lambda"], tuned[("SNP", "hair")]["lambda"]
    lam_boa_l, lam_boa_h = tuned[("BOA", "load")]["lambda"], tuned[("BOA", "hair")]["lambda"]
    (w_g_l, w_b_l), lam_comb_l = tuned[("Combined", "load")]["weights"], tuned[("Combined", "load")]["lambda"]
    (w_g_h, w_b_h), lam_comb_h = tuned[("Combined", "hair")]["weights"], tuned[("Combined", "hair")]["lambda"]
    for (model, trait), res in tuned.items():
        print(f"  {model:9s} {trait:5s}: lambda {res['lambda']:g}, kernel weights {res['weights']}, training CV r {res['cv_r']:.3f}")
    K_comb_tr_l = w_g_l * G_tr + w_b_l * B_tr
    K_comb_tr_h = w_g_h * G_tr + w_b_h * B_tr

    # Model 1: Dense SNP GBLUP
    alpha_snp_l = np.linalg.solve(G_tr + lam_snp_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_snp_h = np.linalg.solve(G_tr + lam_snp_h * np.eye(len(tr_idx)), y_tr_cent_h)

    # Model 2: Dense BOA GBLUP
    alpha_boa_l = np.linalg.solve(B_tr + lam_boa_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_boa_h = np.linalg.solve(B_tr + lam_boa_h * np.eye(len(tr_idx)), y_tr_cent_h)

    # Model 3: Combined SNP+BOA
    alpha_comb_l = np.linalg.solve(K_comb_tr_l + lam_comb_l * np.eye(len(tr_idx)), y_tr_cent_l)
    alpha_comb_h = np.linalg.solve(K_comb_tr_h + lam_comb_h * np.eye(len(tr_idx)), y_tr_cent_h)

    # Precompute GEBVs for ALL cattle
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

    # Held-out accuracy of each GEBV model on the prospective cohort. Used only by the Monte Carlo
    # truth model (step 5), never by the mating optimizer.
    acc_df, model_accuracy = measured_gebv_accuracy(
        {"SNP": (gebv_snp_l, gebv_snp_h), "BOA": (gebv_boa_l, gebv_boa_h), "Combined": (gebv_comb_l, gebv_comb_h)},
        te_idx, split_load.y_test_adj, split_hair.y_test_adj, ASSUMED_H2,
    )
    acc_df.to_csv(OUT_DIR / "layer4a_gebv_accuracy.csv", index=False)
    print("  Prospective-cohort predictive ability and implied accuracy (h2 assumed):")
    print("  " + acc_df.round(4).to_string(index=False).replace("\n", "\n  "))
    reference_accuracy = model_accuracy["SNP"]


    # 4. Form Candidate Batteries
    print("\n[3. Formulating Candidate Parental Batteries under Real Data Constraints...]")
    dam_ids = list(test_ids)
    sire_ids = select_candidate_sires(animal_ids, train_ids, pheno_df, n_sires=40, seed=42)

    print(f"  Candidate Sire-Role Battery: N_s = {len(sire_ids)} genomes (stratified across ancestry)")
    print(f"  Commercial Heifer Dam Herd:  N_d = {len(dam_ids)} replacement heifers")

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

    # 5. Initialize Predictor and Optimizer with Coancestry Ceiling
    predictor = RealOffspringPredictor(grm_matrix=G_122k, animal_ids=animal_ids)
    # Theta_max = 0.0625 corresponds to half-sib coancestry (G_sd <= 0.125)
    optimizer = RealMatingOptimizer(
        predictor=predictor,
        sires=sires_merit,
        dams=dams_merit,
        max_sire_quota=25,
        max_coancestry_threshold=0.0625,
        ancestry_target=0.375,
    )

    # 6. Execute pair-specific optimization (mid-parent GEBV + Gaussian Mendelian sampling) for 5 policies
    print("\n[4. Executing Pair-Specific Constrained Mating Optimization...]")
    scen_a = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]
    scen_b = EnvironmentEvaluator.CANONICAL_SCENARIOS["Elevated_Heat_Shifted"]
    tau_l = float(np.quantile(y_tr_l, 0.35))   # 28.51
    tau_h = float(np.quantile(y_tr_h, 0.35))   # 5.93 mm

    policy_specs = [
        (PolicyType.RANDOM, gebv_snp_l, gebv_snp_h, "Random Mating (Control)"),
        (PolicyType.GLOBAL_ANCESTRY, gebv_snp_l, gebv_snp_h, "Global Ancestry (5/8 Target)"),
        (PolicyType.DENSE_SNP, gebv_snp_l, gebv_snp_h, "Dense SNP Genomic Mating (M1)"),
        (PolicyType.DENSE_BOA, gebv_boa_l, gebv_boa_h, "Dense BOA Genomic Mating (M2)"),
        (PolicyType.COMBINED_SNP_BOA, gebv_comb_l, gebv_comb_h, "Combined SNP+BOA Mating (M3)"),
    ]

    policy_summaries: dict = {}
    summary_records: list[dict] = []

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
            scenario_alt=scen_b,
            tau_load=tau_l,
            tau_hair=tau_h,
            seed=42,
        )

        policy_summaries[pol_type.value] = summary
        summary_records.append({
            "Policy": pol_type.value,
            "Label": label,
            "N_Matings": summary.num_matings,
            "Sires_Used": summary.num_unique_sires_used,
            "Max_Sire_Usage": summary.max_sire_usage,
            "Mean_Ancestry": summary.mean_offspring_ancestry,
            "Ancestry_Target_MAD": summary.ancestry_target_mad,
            "Mean_GEBV_Load": summary.mean_gebv_load,
            "Mean_GEBV_Hair": summary.mean_gebv_hair,
            "Mean_Parental_Coancestry": summary.mean_parental_coancestry,
            "Max_Parental_Coancestry": summary.max_parental_coancestry,
            "Population_Delta_F": summary.population_delta_f,
            "Expected_Attainment_Env_A_Pct": summary.mean_prob_attainment_env_a * 100.0,
            "Expected_Attainment_Env_B_Pct": summary.mean_prob_attainment_env_b * 100.0,
        })

    summary_df = pd.DataFrame(summary_records)

    # 7. Print Optimization Scorecard
    print("\n" + "=" * 115)
    print("LAYER 4A-v2: MATING OPTIMIZATION SCORECARD (COANCESTRY CEILING Theta_sd <= 0.0625, QUOTA C_sire <= 25)")
    print("=" * 115)
    print(summary_df[[
        "Policy", "Sires_Used", "Max_Sire_Usage", "Ancestry_Target_MAD",
        "Mean_GEBV_Load", "Mean_GEBV_Hair", "Mean_Parental_Coancestry", "Population_Delta_F",
        "Expected_Attainment_Env_A_Pct", "Expected_Attainment_Env_B_Pct"
    ]].to_string(index=False))

    csv_path = OUT_DIR / "layer4a_policy_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved Layer 4A optimization scorecard to: {csv_path}")

    # 8. Monte Carlo forward evaluation (R = 200 replicates). Primary: parents' true breeding values
    # drawn around their GEBVs with the measured accuracy. Legacy: GEBVs treated as true values
    # (circular; kept only to show how much of the apparent advantage it creates).
    print("\n[5. Monte Carlo forward evaluation across R = 200 replicate cohorts (accuracy-based truth model)...]")
    reference_ids = sire_ids + dam_ids
    reference_gebvs = {aid: (float(gebv_snp_l[id_to_idx[aid]]), float(gebv_snp_h[id_to_idx[aid]])) for aid in reference_ids}
    print("  Counterfactual sensitivity under one SNP reference truth model; not empirical offspring validation.")
    mc_simulator = MatingMonteCarloSimulator(predictor=predictor, num_replicates=200, random_state=42, gebv_accuracy=reference_accuracy, reference_parental_gebvs=reference_gebvs, truth_source="SNP reference; accuracy from prospective phenotypes with assumed h2; sensitivity only")
    legacy_simulator = MatingMonteCarloSimulator(predictor=predictor, num_replicates=200, random_state=42, reference_parental_gebvs=reference_gebvs, truth_source="SNP reference GEBVs treated as truth; sensitivity only")

    mc_results_a = mc_simulator.run_replicates(policy_summaries, scen_a, tau_load=tau_l, tau_hair=tau_h)
    mc_results_b = mc_simulator.run_replicates(policy_summaries, scen_b, tau_load=tau_l, tau_hair=tau_h)
    legacy_a = legacy_simulator.run_replicates(policy_summaries, scen_a, tau_load=tau_l, tau_hair=tau_h)

    mc_rows = []
    for p_name in policy_summaries.keys():
        res_a = mc_results_a[p_name]
        res_b = mc_results_b[p_name]
        mc_rows.append({
            "Policy": p_name,
            "Truth_Model": res_a.truth_model,
            "Accuracy_Load": res_a.accuracy_load,
            "Accuracy_Hair": res_a.accuracy_hair,
            "Unassigned_Dams": policy_summaries[p_name].num_unassigned_dams,
            "Env_A_Mean_Attainment_Pct": res_a.mean_attainment_pct,
            "Env_A_SD_Pct": res_a.sd_attainment_pct,
            "Env_A_CI95": f"[{res_a.ci95_attainment_pct[0]:.2f}, {res_a.ci95_attainment_pct[1]:.2f}]",
            "Env_A_Delta_vs_Ancestry_Pct": res_a.mean_delta_vs_ancestry_pct,
            "Env_A_Delta_CI95": f"[{res_a.ci95_delta_vs_ancestry_pct[0]:.2f}, {res_a.ci95_delta_vs_ancestry_pct[1]:.2f}]",
            "Env_A_P(Genomic>Ancestry)": res_a.prob_superior_to_ancestry,
            "Env_B_Mean_Attainment_Pct": res_b.mean_attainment_pct,
            "Env_B_Delta_vs_Ancestry_Pct": res_b.mean_delta_vs_ancestry_pct,
            "Env_B_P(Genomic>Ancestry)": res_b.prob_superior_to_ancestry,
            "Legacy_GEBV_as_Truth_Env_A_Mean_Pct": legacy_a[p_name].mean_attainment_pct,
            "Legacy_GEBV_as_Truth_Env_A_P(>Ancestry)": legacy_a[p_name].prob_superior_to_ancestry,
        })

    mc_df = pd.DataFrame(mc_rows)
    mc_csv_path = OUT_DIR / "layer4a_monte_carlo_results.csv"
    mc_df.to_csv(mc_csv_path, index=False)

    print("\n" + "=" * 115)
    print("MONTE CARLO FORWARD EVALUATION (R = 200 replicate cohorts; Gaussian Mendelian sampling around GEBVs;")
    print(f"true breeding values drawn with the measured accuracy, h2 assumed {ASSUMED_H2}):")
    print("=" * 115)
    print(mc_df[[
        "Policy", "Accuracy_Load", "Accuracy_Hair", "Env_A_Mean_Attainment_Pct", "Env_A_CI95",
        "Env_A_Delta_vs_Ancestry_Pct", "Env_A_Delta_CI95", "Env_A_P(Genomic>Ancestry)",
        "Legacy_GEBV_as_Truth_Env_A_P(>Ancestry)"
    ]].to_string(index=False))

    # 9. Publication-Quality Figures
    print("\n[6. Generating Layer 4A-v2 Publication-Quality Comparative Figures...]")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Offspring GEBV Merit Space
    ax = axes[0, 0]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    markers = ["o", "s", "^", "D", "*"]
    for i, row in summary_df.iterrows():
        ax.scatter(row["Mean_GEBV_Hair"], row["Mean_GEBV_Load"], s=180, color=colors[i], marker=markers[i], label=row["Policy"], zorder=5)
        ax.text(row["Mean_GEBV_Hair"] + 0.005, row["Mean_GEBV_Load"] + 0.01, row["Policy"], fontsize=10, fontweight="bold")
    ax.set_title("A. Expected Offspring Genetic Merit (Lower is Favorable)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Offspring Mean GEBV: Hair Length (mm)", fontsize=11)
    ax.set_ylabel("Offspring Mean GEBV: Heat Load Slope", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: Monte Carlo Attainment Distributions across 200 Replicate Breeding Programs
    ax = axes[0, 1]
    box_data = [mc_results_a[p].replicate_attainments for p in summary_df["Policy"]]
    bp = ax.boxplot(box_data, tick_labels=summary_df["Policy"], patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("B. Monte Carlo Realized Attainment (R = 200 Replicate Cohorts)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Realized Cohort Attainment (%)", fontsize=11)
    ax.set_xticklabels(summary_df["Policy"], rotation=20, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Brahman Fraction Target MAD from 3/8
    ax = axes[1, 0]
    ax.set_xticks(np.arange(len(summary_df)))
    ax.set_xticklabels(summary_df["Policy"], rotation=20, ha="right", fontsize=9)
    ax.bar(np.arange(len(summary_df)), summary_df["Ancestry_Target_MAD"], color=["#7f7f7f", "#3182bd", "#31a354", "#fd8d3c", "#e6550d"], alpha=0.85)
    ax.set_title("C. Brahman Fraction Target Deviation (MAD from 3/8)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Absolute Deviation |q - 0.375|", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Coancestry Control & Sire Utilization
    ax = axes[1, 1]
    x = np.arange(len(summary_df))
    width = 0.35
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Policy"], rotation=20, ha="right", fontsize=9)
    ax2 = ax.twinx()
    l1 = ax.bar(x - width/2, summary_df["Mean_Parental_Coancestry"], width, label="Mean Coancestry (Theta_sd)", color="#756bb1", alpha=0.85)
    l2 = ax2.bar(x + width/2, summary_df["Sires_Used"], width, label="Unique Sires Used", color="#636363", alpha=0.6)
    ax.axhline(0.0625, color="red", linestyle="--", linewidth=1.2, label="Half-Sib Ceiling (0.0625)")
    ax.set_title("D. Coancestry Control & Sire Battery Utilization", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parental Genomic Coancestry", fontsize=11, color="#756bb1")
    ax2.set_ylabel("Number of Active Sires (max 40)", fontsize=11, color="#636363")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "layer4a_policy_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Layer 4A-v2 comparative figure to: {fig_path}")


if __name__ == "__main__":
    run_layer4a_v2_benchmark()

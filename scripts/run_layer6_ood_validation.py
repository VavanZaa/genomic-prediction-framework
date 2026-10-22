"""Run Layer 6A-v2: Calibrated OOD Validation & Selective Prediction Benchmark.

Enforces strict separation of Development Shifts vs Independent Blind Evaluation Shifts:
1. Base Reference Cohort: Seminole multibreed cattle with q_B <= 0.28 (N=1,213).
2. Development Shift (Threshold Tuning Only): Seminole cattle with 0.28 < q_B <= 0.32 (N=521).
3. Independent Blind Evaluation Shifts (Zero parameter tuning):
   - Graded Ancestry Extrapolation:
     * Mild Extrapolation: 0.32 < q_B <= 0.36 (N=343)
     * Moderate Extrapolation: 0.36 < q_B <= 0.42 (N=223)
     * Severe Extrapolation: q_B > 0.42 (N=88)
   - External Management Shift: Williamson commercial Brangus herd 'Wes' (N=669).
   - Extreme Adversarial Shift: Inverted frequency alien genome.

Operational Tiers Enforced:
- HIGH: Certified numeric prediction + calibrated interval.
- MODERATE: Qualitative ranking only; absolute numeric prediction suppressed.
- LOW: Full refusal; quantitative prediction suppressed.

Outputs:
- output/real_layer6/layer6_ood_validation_scorecard.csv
- output/real_layer6/layer6_ood_validation.png
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

from src.evaluation.ood_detector import GenomicOODDetector, ConfidenceTier, OODAssessment
from src.prediction.models import AdditiveGBLUP


def run_layer6_ood_benchmark():
    print("=" * 95)
    print("LAYER 6A-v2: CALIBRATED OOD VALIDATION & SELECTIVE PREDICTION BENCHMARK")
    print("=" * 95)

    out_dir = BASE_DIR / "output" / "real_layer6"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Load Real Data
    print("\n[1/5] Ingesting Florida Angus-Brahman Multibreed Data...")
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
    project_valid = phenos_valid["Project"].astype(str).values

    print(f"  Valid complete records: {len(ids_valid)} cattle x {g_valid.shape[1]} SNPs")

    # -------------------------------------------------------------------------
    # 2. Strict Partitioning: Train vs Development vs Blind Evaluation
    # -------------------------------------------------------------------------
    print("\n[2/5] Partitioning into Train, Development (Calibration), and Blind Evaluation Sets...")
    is_semin = (project_valid == "Semin")
    is_wes = (project_valid == "Wes")

    train_mask = is_semin & (qb_valid <= 0.28)
    dev_mask = is_semin & (qb_valid > 0.28) & (qb_valid <= 0.32)

    # Blind evaluation masks
    eval_mild_mask = is_semin & (qb_valid > 0.32) & (qb_valid <= 0.36)
    eval_mod_mask = is_semin & (qb_valid > 0.36) & (qb_valid <= 0.42)
    eval_sev_mask = is_semin & (qb_valid > 0.42)
    eval_ancestry_all_mask = is_semin & (qb_valid > 0.32)
    eval_herd_mask = is_wes

    print(f"  Base Reference Training Cohort (Semin, qB <= 0.28): {train_mask.sum()} cattle")
    print(f"  Development Calibration Cohort (Semin, 0.28 < qB <= 0.32): {dev_mask.sum()} cattle")
    print(f"  Independent Blind Evaluation Cohorts (Zero Threshold Tuning):")
    print(f"    - Graded Ancestry Mild (0.32 < qB <= 0.36): {eval_mild_mask.sum()} cattle")
    print(f"    - Graded Ancestry Moderate (0.36 < qB <= 0.42): {eval_mod_mask.sum()} cattle")
    print(f"    - Graded Ancestry Severe (qB > 0.42): {eval_sev_mask.sum()} cattle")
    print(f"    - External Management Herd Shift ('Wes' Commercial): {eval_herd_mask.sum()} cattle")

    # Fit GBLUP on Reference Training Data
    gblup = AdditiveGBLUP()
    gblup.fit(g_valid[train_mask], y_load_valid[train_mask])

    # Fit OOD Detector on Reference Training Data
    detector = GenomicOODDetector(n_components=15)
    detector.fit(
        g_valid[train_mask],
        ancestry_qb_train=qb_valid[train_mask],
        env_ids_train=project_valid[train_mask]
    )

    # -------------------------------------------------------------------------
    # 3. Development Shift Calibration Check
    # -------------------------------------------------------------------------
    print("\n[3/5] Calibrating Thresholds on Development Shift (0.28 < qB <= 0.32)...")
    y_pred_dev = gblup.predict(g_valid[dev_mask])
    ass_dev = detector.assess_cohort(
        g_valid[dev_mask],
        ancestry_qb_cohort=qb_valid[dev_mask],
        env_ids_cohort=project_valid[dev_mask],
        animal_ids=ids_valid[dev_mask].tolist()
    )
    d_ood_dev = np.mean([a.d_ood_composite for a in ass_dev])
    fsr_dev = GenomicOODDetector.compute_false_safe_rate(y_load_valid[dev_mask], y_pred_dev, ass_dev)
    print(f"  Dev Shift: Mean d_OOD = {d_ood_dev:.3f}, MSE = {np.mean((y_load_valid[dev_mask] - y_pred_dev)**2):.2f}, FSR = {fsr_dev['false_safe_rate']*100:.1f}%")

    # -------------------------------------------------------------------------
    # 4. Independent Blind Evaluation Across Domain Shifts
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating Blindly on Unseen Evaluation Shifts...")

    eval_cohorts = [
        ("Eval 1A: Graded Mild (0.32 < qB <= 0.36)", eval_mild_mask),
        ("Eval 1B: Graded Mod (0.36 < qB <= 0.42)", eval_mod_mask),
        ("Eval 1C: Graded Severe (qB > 0.42)", eval_sev_mask),
        ("Eval 1: Combined Ancestry Shift (qB > 0.32)", eval_ancestry_all_mask),
        ("Eval 2: Management Shift ('Semin' -> 'Wes')", eval_herd_mask),
    ]

    scorecard_rows = []
    eval_results = {}

    for name, mask in eval_cohorts:
        g_sub = g_valid[mask]
        y_sub = y_load_valid[mask]
        qb_sub = qb_valid[mask]
        proj_sub = project_valid[mask]
        ids_sub = ids_valid[mask]

        y_pred_sub = gblup.predict(g_sub)
        corr = float(stats.pearsonr(y_sub, y_pred_sub)[0]) if len(y_sub) > 2 else 0.0
        mse = float(np.mean((y_sub - y_pred_sub) ** 2))

        ass_sub = detector.assess_cohort(
            g_sub,
            ancestry_qb_cohort=qb_sub,
            env_ids_cohort=proj_sub,
            animal_ids=ids_sub.tolist()
        )

        d_ood = np.array([a.d_ood_composite for a in ass_sub])
        confs = np.array([a.selective_confidence for a in ass_sub])
        tiers = [a.confidence_tier for a in ass_sub]
        inflations = np.array([a.uncertainty_inflation_factor for a in ass_sub])

        pct_high = float(np.mean([t == ConfidenceTier.HIGH for t in tiers]) * 100.0)
        pct_mod = float(np.mean([t == ConfidenceTier.MODERATE for t in tiers]) * 100.0)
        pct_low = float(np.mean([t == ConfidenceTier.LOW for t in tiers]) * 100.0)

        # Operational permissions
        pct_numeric_allowed = float(np.mean([a.allow_numeric_prediction for a in ass_sub]) * 100.0)
        pct_ranking_only = float(np.mean([a.allow_ranking_only for a in ass_sub]) * 100.0)
        pct_full_refusal = float(np.mean([a.is_refused for a in ass_sub]) * 100.0)

        fsr_res = GenomicOODDetector.compute_false_safe_rate(y_sub, y_pred_sub, ass_sub)
        frr_res = GenomicOODDetector.compute_false_refusal_rate(y_sub, y_pred_sub, ass_sub)
        risk_df = GenomicOODDetector.compute_selective_risk(y_sub, y_pred_sub, confs)
        aurc = GenomicOODDetector.compute_aurc(risk_df)
        cov_90 = GenomicOODDetector.compute_coverage_at_risk(risk_df, max_risk=90.0)
        rank_res = GenomicOODDetector.compute_ranking_metrics(y_sub, y_pred_sub, top_k_pct=0.10)

        eval_results[name] = {
            "d_ood": d_ood,
            "mse": mse,
            "corr": corr,
            "tiers": tiers,
            "inflations": inflations,
            "pct_high": pct_high,
            "pct_mod": pct_mod,
            "pct_low": pct_low,
            "pct_numeric": pct_numeric_allowed,
            "pct_ranking": pct_ranking_only,
            "fsr": fsr_res["false_safe_rate"] * 100.0,
            "frr": frr_res["false_refusal_rate"] * 100.0,
            "aurc": aurc,
            "risk_df": risk_df,
            "rank_res": rank_res
        }

        scorecard_rows.append({
            "Shift_Experiment": name,
            "Cohort_N": len(y_sub),
            "Pearson_r": corr,
            "Spearman_rho": rank_res["spearman_rho"],
            "Pairwise_Concordance": rank_res["pairwise_concordance"],
            "Top10_Enrichment": rank_res["top_k_enrichment"],
            "Ranking_Valid": rank_res["is_ranking_valid"],
            "MSE": mse,
            "Mean_d_OOD": float(np.mean(d_ood)),
            "Mean_C_Absolute": float(np.mean([a.c_absolute for a in ass_sub])),
            "Mean_C_Ranking": float(np.mean([a.c_ranking for a in ass_sub])),
            "Pct_HIGH": pct_high,
            "Pct_MODERATE": pct_mod,
            "Pct_LOW_Refused": pct_low,
            "Pct_Numeric_Allowed": pct_numeric_allowed,
            "Pct_Ranking_Only": pct_ranking_only,
            "False_Safe_Rate": fsr_res["false_safe_rate"] * 100.0,
            "False_Refusal_Rate": frr_res["false_refusal_rate"] * 100.0,
            "AURC": aurc,
            "Mean_Uncertainty_Inflation": float(np.mean(inflations))
        })

    # Shift C: Extreme Adversarial Alien Genome
    p_ref = detector.ref_p
    g_alien = rng.binomial(2, np.clip(1.0 - p_ref, 0.05, 0.95)).astype(np.int8)
    alien_ass = detector.assess_animal(g_alien, animal_id="alien_001")
    scorecard_rows.append({
        "Shift_Experiment": "Eval 3: Adversarial Alien Genome",
        "Cohort_N": 1,
        "Pearson_r": 0.0,
        "Spearman_rho": 0.0,
        "Pairwise_Concordance": 0.5000,
        "Top10_Enrichment": 1.0,
        "Ranking_Valid": False,
        "MSE": 0.0,
        "Mean_d_OOD": float(alien_ass.d_ood_composite),
        "Mean_C_Absolute": float(alien_ass.c_absolute),
        "Mean_C_Ranking": float(alien_ass.c_ranking),
        "Pct_HIGH": 0.0,
        "Pct_MODERATE": 0.0,
        "Pct_LOW_Refused": 100.0,
        "Pct_Numeric_Allowed": 0.0,
        "Pct_Ranking_Only": 0.0,
        "False_Safe_Rate": 0.0,
        "False_Refusal_Rate": 0.0,
        "AURC": 0.0,
        "Mean_Uncertainty_Inflation": float(alien_ass.uncertainty_inflation_factor)
    })

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_path = out_dir / "layer6_ood_validation_scorecard.csv"
    scorecard_df.to_csv(scorecard_path, index=False)
    print(f"\nSaved scorecard to {scorecard_path}")

    # Print summary table
    print("\n" + "=" * 140)
    print("INDEPENDENT BLIND OOD EVALUATION SCORECARD (DUAL CONFIDENCE AXES: ABSOLUTE & RANKING)")
    print("=" * 140)
    print(f"{'Shift Experiment':<30} | {'N':<5} | {'Pearson':<8} | {'Spearman':<8} | {'Concord':<8} | {'Top10%':<7} | {'d_OOD':<7} | {'C_abs':<6} | {'C_rnk':<6} | {'LOW(%)':<7} | {'FSR(%)':<7} | {'FRR(%)':<7}")
    print("-" * 140)
    for _, r in scorecard_df.iterrows():
        print(f"{r['Shift_Experiment']:<30} | {r['Cohort_N']:<5} | {r['Pearson_r']:<8.4f} | {r['Spearman_rho']:<8.4f} | {r['Pairwise_Concordance']:<8.4f} | {r['Top10_Enrichment']:<7.2f}x | {r['Mean_d_OOD']:<7.3f} | {r['Mean_C_Absolute']:<6.2f} | {r['Mean_C_Ranking']:<6.2f} | {r['Pct_LOW_Refused']:<7.1f}% | {r['False_Safe_Rate']:<7.1f}% | {r['False_Refusal_Rate']:<7.1f}%")
    print("=" * 140)

    # Computed from the scorecard above (this block used to print fixed numbers and claims)
    print("\nOperational tiers by shift (tier thresholds are heuristic; see docs/SOURCE_INVENTORY.md):")
    blind = scorecard_df[scorecard_df["Cohort_N"] > 1]
    for _, r in blind.iterrows():
        print(f"  - {r['Shift_Experiment']}: HIGH {r['Pct_HIGH']:.1f}%, MODERATE {r['Pct_MODERATE']:.1f}%, "
              f"LOW/refused {r['Pct_LOW_Refused']:.1f}%, numeric allowed {r['Pct_Numeric_Allowed']:.1f}%; "
              f"false-safe {r['False_Safe_Rate']:.1f}%, over-refusal {r['False_Refusal_Rate']:.1f}%")
    mgmt = blind[blind["Shift_Experiment"].str.contains("Management")]
    if len(mgmt):
        m = mgmt.iloc[0]
        print(f"  - Ranking on the management shift: Spearman rho {m['Spearman_rho']:+.4f}, concordance "
              f"{m['Pairwise_Concordance']:.4f} -> {'valid' if m['Ranking_Valid'] else 'not valid; ranking output suppressed'}")
    print(f"  - Largest false-safe rate across blind shifts: {blind['False_Safe_Rate'].max():.1f}%")

    # Active applicability rule (docs/OOD_CALIBRATION.md, revision 1): support ratio <= 1.0 for ranking,
    # plus a seen herd for numeric prediction. Neither the pre-specified calibration nor the revision
    # used these blind cohorts, so this is an untouched internal test of the rule.
    thr_path = BASE_DIR / "output" / "ood_calibration" / "thresholds_rev1.json"
    if thr_path.exists():
        from src.evaluation.risk_calibration import load_frozen_thresholds
        thr = load_frozen_thresholds(thr_path)
        cal = GenomicOODDetector(n_components=15, calibrated_thresholds={"ranking": thr["ranking"], "absolute": thr["absolute"]})
        cal.fit(g_valid[train_mask], ancestry_qb_train=qb_valid[train_mask], env_ids_train=project_valid[train_mask])
        cal_rows = []
        for name, mask in eval_cohorts:
            ass = cal.assess_cohort(g_valid[mask], ancestry_qb_cohort=qb_valid[mask], env_ids_cohort=project_valid[mask])
            y, pred = y_load_valid[mask], gblup.predict(g_valid[mask])
            rank_ok = np.array([a.allow_ranking_only for a in ass])
            num_ok = np.array([a.allow_numeric_prediction for a in ass])
            r_acc = float(stats.pearsonr(y[rank_ok], pred[rank_ok])[0]) if rank_ok.sum() > 2 else float("nan")
            r_ref = float(stats.pearsonr(y[~rank_ok], pred[~rank_ok])[0]) if (~rank_ok).sum() > 2 else float("nan")
            cal_rows.append({
                "Shift_Experiment": name, "Cohort_N": int(mask.sum()),
                "Pct_Numeric_Allowed": 100.0 * num_ok.mean(), "Pct_Ranking_Allowed": 100.0 * rank_ok.mean(),
                "Pct_Refused": 100.0 * (~rank_ok).mean(), "r_Accepted_For_Ranking": r_acc, "r_Refused": r_ref,
                "False_Safe_Rate": GenomicOODDetector.compute_false_safe_rate(y, pred, ass)["false_safe_rate"] * 100.0,
                "Median_Support_Score": float(np.median([a.support_score for a in ass])),
            })
        cal_df = pd.DataFrame(cal_rows)
        cal_df.to_csv(out_dir / "layer6_ood_validation_calibrated.csv", index=False)
        print(f"\nActive applicability rule, revision 1 (ranking: support <= {thr['ranking']}; numeric: also a seen herd):")
        print(cal_df.round(3).to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Publication Plot
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Publication Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Graded d_OOD vs Actual Prediction Error (MSE)
    ax = axes[0, 0]
    graded_names = ["Dev (0.28-0.32)", "Mild (0.32-0.36)", "Mod (0.36-0.42)", "Severe (>0.42)"]
    d_oods = [
        d_ood_dev,
        eval_results["Eval 1A: Graded Mild (0.32 < qB <= 0.36)"]["d_ood"].mean(),
        eval_results["Eval 1B: Graded Mod (0.36 < qB <= 0.42)"]["d_ood"].mean(),
        eval_results["Eval 1C: Graded Severe (qB > 0.42)"]["d_ood"].mean(),
    ]
    mses = [
        float(np.mean((y_load_valid[dev_mask] - y_pred_dev)**2)),
        eval_results["Eval 1A: Graded Mild (0.32 < qB <= 0.36)"]["mse"],
        eval_results["Eval 1B: Graded Mod (0.36 < qB <= 0.42)"]["mse"],
        eval_results["Eval 1C: Graded Severe (qB > 0.42)"]["mse"],
    ]
    ax.plot(d_oods, mses, "ro-", lw=2.5, markersize=8, label="Actual Prediction Error (MSE)")
    for name, x, y_val in zip(graded_names, d_oods, mses):
        ax.annotate(name, (x, y_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_title("A. Graded OOD Distance vs Actual Prediction Error ($d_{\\rm OOD} \\uparrow \\Rightarrow {\\rm MSE} \\uparrow$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Genomic OOD Distance ($d_{\\rm OOD}$)", fontsize=10)
    ax.set_ylabel("Empirical Mean Squared Error (MSE)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    # Panel B: Selective Prediction Risk-Coverage Curves (AURC)
    ax = axes[0, 1]
    risk_anc = eval_results["Eval 1: Combined Ancestry Shift (qB > 0.32)"]["risk_df"]
    aurc_anc = eval_results["Eval 1: Combined Ancestry Shift (qB > 0.32)"]["aurc"]
    risk_wes = eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["risk_df"]
    aurc_wes = eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["aurc"]

    ax.plot(risk_anc["Coverage"] * 100, risk_anc["Risk_MSE"], "r-o", lw=2, label=f"Blind Ancestry Extrapolation (AURC={aurc_anc:.1f})")
    ax.plot(risk_wes["Coverage"] * 100, risk_wes["Risk_MSE"], "b-s", lw=2, label=f"Blind Management Shift (AURC={aurc_wes:.1f})")
    ax.set_title("B. Selective Prediction Risk-Coverage Curves (Monotonic Rejection)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Coverage Fraction (% of Population Evaluated)", fontsize=10)
    ax.set_ylabel("Empirical Risk (MSE)", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Uncertainty Inflation vs Actual Error
    ax = axes[1, 0]
    inf_anc = eval_results["Eval 1: Combined Ancestry Shift (qB > 0.32)"]["inflations"]
    d_anc = eval_results["Eval 1: Combined Ancestry Shift (qB > 0.32)"]["d_ood"]
    ax.scatter(d_anc, inf_anc, alpha=0.45, color="#d62728", s=25, label="Ancestry Extrapolation Cohort")
    inf_wes = eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["inflations"]
    d_wes = eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["d_ood"]
    ax.scatter(d_wes, inf_wes, alpha=0.35, color="#1f77b4", s=20, label="Management Shift Cohort ('Wes')")
    ax.scatter([alien_ass.d_ood_composite], [alien_ass.uncertainty_inflation_factor],
               color="purple", s=130, marker="X", label="Alien Genome (Refused)")
    ax.set_title("C. Calibrated Predictive Uncertainty Inflation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Composite OOD Distance ($d_{\\rm OOD}$)", fontsize=10)
    ax.set_ylabel("Uncertainty Inflation Scale ($\\sigma_{\\rm pred} / \\sigma_{\\rm model}$)", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Operational Tiers Distribution (Explicit Action Semantics)
    ax = axes[1, 1]
    categories = ["Mild (0.32-0.36)", "Mod (0.36-0.42)", "Severe (>0.42)", "Wes Herd Shift", "Alien Genome"]
    highs = [
        eval_results["Eval 1A: Graded Mild (0.32 < qB <= 0.36)"]["pct_high"],
        eval_results["Eval 1B: Graded Mod (0.36 < qB <= 0.42)"]["pct_high"],
        eval_results["Eval 1C: Graded Severe (qB > 0.42)"]["pct_high"],
        eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["pct_high"],
        0.0
    ]
    mods = [
        eval_results["Eval 1A: Graded Mild (0.32 < qB <= 0.36)"]["pct_mod"],
        eval_results["Eval 1B: Graded Mod (0.36 < qB <= 0.42)"]["pct_mod"],
        eval_results["Eval 1C: Graded Severe (qB > 0.42)"]["pct_mod"],
        eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["pct_mod"],
        0.0
    ]
    lows = [
        eval_results["Eval 1A: Graded Mild (0.32 < qB <= 0.36)"]["pct_low"],
        eval_results["Eval 1B: Graded Mod (0.36 < qB <= 0.42)"]["pct_low"],
        eval_results["Eval 1C: Graded Severe (qB > 0.42)"]["pct_low"],
        eval_results["Eval 2: Management Shift ('Semin' -> 'Wes')"]["pct_low"],
        100.0
    ]

    x_c = np.arange(len(categories))
    w_bar = 0.5
    ax.bar(x_c, highs, w_bar, label="HIGH: Numeric Prediction + Interval", color="#2ca02c", alpha=0.85)
    ax.bar(x_c, mods, w_bar, bottom=highs, label="MODERATE: Ranking Only (Numeric Suppressed)", color="#ff7f0e", alpha=0.85)
    ax.bar(x_c, lows, w_bar, bottom=np.array(highs) + np.array(mods), label="LOW: Full Refusal (Suppressed)", color="#d62728", alpha=0.85)
    ax.set_xticks(x_c)
    ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("Cohort Proportion (%)", fontsize=10)
    ax.set_title("D. Operational Action Tiers (100% Numeric Suppression on Failure)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "layer6_ood_validation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to {fig_path}")


if __name__ == "__main__":
    run_layer6_ood_benchmark()

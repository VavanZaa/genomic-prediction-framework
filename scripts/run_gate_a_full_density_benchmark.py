"""Gate A: Full-Density (122,486-Marker) BOA vs SNP Genomic Prediction Benchmark.

Compares:
1. M0: Global Ancestry Baseline (y ~ q_Angus)
2. M1_122k: Dense SNP GBLUP (y ~ G_122k)
3. M2_122k: Dense BOA GBLUP (y ~ B_122k)
4. M3_122k: Combined Dense SNP + BOA (y ~ w_G*G_122k + w_B*B_122k)
under strictly prospective canonical validation (N_train=1,754, N_test=658, Zero-Leakage).

Evaluates whether r(SNP + BOA) > r(full SNP) on the maximum feasible matched marker panel.
"""
import sys
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from src.real_data.canonical_split import CanonicalDataSplitter
from src.real_data.reference_grm import load_reference_grm

OUT_DIR = BASE_DIR / "output" / "real_layer3"
GRM_PATH = BASE_DIR / "data" / "real_cattle" / "florida_multibreed" / "florida_grm_122k_v3.npz"


from src.prediction.kernel_cv import cv_kernel_ridge, TWO_KERNEL_WEIGHTS
from src.prediction.boa_models import BOAValidationStats


def paired_bootstrap(y, p_model, p_base, n_bootstraps=1000, seed=42):
    """Paired bootstrap of r(model) - r(baseline) on the same held-out animals."""
    res = BOAValidationStats.paired_bootstrap_test(np.asarray(y), np.asarray(p_model), np.asarray(p_base),
                                                   n_bootstraps=n_bootstraps, random_seed=seed)
    return (float(res.delta_r), float(res.ci_lower), float(res.ci_upper),
            float(res.p_value_one_sided), float(res.p_value_two_sided), bool(res.is_significant))


def optimize_lambda_cv(K_tr, y_tr_cent, n_folds=5):
    return cv_kernel_ridge([K_tr],y_tr_cent,n_folds=n_folds,seed=42)["lambda"]

def optimize_mixture_weights_cv(G_tr,B_tr,y_tr_cent,n_folds=5):
    best=cv_kernel_ridge([G_tr,B_tr],y_tr_cent,TWO_KERNEL_WEIGHTS,n_folds=n_folds,seed=42)
    return *best["weights"],best["lambda"]


def run_full_density_benchmark():
    print("=" * 80)
    print("GATE A: FULL-DENSITY (122,486 MARKERS) BOA VS SNP GENOMIC PREDICTION")
    print("=" * 80)

    if not GRM_PATH.exists():
        raise FileNotFoundError(f"GRM file missing at {GRM_PATH}. Run scripts/build_full_density_122k_grm.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1. Loading Full-Density GRMs (G_122k and B_122k)...]")
    t0 = time.time()
    grm_data = load_reference_grm(GRM_PATH)
    G_122k = grm_data["G_122k"].astype(np.float64)
    B_122k = grm_data["B_122k"].astype(np.float64)
    animal_ids = [str(x) for x in grm_data["animal_ids"]]
    num_markers = int(grm_data["num_markers"])
    print(f"  Loaded G_122k & B_122k for {len(animal_ids)} cattle across {num_markers} markers ({time.time() - t0:.2f}s).")

    traits = ["LoadSlope", "AvgShortLength"]
    results_rows = []
    bootstrap_rows = []

    for trait in traits:
        print(f"\n" + "-" * 80)
        print(f"EVALUATING TRAIT: {trait}")
        print("-" * 80)

        split = CanonicalDataSplitter.get_prospective_split(
            trait_col=trait,
            valid_animal_ids=animal_ids,
            require_joint_traits=True
        )

        id_to_idx = {aid: i for i, aid in enumerate(animal_ids)}
        tr_idx = [id_to_idx[aid] for aid in split.train_ids]
        te_idx = [id_to_idx[aid] for aid in split.test_ids]

        y_tr = split.y_train_adj
        y_te = split.y_test_adj
        mu_tr = float(np.mean(y_tr))
        y_tr_cent = y_tr - mu_tr

        q_tr = split.q_train.reshape(-1, 1)
        q_te = split.q_test.reshape(-1, 1)

        # Slice GRM blocks
        G_tr = G_122k[np.ix_(tr_idx, tr_idx)]
        G_te = G_122k[np.ix_(te_idx, tr_idx)]

        B_tr = B_122k[np.ix_(tr_idx, tr_idx)]
        B_te = B_122k[np.ix_(te_idx, tr_idx)]

        # Model 0: Global Ancestry
        m0 = Ridge(alpha=10.0).fit(q_tr, y_tr)
        p_m0 = m0.predict(q_te)
        r_m0, _ = stats.pearsonr(y_te, p_m0)
        rmse_m0 = np.sqrt(np.mean((y_te - p_m0)**2))

        # Model 1: Dense SNP GBLUP (M1_122k)
        lam_g = optimize_lambda_cv(G_tr, y_tr_cent)
        alpha_g = np.linalg.solve(G_tr + lam_g * np.eye(len(tr_idx)), y_tr_cent)
        p_m1 = mu_tr + G_te @ alpha_g
        r_m1, _ = stats.pearsonr(y_te, p_m1)
        rmse_m1 = np.sqrt(np.mean((y_te - p_m1)**2))

        # Model 2: Dense BOA GBLUP (M2_122k)
        lam_b = optimize_lambda_cv(B_tr, y_tr_cent)
        alpha_b = np.linalg.solve(B_tr + lam_b * np.eye(len(tr_idx)), y_tr_cent)
        p_m2 = mu_tr + B_te @ alpha_b
        r_m2, _ = stats.pearsonr(y_te, p_m2)
        rmse_m2 = np.sqrt(np.mean((y_te - p_m2)**2))

        # Model 3: Combined Dense SNP + BOA (M3_122k)
        w_g, w_b, lam_mix = optimize_mixture_weights_cv(G_tr, B_tr, y_tr_cent)
        K_mix_tr = w_g * G_tr + w_b * B_tr
        K_mix_te = w_g * G_te + w_b * B_te
        alpha_mix = np.linalg.solve(K_mix_tr + lam_mix * np.eye(len(tr_idx)), y_tr_cent)
        p_m3 = mu_tr + K_mix_te @ alpha_mix
        r_m3, _ = stats.pearsonr(y_te, p_m3)
        rmse_m3 = np.sqrt(np.mean((y_te - p_m3)**2))

        print(f"  M0 (Ancestry):          r = {r_m0:.4f}, RMSE = {rmse_m0:.3f}")
        print(f"  M1 (Dense 122k SNP):    r = {r_m1:.4f}, RMSE = {rmse_m1:.3f} (opt lambda = {lam_g})")
        print(f"  M2 (Dense 122k BOA):    r = {r_m2:.4f}, RMSE = {rmse_m2:.3f} (opt lambda = {lam_b})")
        print(f"  M3 (Combined 122k Mix): r = {r_m3:.4f}, RMSE = {rmse_m3:.3f} (weights: w_G={w_g:.2f}, w_B={w_b:.2f}, lam={lam_mix})")

        results_rows.append({
            "Trait": trait,
            "M0_Ancestry_r": round(float(r_m0), 4),
            "M1_Dense_SNP_r": round(float(r_m1), 4),
            "M2_Dense_BOA_r": round(float(r_m2), 4),
            "M3_Combined_r": round(float(r_m3), 4),
            "Best_Kernel_Weights": f"w_G={w_g:.2f}, w_B={w_b:.2f}",
            "SNP_Lambda": lam_g,
            "BOA_Lambda": lam_b,
            "Mix_Lambda": lam_mix
        })

        # Paired Bootstrap
        # M3 vs M1
        d_m3_m1, ci_l_31, ci_u_31, p1_31, p2_31, sig_31 = paired_bootstrap(y_te, p_m3, p_m1)
        bootstrap_rows.append({
            "Trait": trait,
            "Comparison": "M3 (Combined 122k) vs M1 (Dense SNP)",
            "Delta_r": round(d_m3_m1, 4),
            "CI_95_Low": round(ci_l_31, 4),
            "CI_95_High": round(ci_u_31, 4),
            "p_one_sided": round(p1_31, 4),
            "p_two_sided": round(p2_31, 4),
            "Is_Significant_2s": sig_31
        })

        # M2 vs M1
        d_m2_m1, ci_l_21, ci_u_21, p1_21, p2_21, sig_21 = paired_bootstrap(y_te, p_m2, p_m1)
        bootstrap_rows.append({
            "Trait": trait,
            "Comparison": "M2 (Dense BOA) vs M1 (Dense SNP)",
            "Delta_r": round(d_m2_m1, 4),
            "CI_95_Low": round(ci_l_21, 4),
            "CI_95_High": round(ci_u_21, 4),
            "p_one_sided": round(p1_21, 4),
            "p_two_sided": round(p2_21, 4),
            "Is_Significant_2s": sig_21
        })

        # M3 vs M0
        d_m3_m0, ci_l_30, ci_u_30, p1_30, p2_30, sig_30 = paired_bootstrap(y_te, p_m3, p_m0)
        bootstrap_rows.append({
            "Trait": trait,
            "Comparison": "M3 (Combined 122k) vs M0 (Ancestry)",
            "Delta_r": round(d_m3_m0, 4),
            "CI_95_Low": round(ci_l_30, 4),
            "CI_95_High": round(ci_u_30, 4),
            "p_one_sided": round(p1_30, 4),
            "p_two_sided": round(p2_30, 4),
            "Is_Significant_2s": sig_30
        })

    df_bench = pd.DataFrame(results_rows)
    df_boot = pd.DataFrame(bootstrap_rows)

    print("\n" + "=" * 90)
    print("GATE A: FULL-DENSITY (122,486 MARKERS) BENCHMARK SUMMARY:")
    print("=" * 90)
    print(df_bench.to_string(index=False))

    print("\n" + "=" * 90)
    print("GATE A: PAIRED BOOTSTRAP SIGNIFICANCE (1,000 DRAWS):")
    print("=" * 90)
    print(df_boot.to_string(index=False))

    bench_csv = OUT_DIR / "gate_a_full_density_benchmark.csv"
    boot_csv = OUT_DIR / "gate_a_full_density_bootstrap.csv"
    df_bench.to_csv(bench_csv, index=False)
    df_boot.to_csv(boot_csv, index=False)
    print(f"\nSaved benchmark tables to:\n  {bench_csv}\n  {boot_csv}")


if __name__ == "__main__":
    run_full_density_benchmark()

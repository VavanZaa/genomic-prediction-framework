"""Master runner for Layer 3: Real Genomic Prediction on Real Cattle Dataset.

Evaluates out-of-sample genomic prediction on the University of Florida
Angus-Brahman Multibreed Beef Herd (3,194 real cattle, 122,486 mapped SNPs).
Tests Ancestry-Only baseline, Additive GBLUP, and Non-linear RKHS on withheld
prospective cohort calves (Years 2019-2022) trained on historical calves (Years 2016-2018).
Generates publication-ready diagnostic plots and exports summary tables.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.florida_loader import FloridaLoader
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.prediction.uncertainty import residual_covariance
from src.prediction.models import AdditiveGBLUP, RkhsPredictor


def run_layer3_genomic_prediction():
    print("=" * 80)
    print("LAYER 3: REAL GENOMIC PREDICTION BENCHMARK ON PHYSICAL CATTLE")
    print("  Dataset: University of Florida Angus-Brahman Multibreed Herd")
    print("=" * 80)

    out_dir = BASE_DIR / "output" / "real_layer3"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Load dataset with 5,104 SNPs across 29 bovine autosomes
    data = FloridaLoader.load_dataset(marker_step=24, min_maf=0.03)
    pheno_df = data.phenotypes
    geno = data.genotypes
    ancestry = data.ancestry_matrix
    marker_map = data.marker_map

    print(f"\n[1. Ingestion & Alignment]")
    print(f"  Total matched cattle:   {data.num_animals}")
    print(f"  Autosomal SNPs loaded:  {data.num_markers} (spanning BTA 1 to 29)")
    print(f"  Genotype matrix shape:  {geno.shape} ({geno.nbytes / (1024*1024):.1f} MB)")
    print(f"  Ingestion elapsed time: {time.time() - t0:.2f}s")

    # 2. Canonical Prospective Cohort Partitioning & Zero-Leakage Adjustment
    print(f"\n[2. Canonical Prospective Partitioning & Zero-Leakage Adjustment]")
    from src.real_data.canonical_split import CanonicalDataSplitter
    
    split_load = CanonicalDataSplitter.get_prospective_split(
        trait_col="LoadSlope",
        pheno_df=data.phenotypes,
        valid_animal_ids=data.animal_ids,
        require_joint_traits=True
    )
    split_hair = CanonicalDataSplitter.get_prospective_split(
        trait_col="AvgShortLength",
        pheno_df=data.phenotypes,
        valid_animal_ids=data.animal_ids,
        require_joint_traits=True
    )

    id_to_geno_idx = {aid: idx for idx, aid in enumerate(data.animal_ids)}
    train_geno_idx = [id_to_geno_idx[aid] for aid in split_load.train_ids]
    test_geno_idx = [id_to_geno_idx[aid] for aid in split_load.test_ids]

    train_geno, test_geno = geno[train_geno_idx], geno[test_geno_idx]
    train_anc = split_load.q_train.reshape(-1, 1)
    test_anc = split_load.q_test.reshape(-1, 1)

    y_train_load, y_test_load = split_load.y_train_adj, split_load.y_test_adj
    y_train_hair, y_test_hair = split_hair.y_train_adj, split_hair.y_test_adj

    n_train = len(split_load.train_ids)
    n_test = len(split_load.test_ids)

    print(f"  Training Cohort (Birth Years 2016-2018): {n_train} animals")
    print(f"  Prospective Test Cohort (Birth Years 2019, 2021-2022): {n_test} animals")
    print(f"  Training Environmental R2: LoadSlope={split_load.r2_train_env:.3f}, AvgShortLength={split_hair.r2_train_env:.3f}")

    # Subtropical breeding targets (Lower load slope & shorter slick hair coat)
    tau_load = float(np.median(y_train_load))
    tau_hair = float(np.median(y_train_hair))
    realized_joint = ((y_test_load <= tau_load) & (y_test_hair <= tau_hair)).astype(int)
    realized_joint_pct = float(np.mean(realized_joint) * 100)
    print(f"  Subtropical Target Criteria: LoadSlope <= {tau_load:.2f} AND HairLength <= {tau_hair:.2f}")
    print(f"  Realized Withheld Joint Attainment: {realized_joint_pct:.2f}% ({np.sum(realized_joint)}/{n_test})")

    # 4. Model Training & Out-of-Sample Prediction
    print(f"\n[4. Fitting Prediction Models & Evaluating on Withheld Real Calves...]")

    # Model 1: Ancestry Only
    m_anc_l = Ridge(alpha=10.0).fit(train_anc, y_train_load)
    p_anc_l = m_anc_l.predict(test_anc)
    m_anc_h = Ridge(alpha=10.0).fit(train_anc, y_train_hair)
    p_anc_h = m_anc_h.predict(test_anc)

    # Model 2: Additive GBLUP
    m_gb_l = AdditiveGBLUP().fit(train_geno, y_train_load)
    p_gb_l = m_gb_l.predict(test_geno)
    m_gb_h = AdditiveGBLUP().fit(train_geno, y_train_hair)
    p_gb_h = m_gb_h.predict(test_geno)

    # Model 3: Non-linear RKHS
    m_rk_l = RkhsPredictor().fit(train_geno, y_train_load)
    p_rk_l = m_rk_l.predict(test_geno)
    m_rk_h = RkhsPredictor().fit(train_geno, y_train_hair)
    p_rk_h = m_rk_h.predict(test_geno)

    model_dict = {
        "AncestryOnly": (p_anc_l, p_anc_h, m_anc_l, m_anc_h, train_anc),
        "AdditiveGBLUP": (p_gb_l, p_gb_h, m_gb_l, m_gb_h, train_geno),
        "NonlinearRKHS": (p_rk_l, p_rk_h, m_rk_l, m_rk_h, train_geno),
    }

    benchmark_rows = []

    for name, (p_l, p_h, m_l, m_h, X_tr) in model_dict.items():
        # Accuracy on LoadSlope
        r_l, _ = stats.pearsonr(y_test_load, p_l)
        rho_l, _ = stats.spearmanr(y_test_load, p_l)
        rmse_l = float(np.sqrt(np.mean((y_test_load - p_l) ** 2)))

        # Accuracy on HairLength
        r_h, _ = stats.pearsonr(y_test_hair, p_h)
        rho_h, _ = stats.spearmanr(y_test_hair, p_h)
        rmse_h = float(np.sqrt(np.mean((y_test_hair - p_h) ** 2)))

        # Entirely held-out training groups for predictive (not fitted) residuals.
        factory = (lambda: Ridge(alpha=10.0)) if name == "AncestryOnly" else type(m_l)
        groups = split_load.df_aligned.iloc[split_load.train_indices]["Set"].to_numpy()
        cov_res = residual_covariance(X_tr,y_train_load,y_train_hair,factory,groups)


        # Joint probability estimation
        joint_probs = []
        for i in range(len(p_l)):
            mvn = multivariate_normal(mean=[p_l[i], p_h[i]], cov=cov_res, allow_singular=True)
            prob = mvn.cdf([tau_load, tau_hair])
            joint_probs.append(prob)
        joint_probs = np.array(joint_probs)

        brier = float(brier_score_loss(realized_joint, joint_probs))
        auc = float(roc_auc_score(realized_joint, joint_probs))
        mean_pred_prob = float(np.mean(joint_probs) * 100)

        benchmark_rows.append({
            "Model": name,
            "LoadSlope r": round(float(r_l), 4),
            "LoadSlope rho": round(float(rho_l), 4),
            "LoadSlope RMSE": round(rmse_l, 3),
            "HairLength r": round(float(r_h), 4),
            "HairLength rho": round(float(rho_h), 4),
            "HairLength RMSE": round(rmse_h, 3),
            "Joint Pred Attain (%)": round(mean_pred_prob, 2),
            "Joint Real Attain (%)": round(realized_joint_pct, 2),
            "Brier Score": round(brier, 4),
            "ROC-AUC": round(auc, 3),
        })

    df_results = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 80)
    print("REAL GENOMIC PREDICTION BENCHMARK RESULTS (WITHHELD COHORT 2019-2022):")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Export summary CSVs
    csv_path = out_dir / "layer3_florida_model_benchmark.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nExported benchmark table to: {csv_path}")

    # -------------------------------------------------------------
    # 5. Diagnostic Visualizations
    # -------------------------------------------------------------
    print(f"\n[5. Generating Layer 3 Diagnostic Figures...]")

    # Figure 1: Florida Herd Ancestry & Allele Frequency Spectrum
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    axes1[0].hist(pheno_df["AngusGBC"] * 100, bins=25, color="#1e3a8a", alpha=0.75, edgecolor="black")
    axes1[0].axvline(62.5, color="#dc2626", linestyle="--", lw=2, label="Brangus Standard (62.5% Angus)")
    axes1[0].axvline(np.mean(pheno_df["AngusGBC"]) * 100, color="#10b981", lw=2, label=f"Herd Mean ({np.mean(pheno_df['AngusGBC'])*100:.1f}%)")
    axes1[0].set_xlabel("Angus Genomic Breed Composition (%)")
    axes1[0].set_ylabel("Number of Real Cattle")
    axes1[0].set_title("(A) Florida Herd Genomic Breed Spectrum (N=3,194)")
    axes1[0].legend()
    axes1[0].grid(axis="y", linestyle="--", alpha=0.4)

    p_freq = np.mean(geno, axis=0) / 2.0
    maf = np.minimum(p_freq, 1.0 - p_freq)
    axes1[1].hist(maf, bins=25, color="#7c3aed", alpha=0.75, edgecolor="black")
    axes1[1].axvline(np.mean(maf), color="#f59e0b", lw=2, label=f"Mean MAF = {np.mean(maf):.3f}")
    axes1[1].set_xlabel("Minor Allele Frequency (MAF)")
    axes1[1].set_ylabel("Number of SNP Markers")
    axes1[1].set_title(f"(B) Autosomal SNP Spectrum (M={data.num_markers} on BTA 1-29)")
    axes1[1].legend()
    axes1[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig1_path = out_dir / "florida_ancestry_spectrum.png"
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Environmental Adjustment vs Raw Phenotypes
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    var_raw_l = float(np.var(split_load.y_train_raw))
    var_adj_l = float(np.var(split_load.y_train_adj))
    axes2[0].hist(split_load.y_train_raw, bins=25, alpha=0.55, color="#ef4444", label=f"Raw LoadSlope (Var={var_raw_l:.1f})")
    axes2[0].hist(split_load.y_train_adj, bins=25, alpha=0.65, color="#2563eb", label=f"Adjusted LoadSlope (Var={var_adj_l:.1f})")
    axes2[0].set_xlabel("Thermal Load Slope (°C·hr / ambient load)")
    axes2[0].set_ylabel("Count")
    axes2[0].set_title(f"(A) Thermal Load Slope Adjustment (R²_env = {split_load.r2_train_env:.3f})")
    axes2[0].legend()
    axes2[0].grid(linestyle="--", alpha=0.4)

    var_raw_h = float(np.var(split_hair.y_train_raw))
    var_adj_h = float(np.var(split_hair.y_train_adj))
    axes2[1].hist(split_hair.y_train_raw, bins=25, alpha=0.55, color="#f59e0b", label=f"Raw Hair Length (Var={var_raw_h:.2f})")
    axes2[1].hist(split_hair.y_train_adj, bins=25, alpha=0.65, color="#059669", label=f"Adjusted Hair Length (Var={var_adj_h:.2f})")
    axes2[1].set_xlabel("Hair Coat Length (mm)")
    axes2[1].set_ylabel("Count")
    axes2[1].set_title(f"(B) Hair Coat Length Adjustment (R²_env = {split_hair.r2_train_env:.3f})")
    axes2[1].legend()
    axes2[1].grid(linestyle="--", alpha=0.4)

    fig2_path = out_dir / "florida_environmental_adjustment.png"
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    print(f"  Saved: {fig2_path}")

    # Figure 3: Out-of-Sample Genomic Prediction Benchmark
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    models = df_results["Model"].tolist()
    r_load = df_results["LoadSlope r"].tolist()
    r_hair = df_results["HairLength r"].tolist()
    brier_scores = df_results["Brier Score"].tolist()

    x = np.arange(len(models))
    w = 0.35
    axes3[0].bar(x - w/2, r_load, w, label="LoadSlope Accuracy (r)", color="#3b82f6")
    axes3[0].bar(x + w/2, r_hair, w, label="HairLength Accuracy (r)", color="#10b981")
    axes3[0].set_xticks(x)
    axes3[0].set_xticklabels(models)
    axes3[0].set_ylabel("Pearson Correlation (r) on Unseen Calves")
    axes3[0].set_title("(A) Real Genomic vs Ancestry Out-of-Sample Accuracy")
    axes3[0].legend()
    axes3[0].grid(axis="y", linestyle="--", alpha=0.4)
    for idx in range(len(models)):
        axes3[0].text(idx - w/2, r_load[idx] + 0.005, f"{r_load[idx]:.3f}", ha="center", fontsize=8, fontweight="bold")
        axes3[0].text(idx + w/2, r_hair[idx] + 0.005, f"{r_hair[idx]:.3f}", ha="center", fontsize=8, fontweight="bold")

    axes3[1].bar(models, brier_scores, color=["#94a3b8", "#6366f1", "#8b5cf6"], width=0.5)
    axes3[1].axhline(brier_score_loss(realized_joint, np.full(len(realized_joint), np.mean(realized_joint))),
                     color="#dc2626", linestyle="--", label="Climatological Reference Brier")
    axes3[1].set_ylabel("Joint Target Brier Score (Lower is Better)")
    axes3[1].set_title("(B) Joint Attainment Probability Calibration")
    axes3[1].legend()
    axes3[1].grid(axis="y", linestyle="--", alpha=0.4)
    for idx, b in enumerate(brier_scores):
        axes3[1].text(idx, b + 0.004, f"{b:.4f}", ha="center", fontweight="bold")

    fig3_path = out_dir / "florida_genomic_prediction_benchmark.png"
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()
    print(f"  Saved: {fig3_path}")


    print("\n" + "=" * 80)
    print("LAYER 3 REAL GENOMIC PREDICTION BENCHMARK COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_layer3_genomic_prediction()

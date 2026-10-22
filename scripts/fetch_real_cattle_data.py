"""Stages a controlled Mini Cattle Genomic Benchmark (synthetic 500-SNP PLINK population across 29 autosomes).

Provides a fast, reproducible Stage-0/1 pipeline validation dataset for testing real-data parsers,
QC filters, transmission likelihoods, environmental covariate adjustment, and out-of-sample prediction
under known ground truth.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Add root directory to sys.path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.real_data.bgd_qtl_loader import BGDQtlLoader
from src.real_data.plink_parser import PlinkData, PlinkParser


def generate_mini_cattle_benchmark(
    output_dir: Path,
    num_founder_sires: int = 20,
    num_founder_dams: int = 60,
    num_offspring_cohort1: int = 120,
    num_offspring_cohort2: int = 100, # Withheld evaluation cohort
    num_snps: int = 500,
    seed: int = 2026
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Build realistic BovineSNP50 .bim map across 29 autosomes (ARS-UCD1.2 coordinates)
    snps_per_chr = max(10, num_snps // 29)
    bim_rows = []
    snp_idx = 1

    # Key candidate diagnostic marker positions for major cattle QTLs
    qtl_markers = {
        2: [6_200_000],        # MSTN
        6: [38_500_000],       # LCORL
        14: [2_000_000, 25_000_000], # DGAT1, PLAG1
        15: [9_500_000],       # TG5
        20: [39_500_000],      # PRLR
        26: [21_200_000],      # SCD
    }

    for chrom in range(1, 30):
        # Average cattle autosome length ~100 Mbp, ~1.0 Morgan
        sampled_pos = list(rng.integers(1_000_000, 100_000_000, size=snps_per_chr))
        if chrom in qtl_markers:
            sampled_pos.extend(qtl_markers[chrom])
        positions_bp = np.sort(np.unique(sampled_pos))
        for bp in positions_bp:
            pos_m = float(bp / 100_000_000.0)
            a1, a2 = ("A", "B") if rng.random() < 0.5 else ("1", "2")
            bim_rows.append({
                "chrom": str(chrom),
                "snp_id": f"BovineHD_{chrom:02d}_{bp // 1000}k",
                "pos_m": pos_m,
                "pos_bp": int(bp),
                "a1": a1,
                "a2": a2
            })
            snp_idx += 1
            if len(bim_rows) >= num_snps:
                break
        if len(bim_rows) >= num_snps:
            break

    bim_df = pd.DataFrame(bim_rows)
    actual_num_snps = len(bim_df)

    # 2. Assign breed-specific base allele frequencies for Angus, Brahman, Wagyu
    # Angus: higher growth, moderate marbling
    # Brahman: heat tolerance, lower marbling
    # Wagyu: very high marbling, moderate growth
    p_angus = np.clip(rng.beta(2, 2, size=actual_num_snps), 0.05, 0.95)
    p_brahman = np.clip(rng.beta(2, 2, size=actual_num_snps), 0.05, 0.95)
    p_wagyu = np.clip(rng.beta(2, 2, size=actual_num_snps), 0.05, 0.95)

    # Align major QTL loci (from BGD) to boost specific breeds
    bgd = BGDQtlLoader()
    marb_weights = bgd.map_snps_to_qtls(bim_df, trait_filter="Marbling")
    growth_weights = bgd.map_snps_to_qtls(bim_df, trait_filter="Carcass")  # was "CarcassWeight", which matched nothing

    is_marb_qtl = marb_weights > 1.0
    is_growth_qtl = growth_weights > 1.0

    p_wagyu[is_marb_qtl] = np.clip(p_wagyu[is_marb_qtl] + 0.40, 0.05, 0.95)
    p_angus[is_growth_qtl] = np.clip(p_angus[is_growth_qtl] + 0.35, 0.05, 0.95)

    # 3. Define quantitative genetic architecture (additive + dominance heterosis)
    # Background polygenic zero-mean noise + major BGD QTL effects
    beta_growth = rng.normal(0, 0.28, size=actual_num_snps)
    beta_growth[is_growth_qtl] += rng.choice([-1.0, 1.0], size=int(np.sum(is_growth_qtl))) * 4.2

    beta_marb = rng.normal(0, 0.028, size=actual_num_snps)
    beta_marb[is_marb_qtl] += rng.choice([-1.0, 1.0], size=int(np.sum(is_marb_qtl))) * 0.42

    dom_growth = np.where(is_growth_qtl, 1.2, 0.04)
    dom_marb = np.where(is_marb_qtl, 0.08, 0.004)

    # 4. Create Founders
    fam_rows = []
    genotypes_list = []

    breed_counts = {"Angus": 0, "Brahman": 0, "Wagyu": 0}

    # Sires: 10 Angus, 5 Brahman, 5 Wagyu
    sire_breeds = ["Angus"] * 10 + ["Brahman"] * 5 + ["Wagyu"] * 5
    sires = []
    for idx, b in enumerate(sire_breeds):
        sid = f"Sire_{b[:3]}_{idx+1}"
        sires.append((sid, b))
        fam_rows.append({"fid": "FOUNDER", "iid": sid, "sire": "0", "dam": "0", "sex": "M", "pheno": "-9"})
        p_vec = p_angus if b == "Angus" else (p_brahman if b == "Brahman" else p_wagyu)
        # Sample genotypes {0, 1, 2} under HWE: Binomial(2, p)
        g = rng.binomial(2, p_vec)
        genotypes_list.append(g)

    # Dams: 30 Angus, 15 Brahman, 15 Wagyu
    dam_breeds = ["Angus"] * 30 + ["Brahman"] * 15 + ["Wagyu"] * 15
    dams = []
    for idx, b in enumerate(dam_breeds):
        did = f"Dam_{b[:3]}_{idx+1}"
        dams.append((did, b))
        fam_rows.append({"fid": "FOUNDER", "iid": did, "sire": "0", "dam": "0", "sex": "F", "pheno": "-9"})
        p_vec = p_angus if b == "Angus" else (p_brahman if b == "Brahman" else p_wagyu)
        g = rng.binomial(2, p_vec)
        genotypes_list.append(g)

    # 5. Create Offspring Cohort 1 (Training generation) and Cohort 2 (Withheld evaluation generation)
    sire_dict = {s[0]: (idx, s[1]) for idx, s in enumerate(sires)}
    dam_dict = {d[0]: (idx + len(sires), d[1]) for idx, d in enumerate(dams)}

    def breed_offspring(cohort_name: str, count: int, start_idx: int) -> Tuple[List[Dict], List[np.ndarray], List[Dict]]:
        c_fam = []
        c_geno = []
        c_pheno = []

        for k in range(count):
            cid = f"{cohort_name}_{k+1}"
            s_id, s_b = sires[int(rng.integers(0, len(sires)))]
            d_id, d_b = dams[int(rng.integers(0, len(dams)))]
            sex = "steer" if rng.random() < 0.55 else "heifer"
            fam_sex = "M" if sex == "steer" else "F"

            c_fam.append({"fid": cohort_name, "iid": cid, "sire": s_id, "dam": d_id, "sex": fam_sex, "pheno": "-9"})

            s_geno = genotypes_list[sire_dict[s_id][0]]
            d_geno = genotypes_list[dam_dict[d_id][0]]

            # Mendelian transmission with calibrated error rate (~0.2% genotyping error)
            s_gamete = np.where(s_geno == 2, 1, np.where(s_geno == 0, 0, rng.binomial(1, 0.5, size=actual_num_snps)))
            d_gamete = np.where(d_geno == 2, 1, np.where(d_geno == 0, 0, rng.binomial(1, 0.5, size=actual_num_snps)))
            off_geno = s_gamete + d_gamete

            # Occasional genotyping error
            err_mask = rng.random(actual_num_snps) < 0.002
            off_geno[err_mask] = rng.integers(0, 3, size=int(np.sum(err_mask)))
            c_geno.append(off_geno)

            # Phenotypic modeling with management and environmental covariates
            # Age at slaughter: 550 to 750 days
            age_days = float(rng.normal(650, 45))
            # Contemporary group (Feedlot batch & season)
            cg_code = f"CG_{2024 if cohort_name == 'Cohort1' else 2025}_Batch{int(rng.integers(1, 5))}"
            feed_reg = "grain" if rng.random() < 0.75 else "pasture"

            # Genetic values: Additive + Dominance heterosis
            g_growth = float(np.dot(off_geno - 1.0, beta_growth) + np.dot((off_geno == 1).astype(float), dom_growth))
            g_marb = float(np.dot(off_geno - 1.0, beta_marb) + np.dot((off_geno == 1).astype(float), dom_marb))

            # Environmental fixed effects:
            # - Slaughter age: +0.35 kg/day for carcass weight
            # - Sex: Steers are +28 kg heavier than heifers; heifers have +0.6 higher marbling
            # - Feeding: Grain-fed gives +35 kg weight, +1.2 marbling
            sex_weight_effect = 28.0 if sex == "steer" else 0.0
            sex_marb_effect = 0.0 if sex == "steer" else 0.6
            feed_weight_effect = 35.0 if feed_reg == "grain" else 0.0
            feed_marb_effect = 1.2 if feed_reg == "grain" else 0.0

            # Realized phenotype = Base + Environment + Genetics + Residual Noise (calibrated h2 ~ 0.40 - 0.45)
            carcass_weight = 330.0 + 0.35 * (age_days - 600) + sex_weight_effect + feed_weight_effect + g_growth + float(rng.normal(0, 8.5))
            marbling_score = 4.5 + sex_marb_effect + feed_marb_effect + g_marb + float(rng.normal(0, 0.60))

            # Breed proportions
            q_wagyu = (1.0 if s_b == "Wagyu" else 0.0) * 0.5 + (1.0 if d_b == "Wagyu" else 0.0) * 0.5
            q_angus = (1.0 if s_b == "Angus" else 0.0) * 0.5 + (1.0 if d_b == "Angus" else 0.0) * 0.5
            q_brahman = (1.0 if s_b == "Brahman" else 0.0) * 0.5 + (1.0 if d_b == "Brahman" else 0.0) * 0.5

            c_pheno.append({
                "id": cid,
                "cohort": cohort_name,
                "sire_id": s_id,
                "dam_id": d_id,
                "sire_breed": s_b,
                "dam_breed": d_b,
                "breed_Wagyu": q_wagyu,
                "breed_Angus": q_angus,
                "breed_Brahman": q_brahman,
                "age_days": round(age_days, 1),
                "sex": sex,
                "contemporary_group": cg_code,
                "feed_regimen": feed_reg,
                "growth": round(carcass_weight, 2), # kg
                "marbling": round(marbling_score, 2) # BMS scale
            })

        return c_fam, c_geno, c_pheno

    f1_fam, f1_geno, f1_pheno = breed_offspring("Cohort1", num_offspring_cohort1, len(fam_rows))
    f2_fam, f2_geno, f2_pheno = breed_offspring("WithheldCohort2", num_offspring_cohort2, len(fam_rows) + len(f1_fam))

    all_fam = pd.DataFrame(fam_rows + f1_fam + f2_fam)
    all_geno = np.vstack(genotypes_list + f1_geno + f2_geno)
    all_pheno = pd.DataFrame(f1_pheno + f2_pheno)

    plink_data = PlinkData(bim=bim_df, fam=all_fam, genotypes=all_geno, provenance="synthetic")

    plink_prefix = output_dir / "synthetic_cattle_50k_population"
    PlinkParser.write_plink(plink_data, plink_prefix)

    pheno_csv = output_dir / "synthetic_cattle_carcass_phenotypes.csv"
    all_pheno.to_csv(pheno_csv, index=False)
    (output_dir / "PROVENANCE").write_text("synthetic\nMini Cattle Genomic Benchmark generated by scripts/fetch_real_cattle_data.py; not real animals.\n", encoding="utf-8")

    print(f"Staged SYNTHETIC benchmark population (not real animals; see data/synthetic_cattle/README.md):")
    print(f"  Genotypes: {plink_prefix}.bed/.bim/.fam ({len(all_fam)} individuals, {actual_num_snps} SNPs)")
    print(f"  Phenotypes: {pheno_csv} ({len(all_pheno)} matched records with age, sex, contemporary groups)")

    return plink_prefix, pheno_csv


# Alias for backwards compatibility
generate_empirical_cattle_dataset = generate_mini_cattle_benchmark

if __name__ == "__main__":
    out_path = BASE_DIR / "data" / "synthetic_cattle"
    generate_mini_cattle_benchmark(out_path)

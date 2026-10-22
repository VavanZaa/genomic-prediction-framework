"""Layer 5B-T: how well do simulations from the real parents recover the real calf?

Real data: 171 genotyped Spanish beef-cattle trios (7 breeds, BovineHD, BEAGLE 3 trio phasing).
For every trio, only the two parents are used to predict the calf; the real calf (reconstructed from
the parents' transmitted haplotypes, as the BEAGLE 3.3 manual specifies) is then revealed.

Reports, per breed and overall:
1. Locus-wise recovery: log-likelihood per locus, genotype-call accuracy, and the correlation between
   mid-parent and calf genotype deviations, against breed allele frequencies and against the wrong parents.
2. Mendelian segregation ratios observed across all trios.
3. Calibration of the simulated offspring distribution: where the real calf falls among calves
   simulated from its parents, for heterozygosity, relationship to a parent, and the genetic value of
   20 synthetic additive traits placed on the real genomes (known effects, so the calf's value is known).
4. The constant Mendelian-sampling approximation 0.5 (1 - F) sigma_A^2 used by Layer 4A, and the old
   Layer 5B-P crossover rate (one extra crossover per chromosome), as comparisons.

Outputs (output/real_layer5/): layer5b_trio_recovery_per_trio.csv, _summary.csv, _segregation.csv,
_midparent.csv, and layer5b_trio_recovery.png.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.trio_recovery import BREEDS, analyze_panel, load_trio_panel, midparent_accuracy, segregation_table

DATA_DIR = BASE_DIR / "data" / "real_cattle" / "spanish_trios"
OUT_DIR = BASE_DIR / "output" / "real_layer5"
MARKER_STEP = 4
SIMULATED_CALVES = 60
N_TRAITS, N_QTL = 20, 1000


def coverage(pit: pd.Series) -> float:
    return float(np.mean((pit >= 0.05) & (pit <= 0.95)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_trio, seg = [], []
    for name, slug, breed_id in BREEDS:
        t0 = time.time()
        panel = load_trio_panel(DATA_DIR / f"{slug}.tar.gz", name, breed_id, step=MARKER_STEP)
        df = analyze_panel(panel, k=SIMULATED_CALVES, n_traits=N_TRAITS, n_qtl=N_QTL, seed=breed_id)
        old = analyze_panel(panel, k=SIMULATED_CALVES, n_traits=N_TRAITS, n_qtl=N_QTL, seed=breed_id, extra_crossovers=1.0)
        df["gv_cover90_sim_old_rate"] = old["gv_cover90_sim"].to_numpy()
        df["gv_z_sim_sd_old_rate"] = old["gv_z_sim_sd"].to_numpy()
        df["het_pit_old_rate"] = old["het_pit"].to_numpy()
        per_trio.append(df)
        seg.append(segregation_table(panel))
        print(f"  {name:<26} {panel.n_trios:3d} trios, {panel.haplotypes.shape[0]:7d} markers ({time.time() - t0:.0f}s)")

    trios = pd.concat(per_trio, ignore_index=True)
    mid = midparent_accuracy(trios)
    seg_df = pd.concat(seg, ignore_index=True)

    def summarize(sub: pd.DataFrame, label: str) -> dict:
        breeds = sub["breed"].unique()
        return {
            "breed": label,
            "trios": len(sub),
            "loci_per_trio": int(sub["loci"].mean()),
            "deviation_r_parents": sub["deviation_r_parents"].mean(),
            "call_accuracy_parents": sub["call_accuracy_parents"].mean(),
            "call_accuracy_breed": sub["call_accuracy_breed"].mean(),
            "loglik_parents": sub["loglik_per_locus_parents"].mean(),
            "loglik_breed": sub["loglik_per_locus_breed"].mean(),
            "loglik_wrong_parents": sub["loglik_per_locus_wrong_parents"].mean(),
            "het_cover90": coverage(sub["het_pit"]),
            "het_cover90_old_rate": coverage(sub["het_pit_old_rate"]),
            "rel_parent_cover90": coverage(sub["rel_parent1_pit"]),
            "rel_parent_real_mean": sub["rel_parent1_real"].mean(),
            "gv_cover90_sim": sub["gv_cover90_sim"].mean(),
            "gv_cover90_sim_old_rate": sub["gv_cover90_sim_old_rate"].mean(),
            "gv_cover90_const": sub["gv_cover90_const"].mean(),
            "gv_z_sd_sim": sub["gv_z_sim_sd"].mean(),
            "gv_z_sd_const": sub["gv_z_const_sd"].mean(),
            "ms_sd_ratio_sim_to_const": sub["gv_ms_sd_ratio_sim_to_const"].mean(),
            "r_midparent_child": mid[mid["breed"].isin(breeds)]["r_midparent_child"].mean(),
        }

    summary = pd.DataFrame([summarize(trios[trios["breed"] == b], b) for b in trios["breed"].unique()] + [summarize(trios, "All breeds")])
    trios.drop(columns=["_gv_child", "_gv_mid"]).to_csv(OUT_DIR / "layer5b_trio_recovery_per_trio.csv", index=False)
    summary.to_csv(OUT_DIR / "layer5b_trio_recovery_summary.csv", index=False)
    seg_df.to_csv(OUT_DIR / "layer5b_trio_recovery_segregation.csv", index=False)
    mid.to_csv(OUT_DIR / "layer5b_trio_recovery_midparent.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + summary.round(4).to_string(index=False))
    pooled = seg_df.groupby("mating_type")[["n"]].sum()
    for mt in ("0x1", "1x1", "1x2"):
        w = seg_df[seg_df["mating_type"] == mt]
        obs = [(w[f"obs_{g}"] * w["n"]).sum() / w["n"].sum() for g in range(3)]
        exp = [w[f"exp_{g}"].iloc[0] for g in range(3)]
        print(f"  Segregation {mt}: observed {np.round(obs, 4)} vs Mendel {np.round(exp, 3)} over {int(pooled.loc[mt, 'n']):,} locus-trios")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    s = summary[summary["breed"] != "All breeds"]
    short = [b.split()[0] for b in s["breed"]]
    x = np.arange(len(s))
    ax = axes[0, 0]
    ax.bar(x - 0.2, s["call_accuracy_parents"], 0.4, label="From the two parents", color="#2e7d4f")
    ax.bar(x + 0.2, s["call_accuracy_breed"], 0.4, label="From breed frequencies only", color="#9aa8a0")
    ax.set_xticks(x, short, rotation=20)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Calf genotype calls correct")
    ax.set_title("A. Locus-wise recovery of the real calf")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    cols = ["gv_cover90_sim", "gv_cover90_const", "gv_cover90_sim_old_rate"]
    labels = ["Meiosis simulation (1 cM/Mb)", "Constant 0.5(1-F)σ²A", "Old 5B rate (+1 crossover)"]
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - 1) * 0.27, s[c], 0.27, label=lab)
    ax.axhline(0.90, color="black", ls="--", lw=1)
    ax.set_xticks(x, short, rotation=20)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Real calves inside the 90% interval")
    ax.set_title("B. Calibration of predicted offspring genetic value")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    for mt, g, col in (("1x1", 1, "#3f6b8a"), ("0x1", 1, "#a8432f"), ("1x2", 2, "#2e7d4f")):
        w = seg_df[seg_df["mating_type"] == mt]
        ax.plot([b.split()[0] for b in w["breed"]], w[f"obs_{g}"], "o-", color=col, label=f"{mt}: share of calves with {g} copies")
        ax.axhline(w[f"exp_{g}"].iloc[0], color=col, ls=":", lw=1)
    ax.set_ylim(0.45, 0.55)
    ax.set_ylabel("Observed proportion (dotted: Mendel)")
    ax.set_title("C. Mendelian segregation in real trios")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for b in trios["breed"].unique():
        sub = trios[trios["breed"] == b]
        child = np.vstack(sub["_gv_child"].to_numpy())[:, 0]
        midp = np.vstack(sub["_gv_mid"].to_numpy())[:, 0]
        mu, sd = midp.mean(), midp.std()
        ax.scatter((midp - mu) / sd, (child - mu) / sd, s=14, alpha=0.7, label=b.split()[0])
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Mid-parent genetic value (trait 1, standardized within breed)")
    ax.set_ylabel("Real calf genetic value")
    ax.set_title(f"D. Mid-parent vs calf (mean r over {N_TRAITS} traits: {summary.iloc[-1]['r_midparent_child']:.2f})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer5b_trio_recovery.png", dpi=160)
    plt.close(fig)
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

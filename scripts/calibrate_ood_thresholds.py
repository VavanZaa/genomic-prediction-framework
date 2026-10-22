"""Risk-calibrate the OOD support thresholds on internal Florida holdouts (before Layer 6C).

Implements docs/OOD_CALIBRATION.md, which was written before this script was first run:
calibration pool = Seminole animals with q_B <= 0.32 (Layer 6A training + development cohorts); holdout
schemes = year, contemporary group (Set), ancestry band, relatedness cluster, plus a random 5-fold
reference; score = max(genomic, ancestry) k-NN support ratio; losses = absolute and level-free
(ranking) squared error; c* = largest threshold whose 95 % bootstrap upper risk bound is within
1.10 x the in-domain risk, with at least 100 accepted records. Writes output/ood_calibration/ and
refuses to overwrite a frozen thresholds.json.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.evaluation.ood_detector import GenomicOODDetector
from src.evaluation.risk_calibration import risk_coverage, select_threshold, sha256_file
from src.prediction.models import AdditiveGBLUP
from src.real_data.florida_boa_loader import FloridaBOALoader
from src.real_data.genotype_preprocessing import GenotypeImputer

OUT = ROOT / "output" / "ood_calibration"
DESIGN = ROOT / "docs" / "OOD_CALIBRATION.md"
GRID = np.r_[np.round(np.arange(0.5, 4.01, 0.1), 2), np.inf]
RISK_FACTOR, MIN_ACCEPTED, N_BOOT, SEED = 1.10, 100, 1000, 0
TRAITS = ("LoadSlope", "AvgShortLength")   # LoadSlope selects the thresholds; AvgShortLength is a check


def load_pool():
    dual = FloridaBOALoader.load_dual_dataset(marker_step=24)
    ph = dual.phenotypes.reset_index(drop=True)
    reference = ph["Year"].to_numpy() <= 2018
    G = GenotypeImputer().fit(dual.genotypes_snp[reference]).transform(dual.genotypes_snp)
    q = ph["BrahmanGBC"].to_numpy(dtype=float)
    pool = ((ph["Project"].astype(str) == "Semin").to_numpy() & ph["LoadSlope"].notna().to_numpy()
            & np.isfinite(q) & (q <= 0.32))
    return G[pool], ph.loc[pool].reset_index(drop=True), dual.provenance


def schemes(ph, G):
    n = len(ph)
    rng = np.random.default_rng(SEED)
    folds = {"random": rng.permutation(n) % 5, "year": ph["Year"].astype(int).to_numpy()}
    sets = ph["Set"].astype(str)
    counts = sets.value_counts()
    folds["contemporary_group"] = sets.where(sets.map(counts) >= 30, "small_sets").to_numpy()
    folds["ancestry_band"] = pd.qcut(ph["BrahmanGBC"], 5, labels=False, duplicates="drop").to_numpy()
    p = G.mean(axis=0) / 2.0
    Z = (G - 2 * p) / np.sqrt(np.clip(2 * p * (1 - p), 1e-6, None))
    u, s, _ = np.linalg.svd(Z, full_matrices=False)
    folds["relatedness_cluster"] = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit_predict(u[:, :10] * s[:10])
    return folds


def adjust(train_ph, ph_part, trait):
    """Set deviations estimated on training records; records of unseen Sets stay unadjusted."""
    mu = train_ph[trait].mean()
    eff = (train_ph.groupby("Set")[trait].mean() - mu).to_dict()
    return ph_part[trait].to_numpy(dtype=float) - ph_part["Set"].map(eff).fillna(0.0).to_numpy()


def run_folds(G, ph, folds):
    rows = []
    for scheme, labels in folds.items():
        for fold in np.unique(labels):
            te = labels == fold
            tr = ~te
            det = GenomicOODDetector(n_components=15).fit(G[tr], ancestry_qb_train=ph.loc[tr, "BrahmanGBC"].to_numpy())
            genomic, ancestry = det.support_ratios(G[te], ph.loc[te, "BrahmanGBC"].to_numpy())
            rec = pd.DataFrame({"animal": ph.loc[te, "UFID_str"].to_numpy(), "scheme": scheme, "fold": str(fold),
                                "s_genomic": genomic, "s_ancestry": ancestry, "s": np.maximum(genomic, ancestry)})
            for trait in TRAITS:
                ok_tr = tr & ph[trait].notna().to_numpy()
                ok_te = te & ph[trait].notna().to_numpy()
                y_tr = adjust(ph[ok_tr], ph[ok_tr], trait)
                pred = np.full(te.sum(), np.nan)
                y_te = np.full(te.sum(), np.nan)
                sub = ok_te[te]
                if sub.any():
                    pred[sub] = AdditiveGBLUP().fit(G[ok_tr], y_tr).predict(G[ok_te])
                    y_te[sub] = adjust(ph[ok_tr], ph[ok_te], trait)
                rec[f"loss_abs_{trait}"] = (y_te - pred) ** 2
                rec[f"loss_rank_{trait}"] = ((y_te - np.nanmean(y_te)) - (pred - np.nanmean(pred))) ** 2
            rows.append(rec)
            print(f"  {scheme:20s} fold {str(fold):12s}: held out {te.sum():4d}, median s {np.median(rec['s']):.2f}")
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true", help="Replace a frozen result (recorded in the output)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "thresholds.json").exists() and not args.overwrite:
        raise SystemExit("output/ood_calibration/thresholds.json is frozen; thresholds are not re-tuned")

    G, ph, provenance = load_pool()
    print(f"Calibration pool: {len(ph)} Seminole animals with q_B <= 0.32 ({provenance}); {G.shape[1]} SNPs")
    folds = schemes(ph, G)
    records = run_folds(G, ph, folds)
    records.to_csv(OUT / "records.csv", index=False)

    shift = records[records["scheme"] != "random"]
    ref = records[records["scheme"] == "random"]
    result, curves = {}, []
    for axis in ("rank", "abs"):
        for trait in TRAITS:
            col = f"loss_{axis}_{trait}"
            r_ref = float(ref[col].mean())
            d = shift[shift[col].notna()]
            curve = risk_coverage(d["s"], d[col], GRID, units=d["animal"], n_boot=N_BOOT, seed=SEED)
            curve["axis"], curve["trait"], curve["risk_ref"] = axis, trait, r_ref
            curve["risk_max"] = RISK_FACTOR * r_ref
            curves.append(curve)
            c = select_threshold(curve, RISK_FACTOR * r_ref, MIN_ACCEPTED)
            at = lambda thr: curve.loc[np.isclose(curve["threshold"], thr)].iloc[0].to_dict() if thr is not None else None
            result[(axis, trait)] = {"c_star": c, "risk_ref": r_ref, "risk_max": RISK_FACTOR * r_ref,
                                     "at_c_star": at(c), "at_v1_rule": at(1.0)}
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / "risk_coverage.csv", index=False)

    c_rank, c_abs = result[("rank", "LoadSlope")]["c_star"], result[("abs", "LoadSlope")]["c_star"]
    def clean(v):
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
    thresholds = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overwrite": bool(args.overwrite),
        "c_star": {"ranking": clean(c_rank), "absolute": clean(c_abs)},
        "accept_everything": {"ranking": c_rank is not None and np.isinf(c_rank), "absolute": c_abs is not None and np.isinf(c_abs)},
        "rule": {"score": "max(genomic, ancestry) kNN ratio", "risk_factor": RISK_FACTOR,
                 "min_accepted": MIN_ACCEPTED, "bootstrap": f"{N_BOOT} animal resamples, 95% upper bound",
                 "selection_trait": "LoadSlope"},
        "pool": {"n_animals": int(len(ph)), "definition": "Semin, LoadSlope observed, q_B <= 0.32"},
        "details": {f"{a}_{t}": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in
                                 {kk: vv for kk, vv in r.items() if kk not in ("at_c_star", "at_v1_rule")}.items()}
                    | {"at_c_star": r["at_c_star"], "at_v1_rule": r["at_v1_rule"]}
                    for (a, t), r in result.items()},
        "sha256": {"script": sha256_file(Path(__file__)), "design": sha256_file(DESIGN),
                   "records": sha256_file(OUT / "records.csv")},
    }
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2, default=float), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, axis, title in ((axes[0], "rank", "Ranking loss (level removed)"), (axes[1], "abs", "Absolute loss")):
        for trait, style in zip(TRAITS, ("-", "--")):
            c = curves[(curves["axis"] == axis) & (curves["trait"] == trait) & np.isfinite(curves["threshold"])]
            ax.plot(c["coverage"], c["risk"] / c["risk_ref"], style, label=f"{trait} risk")
            ax.plot(c["coverage"], c["risk_ucb"] / c["risk_ref"], style, alpha=0.4, label=f"{trait} 95 % upper bound")
        ax.axhline(RISK_FACTOR, color="black", lw=1, ls=":", label="target (1.10 x in-domain)")
        ax.set_xlabel("Coverage C(c)"), ax.set_ylabel("Risk / in-domain risk"), ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=150)

    print("\nSelected thresholds (LoadSlope):", thresholds["c_star"])
    for (axis, trait), r in result.items():
        a, v = r["at_c_star"], r["at_v1_rule"]
        print(f"  {axis:4s} {trait:15s}: c* = {r['c_star']}; in-domain risk {r['risk_ref']:.3f}; "
              + (f"at c*: coverage {a['coverage']:.2f}, risk {a['risk']:.3f} (UCB {a['risk_ucb']:.3f}); " if a else "no threshold meets the target; ")
              + f"v1 rule c=1: coverage {v['coverage']:.2f}, risk {v['risk']:.3f}")
    print(f"thresholds.json SHA-256: {sha256_file(OUT / 'thresholds.json')}")


if __name__ == "__main__":
    main()

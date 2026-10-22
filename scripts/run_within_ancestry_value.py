"""Does genomic prediction earn its keep where breed percentage cannot help?

C14 found that an ancestry-only baseline matches or beats genomic prediction on the Florida traits.
But that comparison ranges over animals of DIFFERENT ancestry, which is the one thing a breed
percentage is good at. Once a breeder has fixed the cross, every candidate animal has the same breed
percentage, and the baseline has no variance left to use.

So this restricts the comparison to narrow ancestry bands and asks whether genomic prediction still
ranks animals inside them. The within-ancestry genomic heritability measured earlier (0.160 for hair
length, 0.133 for thermal slope) says there is something there to find; this asks whether the model
finds it.

Bands are quantile-based so each holds enough animals. Within a band: train/test split with a
contemporary-group-adjusted phenotype, GBLUP against an ancestry-only baseline, repeated.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prediction.kernel_cv import cv_kernel_ridge  # noqa: E402

GRM = ROOT / "data/real_cattle/florida_multibreed/florida_grm_122k_v3.npz"
PHENO = ROOT / "data/real_cattle/florida_multibreed/Thermo.Phenotypes.csv"
OUT = ROOT / "output/within_ancestry"
TRAITS = ["AvgShortLength", "LoadSlope"]
N_BANDS, N_REPEATS, TEST_SHARE, SEED = 4, 20, 0.25, 20260917


def adjusted(df, trait):
    """Phenotype with contemporary group (Set) removed, fitted on the band."""
    y = df[trait].to_numpy(float)
    g = pd.get_dummies(df["Set"].astype(str), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(y)), g])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta + float(y.mean())


def main():
    z = np.load(GRM, allow_pickle=False)
    G = z["G_122k"].astype(float)
    ids = [str(x) for x in z["animal_ids"]]
    ph = pd.read_csv(PHENO, dtype={"UFID": str}).set_index("UFID").reindex(ids)
    ph["row"] = np.arange(len(ph))
    rng = np.random.default_rng(SEED)
    results = {}

    for trait in TRAITS:
        sub = ph[ph[trait].notna() & ph["BrahmanGBC"].notna() & ph["Set"].notna()].copy()
        sub["band"] = pd.qcut(sub["BrahmanGBC"], N_BANDS, labels=False)
        rows = []
        for b in range(N_BANDS):
            d = sub[sub["band"] == b]
            if len(d) < 150:
                continue
            q = d["BrahmanGBC"].to_numpy(float)
            y = adjusted(d, trait)
            idx = d["row"].to_numpy()
            g_s, a_s = [], []
            for _ in range(N_REPEATS):
                perm = rng.permutation(len(d))
                cut = int(round(TEST_SHARE * len(d)))
                te, tr = perm[:cut], perm[cut:]
                K = G[np.ix_(idx[tr], idx[tr])]
                K_te = G[np.ix_(idx[te], idx[tr])]
                mu = float(y[tr].mean())
                lam = cv_kernel_ridge([K], y[tr] - mu, n_folds=5, seed=42)["lambda"]
                alpha = np.linalg.solve(K + lam * np.eye(len(tr)), y[tr] - mu)
                pred_g = mu + K_te @ alpha
                A = np.column_stack([np.ones(len(tr)), q[tr]])
                beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
                pred_a = np.column_stack([np.ones(len(te)), q[te]]) @ beta
                g_s.append(float(spearmanr(pred_g, y[te]).statistic))
                a_s.append(float(spearmanr(pred_a, y[te]).statistic))
            g, a = np.array(g_s), np.array(a_s)
            rows.append({
                "band": int(b), "n": int(len(d)),
                "ancestry_range": [round(float(q.min()), 3), round(float(q.max()), 3)],
                "ancestry_sd_within_band": round(float(q.std()), 4),
                "gblup_rho": round(float(np.median(g)), 4),
                "gblup_ci95": [round(float(np.percentile(g, 2.5)), 4),
                               round(float(np.percentile(g, 97.5)), 4)],
                "ancestry_only_rho": round(float(np.median(a)), 4),
                "delta": round(float(np.median(g - a)), 4),
                "gblup_wins_share": round(float(np.mean(g > a)), 3)})
            r = rows[-1]
            print(f"  {trait:16s} band {b} (q {r['ancestry_range'][0]:.2f}-{r['ancestry_range'][1]:.2f}, "
                  f"n={r['n']:>4}): GBLUP {r['gblup_rho']:+.3f} {r['gblup_ci95']}  "
                  f"ancestry {r['ancestry_only_rho']:+.3f}  delta {r['delta']:+.3f}  "
                  f"wins {r['gblup_wins_share']:.2f}", flush=True)
        results[trait] = rows

    payload = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "question": ("does genomic prediction rank animals inside a fixed ancestry band, where a "
                     "breed-percentage baseline has almost no variance to use?"),
        "design": {"bands": N_BANDS, "repeats": N_REPEATS, "test_share": TEST_SHARE,
                   "phenotype": "contemporary-group adjusted within band", "seed": SEED},
        "traits": results,
        "reading": ("C14 compared models across animals of differing ancestry, which favours a breed "
                    "baseline. A breeder who has chosen a cross faces the opposite situation: every "
                    "candidate has the same breed percentage. These bands approximate that case."),
        "limits": [
            "random splits within band, so relatives are shared between train and test; these are "
            "optimistic relative to a new herd",
            "bands still contain some ancestry variation, so the baseline is weakened rather than "
            "eliminated",
            "n per band is roughly a quarter of the herd, which is small for genomic prediction",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "within_ancestry_value.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {(OUT / 'within_ancestry_value.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

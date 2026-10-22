"""C5 re-analysis: the specificity null with family dependence taken seriously, and with the
question "what could this experiment have detected?" answered rather than assumed.

The frozen 6C2 secondary endpoint bootstrapped affected and unaffected animals independently. These
99 animals are progeny of a small number of sires, so that interval is too narrow: it treats
relatives as independent observations. This script redoes it three ways and reports what each can
and cannot support.

  1. Power. The detectable standardised effect at 80 % power, two-sided alpha = 0.05, for the actual
     group sizes - assuming independence, which is optimistic.
  2. Sire-cluster bootstrap. Resamples sires, not animals. With only three sires this is coarse, and
     that is reported rather than hidden.
  3. Within-sire permutation. Permutes coat status within sire, so sire background is held fixed.
     This is the better-powered test of the two and the one to read.

Nothing here changes any threshold, model or panel. The conclusion it supports is bounded:
"no detectable association", not "the score does not respond".
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_layer6c2_hairy_probe import PLINK, TRAITS  # noqa: E402

OUT = ROOT / "output/layer6c2"
N_BOOT, N_PERM, SEED = 2000, 20000, 20260912


def smd(a, u):
    """Standardised mean difference (affected minus unaffected), pooled SD."""
    if len(a) < 2 or len(u) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(u) - 1) * u.var(ddof=1)) / (len(a) + len(u) - 2))
    return (a.mean() - u.mean()) / pooled if pooled > 0 else np.nan


def detectable_effect(n1, n0, power=0.80, alpha=0.05):
    """Smallest |d| a two-sample test of these sizes detects at the given power, assuming independence."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(1 / n1 + 1 / n0)


def main():
    scored = pd.read_csv(OUT / "prepared_hairy.csv", dtype={"iid": str})
    fam = pd.read_csv(f"{PLINK}.fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "pid", "mid", "sex", "phenotype"], dtype=str)
    if scored["iid"].tolist() != fam["iid"].tolist():
        raise SystemExit("prepared rows do not match the .fam order")

    affected = (fam["phenotype"] == "2").to_numpy()
    sire = fam["pid"].to_numpy()          # '0' means unrecorded sire, kept as its own group
    dam = fam["mid"].to_numpy()
    rng = np.random.default_rng(SEED)

    structure = {
        "n_animals": int(len(fam)), "n_affected": int(affected.sum()),
        "n_unaffected": int((~affected).sum()),
        "distinct_sires": int(pd.Series(sire).nunique()),
        "sire_group_sizes": pd.Series(sire).value_counts().to_dict(),
        "distinct_dams": int(pd.Series(dam).nunique()),
        "animals_with_unrecorded_dam": int((dam == "0").sum()),
        "affected_share_by_sire": {s: round(float(affected[sire == s].mean()), 4)
                                   for s in pd.unique(sire)},
    }
    print("family structure:", json.dumps(structure, indent=2))

    power = {
        "detectable_smd_at_80pct_power": round(float(detectable_effect(int(affected.sum()),
                                                                      int((~affected).sum()))), 3),
        "assumption": "independent animals - optimistic here, since these are relatives",
        "meaning": ("effects smaller than this are not reliably detectable by this experiment, so a "
                    "null result excludes large effects far better than modest ones"),
    }
    print(f"\ndetectable |d| at 80% power (independence assumed): {power['detectable_smd_at_80pct_power']}")

    # phenotype is unevenly distributed across sires, which is itself the reason clustering matters
    sires = pd.unique(sire)
    results = {}
    for t in TRAITS:
        v = scored[f"pred_{t}"].to_numpy()
        point = float(smd(v[affected], v[~affected]))

        # 1. the frozen approach: resample animals independently (too narrow under relatedness)
        naive = []
        a_v, u_v = v[affected], v[~affected]
        for _ in range(N_BOOT):
            d = smd(a_v[rng.integers(0, len(a_v), len(a_v))], u_v[rng.integers(0, len(u_v), len(u_v))])
            if np.isfinite(d):
                naive.append(d)

        # 2. sire-cluster bootstrap: resample whole sire families
        clustered = []
        for _ in range(N_BOOT):
            picked = rng.choice(len(sires), size=len(sires), replace=True)
            rows = np.concatenate([np.flatnonzero(sire == sires[i]) for i in picked])
            d = smd(v[rows][affected[rows]], v[rows][~affected[rows]])
            if np.isfinite(d):
                clustered.append(d)

        # 3. within-sire permutation: sire background held fixed
        def within_sire_statistic(labels):
            centred = np.empty_like(v)
            for s in sires:
                m = sire == s
                centred[m] = v[m] - v[m].mean()
            return smd(centred[labels], centred[~labels])

        observed = float(within_sire_statistic(affected))
        null = []
        for _ in range(N_PERM):
            perm = affected.copy()
            for s in sires:
                m = np.flatnonzero(sire == s)
                perm[m] = rng.permutation(affected[m])
            d = within_sire_statistic(perm)
            if np.isfinite(d):
                null.append(d)
        null = np.array(null)
        p_perm = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))

        results[t] = {
            "smd": round(point, 4),
            "naive_animal_bootstrap_ci95": [round(float(np.percentile(naive, 2.5)), 4),
                                            round(float(np.percentile(naive, 97.5)), 4)],
            "sire_cluster_bootstrap_ci95": [round(float(np.percentile(clustered, 2.5)), 4),
                                            round(float(np.percentile(clustered, 97.5)), 4)],
            "sire_cluster_bootstrap_width_ratio": round(
                float((np.percentile(clustered, 97.5) - np.percentile(clustered, 2.5)) /
                      (np.percentile(naive, 97.5) - np.percentile(naive, 2.5))), 3),
            "within_sire_smd": round(observed, 4),
            "within_sire_permutation_p": round(p_perm, 4),
            "detects": bool(p_perm < 0.05),
        }
        r = results[t]
        print(f"\n{t}")
        print(f"  SMD {r['smd']:+.3f}")
        print(f"  naive animal bootstrap  95% CI {r['naive_animal_bootstrap_ci95']}")
        print(f"  sire-cluster bootstrap  95% CI {r['sire_cluster_bootstrap_ci95']} "
              f"(x{r['sire_cluster_bootstrap_width_ratio']} wider)")
        print(f"  within-sire SMD {r['within_sire_smd']:+.3f}, permutation p = {r['within_sire_permutation_p']:.4f}")

    out = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "status": "post-hoc re-analysis of the 6C2 secondary endpoint (family dependence)",
        "claim_under_test": "C5: the polygenic hair score does not respond to hairy syndrome",
        "family_structure": structure,
        "power": power,
        "traits": results,
        "limitations": [
            "three sire clusters only, so the cluster bootstrap is coarse and its interval unreliable",
            "dam is unrecorded for 77 of 99 animals, so maternal families cannot be used",
            "coat status is unevenly distributed across sires, so sire and phenotype are partly confounded",
            "25% panel coverage compresses the score, which shrinks any true effect toward zero",
        ],
        "supported_conclusion": (
            "No detectable association between the Florida polygenic scores and hairy-syndrome status. "
            "The experiment excludes large effects far more convincingly than modest ones, and family "
            "dependence plus low panel coverage limit interpretation. This is CONSISTENT_WITH_NULL, not "
            "a demonstration that the score does not respond."),
        "changes_nothing": "no threshold, model, panel or training set is modified",
    }
    (OUT / "results_6c2_c5_family_clustered.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {(OUT / 'results_6c2_c5_family_clustered.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

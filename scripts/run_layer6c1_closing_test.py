"""6C-1 closing test: accepted vs refused rank transfer in the Mizzou near-domain arm, clustered.

Implements Amendments 1 and 2 of docs/LAYER6C_PROTOCOL.md (post-hoc by construction; the frozen primary
endpoint is unchanged). Amendment 2 corrects the contemporary unit: `score_group` has only 16 values (a
within-herd window index), while date + apparent high + day length identify 433 scoring events. Every
statistic is computed for both units side by side:
- rho among accepted and among refused animals;
- delta rho with 2,000 cluster bootstrap replicates over units;
- an accepted x prediction interaction with CR1 cluster-robust standard errors;
- the same interaction with unit fixed effects, which compares accepted and refused animals inside the
  same unit, plus how many units actually hold both kinds;
- rho among accepted with a cluster-bootstrap interval (robustness check of the primary endpoint).
Writes output/layer6c/results_6c1_closing.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

OUT = ROOT / "output" / "layer6c"
PREP = OUT / "prepared_mizzou.csv"
PHENO = ROOT / "data/external/mizzou_hair_shedding/Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv"
MIN_GROUP, MIN_SIDE, MIN_MIXED_UNITS, N_BOOT, SEED = 5, 10, 10, 2000, 0
PRED = "pred_AvgShortLength"


def records() -> pd.DataFrame:
    ph = pd.read_csv(PHENO, dtype={"id": str, "score_group": str})
    ph["event"] = (ph["date_score_recorded"].astype(str) + "|" + ph["mean_apparent_high"].round(3).astype(str)
                   + "|" + ph["mean_day_length"].round(3).astype(str))
    return ph


def animal_table(ph: pd.DataFrame, unit: str) -> pd.DataFrame:
    """One row per animal: unit-centred mean score, dominant unit, prediction and acceptance."""
    d = ph[ph.groupby(unit)["hair_score"].transform("size") >= MIN_GROUP].copy()
    d["centred"] = d["hair_score"] - d.groupby(unit)["hair_score"].transform("mean")
    agg = d.groupby("id").agg(phenotype=("centred", "mean"), records=("centred", "size"))
    dominant = (d.groupby(["id", unit]).size().rename("n").reset_index()
                 .sort_values(["id", "n"], ascending=[True, False]).drop_duplicates("id").set_index("id")[unit])
    prep = pd.read_csv(PREP, dtype={"id": str})
    out = prep[(~prep["duplicate"]) & (prep["arm"] == "near")].merge(agg.join(dominant), left_on="id",
                                                                    right_index=True, how="inner")
    return out.rename(columns={unit: "cluster"}).reset_index(drop=True)


def rho(x, y):
    return float(stats.spearmanr(x, y)[0]) if len(x) > 2 else np.nan


def cluster_bootstrap(df: pd.DataFrame):
    """Cluster bootstrap over units: delta rho (accepted - refused) and rho among accepted."""
    rng = np.random.default_rng(SEED)
    groups = df["cluster"].to_numpy()
    by_unit = {g: df.index[groups == g].to_numpy() for g in np.unique(groups)}
    keys = np.array(list(by_unit))
    deltas, accepted_rhos, skipped = [], [], 0
    for _ in range(N_BOOT):
        idx = np.concatenate([by_unit[g] for g in rng.choice(keys, size=len(keys), replace=True)])
        sub = df.loc[idx]
        acc, ref = sub[sub["ranking_ok"]], sub[~sub["ranking_ok"]]
        if len(acc) < MIN_SIDE or len(ref) < MIN_SIDE:
            skipped += 1
            continue
        accepted_rhos.append(rho(acc[PRED], acc["phenotype"]))
        deltas.append(accepted_rhos[-1] - rho(ref[PRED], ref["phenotype"]))
    d, a = np.asarray(deltas), np.asarray(accepted_rhos)
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"units": int(len(keys)), "replicates_used": int(d.size), "replicates_skipped": skipped,
            "delta_rho_mean": float(d.mean()), "delta_rho_ci95": pct(d),
            "delta_rho_p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "rho_accepted_mean": float(a.mean()), "rho_accepted_ci95": pct(a)}


def interaction(df: pd.DataFrame, unit_fixed_effects: bool):
    """Rank-transformed OLS with accepted x prediction, CR1 errors clustered on the unit."""
    r = lambda v: stats.rankdata(v) / len(v)
    x, y = r(df[PRED].to_numpy()), r(df["phenotype"].to_numpy())
    a = df["ranking_ok"].to_numpy(dtype=float)
    clusters = df["cluster"].to_numpy()
    keys = np.unique(clusters)
    parts = [x, a, x * a]
    if unit_fixed_effects:
        parts += [np.column_stack([(clusters == g).astype(float) for g in keys])]
    else:
        parts = [np.ones_like(x)] + parts
    X = np.column_stack(parts)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in keys:
        m = clusters == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    n, k, G = len(y), int(np.linalg.matrix_rank(X)), len(keys)
    V = (n - 1) / max(n - k, 1) * G / max(G - 1, 1) * XtX_inv @ meat @ XtX_inv
    j = 2 if unit_fixed_effects else 3
    se = float(np.sqrt(V[j, j]))
    t = float(beta[j] / se)
    sizes = df.groupby("cluster")["ranking_ok"].agg(["size", "sum"])
    mixed = sizes[(sizes["sum"] >= MIN_SIDE) & (sizes["size"] - sizes["sum"] >= MIN_SIDE)]
    return {"interaction_coef": float(beta[j]), "cluster_robust_se": se, "t": t,
            "p_two_sided": float(2 * stats.t.sf(abs(t), df=max(G - 1, 1))), "clusters": int(G),
            "slope_refused": float(beta[0 if unit_fixed_effects else 1]),
            "slope_accepted": float(beta[0 if unit_fixed_effects else 1] + beta[j]),
            "units_with_both_sides": int(len(mixed)), "animals_in_mixed_units": int(mixed["size"].sum()) if len(mixed) else 0}


def main():
    ph = records()
    res = {"structure": {"records": int(len(ph)), "animals": int(ph["id"].nunique()),
                         "score_group_values": int(ph["score_group"].nunique()), "events": int(ph["event"].nunique())},
           "frozen_primary_endpoint": {"rho_accepted": 0.16650276556079205, "ci95": [0.10274610927657139, 0.2307924868448599],
                                       "note": "score_group centring, bootstrap over animals; unchanged"}}
    for unit in ("score_group", "event"):
        df = animal_table(ph, unit)
        acc, ref = df[df["ranking_ok"]], df[~df["ranking_ok"]]
        res[unit] = {
            "animals": int(len(df)), "accepted": int(len(acc)), "refused": int(len(ref)),
            "rho_accepted": rho(acc[PRED], acc["phenotype"]), "rho_refused": rho(ref[PRED], ref["phenotype"]),
            "cluster_bootstrap": cluster_bootstrap(df),
            "interaction": interaction(df, unit_fixed_effects=False),
            "interaction_within_unit": interaction(df, unit_fixed_effects=True),
        }
    e = res["event"]
    ok = (e["cluster_bootstrap"]["delta_rho_ci95"][0] > 0 or e["cluster_bootstrap"]["delta_rho_ci95"][1] < 0)
    within = e["interaction_within_unit"]
    res["decision_6c1"] = ("accepted transfer significantly stronger (event-clustered): 6C-1 can close"
                           if ok and within["p_two_sided"] < 0.05 and within["units_with_both_sides"] >= MIN_MIXED_UNITS
                           else "unresolved under Amendment 2's rule: 6C-1 stays open")
    res["interpretation_limit"] = "acceptance follows genotype-based support, not randomisation"
    (OUT / "results_6c1_closing.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

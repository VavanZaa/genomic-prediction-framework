"""Is there genetic signal left once breed composition is accounted for?

C14 found that an ancestry-only baseline matches or beats genomic prediction for these traits, and
the project's own calibration says between-breed variance is roughly 12-22x the within-breed
additive variance. So the question that decides whether more training animals would help is not
"what is the heritability" but:

    how much heritability survives after ancestry is fitted as a fixed effect?

If it is near zero, breed percentage is the ceiling for these traits and more animals change
nothing. If it is appreciable, the signal exists and the sample size is what limits accuracy.

Single-component GREML on the 122k VanRaden GRM, by eigendecomposition: with G = U D U', the
REML likelihood is cheap to evaluate on a grid of h2, and the confidence interval comes from the
profile likelihood (a drop of 1.92 log-likelihood units).

Three models per trait, each adding one layer of fixed effects:
  1. intercept only                      - total genomic heritability, breed differences included
  2. + contemporary group (Set)          - removes herd, year and management
  3. + ancestry (Brahman fraction)       - the number this script exists for
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRM = ROOT / "data/real_cattle/florida_multibreed/florida_grm_122k_v3.npz"
PHENO = ROOT / "data/real_cattle/florida_multibreed/Thermo.Phenotypes.csv"
OUT = ROOT / "output/heritability"
TRAITS = ["AvgShortLength", "LoadSlope"]
GRID = np.linspace(1e-4, 0.999, 400)


def reml_profile(y, X, d, Ut_y, Ut_X):
    """Profile REML log-likelihood over h2, using the GRM eigenvalues."""
    n, p = X.shape[0], X.shape[1]
    out = []
    for h2 in GRID:
        w = h2 * d + (1.0 - h2)
        w = np.maximum(w, 1e-10)
        XtWX = Ut_X.T @ (Ut_X / w[:, None])
        XtWy = Ut_X.T @ (Ut_y / w)
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            sign, logdet_XtWX = np.linalg.slogdet(XtWX)
        except np.linalg.LinAlgError:
            out.append(-np.inf)
            continue
        resid = Ut_y - Ut_X @ beta
        q = float(np.sum(resid ** 2 / w))
        ll = -0.5 * (np.sum(np.log(w)) + (n - p) * np.log(q / (n - p)) + logdet_XtWX)
        out.append(ll)
    return np.array(out)


def fit(y, X, d, U):
    Ut_y, Ut_X = U.T @ y, U.T @ X
    ll = reml_profile(y, X, d, Ut_y, Ut_X)
    i = int(np.nanargmax(ll))
    h2 = float(GRID[i])
    inside = np.flatnonzero(ll >= ll[i] - 1.92)
    lo, hi = float(GRID[inside[0]]), float(GRID[inside[-1]])
    # variance scale at the optimum
    w = h2 * d + (1 - h2)
    XtWX = Ut_X.T @ (Ut_X / w[:, None])
    beta = np.linalg.solve(XtWX, Ut_X.T @ (Ut_y / w))
    resid = Ut_y - Ut_X @ beta
    var_p = float(np.sum(resid ** 2 / w) / (len(y) - X.shape[1]))
    return {"h2": round(h2, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "var_phenotypic": round(var_p, 4), "var_additive": round(h2 * var_p, 4),
            "n": int(len(y)), "fixed_effects": int(X.shape[1]),
            "at_boundary": bool(h2 <= GRID[1] or h2 >= GRID[-2])}


def design(df, group=False, ancestry=False):
    X = [np.ones(len(df))]
    names = ["intercept"]
    if group:
        codes = pd.get_dummies(df["Set"].astype(str), drop_first=True).to_numpy(dtype=float)
        X.append(codes)
        names += [f"Set_{i}" for i in range(codes.shape[1])]
    if ancestry:
        q = df["BrahmanGBC"].to_numpy(dtype=float)
        X.append(np.column_stack([q, q ** 2]))
        names += ["BrahmanGBC", "BrahmanGBC^2"]
    return np.column_stack(X), names


def main():
    z = np.load(GRM, allow_pickle=False)
    G = z["G_122k"].astype(float)                 # SNP-based VanRaden GRM at 122,486 markers
    ids = [str(x) for x in z["animal_ids"]]
    print(f"GRM {G.shape} over {int(z['num_markers'])} markers, {len(ids)} ids")

    ph = pd.read_csv(PHENO, dtype={"UFID": str}).set_index("UFID")
    ph = ph.reindex(ids)
    results = {}
    for trait in TRAITS:
        ok = ph[trait].notna() & ph["BrahmanGBC"].notna() & ph["Set"].notna()
        idx = np.flatnonzero(ok.to_numpy())
        sub = ph.iloc[idx]
        Gs = G[np.ix_(idx, idx)]
        # scale so the average diagonal is 1: h2 is then on the usual scale
        Gs = Gs / np.mean(np.diag(Gs))
        d, U = np.linalg.eigh(Gs)
        d = np.maximum(d, 1e-8)
        y = sub[trait].to_numpy(dtype=float)
        print(f"\n{trait}: n = {len(y)}")
        models = {}
        for name, kw in (("1_intercept_only", {}),
                         ("2_plus_contemporary_group", {"group": True}),
                         ("3_plus_ancestry", {"group": True, "ancestry": True})):
            X, _ = design(sub, **kw)
            r = fit(y, X, d, U)
            models[name] = r
            print(f"  {name:28s} h2 = {r['h2']:.3f}  95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
                  f"  (n={r['n']}, {r['fixed_effects']} fixed effects)"
                  + ("  [at boundary]" if r["at_boundary"] else ""))
        drop = models["1_intercept_only"]["h2"] - models["3_plus_ancestry"]["h2"]
        models["heritability_lost_to_fixed_effects"] = round(float(drop), 4)
        results[trait] = models

    verdict = {
        t: ("appreciable within-ancestry signal: sample size is plausibly the limit"
            if r["3_plus_ancestry"]["h2"] >= 0.15 else
            "little within-ancestry signal: more animals would not obviously help")
        for t, r in results.items()}
    payload = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "question": "how much genomic heritability survives after ancestry is fitted",
        "method": "single-component GREML on the 122k VanRaden GRM, eigendecomposition, "
                  "profile-likelihood 95% interval",
        "grm": str(GRM.relative_to(ROOT)), "traits": results, "verdict": verdict,
        "interpretation_limits": [
            "one herd and two traits; not a general statement about cattle",
            "ancestry is fitted as BrahmanGBC linear plus quadratic; a different functional form "
            "would move the residual heritability somewhat",
            "contemporary group is Set, the unit used in the canonical split",
            "heritability bounds what prediction can reach; it does not promise that accuracy will "
            "follow, and a non-zero h2 is necessary but not sufficient for genomic prediction to "
            "beat an ancestry baseline",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "genomic_heritability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nverdict:", json.dumps(verdict, indent=2))
    print(f"wrote {(OUT / 'genomic_heritability.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

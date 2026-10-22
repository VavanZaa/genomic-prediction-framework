"""Layer 7 / H7b: does the genomic prediction machinery work in a second species?

Protocol: docs/LAYER7_H7B_PROTOCOL.md.

  --stage prepare   build the mouse GRM and the trait table by the pre-registered selection rule.
                    Reads phenotype columns to count completeness and to assemble the table; fits
                    no model and computes no accuracy.
  --stage unblind   verify the lock, then fit GBLUP and the covariates-only baseline once, over
                    held-out-family splits, and evaluate E1, E2 and E3.

Nothing is tuned on mouse phenotypes: lambda comes from training-only cross-validation and the
support region is fitted on training mice. No cattle result is touched.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402
from src.prediction.kernel_cv import cv_kernel_ridge  # noqa: E402

HS = ROOT / "data/real_mice/hs_mice"
OUT = ROOT / "output/layer7b"
LOCK = OUT / "PROTOCOL_LOCK_H7B.json"
CHROMS = [str(c) for c in range(1, 20)]          # X excluded, as in every other layer
PREFIXES = {"Weight.txt": ("Weight.", "Start.", "End."), "Biochemistry.txt": ("Biochem.",),
            "Haematology.txt": ("Haem.",), "Immunology.txt": ("Imm.",)}
TRAITS_PER_FILE = 2
COVARIATE_CANDIDATES = ["GENDER", "Age", "End.Age", "Date.Year", "DATE.OF.BIRTH.Year"]
EXCLUDE = ("Date", "Cage", "Litter", "Age", "Year", "Month", "Day", "Hour", "Seconds", "Study")
N_REPEATS, TEST_FAMILY_SHARE, SEED = 10, 0.20, 20260917
E1_MIN_TRAITS, E2_MIN_TRAITS, E3_MIN_ACCEPT = 4, 4, 0.90


def read_genotypes():
    """Counted-allele dosages for the autosomes, plus subject ids and family labels."""
    ids, fams, blocks = None, None, []
    for c in CHROMS:
        rows, sub, fam = [], [], []
        with open(HS / f"genotypes/chr{c}.build37.data") as fh:
            for line in fh:
                f = line.split()
                fam.append(f[0])
                sub.append(f[1])
                rows.append(f[6:])
        arr = np.array(rows, dtype=object)
        n_marker = arr.shape[1] // 2
        dose = np.full((len(sub), n_marker), np.nan, dtype=np.float32)
        for j in range(n_marker):
            a, b = arr[:, 2 * j], arr[:, 2 * j + 1]
            seen = [x for x in np.unique(np.concatenate([a, b])) if x != "NA"]
            if not seen:
                continue
            ref = sorted(seen)[0]
            called = (a != "NA") & (b != "NA")
            dose[called, j] = ((a[called] == ref).astype(np.float32)
                               + (b[called] == ref).astype(np.float32))
        blocks.append(dose)
        if ids is None:
            ids, fams = sub, fam
        elif sub != ids:
            raise SystemExit(f"chr{c}: animal order differs from chr1")
        print(f"  chr{c}: {n_marker} markers", flush=True)
    return ids, np.array(fams), np.hstack(blocks)


def select_traits():
    """The pre-registered rule: two most complete biological traits per file."""
    subs = {l.split()[1] for l in open(HS / "genotypes/chr1.build37.data")}
    chosen, tables = [], {}
    for fname, prefixes in PREFIXES.items():
        t = pd.read_csv(HS / "phenotypes" / fname, sep="\t", dtype={"SUBJECT.NAME": str}, low_memory=False)
        t = t[t["SUBJECT.NAME"].isin(subs)].copy()
        tables[fname] = t
        num = t.select_dtypes("number")
        cand = [c for c in num.columns
                if c.startswith(prefixes) and not any(k in c for k in EXCLUDE)]
        counts = num[cand].notna().sum().sort_values(ascending=False)
        for trait in list(counts.index[:TRAITS_PER_FILE]):
            chosen.append({"file": fname, "trait": trait, "n_complete": int(counts[trait])})
    return chosen, tables


def build_design(t, trait):
    cols = [c for c in COVARIATE_CANDIDATES if c in t.columns]
    df = t[["SUBJECT.NAME", trait] + cols].copy()
    df = df[df[trait].notna()]
    X = [np.ones(len(df))]
    for c in cols:
        v = df[c]
        # pandas may give a string dtype that is not `object`, so test numeric-ness directly
        if not pd.api.types.is_numeric_dtype(v):
            d = pd.get_dummies(v.astype(str), drop_first=True).to_numpy(dtype=float)
            if d.size:
                X.append(d)
        else:
            X.append(v.fillna(v.median()).to_numpy(dtype=float)[:, None])
    return df["SUBJECT.NAME"].tolist(), df[trait].to_numpy(dtype=float), np.column_stack(X), cols


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    print("reading genotypes ...", flush=True)
    ids, fams, dose = read_genotypes()
    missing = float(np.isnan(dose).mean())
    mean = np.nanmean(dose, axis=0)
    dose = np.where(np.isnan(dose), mean, dose).astype(np.float32)
    keep = np.nanstd(dose, axis=0) > 0
    dose = dose[:, keep]
    p = np.clip(mean[keep].astype(np.float64) / 2.0, 1e-4, 1 - 1e-4)
    Z = dose - (2 * p).astype(np.float32)
    G = (Z @ Z.T).astype(np.float64) / float(2.0 * np.sum(p * (1 - p)))
    print(f"mice {len(ids)}, markers kept {int(keep.sum())}, missing {missing:.4f}, GRM {G.shape}")

    chosen, _ = select_traits()
    np.savez_compressed(OUT / "mouse_grm.npz", ids=np.array(ids, dtype=str),
                        families=fams.astype(str), G=G)
    summary = {"stage": "prepare", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "evidence": "EMPIRICAL", "mice": len(ids), "families": int(len(set(fams))),
               "markers_used": int(keep.sum()), "genotype_missing_share": round(missing, 5),
               "selected_traits": chosen,
               "computed_here": "GRM and trait selection only; no model fitted, no accuracy computed"}
    (OUT / "prepare_summary_h7b.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the H7b protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def unblind():
    verify_lock()
    z = np.load(OUT / "mouse_grm.npz", allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    fams = np.array([str(x) for x in z["families"]])
    G = z["G"]
    pos = {a: i for i, a in enumerate(ids)}
    chosen, tables = select_traits()
    rng = np.random.default_rng(SEED)
    families = np.unique(fams)
    n_test_fam = max(2, int(round(TEST_FAMILY_SHARE * len(families))))

    results, accept_rates = [], []
    for spec in chosen:
        t = tables[spec["file"]]
        sub_ids, y, X, cov = build_design(t, spec["trait"])
        idx = np.array([pos[s] for s in sub_ids])
        fam_of = fams[idx]
        g, b, acc = [], [], []
        for _ in range(N_REPEATS):
            te_fams = rng.choice(families, n_test_fam, replace=False)
            te = np.flatnonzero(np.isin(fam_of, te_fams))
            tr = np.flatnonzero(~np.isin(fam_of, te_fams))
            if len(te) < 30 or len(tr) < 100:
                continue
            K = G[np.ix_(idx[tr], idx[tr])]
            K_te = G[np.ix_(idx[te], idx[tr])]
            # covariates-only baseline
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pred_b = X[te] @ beta
            # GBLUP on the covariate residual, the genomic term added on top
            resid = y[tr] - X[tr] @ beta
            lam = cv_kernel_ridge([K], resid - resid.mean(), n_folds=5, seed=42)["lambda"]
            alpha = np.linalg.solve(K + lam * np.eye(len(tr)), resid - resid.mean())
            pred_g = pred_b + resid.mean() + K_te @ alpha
            g.append(float(spearmanr(pred_g, y[te]).statistic))
            b.append(float(spearmanr(pred_b, y[te]).statistic))
            # E3: support rule on held-out families, fitted on training mice only
            from src.evaluation.support import KNNSupport
            d_tr, v = np.linalg.eigh(K)
            feats_tr = v[:, -10:] * np.sqrt(np.maximum(d_tr[-10:], 1e-8))
            sup = KNNSupport(level=0.95).fit(feats_tr)
            feats_te = K_te @ v[:, -10:] / np.maximum(np.sqrt(np.maximum(d_tr[-10:], 1e-8)), 1e-8)
            acc.append(float(np.mean(sup.supported(feats_te))))
        gm, bm = np.array(g), np.array(b)
        row = {"file": spec["file"], "trait": spec["trait"], "n": int(len(y)), "repeats": len(g),
               "gblup_rho_median": round(float(np.median(gm)), 4),
               "gblup_ci95": [round(float(np.percentile(gm, 2.5)), 4),
                              round(float(np.percentile(gm, 97.5)), 4)],
               "covariates_rho_median": round(float(np.median(bm)), 4),
               "delta_median": round(float(np.median(gm - bm)), 4),
               "gblup_positive": bool(np.percentile(gm, 2.5) > 0),
               "gblup_beats_covariates": bool(np.median(gm) > np.median(bm)),
               "support_accept_rate": round(float(np.mean(acc)), 4)}
        results.append(row)
        accept_rates.append(np.mean(acc))
        print(f"  {row['trait']:26s} n={row['n']:>5}  GBLUP {row['gblup_rho_median']:+.3f} "
              f"{row['gblup_ci95']}  covars {row['covariates_rho_median']:+.3f}  "
              f"delta {row['delta_median']:+.3f}  accept {row['support_accept_rate']:.2f}", flush=True)

    e1 = sum(r["gblup_positive"] for r in results)
    e2 = sum(r["gblup_beats_covariates"] for r in results)
    e3 = float(np.mean(accept_rates))
    payload = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": "EMPIRICAL", "protocol": "docs/LAYER7_H7B_PROTOCOL.md",
        "lock": str(LOCK.relative_to(ROOT)), "axis": "G_genomic (machinery, second species)",
        "traits": results,
        "E1_primary": {"traits_with_positive_interval": e1, "required": E1_MIN_TRAITS,
                       "of": len(results), "pass": bool(e1 >= E1_MIN_TRAITS)},
        "E2": {"traits_where_gblup_beats_covariates": e2, "required": E2_MIN_TRAITS,
               "of": len(results), "pass": bool(e2 >= E2_MIN_TRAITS)},
        "E3": {"mean_support_accept_rate": round(e3, 4), "required": E3_MIN_ACCEPT,
               "pass": bool(e3 >= E3_MIN_ACCEPT)},
        "does_not_claim": ("machinery only. Says nothing about cattle accuracy, nothing about "
                           "cross-species transfer, and nothing about H7 inheritance, which stays void."),
        "changes_nothing": "no cattle threshold, panel, model or result is modified",
    }
    (OUT / "results_h7b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: payload[k] for k in ("E1_primary", "E2", "E3")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "unblind"], required=True)
    args = ap.parse_args()
    prepare() if args.stage == "prepare" else unblind()

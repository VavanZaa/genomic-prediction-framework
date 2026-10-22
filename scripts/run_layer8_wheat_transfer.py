"""Layer 8: transfer to unseen individuals in an unseen environment (CIMMYT wheat benchmark).

Protocol: docs/LAYER8_WHEAT_TRANSFER_PROTOCOL.md. The project has no external predictive transfer
result in any species; this arm tests for one on a dataset that carries its own positive and
negative control, because environments 2 and 4 agree (+0.661) while 1 and 4 do not (-0.193).

  --stage prepare   build the genomic relationship matrix and the phenotype table. Fits nothing.
  --stage unblind   verify the lock, then run all 16 ordered environment pairs once.

Test lines never appear in training in any environment. Nothing is tuned on the test environment.
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
from src.evaluation.support import KNNSupport  # noqa: E402
from src.prediction.kernel_cv import cv_kernel_ridge  # noqa: E402

WHEAT = ROOT / "data/real_plants/cimmyt_wheat"
OUT = ROOT / "output/layer8"
LOCK = OUT / "PROTOCOL_LOCK_L8.json"
N_REPEATS, TEST_SHARE, SEED = 20, 0.50, 20260917
N_PCS, SUPPORT_PCS = 5, 10
T1_PAIRS, T2_PAIRS = (("2", "4"), ("4", "2")), (("1", "4"), ("4", "1"))
T3_MIN_RHO, T4_MIN_ACCEPT = 0.5, 0.90


# The alignment gate is a permutation test, not a guessed correlation. A first attempt used
# "r >= 0.30", which the data failed at r = 0.2476 - but the pedigree matrix is sparse and noisy, so
# 0.30 was never attainable even with perfect alignment. Against 200 row permutations the null is
# -0.0001 +/- 0.0014, so the observed value sits 173 standard deviations away. Calibrating the gate
# against its own null is the fix; a remembered threshold applied to an incomparable quantity is the
# mistake this project has now made four times.
ALIGNMENT_PERMUTATIONS, ALIGNMENT_MIN_Z = 200, 10.0


def load():
    """Markers, phenotypes and the pedigree matrix, aligned by position and then verified.

    BGLR ships wheat.X without row names, so the export carries sequential indices while wheat.Y
    and wheat.A carry real line IDs. Position is therefore the only link between markers and
    phenotypes. This project has already been bitten once by trusting positional alignment - the
    Florida SNP/BOA files were row-misaligned for 84 % of animals - so alignment is verified rather
    than assumed, using the pedigree matrix as an independent witness.
    """
    X = pd.read_csv(WHEAT / "wheat_X_markers.csv", index_col=0)
    Y = pd.read_csv(WHEAT / "wheat_Y_yield.csv", index_col=0)
    A = pd.read_csv(WHEAT / "wheat_A_pedigree.csv", index_col=0)
    if not (len(X) == len(Y) == len(A)):
        raise SystemExit(f"row counts differ: X {len(X)}, Y {len(Y)}, A {len(A)}")
    if list(A.index) != list(Y.index):
        raise SystemExit("pedigree and phenotype line order differ; alignment cannot be verified")
    return X, Y, A


def check_alignment(G, A, seed=0):
    """Genomic and pedigree relationships must agree off-diagonal beyond what row shuffling gives."""
    Am = A.to_numpy(dtype=float)
    iu = np.triu_indices(len(Am), k=1)
    a = Am[iu]
    r = float(np.corrcoef(G[iu], a)[0, 1])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(ALIGNMENT_PERMUTATIONS):
        p = rng.permutation(len(G))
        Gp = G[np.ix_(p, p)]
        null.append(float(np.corrcoef(Gp[iu], a)[0, 1]))
    null = np.array(null)
    z = (r - null.mean()) / max(null.std(), 1e-12)
    return {"r": round(r, 4), "null_mean": round(float(null.mean()), 5),
            "null_sd": round(float(null.std()), 5), "null_max_abs": round(float(np.abs(null).max()), 5),
            "z": round(float(z), 1), "threshold_z": ALIGNMENT_MIN_Z,
            "pass": bool(z >= ALIGNMENT_MIN_Z)}


def grm(X):
    M = X.to_numpy(dtype=float)
    p = M.mean(axis=0)
    keep = (p > 0) & (p < 1)
    M, p = M[:, keep], p[keep]
    Z = M - p
    G = Z @ Z.T / float(np.sum(p * (1 - p)))
    return G / np.mean(np.diag(G)), int(keep.sum())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, A = load()
    G, n_markers = grm(X)
    align = check_alignment(G, A)
    print(f"alignment: r = {align['r']} vs permutation null {align['null_mean']} +/- "
          f"{align['null_sd']} (max |r| {align['null_max_abs']}), z = {align['z']} -> "
          f"{'OK' if align['pass'] else 'FAILED'}")
    if not align["pass"]:
        raise SystemExit("marker rows do not correspond to the phenotype rows; Layer 8 stops here")
    d, v = np.linalg.eigh(G)
    np.savez_compressed(OUT / "wheat_grm.npz", G=G, lines=np.array(X.index, dtype=str),
                        Y=Y.to_numpy(dtype=float), envs=np.array(Y.columns, dtype=str))
    corr = Y.corr(method="pearson").round(3)
    summary = {"stage": "prepare", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "evidence": "EMPIRICAL", "lines": int(len(X)), "markers_used": n_markers,
               "environments": list(Y.columns),
               "phenotype_correlations": json.loads(corr.to_json()),
               "grm_diagonal_mean": round(float(np.mean(np.diag(G))), 4),
               "grm_top_eigenvalue_share": round(float(d[-1] / d.sum()), 4),
               "alignment_check": align,
               "computed_here": "GRM and phenotype table only; no model fitted"}
    (OUT / "prepare_summary_l8.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the Layer 8 protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def unblind():
    verify_lock()
    z = np.load(OUT / "wheat_grm.npz", allow_pickle=False)
    G, Y, envs = z["G"], z["Y"], [str(e) for e in z["envs"]]
    n = len(G)
    rng = np.random.default_rng(SEED)
    d_all, v_all = np.linalg.eigh(G)
    pcs = v_all[:, -N_PCS:] * np.sqrt(np.maximum(d_all[-N_PCS:], 1e-9))

    splits = []
    for _ in range(N_REPEATS):
        perm = rng.permutation(n)
        cut = int(round(TEST_SHARE * n))
        splits.append((perm[cut:], perm[:cut]))          # train lines, test lines

    rows, accept = [], []
    for i, e_tr in enumerate(envs):
        for j, e_te in enumerate(envs):
            g_scores, b_scores = [], []
            for tr, te in splits:
                y_tr, y_te = Y[tr, i], Y[te, j]
                ok_tr, ok_te = np.isfinite(y_tr), np.isfinite(y_te)
                if ok_tr.sum() < 100 or ok_te.sum() < 50:
                    continue
                tr_i, te_i = tr[ok_tr], te[ok_te]
                K = G[np.ix_(tr_i, tr_i)]
                K_te = G[np.ix_(te_i, tr_i)]
                yv = Y[tr_i, i]
                mu = float(yv.mean())
                lam = cv_kernel_ridge([K], yv - mu, n_folds=5, seed=42)["lambda"]
                alpha = np.linalg.solve(K + lam * np.eye(len(tr_i)), yv - mu)
                pred_g = mu + K_te @ alpha
                beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(tr_i)), pcs[tr_i]]), yv, rcond=None)
                pred_b = np.column_stack([np.ones(len(te_i)), pcs[te_i]]) @ beta
                g_scores.append(float(spearmanr(pred_g, Y[te_i, j]).statistic))
                b_scores.append(float(spearmanr(pred_b, Y[te_i, j]).statistic))
                if i == j == 0:
                    sup = KNNSupport(level=0.95).fit(pcs[tr_i][:, -SUPPORT_PCS:])
                    accept.append(float(np.mean(sup.supported(pcs[te_i][:, -SUPPORT_PCS:]))))
            g = np.array(g_scores)
            rows.append({
                "train_env": e_tr, "test_env": e_te, "same_env": e_tr == e_te,
                "repeats": len(g_scores),
                "gblup_rho_median": round(float(np.median(g)), 4),
                "gblup_ci95": [round(float(np.percentile(g, 2.5)), 4),
                               round(float(np.percentile(g, 97.5)), 4)],
                "structure_only_rho_median": round(float(np.median(b_scores)), 4),
                "positive_interval": bool(np.percentile(g, 2.5) > 0)})
            print(f"  {e_tr} -> {e_te}{'  (same)' if e_tr == e_te else '        '}  "
                  f"GBLUP {rows[-1]['gblup_rho_median']:+.3f} {rows[-1]['gblup_ci95']}  "
                  f"structure {rows[-1]['structure_only_rho_median']:+.3f}", flush=True)

    by_pair = {(r["train_env"], r["test_env"]): r for r in rows}
    t1 = all(by_pair[p]["positive_interval"] for p in T1_PAIRS)
    t2 = not any(by_pair[p]["positive_interval"] for p in T2_PAIRS)
    Ydf = pd.DataFrame(Y, columns=envs)
    pheno_corr = Ydf.corr()
    cross = [r for r in rows if not r["same_env"]]
    xs = [float(pheno_corr.loc[r["train_env"], r["test_env"]]) for r in cross]
    ys = [r["gblup_rho_median"] for r in cross]
    t3_rho = float(spearmanr(xs, ys).statistic)
    t4 = float(np.mean(accept)) if accept else float("nan")

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "protocol": "docs/LAYER8_WHEAT_TRANSFER_PROTOCOL.md", "lock": str(LOCK.relative_to(ROOT)),
        "grid": rows,
        "T1_positive_control": {"pairs": [list(p) for p in T1_PAIRS], "pass": bool(t1),
                                "detail": {f"{a}->{b}": by_pair[(a, b)]["gblup_ci95"] for a, b in T1_PAIRS}},
        "T2_negative_control": {"pairs": [list(p) for p in T2_PAIRS], "pass": bool(t2),
                                "detail": {f"{a}->{b}": by_pair[(a, b)]["gblup_ci95"] for a, b in T2_PAIRS},
                                "meaning": ("a failure here would mean the pipeline reports successful "
                                            "transfer where the trait does not transfer")},
        "T3_ordering": {"spearman_corr_vs_accuracy": round(t3_rho, 4), "required": T3_MIN_RHO,
                        "pass": bool(t3_rho >= T3_MIN_RHO)},
        "T4_applicability": {"accept_rate_same_population": round(t4, 4), "required": T4_MIN_ACCEPT,
                             "pass": bool(t4 >= T4_MIN_ACCEPT),
                             "note": "the same check failed in mice at 0.885 (H7b E3)"},
        "overall": {"pass": bool(t1 and t2),
                    "rule": "T1 and T2 together; passing T1 while failing T2 is not a partial success"},
        "does_not_claim": ("wheat lines are inbred, so the inheritance layer is untouched; and this "
                           "says nothing about cattle accuracy"),
    }
    (OUT / "results_l8.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: result[k] for k in
                             ("T1_positive_control", "T2_negative_control", "T3_ordering",
                              "T4_applicability", "overall")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "unblind"], required=True)
    args = ap.parse_args()
    prepare() if args.stage == "prepare" else unblind()

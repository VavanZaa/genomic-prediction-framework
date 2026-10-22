"""How many shared markers does the k-NN support rule need? (docs/COVERAGE_CALIBRATION.md)

The 5,000-marker clause in the 6C3 protocol bounded a method Amendment 4 withdrew (mean-filling).
The shared-marker subspace that replaced it has never had a coverage requirement established. This
derives one from Florida data alone: no external genotype is read.

Out-of-domain animals are created by ancestry holdout rather than borrowed from an external
population, so the requirement can be set without touching the data it will later gate.

The criterion and decision rule are fixed in the document above before this runs.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.ood_detector import GenomicOODDetector  # noqa: E402
from src.evaluation.risk_calibration import load_frozen_thresholds  # noqa: E402
from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction  # noqa: E402
from run_layer6c_external_validation import FloridaReference  # noqa: E402

OUT = ROOT / "output/coverage_calibration"
THRESH = ROOT / "output/ood_calibration/thresholds_rev1.json"
GRID = [500, 1000, 2000, 2891, 4000, 5067, 8000, 12000, 20087]
DRAWS = 10
# Quantiles, not absolute cut-offs: absolute ones left only 10 out-of-domain animals (see the doc).
Q_TRAIN_PCT, Q_OUT_PCT, IN_HOLDOUT = 70, 90, 0.25
R_OUT_MIN, R_IN_MAX = 0.90, 0.05
SEED = 20260916
NOTE = {2891: "Decker worldwide panel", 5067: "slick/hairy shared subspace", 20087: "full 6C panel"}


def main():
    ref = FloridaReference()
    pos = {a: i for i, a in enumerate(ref.ids)}
    tr = np.array([pos[a] for a in ref.train_ids])
    G = ref.snp[tr]
    mean = np.nanmean(G, axis=0)
    G = np.where(np.isnan(G), mean, G)

    pa, pb, _, _ = breed_frequencies(ref.snp, ref.boa)
    q = estimate_brahman_fraction(G, pa, pb)
    q_train_max = float(np.percentile(q, Q_TRAIN_PCT))
    q_out_min = float(np.percentile(q, Q_OUT_PCT))
    in_pool = np.flatnonzero(q <= q_train_max)
    out_idx = np.flatnonzero(q >= q_out_min)
    rng = np.random.default_rng(SEED)
    hold = rng.choice(in_pool, int(round(IN_HOLDOUT * len(in_pool))), replace=False)
    fit_idx = np.setdiff1d(in_pool, hold)
    print(f"ancestry coordinate: P{Q_TRAIN_PCT} = {q_train_max:.3f}, P{Q_OUT_PCT} = {q_out_min:.3f}")
    print(f"support fit: {len(fit_idx)} animals (q <= {q_train_max:.3f})")
    print(f"in-domain holdout: {len(hold)}   out-of-domain holdout: {len(out_idx)} (q >= {q_out_min:.3f})")
    if len(out_idx) < 30:
        raise SystemExit("too few out-of-domain animals for a stable estimate")

    thr = load_frozen_thresholds(THRESH)
    n_markers = G.shape[1]
    rows = []
    for m in GRID:
        r_out, r_in = [], []
        for d in range(DRAWS):
            sel = (np.arange(n_markers) if m >= n_markers
                   else rng.choice(n_markers, m, replace=False))
            Gf = G[np.ix_(fit_idx, sel)]
            qf = estimate_brahman_fraction(Gf, pa[sel], pb[sel])
            det = GenomicOODDetector(
                n_components=15,
                calibrated_thresholds={"ranking": thr["ranking"], "absolute": thr["absolute"]})
            det.fit(Gf, ancestry_qb_train=qf, env_ids_train=["Semin"] * len(fit_idx))
            for idx, sink in ((out_idx, r_out), (hold, r_in)):
                X = G[np.ix_(idx, sel)]
                qq = estimate_brahman_fraction(X, pa[sel], pb[sel])
                a = det.assess_cohort(X, ancestry_qb_cohort=qq, env_ids_cohort=["Semin"] * len(idx))
                sink.append(float(np.mean([not x.allow_ranking_only for x in a])))
            if m >= n_markers:
                break  # the full panel has only one possible subset
        row = {"markers": m, "draws": len(r_out),
               "R_out_median": round(float(np.median(r_out)), 4),
               "R_in_median": round(float(np.median(r_in)), 4),
               "R_out_min": round(float(np.min(r_out)), 4),
               "R_in_max": round(float(np.max(r_in)), 4),
               "sufficient": bool(np.median(r_out) >= R_OUT_MIN and np.median(r_in) <= R_IN_MAX)}
        if m in NOTE:
            row["note"] = NOTE[m]
        rows.append(row)
        print(f"  m={m:>6,}: R_out {row['R_out_median']:.3f}  R_in {row['R_in_median']:.3f}  "
              f"{'sufficient' if row['sufficient'] else 'not sufficient'}"
              f"{'   <- ' + NOTE[m] if m in NOTE else ''}", flush=True)

    # smallest sufficient m for which every larger grid value is also sufficient
    minimum = None
    for i, row in enumerate(rows):
        if row["sufficient"] and all(r["sufficient"] for r in rows[i:]):
            minimum = row["markers"]
            break

    decker_ok = minimum is not None and minimum <= 2891
    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "question": "minimum shared markers for the k-NN support rule in the subspace method",
        "protocol": "docs/COVERAGE_CALIBRATION.md (rule fixed before running)",
        "design": {"support_fit_animals": int(len(fit_idx)),
                   "q_train_percentile": Q_TRAIN_PCT, "q_train_max": round(q_train_max, 4),
                   "in_domain_holdout": int(len(hold)), "out_of_domain_holdout": int(len(out_idx)),
                   "q_out_percentile": Q_OUT_PCT, "q_out_min": round(q_out_min, 4),
                   "draws_per_grid_value": DRAWS, "seed": SEED},
        "criterion": {"R_out_min": R_OUT_MIN, "R_in_max": R_IN_MAX},
        "curve": rows,
        "minimum_required_markers": minimum,
        "decker_2891_sufficient": bool(decker_ok),
        "consequence": (
            "The 6C3 coverage clause is superseded by this minimum and Decker becomes testable under a "
            "new pre-registration; any result carries the label that its coverage requirement was set "
            "after its coverage was known." if decker_ok else
            "6C3 stays inconclusive and the Decker panel is not analysed. No second calibration is run."),
        "disclosure": ("Run with knowledge that Decker yields 2,891 markers, at the project owner's "
                       "request. Florida data only; grid includes 2,891 with values above and below; "
                       "full curve reported; the rule was fixed before the run."),
        "limits": [
            "the out-of-domain group is high-Brahman Florida cattle, closer to training than a foreign "
            "breed would be, so this requirement is conservative rather than permissive",
            "markers are drawn at random; Decker's 2,891 are the SNP50 backbone, not a random subset",
            "this gates the support rule only, not the coverage a prediction score needs",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nminimum required markers: {minimum}")
    print(f"Decker's 2,891 sufficient: {decker_ok}")
    print(f"wrote {(OUT / 'coverage_calibration.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Does genomic prediction beat an ancestry baseline once the training set is large enough?

C14 found that for the Florida traits no genomic model beats breed composition, on 1,754 training
animals. The heritability estimate then showed genuine within-ancestry signal (hair length 0.160
[0.100, 0.230]), so the obvious suspect is sample size. The Mizzou hair-shedding project has
10,393 genotyped animals on the same harmonised panel, which is six times Florida's training set.

This is a **methodological study, not a blind test**: these phenotypes were already opened during
6C1, so nothing here is pre-registered evidence about transfer. It answers one question only -
whether the accuracy gap between genomic prediction and an ancestry baseline closes as n grows.

Design, fixed before the curve was computed:
  - phenotype: hair-shedding score centred on the scoring EVENT (date + mean_apparent_high +
    mean_day_length), the corrected contemporary unit from Amendment 2, averaged per animal;
  - models: AncestryOnly (estimated Brahman coordinate, linear + quadratic) and GBLUP on the
    VanRaden GRM at the shared panel, lambda chosen by training-only 5-fold CV;
  - training sizes 250, 500, 1000, 2000, 4000, 6000, with 10 replicate draws each;
  - two splits, both reported: RANDOM (optimistic - relatives shared between train and test) and
    EVENT-HELD-OUT (whole scoring events held out, closer to a new herd);
  - metric: Spearman correlation between prediction and event-centred score in the held-out set.

  --stage extract   read panel dosages out of the 3.2 GB VCF once and cache them
  --stage curve     run the learning curve from the cache
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.prediction.kernel_cv import cv_kernel_ridge  # noqa: E402
from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction  # noqa: E402
from src.real_data.harmonisation import vcf_to_counted_dosage  # noqa: E402

EXT = ROOT / "data/external"
MIZ_VCF = EXT / "mizzou_hair_shedding/Durbin_etal_2021_MizzouHairShedding_genotypes.vcf.gz"
MIZ_PHENO = EXT / "mizzou_hair_shedding/Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv"
LOOKUP, PANEL, FL = EXT / "allele_lookup_f250.csv", EXT / "panel_6c_markers.csv", EXT / "florida_panel_6c.npz"
OUT = ROOT / "output/mizzou_scale"
CACHE = OUT / "mizzou_panel_dosages.npz"

# extended to the largest training set the data allow (10,225 animals minus the held-out test set),
# because the curve had not flattened at 6,000 and the stopping point had been arbitrary
TRAIN_SIZES = [250, 500, 1000, 2000, 4000, 6000, 8000, 8700]
N_REPLICATES = 10
TEST_SIZE = 1500
SEED = 20260916


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)
    lookup = pd.read_csv(LOOKUP)
    print("reading the VCF; this is the slow step ...", flush=True)
    ids, got, X, reasons = vcf_to_counted_dosage(MIZ_VCF, lookup, markers=panel["marker"], key="position")
    markers = [str(m) for m in got]
    print(f"animals {len(ids)}, markers recovered {len(markers)}, missing {np.isnan(X).mean():.4f}")
    np.savez_compressed(CACHE, ids=np.array(ids, dtype=str), markers=np.array(markers, dtype=str),
                        dosage=X.astype(np.float32))
    (OUT / "extract_summary.json").write_text(json.dumps(
        {"animals": len(ids), "markers": len(markers), "exclusions": {k: int(v) for k, v in reasons.items()},
         "missing_share": round(float(np.isnan(X).mean()), 5)}, indent=2), encoding="utf-8")
    print(f"cached {CACHE.relative_to(ROOT)}")


def phenotypes(ids):
    """Event-centred hair-shedding score per animal, on the corrected contemporary unit."""
    ph = pd.read_csv(MIZ_PHENO, dtype={"id": str})
    ph = ph[ph["hair_score"].notna()].copy()
    # 454 of 36,899 records carry no scoring date or weather covariates, so they have no identifiable
    # contemporary unit. Under the Amendment 2 rule they cannot be event-centred and are dropped
    # rather than centred on a group that does not exist. Forced by the data, before any model runs.
    has_unit = (ph["date_score_recorded"].notna() & ph["mean_apparent_high"].notna()
                & ph["mean_day_length"].notna())
    dropped = int((~has_unit).sum())
    ph = ph[has_unit].copy()
    ph["event"] = (ph["date_score_recorded"].astype(str) + "|"
                   + ph["mean_apparent_high"].astype(str) + "|" + ph["mean_day_length"].astype(str))
    ph["centred"] = ph["hair_score"] - ph.groupby("event")["hair_score"].transform("mean")
    per = ph.groupby("id").agg(phenotype=("centred", "mean"),
                               event=("event", lambda s: s.value_counts().index[0]),
                               records=("centred", "size")).reset_index()
    print(f"phenotype records: {len(ph):,} used, {dropped:,} dropped for having no scoring event; "
          f"{ph['event'].nunique():,} events, {len(per):,} animals")
    per = per.set_index("id").reindex([str(i) for i in ids])
    return per


def gblup_predict(K_tr, y_tr, K_te_tr, lam):
    mu = float(np.mean(y_tr))
    alpha = np.linalg.solve(K_tr + lam * np.eye(len(y_tr)), y_tr - mu)
    return mu + K_te_tr @ alpha


def curve():
    d = np.load(CACHE, allow_pickle=False)
    ids, X = [str(i) for i in d["ids"]], d["dosage"]          # float32: two float64 copies of a
    markers = [str(m) for m in d["markers"]]                  # 10k x 20k matrix exhausted memory
    per = phenotypes(ids)
    ok = per["phenotype"].notna().to_numpy()
    X, per = X[ok], per[ok]
    ids = list(np.array(ids)[ok])
    print(f"animals with genotype and phenotype: {len(ids):,}  markers: {len(markers):,}")

    mean = np.nanmean(X, axis=0).astype(np.float32)
    X = np.where(np.isnan(X), mean, X).astype(np.float32)
    p = np.clip(mean.astype(np.float64) / 2.0, 1e-3, 1 - 1e-3)
    Z = (X - (2 * p).astype(np.float32)).astype(np.float32)
    denom = float(2.0 * np.sum(p * (1 - p)))

    # ancestry coordinate from Florida-derived breed frequencies, as elsewhere in the project
    fl = np.load(FL, allow_pickle=False)
    fl_markers = [str(m) for m in fl["markers"]]
    pa_full, pb_full, _, _ = breed_frequencies(fl["snp"].astype(float), fl["boa"].astype(float))
    idx = {m: i for i, m in enumerate(fl_markers)}
    take = np.array([idx[m] for m in markers])
    q = estimate_brahman_fraction(X.astype(np.float64), pa_full[take], pb_full[take])
    del X

    y = per["phenotype"].to_numpy(dtype=float)
    events = per["event"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ("random", "event_held_out"):
        for n_train in TRAIN_SIZES:
            g_scores, a_scores = [], []
            for rep in range(N_REPLICATES):
                if split == "random":
                    perm = rng.permutation(len(y))
                    te, tr = perm[:TEST_SIZE], perm[TEST_SIZE:TEST_SIZE + n_train]
                else:
                    uniq = rng.permutation(np.unique(events))
                    te_mask = np.zeros(len(y), dtype=bool)
                    for ev in uniq:
                        te_mask |= events == ev
                        if te_mask.sum() >= TEST_SIZE:
                            break
                    te = np.flatnonzero(te_mask)[:TEST_SIZE]
                    pool = np.flatnonzero(~te_mask)
                    tr = rng.permutation(pool)[:n_train]
                if len(tr) < n_train:
                    continue
                Ztr = Z[tr]
                K = (Ztr @ Ztr.T).astype(np.float64) / denom
                lam = cv_kernel_ridge([K], y[tr] - y[tr].mean(), n_folds=5, seed=42)["lambda"]
                K_te = (Z[te] @ Ztr.T).astype(np.float64) / denom
                del Ztr
                pred_g = gblup_predict(K, y[tr], K_te, lam)
                A = np.column_stack([np.ones(len(tr)), q[tr], q[tr] ** 2])
                beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
                pred_a = np.column_stack([np.ones(len(te)), q[te], q[te] ** 2]) @ beta
                g_scores.append(float(spearmanr(pred_g, y[te]).statistic))
                a_scores.append(float(spearmanr(pred_a, y[te]).statistic))
            row = {"split": split, "n_train": n_train, "replicates": len(g_scores),
                   "gblup_rho": round(float(np.median(g_scores)), 4),
                   "ancestry_rho": round(float(np.median(a_scores)), 4),
                   "delta_median": round(float(np.median(np.array(g_scores) - np.array(a_scores))), 4),
                   "delta_ci95": [round(float(np.percentile(np.array(g_scores) - np.array(a_scores), 2.5)), 4),
                                  round(float(np.percentile(np.array(g_scores) - np.array(a_scores), 97.5)), 4)],
                   "gblup_beats_ancestry_share": round(float(np.mean(np.array(g_scores) > np.array(a_scores))), 3)}
            rows.append(row)
            print(f"  {split:14s} n={n_train:>5}: GBLUP {row['gblup_rho']:+.3f}  "
                  f"ancestry {row['ancestry_rho']:+.3f}  delta {row['delta_median']:+.3f} "
                  f"{row['delta_ci95']}  wins {row['gblup_beats_ancestry_share']:.2f}", flush=True)

    # Fit the standard genomic-prediction scaling law r(n) = sqrt(n h2 / (n h2 + Me)) to each split.
    # Me, the effective number of independent chromosome segments, is the only free parameter once
    # h2 is fixed, and it converts the curve into "how many animals for a target accuracy".
    scaling = {}
    for split in ("random", "event_held_out"):
        pts = [(r["n_train"], r["gblup_rho"]) for r in rows if r["split"] == split]
        n_arr = np.array([p[0] for p in pts], dtype=float)
        r_arr = np.array([p[1] for p in pts], dtype=float)
        best = None
        for h2 in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for me in np.logspace(2.5, 5.5, 400):
                pred = np.sqrt(n_arr * h2 / (n_arr * h2 + me))
                sse = float(np.sum((pred - r_arr) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, h2, float(me))
        _, h2, me = best
        needed = {}
        for target in (0.35, 0.40, 0.50, 0.60):
            n_req = me * target ** 2 / (h2 * (1 - target ** 2))
            needed[str(target)] = int(round(n_req)) if n_req < 1e7 else None
        scaling[split] = {"fitted_h2": h2, "fitted_Me": round(me), "rmse": round(np.sqrt(best[0] / len(pts)), 4),
                          "animals_needed_for_accuracy": needed}
        print(f"\n{split}: fitted h2 {h2}, Me {round(me)}, animals needed -> {needed}")

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "scaling_law": scaling,
        "status": "methodological study, NOT a blind test: these phenotypes were opened in 6C1",
        "question": "does the genomic-vs-ancestry gap close as the training set grows?",
        "population": "Mizzou hair shedding, event-centred score, same 20,087-marker panel",
        "animals_used": len(ids), "markers": len(markers),
        "design": {"train_sizes": TRAIN_SIZES, "replicates": N_REPLICATES, "test_size": TEST_SIZE,
                   "seed": SEED, "splits": ["random", "event_held_out"]},
        "curve": rows,
        "limits": [
            "the trait is a 1-5 shedding score, not the Florida hair-length measurement",
            "the random split shares relatives between train and test and is optimistic",
            "event-held-out approximates a new herd-day, but the same farm can recur across events",
            "ancestry here is the Angus-Brahman coordinate, which for US beef breeds is a projection "
            "rather than a true ancestry fraction",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale_curve.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "scale_curve.csv", index=False)
    print(f"\nwrote {(OUT / 'scale_curve.json').relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["extract", "curve"], required=True)
    args = ap.parse_args()
    extract() if args.stage == "extract" else curve()

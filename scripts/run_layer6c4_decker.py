"""Layer 6C4: confirmation test for C3 on the Decker worldwide panel.

Protocol: docs/LAYER6C4_DECKER_PROTOCOL.md. Two stages:

  --stage prepare   harmonise the Decker SNP50 genotypes onto the 6C panel and run the gates:
                    usable-marker count, A/B manifest consistency, duplicate check. **Computes no
                    ancestry coordinate, no support score and no acceptance decision** - those are
                    the endpoint quantities, and this arm is a confirmation test, so they wait for
                    the lock.
  --stage unblind   verify the 6C4 lock, then compute the endpoints once.

Decker genotypes are Illumina A/B coded (1 = A, 2 = B, 0 = missing), and the allele lookup records
which Illumina allele Florida counted, so dosages map directly with no strand or nucleotide
inference. Whether the two manifests agree on A/B is what the consistency gate tests.
"""
import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402
from src.real_data.liftover import ChainLiftOver  # noqa: E402

EXT = ROOT / "data/external"
DECKER = EXT / "decker2014_worldwide"
ZIP = DECKER / "WorldWideCattleDryad.zip"
PED = "DataDryad/ww_cattle_dryad_20Sept2013.ped"
MAP = DECKER / "extracted/ww_cattle_dryad_20Sept2013.map"
PEDIND = DECKER / "extracted/ww_cattle_dryad_20Sept2013.pedind"
CHAIN = EXT / "ucsc_chain/bosTau6ToBosTau9.over.chain.gz"
PANEL = EXT / "panel_6c_markers.csv"
LOOKUP = EXT / "allele_lookup_f250.csv"
FL = EXT / "florida_panel_6c.npz"
PHENO = ROOT / "data/real_cattle/florida_multibreed/Thermo.Phenotypes.csv"
OUT = ROOT / "output/layer6c4"
LOCK = OUT / "PROTOCOL_LOCK_6C4.json"
THRESH = ROOT / "output/ood_calibration/thresholds_rev1.json"

MIN_USABLE_MARKERS = 2000
# Amendment 1: the A/B gate is a direct flip test, not a correlation. A flipped manifest assignment
# moves a frequency from p to 1-p, which is unambiguous only away from p = 0.5, so the test is
# evaluated where the Florida frequency is extreme. The correlation it replaces is still reported,
# descriptively: at n = 20 Decker Angus it measures sampling noise more than agreement.
AB_FLIP_DIFF = 0.5
AB_FLIP_MAX_SHARE = 0.01
AB_INFORMATIVE_LOW, AB_INFORMATIVE_HIGH = 0.15, 0.85
AB_MIN_INFORMATIVE = 200
DUPLICATE_CORR = 0.95
FLORIDA_ANGUS_DECILE = 0.10
E1_REFUSAL_MIN, E1_MIN_BREEDS = 0.90, 10
FLORIDA_Q_RANGE = (0.035, 0.915)
N_BOOT, SEED = 2000, 20260916
EXCLUDED_LABELS = ("AN", "BR")  # reference-adjacent: excluded from endpoints, see the protocol


def matched_markers():
    """Panel markers recoverable from the Decker map, with the Illumina allele Florida counted."""
    panel = pd.read_csv(PANEL, dtype={"chrom": str})
    lookup = pd.read_csv(LOOKUP, usecols=["marker", "florida_counted_code"])
    panel = panel.merge(lookup, on="marker", how="left")
    mp = pd.read_csv(MAP, sep=r"\s+", header=None, names=["chr", "snp", "cm", "bp"],
                     dtype={"chr": str, "snp": str})
    mp["ped_index"] = np.arange(len(mp))
    lifted = ChainLiftOver(CHAIN).convert_many(mp["chr"], mp["bp"])
    mp["new_chrom"] = [x[0] if x else None for x in lifted]
    mp["new_pos"] = [x[1] if x else -1 for x in lifted]
    m = panel.merge(mp.dropna(subset=["new_chrom"]), left_on=["chrom", "pos_bp"],
                    right_on=["new_chrom", "new_pos"], how="inner").drop_duplicates(subset=["marker"])
    usable = m[m["florida_counted_code"].isin(["A", "B"])].copy()
    usable["counted_ped_code"] = np.where(usable["florida_counted_code"] == "A", "1", "2")
    stats = {"decker_markers": int(len(mp)), "lifted": int(mp["new_chrom"].notna().sum()),
             "matched_by_position": int(len(m)),
             "dropped_no_florida_ab_code": int(len(m) - len(usable)),
             "usable_markers": int(len(usable))}
    return usable.reset_index(drop=True), stats


def read_ped_dosages(usable):
    """Counted-allele dosage for each animal at the usable markers, streamed from the zipped .ped."""
    idx = usable["ped_index"].to_numpy()
    counted = usable["counted_ped_code"].to_numpy()
    a_cols, b_cols = 6 + 2 * idx, 6 + 2 * idx + 1
    ids, rows = [], []
    with zipfile.ZipFile(ZIP) as z, z.open(PED) as fh:
        for line in fh:
            f = line.split()
            ids.append(f[1].decode())
            a = np.array([f[i] for i in a_cols])
            b = np.array([f[i] for i in b_cols])
            want = np.char.encode(counted.astype(str))
            missing = (a == b"0") | (b == b"0")
            d = (a == want).astype(float) + (b == want).astype(float)
            d[missing] = np.nan
            rows.append(d)
    return ids, np.vstack(rows)


def florida_angus_frequencies(usable):
    """Counted-allele frequency among the most Angus-like Florida animals, by published ancestry."""
    z = np.load(FL, allow_pickle=False)
    ids, markers = [str(x) for x in z["ids"]], [str(x) for x in z["markers"]]
    snp = z["snp"].astype(float)
    ph = pd.read_csv(PHENO, dtype={"UFID": str}).set_index("UFID")
    gbc = ph.reindex(ids)["BrahmanGBC"].to_numpy(dtype=float)
    ok = np.isfinite(gbc)
    cut = np.quantile(gbc[ok], FLORIDA_ANGUS_DECILE)
    sel = ok & (gbc <= cut)
    cols = {m: i for i, m in enumerate(markers)}
    take = np.array([cols[m] for m in usable["marker"]])
    sub = snp[np.ix_(np.flatnonzero(sel), take)]
    return np.nanmean(sub, axis=0) / 2.0, int(sel.sum()), float(cut)


def prepare():
    usable, stats = matched_markers()
    print(json.dumps(stats, indent=2))
    if stats["usable_markers"] < MIN_USABLE_MARKERS:
        raise SystemExit(f"usable markers {stats['usable_markers']} < {MIN_USABLE_MARKERS}: "
                         "6C4 stops here and is reported as insufficient coverage")

    print("streaming the .ped ...", flush=True)
    ids, X = read_ped_dosages(usable)
    print(f"animals {len(ids)}, markers {X.shape[1]}, missing {np.isnan(X).mean():.4f}")

    pedind = pd.read_csv(PEDIND, sep=r"\s+", header=None,
                         names=["breed_code", "id", "pid", "mid", "sex", "breed"], dtype=str)
    if pedind["id"].tolist() != ids:
        raise SystemExit("the .ped and .pedind disagree on animal order")

    # --- A/B manifest consistency gate: Angus against Angus, frequencies only -------------------
    fl_freq, n_fl, cut = florida_angus_frequencies(usable)
    an = pedind["breed"].to_numpy() == "AN"
    dk_freq = np.nanmean(X[an], axis=0) / 2.0
    ok = np.isfinite(fl_freq) & np.isfinite(dk_freq)
    r = float(np.corrcoef(fl_freq[ok], dk_freq[ok])[0, 1])
    informative = ok & ((fl_freq < AB_INFORMATIVE_LOW) | (fl_freq > AB_INFORMATIVE_HIGH))
    n_inf = int(informative.sum())
    flipped = int((np.abs(fl_freq[informative] - dk_freq[informative]) > AB_FLIP_DIFF).sum())
    flip_share = flipped / n_inf if n_inf else float("nan")
    gate_pass = n_inf >= AB_MIN_INFORMATIVE and flip_share <= AB_FLIP_MAX_SHARE
    print(f"A/B flip test: {flipped} of {n_inf:,} informative markers flipped "
          f"(share {flip_share:.5f}, limit {AB_FLIP_MAX_SHARE})")
    print(f"  descriptive only: frequency correlation r = {r:.4f} over {int(ok.sum()):,} markers "
          f"(Decker AN n={int(an.sum())}, Florida BrahmanGBC <= {cut:.3f} n={n_fl})")

    # --- duplicate check ------------------------------------------------------------------------
    z = np.load(FL, allow_pickle=False)
    markers = [str(x) for x in z["markers"]]
    cols = {m: i for i, m in enumerate(markers)}
    take = np.array([cols[m] for m in usable["marker"]])
    F = z["snp"].astype(float)[:, take]
    fmean = np.nanmean(F, axis=0)
    F = np.where(np.isnan(F), fmean, F) - fmean
    Xi = np.where(np.isnan(X), fmean, X) - fmean
    num = Xi @ F.T
    corr = num / np.sqrt(np.einsum("ij,ij->i", Xi, Xi)[:, None] * np.einsum("ij,ij->i", F, F)[None, :])
    max_corr = corr.max(axis=1)
    duplicates = int((max_corr > DUPLICATE_CORR).sum())
    print(f"duplicates with Florida animals: {duplicates} (max correlation {max_corr.max():.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "prepared_decker.npz", ids=np.array(ids, dtype=str), dosage=X,
                        markers=usable["marker"].to_numpy().astype(str),
                        breed=pedind["breed"].to_numpy().astype(str),
                        max_corr_florida=max_corr)
    summary = {
        "stage": "prepare", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": "EMPIRICAL", "protocol": "docs/LAYER6C4_DECKER_PROTOCOL.md",
        "harmonisation": stats,
        "coverage_requirement": MIN_USABLE_MARKERS,
        "coverage_note": ("between 2,000 and 2,891 the arm runs with expected out-of-domain refusal "
                          "about 0.97 rather than 0.99" if stats["usable_markers"] < 2891 else
                          "at or above the 2,891 marker point the calibration measured directly"),
        "animals": len(ids), "breeds": int(pedind["breed"].nunique()),
        "genotype_missing_share": round(float(np.isnan(X).mean()), 5),
        "ab_consistency": {
            "test": "flip test on markers with extreme Florida frequency (Amendment 1)",
            "informative_markers": n_inf, "flipped_markers": flipped,
            "flip_share": round(flip_share, 5), "limit": AB_FLIP_MAX_SHARE,
            "informative_window": [AB_INFORMATIVE_LOW, AB_INFORMATIVE_HIGH],
            "frequency_correlation_descriptive_only": round(r, 4),
            "markers_compared": int(ok.sum()), "decker_angus_n": int(an.sum()),
            "florida_n": n_fl, "pass": bool(gate_pass)},
        "duplicates_with_florida": duplicates,
        "max_correlation_with_florida": round(float(max_corr.max()), 4),
        "computed_here": "harmonisation, allele-frequency gate, duplicate check",
        "not_computed_here": "ancestry coordinates, support scores, acceptance decisions",
    }
    (OUT / "prepare_summary_6c4.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nA/B consistency gate: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("the arm is a harmonisation failure; no endpoint may be computed")
    print(f"wrote {(OUT / 'prepare_summary_6c4.json').relative_to(ROOT)}")


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the 6C4 protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def unblind():
    verify_lock()
    from src.evaluation.ood_detector import GenomicOODDetector
    from src.evaluation.risk_calibration import load_frozen_thresholds
    from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction
    from run_layer6c_external_validation import FloridaReference

    summary = json.loads((OUT / "prepare_summary_6c4.json").read_text(encoding="utf-8"))
    if not summary["ab_consistency"]["pass"]:
        raise SystemExit("the A/B consistency gate failed; this arm has no endpoints")
    d = np.load(OUT / "prepared_decker.npz", allow_pickle=False)
    X, breed = d["dosage"].astype(float), np.array([str(b) for b in d["breed"]])
    keep_markers = [str(m) for m in d["markers"]]

    ref = FloridaReference()
    cols = {m: i for i, m in enumerate(ref.markers)}
    take = np.array([cols[m] for m in keep_markers])
    pos = {a: i for i, a in enumerate(ref.ids)}
    tr = np.array([pos[a] for a in ref.train_ids])
    G = ref.snp[tr][:, take]
    mean = np.nanmean(G, axis=0)
    G = np.where(np.isnan(G), mean, G)
    Xf = np.where(np.isnan(X), mean, X)

    pa_full, pb_full, _, _ = breed_frequencies(ref.snp, ref.boa)
    pa, pb = pa_full[take], pb_full[take]
    q_tr = estimate_brahman_fraction(G, pa, pb)
    q_dk = estimate_brahman_fraction(Xf, pa, pb)
    thr = load_frozen_thresholds(THRESH)
    det = GenomicOODDetector(n_components=15,
                             calibrated_thresholds={"ranking": thr["ranking"], "absolute": thr["absolute"]})
    det.fit(G, ancestry_qb_train=q_tr, env_ids_train=["Semin"] * len(G))
    ass = det.assess_cohort(Xf, ancestry_qb_cohort=q_dk, env_ids_cohort=["Decker"] * len(Xf))
    refused = np.array([not a.allow_ranking_only for a in ass])
    numeric_refused = np.array([not a.allow_numeric_prediction for a in ass])

    out_of_ref = ~np.isin(breed, EXCLUDED_LABELS)
    inside = (q_dk >= FLORIDA_Q_RANGE[0]) & (q_dk <= FLORIDA_Q_RANGE[1])
    e1_set = out_of_ref & inside
    breed_median_q = pd.Series(q_dk[out_of_ref]).groupby(breed[out_of_ref]).median()
    breeds_inside = breed_median_q[(breed_median_q >= FLORIDA_Q_RANGE[0])
                                   & (breed_median_q <= FLORIDA_Q_RANGE[1])]
    rng = np.random.default_rng(SEED)
    sub_breeds = breed[e1_set]
    uniq = np.unique(sub_breeds)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        vals = np.concatenate([refused[e1_set][sub_breeds == b] for b in pick])
        boot.append(vals.mean())
    e1_rate = float(refused[e1_set].mean()) if e1_set.any() else float("nan")
    feasible = len(breeds_inside) >= E1_MIN_BREEDS

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "protocol": "docs/LAYER6C4_DECKER_PROTOCOL.md", "lock": str(LOCK.relative_to(ROOT)),
        "usable_markers": summary["harmonisation"]["usable_markers"],
        "E1_primary": {
            "animals_in_set": int(e1_set.sum()),
            "breeds_with_median_q_inside_florida_range": int(len(breeds_inside)),
            "required_breeds": E1_MIN_BREEDS, "feasible": bool(feasible),
            "knn_refusal_rate": round(e1_rate, 4),
            "breed_bootstrap_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                     round(float(np.percentile(boot, 97.5)), 4)] if boot else None,
            "threshold": E1_REFUSAL_MIN,
            "pass": bool(feasible and e1_rate >= E1_REFUSAL_MIN),
            "inconclusive_reason": None if feasible else
            f"only {len(breeds_inside)} breeds have a median coordinate inside the Florida range",
        },
        "E2_refusal_breadth": {
            "out_of_reference_refusal_rate": round(float(refused[out_of_ref].mean()), 4),
            "numeric_refusal_rate": round(float(numeric_refused.mean()), 4)},
        "E3_coordinate_placement": {
            "breeds_total": int(len(breed_median_q)),
            "share_of_breeds_inside_florida_range": round(float(len(breeds_inside) / len(breed_median_q)), 4),
            "median_q_across_breeds": round(float(breed_median_q.median()), 4)},
        "per_breed": {b: {"n": int((breed == b).sum()),
                          "median_q_AB_axis": round(float(np.median(q_dk[breed == b])), 4),
                          "refusal_rate": round(float(refused[breed == b].mean()), 4)}
                      for b in sorted(set(breed))},
        "terminology": "q_AB_axis is a coordinate on the Angus-Brahman axis, not a Brahman ancestry fraction",
        "disclosure": ("the coverage requirement enabling this arm was re-derived after this dataset's "
                       "coverage was known; see docs/COVERAGE_CALIBRATION.md and Amendment 5"),
        "changes_nothing": "no threshold, panel, model or training set is modified by this arm",
    }
    (OUT / "results_6c4.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("E1_primary", "E2_refusal_breadth",
                                             "E3_coordinate_placement")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "unblind"], required=True)
    args = ap.parse_args()
    prepare() if args.stage == "prepare" else unblind()

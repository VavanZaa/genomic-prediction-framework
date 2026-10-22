"""Layer 7 / H7 v2: feasibility gate and endpoint evaluation, applied symmetrically.

Draft protocol: docs/LAYER7_H7v2_PROTOCOL.md. Stages:

  --stage gate     read both reconstructions (observed mice, painted simulated cohort) at every
                   minprob on the ladder and apply the feasibility gate. Uses only call-quality
                   properties of each reconstruction on its own: no switch count, segment length or
                   founder share is computed or written, and nothing compares the two cohorts.
  --stage unblind  verify the v2 lock, then compute the endpoints once at the gate-selected minprob.

Per-haplotype conventions shared by both stages (defined in the protocol before any statistic):
  - consecutive called runs abut when one ends where the next starts; otherwise an uncalled gap
    separates them;
  - a transition between runs of the SAME founder across a gap is not a switch: the gap hides no
    change, so the runs are one segment;
  - a transition between DIFFERENT founders is one switch, whether abutting or across a gap, since
    at least one change must have occurred. That is the minimum number of switches consistent with
    the calls.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402

OUT = ROOT / "output/layer7v2"
SIDES = {"observed": OUT / "observed_recon", "painted": OUT / "painted_recon"}
COVARS = {"observed": ROOT / "data/real_mice/svenson/svenson_covar.csv",
          "painted": OUT / "painted_cross/painted_covar.csv"}
LOCK = OUT / "PROTOCOL_LOCK_H7v2.json"
V1_SIM = ROOT / "output/layer7/simulated_mosaics.json"

LADDER = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
GATE = {"uncalled_share_max": 0.02, "gap_identical_share_max": 0.05, "bound_ratio_max": 1.25}
SHORT_RUN_CM = 2.0
ABUT_TOL_CM = 1e-6
E1_TOL, E2_TOL, E3_TOL = 0.20, 0.20, 0.15
N_BOOT, SEED = 2000, 20260912


def tag(mp):
    return f"{int(round(mp * 100)):03d}"


def load_runs(side, mp):
    seg = pd.read_csv(SIDES[side] / f"segments_mp{tag(mp)}.csv", dtype={"id": str})
    return seg.sort_values(["id", "chr", "hap", "start_cM"]).reset_index(drop=True)


def classify_transitions(seg):
    """Counts of abutting changes, gap-identical and gap-different transitions."""
    g = seg.groupby(["id", "chr", "hap"], sort=False)
    prev_end, prev_founder = g["end_cM"].shift(1), g["founder"].shift(1)
    is_t = prev_end.notna()
    gap = (seg["start_cM"] - prev_end) > ABUT_TOL_CM
    same = seg["founder"] == prev_founder
    return {"total": int(is_t.sum()),
            "abutting_change": int((is_t & ~gap & ~same).sum()),
            "gap_identical": int((is_t & gap & same).sum()),
            "gap_different": int((is_t & gap & ~same).sum())}


def gate():
    rows, selected = [], None
    for mp in LADDER:
        rung = {"minprob": mp}
        ok_both = True
        for side in SIDES:
            calls = json.loads((SIDES[side] / f"calls_mp{tag(mp)}.json").read_text(encoding="utf-8"))
            uncalled = 1 - calls["called_cells"] / calls["total_cells"]
            t = classify_transitions(load_runs(side, mp))
            gap_identical = t["gap_identical"] / t["total"] if t["total"] else float("nan")
            lower = t["abutting_change"]
            upper = t["abutting_change"] + t["gap_different"]
            ratio = upper / lower if lower else float("inf")
            passed = (uncalled <= GATE["uncalled_share_max"]
                      and gap_identical <= GATE["gap_identical_share_max"]
                      and ratio <= GATE["bound_ratio_max"])
            # shares and a within-cohort ratio only: no count that could be compared across cohorts
            rung[side] = {"uncalled_share": round(uncalled, 4),
                          "gap_identical_share": round(gap_identical, 4),
                          "bound_ratio": round(ratio, 4), "passes": bool(passed)}
            ok_both = ok_both and passed
        rung["passes_on_both"] = bool(ok_both)
        rows.append(rung)
        print(f"minprob {mp:.2f}: " + "  ".join(
            f"{s}: uncalled {rung[s]['uncalled_share']:.4f}, gap-identical "
            f"{rung[s]['gap_identical_share']:.4f}, bound ratio {rung[s]['bound_ratio']:.3f} "
            f"[{'ok' if rung[s]['passes'] else 'fail'}]" for s in SIDES), flush=True)
        if ok_both and selected is None:
            selected = mp          # the ladder runs from high to low, so the first pass is the largest

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "feasibility gate", "protocol": "docs/LAYER7_H7v2_PROTOCOL.md (draft)",
        "criteria": GATE, "ladder": LADDER,
        "rule": "largest minprob meeting all three criteria on both cohorts; none -> INFEASIBLE",
        "rungs": rows, "selected_minprob": selected,
        "decision": "FEASIBLE" if selected is not None else "INFEASIBLE",
        "blinding": ("only call-quality shares and within-cohort bound ratios were computed; no "
                     "switch count, segment length or founder share, and no cross-cohort comparison"),
    }
    (OUT / "gate_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ndecision: {result['decision']}"
          + (f" at minprob {selected}" if selected is not None else ""))
    print(f"wrote {(OUT / 'gate_result.json').relative_to(ROOT)}")


def filtered_runs(seg):
    """Per haplotype and chromosome: merge same-founder neighbours, then absorb runs shorter than
    SHORT_RUN_CM into the longer neighbour, repeating until none remain. Identical on both cohorts."""
    out = []
    absorbed = 0
    for (aid, chrom, hap), sub in seg.groupby(["id", "chr", "hap"], sort=False):
        runs = [[f, s, e] for f, s, e in zip(sub["founder"], sub["start_cM"], sub["end_cM"])]

        def merge(rs):
            merged = [rs[0]]
            for f, s, e in rs[1:]:
                if f == merged[-1][0]:
                    merged[-1][2] = e
                else:
                    merged.append([f, s, e])
            return merged

        runs = merge(runs)
        while len(runs) > 1:
            lengths = [e - s for _, s, e in runs]
            i = int(np.argmin(lengths))
            if lengths[i] >= SHORT_RUN_CM:
                break
            left = lengths[i - 1] if i > 0 else -1
            right = lengths[i + 1] if i < len(runs) - 1 else -1
            if left >= right:
                runs[i - 1][2] = runs[i][2]
            else:
                runs[i + 1][1] = runs[i][1]
            del runs[i]
            absorbed += 1
            runs = merge(runs)
        for f, s, e in runs:
            out.append((aid, chrom, hap, f, s, e))
    df = pd.DataFrame(out, columns=["id", "chr", "hap", "founder", "start_cM", "end_cM"])
    return df, absorbed


def cohort_statistics(side, mp):
    seg = load_runs(side, mp)
    runs, absorbed = filtered_runs(seg)
    covar = pd.read_csv(COVARS[side], dtype={"id": str})
    ngen = dict(zip(covar["id"], covar["ngen"].astype(int)))
    runs["ngen"] = runs["id"].map(ngen)
    if runs["ngen"].isna().any():
        raise SystemExit(f"{side}: animals without a generation number")
    stats = {"runs_absorbed_by_filter": absorbed, "runs_before_filter": int(len(seg))}
    for g, sub in runs.groupby("ngen"):
        per_hap = sub.groupby(["id", "hap"]).size() - sub.groupby(["id", "hap"])["chr"].nunique()
        lengths = ((sub["end_cM"] - sub["start_cM"]) / 100.0).to_numpy()
        founder_share = (sub.assign(L=sub["end_cM"] - sub["start_cM"])
                         .groupby("founder")["L"].sum() / (sub["end_cM"] - sub["start_cM"]).sum())
        stats[int(g)] = {"switches": per_hap.to_numpy(float), "lengths": lengths,
                         "founder_share": founder_share.round(4).to_dict()}
    return stats


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the H7 v2 protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def unblind():
    verify_lock()
    g = json.loads((OUT / "gate_result.json").read_text(encoding="utf-8"))
    if g["decision"] != "FEASIBLE":
        raise SystemExit("the feasibility gate did not pass; H7 v2 is INFEASIBLE and has no endpoints")
    mp = g["selected_minprob"]
    obs, pnt = cohort_statistics("observed", mp), cohort_statistics("painted", mp)
    v1 = json.loads(V1_SIM.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    per_gen = {}
    for gen in (8, 11):
        o, p = obs[gen], pnt[gen]
        boot = np.array([rng.choice(p["switches"], len(p["switches"])).mean() for _ in range(N_BOOT)])
        # drift: replicate spread of the frozen v1 exact simulation, as a coefficient of variation
        lo, hi = v1["generations"][str(gen)]["replicate_mean_ci95"]
        drift_cv = ((hi - lo) / 3.92) / v1["generations"][str(gen)]["mean_switches_per_haplotype"]
        p_mean = float(p["switches"].mean())
        sd = float(np.sqrt(boot.var(ddof=1) + (drift_cv * p_mean) ** 2))
        interval = [p_mean - 1.96 * sd, p_mean + 1.96 * sd]
        o_mean = float(o["switches"].mean())
        rel = abs(o_mean - p_mean) / p_mean
        med = (float(np.median(o["lengths"])), float(np.median(p["lengths"])))
        p90 = (float(np.percentile(o["lengths"], 90)), float(np.percentile(p["lengths"], 90)))
        per_gen[str(gen)] = {
            "observed_mean_switches": round(o_mean, 3), "painted_mean_switches": round(p_mean, 3),
            "painted_interval95": [round(x, 3) for x in interval],
            "interval_components_sd": {"bootstrap": round(float(boot.std(ddof=1)), 4),
                                       "drift_from_v1_exact": round(drift_cv * p_mean, 4)},
            "relative_difference": round(rel, 4),
            "E1_prime_pass": bool(interval[0] <= o_mean <= interval[1] and rel <= E1_TOL),
            "observed_median_segment_M": round(med[0], 5), "painted_median_segment_M": round(med[1], 5),
            "observed_p90_segment_M": round(p90[0], 5), "painted_p90_segment_M": round(p90[1], 5),
            "E2_prime_pass": bool(abs(med[0] - med[1]) / med[1] <= E2_TOL
                                  and abs(p90[0] - p90[1]) / p90[1] <= E2_TOL),
            "founder_share_observed": o["founder_share"], "founder_share_painted": p["founder_share"],
        }
    r_o = obs[11]["switches"].mean() / obs[8]["switches"].mean()
    r_p = pnt[11]["switches"].mean() / pnt[8]["switches"].mean()
    e3 = abs(r_o / r_p - 1)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": "EMPIRICAL+SIMULATION", "protocol": "docs/LAYER7_H7v2_PROTOCOL.md",
        "lock": str(LOCK.relative_to(ROOT)), "selected_minprob": mp,
        "filter": {"short_run_cM": SHORT_RUN_CM,
                   "absorbed_observed": obs["runs_absorbed_by_filter"],
                   "absorbed_painted": pnt["runs_absorbed_by_filter"]},
        "per_generation": per_gen,
        "E1_prime_primary": {k: v["E1_prime_pass"] for k, v in per_gen.items()},
        "E2_prime": {k: v["E2_prime_pass"] for k, v in per_gen.items()},
        "E3_prime": {"observed_ratio": round(float(r_o), 4), "painted_ratio": round(float(r_p), 4),
                     "relative_difference": round(float(e3), 4), "pass": bool(e3 <= E3_TOL)},
        "evidence_labels": [
            "Cross-species inheritance test (G_inheritance): v2, symmetric measurement.",
            "Cross-species genomic test (G_genomic): not available - no phenotype in hand.",
            "Same-trait cross-species phenotype validation: not available.",
        ],
    }
    (OUT / "results_h7v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("E1_prime_primary", "E2_prime", "E3_prime")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["gate", "unblind"], required=True)
    args = ap.parse_args()
    gate() if args.stage == "gate" else unblind()

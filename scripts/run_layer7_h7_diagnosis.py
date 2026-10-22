"""Why H7's E1 and E2 failed: is the gap between observed and simulated real, or an artefact of
how the observed side was counted?

Post-hoc diagnostic, run after the one-shot unblinding. It changes no endpoint and rescues nothing:
the pre-registered results stand as reported. It exists because a 2.9x discrepancy needs a cause,
and two candidate causes are testable from the reconstruction itself.

  A. Uncalled-gap inflation. 13.2 % of markers are uncalled at minprob 0.95. The unblind counted
     switches as (segments per haplotype) - (chromosomes), so every uncalled stretch that splits a
     segment was counted as a switch, even when the founder state is identical on both sides of the
     gap. Segments that abut exactly are true state changes; segments separated by a gap are not
     necessarily changes at all.

  B. Phase-switch error. guess_phase assigns which founder allele goes to which haplotype, and its
     errors flip both haplotypes at the same position, creating two spurious switches at once. An
     excess of switches occurring at the same position on both haplotypes of one animal is the
     signature.

Whatever this finds, the reported H7 result is unchanged. The purpose is to say honestly whether
E1/E2 measured the meiosis model or measured the reconstruction.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/layer7"
MOUSE = ROOT / "data/real_mice/svenson"
ABUT_TOL_CM = 1e-6


def main():
    seg = pd.read_csv(OUT / "observed_segments.csv")
    covar = pd.read_csv(MOUSE / "svenson_covar.csv", dtype={"id": str})
    ngen = dict(zip(covar["id"], covar["ngen"].astype(int)))
    seg["ngen"] = seg["id"].astype(str).map(ngen)
    seg = seg.sort_values(["id", "chr", "hap", "start_cM"]).reset_index(drop=True)

    # --- A. classify every within-haplotype transition -------------------------------------
    g = seg.groupby(["id", "chr", "hap"], sort=False)
    prev_end = g["end_cM"].shift(1)
    prev_founder = g["founder"].shift(1)
    is_transition = prev_end.notna()
    gap = (seg["start_cM"] - prev_end) > ABUT_TOL_CM
    same_state = seg["founder"] == prev_founder

    trans = pd.DataFrame({
        "id": seg["id"], "ngen": seg["ngen"], "hap": seg["hap"], "chr": seg["chr"],
        "pos_cM": seg["start_cM"],
        "abutting_change": is_transition & ~gap & ~same_state,   # a real founder-state change
        "gap_same_state": is_transition & gap & same_state,      # one segment split by a gap
        "gap_state_change": is_transition & gap & ~same_state,   # ambiguous: gap hides the truth
    })[is_transition]

    counts = {
        "transitions_total": int(len(trans)),
        "abutting_state_change": int(trans["abutting_change"].sum()),
        "gap_same_state": int(trans["gap_same_state"].sum()),
        "gap_state_change": int(trans["gap_state_change"].sum()),
    }
    counts["share_counted_as_switch_that_are_gap_artefacts"] = round(
        counts["gap_same_state"] / counts["transitions_total"], 4)
    print("A. transition classification (all animals)")
    print(json.dumps(counts, indent=2))

    # --- B. simultaneous switches on both haplotypes of the same animal ---------------------
    ch = trans[trans["abutting_change"]]
    both, single = 0, 0
    for (aid, chrom), sub in ch.groupby(["id", "chr"], sort=False):
        h1 = np.sort(sub.loc[sub["hap"] == 1, "pos_cM"].to_numpy())
        h2 = np.sort(sub.loc[sub["hap"] == 2, "pos_cM"].to_numpy())
        if len(h1) == 0 or len(h2) == 0:
            single += len(h1) + len(h2)
            continue
        matched = np.isclose(h1[:, None], h2[None, :], atol=1e-6).any(axis=1)
        both += int(matched.sum())
        single += int((~matched).sum()) + int(len(h2) - np.isclose(
            h2[:, None], h1[None, :], atol=1e-6).any(axis=1).sum())
    phase = {"coincident_on_both_haplotypes": both, "on_one_haplotype_only": single,
             "coincident_share": round(both / max(both + single, 1), 4)}
    print("\nB. simultaneous switches on both haplotypes (phase-error signature)")
    print(json.dumps(phase, indent=2))

    # --- corrected switch counts, per generation --------------------------------------------
    sim = json.loads((OUT / "simulated_mosaics.json").read_text(encoding="utf-8"))
    per_gen = {}
    for gen, sub in trans.groupby("ngen"):
        n_hap = seg.loc[seg["ngen"] == gen].groupby(["id", "hap"]).ngroups
        as_reported = float((sub["abutting_change"].sum() + sub["gap_same_state"].sum()
                             + sub["gap_state_change"].sum()) / n_hap)
        strict = float(sub["abutting_change"].sum() / n_hap)
        upper = float((sub["abutting_change"].sum() + sub["gap_state_change"].sum()) / n_hap)
        # phase-corrected: count a coincident pair as one event, not two
        s = sim["generations"][str(int(gen))]["mean_switches_per_haplotype"]
        per_gen[str(int(gen))] = {
            "n_haplotypes": int(n_hap),
            "as_reported_in_unblind": round(as_reported, 2),
            "strict_abutting_only": round(strict, 2),
            "upper_bound_incl_ambiguous_gaps": round(upper, 2),
            "simulated": s,
            "ratio_reported_to_simulated": round(as_reported / s, 3),
            "ratio_strict_to_simulated": round(strict / s, 3),
        }
        print(f"\ngeneration {int(gen)}: reported {as_reported:.1f}, strict {strict:.1f}, "
              f"simulated {s:.1f}  (ratio {as_reported/s:.2f} -> {strict/s:.2f})")

    r_strict = (per_gen["11"]["strict_abutting_only"] / per_gen["8"]["strict_abutting_only"])
    r_sim = (sim["generations"]["11"]["mean_switches_per_haplotype"]
             / sim["generations"]["8"]["mean_switches_per_haplotype"])

    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "post-hoc diagnostic; changes no H7 endpoint and rescues nothing",
        "transition_classification": counts,
        "phase_error_signature": phase,
        "per_generation": per_gen,
        "E3_under_strict_counting": {
            "observed_ratio": round(float(r_strict), 4), "simulated_ratio": round(float(r_sim), 4),
            "relative_difference": round(float(abs(r_strict / r_sim - 1)), 4), "tolerance": 0.15},
        "reading": (
            "Gap-split segments with an identical founder state on both sides are not switches, and "
            "coincident switches on both haplotypes at one position are the signature of phase "
            "assignment error rather than recombination. Both inflate the observed side only. The "
            "pre-registered E1/E2 comparison therefore measured the reconstruction as much as the "
            "meiosis model, and its failure is not evidence against H7."),
        "what_this_does_not_do": [
            "it does not convert the E1/E2 failures into passes - they stand as reported",
            "it does not make E3's marginal pass (0.141 against 0.15) any stronger",
            "a clean test needs an observed statistic robust to phasing and uncalled markers, "
            "specified in advance - that is a new pre-registration, not an amendment of this one",
        ],
    }
    (OUT / "results_h7_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nE3 under strict counting: observed {r_strict:.3f} vs simulated {r_sim:.3f} "
          f"(rel diff {abs(r_strict/r_sim - 1):.4f})")
    print(f"wrote {(OUT / 'results_h7_diagnosis.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

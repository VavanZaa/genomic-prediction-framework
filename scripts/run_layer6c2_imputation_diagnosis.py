"""Why the 6C2 refusal endpoint failed: mean-imputation, not the applicability rule.

Post-hoc (Amendment 3 to docs/LAYER6C_PROTOCOL.md). The pre-registered 6C2 run filled the
74.8 % of panel markers absent from the external array with Florida training means, which is
what the frozen harmonisation clause says. On those columns every external animal then *is* the
Florida training mean, so it is dragged toward the training centroid in exactly the space the
applicability rule measures.

Three comparisons decide it, all on data already on disk and none of them fitted on external data:

  A  the 99 external animals, mean-filled over the full 20,087-marker panel   (the frozen run)
  B  the same 99 animals in the 5,067-marker shared subspace, with the support
     region and the breed frequencies recomputed on Florida *training* animals
     restricted to the same markers
  C  Florida held-out animals in that same subspace, as a control that the
     subspace detector still accepts animals it should accept

If B refuses what A accepted while C keeps accepting, the frozen run measured the imputation and
not the population, and the 6C2 refusal endpoint is void rather than failed.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.ood_detector import GenomicOODDetector  # noqa: E402
from src.evaluation.risk_calibration import load_frozen_thresholds  # noqa: E402
from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction  # noqa: E402
from src.real_data.canonical_split import CanonicalDataSplitter  # noqa: E402
from run_layer6c2_hairy_probe import harmonise, read_bed_dosages, PLINK  # noqa: E402
from run_layer6c_external_validation import FloridaReference  # noqa: E402

OUT = ROOT / "output/layer6c2"
THRESH = ROOT / "output/ood_calibration/thresholds_rev1.json"


def main():
    ref = FloridaReference()
    _, keep, stats = harmonise()
    cols = {m: i for i, m in enumerate(ref.markers)}
    idx = np.array([cols[m] for m in keep["marker"]])
    print(f"shared subspace: {len(idx):,} of {len(ref.markers):,} panel markers "
          f"({100*len(idx)/len(ref.markers):.1f}%)")

    pos = {a: i for i, a in enumerate(ref.ids)}
    tr = np.array([pos[a] for a in ref.train_ids])
    G_tr = ref.snp[tr][:, idx]
    mean_sub = np.nanmean(G_tr, axis=0)
    G_tr = np.where(np.isnan(G_tr), mean_sub, G_tr)

    # breed frequencies and support region, both from Florida training animals only
    pa_full, pb_full, _, _ = breed_frequencies(ref.snp, ref.boa)
    pa, pb = pa_full[idx], pb_full[idx]
    q_tr = estimate_brahman_fraction(G_tr, pa, pb)
    thr = load_frozen_thresholds(THRESH)
    det = GenomicOODDetector(n_components=15,
                             calibrated_thresholds={"ranking": thr["ranking"], "absolute": thr["absolute"]})
    det.fit(G_tr, ancestry_qb_train=q_tr, env_ids_train=["Semin"] * len(tr))

    fam = pd.read_csv(f"{PLINK}.fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "pid", "mid", "sex", "phenotype"], dtype=str)
    X_ext = read_bed_dosages(keep, len(fam))
    X_ext = np.where(np.isnan(X_ext), mean_sub, X_ext)
    q_ext = estimate_brahman_fraction(X_ext, pa, pb)
    ass_ext = det.assess_cohort(X_ext, ancestry_qb_cohort=q_ext, env_ids_cohort=["LittlejohnNZ"] * len(X_ext))

    split = CanonicalDataSplitter.get_prospective_split("AvgShortLength", valid_animal_ids=ref.ids,
                                                        require_joint_traits=True)
    te = np.array([pos[a] for a in split.test_ids])
    X_fl = ref.snp[te][:, idx]
    X_fl = np.where(np.isnan(X_fl), mean_sub, X_fl)
    q_fl = estimate_brahman_fraction(X_fl, pa, pb)
    ass_fl = det.assess_cohort(X_fl, ancestry_qb_cohort=q_fl, env_ids_cohort=["Willi"] * len(X_fl))

    frozen = pd.read_csv(OUT / "prepared_hairy.csv")
    rows = {
        "A_external_mean_filled_full_panel": {
            "n": int(len(frozen)), "ranking_refusal_rate": round(float(1 - frozen["ranking_ok"].mean()), 4),
            "q_brahman_mean": round(float(frozen["q_brahman"].mean()), 4),
            "q_brahman_max": round(float(frozen["q_brahman"].max()), 4),
            "support_score_median": round(float(frozen["support_score"].median()), 4)},
        "B_external_shared_subspace": {
            "n": int(len(X_ext)),
            "ranking_refusal_rate": round(float(1 - np.mean([a.allow_ranking_only for a in ass_ext])), 4),
            "q_brahman_mean": round(float(q_ext.mean()), 4), "q_brahman_max": round(float(q_ext.max()), 4),
            "support_score_median": round(float(np.median([a.support_score for a in ass_ext])), 4)},
        "C_florida_holdout_shared_subspace": {
            "n": int(len(X_fl)),
            "ranking_refusal_rate": round(float(1 - np.mean([a.allow_ranking_only for a in ass_fl])), 4),
            "q_brahman_mean": round(float(q_fl.mean()), 4), "q_brahman_max": round(float(q_fl.max()), 4),
            "support_score_median": round(float(np.median([a.support_score for a in ass_fl])), 4)},
    }
    for name, r in rows.items():
        print(f"\n{name}  (n = {r['n']})")
        print(f"  ranking refusal rate : {r['ranking_refusal_rate']:.3f}")
        print(f"  Brahman fraction     : mean {r['q_brahman_mean']:.3f}, max {r['q_brahman_max']:.3f}")
        print(f"  median support score : {r['support_score_median']:.3f}")

    diagnosed = (rows["B_external_shared_subspace"]["ranking_refusal_rate"]
                 > rows["A_external_mean_filled_full_panel"]["ranking_refusal_rate"]
                 and rows["C_florida_holdout_shared_subspace"]["ranking_refusal_rate"] < 0.5)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "evidence": "EMPIRICAL",
        "status": "post-hoc diagnosis of the frozen 6C2 refusal endpoint (Amendment 3)",
        "shared_markers": int(len(idx)), "panel_markers": int(len(ref.markers)),
        "coverage": round(float(len(idx) / len(ref.markers)), 4),
        "comparisons": rows,
        "diagnosis_supported": bool(diagnosed),
        "conclusion": (
            "Mean-filling unobserved markers moves an external animal toward the Florida training "
            "centroid in the same space the applicability rule measures, so at 25 % coverage the "
            "frozen 6C2 refusal endpoint measured the imputation rather than the population. The "
            "endpoint is void, not failed." if diagnosed else
            "The comparisons do not support the imputation explanation; the refusal failure stands "
            "as a property of the applicability rule and must be investigated further."),
        "what_this_does_not_do": [
            "it does not rescue the 6C2 arm: 25 % coverage still cannot support a transfer claim",
            "no threshold, model, panel or training set is changed",
            "the support region here is refitted on Florida training animals only, never on external data",
        ],
    }
    (OUT / "results_6c2_imputation_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ndiagnosis supported: {diagnosed}")
    print(f"wrote {(OUT / 'results_6c2_imputation_diagnosis.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

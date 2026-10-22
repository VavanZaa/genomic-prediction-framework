"""Layer 6C external validation runner (docs/LAYER6C_PROTOCOL.md, frozen specification section).

Stages
- prepare: everything that needs no external phenotype. Fits the Florida SNP GBLUP models (canonical
  2016-2018 cohort, lambda by training-only CV) on the 6C panel; converts the Seminole-steer and Mizzou
  genotypes to Florida coding; estimates every animal's Brahman fraction with one estimator; flags
  duplicates of Florida animals; assigns the Mizzou arms; scores every animal with the revision-1
  applicability rule (environment unit = herd). Writes output/layer6c/prepared_*.csv and prepare_summary.json.
- unblind: refuses to run unless output/layer6c/PROTOCOL_LOCK.json exists and every hashed file still
  matches. Only then reads the Mizzou hair scores and computes the pre-set endpoints.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.ood_detector import GenomicOODDetector
from src.evaluation.risk_calibration import load_frozen_thresholds, sha256_file
from src.prediction.kernel_cv import cv_kernel_ridge
from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction
from src.real_data.canonical_split import CanonicalDataSplitter
from src.real_data.harmonisation import vcf_to_counted_dosage

EXT = ROOT / "data" / "external"
OUT = ROOT / "output" / "layer6c"
LOOKUP, PANEL, FL = EXT / "allele_lookup_f250.csv", EXT / "panel_6c_markers.csv", EXT / "florida_panel_6c.npz"
THRESH = ROOT / "output" / "ood_calibration" / "thresholds_rev1.json"
STEERS_VCF = EXT / "seminole_steers_eva" / "Seminole.Steers.May.2024.vcf.gz"
MIZ_VCF = EXT / "mizzou_hair_shedding" / "Durbin_etal_2021_MizzouHairShedding_genotypes.vcf.gz"
MIZ_CSV = EXT / "mizzou_hair_shedding" / "Durbin_etal_2021_MizzouHairShedding_pheno_metadata.csv"
LOCK = OUT / "PROTOCOL_LOCK.json"
TRAITS = ("AvgShortLength", "LoadSlope")
DUPLICATE_CORR, FAR_Q, MIN_N, N_BOOT, SEED, MIN_GROUP = 0.95, 0.05, 50, 2000, 0, 5


class FloridaReference:
    """Florida training genotypes at the panel, reference frequencies and fitted models."""

    def __init__(self):
        z = np.load(FL, allow_pickle=False)
        self.ids, self.markers = [str(x) for x in z["ids"]], [str(x) for x in z["markers"]]
        self.snp, self.boa = z["snp"].astype(np.float64), z["boa"].astype(np.float64)
        pos = {a: i for i, a in enumerate(self.ids)}
        split = CanonicalDataSplitter.get_prospective_split("AvgShortLength", valid_animal_ids=self.ids,
                                                            require_joint_traits=True)
        self.train_ids = split.train_ids
        tr = np.array([pos[a] for a in self.train_ids])
        G = self.snp[tr]
        self.mean = np.nanmean(G, axis=0)
        self.p = np.clip(self.mean / 2.0, 1e-3, 1 - 1e-3)
        self.denom = 2.0 * np.sum(self.p * (1 - self.p))
        self.Z_tr = np.where(np.isnan(G), self.mean, G) - 2 * self.p
        self.G_tr = np.where(np.isnan(G), self.mean, G)
        K = self.Z_tr @ self.Z_tr.T / self.denom
        self.models = {}
        for trait in TRAITS:
            s = CanonicalDataSplitter.get_prospective_split(trait, valid_animal_ids=self.ids, require_joint_traits=True)
            assert s.train_ids == self.train_ids
            y = s.y_train_adj
            mu = float(np.mean(y))
            lam = cv_kernel_ridge([K], y - mu, n_folds=5, seed=42)["lambda"]
            alpha = np.linalg.solve(K + lam * np.eye(len(y)), y - mu)
            self.models[trait] = {"mu": mu, "lambda": float(lam), "alpha": alpha}
        pa, pb, _, _ = breed_frequencies(self.snp, self.boa)
        self.p_angus, self.p_brahman = pa, pb
        self.q_all = estimate_brahman_fraction(self.snp, pa, pb)
        ph = pd.read_csv(ROOT / "data/real_cattle/florida_multibreed/Thermo.Phenotypes.csv", dtype={"UFID": str}).set_index("UFID")
        gbc = ph.reindex(self.ids)["BrahmanGBC"].to_numpy(dtype=float)
        ok = np.isfinite(gbc)
        self.q_validation = {"r_with_BrahmanGBC": float(np.corrcoef(self.q_all[ok], gbc[ok])[0, 1]),
                             "mean_difference": float(np.mean(self.q_all[ok] - gbc[ok])), "n": int(ok.sum())}
        self.q_train = self.q_all[tr]
        thr = load_frozen_thresholds(THRESH)
        self.detector = GenomicOODDetector(n_components=15, calibrated_thresholds={"ranking": thr["ranking"], "absolute": thr["absolute"]})
        self.detector.fit(self.G_tr, ancestry_qb_train=self.q_train, env_ids_train=["Semin"] * len(tr))
        # all Florida animals, for the duplicate check
        A = np.where(np.isnan(self.snp), self.mean, self.snp) - 2 * self.p
        self.Z_all, self.self_all = A, np.einsum("ij,ij->i", A, A)

    def score(self, X: np.ndarray, herd: str, chunk: int = 1000) -> pd.DataFrame:
        """Predictions, Brahman fraction, duplicate check and applicability for external animals."""
        rows = []
        for start in range(0, len(X), chunk):
            Xc = X[start:start + chunk].astype(np.float64)
            q = estimate_brahman_fraction(Xc, self.p_angus, self.p_brahman)
            Xi = np.where(np.isnan(Xc), self.mean, Xc)
            Z = Xi - 2 * self.p
            K_ext = Z @ self.Z_tr.T / self.denom
            preds = {t: m["mu"] + K_ext @ m["alpha"] for t, m in self.models.items()}
            cross = Z @ self.Z_all.T
            corr = cross / np.sqrt(np.einsum("ij,ij->i", Z, Z)[:, None] * self.self_all[None, :])
            ass = self.detector.assess_cohort(Xi, ancestry_qb_cohort=q, env_ids_cohort=[herd] * len(Xi))
            for i, a in enumerate(ass):
                rows.append({"q_brahman": q[i], "max_corr_florida": float(corr[i].max()),
                             "support_score": a.support_score, "ranking_ok": a.allow_ranking_only,
                             "numeric_ok": a.allow_numeric_prediction, "tier": a.confidence_tier.name,
                             **{f"pred_{t}": float(preds[t][i]) for t in TRAITS}})
        df = pd.DataFrame(rows)
        df["duplicate"] = df["max_corr_florida"] > DUPLICATE_CORR
        df["ancestry_ratio"] = self.detector.ancestry_support.ratio(df["q_brahman"].to_numpy())
        return df


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = FloridaReference()
    lookup, panel = pd.read_csv(LOOKUP), pd.read_csv(PANEL)
    summary = {"florida": {"train_n": len(ref.train_ids), "panel_markers": len(ref.markers),
                           "models": {t: {"lambda": m["lambda"], "mu": m["mu"]} for t, m in ref.models.items()},
                           "q_estimate_validation": ref.q_validation}}
    for name, vcf, key, herd in (("steers", STEERS_VCF, "id", "Semin"), ("mizzou", MIZ_VCF, "position", "Mizzou")):
        ids, got, X, reasons = vcf_to_counted_dosage(vcf, lookup, markers=panel["marker"], key=key)
        full = np.full((len(ids), len(ref.markers)), np.nan, dtype=np.float32)
        col = {m: j for j, m in enumerate(ref.markers)}
        full[:, [col[m] for m in got]] = X
        df = ref.score(full, herd)
        df.insert(0, "id", ids)
        if name == "mizzou":
            df["arm"] = np.where(df["q_brahman"] < FAR_Q, "far",
                                 np.where(df["ancestry_ratio"] <= 1.0, "near", "other"))
        df.to_csv(OUT / f"prepared_{name}.csv", index=False)
        clean = df[~df["duplicate"]]
        summary[name] = {"animals": len(df), "duplicates": int(df["duplicate"].sum()),
                         "panel_markers_found": len(got), "marker_exclusions": dict(reasons),
                         "missing_share": float(np.isnan(full).mean()),
                         "q_brahman_quantiles": clean["q_brahman"].quantile([0, .05, .25, .5, .75, .95, 1]).round(3).to_dict(),
                         "ranking_accepted_share": float(clean["ranking_ok"].mean()),
                         "numeric_accepted_share": float(clean["numeric_ok"].mean())}
        if name == "mizzou":
            summary[name]["arms"] = clean["arm"].value_counts().to_dict()
            summary[name]["ranking_accepted_by_arm"] = clean.groupby("arm")["ranking_ok"].mean().round(4).to_dict()
    summary["steers"]["qa_gate_pass"] = summary["steers"]["ranking_accepted_share"] >= 0.80
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def verify_lock():
    if not LOCK.exists():
        raise SystemExit("No PROTOCOL_LOCK.json: freeze the protocol (scripts/lock_layer6c_protocol.py) before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [p for p, h in lock["sha256"].items() if sha256_file(ROOT / p) != h]
    if bad:
        raise SystemExit(f"Files changed since the protocol was locked: {bad}")
    return lock


def bootstrap_ci(x, y, fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def unblind():
    verify_lock()
    prep = pd.read_csv(OUT / "prepared_mizzou.csv", dtype={"id": str})
    ph = pd.read_csv(MIZ_CSV, dtype={"id": str, "score_group": str})
    size = ph.groupby("score_group")["hair_score"].transform("size")
    ph = ph[size >= MIN_GROUP].copy()
    ph["centred"] = ph["hair_score"] - ph.groupby("score_group")["hair_score"].transform("mean")
    animal = ph.groupby("id").agg(phenotype=("centred", "mean"), records=("centred", "size")).reset_index()
    d = prep[~prep["duplicate"]].merge(animal, on="id", how="inner")
    rho = lambda a, b: stats.spearmanr(a, b)[0]
    res = {"phenotyped_genotyped_animals": len(d)}
    for arm in ("near", "far", "other"):
        sub = d[d["arm"] == arm]
        acc = sub[sub["ranking_ok"]]
        r = {"n": len(sub), "ranking_accepted": len(acc), "refusal_rate": float(1 - sub["ranking_ok"].mean()) if len(sub) else None}
        for label, part in (("accepted", acc), ("all", sub), ("refused", sub[~sub["ranking_ok"]])):
            if len(part) >= 3:
                x, y = part["pred_AvgShortLength"].to_numpy(), part["phenotype"].to_numpy()
                r[f"spearman_{label}"] = float(rho(x, y))
                r[f"pearson_{label}"] = float(np.corrcoef(x, y)[0, 1])
                if label == "accepted" and len(part) >= MIN_N:
                    r["spearman_accepted_ci95"] = bootstrap_ci(x, y, rho)
        res[arm] = r
    near = res["near"]
    res["endpoints"] = {
        "G_genomic_related_trait_near": ("inconclusive (n < 50)" if near["ranking_accepted"] < MIN_N else
                                         "pass" if near["spearman_accepted_ci95"][0] > 0 else "fail"),
        "far_domain_refusal": ("inconclusive (n < 50)" if res["far"]["n"] < MIN_N else
                               "pass" if res["far"]["refusal_rate"] >= 0.90 else "fail"),
        "G_phenotype": "not tested (different trait scale)", "G_inheritance": "not tested (no trios)",
        "LoadSlope transfer": "not tested (no external LoadSlope)",
    }
    (OUT / "results_mizzou.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=["prepare", "unblind"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else unblind()

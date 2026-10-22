"""Layer 6C2: slick / hairy specificity probe (Littlejohn et al. 2014, Dryad 10.5061/dryad.nh6v4).

Pre-registered in docs/LAYER6C_PROTOCOL.md, section "6C2". Two stages, as in 6C1:

  --stage prepare   harmonise the hairy-syndrome HD genotypes onto the 6C panel and score them.
                    Genotype side only: the .fam phenotype column is never read here.
  --stage unblind   verify the 6C2 lock, then read the phenotype column once and compute the
                    pre-registered endpoints.

The slick arm is not implemented on purpose: those 82 animals carry 26 chromosome-20 markers,
so the Florida score cannot be computed for them. That is recorded as not tested.

The primary endpoint is refusal. The score-phenotype contrast is descriptive, and cannot change
any threshold, panel or model.
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402
from src.real_data.liftover import ChainLiftOver  # noqa: E402
from run_layer6c_external_validation import FloridaReference, TRAITS  # noqa: E402

DATA = ROOT / "data/external/slick_hairy_prlr/extracted"
PLINK = DATA / "gen_phen_data_for_paper/hairy_syndrome_gens_phen"
CHAIN = ROOT / "data/external/ucsc_chain/bosTau6ToBosTau9.over.chain.gz"
PANEL = ROOT / "data/external/panel_6c_markers.csv"
OUT = ROOT / "output/layer6c2"
LOCK = OUT / "PROTOCOL_LOCK_6C2.json"
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
MIN_PANEL_MARKERS = 5000
REFUSAL_TARGET = 0.95
N_BOOT, SEED = 2000, 20260912


def harmonise():
    """Panel markers recoverable from the hairy-syndrome HD set, with allele orientation."""
    panel = pd.read_csv(PANEL, dtype={"chrom": str})
    bim = pd.read_csv(f"{PLINK}.bim", sep="\t", header=None,
                      names=["chr", "snp", "cm", "bp", "a1", "a2"], dtype={"chr": str, "snp": str})
    bim["bim_index"] = np.arange(len(bim))
    lifted = ChainLiftOver(CHAIN).convert_many(bim["chr"].astype(str), bim["bp"].astype(int))
    bim["new_chrom"] = [x[0] if x else None for x in lifted]
    bim["new_pos"] = [x[1] if x else -1 for x in lifted]
    n_lifted = int(bim["new_chrom"].notna().sum())

    m = panel.merge(bim.dropna(subset=["new_chrom"]), left_on=["chrom", "pos_bp"],
                    right_on=["new_chrom", "new_pos"], how="inner").drop_duplicates(subset=["marker"])
    pa, pb = m["counted_nucleotide"].str.upper(), m["other_nucleotide"].str.upper()
    ba, bb = m["a1"].str.upper(), m["a2"].str.upper()
    ambiguous = (pa + pb).isin(["AT", "TA", "CG", "GC"])
    monomorphic = (ba == "0") | (bb == "0")
    direct = ((pa == ba) & (pb == bb)) | ((pa == bb) & (pb == ba))
    flip = ((pa == ba.map(COMP)) & (pb == bb.map(COMP))) | ((pa == bb.map(COMP)) & (pb == ba.map(COMP)))
    usable = (direct | flip) & ~ambiguous & ~monomorphic
    keep = m.loc[usable].copy()
    keep["counted_is_a1"] = (keep["counted_nucleotide"].str.upper() == keep["a1"].str.upper()) | \
                            (keep["counted_nucleotide"].str.upper() == keep["a1"].str.upper().map(COMP))
    stats = {"bim_markers": int(len(bim)), "lifted": n_lifted, "matched_by_position": int(len(m)),
             "strand_ambiguous": int(ambiguous.sum()), "monomorphic_external": int(monomorphic.sum()),
             "alleles_direct": int(direct.sum()), "alleles_complement": int(flip.sum()),
             "usable_panel_markers": int(usable.sum()), "panel_total": int(len(panel)),
             "coverage": round(float(usable.sum() / len(panel)), 4)}
    return panel, keep, stats


def read_bed_dosages(keep, n_samples, bed_path=None):
    """Counted-allele dosages for the kept markers, read by byte offset from the SNP-major .bed."""
    bed_path = bed_path or f"{PLINK}.bed"
    bytes_per_snp = (n_samples + 3) // 4
    copies_a1 = np.array([2.0, np.nan, 1.0, 0.0])  # 2-bit codes 00,01,10,11
    X = np.full((n_samples, len(keep)), np.nan)
    with open(bed_path, "rb") as fh:
        magic = fh.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise SystemExit(f"{bed_path} is not a PLINK binary file")
        if magic[2] != 1:
            raise SystemExit("only SNP-major .bed files are supported")
        for col, (idx, counted_is_a1) in enumerate(zip(keep["bim_index"].to_numpy(),
                                                       keep["counted_is_a1"].to_numpy())):
            fh.seek(3 + int(idx) * bytes_per_snp)
            raw = np.frombuffer(fh.read(bytes_per_snp), dtype=np.uint8)
            codes = np.empty(bytes_per_snp * 4, dtype=np.uint8)
            for shift in range(4):
                codes[shift::4] = (raw >> (2 * shift)) & 0b11
            d = copies_a1[codes[:n_samples]]
            X[:, col] = d if counted_is_a1 else 2.0 - d
    return X


def prepare():
    panel, keep, stats = harmonise()
    if stats["usable_panel_markers"] < MIN_PANEL_MARKERS:
        raise SystemExit(f"coverage {stats['usable_panel_markers']} < {MIN_PANEL_MARKERS}: 6C2 is inconclusive")
    fam = pd.read_csv(f"{PLINK}.fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "pid", "mid", "sex", "phenotype"], dtype=str)
    n = len(fam)
    print(f"animals {n}; usable panel markers {stats['usable_panel_markers']:,} "
          f"({100*stats['coverage']:.1f}% of the panel)")

    X_small = read_bed_dosages(keep, n)
    print(f"genotype missingness at those markers: {100*np.isnan(X_small).mean():.2f}%")

    ref = FloridaReference()
    cols = {mk: i for i, mk in enumerate(ref.markers)}
    X = np.full((n, len(ref.markers)), np.nan)
    X[:, [cols[mk] for mk in keep["marker"]]] = X_small

    scored = ref.score(X, herd="LittlejohnNZ")
    scored.insert(0, "iid", fam["iid"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT / "prepared_hairy.csv", index=False)

    summary = {"stage": "prepare", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "evidence": "EMPIRICAL", "arm": "far-domain specificity probe",
               "n_animals": n, "harmonisation": stats,
               "genotype_missingness_at_kept_markers": round(float(np.isnan(X_small).mean()), 4),
               "q_brahman": {"mean": round(float(scored["q_brahman"].mean()), 4),
                             "max": round(float(scored["q_brahman"].max()), 4)},
               "duplicates_with_florida": int(scored["duplicate"].sum()),
               "ranking_refusal_rate": round(float(1 - scored["ranking_ok"].mean()), 4),
               "numeric_refusal_rate": round(float(1 - scored["numeric_ok"].mean()), 4),
               "slick_arm": "not tested: 26 chromosome-20 markers only, the score cannot be computed"}
    (OUT / "prepare_summary_6c2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("q_brahman", "duplicates_with_florida",
                                              "ranking_refusal_rate", "numeric_refusal_rate")}, indent=2))
    print(f"wrote {(OUT / 'prepared_hairy.csv').relative_to(ROOT)}")


def verify_lock():
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: freeze the 6C2 protocol before unblinding")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [f for f, h in lock["sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to unblind: {bad}")
    print(f"lock verified ({len(lock['sha256'])} files)")


def unblind():
    verify_lock()
    scored = pd.read_csv(OUT / "prepared_hairy.csv", dtype={"iid": str})
    fam = pd.read_csv(f"{PLINK}.fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "pid", "mid", "sex", "phenotype"], dtype=str)
    # PLINK binary coding: 1 = unaffected, 2 = affected.
    affected = fam["phenotype"].to_numpy() == "2"
    if scored["iid"].tolist() != fam["iid"].tolist():
        raise SystemExit("prepared rows do not match the .fam order")

    refusal_ranking = float(1 - scored["ranking_ok"].mean())
    refusal_numeric = float(1 - scored["numeric_ok"].mean())
    primary = bool(refusal_numeric == 1.0 and refusal_ranking >= REFUSAL_TARGET)

    rng = np.random.default_rng(SEED)
    trait_rows = {}
    for t in TRAITS:
        v = scored[f"pred_{t}"].to_numpy()
        a, u = v[affected], v[~affected]
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(u) - 1) * u.var(ddof=1)) / (len(a) + len(u) - 2))
        d = float((a.mean() - u.mean()) / pooled) if pooled > 0 else float("nan")
        boot = []
        for _ in range(N_BOOT):
            ia, iu = rng.integers(0, len(a), len(a)), rng.integers(0, len(u), len(u))
            ba, bu = a[ia], u[iu]
            p = np.sqrt(((len(ba) - 1) * ba.var(ddof=1) + (len(bu) - 1) * bu.var(ddof=1)) / (len(ba) + len(bu) - 2))
            if p > 0:
                boot.append((ba.mean() - bu.mean()) / p)
        lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)
        trait_rows[t] = {"mean_affected": round(float(a.mean()), 4), "mean_unaffected": round(float(u.mean()), 4),
                         "standardised_mean_difference": round(d, 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "detects": bool(lo > 0 or hi < 0)}
        print(f"{t:16s} affected {a.mean():+.4f} vs unaffected {u.mean():+.4f}  "
              f"SMD {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    result = {
        "stage": "unblind", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": "EMPIRICAL", "protocol": "docs/LAYER6C_PROTOCOL.md section 6C2",
        "lock": str(LOCK.relative_to(ROOT)),
        "n_animals": int(len(scored)), "n_affected": int(affected.sum()),
        "n_unaffected": int((~affected).sum()),
        "panel_coverage": json.loads((OUT / "prepare_summary_6c2.json").read_text())["harmonisation"]["coverage"],
        "primary_endpoint_refusal": {
            "numeric_refusal_rate": round(refusal_numeric, 4),
            "ranking_refusal_rate": round(refusal_ranking, 4),
            "target": REFUSAL_TARGET, "pass": primary},
        "secondary_descriptive_contrast": trait_rows,
        "interpretation_limits": [
            "25% panel coverage: predictions keep rank order (rho 0.81 internally) but lose ~55% of their spread",
            "the k-NN support score is unreliable at this coverage (r = 0.555 internally) and carries no endpoint",
            "these animals are close relatives, so any non-null contrast is more likely family structure than trait transfer",
            "different trait (near-monogenic PRLR coat syndrome) and different breed: not same-trait validation",
        ],
        "slick_arm": "not tested: 26 chromosome-20 markers only, the Florida score cannot be computed",
        "changes_nothing": "no threshold, panel, model or training set is modified by this arm",
    }
    (OUT / "results_6c2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nprimary (refusal): {'pass' if primary else 'FAIL'} "
          f"(numeric {refusal_numeric:.3f}, ranking {refusal_ranking:.3f})")
    print(f"wrote {(OUT / 'results_6c2.json').relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "unblind"], required=True)
    args = ap.parse_args()
    prepare() if args.stage == "prepare" else unblind()

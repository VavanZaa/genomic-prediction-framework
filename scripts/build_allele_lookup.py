"""Build the Illumina A/B -> nucleotide allele map for the Florida F250 panel from the Seminole steers.

The Florida SNP.raw.gz counts an Illumina A or B allele per marker; the Seminole steers VCF (EVA
PRJEB75981, same ranch as the training heifers) lists nucleotides with a mostly provisional REF. For
each shared marker, the counted-allele frequency in the Florida Seminole animals is compared with the
steers' ALT frequency (src/real_data/harmonisation.orient_by_frequency) to decide which nucleotide is
the counted allele. Output: data/external/allele_lookup_f250.csv and a JSON summary.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.real_data.harmonisation import is_strand_ambiguous, orient_by_frequency

FLORIDA = ROOT / "data" / "real_cattle" / "florida_multibreed"
STEERS = ROOT / "data" / "external" / "seminole_steers_eva" / "Seminole.Steers.May.2024.vcf.gz"
OUT = ROOT / "data" / "external" / "allele_lookup_f250.csv"


def steer_frequencies():
    recs = {}
    with gzip.open(STEERS, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, pos, mid, ref, alt, _q, _f, _i, _fmt, gts = line.rstrip("\n").split("\t", 9)
            gts = gts.replace("|", "/")
            hom_alt, het, missing = gts.count("1/1"), gts.count("0/1") + gts.count("1/0"), gts.count("./.")
            n = gts.count("\t") + 1 - missing
            recs[mid] = (chrom, int(pos), ref, alt, (2 * hom_alt + het) / (2 * n) if n else np.nan, n)
    return recs


def florida_frequencies(names_needed):
    ph = pd.read_csv(FLORIDA / "Thermo.Phenotypes.csv", dtype={"UFID": str})
    semin = set(ph.loc[ph["Project"] == "Semin", "UFID"])
    with gzip.open(FLORIDA / "SNP.raw.gz", "rt") as f:
        header = f.readline().split()[1:]
        cols = [i for i, c in enumerate(header) if c.rsplit("_", 1)[0] in names_needed]
        names = [header[i].rsplit("_", 1)[0] for i in cols]
        codes = [header[i].rsplit("_", 1)[1] for i in cols]
        idx = np.asarray(cols) + 1
        total, count, used = np.zeros(len(cols)), np.zeros(len(cols)), 0
        for line in f:
            parts = line.split()
            if parts[0] not in semin:
                continue
            vals = np.asarray(parts, dtype=object)[idx]
            ok = vals != "NA"
            dose = np.zeros(len(cols))
            dose[ok] = vals[ok].astype(float)
            total += dose
            count += ok
            used += 1
    return names, codes, total / (2 * np.maximum(count, 1)), count, used


def main():
    steers = steer_frequencies()
    names, codes, p_fl, n_fl, used = florida_frequencies(set(steers))
    rows = []
    for name, code, p, n in zip(names, codes, p_fl, n_fl):
        chrom, pos, ref, alt, q, nq = steers[name]
        rows.append((name, chrom, pos, code, ref, alt, p, q, int(n), nq))
    df = pd.DataFrame(rows, columns=["marker", "chrom", "pos_bp", "florida_counted_code", "vcf_ref", "vcf_alt",
                                     "p_florida_counted", "p_steers_alt", "n_florida", "n_steers"])
    df["orientation"] = orient_by_frequency(df["p_florida_counted"], df["p_steers_alt"])
    df["counted_nucleotide"] = np.where(df["orientation"] == 1, df["vcf_alt"],
                                        np.where(df["orientation"] == -1, df["vcf_ref"], ""))
    df["other_nucleotide"] = np.where(df["orientation"] == 1, df["vcf_ref"],
                                      np.where(df["orientation"] == -1, df["vcf_alt"], ""))
    df["strand_ambiguous"] = [is_strand_ambiguous(r, a) for r, a in zip(df["vcf_ref"], df["vcf_alt"])]
    aligned = np.where(df["orientation"] == 1, df["p_steers_alt"], 1 - df["p_steers_alt"])
    df["abs_freq_diff"] = np.abs(df["p_florida_counted"] - aligned)
    df.to_csv(OUT, index=False)
    ok = df["orientation"] != 0
    summary = {
        "florida_seminole_animals": used, "steers": int(df["n_steers"].max()), "shared_markers": len(df),
        "oriented": int(ok.sum()), "undecided": int((~ok).sum()),
        "usable_for_letter_matching": int((ok & ~df["strand_ambiguous"]).sum()),
        "orientation_alt_is_counted": int((df["orientation"] == 1).sum()),
        "orientation_ref_is_counted": int((df["orientation"] == -1).sum()),
        "freq_r_oriented": float(np.corrcoef(df.loc[ok, "p_florida_counted"], aligned[ok])[0, 1]),
        "median_abs_diff_oriented": float(df.loc[ok, "abs_freq_diff"].median()),
        "counted_code_vs_alt": pd.crosstab(df.loc[ok, "florida_counted_code"], df.loc[ok, "orientation"]).to_dict(),
    }
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

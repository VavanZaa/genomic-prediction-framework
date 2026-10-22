"""Fix the Layer 6C marker panel before any external phenotype is opened.

A marker enters the panel when it is:
1. oriented and strand-unambiguous in data/external/allele_lookup_f250.csv (Florida A/B -> nucleotide);
2. present in the Mizzou VCF at the same chromosome and position, with letters that match the lookup
   on either strand;
3. imputed with Minimac R2 >= 0.80 in the Mizzou VCF.
The qualifying markers are then thinned evenly along the genome (every k-th in chromosome/position
order) to about 20,000 markers, the size of the Layer 3B step-6 panel. Output:
data/external/panel_6c_markers.csv and a summary JSON with the panel's SHA-256 (for the protocol lock).
Only the VCF's marker columns (CHROM..INFO) are read; no phenotype file is touched.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.real_data.harmonisation import alt_is_counted, usable_lookup

LOOKUP = ROOT / "data" / "external" / "allele_lookup_f250.csv"
MIZZOU = ROOT / "data" / "external" / "mizzou_hair_shedding" / "Durbin_etal_2021_MizzouHairShedding_genotypes.vcf.gz"
OUT = ROOT / "data" / "external" / "panel_6c_markers.csv"
MIN_R2, TARGET = 0.80, 20000


def info_fields(info: str) -> dict:
    out = {}
    for item in info.split(";"):
        k, _, v = item.partition("=")
        out[k] = v if v else True
    return out


def main():
    lookup = pd.read_csv(LOOKUP)
    table = usable_lookup(lookup)
    pos = {f"{c}:{p}": m for m, c, p in zip(lookup["marker"], lookup["chrom"].astype(str), lookup["pos_bp"].astype(int))
           if m in table}
    rows, counts = [], {"usable_lookup": len(table), "in_mizzou": 0, "letters_match": 0, "r2_ok": 0}
    with gzip.open(MIZZOU, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, p, _id, ref, alt, _q, _f, info, _rest = line.split("\t", 8)
            m = pos.get(f"{chrom}:{p}")
            if m is None:
                continue
            counts["in_mizzou"] += 1
            if len(ref) != 1 or len(alt) != 1 or alt_is_counted(ref, alt, *table[m]) is None:
                continue
            counts["letters_match"] += 1
            inf = info_fields(info)
            r2 = float(inf.get("R2", "nan"))
            typed = "TYPED" in inf or "TYPED_ONLY" in inf
            if not r2 >= MIN_R2:
                continue
            counts["r2_ok"] += 1
            rows.append((m, chrom, int(p), table[m][0], table[m][1], ref, alt, r2, typed))
    df = pd.DataFrame(rows, columns=["marker", "chrom", "pos_bp", "counted_nucleotide", "other_nucleotide",
                                     "mizzou_ref", "mizzou_alt", "mizzou_r2", "mizzou_typed"])
    df["chrom_n"] = pd.to_numeric(df["chrom"], errors="coerce")
    df = df[df["chrom_n"].between(1, 29)].sort_values(["chrom_n", "pos_bp"]).drop(columns="chrom_n")
    k = max(1, len(df) // TARGET)
    panel = df.iloc[::k].reset_index(drop=True)
    panel.to_csv(OUT, index=False)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    summary = {**counts, "autosomal_qualifying": len(df), "thinning_k": k, "panel_markers": len(panel),
               "panel_typed_share": float(panel["mizzou_typed"].mean()), "panel_median_r2": float(panel["mizzou_r2"].median()),
               "rule": f"usable lookup; Mizzou position + letters match; Minimac R2 >= {MIN_R2}; autosomes; every k-th",
               "sha256": sha}
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

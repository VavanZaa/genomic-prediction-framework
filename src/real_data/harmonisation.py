"""Marker harmonisation between Illumina A/B-coded panels and nucleotide-coded VCFs.

The Florida files count Illumina A/B alleles; external VCFs list nucleotides, and a PLINK-exported VCF may
carry a provisional (not genome-reference) REF allele. Allele identity is therefore decided empirically
on a population that shares the reference panel's allele frequencies (the Seminole steers share the
training heifers' ranch): the external ALT allele is the reference panel's counted allele when its
frequency matches p, and the other allele when it matches 1 - p. Markers where frequency cannot tell
the two orientations apart are left undecided rather than guessed.
"""
from __future__ import annotations
import gzip
from collections import Counter
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd

STRAND_AMBIGUOUS = ({"A", "T"}, {"C", "G"})
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def is_strand_ambiguous(a1: str, a2: str) -> bool:
    """A/T and C/G pairs look identical on both strands, so letters cannot align them across datasets."""
    return {a1.upper(), a2.upper()} in STRAND_AMBIGUOUS


def orient_by_frequency(p_counted, p_alt, tol: float = 0.10, margin: float = 0.05) -> np.ndarray:
    """Per marker: 1 if the external ALT allele is the panel's counted allele, -1 if REF is, 0 if undecided.

    p_counted: counted-allele frequency in the reference panel; p_alt: ALT-allele frequency in an external
    population expected to share those frequencies. Undecided when the better orientation misses by more
    than `tol`, or the two orientations fit within `margin` of each other (frequencies near 0.5).
    """
    p, q = np.asarray(p_counted, dtype=float), np.asarray(p_alt, dtype=float)
    same, flip = np.abs(p - q), np.abs(p - (1.0 - q))
    code = np.where(same < flip, 1, -1)
    undecided = (np.minimum(same, flip) > tol) | (np.abs(same - flip) < margin) | ~np.isfinite(p) | ~np.isfinite(q)
    return np.where(undecided, 0, code).astype(int)


def usable_lookup(lookup: pd.DataFrame) -> Dict[str, tuple]:
    """marker -> (counted nucleotide, other nucleotide) for oriented, strand-unambiguous markers."""
    ok = (lookup["orientation"] != 0) & ~lookup["strand_ambiguous"].astype(bool)
    return {m: (c, o) for m, c, o in zip(lookup.loc[ok, "marker"], lookup.loc[ok, "counted_nucleotide"],
                                         lookup.loc[ok, "other_nucleotide"])}


def _alt_is_counted(ref: str, alt: str, counted: str, other: str) -> Optional[bool]:
    """True if ALT is the counted allele, False if REF is, None if the letters cannot be matched."""
    for f in (lambda x: x, lambda x: COMPLEMENT.get(x, "?")):     # same strand, then opposite strand
        r, a = f(ref), f(alt)
        if (a, r) == (counted, other):
            return True
        if (r, a) == (counted, other):
            return False
    return None


alt_is_counted = _alt_is_counted


def _alt_counts(gt_block: str, n: int, fixed_width: bool) -> np.ndarray:
    """ALT-allele count per sample from the genotype columns; NaN where missing."""
    if fixed_width:                                   # "0/1\t1/1\t..." : alleles at offsets 4j and 4j+2
        b = np.frombuffer(gt_block.encode("ascii"), dtype="S1")
        a1, a2 = b[0::4][:n], b[2::4][:n]
    else:
        gts = [g.split(":", 1)[0].replace("|", "/") for g in gt_block.split("\t")]
        a1 = np.array([g[0:1] for g in gts], dtype="S1")
        a2 = np.array([g[2:3] if len(g) > 2 else b"." for g in gts], dtype="S1")
    dose = (a1 == b"1").astype(float) + (a2 == b"1").astype(float)
    dose[(a1 == b".") | (a2 == b".")] = np.nan
    return dose


def vcf_to_counted_dosage(vcf_path, lookup: pd.DataFrame, markers: Optional[Sequence[str]] = None,
                          samples: Optional[Sequence[str]] = None, key: str = "id"):
    """Reads a biallelic VCF into dosages of the Florida counted allele (0/1/2, NaN missing).

    Only markers in the usable lookup (and in `markers`, if given) are read; each is aligned by letters,
    directly or via the complement strand. `key="id"` matches VCF IDs to lookup marker names;
    `key="position"` matches CHROM:POS to the lookup's chrom/pos_bp (same assembly required). Returns
    (sample_ids, lookup marker names, matrix animals x markers, Counter of exclusion reasons).
    """
    table = usable_lookup(lookup)
    wanted = set(table) if markers is None else set(markers) & set(table)
    by_position = None
    if key == "position":
        by_position = {f"{c}:{p}": m for m, c, p in zip(lookup["marker"], lookup["chrom"].astype(str),
                                                         lookup["pos_bp"].astype(int)) if m in wanted}
    elif key != "id":
        raise ValueError("key must be 'id' or 'position'")
    reasons, cols, names = Counter(), [], []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                all_samples = line.rstrip("\n").split("\t")[9:]
                keep = np.arange(len(all_samples)) if samples is None else \
                    np.array([i for i, s in enumerate(all_samples) if s in set(samples)])
                n = len(all_samples)
                continue
            head = line.split("\t", 9)
            mid = by_position.get(f"{head[0]}:{head[1]}") if by_position is not None else head[2]
            if mid not in wanted:
                continue
            ref, alt, fmt = head[3], head[4], head[8]
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                reasons["not biallelic SNP"] += 1
                continue
            alt_counted = _alt_is_counted(ref, alt, *table[mid])
            if alt_counted is None:
                reasons["letters do not match the lookup"] += 1
                continue
            block = head[9].rstrip("\n")
            dose = _alt_counts(block, n, fmt == "GT" and len(block) == 4 * n - 1)
            cols.append((dose if alt_counted else 2.0 - dose)[keep].astype(np.float32))
            names.append(mid)
    reasons["not in VCF"] = len(wanted) - len(names) - sum(v for k, v in reasons.items() if k != "not in VCF")
    ids = [all_samples[i] for i in keep]
    X = np.column_stack(cols) if cols else np.zeros((len(ids), 0), dtype=np.float32)
    return ids, names, X, reasons

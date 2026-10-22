"""Allele orientation between A/B-coded and nucleotide-coded panels."""
import gzip

import numpy as np
import pandas as pd

from src.real_data.harmonisation import is_strand_ambiguous, orient_by_frequency, vcf_to_counted_dosage


def test_vcf_dosages_follow_the_counted_allele(tmp_path):
    lookup = pd.DataFrame({
        "marker": ["m_alt", "m_ref", "m_comp", "m_amb", "m_bad", "m_undecided"],
        "counted_nucleotide": ["G", "C", "A", "A", "C", ""],
        "other_nucleotide": ["A", "T", "G", "T", "T", ""],
        "orientation": [1, -1, 1, 1, 1, 0],
        "strand_ambiguous": [False, False, False, True, False, False],
    })
    records = [
        ("1", "100", "m_alt", "A", "G", "GT", ["0/0", "0/1", "1/1"]),      # ALT is counted: dose = ALT count
        ("1", "200", "m_ref", "C", "T", "GT", ["0/0", "0/1", "./."]),      # REF is counted: dose = 2 - ALT count
        ("1", "300", "m_comp", "C", "T", "GT", ["1/1", "0/1", "0/0"]),     # opposite strand: T=A counted, C=G
        ("1", "400", "m_amb", "A", "T", "GT", ["0/1", "0/1", "0/1"]),      # strand-ambiguous: excluded
        ("1", "500", "m_bad", "A", "C", "GT", ["0/1", "0/1", "0/1"]),      # matches C/T on neither strand: excluded
        ("1", "600", "m_undecided", "A", "G", "GT", ["0/1", "0/1", "0/1"]),
    ]
    path = tmp_path / "t.vcf.gz"
    with gzip.open(path, "wt") as f:
        f.write("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n")
        for c, p, i, r, a, fmt, g in records:
            f.write("\t".join([c, p, i, r, a, ".", "PASS", ".", fmt] + g) + "\n")
    ids, names, X, reasons = vcf_to_counted_dosage(path, lookup)
    assert ids == ["s1", "s2", "s3"] and names == ["m_alt", "m_ref", "m_comp"]
    np.testing.assert_array_equal(X[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(X[:2, 1], [2, 1]) and np.isnan(X[2, 1])
    np.testing.assert_array_equal(X[:, 2], [2, 1, 0])
    assert reasons["letters do not match the lookup"] == 1
    sub_ids, _, sub_X, _ = vcf_to_counted_dosage(path, lookup, markers=["m_alt"], samples=["s3", "s1"])
    assert sub_ids == ["s1", "s3"] and sub_X.shape == (2, 1) and list(sub_X[:, 0]) == [0, 2]


def test_orientation_follows_frequencies_and_refuses_to_guess():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.35, 2000)                      # clearly away from 0.5
    same = np.clip(p + rng.normal(0, 0.02, p.size), 0, 1)  # external ALT = counted allele
    flip = 1 - same                                        # external ALT = the other allele
    assert (orient_by_frequency(p, same) == 1).mean() > 0.99
    assert (orient_by_frequency(p, flip) == -1).mean() > 0.99
    near_half = orient_by_frequency(np.full(100, 0.5), np.full(100, 0.51))
    assert (near_half == 0).all()                          # both orientations fit: undecided
    misfit = orient_by_frequency(np.full(100, 0.1), np.full(100, 0.35))
    assert (misfit == 0).all()                             # neither orientation fits: undecided
    assert orient_by_frequency([np.nan], [0.2])[0] == 0


def test_strand_ambiguous_pairs():
    assert is_strand_ambiguous("A", "T") and is_strand_ambiguous("g", "c")
    assert not is_strand_ambiguous("A", "G") and not is_strand_ambiguous("C", "T")

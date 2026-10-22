"""Coordinate conversion and PLINK .bed decoding, the two new genotype-handling paths in 6C2.

Both are silent-failure risks: a wrong conversion matches the wrong variant, and a wrong .bed
decode flips dosages. Neither shows up as an exception, so they are checked against hand-worked
examples here.
"""
import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.real_data.liftover import ChainLiftOver  # noqa: E402


@pytest.fixture
def chain(tmp_path):
    """Two chains: chr1 forward with a gap, chr2 on the reverse strand."""
    text = (
        "chain 100 chr1 1000 + 100 260 1 900 + 50 220 1\n"
        "100 10 20\n"
        "50\n"
        "\n"
        "chain 100 chr2 500 + 0 100 2 900 - 0 100 2\n"
        "100\n"
        "\n"
    )
    path = tmp_path / "test.over.chain.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return ChainLiftOver(path)


def test_positions_convert_within_a_block(chain):
    # target block 100-200 (0-based) maps to query 50-150; 1-based 101 is the first base
    assert chain.convert("chr1", 101) == ("1", 51)
    assert chain.convert("chr1", 200) == ("1", 150)


def test_second_block_picks_up_the_chain_offsets(chain):
    # after dt=10/dq=20 the next block starts at target 210, query 170
    assert chain.convert("chr1", 211) == ("1", 171)


def test_positions_inside_a_gap_do_not_convert(chain):
    assert chain.convert("chr1", 205) is None


def test_positions_outside_every_block_do_not_convert(chain):
    assert chain.convert("chr1", 50) is None
    assert chain.convert("chr1", 900) is None


def test_unknown_chromosome_does_not_convert(chain):
    assert chain.convert("chr99", 101) is None


def test_chromosome_names_match_with_or_without_the_chr_prefix(chain):
    assert chain.convert("1", 101) == chain.convert("chr1", 101)


def test_reverse_strand_chains_are_counted_from_the_far_end(chain):
    # qSize 900, qStart 0: target base 0 maps to query base 899 (0-based) = 900 (1-based)
    assert chain.convert("chr2", 1) == ("2", 900)
    assert chain.convert("chr2", 2) == ("2", 899)


def test_convert_many_returns_one_entry_per_input(chain):
    got = chain.convert_many(["chr1", "chr1", "chr9"], [101, 205, 101])
    assert got == [("1", 51), None, None]


def _write_bed(path, snp_bytes):
    with open(path, "wb") as fh:
        fh.write(struct.pack("BBB", 0x6C, 0x1B, 0x01))  # magic + SNP-major
        fh.write(bytes(snp_bytes))


def test_bed_dosages_follow_the_counted_allele(tmp_path):
    from run_layer6c2_hairy_probe import read_bed_dosages

    # three samples in one byte: codes 00 (two A1), 10 (het), 11 (zero A1)
    _write_bed(tmp_path / "t.bed", [0b00111000])
    keep = pd.DataFrame({"bim_index": [0], "counted_is_a1": [True]})
    got = read_bed_dosages(keep, 3, bed_path=str(tmp_path / "t.bed"))
    assert got[:, 0].tolist() == [2.0, 1.0, 0.0]

    keep["counted_is_a1"] = [False]
    flipped = read_bed_dosages(keep, 3, bed_path=str(tmp_path / "t.bed"))
    assert flipped[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_bed_missing_calls_become_nan(tmp_path):
    from run_layer6c2_hairy_probe import read_bed_dosages

    # codes 01 (missing), 00, 00
    _write_bed(tmp_path / "m.bed", [0b00000001])
    keep = pd.DataFrame({"bim_index": [0], "counted_is_a1": [True]})
    got = read_bed_dosages(keep, 3, bed_path=str(tmp_path / "m.bed"))
    assert np.isnan(got[0, 0])
    assert got[1, 0] == 2.0


def test_bed_markers_are_read_by_index_not_in_order(tmp_path):
    from run_layer6c2_hairy_probe import read_bed_dosages

    # three SNPs, one byte each; ask for the third then the first
    _write_bed(tmp_path / "i.bed", [0b00000000, 0b00001010, 0b00111111])
    keep = pd.DataFrame({"bim_index": [2, 0], "counted_is_a1": [True, True]})
    got = read_bed_dosages(keep, 3, bed_path=str(tmp_path / "i.bed"))
    assert got[:, 0].tolist() == [0.0, 0.0, 0.0]  # SNP 2: all 11
    assert got[:, 1].tolist() == [2.0, 2.0, 2.0]  # SNP 0: all 00


def test_the_stored_chain_file_matches_the_panel_assembly():
    """The real chain converts the bulk of a UMD3.1 map; a wrong file would convert almost none."""
    chain_path = ROOT / "data/external/ucsc_chain/bosTau6ToBosTau9.over.chain.gz"
    map_path = ROOT / "data/external/decker2014_worldwide/extracted/ww_cattle_dryad_20Sept2013.map"
    if not chain_path.exists() or not map_path.exists():
        pytest.skip("external chain or Decker map not present")
    mp = pd.read_csv(map_path, sep=r"\s+", header=None, names=["chr", "snp", "cm", "bp"],
                     dtype={"chr": str}).head(2000)
    lifted = ChainLiftOver(chain_path).convert_many(mp["chr"], mp["bp"])
    assert np.mean([x is not None for x in lifted]) > 0.95

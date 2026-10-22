"""Assembly coordinate conversion from a UCSC chain file.

The Florida/Mizzou 6C panel carries ARS-UCD1.2 coordinates, while public cattle datasets
published before ~2018 (Decker et al. 2014, Littlejohn et al. 2014) carry UMD3.1. Matching
markers by position across those two needs a coordinate conversion, so this module reads a
UCSC chain file and maps single positions.

Conversion alone is not evidence that two markers are the same variant: callers must still
check that the alleles agree (directly or on the complement strand), which is what
`src.real_data.harmonisation` does. A high allele-agreement rate after conversion is the
signal that the mapping is real rather than coincidental.
"""
import bisect
import gzip
from pathlib import Path


class ChainLiftOver:
    """Positions in the chain's target assembly -> positions in its query assembly.

    Coordinates in and out are 1-based, as in PLINK map/bim files; the chain file itself is
    0-based half-open, and that difference is handled here.
    """

    def __init__(self, chain_path):
        self.path = Path(chain_path)
        self._blocks = self._load(self.path)

    @staticmethod
    def _load(path):
        """chrom -> (block starts, blocks), each block (t_start, t_end, q_start, q_name, q_strand, q_size)."""
        by_chrom: dict[str, list] = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            t_name = q_name = q_strand = None
            t_pos = q_pos = q_size = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("chain"):
                    f = line.split()
                    t_name, t_pos = f[2], int(f[5])
                    q_name, q_size, q_strand, q_pos = f[7], int(f[8]), f[9], int(f[10])
                    continue
                if t_name is None:
                    continue
                f = line.split()
                size = int(f[0])
                by_chrom.setdefault(t_name, []).append(
                    (t_pos, t_pos + size, q_pos, q_name, q_strand, q_size))
                if len(f) == 3:
                    t_pos += size + int(f[1])
                    q_pos += size + int(f[2])
                else:
                    t_name = None  # last block of this chain
        for chrom in by_chrom:
            by_chrom[chrom].sort()
        return {c: ([b[0] for b in v], v) for c, v in by_chrom.items()}

    def convert(self, chrom, position):
        """(chrom, 1-based position) in the new assembly, or None where the position does not map.

        Chromosome names match with or without a `chr` prefix on either side.
        """
        chrom = str(chrom)
        key = chrom if chrom in self._blocks else f"chr{chrom}"
        if key not in self._blocks:
            return None
        starts, blocks = self._blocks[key]
        p0 = int(position) - 1
        i = bisect.bisect_right(starts, p0) - 1
        if i < 0:
            return None
        t_start, t_end, q_start, q_name, q_strand, q_size = blocks[i]
        if not (t_start <= p0 < t_end):
            return None  # inside a gap between blocks
        offset = p0 - t_start
        q0 = q_start + offset if q_strand == "+" else q_size - (q_start + offset) - 1
        return q_name.replace("chr", ""), q0 + 1

    def convert_many(self, chroms, positions):
        """Vectorised `convert`: a list of (chrom, position) tuples or None, one per input."""
        return [self.convert(c, p) for c, p in zip(chroms, positions)]

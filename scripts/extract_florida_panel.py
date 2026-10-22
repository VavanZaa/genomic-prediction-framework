"""Extract the Florida SNP dosages and breed-of-origin counts at the Layer 6C panel markers.

Reads data/external/panel_6c_markers.csv, streams SNP.raw.gz and BOA.raw.gz by column name, joins rows by
animal ID (BOA defines the animal set, as in the v3 loader) and writes data/external/florida_panel_6c.npz:
- snp: counted-allele dosage (NaN missing), animals x panel markers;
- boa: Brahman-origin allele count (NaN missing, or where the marker is not in the BOA file);
- ids, markers, in_boa (marker present in BOA.raw.gz).
BOA counts are flipped to Brahman-origin counts where the file counts the Angus allele (suffix _A).
"""
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

FLORIDA = ROOT / "data" / "real_cattle" / "florida_multibreed"
PANEL = ROOT / "data" / "external" / "panel_6c_markers.csv"
OUT = ROOT / "data" / "external" / "florida_panel_6c.npz"


def read_columns(path: Path, markers):
    with gzip.open(path, "rt") as f:
        header = f.readline().split()[1:]
        where = {c.rsplit("_", 1)[0]: (i + 1, c.rsplit("_", 1)[1]) for i, c in enumerate(header)}
        present = [m for m in markers if m in where]
        idx = np.array([where[m][0] for m in present])
        suffix = [where[m][1] for m in present]
        rows = {}
        for line in f:
            parts = line.split()
            vals = np.asarray(parts, dtype=object)[idx]
            ok = vals != "NA"
            x = np.full(len(idx), np.nan, dtype=np.float32)
            x[ok] = vals[ok].astype(np.float32)
            x[x < 0] = np.nan
            rows[parts[0]] = x
    return present, suffix, rows


def main():
    markers = pd.read_csv(PANEL)["marker"].tolist()
    snp_markers, _, snp_rows = read_columns(FLORIDA / "SNP.raw.gz", markers)
    if len(snp_markers) != len(markers):
        raise SystemExit(f"{len(markers) - len(snp_markers)} panel markers missing from SNP.raw.gz")
    boa_markers, boa_suffix, boa_rows = read_columns(FLORIDA / "BOA.raw.gz", markers)
    ids = [i for i in boa_rows if i in snp_rows]
    col = {m: j for j, m in enumerate(boa_markers)}
    in_boa = np.array([m in col for m in markers])
    flip = np.array([s.upper() == "A" for s in boa_suffix])
    snp = np.stack([snp_rows[i] for i in ids])
    boa = np.full(snp.shape, np.nan, dtype=np.float32)
    if boa_markers:
        B = np.stack([boa_rows[i] for i in ids])
        B[:, flip] = 2 - B[:, flip]
        boa[:, [markers.index(m) for m in boa_markers]] = B
    np.savez_compressed(OUT, snp=snp, boa=boa, ids=np.array(ids), markers=np.array(markers), in_boa=in_boa)
    print(f"animals {len(ids)}, panel markers {len(markers)}, in BOA file {int(in_boa.sum())}, "
          f"SNP missing {np.isnan(snp).mean():.4f} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""SNP-array ascertainment: turn simulated segregating sites into an array-like marker panel.

Real genotypes come from a SNP array; a coalescent simulation returns every segregating site, most of
them rare. Comparing the two directly mixes biology with the measurement process. This operator,
A: all simulated sites -> array-like markers, is fitted on the real panel and matches
- the minor-allele-frequency spectrum of the real markers (in `n_bins` MAF strata),
- the MAF floor used in the real-data QC (`min_maf`),
- even spacing along each chromosome within every MAF stratum,
- the real missing-call rate (`add_missing_calls`; impute afterwards exactly as for the real data).
The MAF of each simulated site comes from a discovery panel chosen by the caller. Marker density per
Mb is not matched when the simulated genome is shorter than the real one; `markers_per_mb_` and
`median_gap_bp_` are recorded so the comparison can report the difference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class ArrayAscertainment:
    def __init__(self, min_maf: float = 0.03, n_bins: int = 10):
        self.min_maf, self.n_bins = min_maf, n_bins

    def fit(self, genotypes, marker_map: pd.DataFrame) -> "ArrayAscertainment":
        """Learns the real panel's MAF spectrum, missing-call rate and spacing (chrom, pos_bp columns)."""
        X = np.asarray(genotypes, dtype=float)
        called = np.isfinite(X) & (X >= 0)
        self.missing_rate_ = float(1.0 - called.mean())
        n_called = called.sum(axis=0)
        p = np.divide(np.where(called, X, 0.0).sum(axis=0), 2.0 * n_called,
                      out=np.full(X.shape[1], np.nan), where=n_called > 0)
        maf = np.minimum(p, 1.0 - p)
        self.target_maf_ = maf[np.isfinite(maf) & (maf >= self.min_maf)]
        if len(self.target_maf_) == 0:
            raise ValueError("No real markers pass the MAF floor")
        self.edges_ = np.linspace(self.min_maf, 0.5, self.n_bins + 1)
        counts, _ = np.histogram(self.target_maf_, bins=self.edges_)
        self.bin_share_ = counts / counts.sum()
        gaps, span_mb = [], 0.0
        for _, g in marker_map.groupby("chrom"):
            pos = np.sort(g["pos_bp"].to_numpy(dtype=float))
            if len(pos) > 1:
                gaps.append(np.diff(pos))
                span_mb += (pos[-1] - pos[0]) / 1e6
        self.median_gap_bp_ = float(np.median(np.concatenate(gaps))) if gaps else float("nan")
        self.markers_per_mb_ = float(len(marker_map) / span_mb) if span_mb > 0 else float("nan")
        return self

    def _allocate(self, n: int) -> np.ndarray:
        """Largest-remainder split of n markers across the MAF strata."""
        raw = n * self.bin_share_
        base = np.floor(raw).astype(int)
        base[np.argsort(-(raw - base))[: n - base.sum()]] += 1
        return base

    def select(self, discovery_freq, positions, n_markers: int) -> np.ndarray:
        """Indices of up to `n_markers` sites matching the real MAF spectrum, evenly spaced per stratum."""
        freq, pos = np.asarray(discovery_freq, dtype=float), np.asarray(positions)
        maf = np.minimum(freq, 1.0 - freq)
        eligible = np.where(maf >= self.min_maf)[0]
        if len(eligible) <= n_markers:
            self.last_shortfall_ = int(n_markers - len(eligible))
            return eligible
        stratum = np.clip(np.searchsorted(self.edges_, maf[eligible], side="right") - 1, 0, self.n_bins - 1)
        chosen = []
        for b, want in enumerate(self._allocate(n_markers)):
            cand = eligible[stratum == b]
            cand = cand[np.argsort(pos[cand], kind="stable")]
            take = min(int(want), len(cand))
            if take:
                chosen.extend(cand[np.round(np.linspace(0, len(cand) - 1, take)).astype(int)])
        chosen = np.unique(np.asarray(chosen, dtype=int))
        short = n_markers - len(chosen)
        self.last_shortfall_ = int(max(short, 0))
        if short > 0:
            # Strata the simulation cannot fill: take the remaining eligible sites, evenly spaced
            rest = np.setdiff1d(eligible, chosen)
            rest = rest[np.argsort(pos[rest], kind="stable")]
            chosen = np.union1d(chosen, rest[np.round(np.linspace(0, len(rest) - 1, short)).astype(int)])
        return np.sort(chosen)

    def add_missing_calls(self, genotypes, rng: np.random.Generator) -> np.ndarray:
        """Sets calls to NaN at the real missing rate (impute them as the real data are imputed)."""
        X = np.asarray(genotypes, dtype=float).copy()
        X[rng.random(X.shape) < self.missing_rate_] = np.nan
        return X

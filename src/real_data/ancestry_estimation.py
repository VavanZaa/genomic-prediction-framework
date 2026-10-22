"""Supervised Brahman-fraction estimation from genotypes, with breed-specific allele frequencies.

Breed-specific frequencies come from the Florida files, which give each animal's SNP dosage x (copies of
the counted allele) and breed-of-origin count o (Brahman-origin alleles) at the same marker. Where o = 0
both alleles are Angus-origin, so x/2 samples the Angus frequency; where o = 2 both are Brahman-origin.
Loci with o = 1 do not say which allele came from which breed and are skipped.

An animal's Brahman fraction q maximises sum_j log Binomial(x_j; 2, (1 - q) pA_j + q pB_j) over a grid.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-3


def breed_frequencies(snp: np.ndarray, boa: np.ndarray, min_count: int = 20):
    """Angus and Brahman counted-allele frequencies per marker from SNP dosages and BOA counts.

    Returns (p_angus, p_brahman, n_angus, n_brahman); frequencies are NaN where fewer than `min_count`
    animals are homozygous for that breed's origin at the marker.
    """
    x = np.asarray(snp, dtype=float)
    o = np.asarray(boa, dtype=float)
    out = []
    for origin in (0.0, 2.0):
        m = (o == origin) & np.isfinite(x)
        n = m.sum(axis=0)
        p = np.where(n >= min_count, np.where(m, x, 0.0).sum(axis=0) / (2.0 * np.maximum(n, 1)), np.nan)
        out.append((p, n))
    (pa, na), (pb, nb) = out
    return pa, pb, na, nb


def estimate_brahman_fraction(genotypes: np.ndarray, p_angus: np.ndarray, p_brahman: np.ndarray,
                              grid_step: float = 0.005) -> np.ndarray:
    """Maximum-likelihood Brahman fraction per animal (rows of `genotypes`, counted-allele dosages)."""
    G = np.asarray(genotypes, dtype=float)
    ok = np.isfinite(p_angus) & np.isfinite(p_brahman)
    G, pa, pb = G[:, ok], np.clip(p_angus[ok], EPS, 1 - EPS), np.clip(p_brahman[ok], EPS, 1 - EPS)
    grid = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
    P = (1 - grid)[:, None] * pa[None, :] + grid[:, None] * pb[None, :]          # (grid, markers)
    logp, log1p = np.log(P), np.log(1 - P)
    called = np.isfinite(G)
    X = np.where(called, G, 0.0)
    # log-likelihood (up to a constant) = x log p + (2 - x) log(1 - p), over called markers only
    ll = X @ logp.T + (2.0 * called - X) @ log1p.T                               # (animals, grid)
    return grid[np.argmax(ll, axis=1)]

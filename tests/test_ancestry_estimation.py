"""Supervised Brahman-fraction estimation from breed-specific allele frequencies."""
import numpy as np

from src.real_data.ancestry_estimation import breed_frequencies, estimate_brahman_fraction


def _simulate(rng, q, pa, pb):
    """Genotypes and BOA counts for animals of Brahman fraction q (alleles drawn independently)."""
    n, m = len(q), len(pa)
    origin = rng.random((n, m, 2)) < q[:, None, None]                   # True = Brahman-origin allele
    p = np.where(origin, pb[None, :, None], pa[None, :, None])
    alleles = rng.random((n, m, 2)) < p
    return alleles.sum(axis=2).astype(float), origin.sum(axis=2).astype(float)


def test_breed_frequencies_and_fraction_are_recovered():
    rng = np.random.default_rng(0)
    m = 3000
    pa, pb = rng.uniform(0.05, 0.95, m), rng.uniform(0.05, 0.95, m)
    ref_q = rng.uniform(0.0, 1.0, 1500)
    snp, boa = _simulate(rng, ref_q, pa, pb)
    est_pa, est_pb, na, nb = breed_frequencies(snp, boa)
    ok = np.isfinite(est_pa) & np.isfinite(est_pb)
    assert ok.mean() > 0.95
    assert np.corrcoef(est_pa[ok], pa[ok])[0, 1] > 0.97 and np.corrcoef(est_pb[ok], pb[ok])[0, 1] > 0.97

    new_q = np.array([0.0, 0.1, 0.375, 0.5, 0.8, 1.0])
    new_snp, _ = _simulate(rng, new_q, pa, pb)
    new_snp[rng.random(new_snp.shape) < 0.05] = np.nan                  # missing calls are skipped
    est = estimate_brahman_fraction(new_snp, est_pa, est_pb)
    np.testing.assert_allclose(est, new_q, atol=0.03)

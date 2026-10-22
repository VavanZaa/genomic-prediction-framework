"""Vectorized bivariate standard-normal CDF.

Sheppard's integral:
    Phi2(h, k; rho) = Phi(h) Phi(k) + 1/(2 pi) * int_0^{asin(rho)} exp(-(h^2 - 2hk sin t + k^2) / (2 cos^2 t)) dt
evaluated with 24-point Gauss-Legendre quadrature. Accurate to ~1e-7 for |rho| <= 0.95; rho is
clipped to (-0.999999, 0.999999).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)


def bivariate_normal_cdf(h, k, rho) -> np.ndarray:
    """P(Z1 <= h, Z2 <= k) for standard normals with correlation rho (broadcast over inputs)."""
    h, k, rho = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(k, dtype=float), np.asarray(rho, dtype=float))
    upper = np.arcsin(np.clip(rho, -0.999999, 0.999999))
    t = 0.5 * upper[..., None] * (_NODES + 1.0)          # nodes mapped to [0, asin(rho)]
    hh, kk = h[..., None], k[..., None]
    integrand = np.exp(-(hh ** 2 - 2.0 * hh * kk * np.sin(t) + kk ** 2) / (2.0 * np.cos(t) ** 2))
    integral = 0.5 * upper * np.sum(_WEIGHTS * integrand, axis=-1)
    ph, pk = norm.cdf(h), norm.cdf(k)
    p = ph * pk + integral / (2.0 * np.pi)
    return np.clip(p, np.maximum(0.0, ph + pk - 1.0), np.minimum(ph, pk))

"""Training-reference genotype validation, imputation and input provenance."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np

def validate_ids(values, label="animal IDs"):
    ids = [str(v) for v in values]
    if any(v.strip() in ("", "nan", "None", "<NA>") for v in ids):
        raise ValueError(f"Missing {label}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label}; this API requires one record per animal")
    return ids

class GenotypeImputer:
    """Mean imputation learned on training rows; -1/-9 and NaN are missing.

    An entirely missing training column has no estimable frequency and is rejected.
    """
    @staticmethod
    def _array(X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim != 2 or not all(X.shape):
            raise ValueError("Expected a nonempty animal x marker matrix")
        X[(X == -1) | (X == -9)] = np.nan
        if np.isinf(X).any() or np.any((X < 0) | (X > 2)):
            raise ValueError("Genotype dosages must be in [0, 2] or missing")
        return X

    def fit(self, X):
        X = self._array(X)
        n = np.sum(np.isfinite(X), axis=0)
        if np.any(n == 0):
            raise ValueError("Entirely missing training marker; drop it using training data first")
        self.means_ = np.nansum(X, axis=0) / n
        return self

    def transform(self, X):
        if not hasattr(self, "means_"):
            raise RuntimeError("Fit genotype imputation on training rows first")
        X = self._array(X)
        if X.shape[1] != len(self.means_):
            raise ValueError("Marker count differs from training")
        return np.where(np.isnan(X), self.means_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

@lru_cache(maxsize=32)
def _hash(path, size, mtime):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def source_hashes(paths):
    return {Path(p).name: _hash(str(Path(p).resolve()), Path(p).stat().st_size,
                               Path(p).stat().st_mtime_ns) for p in paths}

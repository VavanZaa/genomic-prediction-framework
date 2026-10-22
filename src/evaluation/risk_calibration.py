"""Risk-controlled selection of an applicability (support) threshold.

For candidate thresholds c on a support score s (larger = further from the training data):
    coverage C(c) = P(s <= c),   risk R(c) = E[loss | s <= c].
The chosen threshold c* is the largest c whose risk meets a target fixed in advance, using the
one-sided 95 % bootstrap upper bound of R(c) (records resampled by unit, e.g. animal). If no candidate
meets the target, no threshold is chosen and that kind of prediction is refused.
See docs/OOD_CALIBRATION.md.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd


def risk_coverage(scores, losses, grid: Sequence[float], units=None, n_boot: int = 1000,
                  seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Coverage, point risk and bootstrap upper risk bound at each threshold in `grid`."""
    s, loss = np.asarray(scores, dtype=float), np.asarray(losses, dtype=float)
    if s.shape != loss.shape or not np.isfinite(s).all() or not np.isfinite(loss).all():
        raise ValueError("Scores and losses must be finite and of equal length")
    grid = np.asarray(grid, dtype=float)
    units = np.arange(len(s)) if units is None else np.asarray(units)
    uniq, inverse = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(seed)
    # Bootstrap weight of each record = number of times its unit is drawn
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    unit_counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])      # (B, U)
    weights = unit_counts[:, inverse].astype(float)                                   # (B, N)
    accepted = s[None, :] <= grid[:, None]                                            # (G, N)
    n_acc = accepted.sum(axis=1)
    risk = np.array([loss[a].mean() if a.any() else np.nan for a in accepted])
    num = weights @ (accepted * loss).T                                               # (B, G)
    den = weights @ accepted.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num / den
    ucb = np.nanquantile(boot, level, axis=0)
    return pd.DataFrame({"threshold": grid, "n_accepted": n_acc, "coverage": n_acc / len(s),
                         "risk": risk, "risk_ucb": ucb})


def select_threshold(curve: pd.DataFrame, risk_max: float, min_accepted: int = 100) -> Optional[float]:
    """Largest threshold whose upper risk bound is within `risk_max` with enough accepted records."""
    ok = curve[(curve["n_accepted"] >= min_accepted) & (curve["risk_ucb"] <= risk_max)]
    return None if ok.empty else float(ok["threshold"].max())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_thresholds(path, expected_sha256: Optional[str] = None) -> dict:
    """Loads calibrated thresholds; refuses a file whose hash differs from the frozen one."""
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"{path} does not match the frozen SHA-256; thresholds must not change after freezing")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"ranking": data["c_star"]["ranking"], "absolute": data["c_star"]["absolute"], "metadata": data}

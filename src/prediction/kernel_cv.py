"""Training-only K-fold selection of kernel-ridge (GBLUP) penalties and kernel weights.

For each candidate weight vector the combined kernel K = sum_i w_i K_i is eigendecomposed once per
fold, so every ridge penalty is scored from the same decomposition:
    y_hat_val = mu + K_val,tr V diag(1 / (s + lambda)) V' (y_tr - mu).
Selection minimizes out-of-fold MSE; only the rows passed in (the training cohort) are used.
"""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd

DEFAULT_LAMBDAS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
TWO_KERNEL_WEIGHTS: Tuple[Tuple[float, float], ...] = tuple((w, round(1.0 - w, 2)) for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


def cv_kernel_ridge(
    kernels: Sequence[np.ndarray],
    y: np.ndarray,
    weight_grid: Sequence[Sequence[float]] = ((1.0,),),
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    n_folds: int = 5,
    seed: int = 0,
    groups=None,
) -> Dict:
    """Pick kernel weights and ridge penalty by out-of-fold MSE on the training rows.

    kernels: training x training kernel matrices; weight_grid: candidate weights, one per kernel.
    Returns the best {"weights", "lambda", "cv_mse", "cv_r"} plus the full score "table".
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4 or n_folds < 2:
        raise ValueError("Cross-validation requires at least four records and two folds")
    if groups is None:
        folds = np.array_split(np.random.default_rng(seed).permutation(n), min(n_folds,n))
    else:
        from sklearn.model_selection import GroupKFold
        groups=np.asarray(groups)
        if len(np.unique(groups))<2: raise ValueError("At least two independent groups required")
        folds=[va for _,va in GroupKFold(min(n_folds,len(np.unique(groups)))).split(y,groups=groups)]
    rows = []
    for w in weight_grid:
        K = sum(wi * Ki for wi, Ki in zip(w, kernels))
        oof = np.zeros((len(lambdas), n))
        for val in folds:
            tr = np.setdiff1d(np.arange(n), val)
            mu = float(np.mean(y[tr]))
            kt,kv=center_kernel(K[np.ix_(tr,tr)],K[np.ix_(val,tr)])
            s, V = np.linalg.eigh(kt)
            s=np.maximum(s,0)
            proj_y = V.T @ (y[tr] - mu)
            k_val = kv @ V
            for li, lam in enumerate(lambdas):
                oof[li, val] = mu + k_val @ (proj_y / (s + lam))
        for li, lam in enumerate(lambdas):
            rows.append({
                "weights": tuple(float(x) for x in w),
                "lambda": float(lam),
                "cv_mse": float(np.mean((y - oof[li]) ** 2)),
                "cv_r": float(np.corrcoef(y, oof[li])[0, 1]) if np.std(oof[li]) > 0 else 0.0,
            })
    table = pd.DataFrame(rows)
    best = table.loc[table["cv_mse"].idxmin()].to_dict()
    return {**best, "table": table}


def tune_snp_boa_models(
    G: np.ndarray,
    B: np.ndarray,
    traits: Dict[str, np.ndarray],
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[Dict[Tuple[str, str], Dict], pd.DataFrame]:
    """Tune the SNP (G), breed-of-origin (B) and combined (w G + (1 - w) B) GBLUP models per trait.

    Returns {(model, trait): best} and one table with every weight/penalty scored.
    """
    best, tables = {}, []
    for trait, y in traits.items():
        for model, kernels, grid in (("SNP", [G], ((1.0,),)), ("BOA", [B], ((1.0,),)), ("Combined", [G, B], TWO_KERNEL_WEIGHTS)):
            res = cv_kernel_ridge(kernels, y, grid, n_folds=n_folds, seed=seed)
            best[(model, trait)] = res
            tables.append(res["table"].assign(model=model, trait=trait))
    return best, pd.concat(tables, ignore_index=True)


def center_kernel(train, cross):
    """Center a feature kernel using only its training rows; supports new batches."""
    train=np.asarray(train,dtype=float); cross=np.asarray(cross,dtype=float)
    column=train.mean(axis=0); grand=float(train.mean())
    return train-column[None,:]-train.mean(axis=1)[:,None]+grand, cross-column[None,:]-cross.mean(axis=1)[:,None]+grand


def predict_kernel(train, cross, y, lam):
    kt,kv=center_kernel(train,cross)
    mu=float(np.mean(y))
    return mu+kv@np.linalg.solve(kt+lam*np.eye(len(y)),np.asarray(y)-mu)


def nested_kernel_oof(kernels,y,groups=None,weight_grid=((1.,),),n_folds=5,seed=42,fixed_lambda=None):
    """Outer OOF residual predictions; all penalty/weight selection stays inside each fold.

    Input kernels use a declared historical genomic reference. Fold centering and
    all phenotype-dependent fitting exclude the scored fold.
    """
    from sklearn.model_selection import KFold,GroupKFold
    y=np.asarray(y);n=len(y)
    if groups is None:
        folds=KFold(min(n_folds,n),shuffle=True,random_state=seed).split(y)
    else:
        groups=np.asarray(groups)
        folds=GroupKFold(min(n_folds,len(np.unique(groups)))).split(y,groups=groups)
    pred=np.empty(n)
    for tr,va in folds:
        ks=[k[np.ix_(tr,tr)] for k in kernels]
        if fixed_lambda is None:
            innergroups=None if groups is None else groups[tr]
            # A two-group outer split leaves one group: fall back to record CV
            # within that training group, while keeping the outer group untouched.
            if innergroups is not None and len(np.unique(innergroups))<2: innergroups=None
            best=cv_kernel_ridge(ks,y[tr],weight_grid,n_folds=min(3,len(tr)),seed=seed,groups=innergroups)
            weights,lam=best['weights'],best['lambda']
        else:
            weights,lam=weight_grid[0],fixed_lambda
        kt=sum(w*k for w,k in zip(weights,ks))
        kv=sum(w*k[np.ix_(va,tr)] for w,k in zip(weights,kernels))
        pred[va]=predict_kernel(kt,kv,y[tr],lam)
    return pred

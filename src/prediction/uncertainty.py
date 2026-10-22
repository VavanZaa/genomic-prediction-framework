"""Predictive residuals from folds that exclude each scored record."""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

def out_of_fold_predictions(X, y, factory, groups=None, n_folds=5, seed=0):
    X=np.asarray(X); y=np.asarray(y,dtype=float)
    n=len(y)
    if n<4: raise ValueError("At least four training records are required for predictive uncertainty")
    if groups is None:
        splits=KFold(n_splits=min(n_folds,n),shuffle=True,random_state=seed).split(X)
    else:
        groups=np.asarray(groups); k=min(n_folds,len(np.unique(groups)))
        if k<2: raise ValueError("Predictive uncertainty requires at least two held-out groups")
        splits=GroupKFold(n_splits=k).split(X,y,groups)
    pred=np.full(n,np.nan)
    for tr,va in splits:
        pred[va]=factory().fit(X[tr],y[tr]).predict(X[va])
    if not np.isfinite(pred).all(): raise ValueError("Non-finite out-of-fold predictions")
    return pred

def residual_covariance(X,y1,y2,factory,groups=None):
    r=np.column_stack([y1-out_of_fold_predictions(X,y1,factory,groups),
                       y2-out_of_fold_predictions(X,y2,factory,groups)])
    cov=np.cov(r,rowvar=False)
    return cov + np.eye(2)*1e-10

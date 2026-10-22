"""Fixed-effect environmental model for adjusting raw cattle phenotypes by age, sex, and contemporary group.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass
class EnvironmentalFitResult:
    trait_name: str
    r2_environment: float
    variance_raw: float
    variance_adjusted: float
    coefficients: Dict[str, float]
    intercept: float
    adjusted_phenotypes: np.ndarray


class EnvironmentalCovariateModel:
    """Adjusts raw carcass and growth phenotypes for management, sex, and slaughter age covariates."""

    def __init__(self, continuous_covariates: Optional[List[str]] = None, categorical_covariates: Optional[List[str]] = None):
        self.cont_covs = ["age_days"] if continuous_covariates is None else continuous_covariates
        self.cat_covs = ["sex", "contemporary_group", "feed_regimen"] if categorical_covariates is None else categorical_covariates
        self.encodings = {}
        self.training_means = {}
        self.fitted_models: Dict[str, LinearRegression] = {}
        self.design_columns: Dict[str, List[str]] = {}

    def _build_design_matrix(self, df, trait_col, fitting=False):
        if fitting:
            self.encodings[trait_col] = {
                "continuous": {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in self.cont_covs if c in df},
                "categorical": {c: sorted(df[c].fillna("<missing>").astype(str).unique()) for c in self.cat_covs if c in df},
            }
        enc=self.encodings[trait_col]; parts=[]; names=[]
        for c,median in enc["continuous"].items():
            vals=df[c] if c in df else pd.Series(np.nan,index=df.index)
            parts.append(vals.fillna(median).to_numpy(dtype=float)); names.append(c)
        for c,levels in enc["categorical"].items():
            vals=df[c].fillna("<missing>").astype(str) if c in df else pd.Series("<missing>",index=df.index)
            for level in levels[1:]:
                parts.append((vals==level).to_numpy(dtype=float)); names.append(f"{c}_{level}")
        return (np.column_stack(parts), names) if parts else (np.ones((len(df),1)),["intercept"])

    def transform(self, df, trait_col):
        if trait_col not in self.fitted_models:
            raise RuntimeError("Fit the environmental model on training records first")
        X,_=self._build_design_matrix(df,trait_col)
        return df[trait_col].to_numpy(dtype=float) - (self.fitted_models[trait_col].predict(X)-self.training_means[trait_col])

    def fit_and_adjust(self, df: pd.DataFrame, trait_col: str) -> EnvironmentalFitResult:
        """Fits fixed-effect model and returns adjusted phenotype: y* = y - X * beta."""
        y_raw = df[trait_col].values.astype(np.float64)
        X, col_names = self._build_design_matrix(df, trait_col, fitting=True)

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y_raw)
        y_pred_env = reg.predict(X)

        # Compute adjusted phenotype centered around raw mean
        raw_mean = float(np.mean(y_raw))
        self.training_means[trait_col] = raw_mean
        y_adjusted = y_raw - (y_pred_env - raw_mean)

        r2 = float(r2_score(y_raw, y_pred_env))
        coef_dict = {name: float(coef) for name, coef in zip(col_names, reg.coef_)}

        self.fitted_models[trait_col] = reg
        self.design_columns[trait_col] = col_names

        return EnvironmentalFitResult(
            trait_name=trait_col,
            r2_environment=r2,
            variance_raw=float(np.var(y_raw)),
            variance_adjusted=float(np.var(y_adjusted)),
            coefficients=coef_dict,
            intercept=float(reg.intercept_),
            adjusted_phenotypes=y_adjusted
        )

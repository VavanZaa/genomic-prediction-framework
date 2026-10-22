"""Decoupled Environmental Scenario Specification & Phenotypic Likelihood Engine.

Decouples intrinsic genetic merit (GEBVs, segregation variance) from the macro-environment
and contemporary group management conditions:
    y = G_offspring + E + epsilon

Enables commercial producers to evaluate:
"What mating is optimal for Florida subtropical conditions?"
versus
"What mating is optimal for a cooler or elevated-stress environment?"
while keeping the genetic inheritance engine invariant.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm
from .bivariate_normal import bivariate_normal_cdf


@dataclass
class EnvironmentScenario:
    """Defines a macro-environmental context E = (mu_env, Sigma_e)."""
    name: str
    description: str
    mu_load: float          # Mean environmental effect for LoadSlope
    mu_hair: float          # Mean environmental effect for AvgShortLength (mm)
    sigma_e_load: float     # Residual standard deviation for LoadSlope
    sigma_e_hair: float     # Residual standard deviation for AvgShortLength
    rho_e: float = 0.0      # Environmental residual correlation between traits
    source: str = "assumed" # provenance of the numbers (see docs/SOURCE_INVENTORY.md)


class EnvironmentEvaluator:
    """Evaluates offspring phenotypic threshold attainment conditional on an environmental scenario."""

    # Default canonical Florida multibreed environments estimated from empirical data
    CANONICAL_SCENARIOS: Dict[str, EnvironmentScenario] = {
        "Subtropical_Baseline": EnvironmentScenario(
            name="Subtropical_Baseline",
            description="Florida Subtropical Research Herd (Seminole Tribe conditions 2016-2018)",
            mu_load=29.58,
            mu_hair=6.80,
            sigma_e_load=8.28,
            sigma_e_hair=3.21,
            rho_e=0.0245,
            source="unverified: Seminole 2016-2018 records give LoadSlope mean 31.58 (sd 10.18) and hair 6.97 (sd 2.14)",
        ),
        "Elevated_Heat_Shifted": EnvironmentScenario(
            name="Elevated_Heat_Shifted",
            description="Commercial Pasture Conditions with Hair Length Shift (Williamson cohort 2021-2022)",
            mu_load=29.58,
            mu_hair=8.56,       # +1.76 mm shift (+25.8%)
            sigma_e_load=8.28,
            sigma_e_hair=3.21,
            rho_e=0.0245,
            source="hair mean measured (Wes 2021-2022 mean 8.56); load mean and SDs unverified",
        ),
        "Temperate_Cool": EnvironmentScenario(
            name="Temperate_Cool",
            description="Moderate/Temperate Environment (Low heat stress, short winter coat)",
            mu_load=20.00,
            mu_hair=5.50,
            sigma_e_load=7.50,
            sigma_e_hair=2.80,
            rho_e=0.0100,
            source="assumed (no data)",
        ),
    }

    @staticmethod
    def evaluate_attainment(
        gebv_load: Union[float, np.ndarray],
        gebv_hair: Union[float, np.ndarray],
        tau_load: float,
        tau_hair: float,
        scenario: EnvironmentScenario,
        var_seg_load: Union[float, np.ndarray] = 0.0,
        var_seg_hair: Union[float, np.ndarray] = 0.0,
        cov_seg: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes marginal and bivariate normal CDF attainment probabilities for lower-is-better traits.

        Returns:
            prob_load: P(y_load < tau_load | gebv, scenario)
            prob_hair: P(y_hair < tau_hair | gebv, scenario)
            prob_joint: P(y_load < tau_load AND y_hair < tau_hair | gebv, scenario)
        """
        gebv_load = np.asarray(gebv_load, dtype=np.float64)
        gebv_hair = np.asarray(gebv_hair, dtype=np.float64)
        is_scalar = (gebv_load.ndim == 0)

        gebv_l_flat = np.atleast_1d(gebv_load)
        gebv_h_flat = np.atleast_1d(gebv_hair)
        v_seg_l = np.atleast_1d(np.asarray(var_seg_load, dtype=np.float64))
        v_seg_h = np.atleast_1d(np.asarray(var_seg_hair, dtype=np.float64))
        c_seg = np.atleast_1d(np.asarray(cov_seg, dtype=np.float64))

        if len(v_seg_l) == 1 and len(gebv_l_flat) > 1:
            v_seg_l = np.repeat(v_seg_l, len(gebv_l_flat))
        if len(v_seg_h) == 1 and len(gebv_h_flat) > 1:
            v_seg_h = np.repeat(v_seg_h, len(gebv_h_flat))
        if len(c_seg) == 1 and len(gebv_l_flat) > 1:
            c_seg = np.repeat(c_seg, len(gebv_l_flat))

        var_e_l = scenario.sigma_e_load ** 2
        var_e_h = scenario.sigma_e_hair ** 2
        cov_e = scenario.rho_e * scenario.sigma_e_load * scenario.sigma_e_hair

        mu_l = gebv_l_flat + scenario.mu_load
        mu_h = gebv_h_flat + scenario.mu_hair

        tot_var_l = np.maximum(var_e_l + v_seg_l, 1e-4)
        tot_var_h = np.maximum(var_e_h + v_seg_h, 1e-4)
        tot_cov = cov_e + c_seg

        sd_l = np.sqrt(tot_var_l)
        sd_h = np.sqrt(tot_var_h)

        # Standardized coordinates for upper bounds (lower is better)
        z_l = (tau_load - mu_l) / sd_l
        z_h = (tau_hair - mu_h) / sd_h

        # Vectorized marginal probabilities
        prob_load = norm.cdf(z_l)
        prob_hair = norm.cdf(z_h)

        # Joint lower-orthant probability, exact for any correlation (vectorized quadrature)
        corr = np.clip(tot_cov / (sd_l * sd_h), -0.99, 0.99)
        p_joint = bivariate_normal_cdf(z_l, z_h, corr)

        if is_scalar:
            return float(prob_load[0]), float(prob_hair[0]), float(p_joint[0])
        return prob_load, prob_hair, p_joint

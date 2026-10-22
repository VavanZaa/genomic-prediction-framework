"""Genetic architecture models: Additive QTL, dominance, epistasis, and pleiotropy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TraitDefinition:
    """Quantitative genetics parameterization of a single trait.

    Decomposes total genetic value: g = mu + A + D + I
    where:
      A: Additive breeding value (sum a_j * (G_j - 1))
      D: Dominance value (sum d_j * [G_j == 1])
      I: Epistatic interaction (sum aa_jk * (G_j - 1)*(G_k - 1))
    """
    name: str
    num_qtls: int
    mean: float = 100.0
    additive_effects: np.ndarray = field(default_factory=lambda: np.array([]))
    dominance_effects: np.ndarray = field(default_factory=lambda: np.array([]))
    epistatic_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    residual_variance: float = 10.0
    target_h2: float = 0.40
    is_ordinal: bool = False
    ordinal_min: int = 1
    ordinal_max: int = 12
    ordinal_cutpoints: Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.additive_effects) == 0:
            self.additive_effects = np.zeros(self.num_qtls, dtype=np.float64)
        if len(self.dominance_effects) == 0:
            self.dominance_effects = np.zeros(self.num_qtls, dtype=np.float64)

    @classmethod
    def create_calibrated(
        cls,
        name: str,
        num_qtls: int,
        mean: float = 100.0,
        target_genetic_var: float = 20.0,
        target_h2: float = 0.40,
        dominance_ratio: float = 0.20,  # Var(D) / Var(A)
        directional_dominance: float = 0.50,  # Fraction of d_j that are positive
        num_epistatic_pairs: int = 5,
        positive_effects: bool = False,
        is_ordinal: bool = False,
        ordinal_min: int = 1,
        ordinal_max: int = 12,
        ordinal_cutpoints: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> TraitDefinition:
        """Create a trait with calibrated additive, dominance, and epistasis variances."""
        if rng is None:
            rng = np.random.default_rng(42)

        # 1. Sample additive effects: Gamma / Laplace-like distribution
        if positive_effects:
            raw_a = np.abs(rng.normal(loc=0.0, scale=1.0, size=num_qtls))
        else:
            raw_a = rng.normal(loc=0.0, scale=1.0, size=num_qtls)
        # Normalize so the additive variance at p = 0.5 equals target_genetic_var:
        # Var(A) = sum_j a_j^2 * 2p(1-p) = 0.5 * sum_j a_j^2. Scale by the RMS rather than the SD so
        # this still holds when every effect has the same sign (positive_effects=True).
        rms = float(np.sqrt(np.mean(raw_a ** 2))) or 1.0
        raw_a = raw_a / rms * np.sqrt(target_genetic_var / (num_qtls * 0.5))

        # 2. Sample dominance effects: directional dominance creates heterosis
        d_signs = np.where(rng.random(num_qtls) < directional_dominance, 1.0, -1.0)
        d_mags = np.abs(raw_a) * np.sqrt(dominance_ratio) * rng.uniform(0.5, 1.5, size=num_qtls)
        raw_d = d_signs * d_mags

        # 3. Sample epistatic pairs
        epistatic_pairs = []
        num_epistatic_pairs = min(num_epistatic_pairs, num_qtls * (num_qtls - 1) // 2)
        if num_epistatic_pairs > 0 and num_qtls >= 2:
            pairs = set()
            while len(pairs) < num_epistatic_pairs:
                j = rng.integers(0, num_qtls)
                k = rng.integers(0, num_qtls)
                if j != k:
                    pair = tuple(sorted([j, k]))
                    pairs.add(pair)
            for j, k in pairs:
                coeff = float(rng.normal(loc=0.0, scale=0.5 * np.mean(np.abs(raw_a))))
                epistatic_pairs.append((j, k, coeff))

        # Provisional residual variance from the nominal variances:
        # h^2 = Var(A) / (Var(G) + Var(E)) => Var(E) = Var(A)/h^2 - Var(G).
        # The realized Var(A) depends on allele frequencies, so this only holds approximately;
        # TraitSimulator.calibrate_residuals sets the exact value for a stated reference population.
        approx_var_g = target_genetic_var * (1.0 + dominance_ratio)
        res_var = max(1.0, (target_genetic_var / target_h2) - approx_var_g)

        return cls(
            name=name,
            num_qtls=num_qtls,
            mean=mean,
            additive_effects=raw_a,
            dominance_effects=raw_d,
            epistatic_pairs=epistatic_pairs,
            residual_variance=res_var,
            target_h2=target_h2,
            is_ordinal=is_ordinal,
            ordinal_min=ordinal_min,
            ordinal_max=ordinal_max,
            ordinal_cutpoints=ordinal_cutpoints
        )

    def evaluate_components(self, qtl_genotypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute (total_g, A, D, I) for a matrix of QTL genotypes (N, Q)."""
        # qtl_genotypes has values in {0, 1, 2}
        N, Q = qtl_genotypes.shape
        if Q != self.num_qtls:
            raise ValueError(f"Genotype matrix has {Q} QTLs, expected {self.num_qtls}")

        # Additive component: a * (G - 1)
        centered = qtl_genotypes - 1.0
        A = np.dot(centered, self.additive_effects)

        # Dominance component: d * [G == 1] (heterozygotes)
        is_het = (qtl_genotypes == 1).astype(np.float64)
        D = np.dot(is_het, self.dominance_effects)

        # Epistatic component: aa_jk * (G_j - 1) * (G_k - 1)
        if len(self.epistatic_pairs) > 0:
            I = np.zeros(N, dtype=np.float64)
            for j, k, coeff in self.epistatic_pairs:
                I += coeff * (centered[:, j] * centered[:, k])
        else:
            I = np.zeros(N, dtype=np.float64)

        total_g = self.mean + A + D + I
        return total_g, A, D, I


@dataclass
class MultiTraitArchitecture:
    """Coordinates multiple traits with correlated genetic and environmental effects."""
    traits: List[TraitDefinition]
    environmental_correlation: Optional[np.ndarray] = None

    def __post_init__(self):
        self.trait_map = {t.name: t for t in self.traits}
        n_t = len(self.traits)
        if self.environmental_correlation is None:
            self.environmental_correlation = np.eye(n_t)

    def evaluate(self, qtl_genotypes: np.ndarray, rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
        """Simulate true genetic values and realized phenotypes with correlated noise."""
        N = qtl_genotypes.shape[0]
        n_t = len(self.traits)

        # Compute genetic values for each trait
        results = {}
        for t in self.traits:
            g, a, d, i = t.evaluate_components(qtl_genotypes)
            results[t.name] = {
                "phenotype": np.zeros(N),
                "total_g": g,
                "additive_a": a,
                "dominance_d": d,
                "epistasis_i": i,
                "noise_e": np.zeros(N)
            }

        # Generate correlated environmental residuals
        std_devs = np.array([np.sqrt(t.residual_variance) for t in self.traits])
        cov_matrix = np.outer(std_devs, std_devs) * self.environmental_correlation
        noise = rng.multivariate_normal(mean=np.zeros(n_t), cov=cov_matrix, size=N)

        for idx, t in enumerate(self.traits):
            results[t.name]["noise_e"] = noise[:, idx]
            latent_z = results[t.name]["total_g"] + noise[:, idx]
            results[t.name]["latent_z"] = latent_z
            if t.is_ordinal:
                if t.ordinal_cutpoints is not None:
                    cuts = np.asarray(t.ordinal_cutpoints)
                    bms = t.ordinal_min + np.sum(latent_z[:, None] > cuts[None, :], axis=1)
                    results[t.name]["phenotype"] = np.clip(bms, t.ordinal_min, t.ordinal_max).astype(np.float64)
                else:
                    results[t.name]["phenotype"] = np.clip(np.round(latent_z), t.ordinal_min, t.ordinal_max).astype(np.float64)
            else:
                results[t.name]["phenotype"] = latent_z

        return results

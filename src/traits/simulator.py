"""Trait simulation orchestration, breed divergence configurations, and phenotypic data generation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import Population, SimulatedIndividual
from .architecture import MultiTraitArchitecture, TraitDefinition


class TraitSimulator:
    """Simulates multi-trait phenotypes from individual genotypes using defined genetic architectures."""

    def __init__(self, architecture: MultiTraitArchitecture):
        self.architecture = architecture
        # trait name -> record of the reference population used to set its residual variance
        self.calibration: Dict[str, Dict] = {}

    def variance_components(
        self,
        individuals: List[SimulatedIndividual],
        genome_spec: GenomeSpec
    ) -> pd.DataFrame:
        """Realized genetic architecture of a population, per trait.

        Reports the variance of each simulated genetic component (A, D, I), the total genetic
        variance, the configured residual variance and the implied h^2 = Var(A)/Var(P) and
        H^2 = Var(G)/Var(P). For ordinal traits these refer to the latent liability scale.
        """
        qtl_matrix = np.vstack([ind.get_qtl_genotypes(genome_spec) for ind in individuals])
        rows = []
        for t in self.architecture.traits:
            g, a, d, i = t.evaluate_components(qtl_matrix)
            var_g = float(np.var(g))
            var_a = float(np.var(a))
            var_p = var_g + t.residual_variance
            rows.append({
                "trait": t.name,
                "n": len(individuals),
                "mean_g": float(np.mean(g)),
                "var_A": var_a,
                "var_D": float(np.var(d)),
                "var_I": float(np.var(i)),
                "var_G": var_g,
                "var_E": float(t.residual_variance),
                "h2_realized": var_a / var_p if var_p > 0 else float("nan"),
                "H2_realized": var_g / var_p if var_p > 0 else float("nan"),
                "h2_target": t.target_h2,
            })
        return pd.DataFrame(rows)

    def calibrate_residuals(
        self,
        reference_individuals: List[SimulatedIndividual],
        genome_spec: GenomeSpec,
        reference_label: str = "reference",
        min_residual_variance: float = 1e-6,
        group_by_breed: bool = False
    ) -> pd.DataFrame:
        """Set each trait's residual variance so h^2 equals target_h2 in the reference population.

        Var(E) = Var(A)/h^2 - Var(G). If Var(A)/Var(G) < target_h2 the target is unreachable
        (too much non-additive variance); Var(E) is then floored and the achieved h^2 recorded.

        group_by_breed=True averages the variance components computed within each breed (dominant
        pedigree breed) of the reference animals, so h^2 is the within-breed heritability, the scale
        on which breed heritabilities are normally reported. Pooling breeds instead counts the
        between-breed difference as additive variance and makes within-breed h^2 far lower.
        """
        if group_by_breed:
            groups: Dict[str, List[SimulatedIndividual]] = {}
            for ind in reference_individuals:
                comp = ind.pedigree_composition.composition
                groups.setdefault(max(sorted(comp), key=comp.get), []).append(ind)
            comps = (
                pd.concat([self.variance_components(g, genome_spec) for g in groups.values()])
                .groupby("trait", sort=False)[["var_A", "var_G"]].mean().reset_index()
            )
            comps["n"] = len(reference_individuals)
        else:
            comps = self.variance_components(reference_individuals, genome_spec)
        for row in comps.itertuples():
            t = self.architecture.trait_map[row.trait]
            desired = row.var_A / t.target_h2 - row.var_G
            feasible = desired > min_residual_variance
            t.residual_variance = float(desired if feasible else min_residual_variance)
            var_p = row.var_G + t.residual_variance
            self.calibration[t.name] = {
                "reference": reference_label,
                "n_reference": int(row.n),
                "var_A": float(row.var_A),
                "var_G": float(row.var_G),
                "var_E": t.residual_variance,
                "h2_target": t.target_h2,
                "h2_achieved": float(row.var_A / var_p),
                "feasible": bool(feasible),
                "within_breed": bool(group_by_breed),
            }
        return pd.DataFrame([{"trait": k, **v} for k, v in self.calibration.items()])

    @classmethod
    def create_brangus_architecture(
        cls,
        genome_spec: GenomeSpec,
        seed: int = 101
    ) -> Tuple[TraitSimulator, Dict[str, Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]]:
        """Create genetic architecture for Case 1 (Brangus):

        Traits:
        1. 'Production' (Growth/carcass yield): Angus has high allele frequencies, Brahman lower.
        2. 'EnvironmentalResponse' (Heat & parasite tolerance): Brahman has high allele frequencies, Angus low.
        Both traits have positive dominance (heterosis in crosses).
        """
        rng = np.random.default_rng(seed)
        num_qtls = genome_spec.total_qtls

        # Trait 1: Production (mean 100, h2 = 0.35)
        prod = TraitDefinition.create_calibrated(
            name="Production",
            num_qtls=num_qtls,
            mean=100.0,
            target_genetic_var=25.0,
            target_h2=0.35,
            dominance_ratio=0.25,
            directional_dominance=0.80,  # mostly positive dominance -> heterosis
            num_epistatic_pairs=10,
            positive_effects=True,
            rng=rng
        )

        # Trait 2: EnvironmentalResponse (mean 50, h2 = 0.40)
        env_resp = TraitDefinition.create_calibrated(
            name="EnvironmentalResponse",
            num_qtls=num_qtls,
            mean=50.0,
            target_genetic_var=20.0,
            target_h2=0.40,
            dominance_ratio=0.20,
            directional_dominance=0.75,
            num_epistatic_pairs=8,
            positive_effects=True,
            rng=rng
        )

        # Modular architecture with minor pleiotropy:
        # Production primarily driven by first half of QTLs; EnvironmentalResponse by second half
        half_qtl = num_qtls // 2
        prod.additive_effects[half_qtl:] *= 0.1
        prod.dominance_effects[half_qtl:] *= 0.1
        env_resp.additive_effects[:half_qtl] *= 0.1
        env_resp.dominance_effects[:half_qtl] *= 0.1

        # Slight negative environmental correlation (common in hot climates: high productivity animals stress more)
        env_corr = np.array([
            [1.0, -0.15],
            [-0.15, 1.0]
        ])

        multi_arch = MultiTraitArchitecture(traits=[prod, env_resp], environmental_correlation=env_corr)

        # Create breed-divergent allele frequency maps
        # Angus: favorable at Production QTLs (first half), low at EnvironmentalResponse (second half)
        # Brahman: opposite
        founder_freqs: Dict[str, Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]] = {
            "Angus": ({}, {}),
            "Brahman": ({}, {})
        }

        qtl_offset = 0
        half_qtl = num_qtls // 2
        for c in genome_spec.chromosomes:
            n_m = c.num_markers
            n_q = c.num_qtls

            # Markers: drift frequencies around 0.5
            angus_m = np.clip(rng.beta(2, 2, size=n_m), 0.05, 0.95)
            brahman_m = np.clip(rng.beta(2, 2, size=n_m), 0.05, 0.95)

            # QTLs: divergent
            angus_q = np.empty(n_q)
            brahman_q = np.empty(n_q)
            for idx in range(n_q):
                global_idx = qtl_offset + idx
                if global_idx < half_qtl:
                    # Production QTL: Angus high (0.80), Brahman lower (0.25)
                    angus_q[idx] = rng.uniform(0.70, 0.90)
                    brahman_q[idx] = rng.uniform(0.15, 0.35)
                else:
                    # Environmental response QTL: Brahman high (0.85), Angus lower (0.20)
                    angus_q[idx] = rng.uniform(0.10, 0.30)
                    brahman_q[idx] = rng.uniform(0.75, 0.95)

            founder_freqs["Angus"][0][c.chrom_id] = angus_m
            founder_freqs["Angus"][1][c.chrom_id] = angus_q

            founder_freqs["Brahman"][0][c.chrom_id] = brahman_m
            founder_freqs["Brahman"][1][c.chrom_id] = brahman_q

            qtl_offset += n_q

        return cls(multi_arch), founder_freqs

    @classmethod
    def create_wagyu_angus_architecture(
        cls,
        genome_spec: GenomeSpec,
        seed: int = 202
    ) -> Tuple[TraitSimulator, Dict[str, Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]]:
        """Create genetic architecture for Case 2 (Wagyu x Angus):

        Traits:
        1. 'Marbling' (Intramuscular fat / BMS): Wagyu has high favorable alleles, Angus moderate.
        2. 'Growth' (Carcass gain / weight): Angus has high favorable alleles, Wagyu moderate.
        """
        rng = np.random.default_rng(seed)
        num_qtls = genome_spec.total_qtls

        # Marbling: h2 = 0.45 (highly heritable trait in cattle)
        # Bounded ordinal scale 1-12 (Japanese Meat Grading Association / JMGA BMS)
        # Modeled as continuous latent liability z = mu + G + D + I + E, followed by
        # an ordinal threshold observation model: BMS = 1 + sum(z > c_k) in {1, ..., 12}
        bms_cutpoints = np.linspace(4.5, 17.5, 11)  # 11 thresholds partition liability into 12 BMS grades
        marbling = TraitDefinition.create_calibrated(
            name="Marbling",
            num_qtls=num_qtls,
            mean=6.0,  # latent liability baseline
            target_genetic_var=4.0,
            target_h2=0.45,
            dominance_ratio=0.15,
            directional_dominance=0.60,
            num_epistatic_pairs=8,
            positive_effects=True,
            is_ordinal=True,
            ordinal_min=1,
            ordinal_max=12,
            ordinal_cutpoints=bms_cutpoints,
            rng=rng
        )

        # Growth: h2 = 0.35
        growth = TraitDefinition.create_calibrated(
            name="Growth",
            num_qtls=num_qtls,
            mean=350.0,  # kg
            target_genetic_var=400.0,
            target_h2=0.35,
            dominance_ratio=0.20,
            directional_dominance=0.70,
            num_epistatic_pairs=10,
            positive_effects=True,
            rng=rng
        )

        # Modular architecture: Marbling driven by first half, Growth by second half
        half_qtl = num_qtls // 2
        marbling.additive_effects[half_qtl:] *= 0.1
        marbling.dominance_effects[half_qtl:] *= 0.1
        growth.additive_effects[:half_qtl] *= 0.1
        growth.dominance_effects[:half_qtl] *= 0.1

        env_corr = np.array([
            [1.0, 0.05],
            [0.05, 1.0]
        ])

        multi_arch = MultiTraitArchitecture(traits=[marbling, growth], environmental_correlation=env_corr)

        founder_freqs: Dict[str, Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]] = {
            "Wagyu": ({}, {}),
            "Angus": ({}, {})
        }

        qtl_offset = 0
        half_qtl = num_qtls // 2
        for c in genome_spec.chromosomes:
            n_m = c.num_markers
            n_q = c.num_qtls

            wagyu_m = np.clip(rng.beta(2, 2, size=n_m), 0.05, 0.95)
            angus_m = np.clip(rng.beta(2, 2, size=n_m), 0.05, 0.95)

            wagyu_q = np.empty(n_q)
            angus_q = np.empty(n_q)
            for idx in range(n_q):
                global_idx = qtl_offset + idx
                if global_idx < half_qtl:
                    # Marbling QTL: Wagyu very high (0.85-0.95), Angus moderate (0.35-0.55)
                    wagyu_q[idx] = rng.uniform(0.85, 0.95)
                    angus_q[idx] = rng.uniform(0.35, 0.55)
                else:
                    # Growth QTL: Angus high (0.80-0.92), Wagyu moderate (0.30-0.50)
                    wagyu_q[idx] = rng.uniform(0.30, 0.50)
                    angus_q[idx] = rng.uniform(0.80, 0.92)

            founder_freqs["Wagyu"][0][c.chrom_id] = wagyu_m
            founder_freqs["Wagyu"][1][c.chrom_id] = wagyu_q

            founder_freqs["Angus"][0][c.chrom_id] = angus_m
            founder_freqs["Angus"][1][c.chrom_id] = angus_q

            qtl_offset += n_q

        return cls(multi_arch), founder_freqs

    def simulate_records(
        self,
        individuals: List[SimulatedIndividual],
        genome_spec: GenomeSpec,
        rng: np.random.Generator
    ) -> pd.DataFrame:
        """Simulate trait phenotypes and return records DataFrame."""
        if not individuals:
            return pd.DataFrame()

        qtl_matrix = np.vstack([ind.get_qtl_genotypes(genome_spec) for ind in individuals])
        results = self.architecture.evaluate(qtl_matrix, rng)

        records = []
        for i, ind in enumerate(individuals):
            row = {
                "id": ind.id,
                "sex": ind.sex,
                "sire_id": ind.sire_id or "",
                "dam_id": ind.dam_id or "",
                "generation": ind.generation,
                "heterozygosity": ind.heterozygosity(genome_spec)
            }
            # Record breed fractions
            for b, frac in ind.pedigree_composition.composition.items():
                row[f"pedigree_{b}"] = float(frac)
            for b, frac in ind.all_realized_ancestry(genome_spec).items():
                row[f"genomic_{b}"] = frac

            # Record trait values
            for t_name, t_data in results.items():
                row[f"phenotype_{t_name}"] = t_data["phenotype"][i]
                if "latent_z" in t_data:
                    row[f"latent_z_{t_name}"] = t_data["latent_z"][i]
                row[f"true_g_{t_name}"] = t_data["total_g"][i]
                row[f"true_a_{t_name}"] = t_data["additive_a"][i]
                row[f"true_d_{t_name}"] = t_data["dominance_d"][i]
                row[f"true_i_{t_name}"] = t_data["epistasis_i"][i]
            records.append(row)

        return pd.DataFrame(records)

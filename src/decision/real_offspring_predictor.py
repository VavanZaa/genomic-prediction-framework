"""Real Physical Offspring Distribution Predictor for Layer 4A-v2.

Computes the pair-specific transition kernel P(G_offspring | G_s, G_d) from real cattle genomes:
1. Expected offspring genetic merit: E[g_offspring] = 0.5 * (g_sire + g_dam)
2. Pair-specific segregation variance: Var(g_offspring | s, d) based on parental inbreeding and heterozygosity
3. Parental genomic coancestry: Theta_{sd} = 0.5 * G_{s, d} (centered relative to reference allele frequencies)
4. Expected offspring inbreeding: E[F_offspring] = Theta_{sd} = 0.5 * G_{s, d}
5. Virtual offspring draws: K Gaussian draws per pair around the mid-parent GEBV (no physical meiosis of genotypes)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..real_data.genotype_preprocessing import validate_ids


@dataclass
class ParentalMerit:
    """Genetic merit profile for a candidate parent."""
    animal_id: str
    sex: str                          # 'M' for sire, 'F' for dam
    ancestry_brahman: float          # Genomic Brahman fraction q
    gebv_load: float                  # GEBV for LoadSlope (lower is better)
    gebv_hair: float                  # GEBV for AvgShortLength (lower is better)
    inbreeding_f: float = 0.0         # Genomic inbreeding coefficient F_i = G_ii - 1


@dataclass
class VirtualOffspringProfile:
    """Predicted distribution of genetic merit and diversity for a candidate mating (sire, dam)."""
    sire_id: str
    dam_id: str
    expected_ancestry: float          # E[q] = 0.5*(q_s + q_d)
    expected_gebv_load: float         # E[g_load] = 0.5*(g_s + g_d)
    expected_gebv_hair: float         # E[g_hair] = 0.5*(g_s + g_d)
    parental_coancestry: float        # Theta_{sd} = 0.5 * G_{s, d}
    expected_inbreeding: float        # E[F_offspring] = Theta_{sd} = 0.5 * G_{s, d}
    var_seg_load: float               # Pair-specific segregation variance for LoadSlope
    var_seg_hair: float               # Pair-specific segregation variance for AvgShortLength
    cov_seg: float = 0.0              # Segregation covariance


class RealOffspringPredictor:
    """Predicts offspring distributions from physical parental genomes and genomic matrices."""

    def __init__(
        self,
        grm_matrix: Optional[np.ndarray] = None,
        animal_ids: Optional[List[str]] = None,
        sigma_a_load: float = 2.50,
        sigma_a_hair: float = 1.10,
        rho_a: float = 0.05,
        assume_unrelated: bool = False,
    ):
        self.assume_unrelated = assume_unrelated
        self.grm = None if grm_matrix is None else np.asarray(grm_matrix, dtype=float)
        self.animal_ids = validate_ids([] if animal_ids is None else animal_ids)
        if self.grm is not None:
            if self.grm.shape != (len(self.animal_ids), len(self.animal_ids)):
                raise ValueError("GRM dimensions do not match unique animal IDs")
            if not np.isfinite(self.grm).all() or not np.allclose(self.grm, self.grm.T):
                raise ValueError("GRM must be finite and symmetric")
        self.id_to_idx = {aid: i for i, aid in enumerate(self.animal_ids)}
        self.sigma_a_load = sigma_a_load
        self.sigma_a_hair = sigma_a_hair
        self.rho_a = rho_a

    def get_genomic_relationship(self, sire_id: str, dam_id: str) -> float:
        """Returns VanRaden genomic relationship G_{s, d} between sire and dam relative to base reference."""
        if self.grm is None or sire_id not in self.id_to_idx or dam_id not in self.id_to_idx:
            if self.assume_unrelated:
                return 0.0
            raise ValueError(f"Unknown genomic relationship for {sire_id}, {dam_id}; supply a complete GRM")
        i = self.id_to_idx[sire_id]
        j = self.id_to_idx[dam_id]
        return float(self.grm[i, j])

    def get_parental_coancestry(self, sire_id: str, dam_id: str) -> float:
        """Returns parental genomic coancestry Theta_{sd} = 0.5 * G_{s, d}."""
        return 0.5 * self.get_genomic_relationship(sire_id, dam_id)

    def predict_mating(
        self,
        sire: ParentalMerit,
        dam: ParentalMerit,
    ) -> VirtualOffspringProfile:
        """Predicts the complete offspring distribution for a specific candidate mating."""
        # 1. Mid-parent expected genetic merit
        exp_gebv_l = 0.5 * (sire.gebv_load + dam.gebv_load)
        exp_gebv_h = 0.5 * (sire.gebv_hair + dam.gebv_hair)
        exp_q = 0.5 * (sire.ancestry_brahman + dam.ancestry_brahman)

        # 2. Parental genomic coancestry and expected offspring inbreeding:
        # VanRaden G is centered relative to base reference frequencies.
        # Negative G_{sd} denotes 'less related than reference population average'.
        # E[F_o] = Theta_{sd} = 0.5 * G_{sd}.
        g_sd = self.get_genomic_relationship(sire.animal_id, dam.animal_id)
        coancestry = 0.5 * g_sd
        exp_f = coancestry

        # 3. Pair-specific segregation variance:
        # Var(segregation | s, d) = 0.5 * (1 - 0.5*(F_s + F_d)) * sigma_A^2
        # Decreases as parents become more inbred/homozygous; increases with parental heterozygosity.
        f_mean_parents = 0.5 * (sire.inbreeding_f + dam.inbreeding_f)
        seg_scale = max(0.5 * (1.0 - f_mean_parents), 0.1)

        var_seg_l = seg_scale * (self.sigma_a_load ** 2)
        var_seg_h = seg_scale * (self.sigma_a_hair ** 2)
        cov_seg = seg_scale * (self.rho_a * self.sigma_a_load * self.sigma_a_hair)

        return VirtualOffspringProfile(
            sire_id=sire.animal_id,
            dam_id=dam.animal_id,
            expected_ancestry=exp_q,
            expected_gebv_load=exp_gebv_l,
            expected_gebv_hair=exp_gebv_h,
            parental_coancestry=coancestry,
            expected_inbreeding=exp_f,
            var_seg_load=var_seg_l,
            var_seg_hair=var_seg_h,
            cov_seg=cov_seg,
        )

    def sample_meiotic_offspring_gebvs(
        self,
        sire: ParentalMerit,
        dam: ParentalMerit,
        num_draws: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draws K virtual offspring GEBVs from the pair-specific Gaussian segregation distribution (no genotype meiosis).

        g_offspring^(k) = 0.5 * (g_s + g_d) + m^(k)
        where m^(k) ~ N(0, Sigma_seg(s, d))
        """
        if rng is None:
            rng = np.random.default_rng(42)

        profile = self.predict_mating(sire, dam)
        mean_l = profile.expected_gebv_load
        mean_h = profile.expected_gebv_hair

        cov_matrix = np.array([
            [profile.var_seg_load, profile.cov_seg],
            [profile.cov_seg, profile.var_seg_hair]
        ])

        seg_draws = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix, size=num_draws)
        draws_load = mean_l + seg_draws[:, 0]
        draws_hair = mean_h + seg_draws[:, 1]

        return draws_load, draws_hair

    def predict_all_pairs(
        self,
        sires: List[ParentalMerit],
        dams: List[ParentalMerit],
    ) -> Dict[Tuple[str, str], VirtualOffspringProfile]:
        """Precomputes predicted offspring distributions for all candidate (sire, dam) pairs."""
        profiles = {}
        for s in sires:
            for d in dams:
                profiles[(s.animal_id, d.animal_id)] = self.predict_mating(s, d)
        return profiles

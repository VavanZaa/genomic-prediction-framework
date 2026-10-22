"""Parent-to-offspring recovery on real genotyped trios (Layer 5B-T).

Data: Spanish autochthonous beef-cattle trios (BovineHD, 7 breeds, ~25 trios each), phased by
BEAGLE 3 in trio mode. The BEAGLE 3.3 manual (section "Phased trio data") stores each trio as four
haplotype columns: parent 1 transmitted, parent 1 untransmitted, parent 2 transmitted, parent 2
untransmitted, and states that offspring genotypes are reconstructed from the first column of each
parent. So the real calf is  T1 + T2  and its parents are  T1 + U1  and  T2 + U2 . Which parent is
the sire is not recorded; autosomal analyses do not need it.

Question: knowing only the two parents, how well does the engine's model of inheritance recover the
real calf, compared with knowing only the breed (allele frequencies) or the wrong parents?

Caveats built into the design:
- BEAGLE sets Mendelian-inconsistent genotypes to missing and imputes them, so Mendelian consistency
  holds by construction and is not reported as a finding.
- Parental phase was inferred with the child, so statistics about shared haplotype segments would be
  optimistic; only locus-wise and additive genome-wide statistics are used, whose expectations do not
  depend on phase.
- Physical positions are not in the phased files: crossovers are placed uniformly over marker order
  and genetic length is taken as 1 cM/Mb of the ARS-UCD1.2 chromosome length (assumed).
"""
from __future__ import annotations
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

BREEDS: Tuple[Tuple[str, str, int], ...] = (
    ("Asturiana de los Valles", "asturiana", 1),
    ("Avileña-Negra Ibérica", "avilena", 2),
    ("Bruna dels Pirineus", "bruna", 3),
    ("Morucha", "morucha", 4),
    ("Pirenaica", "pirenaica", 5),
    ("Retinta", "retinta", 6),
    ("Rubia Gallega", "rubia", 7),
)
# ARS-UCD1.2 autosome lengths (Mb); genetic length assumed at CM_PER_MB
CHROM_LENGTH_MB: Dict[int, float] = {
    1: 158.5, 2: 136.2, 3: 121.0, 4: 120.0, 5: 120.1, 6: 117.8, 7: 110.7, 8: 113.3, 9: 105.5, 10: 103.3,
    11: 107.0, 12: 87.2, 13: 83.5, 14: 82.4, 15: 85.0, 16: 81.0, 17: 73.2, 18: 65.8, 19: 63.4, 20: 72.0,
    21: 69.9, 22: 60.8, 23: 52.5, 24: 62.3, 25: 42.4, 26: 52.0, 27: 45.6, 28: 45.9, 29: 51.1,
}
CM_PER_MB = 1.0
GENOTYPE_ERROR = 1e-3


@dataclass
class TrioPanel:
    """Phased trio haplotypes of one breed: columns [T1, U1, T2, U2] per trio, rows = markers."""
    breed: str
    haplotypes: np.ndarray  # (markers, 4 * n_trios) int8; 1 = alphabetically later allele
    chrom: np.ndarray       # (markers,) chromosome of each row, in chromosome order

    @property
    def n_trios(self) -> int:
        return self.haplotypes.shape[1] // 4

    def trio(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = self.haplotypes
        return h[:, 4 * t], h[:, 4 * t + 1], h[:, 4 * t + 2], h[:, 4 * t + 3]

    def chrom_bounds(self) -> Dict[int, Tuple[int, int]]:
        bounds = {}
        for c in np.unique(self.chrom):
            idx = np.flatnonzero(self.chrom == c)
            bounds[int(c)] = (int(idx[0]), int(idx[-1]) + 1)
        return bounds

    def allele_freq_without(self, t: int) -> np.ndarray:
        """Breed allele frequency from all parental haplotypes except trio t's own (leave-one-out)."""
        total = self.haplotypes.sum(axis=1, dtype=np.int32)
        own = self.haplotypes[:, 4 * t: 4 * t + 4].sum(axis=1, dtype=np.int32)
        return (total - own) / (self.haplotypes.shape[1] - 4)


def load_trio_panel(tar_path: Path, breed: str, breed_id: int, chromosomes: Iterable[int] = range(1, 30), step: int = 4) -> TrioPanel:
    """Read every `step`-th marker of the BEAGLE trio files for one breed."""
    blocks, chrom_ids = [], []
    with tarfile.open(tar_path, "r:gz") as tar:
        for c in chromosomes:
            f = tar.extractfile(f"Beagle.{breed_id}.cromo{c}.pre_phase.bgl.phased")
            rows = [line.split()[2:] for i, line in enumerate(l for l in f if l.startswith(b"M ")) if i % step == 0]
            n_cols = len(rows[0])
            codes = np.frombuffer(b"".join(b"".join(r) for r in rows), dtype=np.uint8)
            if codes.size != len(rows) * n_cols:
                raise ValueError(f"{tar_path.name} chromosome {c}: alleles are not single characters.")
            codes = codes.reshape(len(rows), n_cols)
            blocks.append((codes != codes.min(axis=1, keepdims=True)).astype(np.int8))
            chrom_ids.append(np.full(len(rows), c, dtype=np.int16))
    return TrioPanel(breed=breed, haplotypes=np.vstack(blocks), chrom=np.concatenate(chrom_ids))


def mendelian_log_table(eps: float = GENOTYPE_ERROR) -> np.ndarray:
    """log P(child genotype | parent genotypes) indexed [g1, g2, child], with a small error rate."""
    transmit_one = np.array([0.0, 0.5, 1.0])
    table = np.zeros((3, 3, 3))
    for g1 in range(3):
        for g2 in range(3):
            a, b = transmit_one[g1], transmit_one[g2]
            p = np.array([(1 - a) * (1 - b), a * (1 - b) + b * (1 - a), a * b])
            table[g1, g2] = np.log((1 - eps) * p + eps / 3)
    return table


LOG_TABLE = mendelian_log_table()


def locus_recovery(g1: np.ndarray, g2: np.ndarray, child: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Locus-by-locus recovery of the child from its parents versus from breed allele frequencies."""
    hwe = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], axis=1)
    hwe = (1 - GENOTYPE_ERROR) * hwe + GENOTYPE_ERROR / 3
    idx = np.arange(len(child))
    par_probs = np.exp(LOG_TABLE[g1, g2])  # (M, 3)
    poly = (p > 0.02) & (p < 0.98)
    sd = np.sqrt(2 * p[poly] * (1 - p[poly]))
    z_pred = ((g1[poly] + g2[poly]) / 2.0 - 2 * p[poly]) / sd
    z_child = (child[poly] - 2 * p[poly]) / sd
    return {
        "loci": int(len(child)),
        "loglik_per_locus_parents": float(LOG_TABLE[g1, g2, child].mean()),
        "loglik_per_locus_breed": float(np.log(hwe[idx, child]).mean()),
        "call_accuracy_parents": float(np.mean(par_probs.argmax(axis=1) == child)),
        "call_accuracy_breed": float(np.mean(hwe.argmax(axis=1) == child)),
        "deviation_r_parents": float(np.corrcoef(z_pred, z_child)[0, 1]),
    }


def simulate_gametes(hap_a: np.ndarray, hap_b: np.ndarray, bounds: Dict[int, Tuple[int, int]],
                     lengths_morgan: Dict[int, float], k: int, rng: np.random.Generator,
                     extra_crossovers: float = 0.0) -> np.ndarray:
    """k gametes from a parent with haplotypes (hap_a, hap_b): Poisson crossovers per chromosome,
    placed uniformly over marker order, random starting strand. Returns (k, markers) int8."""
    out = np.empty((k, len(hap_a)), dtype=np.int8)
    for c, (s, e) in bounds.items():
        n = e - s
        n_xo = rng.poisson(lengths_morgan[c] + extra_crossovers, size=k)
        marks = np.zeros((k, n), dtype=np.int16)
        if n > 1 and n_xo.sum():
            np.add.at(marks, (np.repeat(np.arange(k), n_xo), rng.integers(1, n, size=n_xo.sum())), 1)
        strand = (rng.integers(0, 2, size=(k, 1)) + np.cumsum(marks, axis=1)) % 2
        out[:, s:e] = np.where(strand == 0, hap_a[s:e], hap_b[s:e])
    return out


def _relationship(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """VanRaden genomic relationship between rows of x and the genotype vector y."""
    return ((x - 2 * p) @ (y - 2 * p)) / (2 * np.sum(p * (1 - p)))


def _pit(sim: np.ndarray, real: float) -> float:
    return float((np.sum(sim < real) + 0.5 * np.sum(sim == real)) / len(sim))


def analyze_panel(panel: TrioPanel, k: int = 60, n_traits: int = 20, n_qtl: int = 1000,
                  extra_crossovers: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """One row per trio: locus-wise recovery, and calibration of simulated offspring against the real calf.

    Synthetic additive traits (n_traits architectures of n_qtl loci with N(0, 1) effects) are placed on
    the real genomes, so each calf has a known genetic value to recover. For each trio the calf is
    compared with (a) the distribution of k calves simulated from the parents by meiosis and (b) the
    constant approximation Var = 0.5 (1 - mean parental F) sigma_A^2 used by the Layer 4A predictor.
    """
    rng = np.random.default_rng(seed)
    bounds = panel.chrom_bounds()
    lengths = {c: CHROM_LENGTH_MB[c] * CM_PER_MB / 100.0 for c in bounds}
    m = panel.haplotypes.shape[0]
    p_all = panel.haplotypes.mean(axis=1)
    candidates = np.flatnonzero((p_all > 0.05) & (p_all < 0.95))
    qtl = [rng.choice(candidates, size=min(n_qtl, len(candidates)), replace=False) for _ in range(n_traits)]
    effects = [rng.standard_normal(len(q)) for q in qtl]

    def genetic_values(genotypes: np.ndarray) -> np.ndarray:
        g = np.atleast_2d(genotypes)
        return np.column_stack([g[:, q] @ a for q, a in zip(qtl, effects)])  # (rows, n_traits)

    parents = np.column_stack([panel.haplotypes[:, 4 * t] + panel.haplotypes[:, 4 * t + 1] for t in range(panel.n_trios)]
                              + [panel.haplotypes[:, 4 * t + 2] + panel.haplotypes[:, 4 * t + 3] for t in range(panel.n_trios)]).T
    sigma2_a = genetic_values(parents).var(axis=0, ddof=1)  # additive variance among the breed's parents

    rows = []
    for t in range(panel.n_trios):
        t1, u1, t2, u2 = panel.trio(t)
        g1, g2, child = t1 + u1, t2 + u2, t1 + t2
        p = panel.allele_freq_without(t)
        row = {"breed": panel.breed, "trio": t + 1, **locus_recovery(g1, g2, child, p)}

        # Wrong-parents control: the calf scored against another trio's parents from the same breed
        w = (t + 1) % panel.n_trios
        wt1, wu1, wt2, wu2 = panel.trio(w)
        row["loglik_per_locus_wrong_parents"] = float(LOG_TABLE[wt1 + wu1, wt2 + wu2, child].mean())

        sims = simulate_gametes(t1, u1, bounds, lengths, k, rng, extra_crossovers) + \
            simulate_gametes(t2, u2, bounds, lengths, k, rng, extra_crossovers)
        het_real, het_sim = float(np.mean(child == 1)), np.mean(sims == 1, axis=1)
        rel1_real, rel1_sim = float(_relationship(child[None, :], g1, p)[0]), _relationship(sims, g1, p)
        row.update({
            "het_real": het_real, "het_sim_mean": float(het_sim.mean()), "het_pit": _pit(het_sim, het_real),
            "rel_parent1_real": rel1_real, "rel_parent1_sim_mean": float(rel1_sim.mean()), "rel_parent1_pit": _pit(rel1_sim, rel1_real),
        })

        gv_child = genetic_values(child)[0]
        gv_mid = 0.5 * (genetic_values(g1)[0] + genetic_values(g2)[0])
        gv_sims = genetic_values(sims)  # (k, n_traits)
        lo, hi = np.quantile(gv_sims, 0.05, axis=0), np.quantile(gv_sims, 0.95, axis=0)
        z_sim = (gv_child - gv_sims.mean(axis=0)) / gv_sims.std(axis=0, ddof=1)
        f_parents = np.mean([_relationship(g[None, :], g, p)[0] - 1.0 for g in (g1, g2)])
        sd_const = np.sqrt(0.5 * (1.0 - f_parents) * sigma2_a)
        z_const = (gv_child - gv_mid) / sd_const
        row.update({
            "gv_cover90_sim": float(np.mean((gv_child >= lo) & (gv_child <= hi))),
            "gv_z_sim_mean": float(z_sim.mean()),
            "gv_z_sim_sd": float(z_sim.std()),
            "gv_cover90_const": float(np.mean(np.abs(z_const) <= 1.645)),
            "gv_z_const_sd": float(z_const.std()),
            "gv_ms_sd_ratio_sim_to_const": float(np.mean(gv_sims.std(axis=0, ddof=1) / sd_const)),
            "parent_F_mean": float(f_parents),
        })
        rows.append(row)
        # stash for across-trio correlations
        row["_gv_child"], row["_gv_mid"] = gv_child, gv_mid
    return pd.DataFrame(rows)


def midparent_accuracy(per_trio: pd.DataFrame) -> pd.DataFrame:
    """Across trios within each breed and trait: corr(mid-parent genetic value, real calf genetic value)."""
    out = []
    for breed, sub in per_trio.groupby("breed", sort=False):
        child = np.vstack(sub["_gv_child"].to_numpy())
        mid = np.vstack(sub["_gv_mid"].to_numpy())
        for j in range(child.shape[1]):
            out.append({"breed": breed, "trait": j, "r_midparent_child": float(np.corrcoef(mid[:, j], child[:, j])[0, 1])})
    return pd.DataFrame(out)


def segregation_table(panel: TrioPanel) -> pd.DataFrame:
    """Observed calf genotype proportions per parental mating type, pooled over trios, vs Mendel."""
    counts: Dict[Tuple[int, int], np.ndarray] = {}
    for t in range(panel.n_trios):
        t1, u1, t2, u2 = panel.trio(t)
        g_lo, g_hi = np.minimum(t1 + u1, t2 + u2), np.maximum(t1 + u1, t2 + u2)
        child = t1 + t2
        for a in range(3):
            for b in range(a, 3):
                mask = (g_lo == a) & (g_hi == b)
                counts.setdefault((a, b), np.zeros(3, dtype=np.int64))
                counts[(a, b)] += np.bincount(child[mask], minlength=3)
    rows = []
    for (a, b), c in counts.items():
        expected = np.exp(LOG_TABLE[a, b])
        expected = (expected - GENOTYPE_ERROR / 3) / (1 - GENOTYPE_ERROR)
        n = int(c.sum())
        rows.append({"breed": panel.breed, "mating_type": f"{a}x{b}", "n": n,
                     **{f"obs_{g}": (c[g] / n if n else np.nan) for g in range(3)},
                     **{f"exp_{g}": expected[g] for g in range(3)}})
    return pd.DataFrame(rows)

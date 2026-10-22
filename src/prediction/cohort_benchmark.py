"""Synthetic multi-cohort data and the Stage D prediction benchmark.

Generation 0: purebred Angus and Brahman founders. Generation 1: F1 families (Angus sires x Brahman
dams), each assigned to one of `n_families` families. Generation 2: backcrosses of F1 dams to Angus
sires, inheriting the dam's family. Two held-out evaluations follow the workplan:
- withheld_family: train on generations <= 1 of all but the last family, test on the last family;
- later_cohort: train on generations <= 1, test on generation 2.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from ..inheritance.chromosome import GenomeSpec
from ..inheritance.meiosis import MeiosisSimulator, Population
from ..traits.simulator import TraitSimulator
from .models import PredictionBenchmark


def build_multi_cohort_dataset(
    seed: int,
    genome: Optional[GenomeSpec] = None,
    n_founders_per_breed: int = 30,
    n_f1: int = 40,
    n_backcross: int = 60,
    n_families: int = 3
) -> Tuple[pd.DataFrame, np.ndarray, GenomeSpec]:
    if n_families < 2 or n_founders_per_breed < 2*n_families:
        raise ValueError("Need at least two male/female founder lines per family")
    rng = np.random.default_rng(seed)
    genome = genome or GenomeSpec.default_bovine(num_chromosomes=8, markers_per_chrom=25, qtls_per_chrom=5)
    simulator, freqs = TraitSimulator.create_brangus_architecture(genome, seed=seed)
    pop = Population(genome)

    angus = [pop.create_founder(f"A_F{i // 10}_{i}", sex="M" if i % 2 == 0 else "F", breed="Angus", rng=rng,
                                founder_allele_frequencies=freqs) for i in range(n_founders_per_breed)]
    brahman = [pop.create_founder(f"B_F{i // 10}_{i}", sex="M" if i % 2 == 0 else "F", breed="Brahman", rng=rng,
                                  founder_allele_frequencies=freqs) for i in range(n_founders_per_breed)]
    for group in (angus,brahman):
        for i,ind in enumerate(group): ind.meta["family"]=f"Fam_{(i//2)%n_families}"
    am={f:[x for x in angus if x.sex=="M" and x.meta["family"]==f"Fam_{f}"] for f in range(n_families)}
    bf={f:[x for x in brahman if x.sex=="F" and x.meta["family"]==f"Fam_{f}"] for f in range(n_families)}
    f1=[]
    for i in range(n_f1):
        family=i%n_families; k=i//n_families
        f1.append(MeiosisSimulator.mate(am[family][k%len(am[family])],bf[family][k%len(bf[family])],
                  f"F1_{i}", "F" if k%2 else "M",genome,rng,meta={"family":f"Fam_{family}"}))
    females=[x for x in f1 if x.sex=="F"]
    if not females: raise ValueError("Need female F1 offspring to form the later cohort")
    backcross=[]
    for i in range(n_backcross):
        dam=females[i%len(females)]; family=int(dam.meta["family"].split("_")[-1])
        candidates=am[family]
        sire=candidates[(i+1)%len(candidates)]
        backcross.append(MeiosisSimulator.mate(sire,dam,f"G2_{i}","M" if i%2==0 else "F",genome,rng,
                                               meta={"family":dam.meta["family"]}))

    all_inds = angus + brahman + f1 + backcross
    df = simulator.simulate_records(all_inds, genome, rng)
    markers = np.vstack([ind.get_marker_genotypes(genome) for ind in all_inds])
    df["family"] = [ind.meta.get("family", f"Founder_{ind.id.split('_')[0]}") for ind in all_inds]
    df["idx"] = range(len(all_inds))
    return df, markers, genome


def stage_d_splits(df: pd.DataFrame, n_families: int = 3) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    held_out = f"Fam_{n_families - 1}"
    train_fams = [f"Fam_{i}" for i in range(n_families - 1)]
    return {
        "withheld_family": (
            df[df["family"].isin(train_fams) & (df["generation"] <= 1)]["idx"].to_numpy(),
            df[(df["family"] == held_out) & (df["generation"] >= 1)]["idx"].to_numpy(),
        ),
        "later_cohort": (
            df[df["generation"] <= 1]["idx"].to_numpy(),
            df[df["generation"] == 2]["idx"].to_numpy(),
        ),
    }


def run_stage_d_benchmark(seeds: Iterable[int], trait: str = "Production") -> pd.DataFrame:
    """One row per seed x split x model with accuracy (Pearson), ranking (Spearman) and RMSE."""
    rows = []
    ancestry_cols = ["genomic_Angus", "genomic_Brahman"]
    for seed in seeds:
        df, markers, _ = build_multi_cohort_dataset(seed)
        for split, (tr, te) in stage_d_splits(df).items():
            results = PredictionBenchmark.run_comparison(
                train_ancestry=df.loc[tr, ancestry_cols].fillna(0.0).values,
                train_markers=markers[tr],
                train_y=df.loc[tr, f"phenotype_{trait}"].values,
                test_ancestry=df.loc[te, ancestry_cols].fillna(0.0).values,
                test_markers=markers[te],
                test_y=df.loc[te, f"phenotype_{trait}"].values,
                trait_name=trait,
            )
            for model, m in results.items():
                rows.append({"seed": seed, "split": split, "model": model, "n_train": len(tr), "n_test": len(te),
                             "pearson_r": m.pearson_r, "spearman_rho": m.spearman_rho, "rmse": m.rmse})
    return pd.DataFrame(rows)

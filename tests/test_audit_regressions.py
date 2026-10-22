"""Regression probes for the 11 September 2026 current-code audit (findings 1-15).

Each test reproduces the input the audit used to expose a defect and asserts the corrected
behaviour. The success-only gates in run_all_stages cannot detect these failures.
"""
from __future__ import annotations
import copy
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.decision.environment_scenario import EnvironmentEvaluator
from src.decision.multitrait_cross_optimizer import MultiTraitCrossOptimizer
from src.decision.real_mating_optimizer import PolicyType, RealMatingOptimizer
from src.decision.real_offspring_predictor import ParentalMerit, RealOffspringPredictor
from src.evaluation.mating_monte_carlo import MatingMonteCarloSimulator
from src.evaluation.wagyu_angus_case import POLICY_ORDER, WagyuAngusCaseStudy
from src.inheritance.chromosome import GenomeSpec
from src.prediction.cohort_benchmark import build_multi_cohort_dataset, stage_d_splits
from src.prediction.models import AdditiveGBLUP, RkhsPredictor
from src.prediction.uncertainty import out_of_fold_predictions
from src.real_data.environmental_model import EnvironmentalCovariateModel
from src.real_data.florida_boa_loader import FloridaBOALoader
from src.real_data.florida_loader import FloridaLoader
from src.real_data.genotype_preprocessing import GenotypeImputer
from src.real_data.phenotype_loader import PhenotypeDataset, PhenotypeLoader
from src.real_data.plink_parser import PlinkData
from src.real_data.reference_grm import load_reference_grm

SCENARIO = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]


# ---------------------------------------------------------------- Florida loader fixture (1, 5, 9, 11)

MARKERS = ["m1", "m2", "m3", "m4"]
BOA_SUFFIX = ["B", "A", "B", "A"]          # _A columns count the Angus allele and are reversed
BOA = {"001": [0, 2, 1, 2], "002": [2, 0, 1, 0], "003": [1, 1, 0, 2], "010": [2, 2, 2, 0]}
SNP = {"001": [2, 0, "NA", 1], "002": [0, 1, 2, 2], "003": [1, 2, 0, 0], "010": [0, 0, 1, 1],
       "999": [1, 1, 1, 1]}                 # 999 has SNP calls but no BOA record


def _write_raw(path, suffixes, rows):
    with gzip.open(path, "wt") as f:
        f.write(" ".join(["IID"] + [f"{m}_{s}" for m, s in zip(MARKERS, suffixes)]) + "\n")
        for aid, vals in rows:
            f.write(" ".join([aid] + [str(v) for v in vals]) + "\n")


def _snp(aid):
    return np.array([np.nan if v == "NA" else float(v) for v in SNP[aid]])


def _boa_brahman(aid):
    raw = np.array(BOA[aid], dtype=float)
    return np.where(np.array(BOA_SUFFIX) == "A", 2 - raw, raw)


@pytest.fixture
def florida_fixture(tmp_path):
    (tmp_path / "BOA.map").write_text("SNP CHR POS\n" + "".join(f"{m} 1 {1000 * (i + 1)}\n" for i, m in enumerate(MARKERS)))
    _write_raw(tmp_path / "BOA.raw.gz", BOA_SUFFIX, list(BOA.items()))
    # SNP rows in reversed order: pairing rows by position would swap every animal
    _write_raw(tmp_path / "SNP.raw.gz", ["A", "B", ".", "A"], list(reversed(list(SNP.items()))))
    pd.DataFrame({
        "UFID": ["001", "002", "003", "010", "999"], "Set": ["s1"] * 5, "Year": [2017, 2017, 2018, 2020, 2020],
        "Project": ["p"] * 5, "LoadSlope": [30.0, 31.0, 29.0, 28.0, 27.0], "AvgShortLength": [6.0] * 5,
        "AngusGBC": [0.7, 0.6, 0.8, 0.5, 0.5], "BrahmanGBC": [0.3, 0.4, 0.2, 0.5, 0.5],
    }).to_csv(tmp_path / "Thermo.Phenotypes.csv", index=False)
    return tmp_path


def test_dual_loader_joins_snp_and_boa_by_animal_id(florida_fixture):
    """Findings 5, 9, 11: rows joined by ID, leading zeros kept, missing calls left for training imputation."""
    d = FloridaBOALoader.load_dual_dataset(florida_fixture, marker_step=1, cache=False)
    assert d.animal_ids == ["001", "002", "003", "010"]
    for i, aid in enumerate(d.animal_ids):
        np.testing.assert_array_equal(d.genotypes_snp[i], _snp(aid))
        np.testing.assert_array_equal(d.alleles_boa[i], _boa_brahman(aid))
    assert list(d.phenotypes["UFID_str"]) == d.animal_ids


def test_florida_loader_returns_snp_dosages_not_boa(florida_fixture):
    """Finding 1: the legacy loader's `genotypes` are SNP dosages, which differ from BOA counts here."""
    data = FloridaLoader.load_dataset(florida_fixture, marker_step=1, min_maf=0.0)
    expected = np.array([_snp(a) for a in data.animal_ids])
    np.testing.assert_array_equal(data.genotypes, expected)
    boa = np.array([_boa_brahman(a) for a in data.animal_ids])
    assert not np.array_equal(np.nan_to_num(data.genotypes), boa)


def test_dual_loader_rejects_duplicate_animal_ids(florida_fixture):
    """Finding 5: a duplicated animal must not silently overwrite or truncate another record."""
    _write_raw(florida_fixture / "SNP.raw.gz", ["A", "B", ".", "A"], list(SNP.items()) + [("002", [0, 0, 0, 0])])
    with pytest.raises(ValueError, match="Duplicate"):
        FloridaBOALoader.load_dual_dataset(florida_fixture, marker_step=1, cache=False)


# ---------------------------------------------------------------- 2. split before adjustment

def test_training_targets_do_not_depend_on_test_outcomes():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age_days": rng.uniform(300, 500, n), "sex": rng.choice(["steer", "heifer"], n),
        "contemporary_group": rng.choice(["g1", "g2", "g3"], n), "feed_regimen": rng.choice(["grain", "pasture"], n),
        "growth": rng.normal(350, 30, n),
    })
    train = np.arange(n) < 40
    df.loc[~train, "contemporary_group"] = "g4_new"      # a level never seen in training

    def pipeline(frame):
        model = EnvironmentalCovariateModel()
        fit = model.fit_and_adjust(frame[train], "growth")
        adjusted = model.transform(frame, "growth")
        return fit, adjusted

    fit, adjusted = pipeline(df)
    np.testing.assert_allclose(adjusted[train], fit.adjusted_phenotypes)   # transform reproduces the fit
    assert np.isfinite(adjusted[~train]).all()
    shocked = df.copy()
    shocked.loc[~train, "growth"] += 500.0
    fit2, adjusted2 = pipeline(shocked)
    np.testing.assert_allclose(adjusted2[train], adjusted[train])
    assert np.median(adjusted2[train]) == np.median(adjusted[train])        # thresholds from training only
    with pytest.raises(RuntimeError):
        EnvironmentalCovariateModel().transform(df, "growth")


# ---------------------------------------------------------------- 3. Layer 4B constraints

def test_multitrait_allocation_never_violates_constraints():
    opt, w = MultiTraitCrossOptimizer(), np.ones(4)
    infeasible = opt.allocate_real_matings_multitrait(["S1"], ["D1", "D2"], np.array([0.3]), np.array([0.4, 0.4]),
                                                      np.array([[0.5, 0.5]]), w, max_kinship=0.125, sire_quota=1)
    assert len(infeasible) == 0
    assert infeasible.attrs["status"] == "infeasible" and infeasible.attrs["num_unassigned_dams"] == 2

    partial = opt.allocate_real_matings_multitrait(["S1"], ["D1", "D2"], np.array([0.3]), np.array([0.4, 0.4]),
                                                   np.zeros((1, 2)), w, max_kinship=0.125, sire_quota=1)
    assert len(partial) == 1 and partial.attrs["status"] == "partial" and partial.attrs["num_unassigned_dams"] == 1

    rng = np.random.default_rng(3)
    plan = opt.allocate_real_matings_multitrait([f"S{i}" for i in range(3)], [f"D{j}" for j in range(10)],
                                                rng.uniform(0, 1, 3), rng.uniform(0, 1, 10),
                                                rng.uniform(-0.1, 0.3, (3, 10)), w, max_kinship=0.125, sire_quota=3)
    assert (plan["Kinship_Gsd"] <= 0.125).all()
    assert plan["Sire_ID"].value_counts().max() <= 3
    assert plan["Dam_ID"].is_unique and len(plan) + plan.attrs["num_unassigned_dams"] == 10


# ---------------------------------------------------------------- 4A candidates (4, 6, 10)

def _candidates():
    sires = [ParentalMerit(f"S{i}", "M", 0.3 + 0.1 * i, gebv_load=-1.0 + i, gebv_hair=0.5 * i) for i in range(3)]
    dams = [ParentalMerit(f"D{j}", "F", 0.2 + 0.05 * j, gebv_load=0.3 * j - 1, gebv_hair=-0.2 * j) for j in range(6)]
    return sires, dams, [p.animal_id for p in sires + dams]


def _gebvs(sires, dams):
    return (np.array([s.gebv_load for s in sires]), np.array([s.gebv_hair for s in sires]),
            np.array([d.gebv_load for d in dams]), np.array([d.gebv_hair for d in dams]))


def test_monte_carlo_truth_is_common_across_policies():
    """Finding 4: an identical plan scores identically whatever its label or GEBV scale."""
    sires, dams, ids = _candidates()
    predictor = RealOffspringPredictor(grm_matrix=np.eye(len(ids)), animal_ids=ids)
    base = RealMatingOptimizer(predictor, sires, dams, max_sire_quota=2).optimize_policy(
        PolicyType.GLOBAL_ANCESTRY, *_gebvs(sires, dams))
    relabelled = copy.deepcopy(base)
    relabelled.policy_name = "DenseSNP"
    for a in relabelled.assignments:
        a.sire_gebv_load, a.dam_gebv_load = 10 * a.sire_gebv_load + 5, 10 * a.dam_gebv_load + 5
        a.sire_gebv_hair, a.dam_gebv_hair = 3 * a.sire_gebv_hair - 2, 3 * a.dam_gebv_hair - 2
    plans = {"GlobalAncestry": base, "DenseSNP": relabelled}
    reference = {p.animal_id: (p.gebv_load, p.gebv_hair) for p in sires + dams}
    sim = MatingMonteCarloSimulator(predictor, num_replicates=50, gebv_accuracy={"load": 0.3, "hair": 0.3},
                                    reference_parental_gebvs=reference)
    res = sim.run_replicates(plans, SCENARIO)
    np.testing.assert_array_equal(res["GlobalAncestry"].replicate_joint_attainments,
                                  res["DenseSNP"].replicate_joint_attainments)
    with pytest.raises(ValueError, match="common reference"):
        MatingMonteCarloSimulator(predictor, num_replicates=5).run_replicates(plans, SCENARIO)
    with pytest.raises(ValueError, match="Per-policy accuracy"):
        MatingMonteCarloSimulator(predictor, gebv_accuracy={"DenseSNP": {"load": 0.3, "hair": 0.3}})


def test_unknown_relationships_are_rejected_or_recorded():
    """Finding 6: a misspelled ID is an error, not relationship zero."""
    sires, dams, ids = _candidates()
    grm = np.eye(len(ids))
    grm[0, 3] = grm[3, 0] = 1.0                          # S0 and D0 closely related
    predictor = RealOffspringPredictor(grm_matrix=grm, animal_ids=ids)
    assert predictor.get_parental_coancestry("S0", "D0") == 0.5
    with pytest.raises(ValueError, match="Unknown"):
        predictor.get_parental_coancestry("S0", "DO")
    with pytest.raises(ValueError):
        RealOffspringPredictor(grm_matrix=np.eye(3), animal_ids=ids)
    with pytest.raises(ValueError):
        RealOffspringPredictor(grm_matrix=np.eye(2), animal_ids=["A", "A"])
    with pytest.raises(ValueError, match="Unknown"):
        RealMatingOptimizer(RealOffspringPredictor(), sires, dams, max_sire_quota=2)
    lenient = RealOffspringPredictor(assume_unrelated=True)
    summary = RealMatingOptimizer(lenient, sires, dams, max_sire_quota=2).optimize_policy(
        PolicyType.RANDOM, *_gebvs(sires, dams))
    assert summary.relationship_assumption == "assumed_unrelated_for_unknowns"


def test_no_feasible_mating_returns_infeasible_plan():
    """Finding 10: rejecting every pair yields a reported infeasible plan, not a crash."""
    sires, dams, ids = _candidates()
    predictor = RealOffspringPredictor(grm_matrix=np.ones((len(ids), len(ids))), animal_ids=ids)
    summary = RealMatingOptimizer(predictor, sires, dams, max_sire_quota=2).optimize_policy(
        PolicyType.DENSE_SNP, *_gebvs(sires, dams))
    assert summary.status == "infeasible" and summary.num_matings == 0
    assert summary.num_unassigned_dams == len(dams) and np.isnan(summary.max_parental_coancestry)
    with pytest.raises(ValueError, match="empty or infeasible"):
        MatingMonteCarloSimulator(predictor, num_replicates=5).run_replicates({"GlobalAncestry": summary}, SCENARIO)


# ---------------------------------------------------------------- 7. family holdout

@pytest.mark.parametrize("n_founders", [6, 20, 30])
def test_withheld_family_has_no_parent_or_full_sib_in_training(n_founders):
    df, _, _ = build_multi_cohort_dataset(seed=1, n_founders_per_breed=n_founders)
    tr, te = stage_d_splits(df)["withheld_family"]
    train, test = df.loc[tr], df.loc[te]
    assert len(test) > 0
    assert not (set(test["sire_id"]) | set(test["dam_id"])) & set(train["id"])
    train_pairs = set(zip(train["sire_id"], train["dam_id"])) - {("", "")}
    assert not set(zip(test["sire_id"], test["dam_id"])) & train_pairs


# ---------------------------------------------------------------- 8. missing calls

def test_missing_calls_are_imputed_not_used_as_allele_counts():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, (40, 15)).astype(float)
    y = X[:, :3].sum(axis=1) + rng.normal(0, 0.5, 40)
    X_minus1 = X.copy()
    X_minus1[rng.random(X.shape) < 0.1] = -1
    X_nan = np.where(X_minus1 == -1, np.nan, X_minus1)
    for model in (AdditiveGBLUP, RkhsPredictor):
        coded = model().fit(X_minus1[:30], y[:30]).predict(X_minus1[30:])
        missing = model().fit(X_nan[:30], y[:30]).predict(X_nan[30:])
        np.testing.assert_allclose(coded, missing)
    column = X_minus1[:30, 0]
    assert GenotypeImputer().fit(X_minus1[:30]).means_[0] == pytest.approx(column[column >= 0].mean())
    with pytest.raises(ValueError, match="Entirely missing"):
        GenotypeImputer().fit(np.full((5, 2), -1.0))


# ---------------------------------------------------------------- 9. ID validation

def test_phenotype_alignment_rejects_duplicates_and_keeps_leading_zeros(tmp_path):
    csv = tmp_path / "pheno.csv"
    pd.DataFrame({"id": ["007", "010"], "growth": [300.0, 320.0], "marbling": [5.0, 6.0]}).to_csv(csv, index=False)
    ds = PhenotypeLoader.load_from_csv(csv)
    assert list(ds.data["id"]) == ["007", "010"]
    bim = pd.DataFrame({"chrom": [1], "snp_id": ["m1"], "pos_m": [0.0], "pos_bp": [1], "a1": ["A"], "a2": ["B"]})
    fam = pd.DataFrame({"fid": ["f", "f"], "iid": ["010", "007"], "sire": ["0", "0"], "dam": ["0", "0"],
                        "sex": [1, 2], "pheno": [-9, -9]})
    pheno, geno = ds.align_with_plink(PlinkData(bim=bim, fam=fam, genotypes=np.array([[2], [0]])))
    assert list(pheno["id"]) == ["010", "007"] and geno[:, 0].tolist() == [2, 0]
    dup_pheno = PhenotypeDataset(pd.DataFrame({"id": ["007", "007", "010"], "growth": [1.0, 2.0, 3.0],
                                               "marbling": [1.0, 1.0, 1.0]}), "growth", "marbling", [])
    with pytest.raises(ValueError, match="Duplicate"):
        dup_pheno.align_with_plink(PlinkData(bim=bim, fam=fam, genotypes=np.array([[2], [0]])))
    with pytest.raises(ValueError, match="Duplicate"):
        ds.align_with_plink(PlinkData(bim=bim, fam=fam.assign(iid=["007", "007"]), genotypes=np.array([[2], [0]])))


# ---------------------------------------------------------------- 11. reference preprocessing

def test_reference_grm_rejects_obsolete_cache(tmp_path):
    np.savez(tmp_path / "old.npz", G_122k=np.eye(2), B_122k=np.eye(2), animal_ids=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="Obsolete"):
        load_reference_grm(tmp_path / "old.npz")


GRM_V3 = Path(__file__).resolve().parents[1] / "data/real_cattle/florida_multibreed/florida_grm_122k_v3.npz"


@pytest.mark.skipif(not GRM_V3.exists(), reason="v3 reference GRM not built")
def test_reference_grm_frequencies_come_from_historical_animals_only():
    data = load_reference_grm(GRM_V3)
    pheno = pd.read_csv(GRM_V3.parent / "Thermo.Phenotypes.csv", dtype={"UFID": str}).set_index("UFID")
    ref = [str(x) for x in data["reference_ids"]]
    assert (pheno.loc[ref, "Year"] <= 2018).all()
    assert set(ref) < {str(x) for x in data["animal_ids"]}


# ---------------------------------------------------------------- 14. predictive residuals

class _GroupSpy:
    """Predicts zero and fails if a validation group was part of its training fold (column 1 of X)."""
    def fit(self, X, y):
        self.seen = set(X[:, 1])
        return self

    def predict(self, X):
        assert not self.seen & set(X[:, 1]), "a validation group was in the training fold"
        return np.zeros(len(X))


def test_predictive_residuals_come_from_held_out_records():
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 5)
    X = np.column_stack([np.arange(40), groups]).astype(float)
    out_of_fold_predictions(X, rng.normal(size=40), _GroupSpy, groups=groups)
    # A memorising model has zero in-sample residuals but honest out-of-fold residuals
    Xr = rng.normal(size=(60, 3))
    y = Xr[:, 0] + rng.normal(0, 1, 60)
    knn = lambda: KNeighborsRegressor(n_neighbors=1)
    assert np.allclose(y - knn().fit(Xr, y).predict(Xr), 0)
    assert np.var(y - out_of_fold_predictions(Xr, y, knn)) > 0.5


# ---------------------------------------------------------------- Wagyu x Angus (12, 13, 15)

@pytest.fixture(scope="module")
def wagyu_small():
    genome = GenomeSpec.default_bovine(num_chromosomes=3, markers_per_chrom=20, qtls_per_chrom=4)
    return WagyuAngusCaseStudy(genome_spec=genome, seed=42).run(
        num_dams=12, generations=3, calves_per_dam=2, sire_quota=6, n_replicates=2,
        num_virtual_offspring=10, reference_size_per_breed=20)


def test_closed_composite_stays_half_wagyu_across_whole_cohort(wagyu_small):
    """Finding 12: every composite calf, not only the F1 x F1 subset, is exactly 1/2 Wagyu."""
    traj = wagyu_small.trajectory
    composites = ["Closed_Composite", "Composite_GenomicSelection", "Optimization_ML"]
    rows = traj[traj["policy"].isin(composites) & (traj["generation"] >= 2)]
    assert len(rows) and (rows["pedigree_wagyu_min"] == 0.5).all() and (rows["pedigree_wagyu_max"] == 0.5).all()
    rep = wagyu_small.replicate_table
    assert (rep.loc[rep["policy"].isin(composites), "purchased_females"] == 0).all()


def test_production_ledger_balances_each_cohort(wagyu_small):
    """Finding 13: retained and marketed counts refer to the same birth cohort."""
    traj = wagyu_small.trajectory
    assert (traj["births"] == traj["retained_as_parents"] + traj["marketed_calves"]).all()
    assert (traj.loc[traj["generation"] == traj["generation"].max(), "retained_as_parents"] == 0).all()
    rep = wagyu_small.replicate_table
    np.testing.assert_allclose(rep["production_calves_per_dam_slot"],
                               rep["final_births"] / (12 + rep["parental_line_dams"]))


def test_policy_intervals_use_t_critical_values(wagyu_small):
    """Finding 15: replicate and paired intervals use t, not 1.96."""
    rep = wagyu_small.replicate_table
    for key, p in wagyu_small.policies.items():
        att = rep.loc[rep["policy"] == key, "target_attainment"].to_numpy()
        half = stats.t.ppf(0.975, len(att) - 1) * att.std(ddof=1) / np.sqrt(len(att))
        assert p.target_attainment_ci95[0] == pytest.approx(max(0.0, att.mean() - half))
        assert p.target_attainment_ci95[1] == pytest.approx(min(1.0, att.mean() + half))
    rng = np.random.default_rng(0)
    rows = pd.DataFrame([{"replicate": r, "policy": k, "target_attainment": 0.3 + 0.05 * i + rng.normal(0, 0.02)}
                         for r in range(5) for i, k in enumerate(POLICY_ORDER)])
    comp = WagyuAngusCaseStudy._paired_comparisons(rows)
    row = comp[(comp["policy"] == "Optimization_ML") & (comp["baseline"] == "F1_Terminal")].iloc[0]
    wide = rows.pivot(index="replicate", columns="policy", values="target_attainment")
    d = (wide["Optimization_ML"] - wide["F1_Terminal"]).to_numpy()
    assert row["ci95_high"] - row["mean_difference"] == pytest.approx(stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5))
    assert row["interval_method"] == "paired t"

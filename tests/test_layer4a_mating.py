"""Unit and Integration Tests for Layer 4A-v2 Real Genomics- & BOA-Aware Mating Optimization.
"""
import numpy as np
import pytest

from src.decision.cross_fitted_calibrator import CrossFittedCalibrator
from src.decision.environment_scenario import EnvironmentEvaluator, EnvironmentScenario
from src.decision.real_mating_optimizer import PolicyType, RealMatingOptimizer
from src.decision.real_offspring_predictor import ParentalMerit, RealOffspringPredictor
from src.evaluation.mating_monte_carlo import MatingMonteCarloSimulator


class TestEnvironmentEvaluator:
    def test_decoupled_environmental_invariance(self):
        """Tests that intrinsic genetic merit is invariant while environmental scenario shifts."""
        scenario_a = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]
        scenario_b = EnvironmentEvaluator.CANONICAL_SCENARIOS["Elevated_Heat_Shifted"]

        gebv_l = -2.0  # Favorable thermotolerance GEBV
        gebv_h = -1.0  # Favorable short coat GEBV
        tau_l = 28.51
        tau_h = 5.93

        p_l_a, p_h_a, p_j_a = EnvironmentEvaluator.evaluate_attainment(
            gebv_l, gebv_h, tau_l, tau_h, scenario_a
        )
        p_l_b, p_h_b, p_j_b = EnvironmentEvaluator.evaluate_attainment(
            gebv_l, gebv_h, tau_l, tau_h, scenario_b
        )

        # LoadSlope has identical environmental mean (29.58) in both scenarios
        assert pytest.approx(p_l_a, rel=1e-3) == p_l_b

        # Hair length in Scenario B is shifted by +1.76 mm -> attainment probability drops
        assert p_h_a > p_h_b
        assert p_j_a > p_j_b
        assert p_h_b < 0.35  # Reflects shifted mean vs favorable GEBV

    def test_bivariate_bounds(self):
        scenario = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]
        p_l, p_h, p_joint = EnvironmentEvaluator.evaluate_attainment(
            0.0, 0.0, 28.51, 5.93, scenario
        )
        # Joint probability cannot exceed either marginal probability
        assert 0.0 <= p_joint <= p_l
        assert 0.0 <= p_joint <= p_h


class TestRealOffspringPredictor:
    def test_midparent_gebv_and_coancestry(self):
        sire = ParentalMerit(
            animal_id="Sire_1",
            sex="M",
            ancestry_brahman=0.50,
            gebv_load=-2.0,
            gebv_hair=-1.0,
            inbreeding_f=0.05,
        )
        dam = ParentalMerit(
            animal_id="Dam_1",
            sex="F",
            ancestry_brahman=0.25,
            gebv_load=0.0,
            gebv_hair=0.5,
            inbreeding_f=0.02,
        )

        # Mock GRM where G_{sire, dam} = 0.08
        animal_ids = ["Sire_1", "Dam_1"]
        grm = np.array([
            [1.05, 0.08],
            [0.08, 1.02]
        ])

        predictor = RealOffspringPredictor(grm_matrix=grm, animal_ids=animal_ids)
        profile = predictor.predict_mating(sire, dam)

        assert profile.expected_gebv_load == -1.0
        assert profile.expected_gebv_hair == -0.25
        assert profile.expected_ancestry == 0.375  # Exact Brangus composition!
        assert profile.parental_coancestry == pytest.approx(0.04)  # 0.5 * 0.08
        assert profile.expected_inbreeding == pytest.approx(0.04)
        assert profile.var_seg_load > 0.0
        assert profile.var_seg_hair > 0.0

    def test_virtual_meiotic_sampling(self):
        sire = ParentalMerit(animal_id="S1", sex="M", ancestry_brahman=0.5, gebv_load=-1.5, gebv_hair=-0.8)
        dam = ParentalMerit(animal_id="D1", sex="F", ancestry_brahman=0.3, gebv_load=-0.5, gebv_hair=-0.2)
        predictor = RealOffspringPredictor(assume_unrelated=True)

        draws_l, draws_h = predictor.sample_meiotic_offspring_gebvs(sire, dam, num_draws=100)
        assert len(draws_l) == 100
        assert len(draws_h) == 100
        # Expected mean should center around mid-parent
        assert np.abs(np.mean(draws_l) - (-1.0)) < 0.5
        assert np.abs(np.mean(draws_h) - (-0.5)) < 0.3


class TestCrossFittedCalibrator:
    def test_within_training_calibration(self):
        rng = np.random.default_rng(42)
        n_train = 200
        K_mock = rng.standard_normal((n_train, 20))
        K_train = K_mock @ K_mock.T / 20.0 + 0.1 * np.eye(n_train)
        y_train = rng.standard_normal(n_train) * 5.0 + 30.0
        tau = float(np.quantile(y_train, 0.35))

        calibrator = CrossFittedCalibrator(n_folds=3, random_state=42)
        res = calibrator.fit(K_train, y_train, tau=tau, trait_name="MockTrait", lam=5.0)

        assert np.isfinite(res.alpha_cv)
        assert np.isfinite(res.beta_cv)
        assert 0.0 <= res.oof_brier <= 1.0

        raw_prob = 0.35
        cal_prob = calibrator.calibrate(raw_prob)
        assert 0.0 < cal_prob < 1.0


class TestRealMatingOptimizer:
    def test_5_policy_resource_parity_and_constraints(self):
        """Ensures all 5 policies satisfy strict sire quotas and coancestry limits."""
        rng = np.random.default_rng(123)
        num_sires = 10
        num_dams = 50
        max_quota = 6

        sire_ids = [f"S_{i}" for i in range(num_sires)]
        dam_ids = [f"D_{j}" for j in range(num_dams)]

        sires = [
            ParentalMerit(
                animal_id=s_id,
                sex="M",
                ancestry_brahman=float(rng.uniform(0.1, 0.9)),
                gebv_load=float(rng.standard_normal()),
                gebv_hair=float(rng.standard_normal()),
                inbreeding_f=0.02,
            )
            for s_id in sire_ids
        ]

        dams = [
            ParentalMerit(
                animal_id=d_id,
                sex="F",
                ancestry_brahman=float(rng.uniform(0.2, 0.4)),
                gebv_load=float(rng.standard_normal()),
                gebv_hair=float(rng.standard_normal()),
                inbreeding_f=0.01,
            )
            for d_id in dam_ids
        ]

        # Generate candidate GRM
        all_ids = sire_ids + dam_ids
        n_all = len(all_ids)
        M_mock = rng.standard_normal((n_all, 30))
        grm = M_mock @ M_mock.T / 30.0

        predictor = RealOffspringPredictor(grm_matrix=grm, animal_ids=all_ids)
        optimizer = RealMatingOptimizer(
            predictor=predictor,
            sires=sires,
            dams=dams,
            max_sire_quota=max_quota,
            max_coancestry_threshold=0.25,
            ancestry_target=0.375,
        )

        policies = [
            PolicyType.RANDOM,
            PolicyType.GLOBAL_ANCESTRY,
            PolicyType.DENSE_SNP,
            PolicyType.DENSE_BOA,
            PolicyType.COMBINED_SNP_BOA,
        ]

        sire_gebvs_l = np.array([s.gebv_load for s in sires])
        sire_gebvs_h = np.array([s.gebv_hair for s in sires])
        dam_gebvs_l = np.array([d.gebv_load for d in dams])
        dam_gebvs_h = np.array([d.gebv_hair for d in dams])

        for pol in policies:
            summary = optimizer.optimize_policy(
                policy=pol,
                sire_gebvs_load=sire_gebvs_l,
                sire_gebvs_hair=sire_gebvs_h,
                dam_gebvs_load=dam_gebvs_l,
                dam_gebvs_hair=dam_gebvs_h,
            )

            # 1. Parity constraint: exactly num_dams allocations
            assert summary.num_matings == num_dams

            # 2. Strict sire quota constraint: no sire exceeds max_quota
            assert summary.max_sire_usage <= max_quota

            # 3. Coancestry ceiling: half-sib ceiling respected
            assert summary.max_parental_coancestry <= 0.25 + 1e-4

            # 4. Each dam allocated exactly once
            allocated_dams = [a.dam_id for a in summary.assignments]
            assert len(set(allocated_dams)) == num_dams

            # 5. Global Ancestry policy minimizes MAD to 0.375
            if pol == PolicyType.GLOBAL_ANCESTRY:
                mad_ancestry = summary.ancestry_target_mad
                assert mad_ancestry < 0.15


class TestMatingMonteCarloSimulator:
    def test_replicate_simulation_and_statistics(self):
        """Tests forward breeding program Monte Carlo simulation across multiple replicates."""
        rng = np.random.default_rng(42)
        num_sires = 5
        num_dams = 20
        sires = [ParentalMerit(f"S{i}", "M", 0.35, float(rng.standard_normal()), float(rng.standard_normal())) for i in range(num_sires)]
        dams = [ParentalMerit(f"D{j}", "F", 0.35, float(rng.standard_normal()), float(rng.standard_normal())) for j in range(num_dams)]

        all_ids = [s.animal_id for s in sires] + [d.animal_id for d in dams]
        grm = np.eye(len(all_ids))
        predictor = RealOffspringPredictor(grm_matrix=grm, animal_ids=all_ids)

        optimizer = RealMatingOptimizer(
            predictor=predictor, sires=sires, dams=dams, max_sire_quota=5, max_coancestry_threshold=0.1
        )

        s_l = np.array([s.gebv_load for s in sires])
        s_h = np.array([s.gebv_hair for s in sires])
        d_l = np.array([d.gebv_load for d in dams])
        d_h = np.array([d.gebv_hair for d in dams])

        p_rand = optimizer.optimize_policy(PolicyType.RANDOM, s_l, s_h, d_l, d_h)
        p_anc = optimizer.optimize_policy(PolicyType.GLOBAL_ANCESTRY, s_l, s_h, d_l, d_h)
        p_snp = optimizer.optimize_policy(PolicyType.DENSE_SNP, s_l, s_h, d_l, d_h)

        summaries = {"Random": p_rand, "GlobalAncestry": p_anc, "DenseSNP": p_snp}
        scenario = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]

        simulator = MatingMonteCarloSimulator(predictor=predictor, num_replicates=25, random_state=42)
        mc_results = simulator.run_replicates(summaries, scenario)

        assert len(mc_results) == 3
        snp_res = mc_results["DenseSNP"]
        assert snp_res.num_replicates == 25
        assert len(snp_res.replicate_attainments) == 25
        assert 0.0 <= snp_res.mean_attainment_pct <= 100.0
        assert 0.0 <= snp_res.prob_superior_to_ancestry <= 1.0
        assert snp_res.truth_model == "common_gebv_as_truth"
        # backward-compatible aliases used by the layer scripts
        assert snp_res.ci95_attainment_pct == snp_res.ci95_joint_pct
        assert snp_res.mean_delta_vs_ancestry_pct == snp_res.mean_delta_joint_vs_ancestry_pct


def _battery(rng, n_sires=8, n_dams=80, gebv_sd=3.0):
    sires = [ParentalMerit(f"S{i}", "M", 0.35, float(rng.normal(0, gebv_sd)), float(rng.normal(0, gebv_sd / 2))) for i in range(n_sires)]
    dams = [ParentalMerit(f"D{j}", "F", 0.35, float(rng.normal(0, gebv_sd)), float(rng.normal(0, gebv_sd / 2))) for j in range(n_dams)]
    ids = [p.animal_id for p in sires + dams]
    return sires, dams, RealOffspringPredictor(grm_matrix=np.eye(len(ids)), animal_ids=ids)


class TestMonteCarloTruthModel:
    def test_uninformative_gebvs_give_no_advantage_under_measured_accuracy(self):
        """Circularity check: selecting on GEBVs 'wins' only if the GEBVs are treated as the truth."""
        rng = np.random.default_rng(3)
        sires, dams, predictor = _battery(rng)
        opt = RealMatingOptimizer(predictor=predictor, sires=sires, dams=dams, max_sire_quota=20, max_coancestry_threshold=0.1)
        arrays = [np.array([p.gebv_load for p in sires]), np.array([p.gebv_hair for p in sires]),
                  np.array([p.gebv_load for p in dams]), np.array([p.gebv_hair for p in dams])]
        summaries = {
            "Random": opt.optimize_policy(PolicyType.RANDOM, *arrays),
            "GlobalAncestry": opt.optimize_policy(PolicyType.GLOBAL_ANCESTRY, *arrays),
            "DenseSNP": opt.optimize_policy(PolicyType.DENSE_SNP, *arrays),
        }
        scenario = EnvironmentEvaluator.CANONICAL_SCENARIOS["Subtropical_Baseline"]

        legacy = MatingMonteCarloSimulator(predictor, num_replicates=300, random_state=1).run_replicates(summaries, scenario)
        honest = MatingMonteCarloSimulator(predictor, num_replicates=300, random_state=1,
                                           gebv_accuracy={"load": 0.0, "hair": 0.0}).run_replicates(summaries, scenario)
        informed = MatingMonteCarloSimulator(predictor, num_replicates=300, random_state=1,
                                             gebv_accuracy={"load": 1.0, "hair": 1.0}).run_replicates(summaries, scenario)

        gain = lambda res: res["DenseSNP"].mean_joint_pct - res["Random"].mean_joint_pct
        assert gain(legacy) > 3.0                      # looks like a large win when GEBV = truth
        assert abs(gain(honest)) < 1.5                  # vanishes when GEBVs carry no information
        assert honest["DenseSNP"].truth_model == "common_reference_sensitivity"
        # With r = 1 the truth is E[g | gebv] = (sigma_a / sd(gebv)) * gebv: the advantage is real but
        # shrunk when GEBVs are more dispersed than sigma_a, and never larger than the circular estimate.
        assert 1.5 < gain(informed) <= gain(legacy) + 0.5

    def test_coancestry_ceiling_is_hard_and_capacity_is_checked(self):
        rng = np.random.default_rng(4)
        sires, dams, _ = _battery(rng, n_sires=3, n_dams=6)
        ids = [p.animal_id for p in sires + dams]
        grm = np.eye(len(ids))
        grm[:3, 3] = grm[3, :3] = 0.5  # dam D0 is related to every sire: coancestry 0.25 > 0.0625
        predictor = RealOffspringPredictor(grm_matrix=grm, animal_ids=ids)
        arrays = [np.array([p.gebv_load for p in sires]), np.array([p.gebv_hair for p in sires]),
                  np.array([p.gebv_load for p in dams]), np.array([p.gebv_hair for p in dams])]

        opt = RealMatingOptimizer(predictor=predictor, sires=sires, dams=dams, max_sire_quota=3, max_coancestry_threshold=0.0625)
        summary = opt.optimize_policy(PolicyType.DENSE_SNP, *arrays)
        assert summary.num_unassigned_dams == 1
        assert all(a.dam_id != "D0" for a in summary.assignments)
        assert summary.max_parental_coancestry <= 0.0625

        with pytest.raises(ValueError):
            RealMatingOptimizer(predictor=predictor, sires=sires, dams=dams, max_sire_quota=1).optimize_policy(PolicyType.RANDOM, *arrays)

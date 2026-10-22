"""Unit tests for Layer 2: Historical Breed Reconstruction — Brangus Convergence Experiment.
"""
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.brangus_convergence import BrangusHistoricalOptimizer, ConvergenceResult


def test_brangus_unguided_convergence():
    optimizer = BrangusHistoricalOptimizer()
    result = optimizer.run_convergence_test()

    assert isinstance(result, ConvergenceResult)
    # Under the default ASSUMED parameters the optimum lies near 5/8. This checks the model's
    # behaviour with those parameters; it is not evidence that 5/8 is optimal (see module docstring).
    assert 0.59 <= result.optimal_angus_fraction <= 0.65, (
        f"Optimal fraction {result.optimal_angus_fraction} did not converge near 5/8 (0.625)"
    )

    # 5/8 Brangus composition must strictly outperform 1/2 (F1/F2) and 3/4 Backcross
    assert result.attainment_at_five_eighths > result.attainment_at_half, (
        f"Expected 5/8 ({result.attainment_at_five_eighths:.3f}) > 1/2 ({result.attainment_at_half:.3f})"
    )
    assert result.attainment_at_five_eighths > result.attainment_at_three_fourths, (
        f"Expected 5/8 ({result.attainment_at_five_eighths:.3f}) > 3/4 ({result.attainment_at_three_fourths:.3f})"
    )

    assert result.is_converged_to_brangus is True


def test_multi_generation_selection_trajectory():
    optimizer = BrangusHistoricalOptimizer()
    df_traj = optimizer.simulate_selection_trajectory(num_generations=6, population_size=150)

    assert len(df_traj) == 6
    # Mean joint attainment should improve across generations of selection
    assert df_traj["mean_joint_attainment"].iloc[-1] > df_traj["mean_joint_attainment"].iloc[0]

    # Population mean Angus fraction should drift toward ~0.625 from random initial mixture
    final_mean_q = df_traj["mean_angus_fraction"].iloc[-1]
    assert 0.58 <= final_mean_q <= 0.66, f"Final generation mean ancestry was {final_mean_q}"

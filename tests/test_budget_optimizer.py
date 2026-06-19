import copy

import numpy as np
import pytest

from src.budget_optimizer import BudgetResult, optimize_allocation


def test_optimizer_conserves_budget_prefers_better_response_and_respects_bounds():
    response = {
        "search": lambda x: 10 * x / (x + 20),
        "social": lambda x: 4 * x / (x + 20),
    }
    bounds = {"search": (0.0, 100.0), "social": (0.0, 100.0)}
    current = {"search": 50.0, "social": 50.0}

    result = optimize_allocation(response, 100.0, bounds, current)

    assert isinstance(result, BudgetResult)
    assert result.success
    assert sum(result.optimal_allocation.values()) == pytest.approx(100.0, abs=1e-4)
    assert result.optimal_allocation["search"] > result.optimal_allocation["social"]
    assert bounds == {"search": (0.0, 100.0), "social": (0.0, 100.0)}
    assert current == {"search": 50.0, "social": 50.0}
    for channel, spend in result.optimal_allocation.items():
        low, high = bounds[channel]
        assert low <= spend <= high


def test_optimizer_rejects_infeasible_minimum_bounds():
    response = {"a": lambda x: x, "b": lambda x: x}

    result = optimize_allocation(
        response,
        50.0,
        {"a": (40.0, 100.0), "b": (40.0, 100.0)},
        {"a": 25.0, "b": 25.0},
    )

    assert not result.success
    assert "minimum" in result.message.lower()
    assert result.optimal_allocation == {}
    assert result.current_prediction == 50.0
    assert np.isnan(result.optimal_prediction)


def test_optimizer_rejects_infeasible_maximum_bounds():
    response = {"a": lambda x: x, "b": lambda x: x}

    result = optimize_allocation(
        response,
        120.0,
        {"a": (0.0, 50.0), "b": (0.0, 50.0)},
        {"a": 60.0, "b": 60.0},
    )

    assert not result.success
    assert "maximum" in result.message.lower()


def test_optimizer_handles_zero_current_budget_without_mutating_inputs():
    response = {
        "search": lambda x: 8 * x / (x + 10),
        "social": lambda x: 2 * x / (x + 10),
    }
    bounds = {"search": (0.0, 70.0), "social": (0.0, 70.0)}
    current = {"search": 0.0, "social": 0.0}
    bounds_snapshot = copy.deepcopy(bounds)
    current_snapshot = copy.deepcopy(current)

    result = optimize_allocation(response, 70.0, bounds, current)

    assert result.success
    assert sum(result.optimal_allocation.values()) == pytest.approx(70.0, abs=1e-4)
    assert result.optimal_allocation["search"] >= result.optimal_allocation["social"]
    assert bounds == bounds_snapshot
    assert current == current_snapshot


def test_optimizer_returns_clear_failure_for_nonpositive_budget():
    response = {"a": lambda x: x}

    result = optimize_allocation(response, 0.0, {"a": (0.0, 10.0)}, {"a": 1.0})

    assert not result.success
    assert "positive" in result.message.lower()
    assert result.optimal_allocation == {}


@pytest.mark.parametrize(
    "total_budget,bounds,current,message_snippet",
    [
        (np.nan, {"a": (0.0, 10.0)}, {"a": 1.0}, "total budget"),
        (10.0, {"a": (np.nan, 10.0)}, {"a": 1.0}, "finite"),
        (10.0, {"a": (-1.0, 10.0)}, {"a": 1.0}, "nonnegative"),
        (10.0, {"a": (10.0, 5.0)}, {"a": 1.0}, "lower"),
        (10.0, {"a": (0.0, 10.0)}, {"a": np.nan}, "current"),
    ],
)
def test_optimizer_rejects_nonfinite_inputs_with_clear_messages(total_budget, bounds, current, message_snippet):
    response = {"a": lambda x: x}

    result = optimize_allocation(response, total_budget, bounds, current)

    assert not result.success
    assert message_snippet in result.message.lower()
    assert np.isnan(result.optimal_prediction)


def test_optimizer_rejects_nonfinite_response_outputs():
    response = {"a": lambda x: np.nan if x > 0 else 0.0}

    result = optimize_allocation(response, 10.0, {"a": (0.0, 10.0)}, {"a": 1.0})

    assert not result.success
    assert "response" in result.message.lower()
    assert np.isnan(result.optimal_prediction)
    assert np.isnan(result.current_prediction)


def test_optimizer_rejects_solver_success_when_solution_is_infeasible(monkeypatch):
    class FakeResult:
        success = True
        message = "mocked success"
        x = np.array([12.0, -2.0], dtype=float)

    def fake_minimize(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr("src.budget_optimizer.minimize", fake_minimize)

    response = {"search": lambda x: x, "social": lambda x: x}
    bounds = {"search": (0.0, 10.0), "social": (0.0, 10.0)}

    result = optimize_allocation(response, 10.0, bounds, {"search": 5.0, "social": 5.0})

    assert not result.success
    assert "solver" in result.message.lower() or "feasible" in result.message.lower() or "bound" in result.message.lower()
    assert np.isnan(result.optimal_prediction)
    assert np.isfinite(result.current_prediction)

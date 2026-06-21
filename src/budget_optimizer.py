from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

BUDGET_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-8


@dataclass(frozen=True)
class BudgetResult:
    success: bool
    message: str
    optimal_allocation: dict[str, float]
    current_prediction: float
    optimal_prediction: float


class _NonFiniteResponseError(ValueError):
    pass


def _validate_response_outputs(
    channel_response: dict[str, Callable[[float], float]],
    allocation: dict[str, float],
) -> dict[str, float]:
    outputs: dict[str, float] = {}
    for name in channel_response:
        value = float(channel_response[name](float(allocation.get(name, 0.0))))
        if not np.isfinite(value):
            raise _NonFiniteResponseError(f"Non-finite response output for channel '{name}'.")
        outputs[name] = value
    return outputs


def _prediction(channel_response: dict[str, Callable[[float], float]], allocation: dict[str, float]) -> float:
    outputs = _validate_response_outputs(channel_response, allocation)
    return float(sum(outputs.values()))


def _current_prediction(
    channel_response: dict[str, Callable[[float], float]],
    current: dict[str, float],
) -> float:
    if any(not np.isfinite(float(value)) for value in current.values()):
        return float("nan")

    try:
        return _prediction(channel_response, current)
    except (ValueError, TypeError, OverflowError, _NonFiniteResponseError):
        return float("nan")


def _validate_bounds(
    channel_response: dict[str, Callable[[float], float]],
    bounds: dict[str, tuple[float, float]],
) -> tuple[bool, str]:
    response_channels = list(channel_response)
    missing_bounds = [name for name in response_channels if name not in bounds]
    extra_bounds = [name for name in bounds if name not in channel_response]
    if missing_bounds or extra_bounds:
        details: list[str] = []
        if missing_bounds:
            details.append(f"missing bounds for: {', '.join(missing_bounds)}")
        if extra_bounds:
            details.append(f"unexpected bounds for: {', '.join(extra_bounds)}")
        return False, "Channel names must match between responses and bounds (" + "; ".join(details) + ")."

    for name in response_channels:
        lower, upper = bounds[name]
        if not np.isfinite(lower) or not np.isfinite(upper):
            return False, f"Bounds for channel '{name}' must be finite."
        if lower < 0 or upper < 0:
            return False, f"Bounds for channel '{name}' must be nonnegative."
        if lower > upper:
            return False, f"Bounds for channel '{name}' must satisfy lower <= upper."

    return True, ""


def _validate_inputs(
    channel_response: dict[str, Callable[[float], float]],
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
    current: dict[str, float],
) -> tuple[bool, str]:
    if not np.isfinite(total_budget):
        return False, "Total budget must be finite."
    if total_budget <= 0:
        return False, "Total budget must be positive."

    response_channels = list(channel_response)
    if not response_channels:
        return False, "At least one channel response is required."

    bounds_valid, bounds_message = _validate_bounds(channel_response, bounds)
    if not bounds_valid:
        return False, bounds_message

    if any(not np.isfinite(float(value)) for value in current.values()):
        return False, "Current values must be finite."

    lower_total = sum(float(bounds[name][0]) for name in response_channels)
    upper_total = sum(float(bounds[name][1]) for name in response_channels)
    if lower_total > total_budget:
        return False, "Channel minimum bounds exceed the total budget."
    if upper_total < total_budget:
        return False, "Channel maximum bounds cannot absorb the total budget."

    return True, ""


def _stable_initial_allocation(
    channels: list[str],
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
    current: dict[str, float],
) -> np.ndarray:
    lower = np.array([float(bounds[name][0]) for name in channels], dtype=float)
    upper = np.array([float(bounds[name][1]) for name in channels], dtype=float)
    allocation = lower.copy()
    remainder = float(total_budget - allocation.sum())
    headroom = np.maximum(upper - allocation, 0.0)

    if remainder <= 0:
        return allocation

    current_values = np.array([float(current.get(name, 0.0)) for name in channels], dtype=float)
    if float(current_values.sum()) > 0:
        weights = np.clip(current_values, 0.0, None)
    else:
        weights = headroom.copy()

    if float(weights.sum()) <= 0:
        weights = np.ones(len(channels), dtype=float)

    weights = weights / weights.sum()
    allocation += np.minimum(headroom, remainder * weights)

    shortfall = float(total_budget - allocation.sum())
    if shortfall <= 1e-12:
        return allocation

    flexible = np.where(headroom > 0)[0]
    while shortfall > 1e-12 and flexible.size > 0:
        weights = headroom[flexible]
        if float(weights.sum()) <= 0:
            weights = np.ones(flexible.size, dtype=float)
        weights = weights / weights.sum()
        increments = np.minimum(headroom[flexible], shortfall * weights)
        allocation[flexible] += increments
        headroom[flexible] -= increments
        shortfall = float(total_budget - allocation.sum())
        flexible = np.where(headroom > 1e-12)[0]

    return allocation


def optimize_allocation(
    channel_response: dict[str, Callable[[float], float]],
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
    current: dict[str, float],
) -> BudgetResult:
    channels = list(channel_response)
    current_snapshot = dict(current)
    bounds_snapshot = dict(bounds)

    current_prediction = _current_prediction(channel_response, current_snapshot)
    is_valid, message = _validate_inputs(channel_response, total_budget, bounds_snapshot, current_snapshot)
    if not is_valid:
        return BudgetResult(False, message, {}, current_prediction, float("nan"))

    x0 = _stable_initial_allocation(channels, total_budget, bounds_snapshot, current_snapshot)

    def objective(x: np.ndarray) -> float:
        return -_prediction(channel_response, {name: float(value) for name, value in zip(channels, x, strict=True)})

    try:
        solved = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[bounds_snapshot[name] for name in channels],
            constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)},
            options={"maxiter": 1_000, "ftol": 1e-9},
        )
    except _NonFiniteResponseError as exc:
        return BudgetResult(False, str(exc), {}, current_prediction, float("nan"))
    except (ValueError, TypeError, OverflowError) as exc:
        return BudgetResult(False, str(exc), {}, current_prediction, float("nan"))

    if not solved.success:
        return BudgetResult(False, str(solved.message), {}, current_prediction, float("nan"))

    allocation = {name: float(value) for name, value in zip(channels, solved.x, strict=True)}
    if any(not np.isfinite(value) for value in allocation.values()):
        return BudgetResult(False, "Solver returned non-finite allocation values.", {}, current_prediction, float("nan"))

    budget_sum = float(sum(allocation.values()))
    if abs(budget_sum - total_budget) > BUDGET_TOLERANCE:
        return BudgetResult(
            False,
            f"Solver returned an off-budget allocation outside tolerance {BUDGET_TOLERANCE}.",
            {},
            current_prediction,
            float("nan"),
        )

    for name, value in allocation.items():
        lower, upper = bounds_snapshot[name]
        if value < lower - BOUND_TOLERANCE or value > upper + BOUND_TOLERANCE:
            return BudgetResult(
                False,
                f"Solver returned an allocation outside bounds tolerance {BOUND_TOLERANCE} for channel '{name}'.",
                {},
                current_prediction,
                float("nan"),
            )

    try:
        optimal_prediction = _prediction(channel_response, allocation)
    except (ValueError, TypeError, OverflowError, _NonFiniteResponseError) as exc:
        return BudgetResult(False, str(exc), {}, current_prediction, float("nan"))

    return BudgetResult(True, str(solved.message), allocation, current_prediction, optimal_prediction)

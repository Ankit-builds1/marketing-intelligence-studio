from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BudgetResult:
    success: bool
    message: str
    optimal_allocation: dict[str, float]
    current_prediction: float
    optimal_prediction: float


def _prediction(channel_response: dict[str, Callable[[float], float]], allocation: dict[str, float]) -> float:
    return float(
        sum(channel_response[name](float(allocation.get(name, 0.0))) for name in channel_response)
    )


def _validate_inputs(
    channel_response: dict[str, Callable[[float], float]],
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[bool, str]:
    if total_budget <= 0:
        return False, "Total budget must be positive."

    response_channels = list(channel_response)
    if not response_channels:
        return False, "At least one channel response is required."

    missing_bounds = [name for name in response_channels if name not in bounds]
    extra_bounds = [name for name in bounds if name not in channel_response]
    if missing_bounds or extra_bounds:
        details: list[str] = []
        if missing_bounds:
            details.append(f"missing bounds for: {', '.join(missing_bounds)}")
        if extra_bounds:
            details.append(f"unexpected bounds for: {', '.join(extra_bounds)}")
        return False, "Channel names must match between responses and bounds (" + "; ".join(details) + ")."

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

    current_prediction = _prediction(channel_response, current_snapshot)
    is_valid, message = _validate_inputs(channel_response, total_budget, bounds_snapshot)
    if not is_valid:
        return BudgetResult(False, message, {}, current_prediction, 0.0)

    x0 = _stable_initial_allocation(channels, total_budget, bounds_snapshot, current_snapshot)
    objective = lambda x: -sum(
        channel_response[name](float(value)) for name, value in zip(channels, x, strict=True)
    )

    solved = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[bounds_snapshot[name] for name in channels],
        constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)},
    )

    if not solved.success:
        return BudgetResult(False, str(solved.message), {}, current_prediction, 0.0)

    allocation = {name: float(value) for name, value in zip(channels, solved.x, strict=True)}
    optimal_prediction = _prediction(channel_response, allocation)
    return BudgetResult(True, str(solved.message), allocation, current_prediction, optimal_prediction)

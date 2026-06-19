# Task 4 Report: Nonlinear budget optimization

## Scope

- Added `src/budget_optimizer.py`
- Added `tests/test_budget_optimizer.py`

## RED evidence

Focused test run before implementation:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_budget_optimizer.py -v -p no:cacheprovider`

Result: collection failed with `ModuleNotFoundError: No module named 'src.budget_optimizer'`.

## GREEN evidence

After implementation:

- Focused suite: 5 passed
- Full suite: 23 passed

## Review-fix evidence

Focused regression run after tightening validation:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_budget_optimizer.py -v -p no:cacheprovider`

Result: 12 passed.

Full verification run:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest -p no:cacheprovider`

Result: 30 passed.

## Changes added in this pass

- Rejects nonfinite total budget, bounds endpoints, current values, and response outputs
- Validates bounds are finite, nonnegative, and ordered
- Rejects solver results that are finite-looking but violate bounds or budget tolerance
- Returns `optimal_prediction = nan` on any failure
- Returns `current_prediction = nan` when current inputs or response outputs are not evaluable
- Adds regression coverage for nonfinite inputs and solver-contract violations

## Behavior covered

- Feasibility validation for minimum and maximum bounds
- Budget conservation
- Bound-respecting optimal allocations
- Stable initialization when current spend sums to zero
- Clear failure messages for invalid inputs
- No mutation of caller-provided inputs

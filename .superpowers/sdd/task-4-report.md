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

## Behavior covered

- Feasibility validation for minimum and maximum bounds
- Budget conservation
- Bound-respecting optimal allocations
- Stable initialization when current spend sums to zero
- Clear failure messages for invalid inputs
- No mutation of caller-provided inputs


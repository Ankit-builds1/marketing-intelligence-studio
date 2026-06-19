# Task 3 Report: Time-aware MMM, uncertainty, and diagnostics

## Scope

- Implemented `src/mmm_model.py` with `MMMConfig`, `ModelMetrics`, `MMMResult`, `fit_mmm(...)`, and `predict_from_feature_row(...)`.
- Added focused edge-case coverage in `tests/test_mmm_model.py`.
- Left Task 1/2 files untouched.

## TDD evidence

### RED

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_mmm_model.py -v -p no:cacheprovider`

Result:

- `2 failed`
- Both failures were `ModuleNotFoundError: No module named 'src.mmm_model'`

### GREEN

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_mmm_model.py -v -p no:cacheprovider`

Result:

- `2 passed`

## Verification evidence

### Full unit suite

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider`

Result:

- `14 passed in 4.85s`

### Smoke test

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe src\smoke_test.py`

Result:

- Completed successfully
- Ended with `SUCCESS - all checks passed.`

## Implementation notes

- Holdout is time-ordered and reserves the final `max(8, round(n * holdout_fraction))` rows while preserving at least one training row.
- Media coefficients are fit with constrained non-negative ridge regression.
- Holdout metrics come from the training-only fit; final outputs come from an all-data refit.
- `result.fitted` reconciles exactly to `baseline_component + sum(channel_contributions)`.
- Channel summary contribution/ROI and response curves use zero-spend incremental outcome on the transformed scale, while row-level `channel_contributions` stay centered for exact prediction reconciliation.
- Bootstrap intervals are deterministic from the configured seed and use moving contiguous training-row blocks.
- Reliability and warnings reflect negative holdout R², zero-outcome MAPE invalidation, and unstable near-zero intervals.

## Self-review

- Kept all changes inside the three owned files.
- Added a strong-signal recovery test and a weak-signal warning/determinism test rather than broadening the suite.
- Used a tiny zero-crossing tolerance for interval stability so effectively null constrained channels still surface as unstable instead of being hidden by numerical noise.

## Concerns

- The legacy smoke test passes, but it is slow in this environment (about 250 seconds), so future verification should budget for that runtime.

## Task 3 review-fix verification

### RED

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests\test_mmm_model.py -v -p no:cacheprovider`

Result:

- `3 failed`
- Failures covered the three review findings:
  - insufficient-row rejection was missing,
  - moving-block bootstrap still wrapped at the series end,
  - response curves still ignored steady-state adstock.

### GREEN

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests\test_mmm_model.py -v -p no:cacheprovider`

Result:

- `5 passed`

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider`

Result:

- `17 passed in 1.90s`

Command:

`D:\CausalInference_MMM\venv\Scripts\python.exe src\smoke_test.py`

Result:

- Completed successfully
- Ended with `SUCCESS - all checks passed.`

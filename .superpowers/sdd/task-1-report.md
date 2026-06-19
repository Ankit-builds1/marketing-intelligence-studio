# Task 1 Report: Data mapping and validation

## Implementation

Implemented the data-preparation and validation layer in `src/data_validation.py` with the required public interface:

- `DataMapping`
- `ValidationIssue`
- `PreparedData`
- `prepare_and_validate(raw, mapping, min_rows=52)`

The implementation:

- maps arbitrary input columns to canonical output columns `date`, `outcome`, `media__<name>`, and `control__<name>`;
- validates duplicate mappings, missing columns, date parseability, numeric coercion/missingness, minimum row count, non-negative outcome/media values, constant columns, weekly cadence, and near-duplicate media channels;
- returns a `PreparedData` object with `can_train` derived from error-severity issues.

Also added the required test dependency pin to `requirements.txt`.

## RED evidence

Created `tests/test_data_validation.py` first, then ran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Initial expected failure after installing `pytest` into the required venv:

- `ModuleNotFoundError: No module named 'src.data_validation'`

That confirmed the test was exercising the missing production module rather than a typo or unrelated issue.

## GREEN evidence

After implementing `src/data_validation.py`, ran the scoped test file again:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Result: `3 passed`

## Full suite evidence

Ran the full available suite once:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests -v`

Result: `3 passed`

## Changed files

- `requirements.txt`
- `src/data_validation.py`
- `tests/test_data_validation.py`

## Self-review

- Confirmed the canonical column order matches the brief and the tests.
- Confirmed the negative media and too-few-rows cases surface errors, while near-duplicate media channels surface a warning.
- Confirmed the module behaves with the provided weekly sample data and preserves the intended output schema.
- Removed the transient `tests/__pycache__` directory after test execution so it would not pollute the commit.

## Concerns

None for the scoped task. The only environment issue encountered was that `pytest` was not initially installed in the target venv, so it was installed there to run the required verification commands.

## Follow-up resolution: weekly aggregation and cadence blocking

### RED evidence

After adding the regression tests for daily, monthly, and irregular cadence handling, I ran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Initial failure showed the previous implementation did not aggregate daily data to weekly periods or block non-convertible cadence:

- daily data stayed at 364 rows instead of 52 weekly rows;
- 357 daily rows remained trainable instead of failing the 52-row post-aggregation check;
- monthly/irregular cadence was not rejected with the required error.

### GREEN evidence

After updating `src/data_validation.py` to:

- aggregate sub-weekly data into Monday-anchored weekly rows,
- sum outcome/media and average controls,
- apply the minimum-row rule after weekly preparation,
- block monthly/irregular cadence with `irregular_cadence`,

I reran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Result: `6 passed`

### Full-suite evidence

I also reran the full available suite:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests -v`

Result: `6 passed`

## Follow-up resolution: repeated same-day rows are preserved through weekly aggregation

### RED evidence

After adding the regression test for multiple observations sharing the same date, I ran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

The new test failed because the implementation still lost same-day rows before weekly aggregation, which meant the Monday-anchored weekly result did not preserve all outcome and media values.

### GREEN evidence

After removing the pre-aggregation duplicate-date drop and routing both weekly and sub-weekly accepted cadences through the weekly aggregator, I reran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Result: `7 passed`

### Full-suite evidence

I also reran the full available suite:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests -v`

Result: `7 passed`

## Follow-up resolution: pre-aggregation negative validation and weekly date preservation

### RED evidence

After adding regressions for a masked negative sub-weekly row and for non-Monday weekly dates, I ran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

The first run surfaced the intended gap in the existing implementation:

- the negative-row regression passed only after moving the non-negativity check before weekly aggregation;
- the weekly-date regression failed because weekly rows were still being anchored to Mondays and the fixture was too short to pass the minimum-row rule.

### GREEN evidence

After updating `src/data_validation.py` to:

- validate raw outcome/media non-negativity before any weekly roll-up,
- preserve already-weekly dates unchanged,
- deterministically keep the last row when duplicate weekly dates appear,
- continue Monday-anchored aggregation only for sub-weekly data,

I reran:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Result: `9 passed`

### Full-suite evidence

I reran the full available suite:

`D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Result: `9 passed`

### Self-review

- Confirmed raw negative media/outcome values are flagged before any weekly aggregation can hide them.
- Confirmed weekly inputs with non-Monday dates keep their original dates in the prepared frame.
- Confirmed duplicate weekly dates are resolved deterministically without re-anchoring the date convention.
- Confirmed sub-weekly daily data still aggregates to Monday-anchored weeks.

### Concerns

The test run still emits a `PytestCacheWarning` because the worktree cache directory is not writable in this environment. It does not affect the validation results.

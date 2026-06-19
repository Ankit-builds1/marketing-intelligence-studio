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

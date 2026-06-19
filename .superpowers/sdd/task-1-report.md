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

# Task 5 Report — Safe exports and executive report

Status: complete

RED:
- Added `tests/test_reporting.py` first.
- Verified the initial run failed during collection with `ModuleNotFoundError: No module named 'src.reporting'`.

GREEN:
- Implemented `src/reporting.py` with:
  - `ReportContext`
  - `build_html_report(context) -> str`
  - `tables_as_zip(tables) -> bytes`
- HTML output is self-contained, printable, escaped, and includes an explicit observational-data caveat.
- ZIP output is deterministic, uses fixed metadata, and sanitizes entry names to avoid traversal and collisions.
- Empty warnings and empty tables are rendered with explicit messages.

Verification:
- Focused: `D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests/test_reporting.py -v -p no:cacheprovider`
- Full: `D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest -p no:cacheprovider`

Notes:
- No PDF dependencies were added.

Review fix evidence:
- Regression added for reversed insertion order byte-for-byte equality in `tests/test_reporting.py`.
- Focused verification: `D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest tests\test_reporting.py -v -p no:cacheprovider`
- Full verification: `D:\CausalInference_MMM\venv\Scripts\python.exe -m pytest -p no:cacheprovider`
- Result: all 35 tests passed.

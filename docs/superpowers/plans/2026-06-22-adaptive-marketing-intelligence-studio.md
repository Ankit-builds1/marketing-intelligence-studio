# Adaptive Marketing Intelligence Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a schema-adaptive marketing analytics application that safely routes diverse company CSVs to the strongest supported analysis and explains incompatible inputs.

**Architecture:** Pure Python modules profile, classify, prepare, route, and score uploaded data. Streamlit owns session orchestration and presentation only; the existing modelling, transformation, optimizer, and reporting modules remain the analytical core.

**Tech Stack:** Python 3.12–3.14, pandas, NumPy, SciPy, scikit-learn, Plotly, Streamlit, pytest.

## Global Constraints

- Never claim that an arbitrary non-marketing CSV can support MMM.
- Never interpret identifier columns as monetary media spend without user confirmation.
- Preserve daily, weekly, or monthly cadence; do not force every input to weekly.
- Daily modelling requires 28 periods, weekly 52, monthly 24.
- Uploaded data remains session-local and is not sent to external services.
- Disable ROI and optimization when monetary spend evidence is absent.
- All new behavior follows red-green-refactor TDD.

---

### Task 1: Dataset Profiler and Schema Detector

**Files:**
- Create: `src/dataset_profiler.py`
- Create: `src/schema_detection.py`
- Create: `tests/test_dataset_profiler.py`
- Create: `tests/test_schema_detection.py`

**Interfaces:**
- Produces `ColumnProfile`, `DatasetProfile`, `RoleSuggestion`, `SchemaDetection`, `profile_dataset(frame)`, and `detect_schema(frame, profile)`.
- Schema families are `wide_mmm`, `long_campaign`, `event_attribution`, `single_channel`, and `unsupported`.

- [ ] Write failing profiler tests proving numeric, datetime, identifier, missingness, uniqueness, negativity, and cardinality statistics are detected.
- [ ] Run `py -3.12 -m pytest tests/test_dataset_profiler.py -q` and confirm failures identify missing interfaces.
- [ ] Implement bounded profiling with deterministic column-role hints.
- [ ] Run the profiler tests and confirm they pass.
- [ ] Write failing schema tests for weekly wide MMM, long campaign exports, Criteo-style events, single-channel data, misleading numeric IDs, and unsupported payroll data.
- [ ] Run the schema tests and confirm the expected failures.
- [ ] Implement classification, role recommendations, confidence, and human-readable reasons.
- [ ] Run both test modules and commit `feat: add adaptive dataset profiling and schema detection`.

### Task 2: Cadence-Aware Preparation

**Files:**
- Create: `src/cadence.py`
- Create: `tests/test_cadence.py`
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

**Interfaces:**
- Produces `CadenceInfo`, `detect_cadence(dates)`, `minimum_periods(cadence)`, and `aggregate_periods(frame, mapping, cadence)`.
- Extends `PreparedData` with cadence and observed-period metadata without breaking current callers.

- [ ] Write failing tests for daily, weekly, monthly, Unix timestamp, relative timestamp, duplicate period, short-history, and irregular data behavior.
- [ ] Confirm failures with `py -3.12 -m pytest tests/test_cadence.py tests/test_data_validation.py -q`.
- [ ] Implement cadence detection and per-cadence minimums.
- [ ] Refactor validation to preserve cadence, aggregate duplicate periods consistently, and issue exploratory-vs-training guidance.
- [ ] Run cadence and legacy validation tests; commit `feat: support daily weekly and monthly marketing data`.

### Task 3: Event and Long-Campaign Preprocessor

**Files:**
- Create: `src/event_preprocessor.py`
- Create: `tests/test_event_preprocessor.py`

**Interfaces:**
- Produces `PreparationRequest`, `PreparationResult`, and `prepare_marketing_data(frame, request)`.
- Result includes prepared frame, generated `DataMapping`, transformation log, warnings, source-row count, and output-period count.

- [ ] Write failing tests for Criteo event logs, standard campaign exports, Unix timestamps, relative timestamps requiring an anchor, top-channel selection, low-volume channel grouping, aggregation, and no-spend errors.
- [ ] Confirm failures with `py -3.12 -m pytest tests/test_event_preprocessor.py -q`.
- [ ] Implement timestamp interpretation, period aggregation, campaign/channel pivoting, and transparent transformation logs.
- [ ] Verify transformed outputs contain no invented monetary values and preserve source totals within floating-point tolerance.
- [ ] Run tests and commit `feat: prepare event and campaign exports for analysis`.

### Task 4: Analysis Router and Reliability Score

**Files:**
- Create: `src/analysis_router.py`
- Create: `src/reliability.py`
- Create: `tests/test_analysis_router.py`
- Create: `tests/test_reliability.py`

**Interfaces:**
- Produces `AnalysisCapabilities`, `route_analysis(profile, prepared, mapping)` and `ReliabilityAssessment`, `assess_reliability(prepared, model_result=None)`.
- Capability flags control modelling, contribution, monetary ROI, optimization, and exports.

- [ ] Write failing routing tests for full MMM, single-channel response, activity-only response, campaign performance, exploratory-only, and unsupported inputs.
- [ ] Write failing reliability tests for strong, directional, and insufficient cases.
- [ ] Implement explicit capability reasons and reliability factor scoring.
- [ ] Run both modules' tests and commit `feat: route defensible analyses and score reliability`.

### Task 5: Adaptive Service Layer and End-to-End Workflows

**Files:**
- Create: `src/adaptive_service.py`
- Create: `tests/test_adaptive_service.py`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Produces `AdaptiveAnalysisSession`, `understand_upload(frame)`, `prepare_upload(frame, selections)`, and `run_supported_analysis(session, model_config)`.
- Provides one stable API for Streamlit and future Gradio clients.

- [ ] Write failing end-to-end tests for wide weekly MMM, daily long campaign, event attribution, monthly, single-channel, and unsupported datasets.
- [ ] Confirm failures before implementation.
- [ ] Implement orchestration using Tasks 1–4 and existing model/report modules.
- [ ] Ensure incompatible inputs return guidance rather than exceptions or fabricated outputs.
- [ ] Run end-to-end tests and commit `feat: orchestrate adaptive marketing analysis workflows`.

### Task 6: Streamlit Six-Stage Product Redesign

**Files:**
- Modify: `app/budget_optimizer_app.py`
- Modify: `.streamlit/config.toml`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Workflow stages: Upload, Understand, Prepare, Analyze, Optimize, Export.
- Uses `src/adaptive_service.py`; no duplicated classification or preparation logic.

- [ ] Write failing UI-state tests for stage progression, recommended mappings, preparation preview, blocked explanations, capability-based controls, and session resets.
- [ ] Confirm failures.
- [ ] Add navy/indigo/cyan/coral theme CSS, gradient hero, progress ribbon, status cards, themed charts, responsive spacing, and accessible contrast.
- [ ] Implement all six workflow stages with actionable status messages and downloadable prepared data.
- [ ] Keep the bundled demo as an optional quick start, never as the only successful path.
- [ ] Run UI-state and end-to-end tests; commit `feat: redesign adaptive Streamlit experience`.

### Task 7: Reporting, Documentation, and Recruiter Guidance

**Files:**
- Modify: `src/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `README.md`
- Create: `docs/USER_GUIDE.md`
- Create: `docs/DATA_REQUIREMENTS.md`

**Interfaces:**
- Reports include dataset type, transformation history, reliability, capabilities, limitations, diagnostics, and available business outputs.

- [ ] Write failing reporting tests for compatibility-only, exploratory, and full-MMM reports.
- [ ] Confirm failures.
- [ ] Extend HTML/ZIP exports while preserving existing full-MMM report behavior.
- [ ] Document each screen, supported schemas, sample mappings, recruiter test procedure, privacy, interpretation, and limitations in plain language.
- [ ] Run reporting tests and commit `docs: add adaptive workflow and recruiter guide`.

### Task 8: Full Verification and Deployment

**Files:**
- Modify only files required by failures discovered during verification, with a failing regression test first.

- [ ] Run `py -3.12 -m pytest -q` and require zero failures.
- [ ] Run `py -3.12 -m compileall -q app src`.
- [ ] Start Streamlit locally with `streamlit run app/budget_optimizer_app.py --server.headless true` and verify startup logs.
- [ ] Browser-test one wide MMM upload, one long/event upload, and one unsupported upload through the full applicable workflow.
- [ ] Inspect `git status` and exclude the untracked Criteo-derived sample unless explicitly approved.
- [ ] Push the verified branch to GitHub `main`.
- [ ] Monitor Streamlit Cloud until the public app responds successfully.
- [ ] Browser-test the public URL and verify upload, preparation, analysis, optimization, and downloads.
- [ ] Deliver the corrected URL and plain-language project walkthrough.


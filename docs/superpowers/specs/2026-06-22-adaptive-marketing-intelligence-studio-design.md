# Adaptive Marketing Intelligence Studio Design

## Objective

Upgrade the deployed Streamlit application from a weekly-MMM-only demo into an adaptive marketing analytics product. It must accept diverse company CSV schemas, determine which analysis is supported, prepare compatible data transparently, avoid misleading outputs, and present results through a polished, responsive interface.

The product will not claim that every arbitrary CSV can support MMM. Unsupported data receives a compatibility report, explicit missing requirements, and a downloadable template.

## Supported Input Families

1. MMM-ready wide time series with one date, one outcome, two or more media columns, and optional controls.
2. Long campaign exports with date/time, campaign or channel, spend, and outcome/performance metrics.
3. Event-level advertising logs with timestamp, campaign/channel identifier, cost, clicks, and conversions.
4. Single-channel time series, routed to single-channel response analysis rather than multi-channel budget optimization.
5. Unsupported or non-marketing files, routed to compatibility guidance without model fitting.

CSV and compressed CSV uploads are supported. Large inputs are read in bounded chunks where practical.

## Data Understanding and Preparation

The application profiles every upload and produces:

- dataset classification and confidence;
- inferred date/time, outcome, media, spend, campaign/channel, and control roles;
- reasons for recommendations;
- data quality metrics and blocking issues;
- a before/after preparation preview;
- editable mappings so the user remains in control.

Identifiers must never be silently treated as spend. Unix and relative timestamps require explicit interpretation. Long campaign/event data is aggregated and pivoted into analysis-ready channel columns. Every transformation is described and the prepared data is downloadable.

## Cadence Policy

- Daily modelling: at least 28 usable periods.
- Weekly modelling: at least 52 usable periods.
- Monthly modelling: at least 24 usable periods.
- Shorter histories: exploratory summaries only.

The pipeline preserves the selected cadence instead of forcing every dataset to weekly frequency. Duplicate periods are aggregated consistently: outcomes and spend are summed; controls are averaged unless the user chooses another supported rule.

## Analysis Router

The router selects the strongest defensible workflow:

- multiple spend channels, time, outcome: full MMM;
- one spend channel, time, outcome: single-channel response analysis;
- campaign/event cost and conversions: campaign performance plus prepared time-series analysis when history is sufficient;
- media activity without monetary spend: media-response analysis with ROI disabled;
- insufficient or incompatible data: compatibility report only.

The UI must never display ROI, causal lift, or budget optimization when required monetary spend/outcome evidence is absent.

## Modelling and Reliability

The existing adstock, saturation, regularized modelling, time-based validation, uncertainty, contribution, reporting, and constrained optimization modules remain the foundation. New routing and cadence metadata determine which outputs are enabled.

Every trained result receives a reliability label—Strong, Directional, or Insufficient—based on history length, validation performance, collinearity, missingness, and uncertainty. Copy consistently states that outputs are observational model estimates, not guaranteed causal effects.

## User Experience

The primary workflow contains six stages:

1. Upload
2. Understand
3. Prepare
4. Analyze
5. Optimize
6. Export

Each stage displays complete, warning, or blocked status. The app provides actionable next steps instead of exposing disabled controls without explanation.

Visual design uses a deep navy foundation, indigo/cyan gradients, teal and coral accents, high-contrast cards, a progress ribbon, consistent typography, responsive spacing, and themed Plotly charts. Styling must remain readable and professional on desktop and narrow screens.

## Exports

Depending on the routed workflow, users can download:

- prepared analysis CSV;
- compatibility and data-quality report;
- channel contribution and modeled ROI tables;
- uncertainty and diagnostics tables;
- optimized allocation;
- executive HTML report;
- ZIP package containing all available outputs and limitations.

## Architecture

New focused modules:

- `src/dataset_profiler.py`: column statistics and data-quality profile;
- `src/schema_detection.py`: dataset-family classification and role recommendations;
- `src/event_preprocessor.py`: timestamp conversion, aggregation, and long-to-wide pivoting;
- `src/cadence.py`: cadence detection, requirements, and period aggregation;
- `src/analysis_router.py`: supported-workflow selection and capability flags;
- `src/reliability.py`: reliability scoring and explanation.

Existing model, optimizer, validation, reporting, and transformation modules are extended only where required. Streamlit orchestrates the modules but does not duplicate their business logic.

## Error Handling and Privacy

- Uploads remain session-local and are not intentionally persisted by the app.
- Parsing, mapping, transformation, modelling, and download failures show human-readable recovery guidance.
- Large-file and memory limits are explained before processing fails.
- No external service receives uploaded company data.
- Invalid datasets never trigger a model run.

## Verification

Automated coverage includes weekly wide MMM, daily long campaign, event-level attribution, monthly, single-channel, unsupported, missing/duplicate, timestamp, and end-to-end report workflows. Completion requires the full test suite, syntax checks, local Streamlit startup, browser verification of representative workflows, successful GitHub push, and successful public Streamlit deployment verification.


# Marketing Intelligence Studio — Design Specification

## Objective

Transform the existing fixed-schema Streamlit dashboard into a recruiter-ready marketing mix modeling application. A visitor can use a bundled demo dataset or upload compatible weekly marketing data, map its columns, validate its fitness for analysis, fit an interpretable model, explore channel performance, optimize a constrained budget, and download results.

The product must be honest about statistical limits. It reports model-based estimates with uncertainty and diagnostics; it never promises that arbitrary observational data proves causal truth.

## Intended User and Success Criteria

The primary user is a recruiter, interviewer, or analyst evaluating a data-science portfolio. They should be able to complete the full workflow without documentation or private data.

Phase 1 succeeds when:

- the bundled sample works end to end;
- a compatible CSV can be uploaded and mapped without renaming columns;
- unsuitable data produces specific, actionable validation messages;
- the application produces model diagnostics, channel contribution, ROI, response curves, uncertainty ranges, and a constrained budget recommendation;
- results and assumptions can be downloaded;
- core analysis functions have automated tests;
- the application runs locally and is structured for Hugging Face Spaces deployment.

## Scope

### Included

- Guided demo mode and CSV upload mode.
- Flexible mapping for date, outcome, media-spend, and optional control columns.
- Weekly time-series preparation, numeric coercion, sorting, duplicate handling, and missing-value reporting.
- Data-quality gates for row count, date validity, missingness, variance, non-negative spend, and severe multicollinearity.
- Geometric adstock and Hill-style saturation transformations.
- Trend and calendar-seasonality features.
- Regularized regression with time-aware holdout evaluation.
- Bootstrap uncertainty intervals for channel metrics.
- Actual-versus-predicted, residual, contribution, ROI, response-curve, and correlation diagnostics.
- Budget optimization with channel minimum and maximum constraints.
- Downloadable result tables and a self-contained HTML executive report.
- Clear methodology, assumptions, limitations, and reliability warnings.
- Unit and integration tests for validation, transformations, modeling, and optimization.

### Deferred

- User accounts, permanent cloud storage, databases, and collaborative workspaces.
- Fully Bayesian MCMC modeling, because its runtime is unsuitable for the first public demo.
- Automated causal-DAG discovery or claims of causal identification from arbitrary uploads.
- PDF rendering; HTML is more reliable on free hosting and remains printable to PDF.
- Arbitrary data frequencies. Phase 1 accepts weekly data or data that can be consistently aggregated to weeks.

## User Experience

The Streamlit application uses a five-step guided workflow:

1. **Choose data** — load a bundled realistic demo or upload a CSV; show a template download.
2. **Map and validate** — select date, outcome, media channels, and controls; display pass, warning, and blocking checks.
3. **Configure and train** — review sensible defaults for adstock, saturation, holdout size, and bootstrap count; run the model explicitly.
4. **Explore results** — show an executive summary followed by diagnostics, channel performance, response curves, and uncertainty.
5. **Optimize and export** — set total budget and per-channel bounds, compare current and recommended allocations, and download tables and the HTML report.

The bundled demo remains the default so a recruiter sees useful results immediately. Advanced settings stay inside expanders to keep the main flow approachable.

## Architecture

The existing `src/mmm_utils.py` remains available for notebook compatibility, while the production workflow moves into focused modules:

- `src/data_validation.py` — schema mapping, preparation, quality checks, and typed validation results.
- `src/transformations.py` — adstock, saturation, trend, and seasonality features.
- `src/mmm_model.py` — model configuration, fitting, backtesting, bootstrapping, contributions, ROI, and response curves.
- `src/budget_optimizer.py` — constrained nonlinear allocation using fitted response functions.
- `src/reporting.py` — serializable result tables and HTML report generation.
- `app/budget_optimizer_app.py` — orchestration and presentation only.

Domain outputs use dataclasses so the UI does not depend on model internals. No uploaded file is written to disk; it is processed in the Streamlit session.

## Data Contract

Required mapped fields:

- one date column;
- one non-negative numeric outcome column, such as revenue, conversions, or sales;
- at least two non-negative numeric media-spend columns.

Optional fields are numeric controls such as price, promotions, distribution, holiday intensity, or competitor activity.

The default minimum is 52 weekly observations. Between 52 and 103 observations produces a limited-history warning; 104 or more is preferred. The application blocks training when dates cannot be parsed, mapped columns are duplicated, the outcome has no useful variance, spend channels are constant, missingness is excessive, or too few usable rows remain.

## Modeling

Each media channel receives a configurable geometric-adstock decay and Hill saturation curve. Defaults are conservative and can be adjusted in advanced settings. Trend, month/quarter seasonality, and mapped controls form the baseline component.

The initial estimator is non-negative ridge regression for media effects, selected for stability, speed, and interpretability. Time order is preserved: the final observations form the holdout set, and metrics include R², MAE, MAPE when valid, and baseline comparison. Bootstrap refits estimate uncertainty; low stability or poor holdout performance lowers the displayed reliability rating.

Contributions are decomposed into baseline and media components. ROI is incremental modeled outcome divided by channel spend. Results are labeled modeled estimates rather than experimentally verified lift.

## Budget Optimization

Optimization uses each channel's fitted adstock/saturation response while holding controls at representative values. The user chooses a future total budget and channel bounds. The solver must preserve the total budget within tolerance and return either a feasible allocation or an actionable explanation.

The results compare current average allocation with the recommendation, predicted outcome difference, marginal return, and bound constraints. The application warns when the requested budget is far outside the observed spend range.

## Error Handling and Trust

- Blocking validation failures prevent training and explain how to repair the CSV.
- Non-blocking risks remain visible beside results.
- Model exceptions are converted into user-facing messages while technical details remain available in an expander.
- Optimization infeasibility reports which bounds conflict with the total budget.
- The report includes data period, mapping, configuration, diagnostics, assumptions, and warnings.
- Uploaded data remains session-local and is not persisted.

## Testing and Verification

Tests will use deterministic synthetic data and cover:

- valid and invalid CSV schemas;
- date preparation, missing data, and validation severity;
- known adstock and saturation outputs;
- deterministic model fitting and time-aware holdout behavior;
- contribution reconciliation and finite uncertainty intervals;
- optimizer feasibility, budget conservation, and invalid bounds;
- report generation;
- an end-to-end smoke test from demo data to exported results.

Completion requires the automated suite to pass and a browser verification of the Streamlit workflow with no runtime errors.

## Deployment Shape

The repository will include pinned runtime dependencies, a Hugging Face Spaces-compatible README header, and no dependence on local pickle artifacts for the upload workflow. Model training happens on demand from the selected dataset. Expensive steps are cached by data-and-configuration hash and bounded to keep free-hosting runtime practical.

## Delivery Phases

1. Analysis core: validation, transformations, modeling, uncertainty, optimization, and tests.
2. Product UI: guided workflow, charts, warnings, exports, and demo experience.
3. Portfolio finish: README, deployment files, screenshots, browser verification, and Hugging Face readiness.


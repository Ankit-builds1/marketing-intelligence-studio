---
title: Marketing Intelligence Studio
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app/gradio_app.py
pinned: false
short_description: Causal MMM analysis and marketing budget optimization.
tags:
  - streamlit
  - marketing-mix-modeling
  - data-science
  - portfolio
---

# Marketing Intelligence Studio

A recruiter-friendly marketing mix modeling application that turns weekly spend data into a guided workflow: validate the dataset, fit an interpretable model, inspect reliability and ROI, optimize a constrained budget, and export a lightweight executive report.

Live demo: deployment pending. The repository is prepared for a native Hugging Face Gradio Space, but no public URL is published yet.

## Why this project exists

Many marketing dashboards stop at correlation or static attribution. This app is designed to show stronger product thinking:

- it accepts realistic weekly CSV inputs instead of a single fixed schema;
- it blocks bad data before modeling;
- it shows uncertainty and reliability warnings instead of overclaiming;
- it turns model output into a concrete budget-allocation decision.

The repo started as a broader causal-inference/MMM exploration. The current deployable app surface is the Streamlit app in `app/budget_optimizer_app.py`, which uses a lean, reproducible MMM workflow suitable for portfolio review and free-tier hosting.

## Product workflow

1. Choose a bundled demo dataset or upload a session-local CSV.
2. Map date, outcome, media, and optional control columns.
3. Validate weekly fitness, row count, missingness, and channel quality.
4. Train the model with configurable adstock, saturation, holdout, and bootstrap settings.
5. Review results, optimize a future budget, and download HTML/ZIP exports.

The bundled demo is the default so a recruiter can land in a working end-to-end experience without any private data.

## Realistic data contract

The app expects weekly marketing data or higher-frequency data that can be consistently aggregated to weeks.

| Role | Requirement | Typical examples | What happens if it fails |
| --- | --- | --- | --- |
| Date column | Parseable date values | `week`, `date`, `DATE` | Training is blocked if more than 5% are invalid or cadence is irregular |
| Outcome column | One non-negative numeric KPI | `revenue`, `sales`, `orders`, `conversions` | Training is blocked for negative, missing-heavy, or constant outcomes |
| Media columns | At least two non-negative numeric spend columns | `tv_S`, `search_S`, `facebook_S` | Training is blocked for too few channels, negative spend, or no variation |
| Control columns | Optional numeric business drivers | competitor sales, pricing, events, holidays | Kept optional; used when available to absorb non-media effects |

Validation rules in the current app:

- minimum 52 usable weekly rows;
- 52-103 rows triggers a limited-history warning;
- duplicate column-role assignments are blocked;
- excessive missingness is blocked;
- severe channel similarity triggers a warning;
- training is disabled until blocking issues are resolved.

## Methodology

This application does not claim that arbitrary observational data proves causal truth. It reports modeled estimates that are useful for planning and scenario analysis.

- Adstock: each media channel uses geometric adstock to carry spend effects across weeks.
- Saturation: each channel then passes through a Hill-style response curve to capture diminishing returns.
- Baseline structure: optional controls, a normalized trend term, and annual seasonality terms (`sin_52`, `cos_52`) absorb non-media movement.
- Estimator: non-negative ridge-style regression keeps media effects directionally sensible and more stable under collinearity.
- Time-aware validation: the most recent observations are held out, and the app reports holdout R² and MAE before refitting on the full series.
- Uncertainty: moving-block bootstrap refits generate channel contribution intervals.
- ROI: modeled ROI is incremental modeled outcome divided by observed channel spend.
- Optimization: a constrained nonlinear solver reallocates a fixed future budget while respecting user-specified channel bounds and budget conservation.

## Architecture

| File | Responsibility |
| --- | --- |
| `app/gradio_app.py` | Native Hugging Face Space interface for upload, analysis, optimization, and downloads |
| `app/budget_optimizer_app.py` | Five-step Streamlit interface for local use |
| `src/data_validation.py` | Schema mapping, weekly preparation, blocking errors, and warnings |
| `src/transformations.py` | Geometric adstock, Hill saturation, trend, and seasonality features |
| `src/mmm_model.py` | Holdout split, constrained fit, reliability scoring, bootstrap intervals, contributions, and ROI |
| `src/budget_optimizer.py` | Budget-feasible constrained optimization over fitted response curves |
| `src/reporting.py` | Self-contained HTML report and zipped result-table exports |

## Privacy and trust

- Uploaded CSVs remain in the current Streamlit session and are not intentionally written to disk by the app.
- The app keeps validation errors visible instead of silently coercing bad inputs.
- Results are labeled as modeled estimates from observational data.
- Extrapolation outside the observed spend range is explicitly warned.

## Limitations

- Weekly marketing data only; irregular cadence is blocked.
- At least 52 usable weeks are required, and two years of history is preferred.
- Results depend on the chosen mappings and transformation settings.
- High channel collinearity can make channel-level decomposition unstable.
- ROI and optimizer outputs are model-based planning aids, not experimentally verified lift.
- The legacy notebook still contains heavier causal-inference tooling, but that stack is optional and separate from the deployable app requirements.

## Local setup

### Native Gradio interface (same interface used by Hugging Face)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app/gradio_app.py
```

The original five-step Streamlit interface also remains available locally:

```powershell
python -m streamlit run app/budget_optimizer_app.py
```

### Optional notebook workflow

Install the heavier notebook stack only if you want the legacy notebook, downloader utilities, or the original causal-inference exploration:

```powershell
python -m pip install -r requirements-notebook.txt
```

## Tests and verification

```powershell
python -m pip check
python -m pytest tests -q
python -m compileall app src tests
```

The automated suite covers validation, transformations, model behavior, reporting, budget optimization, the native Gradio service layer, and Streamlit end-to-end smoke coverage.

## Deployment

### Hugging Face Spaces

This repository is configured as a native Gradio Space, so Hugging Face installs the Python requirements and launches the declared app directly:

- a valid README YAML header using `sdk: gradio`;
- `app_file: app/gradio_app.py`;
- a tested Gradio version declared in the Space metadata;
- lean runtime dependencies in `requirements.txt`.

Suggested deployment flow:

1. Create a new Hugging Face Space with the Gradio SDK.
2. Push this repository contents to that Space.
3. Hugging Face installs `requirements.txt` and launches `app/gradio_app.py`.
4. Open the Space and run the bundled demo before adding its URL to your resume.

No public deployment URL is included here because deployment has not been published yet.

## Accurate resume bullets

- Built and packaged a native Gradio marketing intelligence app for Hugging Face Spaces that accepts CSV uploads, validates data quality, trains an interpretable MMM, and exports executive-ready outputs; retained a five-step Streamlit interface for local analysis.
- Implemented geometric adstock, Hill saturation, time-based holdout evaluation, moving-block bootstrap uncertainty, modeled ROI, and constrained budget optimization in a reusable Python analysis pipeline.
- Packaged the project for portfolio review with automated tests, session-local privacy handling, lean runtime dependencies, and native Gradio deployment on Hugging Face Spaces.

## Screenshots

Deployment is still pending, so the README intentionally does not claim live screenshots from a published Space yet. Recommended captures to add before public launch:

- Gradio Data & Mapping tab with the bundled demo loaded
- Model Results tab with actual-vs-modeled fit, contribution, ROI, and response curves
- Budget & Downloads tab with a budget-conserving recommendation and report files

## Repository notes

- `requirements.txt` is intentionally lean for the app and test surface.
- `requirements-notebook.txt` keeps the optional notebook and legacy causal-inference stack available.
- `app/gradio_app.py` is the native Hugging Face entry point; no container runtime or local virtualization is required.

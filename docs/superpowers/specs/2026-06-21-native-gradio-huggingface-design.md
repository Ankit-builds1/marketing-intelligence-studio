# Native Gradio Hugging Face Deployment Design

## Goal

Deploy the Marketing Intelligence Studio as a native Hugging Face Gradio Space without Docker or local virtualization. Preserve the existing Streamlit app for local use and reuse the same tested analysis modules for both interfaces.

## Deployment Architecture

Hugging Face launches `app/gradio_app.py` using the README Space metadata:

- `sdk: gradio`
- `app_file: app/gradio_app.py`
- a compatible Gradio SDK version

The Gradio app imports the existing validation, transformation, MMM, optimization, and reporting modules. It does not duplicate modeling logic and does not use model pickle artifacts.

## User Workflow

The Space provides two paths:

1. Load the bundled realistic demo dataset.
2. Upload a marketing CSV.

The user maps a date column, one outcome column, at least two media columns, and optional controls. The app validates the data, trains the model, and returns:

- reliability and holdout metrics;
- data-quality errors and warnings;
- channel contribution, spend, ROI, and uncertainty intervals;
- actual-versus-modeled and channel-performance charts;
- response curves;
- a constrained budget recommendation;
- downloadable HTML and ZIP reports.

Invalid inputs return actionable messages without crashing. Uploaded data is processed only for the active request and is not intentionally persisted by application code.

## Interface Scope

Gradio will use a guided tabbed interface with Data & Mapping, Model Results, and Budget & Downloads sections. The default demo configuration works immediately for recruiters. Advanced transformation settings remain in the Streamlit app; the hosted Gradio demo uses documented conservative defaults to keep the public workflow fast and dependable.

## Files

- Create `app/gradio_app.py` for the native Space interface.
- Create `tests/test_gradio_app.py` for import, demo, validation, result, and optimizer coverage.
- Update `requirements.txt` with Gradio.
- Update `README.md` metadata and deployment instructions.
- Delete `Dockerfile` and remove Docker wording.
- Keep `.streamlit/config.toml` because Streamlit remains the local interface.

## Verification

Completion requires:

- the full Python test suite passing;
- successful import and construction of the Gradio Blocks app;
- demo analysis and optimization completing locally;
- invalid CSV/mapping behavior returning a readable validation result;
- Gradio launching locally and its HTTP endpoint responding;
- no Docker requirement or Docker instruction remaining in deployable files;
- a final code review and clean Git state.

## Resume Positioning

The project will be described as an end-to-end marketing intelligence application deployed with Gradio on Hugging Face Spaces. Claims remain limited to implemented, tested behavior and modeled estimates; no unsupported business uplift or causal guarantee is stated.

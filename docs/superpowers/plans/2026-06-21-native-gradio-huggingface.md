# Native Gradio Hugging Face Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Gradio Space interface that exposes the tested MMM workflow on Hugging Face without Docker.

**Architecture:** `app/gradio_app.py` is a thin UI adapter over the existing `src` analysis modules. It uses serializable UI values and temporary download files, while model logic remains centralized in the existing validated pipeline.

**Tech Stack:** Python 3.12, Gradio, pandas, NumPy, Plotly, SciPy, scikit-learn, pytest

## Global Constraints

- No Docker or virtualization requirement.
- Keep the Streamlit app working for local use.
- Use native Hugging Face metadata with `sdk: gradio` and `app_file: app/gradio_app.py`.
- Do not duplicate modeling logic or use pickle artifacts.
- Do not claim causal proof or guaranteed business uplift.
- Uploaded files are not intentionally persisted beyond temporary files needed for user downloads.
- Keep the public demo fast enough for free Hugging Face hardware.

---

### Task 1: Testable Gradio analysis adapter

**Files:**
- Create: `app/gradio_app.py`
- Create: `tests/test_gradio_app.py`

**Interfaces:**
- Produce `load_source(file_path)`, `suggest_mapping(frame)`, `analyze_dataset(...)`, `optimize_from_analysis(...)`, and `build_demo()`.
- Reuse `DataMapping`, `prepare_and_validate`, `ChannelTransform`, `build_feature_set`, `fit_mmm`, `optimize_allocation`, `ReportContext`, `build_html_report`, and `tables_as_zip`.

- [ ] Write failing tests for demo loading, mapping suggestions, successful analysis, invalid mapping, optimizer budget conservation, and Gradio Blocks construction.
- [ ] Run `python -m pytest tests/test_gradio_app.py -v -p no:cacheprovider` and verify failures are due to the missing module.
- [ ] Implement the minimal adapter and Blocks UI. Use conservative defaults and at most 20 bootstrap samples.
- [ ] Return Plotly figures, result tables, readable status messages, and temporary HTML/ZIP download paths.
- [ ] Run focused and full tests, then commit `feat: add native Gradio Space interface`.

### Task 2: Native Hugging Face configuration

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Delete: `Dockerfile`

- [ ] Add a tested compatible Gradio range to `requirements.txt`.
- [ ] Change README metadata to `sdk: gradio`, `app_file: app/gradio_app.py`, and an installed compatible `sdk_version`.
- [ ] Replace Docker deployment instructions with native Space creation and Git push steps.
- [ ] Keep Streamlit local instructions and remove every Docker requirement/reference.
- [ ] Update resume bullets to describe native Gradio/Hugging Face readiness accurately without claiming a live URL.
- [ ] Commit `docs: configure native Gradio Hugging Face Space`.

### Task 3: Runtime and acceptance verification

**Files:**
- Modify only files required by proven defects.

- [ ] Run `python -m pip check`.
- [ ] Run `python -m pytest tests -v -p no:cacheprovider` and require zero failures.
- [ ] Run `python -m compileall -q app src tests`.
- [ ] Import `app.gradio_app.demo` and verify it is a `gradio.Blocks` instance.
- [ ] Launch Gradio locally on a free port, verify HTTP 200, and inspect logs for tracebacks.
- [ ] Run demo analysis and constrained optimization through the adapter functions.
- [ ] Confirm `git grep -i docker` finds no deployment requirement.

### Task 4: Final review and synchronization

**Files:**
- No planned product edits.

- [ ] Request a broad final review against the approved design.
- [ ] Fix all Critical and Important findings and rerun affected tests.
- [ ] Record clean acceptance evidence in the progress ledger.
- [ ] Push the feature branch into `D:\CausalInference_MMM` through local Git synchronization.
- [ ] Verify the D: repository branch contains the final commits and tests pass there.


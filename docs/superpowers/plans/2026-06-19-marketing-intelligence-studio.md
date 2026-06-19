# Marketing Intelligence Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a recruiter-ready Streamlit application with CSV upload that validates weekly marketing data, fits an interpretable nonlinear MMM, explains uncertainty, optimizes budget, and exports results.

**Architecture:** Keep Streamlit as a thin orchestration layer and move data preparation, transformations, modeling, optimization, and reporting into focused `src` modules. Use deterministic dataclass-based interfaces, time-aware validation, bounded model runtime, and session-local uploaded data.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, scikit-learn, Plotly, Streamlit, pytest

## Global Constraints

- Accept one date column, one non-negative outcome, at least two non-negative media-spend columns, and optional numeric controls.
- Require at least 52 usable weekly rows; warn below 104 rows.
- Never claim arbitrary observational data proves causal truth.
- Do not persist uploaded data.
- Preserve the notebook-facing `src/mmm_utils.py` API.
- Keep model training practical for free Hugging Face Spaces hardware.
- Use time-ordered holdouts and deterministic random seeds.
- Export a self-contained HTML report rather than requiring PDF system dependencies.

---

## File Map

- `src/data_validation.py`: mapping dataclasses, preparation, weekly aggregation, and quality gates.
- `src/transformations.py`: adstock, Hill saturation, calendar features, and model matrix construction.
- `src/mmm_model.py`: constrained ridge estimator, time holdout, metrics, bootstrap intervals, contributions, ROI, response curves, and reliability.
- `src/budget_optimizer.py`: nonlinear response evaluation and feasible constrained allocation.
- `src/reporting.py`: CSV bundles and escaped self-contained HTML executive report.
- `app/budget_optimizer_app.py`: five-step recruiter-facing Streamlit workflow.
- `tests/`: deterministic unit and end-to-end tests.
- `requirements.txt`, `README.md`, `.streamlit/config.toml`: deployment and portfolio finish.

### Task 1: Data mapping and validation

**Files:**
- Create: `src/data_validation.py`
- Create: `tests/test_data_validation.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `DataMapping`, `ValidationIssue`, `PreparedData`, and `prepare_and_validate(raw, mapping, min_rows=52)`.
- `PreparedData.frame` uses canonical columns `date`, `outcome`, `media__<name>`, and `control__<name>`.

- [ ] **Step 1: Add pytest and write failing validation tests**

```python
# tests/test_data_validation.py
import pandas as pd
from src.data_validation import DataMapping, prepare_and_validate

def valid_frame(rows: int = 104) -> pd.DataFrame:
    return pd.DataFrame({
        "week": pd.date_range("2024-01-01", periods=rows, freq="W-MON"),
        "sales": range(100, 100 + rows),
        "search": range(10, 10 + rows),
        "social": range(20, 20 + rows),
        "price": [1.0 + i / 1000 for i in range(rows)],
    })

def test_prepares_canonical_weekly_frame():
    mapping = DataMapping("week", "sales", ("search", "social"), ("price",))
    result = prepare_and_validate(valid_frame(), mapping)
    assert result.can_train
    assert list(result.frame.columns) == [
        "date", "outcome", "media__search", "media__social", "control__price"
    ]
    assert not [issue for issue in result.issues if issue.severity == "error"]

def test_blocks_too_few_rows_and_negative_spend():
    frame = valid_frame(40)
    frame.loc[0, "search"] = -1
    mapping = DataMapping("week", "sales", ("search", "social"), ())
    result = prepare_and_validate(frame, mapping)
    assert not result.can_train
    assert {issue.code for issue in result.issues} >= {"too_few_rows", "negative_media"}

def test_warns_about_near_duplicate_channels():
    frame = valid_frame()
    frame["social"] = frame["search"] * 1.001
    mapping = DataMapping("week", "sales", ("search", "social"), ())
    result = prepare_and_validate(frame, mapping)
    assert "high_collinearity" in {issue.code for issue in result.issues}
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.data_validation'`.

- [ ] **Step 3: Implement validation dataclasses and preparation**

```python
# src/data_validation.py
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DataMapping:
    date_col: str
    outcome_col: str
    channel_cols: tuple[str, ...]
    control_cols: tuple[str, ...] = ()

@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str

@dataclass
class PreparedData:
    frame: pd.DataFrame
    mapping: DataMapping
    issues: list[ValidationIssue]
    @property
    def can_train(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

def prepare_and_validate(raw: pd.DataFrame, mapping: DataMapping, min_rows: int = 52) -> PreparedData:
    selected = (mapping.date_col, mapping.outcome_col, *mapping.channel_cols, *mapping.control_cols)
    issues: list[ValidationIssue] = []
    if len(set(selected)) != len(selected):
        return PreparedData(pd.DataFrame(), mapping, [ValidationIssue("error", "duplicate_mapping", "Each role must use a different column.")])
    if len(mapping.channel_cols) < 2:
        issues.append(ValidationIssue("error", "too_few_channels", "Select at least two media channels."))
    missing = [column for column in selected if column not in raw.columns]
    if missing:
        issues.append(ValidationIssue("error", "missing_columns", f"Missing mapped columns: {', '.join(missing)}"))
        return PreparedData(pd.DataFrame(), mapping, issues)
    frame = raw.loc[:, selected].copy()
    dates = pd.to_datetime(frame[mapping.date_col], errors="coerce")
    if dates.isna().mean() > 0.05:
        issues.append(ValidationIssue("error", "invalid_dates", "More than 5% of date values cannot be parsed."))
    frame[mapping.date_col] = dates
    numeric = [mapping.outcome_col, *mapping.channel_cols, *mapping.control_cols]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[numeric].isna().mean().max() > 0.20:
        issues.append(ValidationIssue("error", "excessive_missingness", "A mapped numeric column has more than 20% missing values."))
    frame = frame.dropna(subset=[mapping.date_col, mapping.outcome_col, *mapping.channel_cols])
    frame = frame.sort_values(mapping.date_col).drop_duplicates(mapping.date_col, keep="last")
    if len(frame) < min_rows:
        issues.append(ValidationIssue("error", "too_few_rows", f"At least {min_rows} usable weekly rows are required."))
    elif len(frame) < 104:
        issues.append(ValidationIssue("warning", "limited_history", "Two years of weekly history is recommended."))
    if (frame[mapping.outcome_col] < 0).any():
        issues.append(ValidationIssue("error", "negative_outcome", "Outcome values must be non-negative."))
    if (frame[list(mapping.channel_cols)] < 0).any().any():
        issues.append(ValidationIssue("error", "negative_media", "Media spend values must be non-negative."))
    constant = [column for column in (mapping.outcome_col, *mapping.channel_cols) if frame[column].nunique(dropna=True) < 2]
    if constant:
        issues.append(ValidationIssue("error", "constant_columns", f"Columns need variation: {', '.join(constant)}"))
    if len(frame) > 1:
        median_gap = frame[mapping.date_col].diff().dt.days.dropna().median()
        if not 5 <= median_gap <= 9:
            issues.append(ValidationIssue("warning", "non_weekly_dates", "Dates do not appear to be consistently weekly."))
    media_corr = frame[list(mapping.channel_cols)].corr().abs()
    np.fill_diagonal(media_corr.values, 0.0)
    if (media_corr > 0.98).any().any():
        issues.append(ValidationIssue("warning", "high_collinearity", "Two media channels are nearly identical, so their separate effects may be unstable."))
    rename = {mapping.date_col: "date", mapping.outcome_col: "outcome"}
    rename.update({name: f"media__{name}" for name in mapping.channel_cols})
    rename.update({name: f"control__{name}" for name in mapping.control_cols})
    frame = frame.rename(columns=rename).reset_index(drop=True)
    return PreparedData(frame, mapping, issues)
```

Add `pytest>=8.0,<9` to `requirements.txt`.

- [ ] **Step 4: Run validation tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_data_validation.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit the validated input pipeline**

```powershell
git add requirements.txt src/data_validation.py tests/test_data_validation.py
git commit -m "feat: add marketing data validation"
```

### Task 2: Marketing transformations and feature matrix

**Files:**
- Create: `src/transformations.py`
- Create: `tests/test_transformations.py`

**Interfaces:**
- Consumes canonical `PreparedData.frame` columns.
- Produces `ChannelTransform`, `FeatureSet`, `geometric_adstock`, `hill_saturation`, and `build_feature_set`.

- [ ] **Step 1: Write deterministic transformation tests**

```python
# tests/test_transformations.py
import numpy as np
import pandas as pd
from src.transformations import ChannelTransform, build_feature_set, geometric_adstock, hill_saturation

def test_geometric_adstock_carries_effect_forward():
    result = geometric_adstock(np.array([10.0, 0.0, 0.0]), decay=0.5)
    np.testing.assert_allclose(result, [10.0, 5.0, 2.5])

def test_hill_saturation_hits_half_at_half_saturation():
    result = hill_saturation(np.array([0.0, 10.0, 100.0]), half_saturation=10.0, slope=1.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 100.0 / 110.0])

def test_feature_set_tracks_channel_columns():
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=8, freq="W-MON"),
        "outcome": np.arange(8.0), "media__search": np.arange(8.0),
        "media__social": np.arange(8.0)[::-1], "control__price": np.ones(8),
    })
    params = {name: ChannelTransform(0.5, 3.0, 1.0) for name in ("search", "social")}
    features = build_feature_set(frame, params)
    assert features.channel_feature_names == {"search": "media__search", "social": "media__social"}
    assert {"trend", "sin_52", "cos_52", "control__price"}.issubset(features.X.columns)
```

- [ ] **Step 2: Verify tests fail because transformations are absent**

Run: `venv\Scripts\python.exe -m pytest tests/test_transformations.py -v`

Expected: FAIL during collection with missing `src.transformations`.

- [ ] **Step 3: Implement bounded adstock, saturation, and calendar features**

```python
# src/transformations.py
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ChannelTransform:
    decay: float = 0.5
    half_saturation: float = 1.0
    slope: float = 1.0

@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series
    channel_feature_names: dict[str, str]
    transforms: dict[str, ChannelTransform]
    original_media: pd.DataFrame

def geometric_adstock(values: np.ndarray, decay: float) -> np.ndarray:
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0, 1)")
    result = np.zeros(len(values), dtype=float)
    for index, value in enumerate(np.asarray(values, dtype=float)):
        result[index] = value + (result[index - 1] * decay if index else 0.0)
    return result

def hill_saturation(values: np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    if half_saturation <= 0 or slope <= 0:
        raise ValueError("half_saturation and slope must be positive")
    safe = np.maximum(np.asarray(values, dtype=float), 0.0)
    powered = np.power(safe, slope)
    return np.divide(powered, powered + half_saturation ** slope, out=np.zeros_like(powered), where=(powered + half_saturation ** slope) != 0)

def build_feature_set(frame: pd.DataFrame, transforms: dict[str, ChannelTransform]) -> FeatureSet:
    X = pd.DataFrame(index=frame.index)
    channel_names: dict[str, str] = {}
    for channel, config in transforms.items():
        source = f"media__{channel}"
        adstocked = geometric_adstock(frame[source].to_numpy(), config.decay)
        X[source] = hill_saturation(adstocked, config.half_saturation, config.slope)
        channel_names[channel] = source
    for column in frame.columns:
        if column.startswith("control__"):
            X[column] = frame[column].interpolate(limit_direction="both").fillna(frame[column].median())
    index = np.arange(len(frame), dtype=float)
    X["trend"] = index / max(len(frame) - 1, 1)
    X["sin_52"] = np.sin(2 * np.pi * index / 52.0)
    X["cos_52"] = np.cos(2 * np.pi * index / 52.0)
    original_media = frame[[f"media__{name}" for name in transforms]].copy()
    original_media.columns = list(transforms)
    return FeatureSet(X, frame["outcome"].astype(float), frame["date"], channel_names, transforms, original_media)
```

- [ ] **Step 4: Run transformation tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_transformations.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit transformations**

```powershell
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add adstock and saturation transforms"
```

### Task 3: Time-aware MMM, uncertainty, and diagnostics

**Files:**
- Create: `src/mmm_model.py`
- Create: `tests/test_mmm_model.py`

**Interfaces:**
- Consumes `FeatureSet`.
- Produces `MMMConfig`, `ModelMetrics`, `MMMResult`, `fit_mmm(features, config)`, and `predict_from_feature_row(result, row)`.
- `MMMResult.channel_summary` columns: `channel`, `coefficient`, `contribution`, `spend`, `roi`, `ci_low`, `ci_high`.

- [ ] **Step 1: Write a synthetic recovery and reconciliation test**

```python
# tests/test_mmm_model.py
import numpy as np
import pandas as pd
from src.mmm_model import MMMConfig, fit_mmm
from src.transformations import ChannelTransform, build_feature_set

def test_model_produces_holdout_metrics_and_reconciled_contributions():
    rng = np.random.default_rng(7)
    rows = 120
    search = rng.uniform(20, 100, rows)
    social = rng.uniform(10, 60, rows)
    frame = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=rows, freq="W-MON"),
        "media__search": search, "media__social": social,
        "outcome": 500 + 3 * search + 2 * social + rng.normal(0, 10, rows),
    })
    params = {"search": ChannelTransform(0.0, 50.0), "social": ChannelTransform(0.0, 30.0)}
    result = fit_mmm(build_feature_set(frame, params), MMMConfig(bootstrap_samples=10, random_state=7))
    assert result.metrics.holdout_rows == 24
    assert np.isfinite(result.metrics.mae)
    assert set(result.channel_summary["channel"]) == {"search", "social"}
    np.testing.assert_allclose(result.fitted, result.baseline_component + result.channel_contributions.sum(axis=1), atol=1e-6)
    assert result.reliability in {"High", "Medium", "Low"}
```

- [ ] **Step 2: Run the model test and confirm it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_mmm_model.py -v`

Expected: FAIL during collection with missing `src.mmm_model`.

- [ ] **Step 3: Implement the constrained model and result contract**

Implement `src/mmm_model.py` with these exact public types and behavior:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.transformations import FeatureSet

@dataclass(frozen=True)
class MMMConfig:
    holdout_fraction: float = 0.20
    alpha: float = 1.0
    bootstrap_samples: int = 50
    random_state: int = 42

@dataclass(frozen=True)
class ModelMetrics:
    r_squared: float
    mae: float
    mape: float | None
    holdout_rows: int

@dataclass
class MMMResult:
    coefficients: pd.Series
    intercept: float
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    channel_feature_names: dict[str, str]
    transforms: dict
    fitted: pd.Series
    holdout_predictions: pd.Series
    metrics: ModelMetrics
    baseline_component: pd.Series
    channel_contributions: pd.DataFrame
    channel_summary: pd.DataFrame
    response_curves: dict[str, pd.DataFrame]
    reliability: str
    warnings: list[str]

def _fit_constrained(X: pd.DataFrame, y: pd.Series, channel_columns: set[str], alpha: float):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    design = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.sqrt(alpha) * np.eye(design.shape[1]); penalty[0, 0] = 0
    augmented_X = np.vstack([design, penalty])
    augmented_y = np.concatenate([y.to_numpy(), np.zeros(design.shape[1])])
    lower = np.full(design.shape[1], -np.inf); lower[0] = -np.inf
    for index, column in enumerate(X.columns, start=1):
        if column in channel_columns: lower[index] = 0.0
    solved = lsq_linear(augmented_X, augmented_y, bounds=(lower, np.full(design.shape[1], np.inf)))
    return float(solved.x[0]), pd.Series(solved.x[1:], index=X.columns), scaler
```

`fit_mmm` must reserve the final `max(8, round(len(X) * holdout_fraction))` rows, fit on earlier rows for holdout metrics, refit on all rows for final outputs, compute media contribution as standardized feature value times coefficient, absorb all non-media terms plus intercept into `baseline_component`, calculate channel ROI against `FeatureSet.original_media`, and create `response_curves[channel]` data frames with `spend` and `incremental_outcome` columns over 50 points from zero to 1.5 times observed maximum spend. Bootstrap contiguous training-row samples with the configured seed, and assign reliability: High when holdout R² ≥ 0.6 and no interval crosses zero, Medium when R² ≥ 0.2, otherwise Low. Add warnings for negative holdout R², MAPE-invalid zero outcomes, and unstable intervals.

- [ ] **Step 4: Run model tests and the existing smoke test**

Run: `venv\Scripts\python.exe -m pytest tests/test_mmm_model.py -v`

Expected: 1 passed.

Run: `venv\Scripts\python.exe src/smoke_test.py`

Expected: existing project smoke test ends with `SUCCESS - all checks passed.`; allow a longer timeout because DoWhy imports are slow.

- [ ] **Step 5: Commit modeling**

```powershell
git add src/mmm_model.py tests/test_mmm_model.py
git commit -m "feat: add time-aware marketing mix model"
```

### Task 4: Nonlinear budget optimization

**Files:**
- Create: `src/budget_optimizer.py`
- Create: `tests/test_budget_optimizer.py`

**Interfaces:**
- Produces `BudgetResult` and `optimize_allocation(channel_response, total_budget, bounds, current)`.
- `channel_response` maps a channel name to a callable accepting weekly spend and returning modeled incremental outcome.

- [ ] **Step 1: Write feasibility and conservation tests**

```python
# tests/test_budget_optimizer.py
import pytest
from src.budget_optimizer import optimize_allocation

def test_optimizer_conserves_budget_and_prefers_better_response():
    response = {"search": lambda x: 10 * x / (x + 20), "social": lambda x: 4 * x / (x + 20)}
    result = optimize_allocation(response, 100.0, {"search": (0, 100), "social": (0, 100)}, {"search": 50, "social": 50})
    assert result.success
    assert sum(result.optimal_allocation.values()) == pytest.approx(100.0, abs=1e-4)
    assert result.optimal_allocation["search"] > result.optimal_allocation["social"]

def test_optimizer_rejects_infeasible_bounds():
    response = {"a": lambda x: x, "b": lambda x: x}
    result = optimize_allocation(response, 50.0, {"a": (40, 100), "b": (40, 100)}, {"a": 25, "b": 25})
    assert not result.success
    assert "minimum" in result.message.lower()
```

- [ ] **Step 2: Confirm tests fail for missing optimizer**

Run: `venv\Scripts\python.exe -m pytest tests/test_budget_optimizer.py -v`

Expected: FAIL during collection with missing module.

- [ ] **Step 3: Implement the optimizer**

```python
# src/budget_optimizer.py
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from scipy.optimize import minimize

@dataclass
class BudgetResult:
    success: bool
    message: str
    optimal_allocation: dict[str, float]
    current_prediction: float
    optimal_prediction: float

def optimize_allocation(channel_response: dict[str, Callable[[float], float]], total_budget: float,
                        bounds: dict[str, tuple[float, float]], current: dict[str, float]) -> BudgetResult:
    channels = list(channel_response)
    if total_budget <= 0:
        return BudgetResult(False, "Total budget must be positive.", {}, 0.0, 0.0)
    if sum(bounds[name][0] for name in channels) > total_budget:
        return BudgetResult(False, "Channel minimum bounds exceed the total budget.", {}, 0.0, 0.0)
    if sum(bounds[name][1] for name in channels) < total_budget:
        return BudgetResult(False, "Channel maximum bounds cannot absorb the total budget.", {}, 0.0, 0.0)
    x0 = np.array([current.get(name, total_budget / len(channels)) for name in channels], dtype=float)
    x0 *= total_budget / x0.sum()
    objective = lambda x: -sum(channel_response[name](float(value)) for name, value in zip(channels, x))
    solved = minimize(objective, x0, method="SLSQP", bounds=[bounds[name] for name in channels],
                      constraints={"type": "eq", "fun": lambda x: float(x.sum() - total_budget)})
    allocation = {name: float(value) for name, value in zip(channels, solved.x)} if solved.success else {}
    current_prediction = sum(channel_response[name](current.get(name, 0.0)) for name in channels)
    return BudgetResult(bool(solved.success), str(solved.message), allocation, float(current_prediction), float(-solved.fun) if solved.success else 0.0)
```

- [ ] **Step 4: Run optimizer tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_budget_optimizer.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit optimization**

```powershell
git add src/budget_optimizer.py tests/test_budget_optimizer.py
git commit -m "feat: add nonlinear budget optimization"
```

### Task 5: Safe exports and executive report

**Files:**
- Create: `src/reporting.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Produces `ReportContext`, `build_html_report(context) -> str`, and `tables_as_zip(tables) -> bytes`.

- [ ] **Step 1: Write report escaping and ZIP tests**

```python
# tests/test_reporting.py
import pandas as pd
from src.reporting import ReportContext, build_html_report, tables_as_zip

def test_report_escapes_user_columns_and_contains_warnings():
    context = ReportContext("<Sales>", "2024-01-01", "2025-12-31", "Medium", ["Limited history"], pd.DataFrame({"channel": ["search"], "roi": [1.2]}))
    html = build_html_report(context)
    assert "&lt;Sales&gt;" in html and "<Sales>" not in html
    assert "Limited history" in html

def test_table_bundle_is_a_zip():
    payload = tables_as_zip({"summary": pd.DataFrame({"x": [1]})})
    assert payload[:2] == b"PK"
```

- [ ] **Step 2: Verify report tests fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_reporting.py -v`

Expected: FAIL during collection with missing module.

- [ ] **Step 3: Implement HTML and ZIP generation**

```python
# src/reporting.py
from dataclasses import dataclass
from html import escape
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import pandas as pd

@dataclass
class ReportContext:
    outcome_name: str
    start_date: str
    end_date: str
    reliability: str
    warnings: list[str]
    channel_summary: pd.DataFrame

def build_html_report(context: ReportContext) -> str:
    rows = context.channel_summary.to_html(index=False, escape=True, border=0)
    warnings = "".join(f"<li>{escape(item)}</li>" for item in context.warnings) or "<li>None</li>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>MMM Executive Report</title>
<style>body{{font-family:Arial;max-width:960px;margin:40px auto;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>
<body><h1>Marketing Mix Modeling Report</h1><p>Outcome: {escape(context.outcome_name)}</p>
<p>Period: {escape(context.start_date)} to {escape(context.end_date)} | Reliability: {escape(context.reliability)}</p>
<h2>Channel performance</h2>{rows}<h2>Warnings and limitations</h2><ul>{warnings}</ul>
<p>Results are model-based estimates from observational data and are not guaranteed causal effects.</p></body></html>"""

def tables_as_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
    return buffer.getvalue()
```

- [ ] **Step 4: Run report tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_reporting.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit reporting**

```powershell
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: add executive analysis exports"
```

### Task 6: Five-step Streamlit product workflow

**Files:**
- Rewrite: `app/budget_optimizer_app.py`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes all public interfaces from Tasks 1–5.
- Produces a session-local guided UI and a callable `run_demo_analysis()` used by the smoke test.

- [ ] **Step 1: Write an end-to-end demo analysis test**

```python
# tests/test_end_to_end.py
from app.budget_optimizer_app import run_demo_analysis

def test_demo_analysis_reaches_exportable_results():
    analysis = run_demo_analysis(bootstrap_samples=5)
    assert analysis.prepared.can_train
    assert len(analysis.model.channel_summary) >= 2
    assert analysis.report_html.startswith("<!doctype html>")
    assert analysis.tables_zip[:2] == b"PK"
```

- [ ] **Step 2: Run end-to-end test and confirm missing interface**

Run: `venv\Scripts\python.exe -m pytest tests/test_end_to_end.py -v`

Expected: FAIL because `run_demo_analysis` is absent.

- [ ] **Step 3: Refactor the app around a testable analysis function**

Add these exact orchestration types before Streamlit rendering:

```python
from dataclasses import dataclass
from src.data_validation import DataMapping, PreparedData, prepare_and_validate
from src.mmm_model import MMMConfig, MMMResult, fit_mmm
from src.reporting import ReportContext, build_html_report, tables_as_zip
from src.transformations import ChannelTransform, build_feature_set

@dataclass
class AnalysisBundle:
    prepared: PreparedData
    model: MMMResult
    report_html: str
    tables_zip: bytes

def run_demo_analysis(bootstrap_samples: int = 30) -> AnalysisBundle:
    raw = pd.read_csv(ROOT / "data" / "robyn_mmm.csv")
    mapping = DataMapping("DATE", "revenue", ("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"), ("competitor_sales_B",))
    prepared = prepare_and_validate(raw, mapping)
    transforms = {
        channel: ChannelTransform(0.5, max(float(prepared.frame[f"media__{channel}"].median()), 1.0), 1.0)
        for channel in mapping.channel_cols
    }
    model = fit_mmm(build_feature_set(prepared.frame, transforms), MMMConfig(bootstrap_samples=bootstrap_samples))
    context = ReportContext(mapping.outcome_col, str(prepared.frame.date.min().date()), str(prepared.frame.date.max().date()), model.reliability, [*model.warnings, *(issue.message for issue in prepared.issues)], model.channel_summary)
    return AnalysisBundle(prepared, model, build_html_report(context), tables_as_zip({"channel_summary": model.channel_summary}))
```

Then render the UI with five tabs named `1 · Data`, `2 · Validate`, `3 · Model`, `4 · Results`, and `5 · Optimize & Export`. Store the raw data, mapping, and `AnalysisBundle` in `st.session_state`; use `st.file_uploader(..., type=["csv"])`; provide selectors for the four mapping roles; disable training when `PreparedData.can_train` is false; cache model execution with `@st.cache_data(show_spinner=False)`; and label every result as a modeled estimate.

Place Streamlit rendering in `main()` and call it only under `if __name__ == "__main__":` so importing `run_demo_analysis` in tests does not create UI side effects. Build optimizer response callables from each response curve with `lambda spend: float(np.interp(spend, curve["spend"], curve["incremental_outcome"]))`.

Required result views:

- KPI row: reliability, holdout R², MAE, modeled incremental outcome.
- Plotly actual-versus-predicted time series.
- Channel contribution and ROI bar charts with interval columns in the data table.
- Residual-over-time and media-correlation diagnostics.
- Per-channel response curves.
- Optimizer controls with min/max bounds, observed-range warning, and current-versus-recommended allocation chart.
- HTML and ZIP download buttons.
- Persistent privacy statement and observational-data limitation.

- [ ] **Step 4: Run end-to-end and full unit suites**

Run: `venv\Scripts\python.exe -m pytest tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the application workflow**

```powershell
git add app/budget_optimizer_app.py tests/test_end_to_end.py
git commit -m "feat: build guided marketing intelligence studio"
```

### Task 7: Portfolio documentation and Hugging Face readiness

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `.streamlit/config.toml`

**Interfaces:**
- No new Python API; produces reproducible setup and deployment instructions.

- [ ] **Step 1: Pin runtime dependencies and configure Streamlit**

Replace broad application dependency floors with tested compatible ranges and keep notebook-only causal packages in a clearly marked optional section. Add:

```toml
# .streamlit/config.toml
[server]
headless = true
maxUploadSize = 50

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#5B5BD6"
backgroundColor = "#F7F8FC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#172033"
```

Add `streamlit.out.log` and `streamlit.err.log` to `.gitignore`.

- [ ] **Step 2: Rewrite README as a portfolio landing page**

The README must contain: product summary, a `Live demo: deployment pending` status until deployment, screenshots section, problem statement, workflow, methodology, data contract, validation rules, interpretation caveats, architecture, local commands, test command, Hugging Face Docker/Streamlit deployment steps, and three accurate resume bullets. Remove unsupported fixed claims such as guaranteed 18% sales improvement.

- [ ] **Step 3: Run clean-environment-oriented checks**

Run: `venv\Scripts\python.exe -m pip check`

Expected: `No broken requirements found.`

Run: `venv\Scripts\python.exe -m pytest tests -q`

Expected: all tests pass with zero failures.

- [ ] **Step 4: Launch and verify the browser workflow**

Run: `venv\Scripts\python.exe -m streamlit run app/budget_optimizer_app.py --server.headless=true --server.port=8501`

Verify in a browser:

- demo dataset loads;
- all five workflow tabs render;
- validation passes;
- training completes without traceback;
- result charts render;
- optimizer preserves total budget;
- HTML and ZIP downloads are enabled;
- an invalid CSV shows actionable validation errors without crashing.

- [ ] **Step 5: Commit portfolio finish**

```powershell
git add README.md requirements.txt .gitignore .streamlit/config.toml
git commit -m "docs: prepare MMM studio for portfolio deployment"
```

### Task 8: Final regression and acceptance verification

**Files:**
- Modify only files needed for defects proven by this task.

**Interfaces:**
- Verifies the complete product contract.

- [ ] **Step 1: Run every automated test with fresh output**

Run: `venv\Scripts\python.exe -m pytest tests -v`

Expected: zero failed, zero errors.

- [ ] **Step 2: Run legacy compatibility verification**

Run: `venv\Scripts\python.exe src/smoke_test.py`

Expected: final line `SUCCESS - all checks passed.`

- [ ] **Step 3: Check repository scope and generated files**

Run: `git status --short`

Expected: no logs, virtual environments, caches, uploaded data, or model artifacts staged; only intentional project files appear.

- [ ] **Step 4: Record final browser acceptance evidence**

Verify `http://localhost:8501/_stcore/health` returns HTTP 200 and capture a screenshot of the result dashboard for the README. Confirm Streamlit stderr contains no `Traceback`, `Exception`, or `Error` after the full interaction.

- [ ] **Step 5: Commit any verified acceptance fixes separately**

```powershell
git diff --name-only | ForEach-Object { git add -- $_ }
git commit -m "fix: resolve final acceptance defects"
```

Skip this commit when no acceptance defect required a code change.

import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.budget_optimizer import BudgetResult, optimize_allocation
from src.data_validation import DataMapping, PreparedData, ValidationIssue, prepare_and_validate
from src.mmm_model import MMMConfig, MMMResult, fit_mmm
from src.reporting import ReportContext, build_html_report, tables_as_zip
from src.transformations import ChannelTransform, build_feature_set

DEMO_DATA_PATH = ROOT / "data" / "robyn_mmm.csv"
DEMO_MAPPING = DataMapping(
    "DATE",
    "revenue",
    ("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
    ("competitor_sales_B",),
)
TAB_LABELS = [
    "1 · Data",
    "2 · Validate",
    "3 · Model",
    "4 · Results",
    "5 · Optimize & Export",
]
DEFAULT_BOOTSTRAP_SAMPLES = 20
MAX_BOOTSTRAP_SAMPLES = 40
DEFAULT_RANDOM_STATE = 42


@dataclass
class AnalysisBundle:
    prepared: PreparedData
    model: MMMResult
    report_html: str
    tables_zip: bytes


def _load_demo_frame() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA_PATH)


def _demo_csv_bytes() -> bytes:
    return DEMO_DATA_PATH.read_bytes()


def _default_mapping_for_frame(raw: pd.DataFrame) -> DataMapping:
    if set(DEMO_MAPPING.channel_cols).issubset(raw.columns) and {DEMO_MAPPING.date_col, DEMO_MAPPING.outcome_col}.issubset(raw.columns):
        controls = tuple(column for column in DEMO_MAPPING.control_cols if column in raw.columns)
        return DataMapping(DEMO_MAPPING.date_col, DEMO_MAPPING.outcome_col, DEMO_MAPPING.channel_cols, controls)

    columns = list(raw.columns)
    numeric_columns = list(raw.select_dtypes(include=["number"]).columns)

    known_date_names = ("date", "week", "day", "ds", "timestamp", "period")
    date_col = next((column for column in columns if column.lower() in known_date_names), columns[0] if columns else "")

    known_outcome_names = ("revenue", "sales", "outcome", "conversion", "conversions", "orders")
    outcome_col = next(
        (
            column
            for column in numeric_columns
            if column.lower() in known_outcome_names
        ),
        numeric_columns[0] if numeric_columns else "",
    )

    media_keywords = ("spend", "_s", "media", "tv", "search", "facebook", "social", "radio", "ooh", "print", "video")
    candidate_channels = [
        column
        for column in numeric_columns
        if column != outcome_col and any(keyword in column.lower() for keyword in media_keywords)
    ]
    if len(candidate_channels) < 2:
        candidate_channels = [column for column in numeric_columns if column != outcome_col][: min(3, max(0, len(numeric_columns) - 1))]

    control_keywords = ("control", "competitor", "price", "event", "holiday", "distribution", "newsletter")
    candidate_controls = [
        column
        for column in numeric_columns
        if column not in {outcome_col, *candidate_channels}
        and any(keyword in column.lower() for keyword in control_keywords)
    ]

    return DataMapping(date_col, outcome_col, tuple(candidate_channels), tuple(candidate_controls))


def _default_transforms(prepared: PreparedData) -> dict[str, ChannelTransform]:
    transforms: dict[str, ChannelTransform] = {}
    for channel in prepared.mapping.channel_cols:
        spend_series = prepared.frame[f"media__{channel}"]
        transforms[channel] = ChannelTransform(
            decay=0.50,
            half_saturation=max(float(spend_series.median()), 1.0),
            slope=1.00,
        )
    return transforms


def _serialize_mapping(mapping: DataMapping) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    return mapping.date_col, mapping.outcome_col, tuple(mapping.channel_cols), tuple(mapping.control_cols)


def _deserialize_mapping(payload: tuple[str, str, tuple[str, ...], tuple[str, ...]]) -> DataMapping:
    date_col, outcome_col, channel_cols, control_cols = payload
    return DataMapping(date_col, outcome_col, tuple(channel_cols), tuple(control_cols))


def _serialize_transforms(transforms: dict[str, ChannelTransform]) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        (channel, config.decay, config.half_saturation, config.slope)
        for channel, config in sorted(transforms.items())
    )


def _deserialize_transforms(payload: tuple[tuple[str, float, float, float], ...]) -> dict[str, ChannelTransform]:
    return {
        channel: ChannelTransform(decay=decay, half_saturation=half_saturation, slope=slope)
        for channel, decay, half_saturation, slope in payload
    }


def _serialize_config(config: MMMConfig) -> tuple[float, float, int, int]:
    return config.holdout_fraction, config.alpha, config.bootstrap_samples, config.random_state


def _deserialize_config(payload: tuple[float, float, int, int]) -> MMMConfig:
    holdout_fraction, alpha, bootstrap_samples, random_state = payload
    return MMMConfig(
        holdout_fraction=float(holdout_fraction),
        alpha=float(alpha),
        bootstrap_samples=int(bootstrap_samples),
        random_state=int(random_state),
    )


def _issue_messages(issues: list[ValidationIssue]) -> list[str]:
    return [issue.message for issue in issues]


def _unique_messages(messages: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = message.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _build_export_tables(prepared: PreparedData, model: MMMResult) -> dict[str, pd.DataFrame]:
    features = build_feature_set(prepared.frame, model.transforms)
    residuals = features.y - model.fitted
    holdout_rows = model.metrics.holdout_rows
    holdout_index = prepared.frame.index[-holdout_rows:]

    timeseries = pd.DataFrame(
        {
            "date": prepared.frame["date"],
            "actual_outcome": features.y,
            "modeled_fit": model.fitted,
            "baseline_component": model.baseline_component,
            "residual": residuals,
            "is_holdout": prepared.frame.index.isin(holdout_index),
        }
    )
    holdout_predictions = pd.DataFrame(
        {
            "date": prepared.frame.loc[holdout_index, "date"],
            "holdout_actual_outcome": features.y.loc[holdout_index].to_numpy(),
            "holdout_prediction": model.holdout_predictions.to_numpy(),
        }
    )
    contribution_table = model.channel_contributions.copy()
    contribution_table.insert(0, "date", prepared.frame["date"])
    response_curves = pd.concat(
        [
            curve.assign(channel=channel)
            for channel, curve in model.response_curves.items()
        ],
        ignore_index=True,
    )
    media_correlation = (
        prepared.frame.filter(regex=r"^media__")
        .rename(columns=lambda value: value.replace("media__", ""))
        .corr()
        .reset_index()
        .rename(columns={"index": "channel"})
    )
    validation_issues = pd.DataFrame(
        [
            {"severity": issue.severity, "code": issue.code, "message": issue.message}
            for issue in prepared.issues
        ]
    )
    model_warnings = pd.DataFrame({"warning": model.warnings})

    return {
        "channel_summary": model.channel_summary,
        "modeled_timeseries": timeseries,
        "holdout_predictions": holdout_predictions,
        "channel_contributions": contribution_table,
        "response_curves": response_curves,
        "media_correlation": media_correlation,
        "validation_issues": validation_issues,
        "model_warnings": model_warnings,
    }


def _build_analysis_bundle(raw: pd.DataFrame, mapping: DataMapping, transforms: dict[str, ChannelTransform], config: MMMConfig) -> AnalysisBundle:
    prepared = prepare_and_validate(raw, mapping)
    if not prepared.can_train:
        messages = "; ".join(_issue_messages(prepared.issues))
        raise ValueError(f"Training is blocked until validation issues are resolved: {messages}")

    features = build_feature_set(prepared.frame, transforms)
    model = fit_mmm(features, config)
    warnings = _unique_messages([*_issue_messages(prepared.issues), *model.warnings])
    context = ReportContext(
        outcome_name=mapping.outcome_col,
        start_date=str(prepared.frame["date"].min().date()),
        end_date=str(prepared.frame["date"].max().date()),
        reliability=model.reliability,
        warnings=warnings,
        channel_summary=model.channel_summary,
    )
    export_tables = _build_export_tables(prepared, model)
    return AnalysisBundle(
        prepared=prepared,
        model=model,
        report_html=build_html_report(context),
        tables_zip=tables_as_zip(export_tables),
    )


@st.cache_data(show_spinner=False)
def _cached_demo_analysis_bundle(
    raw: pd.DataFrame,
    mapping_payload: tuple[str, str, tuple[str, ...], tuple[str, ...]],
    transform_payload: tuple[tuple[str, float, float, float], ...],
    config_payload: tuple[float, float, int, int],
) -> AnalysisBundle:
    mapping = _deserialize_mapping(mapping_payload)
    transforms = _deserialize_transforms(transform_payload)
    config = _deserialize_config(config_payload)
    return _build_analysis_bundle(raw.copy(), mapping, transforms, config)


def _fingerprint_bytes(payload: bytes) -> str:
    """Return a stable, non-reversible identifier for in-memory source data."""
    return hashlib.sha256(payload).hexdigest()


def _mapping_signature(mapping: DataMapping) -> str:
    return repr(_serialize_mapping(mapping))


def _analysis_cache_key(
    source_fingerprint: str,
    mapping: DataMapping,
    transforms: dict[str, ChannelTransform],
    config: MMMConfig,
) -> str:
    payload = repr(
        (
            source_fingerprint,
            _serialize_mapping(mapping),
            _serialize_transforms(transforms),
            _serialize_config(config),
        )
    ).encode("utf-8")
    return _fingerprint_bytes(payload)


def _clear_results(state) -> None:
    state["analysis"] = None
    state["analysis_signature"] = None
    state["optimizer_result"] = None
    state["optimizer_signature"] = None


def _sync_source_state(
    state,
    raw: pd.DataFrame,
    *,
    source_kind: str,
    source_label: str,
    source_fingerprint: str,
) -> bool:
    """Update source state only when the underlying content changes."""
    unchanged = (
        state.get("source_kind") == source_kind
        and state.get("source_fingerprint") == source_fingerprint
    )
    if unchanged:
        return False

    mapping = _default_mapping_for_frame(raw)
    state["raw_data"] = raw.copy()
    state["source_kind"] = source_kind
    state["source_label"] = source_label
    state["data_source"] = source_label
    state["source_fingerprint"] = source_fingerprint
    state["mapping_selection"] = mapping
    state["mapping_signature"] = _mapping_signature(mapping)
    state["uploaded_analysis_cache"] = {}
    _clear_results(state)
    return True


def _sync_mapping_state(state, mapping: DataMapping) -> bool:
    signature = _mapping_signature(mapping)
    if state.get("mapping_signature") == signature:
        state["mapping_selection"] = mapping
        return False
    state["mapping_selection"] = mapping
    state["mapping_signature"] = signature
    _clear_results(state)
    return True


def _get_or_create_analysis(
    state,
    raw: pd.DataFrame,
    *,
    source_kind: str,
    source_fingerprint: str,
    mapping: DataMapping,
    transforms: dict[str, ChannelTransform],
    config: MMMConfig,
) -> AnalysisBundle:
    key = _analysis_cache_key(source_fingerprint, mapping, transforms, config)
    if source_kind == "upload":
        cache = state.setdefault("uploaded_analysis_cache", {})
        if key not in cache:
            cache[key] = _build_analysis_bundle(raw.copy(), mapping, transforms, config)
        return cache[key]

    return _cached_demo_analysis_bundle(
        raw,
        _serialize_mapping(mapping),
        _serialize_transforms(transforms),
        _serialize_config(config),
    )


def _training_gate_state(prepared: PreparedData) -> dict[str, object]:
    if prepared.can_train:
        return {"disabled": False, "help": "Validation passed; training is available."}
    return {
        "disabled": True,
        "help": "Training is disabled until validation errors are resolved.",
    }


def run_demo_analysis(bootstrap_samples: int = 30) -> AnalysisBundle:
    raw = _load_demo_frame()
    prepared = prepare_and_validate(raw, DEMO_MAPPING)
    transforms = _default_transforms(prepared)
    config = MMMConfig(bootstrap_samples=int(bootstrap_samples), random_state=DEFAULT_RANDOM_STATE)
    return _build_analysis_bundle(raw, DEMO_MAPPING, transforms, config)


def _analysis_features(analysis: AnalysisBundle):
    return build_feature_set(analysis.prepared.frame, analysis.model.transforms)


def _current_average_allocation(analysis: AnalysisBundle) -> dict[str, float]:
    current: dict[str, float] = {}
    for channel in analysis.prepared.mapping.channel_cols:
        current[channel] = float(analysis.prepared.frame[f"media__{channel}"].mean())
    return current


def _observed_bounds(analysis: AnalysisBundle) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for channel in analysis.prepared.mapping.channel_cols:
        spend = analysis.prepared.frame[f"media__{channel}"]
        bounds[channel] = (float(spend.min()), float(spend.max()))
    return bounds


def _response_functions(model: MMMResult):
    response: dict[str, callable] = {}
    for channel, curve in model.response_curves.items():
        response[channel] = (
            lambda spend, curve=curve: float(np.interp(spend, curve["spend"], curve["incremental_outcome"]))
        )
    return response


def _safe_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _ensure_session_defaults() -> None:
    if "raw_data" not in st.session_state:
        _sync_source_state(
            st.session_state,
            _load_demo_frame(),
            source_kind="demo",
            source_label="Bundled demo dataset",
            source_fingerprint=_fingerprint_bytes(_demo_csv_bytes()),
        )
    st.session_state.setdefault("mapping_signature", _mapping_signature(st.session_state.mapping_selection))
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("analysis_signature", None)
    st.session_state.setdefault("optimizer_result", None)
    st.session_state.setdefault("optimizer_signature", None)
    st.session_state.setdefault("uploaded_analysis_cache", {})


def _select_data_source() -> None:
    raw = st.session_state.raw_data
    st.markdown(
        "Use the bundled demo to explore the full workflow immediately, or upload a weekly marketing CSV. "
        "Uploaded data stays in this Streamlit session."
    )
    action_col, download_col = st.columns([1, 1])
    with action_col:
        if st.button("Use bundled demo dataset", type="primary", width="stretch"):
            changed = _sync_source_state(
                st.session_state,
                _load_demo_frame(),
                source_kind="demo",
                source_label="Bundled demo dataset",
                source_fingerprint=_fingerprint_bytes(_demo_csv_bytes()),
            )
            if changed:
                st.rerun()
    with download_col:
        st.download_button(
            "Download CSV template",
            data=_demo_csv_bytes(),
            file_name="marketing_mix_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded_file = st.file_uploader(
        "Upload a CSV (session-local only)",
        type=["csv"],
        help="The file is read into memory for this session and is not persisted by the app.",
    )
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        fingerprint = _fingerprint_bytes(upload_bytes)
        try:
            parsed = pd.read_csv(io.BytesIO(upload_bytes))
        except Exception as exc:  # noqa: BLE001
            st.error("We couldn't read that file as a CSV. Please upload a valid comma-separated file with a header row.")
            with st.expander("Technical details"):
                st.exception(exc)
        else:
            _sync_source_state(
                st.session_state,
                parsed,
                source_kind="upload",
                source_label=f"Uploaded file: {uploaded_file.name}",
                source_fingerprint=fingerprint,
            )
            raw = st.session_state.raw_data

    st.caption(f"Current source: {st.session_state.source_label}")
    st.dataframe(raw.head(12), hide_index=True, width="stretch")
    st.caption(f"{len(raw):,} rows × {len(raw.columns):,} columns")


def _mapping_editor(raw: pd.DataFrame) -> DataMapping | None:
    default_mapping = st.session_state.mapping_selection
    all_columns = list(raw.columns)
    numeric_columns = list(raw.select_dtypes(include=["number"]).columns)

    if not all_columns or len(numeric_columns) < 3:
        st.error("Upload a CSV with at least one date column, one outcome column, and two numeric media columns.")
        return None

    left, right = st.columns(2)
    with left:
        date_col = st.selectbox(
            "Date column",
            all_columns,
            index=_safe_index(all_columns, default_mapping.date_col),
            help="Choose the weekly date column, or a date column that can be aggregated to weeks.",
        )
        outcome_col = st.selectbox(
            "Outcome column",
            numeric_columns,
            index=_safe_index(numeric_columns, default_mapping.outcome_col),
            help="Pick the business outcome to model, such as revenue, sales, or conversions.",
        )
    with right:
        default_channels = [column for column in default_mapping.channel_cols if column in numeric_columns]
        channel_cols = st.multiselect(
            "Media spend columns",
            numeric_columns,
            default=default_channels,
            help="Select at least two non-negative media spend columns.",
        )
        control_options = [column for column in numeric_columns if column not in {outcome_col, *channel_cols}]
        default_controls = [column for column in default_mapping.control_cols if column in control_options]
        control_cols = st.multiselect(
            "Optional control columns",
            control_options,
            default=default_controls,
            help="Optional controls help absorb non-media drivers such as price, competition, or events.",
        )

    mapping = DataMapping(date_col, outcome_col, tuple(channel_cols), tuple(control_cols))
    _sync_mapping_state(st.session_state, mapping)
    return mapping


def _render_validation_summary(prepared: PreparedData) -> None:
    error_issues = [issue for issue in prepared.issues if issue.severity == "error"]
    warning_issues = [issue for issue in prepared.issues if issue.severity == "warning"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Usable weekly rows", f"{len(prepared.frame):,}")
    metric_columns[1].metric("Selected media channels", len(prepared.mapping.channel_cols))
    metric_columns[2].metric("Blocking issues", len(error_issues))
    metric_columns[3].metric("Warnings", len(warning_issues))

    if prepared.can_train:
        st.success("Validation passed. Training is enabled for this mapped dataset.")
    else:
        st.error("Validation found blocking issues. Resolve the errors below before training the model.")

    for issue in error_issues:
        st.error(issue.message)
    for issue in warning_issues:
        st.warning(issue.message)

    if not prepared.frame.empty:
        st.dataframe(prepared.frame.head(12), hide_index=True, width="stretch")


def _render_model_tab(raw: pd.DataFrame, mapping: DataMapping | None) -> None:
    if mapping is None:
        st.info("Choose a dataset and map the required columns in the Validate step.")
        return

    prepared = prepare_and_validate(raw, mapping)
    _render_validation_summary(prepared)
    st.markdown("---")

    gate = _training_gate_state(prepared)
    if bool(gate["disabled"]):
        st.button(
            "Train model",
            type="primary",
            disabled=True,
            width="stretch",
            help=str(gate["help"]),
        )
        return

    default_transforms = _default_transforms(prepared)
    with st.expander("Advanced modeling settings", expanded=False):
        holdout_fraction = st.slider(
            "Holdout fraction",
            min_value=0.10,
            max_value=0.35,
            value=0.20,
            step=0.05,
            help="The most recent rows are held out to evaluate reliability before refitting on the full series.",
        )
        alpha = st.number_input(
            "Ridge regularization strength",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Higher values dampen unstable coefficients when channels move together.",
        )
        bootstrap_samples = st.slider(
            "Bootstrap refits",
            min_value=0,
            max_value=MAX_BOOTSTRAP_SAMPLES,
            value=DEFAULT_BOOTSTRAP_SAMPLES,
            step=5,
            help="Bounded to keep runtime practical on local and free-hosted deployments.",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9_999,
            value=DEFAULT_RANDOM_STATE,
            step=1,
        )

        transform_columns = st.columns(3)
        transform_payload: dict[str, ChannelTransform] = {}
        for index, channel in enumerate(prepared.mapping.channel_cols):
            defaults = default_transforms[channel]
            with transform_columns[index % 3]:
                st.markdown(f"**{channel}**")
                decay = st.slider(
                    f"{channel} decay",
                    min_value=0.0,
                    max_value=0.95,
                    value=float(defaults.decay),
                    step=0.05,
                )
                half_saturation = st.number_input(
                    f"{channel} half-saturation",
                    min_value=0.1,
                    value=float(defaults.half_saturation),
                    step=max(float(defaults.half_saturation) / 10.0, 0.1),
                )
                slope = st.slider(
                    f"{channel} curve slope",
                    min_value=0.5,
                    max_value=2.0,
                    value=float(defaults.slope),
                    step=0.1,
                )
                transform_payload[channel] = ChannelTransform(decay=decay, half_saturation=half_saturation, slope=slope)

    config = MMMConfig(
        holdout_fraction=float(holdout_fraction),
        alpha=float(alpha),
        bootstrap_samples=int(bootstrap_samples),
        random_state=int(random_state),
    )

    requested_signature = _analysis_cache_key(
        st.session_state.source_fingerprint,
        mapping,
        transform_payload,
        config,
    )
    if (
        st.session_state.analysis is not None
        and st.session_state.analysis_signature != requested_signature
    ):
        _clear_results(st.session_state)

    if st.button("Train model", type="primary", width="stretch"):
        with st.spinner("Training the model and estimating uncertainty..."):
            try:
                st.session_state.analysis = _get_or_create_analysis(
                    st.session_state,
                    raw,
                    source_kind=st.session_state.source_kind,
                    source_fingerprint=st.session_state.source_fingerprint,
                    mapping=mapping,
                    transforms=transform_payload,
                    config=config,
                )
                st.session_state.analysis_signature = requested_signature
                st.session_state.optimizer_result = None
                st.session_state.optimizer_signature = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.analysis = None
                st.error("Training could not complete with the current data and settings. Review the validation messages and advanced parameters, then try again.")
                with st.expander("Technical details"):
                    st.exception(exc)
                return
        st.success("Model training complete. Review the modeled estimates in the Results and Optimize tabs.")

    analysis = st.session_state.analysis
    if analysis is not None:
        st.info(
            f"Latest trained model: reliability {analysis.model.reliability}, "
            f"holdout R² {analysis.model.metrics.r_squared:.2f}, "
            f"MAE {analysis.model.metrics.mae:,.0f}."
        )


def _render_results_tab(analysis: AnalysisBundle | None) -> None:
    if analysis is None:
        st.info("Train the model in Step 3 to unlock diagnostics, response curves, and modeled ROI.")
        return

    features = _analysis_features(analysis)
    modeled_incremental = float(analysis.model.channel_summary["contribution"].sum())
    residuals = features.y - analysis.model.fitted
    holdout_rows = analysis.model.metrics.holdout_rows
    holdout_dates = analysis.prepared.frame["date"].iloc[-holdout_rows:]

    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Reliability", analysis.model.reliability)
    kpi_columns[1].metric("Holdout R²", f"{analysis.model.metrics.r_squared:.2f}")
    kpi_columns[2].metric("Holdout MAE", f"{analysis.model.metrics.mae:,.0f}")
    kpi_columns[3].metric("Modeled media contribution", f"{modeled_incremental:,.0f}")
    st.caption("All outputs below are modeled estimates from observational data, not experimentally verified lift.")

    timeseries = pd.DataFrame(
        {
            "date": analysis.prepared.frame["date"],
            "Actual outcome": features.y,
            "Modeled fit": analysis.model.fitted,
        }
    ).melt(id_vars="date", var_name="Series", value_name="Outcome")
    timeseries_fig = px.line(
        timeseries,
        x="date",
        y="Outcome",
        color="Series",
        title="Actual vs modeled outcome over time",
    )
    timeseries_fig.add_trace(
        go.Scatter(
            x=holdout_dates,
            y=analysis.model.holdout_predictions,
            mode="lines",
            name="Holdout prediction",
            line={"dash": "dash"},
        )
    )
    st.plotly_chart(timeseries_fig, width="stretch")

    summary = analysis.model.channel_summary.copy()
    summary["ci_range"] = summary.apply(lambda row: f"{row['ci_low']:.0f} to {row['ci_high']:.0f}", axis=1)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        contribution_fig = px.bar(
            summary,
            x="channel",
            y="contribution",
            title="Modeled channel contribution",
            labels={"channel": "Channel", "contribution": "Incremental modeled outcome"},
        )
        st.plotly_chart(contribution_fig, width="stretch")
    with chart_columns[1]:
        roi_fig = px.bar(
            summary,
            x="channel",
            y="roi",
            title="Modeled ROI by channel",
            labels={"channel": "Channel", "roi": "Modeled ROI"},
        )
        st.plotly_chart(roi_fig, width="stretch")

    st.dataframe(
        summary.loc[:, ["channel", "spend", "contribution", "roi", "ci_low", "ci_high", "ci_range"]],
        hide_index=True,
        width="stretch",
    )

    diagnostics_columns = st.columns(2)
    with diagnostics_columns[0]:
        residual_fig = px.line(
            x=analysis.prepared.frame["date"],
            y=residuals,
            title="Residuals over time",
            labels={"x": "Date", "y": "Actual minus modeled fit"},
        )
        residual_fig.add_hline(y=0.0, line_dash="dot")
        st.plotly_chart(residual_fig, width="stretch")
    with diagnostics_columns[1]:
        correlation = (
            analysis.prepared.frame.filter(regex=r"^media__")
            .rename(columns=lambda value: value.replace("media__", ""))
            .corr()
        )
        heatmap = go.Figure(
            data=go.Heatmap(
                z=correlation.to_numpy(),
                x=correlation.columns,
                y=correlation.index,
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                colorbar={"title": "Correlation"},
            )
        )
        heatmap.update_layout(title="Media spend correlation diagnostic")
        st.plotly_chart(heatmap, width="stretch")

    response_curves = pd.concat(
        [
            curve.assign(channel=channel)
            for channel, curve in analysis.model.response_curves.items()
        ],
        ignore_index=True,
    )
    response_fig = px.line(
        response_curves,
        x="spend",
        y="incremental_outcome",
        color="channel",
        title="Response curves by channel",
        labels={"spend": "Weekly spend", "incremental_outcome": "Modeled incremental outcome"},
    )
    st.plotly_chart(response_fig, width="stretch")

    if analysis.model.warnings:
        for warning in analysis.model.warnings:
            st.warning(warning)


def _optimizer_validation(total_budget: float, bounds: dict[str, tuple[float, float]], observed: dict[str, tuple[float, float]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if total_budget <= 0:
        errors.append("Set a positive weekly budget before optimization.")

    if sum(lower for lower, _ in bounds.values()) > total_budget:
        errors.append("Channel minimum bounds exceed the selected total budget.")
    if sum(upper for _, upper in bounds.values()) < total_budget:
        errors.append("Channel maximum bounds cannot absorb the selected total budget.")

    observed_budget_min = sum(lower for lower, _ in observed.values())
    observed_budget_max = sum(upper for _, upper in observed.values())
    if total_budget < observed_budget_min or total_budget > observed_budget_max:
        warnings.append(
            "The requested total budget sits outside the observed weekly spend range, so response-curve extrapolation is less reliable."
        )

    for channel, (lower, upper) in bounds.items():
        if lower > upper:
            errors.append(f"{channel}: minimum bound must be less than or equal to the maximum bound.")
        observed_lower, observed_upper = observed[channel]
        if lower < observed_lower or upper > observed_upper:
            warnings.append(
                f"{channel}: at least one bound is outside the observed spend range ({observed_lower:,.0f} to {observed_upper:,.0f})."
            )

    return errors, warnings


def _render_optimize_tab(analysis: AnalysisBundle | None) -> None:
    if analysis is None:
        st.info("Train the model in Step 3 before optimizing a budget plan or downloading exports.")
        return

    current = _current_average_allocation(analysis)
    observed = _observed_bounds(analysis)
    current_total = float(sum(current.values()))

    st.download_button(
        "Download HTML executive report",
        data=analysis.report_html,
        file_name="mmm_executive_report.html",
        mime="text/html",
        width="stretch",
    )
    st.download_button(
        "Download result tables (ZIP)",
        data=analysis.tables_zip,
        file_name="mmm_result_tables.zip",
        mime="application/zip",
        width="stretch",
    )

    st.markdown("---")
    st.subheader("Budget optimizer")
    st.caption("The optimizer uses fitted response curves to recommend a feasible weekly allocation under your chosen budget and bounds.")

    total_budget = st.number_input(
        "Future weekly budget",
        min_value=0.0,
        value=float(round(current_total, 2)),
        step=max(round(current_total / 20.0, 2), 1.0),
    )

    bounds: dict[str, tuple[float, float]] = {}
    bound_columns = st.columns(2)
    for index, channel in enumerate(analysis.prepared.mapping.channel_cols):
        observed_low, observed_high = observed[channel]
        with bound_columns[index % 2]:
            st.markdown(f"**{channel}**")
            lower = st.number_input(
                f"{channel} minimum",
                min_value=0.0,
                value=float(round(observed_low, 2)),
                step=max(round(observed_high / 20.0, 2), 1.0),
            )
            upper = st.number_input(
                f"{channel} maximum",
                min_value=0.0,
                value=float(round(max(observed_high, lower), 2)),
                step=max(round(observed_high / 20.0, 2), 1.0),
            )
            bounds[channel] = (float(lower), float(upper))

    errors, warnings = _optimizer_validation(total_budget, bounds, observed)
    for error in errors:
        st.error(error)
    for warning in warnings:
        st.warning(warning)

    if st.button("Optimize allocation", type="primary", disabled=bool(errors), width="stretch"):
        response = _response_functions(analysis.model)
        st.session_state.optimizer_result = optimize_allocation(response, float(total_budget), bounds, current)

    result: BudgetResult | None = st.session_state.optimizer_result
    if result is None:
        st.info("Click Optimize allocation to compare the current average plan with the recommended allocation.")
        return

    if not result.success:
        st.error(result.message)
        return

    lift = result.optimal_prediction - result.current_prediction
    lift_pct = (lift / result.current_prediction * 100.0) if result.current_prediction else np.nan
    summary_columns = st.columns(3)
    summary_columns[0].metric("Current modeled outcome", f"{result.current_prediction:,.0f}")
    summary_columns[1].metric("Recommended modeled outcome", f"{result.optimal_prediction:,.0f}", delta=f"{lift:,.0f}")
    summary_columns[2].metric("Modeled change", f"{lift_pct:.1f}%")

    comparison = pd.DataFrame(
        {
            "channel": list(result.optimal_allocation),
            "Current average": [current[channel] for channel in result.optimal_allocation],
            "Recommended": [result.optimal_allocation[channel] for channel in result.optimal_allocation],
        }
    )
    comparison_long = comparison.melt(id_vars="channel", var_name="Scenario", value_name="Spend")
    comparison_fig = px.bar(
        comparison_long,
        x="channel",
        y="Spend",
        color="Scenario",
        barmode="group",
        title="Current vs recommended weekly allocation",
    )
    st.plotly_chart(comparison_fig, width="stretch")
    st.dataframe(comparison, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Marketing Intelligence Studio",
        page_icon="📈",
        layout="wide",
    )
    _ensure_session_defaults()

    st.title("Marketing Intelligence Studio")
    st.caption(
        "A five-step marketing mix workflow for demo data or a session-local CSV upload. "
        "Outputs are framed as modeled estimates with uncertainty, not guaranteed causal truth."
    )

    tabs = st.tabs(TAB_LABELS)
    raw = st.session_state.raw_data

    with tabs[0]:
        _select_data_source()
        raw = st.session_state.raw_data

    with tabs[1]:
        mapping = _mapping_editor(raw)
        if mapping is not None:
            prepared = prepare_and_validate(raw, mapping)
            _render_validation_summary(prepared)
        else:
            prepared = None

    with tabs[2]:
        _render_model_tab(raw, st.session_state.mapping_selection)

    with tabs[3]:
        _render_results_tab(st.session_state.analysis)

    with tabs[4]:
        _render_optimize_tab(st.session_state.analysis)

    st.markdown("---")
    st.caption("Privacy: uploaded CSVs remain in this browser session and are not intentionally written to disk by this app.")
    st.info(
        "Interpretation caveat: these are model-based estimates from observational data. "
        "They are useful for planning, but they do not prove causal effects the way a randomized experiment would."
    )


if __name__ == "__main__":
    main()

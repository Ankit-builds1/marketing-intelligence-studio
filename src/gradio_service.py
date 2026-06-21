"""Framework-independent service layer for the hosted Gradio interface."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.budget_optimizer import optimize_allocation
from src.data_validation import DataMapping, PreparedData, prepare_and_validate
from src.mmm_model import MMMConfig, MMMResult, fit_mmm
from src.reporting import ReportContext, build_html_report, tables_as_zip
from src.transformations import ChannelTransform, build_feature_set

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_PATH = ROOT / "data" / "robyn_mmm.csv"
MAX_BOOTSTRAP_SAMPLES = 20


@dataclass
class AnalysisBundle:
    prepared: PreparedData
    model: MMMResult
    report_html: str
    tables_zip: bytes


@dataclass
class GradioSession:
    bundle: AnalysisBundle
    temp_dir: str


@dataclass
class AnalysisOutput:
    status: str
    channel_summary: pd.DataFrame
    fit_figure: go.Figure
    channel_figure: go.Figure
    response_figure: go.Figure
    session: GradioSession | None
    html_path: str | None
    zip_path: str | None
    suggested_budget: float


def _file_path(file_value: Any) -> str | None:
    if file_value is None or file_value == "":
        return None
    if isinstance(file_value, (str, Path)):
        return str(file_value)
    return str(getattr(file_value, "name", file_value))


def load_source(file_path: Any) -> tuple[pd.DataFrame, str]:
    resolved = _file_path(file_path)
    if resolved is None:
        return pd.read_csv(DEMO_DATA_PATH), "Bundled demo dataset"
    return pd.read_csv(resolved), f"Uploaded file: {Path(resolved).name}"


def suggest_mapping(frame: pd.DataFrame) -> DataMapping:
    columns = list(frame.columns)
    numeric = list(frame.select_dtypes(include=["number"]).columns)
    if not columns:
        return DataMapping("", "", (), ())
    date_names = {"date", "week", "day", "ds", "timestamp", "period"}
    date_col = next((c for c in columns if c.lower() in date_names), columns[0])
    outcome_names = {"revenue", "sales", "outcome", "orders", "conversions"}
    outcome_col = next(
        (c for c in numeric if c.lower() in outcome_names),
        numeric[0] if numeric else "",
    )
    # Prefer explicit spend naming. Broad platform words alone can describe
    # impressions/clicks and must not silently become budget channels.
    channels = [
        c for c in numeric
        if c != outcome_col
        and ("spend" in c.lower() or c.lower().endswith("_s"))
    ]
    if len(channels) < 2:
        media_keywords = ("media", "tv", "social", "radio", "ooh", "print", "video")
        excluded = ("click", "impression", "competitor", "revenue", "sales")
        channels = [
            c for c in numeric
            if c != outcome_col
            and any(key in c.lower() for key in media_keywords)
            and not any(key in c.lower() for key in excluded)
        ]
    if len(channels) < 2:
        channels = [c for c in numeric if c != outcome_col][:3]
    control_keywords = (
        "control", "competitor", "price", "event", "holiday", "distribution",
    )
    controls = [
        c for c in numeric
        if c not in {outcome_col, *channels}
        and any(key in c.lower() for key in control_keywords)
    ]
    return DataMapping(date_col, outcome_col, tuple(channels), tuple(controls))


def _empty_output(status: str) -> AnalysisOutput:
    return AnalysisOutput(
        status, pd.DataFrame(), go.Figure(), go.Figure(), go.Figure(),
        None, None, None, 0.0,
    )


def _default_transforms(prepared: PreparedData) -> dict[str, ChannelTransform]:
    return {
        channel: ChannelTransform(
            decay=0.5,
            half_saturation=max(
                float(prepared.frame[f"media__{channel}"].median()), 1.0
            ),
            slope=1.0,
        )
        for channel in prepared.mapping.channel_cols
    }


def _export_tables(prepared: PreparedData, model: MMMResult) -> dict[str, pd.DataFrame]:
    features = build_feature_set(prepared.frame, model.transforms)
    timeseries = pd.DataFrame(
        {
            "date": prepared.frame["date"],
            "actual_outcome": features.y,
            "modeled_fit": model.fitted,
            "baseline_component": model.baseline_component,
            "residual": features.y - model.fitted,
        }
    )
    curves = pd.concat(
        [curve.assign(channel=channel) for channel, curve in model.response_curves.items()],
        ignore_index=True,
    )
    return {
        "channel_summary": model.channel_summary,
        "modeled_timeseries": timeseries,
        "channel_contributions": model.channel_contributions,
        "response_curves": curves,
    }


def analyze_dataset(
    file_path: Any,
    date_col: str,
    outcome_col: str,
    channel_cols: list[str] | tuple[str, ...] | None,
    control_cols: list[str] | tuple[str, ...] | None,
    bootstrap_samples: int = 10,
) -> AnalysisOutput:
    channels = tuple(channel_cols or ())
    controls = tuple(control_cols or ())
    if len(channels) < 2:
        return _empty_output("Validation blocked: select at least two media channels.")
    try:
        raw, label = load_source(file_path)
        mapping = DataMapping(date_col, outcome_col, channels, controls)
        prepared = prepare_and_validate(raw, mapping)
        if not prepared.can_train:
            errors = [i.message for i in prepared.issues if i.severity == "error"]
            return _empty_output("Validation blocked: " + "; ".join(errors))

        transforms = _default_transforms(prepared)
        model = fit_mmm(
            build_feature_set(prepared.frame, transforms),
            MMMConfig(
                bootstrap_samples=min(
                    max(int(bootstrap_samples), 0), MAX_BOOTSTRAP_SAMPLES
                ),
                random_state=42,
            ),
        )
        warnings = [i.message for i in prepared.issues] + model.warnings
        context = ReportContext(
            outcome_name=outcome_col,
            start_date=str(prepared.frame["date"].min().date()),
            end_date=str(prepared.frame["date"].max().date()),
            reliability=model.reliability,
            warnings=list(dict.fromkeys(warnings)),
            channel_summary=model.channel_summary,
        )
        bundle = AnalysisBundle(
            prepared,
            model,
            build_html_report(context),
            tables_as_zip(_export_tables(prepared, model)),
        )
        temp_dir = tempfile.mkdtemp(prefix="mmm-gradio-")
        html_path = Path(temp_dir) / "mmm_executive_report.html"
        zip_path = Path(temp_dir) / "mmm_result_tables.zip"
        html_path.write_text(bundle.report_html, encoding="utf-8")
        zip_path.write_bytes(bundle.tables_zip)

        features = build_feature_set(prepared.frame, transforms)
        fit_frame = pd.DataFrame(
            {"date": prepared.frame["date"], "Actual": features.y, "Modeled": model.fitted}
        ).melt(id_vars="date", var_name="Series", value_name="Outcome")
        fit_figure = px.line(
            fit_frame, x="date", y="Outcome", color="Series",
            title="Actual vs modeled outcome",
        )
        channel_frame = model.channel_summary[
            ["channel", "contribution", "roi"]
        ].melt(id_vars="channel", var_name="Metric", value_name="Value")
        channel_figure = px.bar(
            channel_frame, x="channel", y="Value", color="Metric", barmode="group",
            title="Modeled channel contribution and ROI",
        )
        response_frame = pd.concat(
            [curve.assign(channel=channel) for channel, curve in model.response_curves.items()],
            ignore_index=True,
        )
        response_figure = px.line(
            response_frame, x="spend", y="incremental_outcome", color="channel",
            title="Channel response curves",
        )
        budget = float(
            sum(prepared.frame[f"media__{channel}"].mean() for channel in channels)
        )
        warning_text = f" Warnings: {'; '.join(warnings)}" if warnings else ""
        status = (
            f"Analysis complete for {label}. Reliability: {model.reliability}; "
            f"holdout R²: {model.metrics.r_squared:.2f}; MAE: {model.metrics.mae:,.2f}."
            f"{warning_text} Results are modeled estimates, not guaranteed causal lift."
        )
        return AnalysisOutput(
            status,
            model.channel_summary,
            fit_figure,
            channel_figure,
            response_figure,
            GradioSession(bundle, temp_dir),
            str(html_path),
            str(zip_path),
            budget,
        )
    except Exception as exc:
        return _empty_output(f"Analysis could not complete: {exc}")


def optimize_from_analysis(
    session: GradioSession | None, total_budget: float
) -> tuple[str, pd.DataFrame, go.Figure]:
    if session is None:
        return "Run an analysis before optimizing a budget.", pd.DataFrame(), go.Figure()
    try:
        bundle = session.bundle
        channels = list(bundle.prepared.mapping.channel_cols)
        current = {
            channel: float(bundle.prepared.frame[f"media__{channel}"].mean())
            for channel in channels
        }
        bounds = {
            channel: (0.0, float(bundle.model.response_curves[channel]["spend"].max()))
            for channel in channels
        }
        response = {
            channel: (
                lambda spend, curve=curve: float(
                    np.interp(spend, curve["spend"], curve["incremental_outcome"])
                )
            )
            for channel, curve in bundle.model.response_curves.items()
        }
        result = optimize_allocation(response, float(total_budget), bounds, current)
        if not result.success:
            return f"Optimization blocked: {result.message}", pd.DataFrame(), go.Figure()
        allocation = pd.DataFrame(
            {
                "Channel": channels,
                "Current average": [current[c] for c in channels],
                "Recommended spend": [result.optimal_allocation[c] for c in channels],
            }
        )
        figure = px.bar(
            allocation.melt(id_vars="Channel", var_name="Scenario", value_name="Spend"),
            x="Channel", y="Spend", color="Scenario", barmode="group",
            title="Current vs recommended weekly allocation",
        )
        change = result.optimal_prediction - result.current_prediction
        return (
            f"Optimization complete. The recommendation conserves the {total_budget:,.2f} "
            f"budget and changes modeled outcome by {change:,.2f}.",
            allocation,
            figure,
        )
    except Exception as exc:
        return f"Optimization could not complete: {exc}", pd.DataFrame(), go.Figure()

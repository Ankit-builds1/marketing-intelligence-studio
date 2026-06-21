from pathlib import Path

import pytest

from src import gradio_service


def test_demo_source_and_mapping_are_ready_for_analysis():
    frame, label = gradio_service.load_source(None)
    mapping = gradio_service.suggest_mapping(frame)

    assert len(frame) >= 104
    assert label == "Bundled demo dataset"
    assert mapping.date_col in frame.columns
    assert mapping.outcome_col in frame.columns
    assert len(mapping.channel_cols) >= 2
    assert set(mapping.channel_cols) == {"tv_S", "ooh_S", "print_S", "facebook_S", "search_S"}
    assert mapping.control_cols == ("competitor_sales_B",)


def test_analyze_demo_returns_results_figures_and_downloads():
    frame, _ = gradio_service.load_source(None)
    mapping = gradio_service.suggest_mapping(frame)
    result = gradio_service.analyze_dataset(
        None,
        mapping.date_col,
        mapping.outcome_col,
        list(mapping.channel_cols),
        list(mapping.control_cols),
        bootstrap_samples=3,
    )

    assert result.session.bundle.prepared.can_train
    assert "complete" in result.status.lower()
    assert not result.channel_summary.empty
    assert len(result.fit_figure.data) >= 2
    assert len(result.channel_figure.data) >= 1
    assert len(result.response_figure.data) >= 1
    assert Path(result.html_path).read_text(encoding="utf-8").startswith("<!doctype html>")
    assert Path(result.zip_path).read_bytes()[:2] == b"PK"


def test_invalid_mapping_returns_actionable_validation_message():
    result = gradio_service.analyze_dataset(
        None, "DATE", "revenue", ["tv_S"], [], bootstrap_samples=0
    )
    assert result.session is None
    assert "at least two" in result.status.lower()


def test_optimizer_conserves_requested_budget():
    frame, _ = gradio_service.load_source(None)
    mapping = gradio_service.suggest_mapping(frame)
    analysis = gradio_service.analyze_dataset(
        None,
        mapping.date_col,
        mapping.outcome_col,
        list(mapping.channel_cols),
        list(mapping.control_cols),
        bootstrap_samples=2,
    )
    current_budget = sum(
        analysis.session.bundle.prepared.frame[f"media__{channel}"].mean()
        for channel in mapping.channel_cols
    )

    status, allocation, figure = gradio_service.optimize_from_analysis(
        analysis.session, float(current_budget)
    )

    assert "complete" in status.lower()
    assert allocation["Recommended spend"].sum() == pytest.approx(current_budget, abs=1e-3)
    assert len(figure.data) >= 1

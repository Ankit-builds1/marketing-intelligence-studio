"""Native Gradio interface launched by Hugging Face Spaces."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gradio_service import (
    MAX_BOOTSTRAP_SAMPLES,
    AnalysisOutput,
    GradioSession,
    analyze_dataset,
    load_source,
    optimize_from_analysis,
    suggest_mapping,
)


def _source_updates(file_path: Any):
    try:
        frame, label = load_source(file_path)
        mapping = suggest_mapping(frame)
        columns = list(frame.columns)
        numeric = list(frame.select_dtypes(include=["number"]).columns)
        return (
            f"Loaded {label}: {len(frame):,} rows × {len(frame.columns):,} columns.",
            frame.head(15),
            gr.Dropdown(choices=columns, value=mapping.date_col),
            gr.Dropdown(choices=numeric, value=mapping.outcome_col),
            gr.Dropdown(choices=numeric, value=list(mapping.channel_cols), multiselect=True),
            gr.Dropdown(choices=numeric, value=list(mapping.control_cols), multiselect=True),
        )
    except Exception as exc:
        return (
            f"Could not load CSV: {exc}", pd.DataFrame(), gr.Dropdown(), gr.Dropdown(),
            gr.Dropdown(multiselect=True), gr.Dropdown(multiselect=True),
        )


def _analysis_updates(*args):
    result: AnalysisOutput = analyze_dataset(*args)
    return (
        result.status,
        result.channel_summary,
        result.fit_figure,
        result.channel_figure,
        result.response_figure,
        result.session,
        result.html_path,
        result.zip_path,
        result.suggested_budget,
    )


def build_demo() -> gr.Blocks:
    demo_frame, _ = load_source(None)
    mapping = suggest_mapping(demo_frame)
    numeric = list(demo_frame.select_dtypes(include=["number"]).columns)

    with gr.Blocks(title="Marketing Intelligence Studio") as interface:
        gr.Markdown(
            "# Marketing Intelligence Studio\n"
            "Upload marketing data or use the bundled demo. Validate inputs, estimate "
            "modeled channel performance, optimize a fixed budget, and download results. "
            "**Outputs are observational model estimates—not guaranteed causal lift.**"
        )
        session_state = gr.State(None)

        with gr.Tab("1 · Data & Mapping"):
            with gr.Row():
                upload = gr.File(label="Marketing CSV", file_types=[".csv"], type="filepath")
                demo_button = gr.Button("Use bundled demo", variant="primary")
            source_status = gr.Markdown(
                f"Loaded bundled demo: {len(demo_frame):,} rows × {len(demo_frame.columns):,} columns."
            )
            preview = gr.Dataframe(value=demo_frame.head(15), interactive=False)
            with gr.Row():
                date_col = gr.Dropdown(
                    choices=list(demo_frame.columns), value=mapping.date_col, label="Date column"
                )
                outcome_col = gr.Dropdown(
                    choices=numeric, value=mapping.outcome_col, label="Outcome column"
                )
            with gr.Row():
                channels = gr.Dropdown(
                    choices=numeric, value=list(mapping.channel_cols), multiselect=True,
                    label="Media spend columns (select at least two)",
                )
                controls = gr.Dropdown(
                    choices=numeric, value=list(mapping.control_cols), multiselect=True,
                    label="Optional control columns",
                )

        with gr.Tab("2 · Model Results"):
            bootstrap = gr.Slider(
                minimum=0, maximum=MAX_BOOTSTRAP_SAMPLES, value=10, step=1,
                label="Bootstrap refits",
            )
            analyze_button = gr.Button("Validate and run analysis", variant="primary")
            analysis_status = gr.Markdown()
            summary = gr.Dataframe(label="Channel summary", interactive=False)
            with gr.Row():
                fit_plot = gr.Plot(label="Actual vs modeled")
                channel_plot = gr.Plot(label="Contribution and ROI")
            response_plot = gr.Plot(label="Response curves")
            with gr.Row():
                html_download = gr.File(label="HTML executive report")
                zip_download = gr.File(label="Result tables ZIP")

        with gr.Tab("3 · Budget & Downloads"):
            budget = gr.Number(label="Future weekly budget", value=0.0, minimum=0.0)
            optimize_button = gr.Button("Optimize allocation", variant="primary")
            optimization_status = gr.Markdown()
            allocation = gr.Dataframe(label="Recommended allocation", interactive=False)
            allocation_plot = gr.Plot(label="Current vs recommended")

        source_outputs = [source_status, preview, date_col, outcome_col, channels, controls]
        upload.change(_source_updates, inputs=[upload], outputs=source_outputs)
        demo_button.click(lambda: _source_updates(None), inputs=None, outputs=source_outputs)
        analyze_button.click(
            _analysis_updates,
            inputs=[upload, date_col, outcome_col, channels, controls, bootstrap],
            outputs=[
                analysis_status, summary, fit_plot, channel_plot, response_plot,
                session_state, html_download, zip_download, budget,
            ],
        )
        optimize_button.click(
            optimize_from_analysis,
            inputs=[session_state, budget],
            outputs=[optimization_status, allocation, allocation_plot],
        )
    return interface


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()

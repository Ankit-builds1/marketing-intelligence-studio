from __future__ import annotations

import importlib.util
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "budget_optimizer_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("budget_optimizer_app_under_test", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_import_is_side_effect_free_and_exposes_analysis_interfaces():
    module = _load_app_module()

    assert hasattr(module, "AnalysisBundle")
    assert hasattr(module, "run_demo_analysis")


def test_demo_analysis_reaches_exportable_results():
    module = _load_app_module()
    analysis = module.run_demo_analysis(bootstrap_samples=5)

    assert isinstance(analysis, module.AnalysisBundle)
    assert analysis.prepared.can_train
    assert len(analysis.model.channel_summary) >= 2
    assert analysis.report_html.startswith("<!doctype html>")
    assert analysis.tables_zip[:2] == b"PK"


def test_streamlit_app_renders_five_guided_tabs():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=120)

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "1 · Data",
        "2 · Validate",
        "3 · Model",
        "4 · Results",
        "5 · Optimize & Export",
    ]

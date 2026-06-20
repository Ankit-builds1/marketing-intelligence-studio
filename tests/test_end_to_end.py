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


def test_same_upload_fingerprint_preserves_existing_state():
    module = _load_app_module()
    raw = module._load_demo_frame()
    state: dict[str, object] = {}
    fingerprint = module._fingerprint_bytes(b"same-upload")

    first_change = module._sync_source_state(
        state,
        raw,
        source_kind="upload",
        source_label="Uploaded file: marketing.csv",
        source_fingerprint=fingerprint,
    )
    assert first_change

    preserved_mapping = module.DataMapping("DATE", "revenue", ("tv_S", "search_S"), ("competitor_sales_B",))
    preserved_analysis = object()
    preserved_optimizer = object()
    state["mapping_selection"] = preserved_mapping
    state["mapping_signature"] = module._mapping_signature(preserved_mapping)
    state["analysis"] = preserved_analysis
    state["analysis_signature"] = "analysis-signature"
    state["optimizer_result"] = preserved_optimizer
    state["optimizer_signature"] = "optimizer-signature"

    second_change = module._sync_source_state(
        state,
        raw.copy(),
        source_kind="upload",
        source_label="Uploaded file: marketing.csv",
        source_fingerprint=fingerprint,
    )

    assert not second_change
    assert state["mapping_selection"] is preserved_mapping
    assert state["analysis"] is preserved_analysis
    assert state["optimizer_result"] is preserved_optimizer


def test_changed_upload_fingerprint_clears_mapping_analysis_and_optimizer_state():
    module = _load_app_module()
    first = module._load_demo_frame()
    second = first.rename(columns={"revenue": "sales"})
    state: dict[str, object] = {}

    module._sync_source_state(
        state,
        first,
        source_kind="upload",
        source_label="Uploaded file: first.csv",
        source_fingerprint=module._fingerprint_bytes(b"first-upload"),
    )
    state["analysis"] = object()
    state["analysis_signature"] = "old-analysis"
    state["optimizer_result"] = object()
    state["optimizer_signature"] = "old-optimizer"
    state["mapping_selection"] = module.DataMapping("DATE", "revenue", ("tv_S", "search_S"), ())
    state["mapping_signature"] = "old-mapping"

    changed = module._sync_source_state(
        state,
        second,
        source_kind="upload",
        source_label="Uploaded file: second.csv",
        source_fingerprint=module._fingerprint_bytes(b"second-upload"),
    )

    assert changed
    assert state["analysis"] is None
    assert state["analysis_signature"] is None
    assert state["optimizer_result"] is None
    assert state["optimizer_signature"] is None
    assert state["source_label"] == "Uploaded file: second.csv"
    assert state["mapping_signature"] == module._mapping_signature(state["mapping_selection"])
    assert state["mapping_selection"].outcome_col == "sales"


def test_mapping_change_invalidates_analysis_and_optimizer():
    module = _load_app_module()
    state: dict[str, object] = {
        "analysis": object(),
        "analysis_signature": "analysis-signature",
        "optimizer_result": object(),
        "optimizer_signature": "optimizer-signature",
        "mapping_signature": module._mapping_signature(module.DEMO_MAPPING),
        "mapping_selection": module.DEMO_MAPPING,
    }
    changed_mapping = module.DataMapping(
        "DATE",
        "revenue",
        ("tv_S", "ooh_S", "facebook_S"),
        ("competitor_sales_B",),
    )

    changed = module._sync_mapping_state(state, changed_mapping)

    assert changed
    assert state["mapping_selection"] == changed_mapping
    assert state["mapping_signature"] == module._mapping_signature(changed_mapping)
    assert state["analysis"] is None
    assert state["analysis_signature"] is None
    assert state["optimizer_result"] is None
    assert state["optimizer_signature"] is None


def test_uploaded_analysis_path_uses_session_cache_not_global_cache(monkeypatch):
    module = _load_app_module()
    raw = module._load_demo_frame()
    prepared = module.prepare_and_validate(raw, module.DEMO_MAPPING)
    transforms = module._default_transforms(prepared)
    config = module.MMMConfig(bootstrap_samples=0, random_state=module.DEFAULT_RANDOM_STATE)
    state: dict[str, object] = {"uploaded_analysis_cache": {}}
    calls = {"count": 0}

    def fake_demo_cache(*args, **kwargs):
        raise AssertionError("global demo cache should not be used for uploaded analysis")

    original_builder = module._build_analysis_bundle

    def counted_builder(*args, **kwargs):
        calls["count"] += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(module, "_cached_demo_analysis_bundle", fake_demo_cache)
    monkeypatch.setattr(module, "_build_analysis_bundle", counted_builder)

    first = module._get_or_create_analysis(
        state,
        raw,
        source_kind="upload",
        source_fingerprint=module._fingerprint_bytes(b"uploaded-demo-bytes"),
        mapping=module.DEMO_MAPPING,
        transforms=transforms,
        config=config,
    )
    second = module._get_or_create_analysis(
        state,
        raw.copy(),
        source_kind="upload",
        source_fingerprint=module._fingerprint_bytes(b"uploaded-demo-bytes"),
        mapping=module.DEMO_MAPPING,
        transforms=transforms,
        config=config,
    )

    assert calls["count"] == 1
    assert first is second


def test_training_gate_blocks_invalid_input():
    module = _load_app_module()
    invalid = module.prepare_and_validate(module._load_demo_frame().head(12), module.DEMO_MAPPING)

    gate = module._training_gate_state(invalid)

    assert gate["disabled"] is True
    assert "validation" in gate["help"].lower()


def test_optimizer_input_change_invalidates_stale_recommendation():
    module = _load_app_module()
    state: dict[str, object] = {
        "optimizer_result": object(),
        "optimizer_signature": "old-signature",
    }
    bounds = {"search": (10.0, 100.0), "social": (5.0, 80.0)}
    signature = module._optimizer_input_signature("analysis-v1", 120.0, bounds)

    changed = module._sync_optimizer_signature(state, signature)

    assert changed
    assert state["optimizer_result"] is None
    assert state["optimizer_signature"] == signature

    preserved_result = object()
    state["optimizer_result"] = preserved_result
    assert not module._sync_optimizer_signature(state, signature)
    assert state["optimizer_result"] is preserved_result


def test_optimizer_signature_tracks_analysis_and_normalizes_bound_order():
    module = _load_app_module()
    forward = {"search": (10.0, 100.0), "social": (5.0, 80.0)}
    reversed_order = {"social": (5.0, 80.0), "search": (10.0, 100.0)}

    signature = module._optimizer_input_signature("analysis-v1", 120.0, forward)

    assert signature == module._optimizer_input_signature(
        "analysis-v1", 120.0, reversed_order
    )
    assert signature != module._optimizer_input_signature(
        "analysis-v2", 120.0, forward
    )


def test_invalid_bound_rerun_clears_previous_optimizer_result():
    module = _load_app_module()
    valid_bounds = {"search": (10.0, 100.0), "social": (5.0, 80.0)}
    invalid_bounds = {"search": (110.0, 100.0), "social": (5.0, 80.0)}
    state: dict[str, object] = {
        "optimizer_result": object(),
        "optimizer_signature": module._optimizer_input_signature(
            "analysis-v1", 120.0, valid_bounds
        ),
    }

    errors, _ = module._optimizer_validation(120.0, invalid_bounds, valid_bounds)
    changed = module._sync_optimizer_signature(
        state,
        module._optimizer_input_signature("analysis-v1", 120.0, invalid_bounds),
    )

    assert errors
    assert changed
    assert state["optimizer_result"] is None

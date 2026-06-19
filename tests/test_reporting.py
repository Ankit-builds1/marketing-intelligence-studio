from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from src.reporting import ReportContext, build_html_report, tables_as_zip


def test_report_escapes_user_values_and_includes_causal_caveat() -> None:
    context = ReportContext(
        "<Sales>",
        "2024-01-01",
        "2025-12-31",
        "Medium",
        ["Limited history"],
        pd.DataFrame({"channel": ["search"], "roi": [1.2], "note": ["<b>bad</b>"]}),
    )

    html = build_html_report(context)

    assert "&lt;Sales&gt;" in html
    assert "<Sales>" not in html
    assert "&lt;b&gt;bad&lt;/b&gt;" in html
    assert "Limited history" in html
    assert "model-based estimates from observational data" in html


def test_report_shows_empty_states_for_warnings_and_tables() -> None:
    context = ReportContext(
        "Revenue",
        "2024-01-01",
        "2025-12-31",
        "Low",
        [],
        pd.DataFrame(columns=["channel", "roi"]),
    )

    html = build_html_report(context)

    assert "No warnings were recorded." in html
    assert "No channel summary data available." in html


def test_tables_as_zip_returns_a_deterministic_valid_zip_archive() -> None:
    payload_one = tables_as_zip({})
    payload_two = tables_as_zip({})

    assert payload_one == payload_two
    with ZipFile(BytesIO(payload_one)) as archive:
        assert archive.namelist() == []


def test_tables_as_zip_sanitizes_entry_names_and_resolves_collisions() -> None:
    payload = tables_as_zip(
        {
            "../summary": pd.DataFrame({"x": [1]}),
            "summary": pd.DataFrame({"x": [2]}),
            "nested/../summary": pd.DataFrame({"x": [3]}),
        }
    )

    with ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()

    assert len(names) == 3
    assert len(set(names)) == 3
    assert names == sorted(names, key=names.index)
    for name in names:
        assert name.endswith(".csv")
        assert name == name.rsplit("/", 1)[-1]
        assert "\\" not in name
        assert ".." not in name

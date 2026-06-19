from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import PurePosixPath
import re
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd


@dataclass(frozen=True)
class ReportContext:
    outcome_name: str
    start_date: str
    end_date: str
    reliability: str
    warnings: list[str]
    channel_summary: pd.DataFrame


def _html_table_or_empty(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="empty-state">No channel summary data available.</p>'
    return frame.to_html(index=False, escape=True, border=0)


def _warnings_html(warnings: list[str]) -> str:
    if not warnings:
        return '<p class="empty-state">No warnings were recorded.</p>'
    items = "".join(f"<li>{escape(message)}</li>" for message in warnings)
    return f"<ul>{items}</ul>"


def build_html_report(context: ReportContext) -> str:
    outcome_name = escape(context.outcome_name)
    start_date = escape(context.start_date)
    end_date = escape(context.end_date)
    reliability = escape(context.reliability)
    channel_summary = _html_table_or_empty(context.channel_summary)
    warnings_html = _warnings_html(context.warnings)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMM Executive Report</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      max-width: 980px;
      margin: 32px auto;
      padding: 0 20px 48px;
      line-height: 1.5;
    }}
    h1, h2, h3 {{
      line-height: 1.2;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 16px;
      border: 1px solid #d7dde8;
      border-radius: 12px;
      background: #f8fafc;
      margin: 20px 0 24px;
    }}
    .meta div {{
      margin: 0;
    }}
    .label {{
      display: block;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #5b6477;
      margin-bottom: 4px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 24px;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid #d7dde8;
      padding: 8px 10px;
      vertical-align: top;
      word-wrap: break-word;
    }}
    th {{
      text-align: left;
      background: #f8fafc;
      position: sticky;
      top: 0;
    }}
    td {{
      text-align: right;
    }}
    td:first-child, th:first-child {{
      text-align: left;
    }}
    .empty-state {{
      padding: 12px 14px;
      border: 1px dashed #b9c3d3;
      border-radius: 10px;
      background: #fbfcfe;
      color: #4a5568;
    }}
    .caveat {{
      padding: 14px 16px;
      border-left: 4px solid #9b6b00;
      background: #fff8e6;
      margin-top: 24px;
    }}
    @media print {{
      body {{
        margin: 0;
        max-width: none;
        padding: 0.35in;
      }}
      .meta, .caveat, .empty-state {{
        break-inside: avoid;
      }}
      a {{
        color: inherit;
        text-decoration: none;
      }}
      table {{
        page-break-inside: auto;
      }}
      tr {{
        page-break-inside: avoid;
        page-break-after: auto;
      }}
    }}
  </style>
</head>
<body>
  <h1>Marketing Mix Modeling Executive Report</h1>
  <p>This report summarizes model output for a single business outcome and is designed to be printed or shared as a standalone file.</p>
  <section class="meta" aria-label="report summary">
    <div><span class="label">Outcome</span>{outcome_name}</div>
    <div><span class="label">Period Start</span>{start_date}</div>
    <div><span class="label">Period End</span>{end_date}</div>
    <div><span class="label">Reliability</span>{reliability}</div>
  </section>
  <section>
    <h2>Channel summary</h2>
    {channel_summary}
  </section>
  <section>
    <h2>Warnings and limitations</h2>
    {warnings_html}
  </section>
  <aside class="caveat">
    Results are model-based estimates from observational data. They support decision-making, but they do not prove causal truth on their own.
  </aside>
</body>
</html>
"""


def _sanitize_entry_name(name: object) -> str:
    raw_name = str(name).strip().replace("\\", "/")
    base_name = PurePosixPath(raw_name).name.strip()
    if base_name.lower().endswith(".csv"):
        base_name = base_name[:-4]
    base_name = base_name.strip().strip(".")
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
    base_name = base_name.strip("._-")
    return base_name or "table"


def tables_as_zip(tables: Mapping[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    used_names: set[str] = set()
    fixed_timestamp = (2024, 1, 1, 0, 0, 0)

    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for original_name, frame in tables.items():
            base_name = _sanitize_entry_name(original_name)
            entry_name = f"{base_name}.csv"
            suffix = 1
            while entry_name in used_names:
                entry_name = f"{base_name}_{suffix}.csv"
                suffix += 1
            used_names.add(entry_name)

            csv_payload = frame.to_csv(index=False, lineterminator="\n")
            info = ZipInfo(entry_name, date_time=fixed_timestamp)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            archive.writestr(info, csv_payload.encode("utf-8"))

    return buffer.getvalue()

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CadenceInfo:
    kind: str
    median_days: float
    period_count: int


def detect_cadence(dates: pd.Series) -> CadenceInfo:
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(parsed) < 2:
        return CadenceInfo("unknown", 0.0, len(parsed))
    median = float(parsed.diff().dt.total_seconds().dropna().median() / 86400)
    if median <= 2:
        kind = "daily"
    elif median <= 10:
        kind = "weekly"
    elif 20 <= median <= 40:
        kind = "monthly"
    else:
        kind = "irregular"
    return CadenceInfo(kind, median, len(parsed))


def minimum_periods(cadence: str) -> int:
    return {"daily": 28, "weekly": 52, "monthly": 24}.get(cadence, 52)


def period_frequency(cadence: str) -> str:
    return {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[cadence]

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    missing_rate: float
    unique_count: int
    is_numeric: bool
    is_datetime: bool
    is_identifier: bool


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile]


def _datetime_like(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not any(token in name.lower() for token in ("date", "time", "week", "month")):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return "timestamp" in name.lower()
    parsed = pd.to_datetime(series.dropna().head(200), errors="coerce")
    return bool(len(parsed) and parsed.notna().mean() >= 0.8)


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    columns = {}
    rows = len(frame)
    for name in frame.columns:
        series = frame[name]
        unique = int(series.nunique(dropna=True))
        numeric = bool(pd.api.types.is_numeric_dtype(series))
        datetime = _datetime_like(series, str(name))
        lower = str(name).lower()
        id_named = lower == "id" or lower.endswith("_id") or lower in {"uid", "campaign_id", "customer_id"}
        near_unique_integer = numeric and rows >= 20 and unique / max(rows, 1) > 0.95 and any(
            token in lower for token in ("id", "uid")
        )
        columns[str(name)] = ColumnProfile(
            str(name), str(series.dtype), float(series.isna().mean()), unique,
            numeric, datetime, bool(id_named or near_unique_integer),
        )
    return DatasetProfile(rows, len(frame.columns), columns)

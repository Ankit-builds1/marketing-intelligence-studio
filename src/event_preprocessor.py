from dataclasses import dataclass

import pandas as pd

from src.data_validation import DataMapping


@dataclass(frozen=True)
class PreparationRequest:
    date_col: str
    outcome_col: str
    spend_col: str
    channel_col: str
    cadence: str = "daily"
    control_cols: tuple[str, ...] = ()
    timestamp_origin: str | None = None
    max_channels: int = 8


@dataclass
class PreparationResult:
    frame: pd.DataFrame
    mapping: DataMapping
    transformations: tuple[str, ...]
    warnings: tuple[str, ...]
    can_analyze: bool


def _dates(series: pd.Series, origin: str | None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if origin:
            magnitude = numeric.dropna().abs().median() if numeric.notna().any() else 0
            if magnitude < 1_000_000_000:
                return pd.Timestamp(origin) + pd.to_timedelta(numeric, unit="s")
        magnitude = numeric.dropna().abs().median() if numeric.notna().any() else 0
        unit = "ms" if magnitude > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def prepare_marketing_data(frame: pd.DataFrame, request: PreparationRequest) -> PreparationResult:
    required = (request.date_col, request.outcome_col, request.spend_col, request.channel_col)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return PreparationResult(pd.DataFrame(), DataMapping("", "", ()), (),
                                 (f"Missing columns: {', '.join(missing)}",), False)
    data = frame.loc[:, [*required, *request.control_cols]].copy()
    data[request.date_col] = _dates(data[request.date_col], request.timestamp_origin)
    data[request.spend_col] = pd.to_numeric(data[request.spend_col], errors="coerce")
    data[request.outcome_col] = pd.to_numeric(data[request.outcome_col], errors="coerce")
    data = data.dropna(subset=[request.date_col, request.spend_col, request.outcome_col, request.channel_col])
    freq = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[request.cadence]
    data["date"] = data[request.date_col].dt.to_period("D").dt.start_time
    if request.cadence == "weekly":
        data["date"] = data["date"] - pd.to_timedelta(data["date"].dt.weekday, unit="D")
    elif request.cadence == "monthly":
        data["date"] = data["date"].dt.to_period("M").dt.start_time
    totals = data.groupby(request.channel_col)[request.spend_col].sum().sort_values(ascending=False)
    keep = list(totals.head(request.max_channels).index)
    data["_channel"] = data[request.channel_col].where(data[request.channel_col].isin(keep), "Other")
    spend = data.pivot_table(index="date", columns="_channel", values=request.spend_col, aggfunc="sum", fill_value=0)
    spend.columns = [f"{str(name)}_spend" for name in spend.columns]
    outcome = data.groupby("date")[request.outcome_col].sum().rename("outcome")
    result = pd.concat([outcome, spend], axis=1).reset_index().sort_values("date")
    mapping = DataMapping("date", "outcome", tuple(spend.columns), ())
    transformations = (f"Parsed {request.date_col} as time.", f"Aggregated to {request.cadence} periods.",
                       f"Pivoted {request.channel_col} into {len(spend.columns)} spend channels.")
    can = len(result) > 0 and len(spend.columns) >= 1
    return PreparationResult(result, mapping, transformations, (), can)

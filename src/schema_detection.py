from dataclasses import dataclass

import pandas as pd

from src.dataset_profiler import DatasetProfile, profile_dataset


@dataclass(frozen=True)
class SchemaDetection:
    family: str
    confidence: float
    suggested_date: str
    suggested_outcome: str
    suggested_channels: tuple[str, ...]
    suggested_spend: str
    suggested_channel_dimension: str
    suggested_controls: tuple[str, ...]
    reasons: tuple[str, ...]


def _first(columns, keywords):
    for keyword in keywords:
        for name in columns:
            if keyword in name.lower():
                return name
    return ""


def detect_schema(frame: pd.DataFrame, profile: DatasetProfile | None = None) -> SchemaDetection:
    profile = profile or profile_dataset(frame)
    names = list(profile.columns)
    date = _first(names, ("date", "week", "month", "timestamp", "time"))
    outcome = _first(names, ("revenue", "sales", "orders", "conversions", "conversion", "leads", "installs"))
    spend = _first(names, ("spend", "cost", "budget", "investment"))
    dimension = _first(names, ("channel", "platform", "campaign", "source", "medium"))
    wide_channels = tuple(
        name for name in names
        if profile.columns[name].is_numeric and not profile.columns[name].is_identifier
        and any(token in name.lower() for token in ("spend", "cost", "budget", "investment"))
        and name != spend
    )
    # A single generic spend field belongs to a long/event schema; named spend fields form wide MMM.
    named_spend = tuple(name for name in names if profile.columns[name].is_numeric and not profile.columns[name].is_identifier
                        and any(token in name.lower() for token in ("spend", "_s", "investment")))
    controls = tuple(name for name in names if any(token in name.lower() for token in
                     ("price", "promo", "holiday", "competitor", "distribution", "click", "impression")))
    has_event_signals = bool(date and spend and dimension and outcome)
    if len(named_spend) >= 2 and date and outcome:
        family, confidence, channels = "wide_mmm", 0.95, named_spend
        reasons = ("Found time, outcome, and multiple named media-spend columns.",)
    elif has_event_signals:
        is_event = profile.row_count > 500 or _first(names, ("uid", "click_pos", "attribution"))
        family = "event_attribution" if is_event else "long_campaign"
        confidence, channels = 0.9, ()
        reasons = ("Found time, campaign/channel, cost, and outcome fields in long form.",)
    elif date and outcome and spend:
        family, confidence, channels = "single_channel", 0.75, (spend,)
        reasons = ("Found one monetary media series and an outcome.",)
    else:
        family, confidence, channels = "unsupported", 0.2, ()
        reasons = ("A usable marketing time, outcome, or spend structure could not be established.",)
    return SchemaDetection(family, confidence, date, outcome, channels, spend, dimension, controls, reasons)

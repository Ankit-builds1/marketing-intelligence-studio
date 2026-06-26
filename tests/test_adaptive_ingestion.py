import numpy as np
import pandas as pd

from src.analysis_router import route_analysis
from src.cadence import detect_cadence, minimum_periods
from src.dataset_profiler import profile_dataset
from src.event_preprocessor import PreparationRequest, prepare_marketing_data
from src.reliability import assess_reliability
from src.schema_detection import detect_schema


def weekly_wide(rows=60):
    x = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "week": pd.date_range("2024-01-01", periods=rows, freq="W-MON"),
        "revenue": 1000 + 4 * x,
        "search_spend": 80 + x % 9,
        "social_spend": 50 + x % 7,
        "price": 10 + np.sin(x / 5),
    })


def test_profiles_ids_dates_and_numeric_metrics():
    frame = weekly_wide()
    frame["customer_id"] = np.arange(len(frame)) + 10000
    profile = profile_dataset(frame)
    assert profile.row_count == 60
    assert profile.columns["week"].is_datetime
    assert profile.columns["customer_id"].is_identifier
    assert profile.columns["revenue"].is_numeric


def test_detects_wide_and_event_schemas_without_using_ids_as_spend():
    wide = detect_schema(weekly_wide())
    assert wide.family == "wide_mmm"
    assert set(wide.suggested_channels) == {"search_spend", "social_spend"}

    event = pd.DataFrame({
        "timestamp": np.arange(100),
        "uid": np.arange(100) + 9000,
        "campaign": np.tile([11, 22, 33], 34)[:100],
        "cost": np.linspace(0.1, 2, 100),
        "click": np.tile([0, 1], 50),
        "conversion": np.tile([0, 0, 0, 1], 25),
    })
    detection = detect_schema(event)
    assert detection.family == "event_attribution"
    assert detection.suggested_spend == "cost"
    assert "uid" not in detection.suggested_channels


def test_cadence_requirements_cover_daily_weekly_monthly():
    assert detect_cadence(pd.Series(pd.date_range("2025-01-01", periods=40, freq="D"))).kind == "daily"
    assert detect_cadence(pd.Series(pd.date_range("2024-01-01", periods=60, freq="W"))).kind == "weekly"
    assert detect_cadence(pd.Series(pd.date_range("2022-01-01", periods=30, freq="MS"))).kind == "monthly"
    assert minimum_periods("daily") == 28
    assert minimum_periods("weekly") == 52
    assert minimum_periods("monthly") == 24


def test_prepares_long_campaign_export_and_preserves_spend_total():
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    frame = pd.DataFrame({
        "date": np.repeat(dates, 3),
        "channel": ["Search", "Social", "Video"] * len(dates),
        "spend": np.tile([10.0, 20.0, 30.0], len(dates)),
        "conversions": np.tile([2.0, 3.0, 4.0], len(dates)),
    })
    result = prepare_marketing_data(frame, PreparationRequest(
        date_col="date", outcome_col="conversions", spend_col="spend",
        channel_col="channel", cadence="daily",
    ))
    assert result.can_analyze
    assert len(result.frame) == 35
    assert set(result.mapping.channel_cols) == {"Search_spend", "Social_spend", "Video_spend"}
    assert result.frame[list(result.mapping.channel_cols)].sum().sum() == frame["spend"].sum()


def test_router_disables_roi_without_monetary_spend_and_scores_history():
    full = route_analysis("wide_mmm", channel_count=3, has_monetary_spend=True, period_count=60, cadence="weekly")
    assert full.workflow == "full_mmm"
    assert full.can_optimize
    activity = route_analysis("wide_mmm", channel_count=3, has_monetary_spend=False, period_count=60, cadence="weekly")
    assert not activity.can_show_roi
    assert not activity.can_optimize
    assert assess_reliability(period_count=120, minimum_required=52, issue_count=0).label == "Strong"
    assert assess_reliability(period_count=20, minimum_required=52, issue_count=0).label == "Insufficient"


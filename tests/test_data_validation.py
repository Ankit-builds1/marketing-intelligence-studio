import pandas as pd

from src.data_validation import DataMapping, prepare_and_validate


def valid_frame(rows: int = 104) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=rows, freq="W-MON"),
            "sales": range(100, 100 + rows),
            "search": [10 + (i % 13) for i in range(rows)],
            "social": [20 + ((i * 7) % 19) for i in range(rows)],
            "price": [1.0 + i / 1000 for i in range(rows)],
        }
    )


def daily_frame(rows: int = 364) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "sales": range(100, 100 + rows),
            "search": [10 + i for i in range(rows)],
            "social": [20 + (i * 2) + (i % 5) for i in range(rows)],
            "price": [1.0 + i / 1000 for i in range(rows)],
        }
    )


def duplicate_date_frame() -> pd.DataFrame:
    rows = []
    for week in range(52):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=week * 7 + 1)
        rows.append(
            {
                "day": day,
                "sales": 100 + week * 10 + 1,
                "search": 10 + week * 2 + 1,
                "social": 20 + week * 3 + 1,
                "price": 1.0 + week / 100,
            }
        )
        rows.append(
            {
                "day": day,
                "sales": 100 + week * 10 + 2,
                "search": 10 + week * 2 + 2,
                "social": 20 + week * 3 + 2,
                "price": 3.0 + week / 100,
            }
        )
    return pd.DataFrame(rows)


def test_prepares_canonical_weekly_frame():
    mapping = DataMapping("week", "sales", ("search", "social"), ("price",))
    result = prepare_and_validate(valid_frame(), mapping)

    assert result.can_train
    assert list(result.frame.columns) == [
        "date",
        "outcome",
        "media__search",
        "media__social",
        "control__price",
    ]
    assert not [issue for issue in result.issues if issue.severity == "error"]


def test_blocks_too_few_rows_and_negative_spend():
    frame = valid_frame(40)
    frame.loc[0, "search"] = -1
    mapping = DataMapping("week", "sales", ("search", "social"), ())

    result = prepare_and_validate(frame, mapping)

    assert not result.can_train
    assert {issue.code for issue in result.issues} >= {"too_few_rows", "negative_media"}


def test_warns_about_near_duplicate_channels():
    frame = valid_frame()
    frame["social"] = frame["search"] * 1.001
    mapping = DataMapping("week", "sales", ("search", "social"), ())

    result = prepare_and_validate(frame, mapping)

    assert "high_collinearity" in {issue.code for issue in result.issues}


def test_aggregates_daily_data_to_weekly_and_is_trainable_at_52_weeks():
    mapping = DataMapping("day", "sales", ("search", "social"), ("price",))
    result = prepare_and_validate(daily_frame(), mapping)

    assert result.can_train
    assert len(result.frame) == 52
    assert list(result.frame.columns) == [
        "date",
        "outcome",
        "media__search",
        "media__social",
        "control__price",
    ]
    assert result.frame.loc[0, "date"] == pd.Timestamp("2024-01-01")
    assert result.frame.loc[0, "outcome"] == sum(range(100, 107))
    assert result.frame.loc[0, "media__search"] == sum(10 + i for i in range(7))
    assert result.frame.loc[0, "media__social"] == sum(20 + (i * 2) + (i % 5) for i in range(7))


def test_blocks_daily_data_with_fewer_than_52_resulting_weeks():
    mapping = DataMapping("day", "sales", ("search", "social"), ())
    result = prepare_and_validate(daily_frame(357), mapping)

    assert not result.can_train
    assert "too_few_rows" in {issue.code for issue in result.issues}


def test_blocks_monthly_and_irregular_cadence():
    monthly = pd.DataFrame(
        {
            "month": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "sales": range(100, 112),
            "search": [10 + i for i in range(12)],
            "social": [20 + ((i * 3) % 9) for i in range(12)],
        }
    )
    irregular = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-08",
                    "2024-01-22",
                    "2024-02-05",
                    "2024-02-06",
                ]
            ),
            "sales": [100, 101, 102, 103, 104, 105],
            "search": [10, 11, 12, 13, 14, 15],
            "social": [20, 21, 22, 23, 24, 25],
        }
    )

    monthly_result = prepare_and_validate(
        monthly, DataMapping("month", "sales", ("search", "social"), ())
    )
    irregular_result = prepare_and_validate(
        irregular, DataMapping("date", "sales", ("search", "social"), ())
    )

    assert not monthly_result.can_train
    assert not irregular_result.can_train
    assert "irregular_cadence" in {issue.code for issue in monthly_result.issues}
    assert "irregular_cadence" in {issue.code for issue in irregular_result.issues}


def test_preserves_weekly_dates_and_resolves_duplicate_rows_deterministically():
    mapping = DataMapping("day", "sales", ("search", "social"), ("price",))
    result = prepare_and_validate(duplicate_date_frame(), mapping)

    assert result.can_train
    assert len(result.frame) == 52
    assert result.frame.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert result.frame.loc[0, "outcome"] == 102
    assert result.frame.loc[0, "media__search"] == 12
    assert result.frame.loc[0, "media__social"] == 22
    assert result.frame.loc[0, "control__price"] == 3.0


def test_blocks_negative_subweekly_rows_even_if_weekly_sum_would_be_positive():
    frame = pd.DataFrame(
        {
            "day": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                ]
            ),
            "sales": [100, 101, 102, 103, 104, 105, 106],
            "search": [10, 11, -100, 13, 14, 15, 16],
            "social": [20, 21, 22, 23, 24, 25, 26],
        }
    )
    mapping = DataMapping("day", "sales", ("search", "social"), ())

    result = prepare_and_validate(frame, mapping)

    assert not result.can_train
    assert "negative_media" in {issue.code for issue in result.issues}


def test_preserves_non_monday_weekly_dates_without_reanchoring():
    weeks = pd.date_range("2024-01-03", periods=52, freq="W-WED")
    frame = pd.DataFrame(
        {
            "week": weeks,
            "sales": range(100, 152),
            "search": range(10, 62),
            "social": range(20, 72),
        }
    )
    mapping = DataMapping("week", "sales", ("search", "social"), ())

    result = prepare_and_validate(frame, mapping)

    assert result.can_train
    assert list(result.frame["date"]) == list(frame["week"])

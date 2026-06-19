import pandas as pd

from src.data_validation import DataMapping, prepare_and_validate


def valid_frame(rows: int = 104) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=rows, freq="W-MON"),
            "sales": range(100, 100 + rows),
            "search": range(10, 10 + rows),
            "social": range(20, 20 + rows),
            "price": [1.0 + i / 1000 for i in range(rows)],
        }
    )


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

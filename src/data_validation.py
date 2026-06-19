from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataMapping:
    date_col: str
    outcome_col: str
    channel_cols: tuple[str, ...]
    control_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class PreparedData:
    frame: pd.DataFrame
    mapping: DataMapping
    issues: list[ValidationIssue]

    @property
    def can_train(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _cadence_kind(dates: pd.Series) -> str:
    unique_dates = dates.sort_values().drop_duplicates()
    if len(unique_dates) < 2:
        return "weekly"

    gaps = unique_dates.diff().dt.days.dropna()
    if gaps.empty:
        return "weekly"

    median_gap = gaps.median()
    min_gap = gaps.min()
    max_gap = gaps.max()

    if max_gap <= 7 and median_gap < 7:
        return "subweekly"

    if 5 <= min_gap and max_gap <= 9 and 5 <= median_gap <= 9:
        return "weekly"

    return "irregular"


def _week_start(date_series: pd.Series) -> pd.Series:
    return date_series.dt.normalize() - pd.to_timedelta(date_series.dt.weekday, unit="D")


def _aggregate_to_weekly(frame: pd.DataFrame, mapping: DataMapping) -> pd.DataFrame:
    grouped = frame.copy()
    grouped["_week_start"] = _week_start(grouped[mapping.date_col])
    agg_spec: dict[str, str] = {mapping.outcome_col: "sum"}
    agg_spec.update({name: "sum" for name in mapping.channel_cols})
    agg_spec.update({name: "mean" for name in mapping.control_cols})

    weekly = grouped.groupby("_week_start", as_index=False).agg(agg_spec)
    return weekly.rename(columns={"_week_start": mapping.date_col})


def prepare_and_validate(
    raw: pd.DataFrame, mapping: DataMapping, min_rows: int = 52
) -> PreparedData:
    selected = (
        mapping.date_col,
        mapping.outcome_col,
        *mapping.channel_cols,
        *mapping.control_cols,
    )
    issues: list[ValidationIssue] = []

    if len(set(selected)) != len(selected):
        return PreparedData(
            pd.DataFrame(),
            mapping,
            [
                ValidationIssue(
                    "error",
                    "duplicate_mapping",
                    "Each role must use a different column.",
                )
            ],
        )

    if len(mapping.channel_cols) < 2:
        issues.append(
            ValidationIssue(
                "error",
                "too_few_channels",
                "Select at least two media channels.",
            )
        )

    missing = [column for column in selected if column not in raw.columns]
    if missing:
        issues.append(
            ValidationIssue(
                "error",
                "missing_columns",
                f"Missing mapped columns: {', '.join(missing)}",
            )
        )
        return PreparedData(pd.DataFrame(), mapping, issues)

    frame = raw.loc[:, selected].copy()

    dates = pd.to_datetime(frame[mapping.date_col], errors="coerce")
    if dates.isna().mean() > 0.05:
        issues.append(
            ValidationIssue(
                "error",
                "invalid_dates",
                "More than 5% of date values cannot be parsed.",
            )
        )
    frame[mapping.date_col] = dates

    numeric = [mapping.outcome_col, *mapping.channel_cols, *mapping.control_cols]
    if numeric:
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        if frame[numeric].isna().mean().max() > 0.20:
            issues.append(
                ValidationIssue(
                    "error",
                    "excessive_missingness",
                    "A mapped numeric column has more than 20% missing values.",
                )
            )

    frame = frame.dropna(subset=[mapping.date_col, mapping.outcome_col, *mapping.channel_cols])
    frame = frame.sort_values(mapping.date_col).drop_duplicates(mapping.date_col, keep="last")

    cadence = _cadence_kind(frame[mapping.date_col])
    if cadence == "subweekly":
        frame = _aggregate_to_weekly(frame, mapping)
    elif cadence == "irregular":
        issues.append(
            ValidationIssue(
                "error",
                "irregular_cadence",
                "Cadence cannot be reliably converted to weekly periods.",
            )
        )
        return PreparedData(pd.DataFrame(), mapping, issues)

    if len(frame) < min_rows:
        issues.append(
            ValidationIssue(
                "error",
                "too_few_rows",
                f"At least {min_rows} usable weekly rows are required.",
            )
        )
    elif len(frame) < 104:
        issues.append(
            ValidationIssue(
                "warning",
                "limited_history",
                "Two years of weekly history is recommended.",
            )
        )

    if (frame[mapping.outcome_col] < 0).any():
        issues.append(
            ValidationIssue(
                "error",
                "negative_outcome",
                "Outcome values must be non-negative.",
            )
        )

    if (frame[list(mapping.channel_cols)] < 0).any().any():
        issues.append(
            ValidationIssue(
                "error",
                "negative_media",
                "Media spend values must be non-negative.",
            )
        )

    constant = [
        column
        for column in (mapping.outcome_col, *mapping.channel_cols)
        if frame[column].nunique(dropna=True) < 2
    ]
    if constant:
        issues.append(
            ValidationIssue(
                "error",
                "constant_columns",
                f"Columns need variation: {', '.join(constant)}",
            )
        )

    if len(mapping.channel_cols) >= 2 and len(frame) > 1:
        media_corr = frame[list(mapping.channel_cols)].corr().abs()
        media_corr_values = media_corr.to_numpy(copy=True)
        np.fill_diagonal(media_corr_values, 0.0)
        if (media_corr_values > 0.98).any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "high_collinearity",
                    "Two media channels are nearly identical, so their separate effects may be unstable.",
                )
            )

    rename = {
        mapping.date_col: "date",
        mapping.outcome_col: "outcome",
    }
    rename.update({name: f"media__{name}" for name in mapping.channel_cols})
    rename.update({name: f"control__{name}" for name in mapping.control_cols})
    frame = frame.rename(columns=rename).reset_index(drop=True)

    return PreparedData(frame, mapping, issues)

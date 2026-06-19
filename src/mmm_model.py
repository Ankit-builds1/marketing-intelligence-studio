from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.transformations import FeatureSet, hill_saturation


@dataclass(frozen=True)
class MMMConfig:
    holdout_fraction: float = 0.20
    alpha: float = 1.0
    bootstrap_samples: int = 50
    random_state: int = 42


@dataclass(frozen=True)
class ModelMetrics:
    r_squared: float
    mae: float
    mape: float | None
    holdout_rows: int


@dataclass
class MMMResult:
    coefficients: pd.Series
    intercept: float
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    channel_feature_names: dict[str, str]
    transforms: dict
    fitted: pd.Series
    holdout_predictions: pd.Series
    metrics: ModelMetrics
    baseline_component: pd.Series
    channel_contributions: pd.DataFrame
    channel_summary: pd.DataFrame
    response_curves: dict[str, pd.DataFrame]
    reliability: str
    warnings: list[str]


def _fit_constrained(
    X: pd.DataFrame,
    y: pd.Series,
    channel_columns: set[str],
    alpha: float,
) -> tuple[float, pd.Series, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    design = np.column_stack([np.ones(len(Xs)), Xs])

    if alpha > 0:
        penalty = np.sqrt(alpha) * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        augmented_X = np.vstack([design, penalty])
        augmented_y = np.concatenate([y.to_numpy(dtype=float), np.zeros(design.shape[1])])
    else:
        augmented_X = design
        augmented_y = y.to_numpy(dtype=float)

    lower = np.full(design.shape[1], -np.inf, dtype=float)
    lower[0] = -np.inf
    for index, column in enumerate(X.columns, start=1):
        if column in channel_columns:
            lower[index] = 0.0

    solved = lsq_linear(
        augmented_X,
        augmented_y,
        bounds=(lower, np.full(design.shape[1], np.inf, dtype=float)),
        lsmr_tol="auto",
    )
    if not solved.success:
        raise RuntimeError(f"Constrained MMM fit failed: {solved.message}")

    return float(solved.x[0]), pd.Series(solved.x[1:], index=X.columns, dtype=float), scaler


def _resolve_holdout_rows(row_count: int, holdout_fraction: float) -> int:
    if row_count < 2:
        raise ValueError("fit_mmm requires at least two rows")

    requested = max(8, round(row_count * holdout_fraction))
    return min(max(requested, 1), row_count - 1)


def _predict(
    X: pd.DataFrame,
    intercept: float,
    coefficients: pd.Series,
    scaler: StandardScaler,
) -> pd.Series:
    transformed = scaler.transform(X.loc[:, coefficients.index])
    values = intercept + transformed @ coefficients.to_numpy(dtype=float)
    return pd.Series(values, index=X.index, name="prediction", dtype=float)


def _standardized_frame(X: pd.DataFrame, scaler: StandardScaler, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    transformed = scaler.transform(X.loc[:, feature_columns])
    return pd.DataFrame(transformed, index=X.index, columns=feature_columns, dtype=float)


def _moving_block_bootstrap_indices(row_count: int, rng: np.random.Generator) -> np.ndarray:
    if row_count == 1:
        return np.array([0], dtype=int)

    block_length = max(2, min(row_count, int(round(np.sqrt(row_count)))))
    positions: list[int] = []
    while len(positions) < row_count:
        start = int(rng.integers(0, row_count))
        block = (start + np.arange(block_length, dtype=int)) % row_count
        positions.extend(block.tolist())
    return np.asarray(positions[:row_count], dtype=int)


def _business_contribution_total(
    X: pd.DataFrame,
    coefficients: pd.Series,
    scaler: StandardScaler,
    feature_column: str,
) -> float:
    scale = float(scaler.scale_[X.columns.get_loc(feature_column)])
    transformed_feature = X[feature_column].to_numpy(dtype=float)
    return float(np.sum((transformed_feature / scale) * float(coefficients[feature_column])))


def _build_response_curves(
    features: FeatureSet,
    coefficients: pd.Series,
    scaler: StandardScaler,
) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    feature_columns = tuple(features.X.columns)

    for channel, feature_column in features.channel_feature_names.items():
        transform = features.transforms[channel]
        max_spend = float(features.original_media[channel].max()) if len(features.original_media) else 0.0
        spend = np.linspace(0.0, max_spend * 1.5, 50, dtype=float)
        transformed_spend = hill_saturation(spend, transform.half_saturation, transform.slope)
        scale = float(scaler.scale_[feature_columns.index(feature_column)])
        incremental_outcome = (transformed_spend / scale) * float(coefficients[feature_column])
        curves[channel] = pd.DataFrame(
            {
                "spend": spend,
                "incremental_outcome": incremental_outcome,
            }
        )

    return curves


def _bootstrap_intervals(
    full_X: pd.DataFrame,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    channel_feature_names: dict[str, str],
    alpha: float,
    bootstrap_samples: int,
    random_state: int,
) -> dict[str, tuple[float, float]]:
    if bootstrap_samples <= 0:
        return {
            channel: (np.nan, np.nan)
            for channel in channel_feature_names
        }

    channel_columns = set(channel_feature_names.values())
    samples: dict[str, list[float]] = {channel: [] for channel in channel_feature_names}
    rng = np.random.default_rng(random_state)

    for _ in range(int(bootstrap_samples)):
        boot_index = _moving_block_bootstrap_indices(len(train_X), rng)
        intercept, coefficients, scaler = _fit_constrained(
            train_X.iloc[boot_index],
            train_y.iloc[boot_index],
            channel_columns,
            alpha,
        )
        _ = intercept
        for channel, feature_column in channel_feature_names.items():
            samples[channel].append(
                _business_contribution_total(full_X, coefficients, scaler, feature_column)
            )

    return {
        channel: tuple(float(value) for value in np.quantile(sample_values, [0.025, 0.975]))
        for channel, sample_values in samples.items()
    }


def _interval_crosses_zero(bounds: tuple[float, float]) -> bool:
    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return low <= 0.0 <= high or np.isclose(low, 0.0, atol=1e-9, rtol=0.0)


def fit_mmm(features: FeatureSet, config: MMMConfig) -> MMMResult:
    X = features.X.astype(float).copy()
    y = features.y.astype(float).copy()
    feature_columns = tuple(X.columns)
    channel_columns = set(features.channel_feature_names.values())

    holdout_rows = _resolve_holdout_rows(len(X), config.holdout_fraction)
    train_end = len(X) - holdout_rows
    train_X, holdout_X = X.iloc[:train_end], X.iloc[train_end:]
    train_y, holdout_y = y.iloc[:train_end], y.iloc[train_end:]

    train_intercept, train_coefficients, train_scaler = _fit_constrained(
        train_X,
        train_y,
        channel_columns,
        config.alpha,
    )
    holdout_predictions = _predict(holdout_X, train_intercept, train_coefficients, train_scaler)

    warnings: list[str] = []
    holdout_r_squared = float(r2_score(holdout_y, holdout_predictions))
    holdout_mae = float(mean_absolute_error(holdout_y, holdout_predictions))
    holdout_mape: float | None
    if (holdout_y == 0).any():
        holdout_mape = None
        warnings.append("MAPE is undefined because holdout outcomes include zero.")
    else:
        holdout_mape = float(np.mean(np.abs((holdout_y - holdout_predictions) / holdout_y)) * 100.0)

    if holdout_r_squared < 0.0:
        warnings.append("Holdout R^2 is negative; the model underperforms a naive baseline.")

    metrics = ModelMetrics(
        r_squared=holdout_r_squared,
        mae=holdout_mae,
        mape=holdout_mape,
        holdout_rows=holdout_rows,
    )

    intercept, coefficients, scaler = _fit_constrained(X, y, channel_columns, config.alpha)
    fitted = _predict(X, intercept, coefficients, scaler)
    standardized = _standardized_frame(X, scaler, feature_columns)

    channel_contributions = pd.DataFrame(index=X.index)
    for channel, feature_column in features.channel_feature_names.items():
        channel_contributions[channel] = standardized[feature_column] * float(coefficients[feature_column])
    channel_contributions = channel_contributions.astype(float)

    baseline_component = (fitted - channel_contributions.sum(axis=1)).rename("baseline_component")

    intervals = _bootstrap_intervals(
        full_X=X,
        train_X=train_X,
        train_y=train_y,
        channel_feature_names=features.channel_feature_names,
        alpha=config.alpha,
        bootstrap_samples=config.bootstrap_samples,
        random_state=config.random_state,
    )

    channel_summary_rows: list[dict[str, float | str]] = []
    unstable_intervals = False
    for channel, feature_column in features.channel_feature_names.items():
        contribution = _business_contribution_total(X, coefficients, scaler, feature_column)
        spend = float(features.original_media[channel].sum())
        ci_low, ci_high = intervals[channel]
        unstable_intervals = unstable_intervals or _interval_crosses_zero((ci_low, ci_high))
        channel_summary_rows.append(
            {
                "channel": channel,
                "coefficient": float(coefficients[feature_column]),
                "contribution": contribution,
                "spend": spend,
                "roi": float(contribution / spend) if spend else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if unstable_intervals:
        warnings.append("Bootstrap intervals cross zero for one or more channels.")

    if holdout_r_squared >= 0.6 and not unstable_intervals:
        reliability = "High"
    elif holdout_r_squared >= 0.2:
        reliability = "Medium"
    else:
        reliability = "Low"

    return MMMResult(
        coefficients=coefficients,
        intercept=intercept,
        scaler=scaler,
        feature_columns=feature_columns,
        channel_feature_names=dict(features.channel_feature_names),
        transforms=dict(features.transforms),
        fitted=fitted,
        holdout_predictions=holdout_predictions,
        metrics=metrics,
        baseline_component=baseline_component,
        channel_contributions=channel_contributions,
        channel_summary=pd.DataFrame(channel_summary_rows),
        response_curves=_build_response_curves(features, coefficients, scaler),
        reliability=reliability,
        warnings=warnings,
    )


def predict_from_feature_row(result: MMMResult, row: pd.Series) -> float:
    missing = [column for column in result.feature_columns if column not in row.index]
    if missing:
        raise KeyError(f"Missing feature columns for prediction: {missing}")

    ordered = row.loc[list(result.feature_columns)].to_numpy(dtype=float)
    standardized = (ordered - result.scaler.mean_) / result.scaler.scale_
    return float(result.intercept + np.dot(standardized, result.coefficients.to_numpy(dtype=float)))

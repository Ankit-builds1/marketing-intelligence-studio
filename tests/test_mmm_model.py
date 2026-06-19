import numpy as np
import pandas as pd

from src.transformations import ChannelTransform, build_feature_set


def _make_frame(rows: int, seed: int, *, include_social_effect: bool = True, corrupt_holdout: bool = False) -> tuple[pd.DataFrame, dict[str, ChannelTransform]]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=rows, freq="W-MON"),
            "media__search": rng.uniform(20.0, 100.0, rows),
            "media__social": rng.uniform(10.0, 60.0, rows),
            "control__price": rng.normal(1.0, 0.05, rows),
            "outcome": np.zeros(rows, dtype=float),
        }
    )
    params = {
        "search": ChannelTransform(0.2, 0.6, 1.0),
        "social": ChannelTransform(0.1, 0.5, 1.0),
    }
    preview = build_feature_set(frame, params)
    outcome = (
        250.0
        + 18.0 * preview.X["control__price"]
        + 40.0 * preview.X["media__search"]
        + (22.0 * preview.X["media__social"] if include_social_effect else 0.0)
        + np.random.default_rng(seed + 99).normal(0.0, 0.25, rows)
    )
    frame["outcome"] = outcome

    if corrupt_holdout:
        holdout_rows = max(8, round(rows * 0.2))
        frame.loc[frame.index[-holdout_rows:], "outcome"] = np.resize(
            np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0], dtype=float),
            holdout_rows,
        )

    return frame, params


def test_model_produces_holdout_metrics_reconciled_components_and_response_curves():
    from src.mmm_model import MMMConfig, fit_mmm, predict_from_feature_row

    frame, params = _make_frame(rows=120, seed=7)
    features = build_feature_set(frame, params)

    result = fit_mmm(features, MMMConfig(bootstrap_samples=12, random_state=7))

    assert result.metrics.holdout_rows == 24
    assert result.metrics.r_squared > 0.7
    assert np.isfinite(result.metrics.mae)
    assert result.metrics.mape is not None
    assert tuple(result.feature_columns) == tuple(features.X.columns)
    assert set(result.channel_summary["channel"]) == {"search", "social"}
    assert list(result.channel_summary.columns) == [
        "channel",
        "coefficient",
        "contribution",
        "spend",
        "roi",
        "ci_low",
        "ci_high",
    ]
    assert (result.channel_summary["coefficient"] >= 0).all()
    assert result.reliability == "High"
    assert result.warnings == []
    assert set(result.response_curves) == {"search", "social"}
    assert all(len(curve) == 50 for curve in result.response_curves.values())
    assert all(curve.columns.tolist() == ["spend", "incremental_outcome"] for curve in result.response_curves.values())
    assert all(curve["spend"].iloc[0] == 0.0 for curve in result.response_curves.values())
    np.testing.assert_allclose(
        result.fitted.to_numpy(),
        (result.baseline_component + result.channel_contributions.sum(axis=1)).to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.channel_summary.set_index("channel")["spend"].sort_index().to_numpy(),
        features.original_media.sum(axis=0).sort_index().to_numpy(),
    )
    assert np.isclose(
        predict_from_feature_row(result, features.X.iloc[0]),
        result.fitted.iloc[0],
    )


def test_fit_mmm_emits_deterministic_warnings_for_zero_mape_negative_holdout_and_unstable_intervals():
    from src.mmm_model import MMMConfig, fit_mmm

    frame, params = _make_frame(rows=60, seed=11, include_social_effect=False, corrupt_holdout=True)
    features = build_feature_set(frame, params)
    config = MMMConfig(bootstrap_samples=16, random_state=3)

    first = fit_mmm(features, config)
    second = fit_mmm(features, config)

    assert first.metrics.holdout_rows == 12
    assert first.metrics.mape is None
    assert first.metrics.r_squared < 0.0
    assert first.reliability == "Low"
    assert any("MAPE" in warning for warning in first.warnings)
    assert any("Holdout R^2" in warning for warning in first.warnings)
    assert any("interval" in warning.lower() for warning in first.warnings)
    pd.testing.assert_frame_equal(first.channel_summary, second.channel_summary)
    assert first.warnings == second.warnings

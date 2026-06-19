import numpy as np
import pandas as pd

from src.transformations import (
    ChannelTransform,
    build_feature_set,
    geometric_adstock,
    hill_saturation,
)


def test_geometric_adstock_carries_effect_forward():
    result = geometric_adstock(np.array([10.0, 0.0, 0.0]), decay=0.5)
    np.testing.assert_allclose(result, [10.0, 5.0, 2.5])


def test_hill_saturation_hits_half_at_half_saturation():
    result = hill_saturation(np.array([0.0, 10.0, 100.0]), half_saturation=10.0, slope=1.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 100.0 / 110.0])


def test_feature_set_tracks_channel_columns_and_original_media():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="W-MON"),
            "outcome": np.arange(8.0),
            "media__search": np.arange(8.0),
            "media__social": np.arange(8.0)[::-1],
            "control__price": np.ones(8),
        }
    )
    params = {name: ChannelTransform(0.5, 3.0, 1.0) for name in ("search", "social")}
    features = build_feature_set(frame, params)

    assert features.channel_feature_names == {
        "search": "media__search",
        "social": "media__social",
    }
    assert {"trend", "sin_52", "cos_52", "control__price"}.issubset(features.X.columns)
    assert list(features.original_media.columns) == ["search", "social"]
    pd.testing.assert_series_equal(features.y, frame["outcome"].astype(float), check_names=False)
    pd.testing.assert_series_equal(features.dates, frame["date"], check_names=False)

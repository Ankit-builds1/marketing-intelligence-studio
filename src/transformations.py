from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelTransform:
    decay: float = 0.5
    half_saturation: float = 1.0
    slope: float = 1.0


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series
    channel_feature_names: dict[str, str]
    transforms: dict[str, ChannelTransform]
    original_media: pd.DataFrame


def geometric_adstock(values: np.ndarray, decay: float) -> np.ndarray:
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0, 1)")

    array = np.asarray(values, dtype=float)
    result = np.zeros(len(array), dtype=float)
    for index, value in enumerate(array):
        result[index] = value + (result[index - 1] * decay if index else 0.0)
    return result


def hill_saturation(values: np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    if half_saturation <= 0 or slope <= 0:
        raise ValueError("half_saturation and slope must be positive")

    safe_values = np.maximum(np.asarray(values, dtype=float), 0.0)
    powered = np.power(safe_values, slope)
    denominator = powered + half_saturation**slope
    return np.divide(powered, denominator, out=np.zeros_like(powered), where=denominator != 0)


def build_feature_set(frame: pd.DataFrame, transforms: dict[str, ChannelTransform]) -> FeatureSet:
    X = pd.DataFrame(index=frame.index)
    channel_feature_names: dict[str, str] = {}

    for channel, config in transforms.items():
        source_column = f"media__{channel}"
        adstocked = geometric_adstock(frame[source_column].to_numpy(), config.decay)
        X[source_column] = hill_saturation(adstocked, config.half_saturation, config.slope)
        channel_feature_names[channel] = source_column

    for column in frame.columns:
        if column.startswith("control__"):
            control_values = frame[column].interpolate(limit_direction="both")
            X[column] = control_values.fillna(control_values.median())

    index = np.arange(len(frame), dtype=float)
    X["trend"] = index / max(len(frame) - 1, 1)
    X["sin_52"] = np.sin(2 * np.pi * index / 52.0)
    X["cos_52"] = np.cos(2 * np.pi * index / 52.0)

    original_media = frame[[f"media__{name}" for name in transforms]].copy()
    original_media.columns = list(transforms)

    return FeatureSet(
        X=X,
        y=frame["outcome"].astype(float),
        dates=frame["date"],
        channel_feature_names=channel_feature_names,
        transforms=transforms,
        original_media=original_media,
    )

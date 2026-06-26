from dataclasses import dataclass

import pandas as pd

from src.analysis_router import AnalysisCapabilities, route_analysis
from src.cadence import detect_cadence, minimum_periods
from src.data_validation import DataMapping, PreparedData, prepare_and_validate
from src.dataset_profiler import DatasetProfile, profile_dataset
from src.event_preprocessor import PreparationRequest, PreparationResult, prepare_marketing_data
from src.reliability import ReliabilityAssessment, assess_reliability
from src.schema_detection import SchemaDetection, detect_schema


@dataclass
class AdaptiveDataset:
    original: pd.DataFrame
    analysis_frame: pd.DataFrame
    mapping: DataMapping
    profile: DatasetProfile
    detection: SchemaDetection
    preparation: PreparationResult | None
    prepared: PreparedData
    capabilities: AnalysisCapabilities
    reliability: ReliabilityAssessment
    messages: tuple[str, ...]

    @property
    def is_transformed(self) -> bool:
        return self.preparation is not None


def _wide_mapping(detection: SchemaDetection) -> DataMapping:
    return DataMapping(
        detection.suggested_date,
        detection.suggested_outcome,
        tuple(detection.suggested_channels),
        tuple(
            column
            for column in detection.suggested_controls
            if column
            not in {
                detection.suggested_date,
                detection.suggested_outcome,
                *detection.suggested_channels,
            }
        ),
    )


def _choose_cadence(frame: pd.DataFrame, date_col: str) -> str:
    info = detect_cadence(frame[date_col])
    if info.kind in {"daily", "weekly", "monthly"}:
        return info.kind
    return "weekly"


def build_adaptive_dataset(frame: pd.DataFrame) -> AdaptiveDataset:
    profile = profile_dataset(frame)
    detection = detect_schema(frame, profile)
    messages: list[str] = [*detection.reasons]
    preparation: PreparationResult | None = None
    analysis_frame = frame.copy()

    if detection.family in {"event_attribution", "long_campaign"}:
        cadence = "daily"
        preparation = prepare_marketing_data(
            frame,
            PreparationRequest(
                date_col=detection.suggested_date,
                outcome_col=detection.suggested_outcome,
                spend_col=detection.suggested_spend,
                channel_col=detection.suggested_channel_dimension,
                cadence=cadence,
                control_cols=(),
                timestamp_origin="1970-01-01" if detection.suggested_date == "timestamp" else None,
                max_channels=8,
            ),
        )
        analysis_frame = preparation.frame
        mapping = preparation.mapping
        messages.extend(preparation.transformations)
    else:
        mapping = _wide_mapping(detection)
        if mapping.date_col:
            cadence = _choose_cadence(frame, mapping.date_col)
        else:
            cadence = "weekly"

    if detection.family == "unsupported" or not mapping.date_col or not mapping.outcome_col:
        prepared = PreparedData(pd.DataFrame(), mapping, [], cadence=cadence, minimum_rows=minimum_periods(cadence))
        capabilities = route_analysis(detection.family, 0, False, 0, cadence)
        reliability = assess_reliability(0, minimum_periods(cadence), 1)
        return AdaptiveDataset(frame, analysis_frame, mapping, profile, detection, preparation, prepared, capabilities, reliability, tuple(messages))

    prepared = prepare_and_validate(
        analysis_frame,
        mapping,
        min_rows=minimum_periods(cadence),
        target_cadence=cadence,
    )
    has_spend = bool(mapping.channel_cols)
    capabilities = route_analysis(
        detection.family,
        len(mapping.channel_cols),
        has_spend,
        len(prepared.frame),
        prepared.cadence,
    )
    issue_count = len([issue for issue in prepared.issues if issue.severity == "error"])
    reliability = assess_reliability(len(prepared.frame), prepared.minimum_rows, issue_count)
    return AdaptiveDataset(frame, analysis_frame, mapping, profile, detection, preparation, prepared, capabilities, reliability, tuple(messages))

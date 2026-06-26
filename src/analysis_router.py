from dataclasses import dataclass

from src.cadence import minimum_periods


@dataclass(frozen=True)
class AnalysisCapabilities:
    workflow: str
    can_train: bool
    can_show_roi: bool
    can_optimize: bool
    reason: str


def route_analysis(family: str, channel_count: int, has_monetary_spend: bool,
                   period_count: int, cadence: str) -> AnalysisCapabilities:
    enough = period_count >= minimum_periods(cadence)
    if not enough:
        return AnalysisCapabilities("exploratory", False, False, False,
                                    f"Need {minimum_periods(cadence)} {cadence} periods; found {period_count}.")
    if channel_count >= 2:
        return AnalysisCapabilities("full_mmm", True, has_monetary_spend, has_monetary_spend,
                                    "Multiple media channels and sufficient history are available.")
    if channel_count == 1:
        return AnalysisCapabilities("single_channel", True, has_monetary_spend, False,
                                    "One media channel supports response analysis but not allocation optimization.")
    return AnalysisCapabilities("compatibility", False, False, False, "No usable media channel was identified.")

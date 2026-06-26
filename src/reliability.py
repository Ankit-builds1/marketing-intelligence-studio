from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityAssessment:
    label: str
    score: int
    explanation: str


def assess_reliability(period_count: int, minimum_required: int, issue_count: int) -> ReliabilityAssessment:
    if period_count < minimum_required:
        return ReliabilityAssessment("Insufficient", 20, "History is shorter than the modelling requirement.")
    ratio = period_count / max(minimum_required, 1)
    score = max(0, min(100, round(55 + min(ratio - 1, 1) * 35 - issue_count * 10)))
    label = "Strong" if score >= 80 else "Directional"
    return ReliabilityAssessment(label, score, "Score reflects history length and detected data-quality issues.")

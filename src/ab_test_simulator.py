"""
Simulate a randomized A/B test to validate causal estimates.

Idea: pretend we ran an RCT where treatment was randomly assigned
(uncorrelated with confounders). Compare the observed effect to the
ATE estimated by DoWhy on the *observational* data. If they agree,
the causal model is well-specified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ABTestResult:
    observed_effect: float
    p_value: float
    n_treatment: int
    n_control: int
    treatment_mean: float
    control_mean: float


def simulate_ab_test(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    threshold: float | None = None,
    seed: int = 42,
) -> ABTestResult:
    """Naive A/B split: bucket rows above the treatment-column median into
    'treatment' and below into 'control', then compare outcome means.

    This is intentionally simple — it captures the *correlational* effect,
    which (because of confounding) should differ from the *causal* ATE.
    The gap between them is the value-add of causal inference.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    if threshold is None:
        threshold = float(df[treatment_col].median())

    is_treated = df[treatment_col] > threshold
    treated = df.loc[is_treated, outcome_col].values
    control = df.loc[~is_treated, outcome_col].values

    t_stat, p_val = stats.ttest_ind(treated, control, equal_var=False)

    return ABTestResult(
        observed_effect=float(treated.mean() - control.mean()),
        p_value=float(p_val),
        n_treatment=int(is_treated.sum()),
        n_control=int((~is_treated).sum()),
        treatment_mean=float(treated.mean()),
        control_mean=float(control.mean()),
    )


def simulate_randomised_experiment(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    true_effect: float,
    n_iter: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap a randomised experiment: for each iteration, randomly
    re-assign 'treatment' status, inject a known causal effect, then
    measure the recovered effect. Returns distribution stats.

    Useful for sanity-checking estimators: if you inject effect=10
    and the estimator recovers ~10, it works.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    effects: list[float] = []

    for _ in range(n_iter):
        assignment = rng.integers(0, 2, n)
        outcome = df[outcome_col].values + true_effect * assignment
        treated = outcome[assignment == 1]
        control = outcome[assignment == 0]
        effects.append(float(treated.mean() - control.mean()))

    arr = np.asarray(effects)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "true_effect": float(true_effect),
        "distribution": arr,
    }

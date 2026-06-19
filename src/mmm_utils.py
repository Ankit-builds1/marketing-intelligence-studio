"""
Reusable causal-inference utilities for the Marketing Mix Modeling project.

Imported by both:
  - notebooks/causal_inference_mmm.ipynb
  - app/budget_optimizer_app.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Causal DAG construction
# ---------------------------------------------------------------------------
def build_causal_dag(
    treatments: Sequence[str],
    outcome: str,
    confounders: Sequence[str],
) -> tuple[str, nx.DiGraph]:
    """Build a DAG in DoWhy's GML-string format and as a networkx DiGraph.

    Each confounder is wired to each treatment AND to the outcome,
    and each treatment is wired to the outcome.
    """
    nodes = list(dict.fromkeys([*treatments, outcome, *confounders]))
    edges: list[tuple[str, str]] = []
    for t in treatments:
        edges.append((t, outcome))
    for c in confounders:
        edges.append((c, outcome))
        for t in treatments:
            edges.append((c, t))

    # GML graph string compatible with DoWhy
    node_lines = "\n".join([f'    node [ id "{n}" label "{n}" ]' for n in nodes])
    edge_lines = "\n".join(
        [f'    edge [ source "{a}" target "{b}" ]' for a, b in edges]
    )
    gml = f"graph [\n  directed 1\n{node_lines}\n{edge_lines}\n]"

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return gml, g


def plot_dag(g: nx.DiGraph, treatments: Sequence[str], outcome: str,
             confounders: Sequence[str], ax=None, title: str = "Causal DAG"):
    """Plot the causal DAG with role-coloured nodes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    pos = nx.spring_layout(g, seed=42, k=1.5)

    color_map = []
    for node in g.nodes():
        if node == outcome:
            color_map.append("#ff6b6b")          # outcome - red
        elif node in treatments:
            color_map.append("#4ecdc4")          # treatment - teal
        elif node in confounders:
            color_map.append("#ffd93d")          # confounder - yellow
        else:
            color_map.append("#a0a0a0")

    nx.draw(
        g, pos, ax=ax, with_labels=True, node_color=color_map,
        node_size=2500, font_size=9, font_weight="bold",
        arrows=True, arrowsize=20, edge_color="#555",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#4ecdc4", label="Treatment"),
        Patch(facecolor="#ff6b6b", label="Outcome"),
        Patch(facecolor="#ffd93d", label="Confounder"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# 2. ATE estimation via DoWhy
# ---------------------------------------------------------------------------
@dataclass
class ATEResult:
    method: str
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    raw: object = None  # underlying DoWhy estimate object


def estimate_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str],
    graph_gml: str,
    method: str = "backdoor.linear_regression",
) -> ATEResult:
    """Estimate ATE with DoWhy using a backdoor-adjustment method."""
    from dowhy import CausalModel

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=list(confounders),
        graph=graph_gml,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(
        identified,
        method_name=method,
        test_significance=True,
        confidence_intervals=True,
    )

    ci_low, ci_high = None, None
    try:
        ci = est.get_confidence_intervals()
        # DoWhy returns various shapes depending on version
        if hasattr(ci, "__iter__"):
            ci_arr = np.asarray(ci).flatten()
            if ci_arr.size >= 2:
                ci_low, ci_high = float(ci_arr[0]), float(ci_arr[1])
    except Exception:
        pass

    return ATEResult(
        method=method,
        estimate=float(est.value),
        ci_low=ci_low,
        ci_high=ci_high,
        raw=est,
    )


# ---------------------------------------------------------------------------
# 3. Refutation tests
# ---------------------------------------------------------------------------
def run_refutation_tests(model, identified_estimand, estimate) -> dict[str, dict]:
    """Run DoWhy's standard refuters; return summary dict.

    Pass the same ``model`` and ``identified_estimand`` you used for the estimate,
    plus the estimate object itself.
    """
    results: dict[str, dict] = {}

    refuter_specs = [
        ("placebo_treatment", "placebo_treatment_refuter",
         {"placebo_type": "permute"}),
        ("random_common_cause", "random_common_cause", {}),
        ("data_subset", "data_subset_refuter",
         {"subset_fraction": 0.8}),
    ]

    for label, method_name, kwargs in refuter_specs:
        try:
            ref = model.refute_estimate(
                identified_estimand, estimate,
                method_name=method_name, **kwargs,
            )
            results[label] = {
                "new_effect": float(ref.new_effect) if ref.new_effect is not None else None,
                "p_value": getattr(ref, "refutation_result", None),
                "raw": ref,
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 4. Double Machine Learning (counterfactuals & heterogeneous effects)
# ---------------------------------------------------------------------------
def estimate_dml(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str],
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Fit EconML LinearDML and return the fitted estimator."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    Y = df[outcome].values
    T = df[treatment].values
    X = df[list(confounders)].values

    dml = LinearDML(
        model_y=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        model_t=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        random_state=random_state,
    )
    dml.fit(Y, T, X=X, W=None)
    return dml


def estimate_causal_forest(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str],
    n_estimators: int = 200,
    random_state: int = 42,
):
    """Fit EconML CausalForestDML for heterogeneous treatment effects."""
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor

    Y = df[outcome].values
    T = df[treatment].values
    X = df[list(confounders)].values

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=random_state),
        model_t=RandomForestRegressor(n_estimators=100, random_state=random_state),
        n_estimators=n_estimators,
        random_state=random_state,
    )
    cf.fit(Y, T, X=X, W=None)
    return cf


# ---------------------------------------------------------------------------
# 5. Budget optimization
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    optimal_allocation: dict[str, float]
    predicted_sales: float
    total_spend: float
    success: bool
    message: str
    raw: object = None


def predict_sales(
    allocation: dict[str, float],
    channel_effects: dict[str, float],
    baseline: float,
) -> float:
    """Linear-additive sales prediction:  Sales = baseline + sum(beta_i * spend_i)."""
    return float(baseline + sum(channel_effects.get(c, 0.0) * v for c, v in allocation.items()))


def optimize_budget(
    channel_effects: dict[str, float],
    total_budget: float,
    channel_bounds: dict[str, tuple[float, float]] | None = None,
    baseline: float = 0.0,
) -> OptimizationResult:
    """Find the spend allocation across channels that maximises predicted sales
    subject to:
      - sum(spend) == total_budget
      - min <= spend_i <= max  (per-channel bounds)
    """
    channels = list(channel_effects.keys())
    n = len(channels)

    if channel_bounds is None:
        channel_bounds = {c: (0.0, total_budget) for c in channels}
    bounds = [channel_bounds[c] for c in channels]

    effects = np.array([channel_effects[c] for c in channels])

    # scipy.minimize minimises - so negate
    def neg_sales(x: np.ndarray) -> float:
        return -float(np.dot(effects, x))

    def grad(x: np.ndarray) -> np.ndarray:
        return -effects

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]

    # Warm start: even allocation respecting bounds
    x0 = np.full(n, total_budget / n)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    # Re-normalise if clipping broke the sum
    if x0.sum() > 0:
        x0 = x0 * (total_budget / x0.sum())

    res = minimize(
        neg_sales, x0, jac=grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 500},
    )

    optimal = {c: float(v) for c, v in zip(channels, res.x)}
    pred = predict_sales(optimal, channel_effects, baseline)
    return OptimizationResult(
        optimal_allocation=optimal,
        predicted_sales=pred,
        total_spend=float(sum(optimal.values())),
        success=bool(res.success),
        message=str(res.message),
        raw=res,
    )


# ---------------------------------------------------------------------------
# 6. Convenience: fit a simple linear MMM and extract channel coefficients
# ---------------------------------------------------------------------------
def fit_linear_mmm(
    df: pd.DataFrame,
    channels: Iterable[str],
    outcome: str,
    confounders: Iterable[str] = (),
) -> dict:
    """Fit OLS Sales ~ channels + confounders. Returns coefficients dict
    plus baseline intercept — the shape expected by ``optimize_budget``
    and ``predict_sales``.
    """
    import statsmodels.api as sm

    feature_cols = list(channels) + list(confounders)
    X = sm.add_constant(df[feature_cols].astype(float))
    y = df[outcome].astype(float)
    model = sm.OLS(y, X).fit()

    return {
        "channel_effects": {c: float(model.params[c]) for c in channels},
        "confounder_effects": {c: float(model.params[c]) for c in confounders},
        "baseline": float(model.params["const"]),
        "summary": model.summary().as_text(),
        "r_squared": float(model.rsquared),
        "raw_model": model,
    }

"""
Streamlit Budget Optimizer Dashboard
====================================

Interactive what-if interface backed by the causal-inference model trained in
notebooks/causal_inference_mmm.ipynb. Drag the per-channel sliders to see live
predicted sales; click "Find Optimal Allocation" for the constrained-optimum.

Run with:
    streamlit run app/budget_optimizer_app.py

The notebook MUST be run first to generate models/mmm_artifacts.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path so we can import src.mmm_utils
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mmm_utils as mu  # noqa: E402

ARTIFACTS_PATH = ROOT / "models" / "mmm_artifacts.pkl"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MMM Budget Optimizer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("Marketing Mix Modeling — Budget Optimizer")
st.caption(
    "Powered by a causal-inference model (DoWhy + EconML). "
    "Drag the sliders to explore what-if budget scenarios in real time."
)


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not ARTIFACTS_PATH.exists():
        return None
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        f"Model artifacts not found at `{ARTIFACTS_PATH}`. "
        "Run the notebook `notebooks/causal_inference_mmm.ipynb` first — "
        "it saves the channel coefficients used by this dashboard."
    )
    st.stop()

channel_effects: dict[str, float] = artifacts["channel_effects"]
baseline: float = artifacts["baseline"]
current_alloc: dict[str, float] = artifacts["current_alloc"]
total_budget_default: float = artifacts["total_budget"]
default_bounds: dict[str, tuple[float, float]] = artifacts["bounds"]


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Budget Configuration")
total_budget = st.sidebar.slider(
    "Total weekly budget",
    min_value=float(total_budget_default * 0.5),
    max_value=float(total_budget_default * 2.0),
    value=float(total_budget_default),
    step=10.0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Per-channel min/max bounds")
bounds: dict[str, tuple[float, float]] = {}
for ch in channel_effects.keys():
    lo_default, hi_default = default_bounds.get(ch, (0.0, total_budget))
    col1, col2 = st.sidebar.columns(2)
    lo = col1.number_input(
        f"{ch} min",
        value=float(lo_default),
        min_value=0.0,
        step=1.0,
        key=f"min_{ch}",
    )
    hi = col2.number_input(
        f"{ch} max",
        value=float(min(hi_default, total_budget)),
        min_value=lo,
        step=1.0,
        key=f"max_{ch}",
    )
    bounds[ch] = (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Main: sliders for allocation + live KPIs
# ---------------------------------------------------------------------------
st.subheader("1. Try your own allocation")

slider_cols = st.columns(len(channel_effects))
allocation: dict[str, float] = {}

for col, ch in zip(slider_cols, channel_effects.keys()):
    lo, hi = bounds[ch]
    default_val = float(min(max(current_alloc.get(ch, lo), lo), hi))
    val = col.slider(
        ch,
        min_value=float(lo),
        max_value=float(hi),
        value=default_val,
        step=max(1.0, (hi - lo) / 100.0),
        key=f"slider_{ch}",
    )
    allocation[ch] = float(val)

total_spent = sum(allocation.values())
predicted_sales = mu.predict_sales(allocation, channel_effects, baseline)
current_pred = mu.predict_sales(current_alloc, channel_effects, baseline)
diff_pct = 100.0 * (predicted_sales - current_pred) / current_pred if current_pred else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total spent", f"{total_spent:,.0f}", delta=f"{total_spent - total_budget:+,.0f}")
kpi2.metric("Predicted sales", f"{predicted_sales:,.0f}", delta=f"{diff_pct:+.1f}%")
kpi3.metric("vs current", f"{predicted_sales - current_pred:+,.0f}")
kpi4.metric("Baseline (no spend)", f"{baseline:,.0f}")

if abs(total_spent - total_budget) > 1e-3:
    st.warning(
        f"Your manual allocation totals {total_spent:,.0f} but the budget is "
        f"{total_budget:,.0f}. Sliders aren't constrained — use the optimizer below to enforce it."
    )


# ---------------------------------------------------------------------------
# Allocation chart
# ---------------------------------------------------------------------------
chart_df = pd.DataFrame({
    "Channel": list(channel_effects.keys()),
    "Current": [current_alloc[c] for c in channel_effects.keys()],
    "Your choice": [allocation[c] for c in channel_effects.keys()],
})
chart_long = chart_df.melt(id_vars="Channel", var_name="Scenario", value_name="Spend")
fig = px.bar(
    chart_long, x="Channel", y="Spend", color="Scenario",
    barmode="group", title="Allocation: Current vs Your Choice",
)
st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Find the optimal allocation")
st.caption(
    "Solves a constrained optimization (`scipy.optimize.minimize`, SLSQP) — "
    "maximizes predicted Sales subject to `sum(spend) = total_budget` "
    "and per-channel bounds."
)

if st.button("Find Optimal Allocation", type="primary"):
    result = mu.optimize_budget(
        channel_effects=channel_effects,
        total_budget=total_budget,
        channel_bounds=bounds,
        baseline=baseline,
    )

    if not result.success:
        st.error(f"Optimization failed: {result.message}")
    else:
        st.success(f"Optimal predicted Sales: **{result.predicted_sales:,.0f}**")

        opt_df = pd.DataFrame({
            "Channel": list(result.optimal_allocation.keys()),
            "Current": [current_alloc[c] for c in result.optimal_allocation],
            "Optimal": list(result.optimal_allocation.values()),
        })

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=opt_df["Channel"],
                values=opt_df["Optimal"],
                hole=0.4,
            )])
            fig_pie.update_layout(title="Optimal Allocation")
            st.plotly_chart(fig_pie, width="stretch")
        with col2:
            opt_long = opt_df.melt(id_vars="Channel", var_name="Scenario", value_name="Spend")
            fig_bar = px.bar(
                opt_long, x="Channel", y="Spend", color="Scenario",
                barmode="group", title="Current vs Optimal",
            )
            st.plotly_chart(fig_bar, width="stretch")

        delta = result.predicted_sales - current_pred
        delta_pct = 100.0 * delta / current_pred if current_pred else 0.0
        st.info(
            f"Switching to the optimal allocation lifts predicted Sales by "
            f"**{delta:+,.0f}** ({delta_pct:+.2f}%) at the same total budget."
        )


# ---------------------------------------------------------------------------
# Model context (collapsed)
# ---------------------------------------------------------------------------
with st.expander("Model details — channel coefficients & ATE estimates"):
    coef_df = pd.DataFrame({
        "Channel": list(channel_effects.keys()),
        "Causal-adjusted coefficient": list(channel_effects.values()),
        "True (simulated)": [artifacts["true_coeffs"].get(c, "n/a") for c in channel_effects.keys()],
    })
    st.dataframe(coef_df, width="stretch")
    st.caption(
        f"Baseline (intercept): {baseline:,.2f}  |  Model R²: {artifacts['r_squared']:.4f}"
    )
    if "ate_estimates" in artifacts:
        st.subheader("DoWhy ATE estimates (Digital_spend)")
        ate_df = pd.DataFrame(list(artifacts["ate_estimates"].items()),
                              columns=["Method", "ATE"])
        st.dataframe(ate_df, width="stretch")

st.caption(
    "Built with DoWhy, EconML, scipy.optimize, and Streamlit. "
    "Project structure: notebooks/ -> src/mmm_utils.py -> app/budget_optimizer_app.py."
)

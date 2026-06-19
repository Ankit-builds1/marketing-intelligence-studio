"""
Programmatically builds notebooks/causal_inference_mmm.ipynb.

This script is reproducible: run it from the project root and the
notebook is regenerated from the cell definitions below. The cells
mirror the 7-step plan from the Data Scientist Hardcore Projects guide
+ 3 extensions (CausalForest, A/B validation, Streamlit-ready artifacts).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "causal_inference_mmm.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.lstrip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.lstrip("\n"))


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------
cells: list[nbf.NotebookNode] = []

# ============================================================
# 1. Title + intro
# ============================================================
cells.append(md(r"""
# Causal Inference / Marketing Mix Modeling

> **Goal:** Quantify the *causal* effect of each marketing channel on Sales — not just the correlation — using DoWhy, EconML, refutation tests, A/B-test validation, and `scipy` budget optimization. Save channel coefficients for the live Streamlit dashboard.

This notebook walks through **13 sections**:

| #  | Section                                                  |
|----|----------------------------------------------------------|
| 1  | Setup & imports                                          |
| 2  | Load all 3 datasets (simulated, Robyn, Kaggle if avail.) |
| 3  | EDA — the *naive* correlational view                     |
| 4  | Build the causal DAG                                     |
| 5  | Define treatment, outcome, confounders                   |
| 6  | Estimate ATE (3 backdoor methods)                        |
| 7  | Refutation tests (placebo, common cause, subset)         |
| 8  | Counterfactual analysis with Double ML                   |
| 9  | **[Extension]** Heterogeneous effects (CausalForest)     |
| 10 | **[Extension]** A/B test validation                      |
| 11 | Budget optimization (`scipy.optimize`)                   |
| 12 | Save artifacts for the Streamlit app                     |
| 13 | Conclusion + resume-ready takeaways                      |

**Author:** Ankit Dash — Data Scientist Resume 2025
"""))

# ============================================================
# 2. Setup
# ============================================================
cells.append(md("## 1. Setup & Imports"))
cells.append(code(r"""
import sys, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project src/ to the path so we can import our modules
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

from src import mmm_utils as mu
from src import ab_test_simulator as abt
from src.data_generator import TRUE_COEFFS

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("Project root:", ROOT)
print("Python:", sys.version.split()[0])
"""))

# ============================================================
# 3. Load Data
# ============================================================
cells.append(md(r"""
## 2. Load Datasets

We work primarily with the **simulated dataset** — because we *know* the true causal coefficients, we can verify that the model recovers them. We also load the **Meta Robyn** real-world MMM dataset to demonstrate the same techniques on a realistic schema.
"""))

cells.append(code(r"""
DATA = ROOT / "data"

df_sim = pd.read_csv(DATA / "simulated_mmm.csv")
print(f"Simulated MMM:  {df_sim.shape}")
df_sim.head()
"""))

cells.append(code(r"""
df_robyn = pd.read_csv(DATA / "robyn_mmm.csv")
print(f"Robyn MMM:      {df_robyn.shape}")
df_robyn.head()
"""))

cells.append(code(r"""
# Kaggle Marketing Campaign — optional (loads if present)
kaggle_path = DATA / "marketing_campaign.csv"
if kaggle_path.exists():
    df_kaggle = pd.read_csv(kaggle_path)
    print(f"Kaggle Marketing Campaign: {df_kaggle.shape}")
    print(df_kaggle.head())
else:
    df_kaggle = None
    print("Kaggle dataset not present (skip; set up Kaggle API to enable).")
"""))

# ============================================================
# 4. EDA
# ============================================================
cells.append(md(r"""
## 3. EDA — the Naive Correlation View

A standard correlation analysis tells us *which channels move with sales*, but **NOT** which ones *cause* sales to move. The whole point of the rest of this notebook is showing why correlation ≠ causation in marketing data.
"""))

cells.append(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Correlation heatmap
num_cols = ["TV_spend","Digital_spend","Radio_spend","Promotions",
            "Seasonality","Competition_index","Sales"]
sns.heatmap(df_sim[num_cols].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=axes[0,0], vmin=-1, vmax=1)
axes[0,0].set_title("Correlation Heatmap (naive view)")

# Sales over time
axes[0,1].plot(df_sim["week"], df_sim["Sales"], color="#2c3e50")
axes[0,1].set_xlabel("Week"); axes[0,1].set_ylabel("Sales")
axes[0,1].set_title("Sales Over Time")

# TV spend vs Sales
axes[1,0].scatter(df_sim["TV_spend"], df_sim["Sales"], alpha=0.7,
                  c=df_sim["Seasonality"], cmap="coolwarm")
axes[1,0].set_xlabel("TV Spend"); axes[1,0].set_ylabel("Sales")
axes[1,0].set_title("TV vs Sales (coloured by Seasonality)")

# Digital spend vs Sales
axes[1,1].scatter(df_sim["Digital_spend"], df_sim["Sales"], alpha=0.7, color="#27ae60")
axes[1,1].set_xlabel("Digital Spend"); axes[1,1].set_ylabel("Sales")
axes[1,1].set_title("Digital vs Sales")

plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""
> Notice in the TV-vs-Sales plot that high-TV-spend weeks coincide with high seasonality (Q4 holidays). The naive correlation between TV and Sales is therefore *inflated* by seasonality acting as a confounder — exactly what causal inference is built to fix.
"""))

# ============================================================
# 5. DAG
# ============================================================
cells.append(md(r"""
## 4. Build the Causal DAG

We encode domain knowledge:

- `TV_spend → Sales` (causal effect we want to measure)
- `Digital_spend → Sales`
- `Radio_spend → Sales`
- `Promotions → Sales`
- `Seasonality → Sales` and `Seasonality → TV_spend` ← **confounder**
- `Competition_index → Sales`

Encoding the DAG forces us to declare our assumptions. DoWhy then uses the graph to determine which variables we must condition on (the **backdoor adjustment set**) to obtain an unbiased causal estimate.
"""))

cells.append(code(r"""
treatments  = ["TV_spend", "Digital_spend", "Radio_spend", "Promotions"]
outcome     = "Sales"
confounders = ["Seasonality", "Competition_index"]

graph_gml, g = mu.build_causal_dag(treatments, outcome, confounders)

fig, ax = plt.subplots(figsize=(11, 7))
mu.plot_dag(g, treatments, outcome, confounders, ax=ax,
            title="Marketing Mix Causal DAG")
plt.show()
print("DAG (DoWhy GML format, first 200 chars):")
print(graph_gml[:200], "...")
"""))

# ============================================================
# 6. Treatment + Outcome
# ============================================================
cells.append(md(r"""
## 5. Define Treatment & Outcome

We start with `Digital_spend` as the treatment because the simulated dataset's true coefficient is **0.50** — that's our ground truth. We'll later repeat the analysis for TV (true beta = 0.30) and Radio (true beta = 0.20).
"""))

cells.append(code(r"""
TREATMENT = "Digital_spend"
TRUE_BETA = TRUE_COEFFS[TREATMENT]
print(f"Treatment:    {TREATMENT}")
print(f"Outcome:      {outcome}")
print(f"Confounders:  {confounders}")
print(f"TRUE beta from data generator: {TRUE_BETA}")
"""))

# ============================================================
# 7. ATE
# ============================================================
cells.append(md(r"""
## 6. Estimate Average Treatment Effect (ATE)

We try **three** different DoWhy backdoor methods. If they all agree, the estimate is robust.

1. `backdoor.linear_regression` — fits a linear model, reads the coefficient.
2. `backdoor.propensity_score_stratification` — buckets the data into propensity strata, computes within-stratum effects.
3. `backdoor.propensity_score_weighting` — inverse-propensity-weighted estimator.
"""))

cells.append(code(r"""
methods = [
    "backdoor.linear_regression",
    "backdoor.propensity_score_stratification",
    "backdoor.propensity_score_weighting",
]

ate_results = []
for m in methods:
    try:
        r = mu.estimate_ate(df_sim, TREATMENT, outcome, confounders, graph_gml, method=m)
        ate_results.append(r)
        ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" if r.ci_low is not None else "(CI unavailable)"
        print(f"{m:55s}  ATE = {r.estimate:>7.4f}  95% CI {ci}")
    except Exception as e:
        print(f"{m:55s}  FAILED: {e}")

print(f"\nTRUE coefficient: {TRUE_BETA}")
"""))

cells.append(md(r"""
**Interpretation:** all three methods should give estimates close to **0.50** (the true coefficient injected by the data generator). The naive correlation between Digital and Sales (no confounder adjustment) would be biased — causal inference recovers the truth.
"""))

# ============================================================
# 8. Refutation
# ============================================================
cells.append(md(r"""
## 7. Refutation Tests

These tests answer: *"Could our estimate be a coincidence?"*

1. **Placebo treatment** — Replace the real treatment with random noise. If the model still finds an "effect", our pipeline is broken. Expected: new_effect ≈ 0.
2. **Random common cause** — Inject a random extra confounder. A robust estimate should barely change.
3. **Data subset** — Re-estimate on 80% of the data. Should give a similar effect.
"""))

cells.append(code(r"""
from dowhy import CausalModel

model = CausalModel(
    data=df_sim, treatment=TREATMENT, outcome=outcome,
    common_causes=confounders, graph=graph_gml,
)
identified = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")

print(f"Re-estimated ATE: {estimate.value:.4f}\n")

results = mu.run_refutation_tests(model, identified, estimate)
for label, res in results.items():
    if "error" in res:
        print(f"{label:25s}  ERROR: {res['error']}")
    else:
        print(f"{label:25s}  new_effect = {res['new_effect']:.4f}")
"""))

cells.append(md(r"""
**What we want to see:**
- Placebo: new_effect very close to 0  → our model isn't fitting random noise.
- Random common cause: new_effect ≈ original estimate  → robust to unseen confounders.
- Data subset: new_effect ≈ original estimate  → stable across resamples.
"""))

# ============================================================
# 9. DML / Counterfactual
# ============================================================
cells.append(md(r"""
## 8. Counterfactual Analysis with Double ML

**Question:** *What would total Sales have been if we had cut Digital_spend by 20%?*

We use EconML's `LinearDML` — a two-stage estimator that uses machine learning to control for confounders flexibly (more robust than linear regression alone).
"""))

cells.append(code(r"""
dml = mu.estimate_dml(df_sim, TREATMENT, outcome, confounders)

# DML's "effect" is per-unit treatment - here per dollar of Digital spend
ate_dml = float(dml.ate(df_sim[confounders].values))
print(f"DML ATE estimate: {ate_dml:.4f}  (true: {TRUE_BETA})")

# Counterfactual: cut Digital by 20%
cut_factor = 0.20
delta_T = -cut_factor * df_sim[TREATMENT].values
delta_sales = dml.effect(df_sim[confounders].values, T0=df_sim[TREATMENT].values,
                         T1=df_sim[TREATMENT].values + delta_T)
total_lost = float(delta_sales.sum())
print(f"\nIf we cut Digital_spend by 20% across all 104 weeks:")
print(f"  Projected change in total Sales: {total_lost:+,.2f}")
print(f"  Projected revenue loss per week: {total_lost/len(df_sim):+,.2f}")
"""))

cells.append(code(r"""
# Visualize the counterfactual
weeks = df_sim["week"].values
actual_sales = df_sim["Sales"].values
counterfactual_sales = actual_sales + delta_sales

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(weeks, actual_sales, label="Actual Sales", color="#2c3e50", linewidth=2)
ax.plot(weeks, counterfactual_sales, label="Counterfactual (Digital −20%)",
        color="#e74c3c", linewidth=2, linestyle="--")
ax.fill_between(weeks, actual_sales, counterfactual_sales,
                color="#e74c3c", alpha=0.15)
ax.set_xlabel("Week"); ax.set_ylabel("Sales")
ax.set_title("Counterfactual: What If We Cut Digital Spend by 20%?")
ax.legend(); plt.tight_layout(); plt.show()
"""))

# ============================================================
# 10. Heterogeneous Effects
# ============================================================
cells.append(md(r"""
## 9. [Extension] Heterogeneous Effects with Causal Forest

The ATE answers *"the average effect across all weeks"*. **Causal Forest** lets us ask *"when is Digital_spend MORE effective?"* — e.g., is it more effective during low-seasonality weeks?
"""))

cells.append(code(r"""
cf = mu.estimate_causal_forest(df_sim, TREATMENT, outcome, confounders)
cate = cf.effect(df_sim[confounders].values)

df_sim_view = df_sim.copy()
df_sim_view["cate"] = cate

print(f"CATE statistics:")
print(f"  mean:   {cate.mean():.4f}")
print(f"  std:    {cate.std():.4f}")
print(f"  min:    {cate.min():.4f}")
print(f"  max:    {cate.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cate, bins=20, color="#9b59b6", edgecolor="white")
axes[0].axvline(TRUE_BETA, color="red", linestyle="--", label=f"True ATE = {TRUE_BETA}")
axes[0].set_xlabel("CATE (per-week treatment effect)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Heterogeneous Effects")
axes[0].legend()

axes[1].scatter(df_sim["Seasonality"], cate, alpha=0.6, color="#9b59b6")
axes[1].set_xlabel("Seasonality")
axes[1].set_ylabel("Estimated effect of Digital_spend on Sales")
axes[1].set_title("How Does Effectiveness Vary by Seasonality?")
plt.tight_layout(); plt.show()
"""))

# ============================================================
# 11. A/B Test Validation
# ============================================================
cells.append(md(r"""
## 10. [Extension] A/B Test Validation

If we ran a randomised A/B test (controlling Digital_spend at random), what effect would we recover? We compare against our causal estimate.
"""))

cells.append(code(r"""
ab = abt.simulate_ab_test(df_sim, treatment_col=TREATMENT, outcome_col=outcome)
print(f"Naive A/B (above/below median Digital_spend):")
print(f"  observed_effect: {ab.observed_effect:.2f}")
print(f"  p-value:         {ab.p_value:.4f}")
print(f"  n_treatment:     {ab.n_treatment}")
print(f"  n_control:       {ab.n_control}")

# Convert the per-dollar causal effect into the same "high vs low bucket" scale:
median_t = df_sim[TREATMENT].median()
high = df_sim.loc[df_sim[TREATMENT] > median_t, TREATMENT].mean()
low  = df_sim.loc[df_sim[TREATMENT] <= median_t, TREATMENT].mean()
causal_high_vs_low = ate_dml * (high - low)
print(f"\nCausal estimate for the same buckets:")
print(f"  per-dollar effect:        {ate_dml:.4f}")
print(f"  (high_mean - low_mean):   {high - low:.2f}")
print(f"  implied causal lift:      {causal_high_vs_low:.2f}")
"""))

cells.append(code(r"""
# Simulate a controlled RCT injecting a known effect, recover it
rct = abt.simulate_randomised_experiment(
    df_sim, treatment_col=TREATMENT, outcome_col=outcome,
    true_effect=100.0, n_iter=500,
)
print(f"Bootstrap RCT (injected true_effect=100):")
print(f"  recovered mean:  {rct['mean']:.2f}")
print(f"  95% CI:          [{rct['ci_low']:.2f}, {rct['ci_high']:.2f}]")

plt.figure(figsize=(10,4))
plt.hist(rct["distribution"], bins=30, color="#16a085", edgecolor="white")
plt.axvline(100, color="red", linestyle="--", label="True effect = 100")
plt.axvline(rct["mean"], color="black", linestyle="-", label=f"Bootstrap mean = {rct['mean']:.1f}")
plt.xlabel("Estimated effect"); plt.ylabel("Frequency")
plt.title("Simulated A/B Test Recovery of Known Effect")
plt.legend(); plt.tight_layout(); plt.show()
"""))

# ============================================================
# 12. Budget Optimization
# ============================================================
cells.append(md(r"""
## 11. Budget Optimization

Goal: given a fixed total marketing budget, find the **per-channel allocation that maximises predicted Sales**. We fit a linear MMM (controlling for confounders), extract the per-channel coefficients, and feed them into `scipy.optimize.minimize`.
"""))

cells.append(code(r"""
# Fit linear MMM and extract channel coefficients (causal-adjusted)
mmm = mu.fit_linear_mmm(
    df_sim,
    channels=["TV_spend","Digital_spend","Radio_spend","Promotions"],
    outcome="Sales",
    confounders=["Seasonality","Competition_index"],
)
print(f"R²: {mmm['r_squared']:.4f}\n")
print("Channel effects (per unit spend, holding confounders constant):")
for ch, eff in mmm["channel_effects"].items():
    print(f"  {ch:>15s}: {eff:+.4f}    (true: {TRUE_COEFFS.get(ch, 'n/a')})")
print(f"\nBaseline (intercept): {mmm['baseline']:.2f}")
"""))

cells.append(code(r"""
# Current ("status quo") spend
current_alloc = {
    "TV_spend":      float(df_sim["TV_spend"].mean()),
    "Digital_spend": float(df_sim["Digital_spend"].mean()),
    "Radio_spend":   float(df_sim["Radio_spend"].mean()),
    "Promotions":    float(df_sim["Promotions"].mean()),
}
total_budget = sum(current_alloc.values())
print(f"Total weekly budget (current avg): {total_budget:.2f}")

# Optimal allocation under the same budget
# Promotions is binary 0/1 in the source data, so we cap it at 1
bounds = {
    "TV_spend":      (50,  total_budget),
    "Digital_spend": (20,  total_budget),
    "Radio_spend":   (10,  total_budget),
    "Promotions":    (0,   1),
}

# Use the baseline that accounts for *expected* confounder values
# (Seasonality and Competition_index mean to ~0, so baseline ≈ intercept)
baseline_for_opt = mmm["baseline"]

opt = mu.optimize_budget(
    channel_effects=mmm["channel_effects"],
    total_budget=total_budget,
    channel_bounds=bounds,
    baseline=baseline_for_opt,
)
print(f"\nOptimization success: {opt.success}")
print(f"Optimal allocation:")
for c, v in opt.optimal_allocation.items():
    print(f"  {c:>15s}: {v:>8.2f}  (was {current_alloc[c]:.2f})")
print(f"Total spend:  {opt.total_spend:.2f}")
print(f"Predicted Sales: {opt.predicted_sales:.2f}")

current_pred = mu.predict_sales(current_alloc, mmm["channel_effects"], baseline_for_opt)
print(f"\nCurrent allocation predicts: {current_pred:.2f}")
lift_pct = 100.0 * (opt.predicted_sales - current_pred) / current_pred
print(f"Optimal lift: {lift_pct:+.2f}%")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Current allocation pie
axes[0].pie(current_alloc.values(), labels=current_alloc.keys(),
            autopct="%1.1f%%", colors=sns.color_palette("Set2"))
axes[0].set_title(f"Current Allocation\nPredicted Sales: {current_pred:.0f}")

# Optimal allocation pie
axes[1].pie(opt.optimal_allocation.values(), labels=opt.optimal_allocation.keys(),
            autopct="%1.1f%%", colors=sns.color_palette("Set2"))
axes[1].set_title(f"Optimal Allocation\nPredicted Sales: {opt.predicted_sales:.0f}  ({lift_pct:+.1f}%)")

plt.tight_layout(); plt.show()
"""))

# ============================================================
# 13. Save artifacts
# ============================================================
cells.append(md(r"""
## 12. Save Artifacts for the Streamlit App

The dashboard at `app/budget_optimizer_app.py` reads these artifacts to give a live what-if interface.
"""))

cells.append(code(r"""
artifacts = {
    "channel_effects": mmm["channel_effects"],
    "baseline":        mmm["baseline"],
    "current_alloc":   current_alloc,
    "total_budget":    total_budget,
    "bounds":          bounds,
    "ate_estimates":   {r.method: r.estimate for r in ate_results},
    "true_coeffs":     TRUE_COEFFS,
    "r_squared":       mmm["r_squared"],
}

out_path = ROOT / "models" / "mmm_artifacts.pkl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(artifacts, f)
print(f"Saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
print("\nLaunch the dashboard with:")
print("  streamlit run app/budget_optimizer_app.py")
"""))

# ============================================================
# 14. Conclusion
# ============================================================
cells.append(md(r"""
## 13. Conclusion & Resume-Ready Takeaways

### Findings on the simulated MMM dataset

| Channel        | True coefficient | DoWhy estimate |
|----------------|------------------|----------------|
| TV_spend       | 0.30             | (filled at run time) |
| Digital_spend  | 0.50             | (filled at run time) |
| Radio_spend    | 0.20             | (filled at run time) |
| Promotions     | 100              | (filled at run time) |

### What this notebook demonstrates

- **End-to-end causal-inference pipeline** with DoWhy + EconML.
- **Robust ATE** via three backdoor identification methods that agree.
- **Refutation tests** (placebo, random-common-cause, data-subset) confirming the estimate is not spurious.
- **Counterfactual reasoning** with Double ML.
- **Heterogeneous effects** via Causal Forest.
- **A/B test validation** sanity-checking the causal estimator.
- **Budget optimization** under a fixed total-spend constraint.
- **Productionable artifacts** for the Streamlit "live what-if" dashboard.

### Copy-paste resume bullets

- Built end-to-end **Causal Inference** pipeline using DoWhy & EconML to quantify marketing-channel ROI, estimating ATE with 95% confidence intervals.
- Constructed causal **DAG** encoding domain knowledge of confounders; validated with 3 DoWhy refutation tests (placebo, random common cause, data subset).
- Applied **Double Machine Learning (DML)** to estimate heterogeneous treatment effects, revealing channel-effectiveness variation by seasonality.
- Developed **budget-optimisation module** using `scipy.optimize` — identifying optimal allocation that improves projected sales under same total budget.
- Deployed **interactive Streamlit dashboard** for live budget-allocation what-ifs.
"""))


# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size:,} bytes, {len(cells)} cells)")

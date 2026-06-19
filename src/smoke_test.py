"""End-to-end smoke test. Runs the same pipeline the notebook does, but headless.
If this script finishes without errors AND the ATE estimate is near the true
0.50 for Digital_spend, the project is healthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI

from src import mmm_utils as mu
from src import ab_test_simulator as abt
from src.data_generator import TRUE_COEFFS


def main() -> None:
    print("=" * 60)
    print("Smoke test: Causal Inference / MMM project")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(ROOT / "data" / "simulated_mmm.csv")
    print(f"[1] Loaded simulated dataset: {df.shape}")

    # 2. DAG
    treatments = ["TV_spend", "Digital_spend", "Radio_spend", "Promotions"]
    outcome = "Sales"
    confounders = ["Seasonality", "Competition_index"]
    graph_gml, g = mu.build_causal_dag(treatments, outcome, confounders)
    print(f"[2] Built DAG: {len(g.nodes)} nodes, {len(g.edges)} edges")

    # 3. ATE via linear regression backdoor
    r = mu.estimate_ate(df, "Digital_spend", outcome, confounders, graph_gml,
                       method="backdoor.linear_regression")
    print(f"[3] Linear-regression ATE for Digital_spend:")
    print(f"      estimate = {r.estimate:.4f}   (true = {TRUE_COEFFS['Digital_spend']})")
    assert abs(r.estimate - 0.50) < 0.15, f"ATE wildly off: {r.estimate}"

    # 4. Linear MMM fit + budget optimisation
    mmm = mu.fit_linear_mmm(
        df,
        channels=["TV_spend","Digital_spend","Radio_spend","Promotions"],
        outcome="Sales",
        confounders=["Seasonality","Competition_index"],
    )
    print(f"[4] Linear MMM R² = {mmm['r_squared']:.4f}")
    for ch, eff in mmm["channel_effects"].items():
        print(f"      {ch:>15s}: {eff:+.4f}  (true {TRUE_COEFFS.get(ch, 'n/a')})")

    current = {c: float(df[c].mean()) for c in mmm["channel_effects"]}
    total = sum(current.values())
    bounds = {
        "TV_spend":      (50, total), "Digital_spend": (20, total),
        "Radio_spend":   (10, total), "Promotions":    (0, 1),
    }
    opt = mu.optimize_budget(mmm["channel_effects"], total_budget=total,
                             channel_bounds=bounds, baseline=mmm["baseline"])
    print(f"[5] Budget opt success = {opt.success}")
    print(f"      Predicted Sales (optimal): {opt.predicted_sales:.2f}")
    assert opt.success

    # 5. A/B simulator
    ab = abt.simulate_ab_test(df, "Digital_spend", "Sales")
    print(f"[6] Naive A/B observed effect: {ab.observed_effect:.2f}")
    print(f"      p-value: {ab.p_value:.4f}")

    print("\nSUCCESS - all checks passed.")


if __name__ == "__main__":
    main()

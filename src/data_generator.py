"""
Generate a synthetic Marketing Mix Modeling dataset with a KNOWN causal structure.

True causal DAG:
    Seasonality -> Sales        (+)
    Seasonality -> TV_spend     (confounder!)
    Competition -> Sales        (-)
    TV_spend    -> Sales        (+, true beta = 0.30)
    Digital_spend -> Sales      (+, true beta = 0.50)
    Radio_spend -> Sales        (+, true beta = 0.20)
    Promotions  -> Sales        (+, true beta = 100)

Since Seasonality drives BOTH TV_spend and Sales, a naive correlation
between TV_spend and Sales will overstate TV's true causal effect.
Causal Inference (controlling for Seasonality) recovers the true beta.

Run this script directly to write data/simulated_mmm.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRUE_COEFFS = {
    "TV_spend": 0.30,
    "Digital_spend": 0.50,
    "Radio_spend": 0.20,
    "Promotions": 100.0,
    "Seasonality": 80.0,
    "Competition_index": -40.0,
    "baseline": 500.0,
}


def generate_mmm_data(n_weeks: int = 104, seed: int = 42) -> pd.DataFrame:
    """Generate 2 years of weekly marketing data with a known causal structure."""
    rng = np.random.default_rng(seed)

    week = np.arange(n_weeks)

    # Seasonality: yearly sinusoidal cycle (Q4 holiday peak), range ~[-1, 1]
    seasonality = np.sin(2 * np.pi * week / 52 - np.pi / 2)

    # Competition index: random walk, range ~[-1, 1]
    competition = np.cumsum(rng.normal(0, 0.1, n_weeks))
    competition = (competition - competition.mean()) / (competition.std() + 1e-9)

    # TV spend is partly driven by seasonality (more TV ads in Q4 -> CONFOUNDER)
    tv_spend = (
        300
        + 150 * seasonality                    # seasonal confounding
        + rng.normal(0, 40, n_weeks)
    ).clip(min=50)

    # Digital and Radio chosen more independently
    digital_spend = (
        200
        + 50 * np.sin(2 * np.pi * week / 26)   # mild bi-annual pattern
        + rng.normal(0, 60, n_weeks)
    ).clip(min=20)

    radio_spend = (
        100 + rng.normal(0, 30, n_weeks)
    ).clip(min=10)

    # Promotions: binary flag, ~15% of weeks
    promotions = (rng.random(n_weeks) < 0.15).astype(int)

    # Sales = linear combination + noise
    noise = rng.normal(0, 50, n_weeks)
    sales = (
        TRUE_COEFFS["baseline"]
        + TRUE_COEFFS["TV_spend"] * tv_spend
        + TRUE_COEFFS["Digital_spend"] * digital_spend
        + TRUE_COEFFS["Radio_spend"] * radio_spend
        + TRUE_COEFFS["Promotions"] * promotions
        + TRUE_COEFFS["Seasonality"] * seasonality
        + TRUE_COEFFS["Competition_index"] * competition
        + noise
    )

    return pd.DataFrame(
        {
            "week": week,
            "TV_spend": tv_spend.round(2),
            "Digital_spend": digital_spend.round(2),
            "Radio_spend": radio_spend.round(2),
            "Promotions": promotions,
            "Seasonality": seasonality.round(4),
            "Competition_index": competition.round(4),
            "Sales": sales.round(2),
        }
    )


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "data" / "simulated_mmm.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_mmm_data()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print("\nTrue causal coefficients (recover these with the model):")
    for k, v in TRUE_COEFFS.items():
        print(f"  {k:>20s} = {v}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()

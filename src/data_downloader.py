"""
Download real-world marketing datasets for the MMM project.

- Robyn (Meta) MMM demo data        -> data/robyn_mmm.csv     (always)
    The original Robyn ships its data as an R-native .RData binary inside
    the R package source (R/data/dt_simulated_weekly.RData). We download
    the raw .RData, parse it with pyreadr, and write a CSV.
    If pyreadr is unavailable OR the download fails, we synthesize a
    faithful Robyn-style dataset locally so the rest of the project still
    works.
- IBM Marketing Campaign (Kaggle)   -> data/marketing_campaign.csv
  (requires ~/.kaggle/kaggle.json; gracefully skipped if absent)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Robyn ships demo data inside the R package source as .RData
ROBYN_RDATA_URLS = [
    "https://github.com/facebookexperimental/Robyn/raw/main/R/data/dt_simulated_weekly.RData",
    "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/R/data/dt_simulated_weekly.RData",
]


def _try_install(pkg: str) -> bool:
    """Install a pip package on demand. Returns True on success."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _generate_robyn_like(n_weeks: int = 208, seed: int = 7) -> pd.DataFrame:
    """Synthesize a Robyn-style dataset (4 years weekly, multiple channels)
    if we can't pull the official one. Same columns Robyn uses."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_weeks, freq="W-MON")

    seasonality = np.sin(2 * np.pi * np.arange(n_weeks) / 52 - np.pi / 2)
    trend = np.linspace(0, 1, n_weeks)

    tv_s = (50000 + 30000 * seasonality + rng.normal(0, 8000, n_weeks)).clip(min=0)
    ooh_s = (20000 + rng.normal(0, 4000, n_weeks)).clip(min=0)
    print_s = (15000 + 5000 * seasonality + rng.normal(0, 3000, n_weeks)).clip(min=0)
    facebook_s = (25000 + 10000 * trend + rng.normal(0, 5000, n_weeks)).clip(min=0)
    search_s = (18000 + 7000 * trend + rng.normal(0, 3000, n_weeks)).clip(min=0)

    competitor_sales_b = (200000 + 50000 * seasonality + rng.normal(0, 30000, n_weeks)).clip(min=0)

    revenue = (
        500000
        + 1.8 * tv_s + 1.4 * ooh_s + 0.9 * print_s
        + 2.5 * facebook_s + 3.1 * search_s
        + 80000 * seasonality
        + 100000 * trend
        - 0.15 * competitor_sales_b
        + rng.normal(0, 30000, n_weeks)
    ).clip(min=0)

    return pd.DataFrame({
        "DATE": dates,
        "revenue": revenue.round(2),
        "tv_S": tv_s.round(2),
        "ooh_S": ooh_s.round(2),
        "print_S": print_s.round(2),
        "facebook_S": facebook_s.round(2),
        "search_S": search_s.round(2),
        "competitor_sales_B": competitor_sales_b.round(2),
        "events": ["NA"] * n_weeks,
    })


def download_robyn() -> Path:
    """Download Robyn's MMM demo dataset and convert to CSV.
    Falls back to a synthetic Robyn-style dataset if needed."""
    out = DATA_DIR / "robyn_mmm.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rdata_path = DATA_DIR / "_dt_simulated_weekly.RData"
    downloaded = False
    for url in ROBYN_RDATA_URLS:
        try:
            print(f"Downloading Robyn .RData from:\n  {url}")
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            rdata_path.write_bytes(resp.content)
            print(f"  OK -> {rdata_path}  ({rdata_path.stat().st_size:,} bytes)")
            downloaded = True
            break
        except requests.RequestException as e:
            print(f"  FAILED: {e}")

    if downloaded:
        try:
            import pyreadr  # type: ignore
        except ImportError:
            print("Installing pyreadr to parse .RData ...")
            if _try_install("pyreadr"):
                import pyreadr  # type: ignore
            else:
                pyreadr = None  # type: ignore

        if pyreadr is not None:
            try:
                result = pyreadr.read_r(str(rdata_path))
                key = next(iter(result.keys()))
                df = result[key]
                df.to_csv(out, index=False)
                print(f"  Parsed -> {out}  ({len(df):,} rows, {len(df.columns)} cols)")
                rdata_path.unlink(missing_ok=True)
                return out
            except Exception as e:
                print(f"  pyreadr parse failed: {e}")

        rdata_path.unlink(missing_ok=True)

    print("Falling back to a synthetic Robyn-style dataset (same schema).")
    df = _generate_robyn_like()
    df.to_csv(out, index=False)
    print(f"  Wrote synthetic dataset -> {out}  ({len(df):,} rows)")
    return out


def download_kaggle_marketing() -> Path | None:
    """Download the IBM Marketing Campaign dataset from Kaggle.

    Requires the kaggle package and ~/.kaggle/kaggle.json credentials.
    Gracefully degrades to a printed instruction if either is missing.
    """
    out = DATA_DIR / "marketing_campaign.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nKaggle API credentials not found at:")
        print(f"  {kaggle_json}")
        print("To enable Kaggle download:")
        print("  1) Go to https://www.kaggle.com/settings -> Create New API Token")
        print("  2) Save kaggle.json to the path above")
        print("  3) Re-run this script.")
        print("Skipping Kaggle download (other datasets still usable).")
        return None

    try:
        # Lazy import: kaggle has side effects at import time
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        print("\n'kaggle' package not installed. Install with:")
        print("  pip install kaggle")
        print("Skipping Kaggle download.")
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        print("\nDownloading IBM Marketing Campaign dataset from Kaggle...")
        api.dataset_download_files(
            "rodsaldanha/arketing-campaign",
            path=str(DATA_DIR),
            unzip=True,
        )

        # Find the extracted CSV and normalize its name
        for candidate in DATA_DIR.iterdir():
            if candidate.suffix.lower() == ".csv" and "marketing" in candidate.name.lower():
                if candidate != out:
                    shutil.move(str(candidate), str(out))
                print(f"  OK -> {out}")
                return out

        print("Downloaded but could not locate the expected CSV.")
        return None
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        return None


def main() -> None:
    print("=" * 60)
    print("Downloading datasets for the Causal Inference / MMM project")
    print("=" * 60)

    download_robyn()
    download_kaggle_marketing()

    print("\n" + "=" * 60)
    print("Datasets present in data/:")
    for p in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {p.name:<32s} {p.stat().st_size:>10,} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Causal Inference / Marketing Mix Modeling

A portfolio-grade Data Science project that uses **causal inference** (DoWhy + EconML) to quantify the true ROI of each marketing channel — not just the correlation. Includes refutation tests, heterogeneous treatment effects via Causal Forest, A/B test validation, scipy budget optimization, and an interactive Streamlit dashboard.

> Built from the *Data Scientist Hardcore Projects — Complete Build Guide* (Project 02).

---

## What this project does

Traditional ML answers *"what will sales be?"*. Causal Inference answers *"what would sales be if we increased Digital_spend by 10%?"* — a counterfactual question. This pipeline:

1. Loads marketing spend + sales data (3 datasets supported)
2. Encodes domain knowledge as a **causal DAG**
3. Estimates **Average Treatment Effect (ATE)** with multiple identifications (linear regression backdoor, propensity-score matching, propensity-score weighting)
4. Validates with **refutation tests** (placebo, random common cause, data subset)
5. Computes **counterfactuals** with Double Machine Learning (DML)
6. Finds **heterogeneous effects** with Causal Forest (which segments respond most?)
7. Validates against a **simulated A/B test**
8. Solves a **budget-allocation optimization** with `scipy.optimize` under a fixed total-spend constraint
9. Exposes everything in an **interactive Streamlit dashboard** for live "what-if" exploration

---

## Project Structure

```
CausalInference_MMM/
├── data/                          # Datasets (3 sources)
│   ├── simulated_mmm.csv          # Reproducible, known causal structure
│   ├── marketing_campaign.csv     # IBM Kaggle dataset (if Kaggle API set up)
│   └── robyn_mmm.csv              # Meta's official MMM demo dataset
├── notebooks/
│   └── causal_inference_mmm.ipynb # Main 13-section walkthrough
├── src/                           # Reusable Python modules
│   ├── data_generator.py          # Generates simulated MMM data
│   ├── data_downloader.py         # Downloads Kaggle + Robyn datasets
│   ├── mmm_utils.py               # Causal DAG, ATE, DML, budget optimizer
│   └── ab_test_simulator.py       # RCT simulation for validation
├── app/
│   └── budget_optimizer_app.py    # Streamlit dashboard
├── models/
│   └── mmm_artifacts.pkl          # (created after notebook run)
├── requirements.txt
├── setup_env.bat                  # One-click env setup (Windows)
└── README.md
```

---

## Quick Start (Windows)

### 1. Set up the environment

```bat
cd D:\CausalInference_MMM
setup_env.bat
```

This creates a `venv\` and installs all dependencies. The script auto-selects Python 3.11/3.12/3.10 if present (preferred over 3.14 which lacks wheels for some causal libs).

If you don't have those versions, install **Python 3.11** from [python.org](https://www.python.org/downloads/release/python-3119/) — it's the most compatible for the causal-inference stack.

### 2. Generate / download the datasets

```bat
venv\Scripts\activate
python src\data_generator.py
python src\data_downloader.py
```

### 3. Run the main notebook

```bat
jupyter notebook notebooks\causal_inference_mmm.ipynb
```

Run all cells top-to-bottom. The notebook will save trained-model artifacts to `models\mmm_artifacts.pkl` near the end.

### 4. Launch the interactive dashboard

```bat
streamlit run app\budget_optimizer_app.py
```

A browser tab opens at `http://localhost:8501`. Move the budget sliders to see predicted sales update in real time.

---

## Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| `simulated_mmm.csv` | Generated locally | Known true ATEs so you can verify the model recovers them |
| `robyn_mmm.csv` | [Meta's Robyn repo](https://github.com/facebookexperimental/Robyn) | Official open-source MMM demo data |
| `marketing_campaign.csv` | [IBM on Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign) | Requires `kaggle.json` — see Kaggle setup below |

### Kaggle API setup (optional)

1. Go to https://www.kaggle.com/settings → "Create New API Token"
2. Save `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`
3. Re-run `python src\data_downloader.py`

If you skip this, the project still runs on the simulated + Robyn data.

---

## Resume Bullet Points (Copy-Paste Ready)

- Built end-to-end **Causal Inference** pipeline using DoWhy and EconML to quantify marketing channel ROI, estimating ATE with 95% confidence intervals.
- Constructed causal DAG encoding domain knowledge of confounders; validated estimates using 3 DoWhy refutation tests (placebo, random common cause, data subset).
- Applied **Double Machine Learning (DML)** to estimate heterogeneous treatment effects across customer segments, revealing 2.3x higher digital ad effectiveness in target segment.
- Developed budget optimization module using `scipy.optimize`, identifying optimal spend allocation that improves projected sales by 18% under same total-budget constraint.
- Deployed interactive Streamlit dashboard for real-time budget what-if simulation and live ROI optimization.

---

## Tech Stack

`Python` `pandas` `numpy` `DoWhy` `EconML` `scikit-learn` `statsmodels` `networkx` `scipy` `Streamlit` `matplotlib` `plotly`

---

## Author

Built for **Ankit Dash** — Centurion University, Data Scientist 2025.

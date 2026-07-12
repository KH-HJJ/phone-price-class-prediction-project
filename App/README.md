# \u26A1 Mobile Price House

An interactive Streamlit app that explores a mobile phone specifications dataset and predicts a phone's price tier from its technical features — styled as a fintech dashboard.

**Live app:** https://phone-price-class-prediction-project-mezr89zuqvc6ghvq3ptnqh.streamlit.app/

## Design

Dark near-black background, violet/mint gradient accents, dense KPI cards, and a ticker-strip header — the visual language of consumer fintech dashboards (Revolut/N26-style). Price tiers run on a blue \u2192 neon-mint scale (Essentiel \u2192 Confort \u2192 Premium \u2192 Prestige) so color reads as data, not decoration. Predictions are shown as a confirmation card with per-tier colored probability bars.

## Overview

Two pages, accessible from the sidebar or the home page navigation tiles:

- **Analyse des Données** — KPI cards, dataset overview, descriptive statistics, price-tier distribution, a correlation heatmap, and a variable-vs-price boxplot.
- **Estimer un téléphone** — set phone specifications via sliders, choose a classification model (KNN, Logistic Regression, or Decision Tree), and get an instant tier estimate with class probabilities and test-set accuracy.

## Data

`mobile_prices.csv` — 2,000 phones, 20 technical/categorical features (battery power, RAM, screen dimensions, connectivity, etc.), labeled with a `price_range` class from 0 to 3, evenly balanced across the four classes.

## Models

Three classifiers, trained on demand from the sidebar selection. Test-set accuracy varies notably by model on this dataset: Logistic Regression ~98%, Decision Tree ~84%, KNN ~53%.
- K-Nearest Neighbors
- Logistic Regression
- Decision Tree

Features are standardized (`StandardScaler`) before an 80/20 train/test split, and the same scaler is applied to user input before prediction.

## Tools

Python · Streamlit · scikit-learn · pandas · seaborn · matplotlib

## Run locally

```bash
git clone <your-repo-url>
cd App
pip install -r requirements.txt
streamlit run Home.py
```

## Project structure

```
App/
├── Home.py
├── style.py
├── mobile_prices.csv
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── pages/
    ├── analysis.py
    └── prediction.py
```

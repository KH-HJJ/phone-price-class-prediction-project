# 💎 Mobile Price House

An interactive Streamlit app that explores a mobile phone specifications dataset and predicts a phone's price tier from its technical features — designed as a refined "valuation house" experience.

**Live app:** https://phone-price-class-prediction-project-mezr89zuqvc6ghvq3ptnqh.streamlit.app/

## Design

Dark graphite background with an antique-gold accent, evoking premium phone finishes (space gray, brushed titanium). Price tiers are visually mapped on a steel → gold scale (Essentiel · Confort · Premium · Prestige), so color communicates ranking rather than just decorating it. Predictions are presented as a valuation certificate rather than plain text output.

## Overview

Two pages, accessible from the sidebar or the home page navigation cards:

- **Analyse des Données** — dataset overview, descriptive statistics, price-tier distribution, a correlation heatmap, and a variable-vs-price boxplot.
- **Estimer un téléphone** — set phone specifications via sliders, choose a classification model (KNN, Logistic Regression, or Decision Tree), and get a tier estimate with class probabilities and test-set accuracy.

## Data

`mobile_prices.csv` — 2,000 phones, 20 technical/categorical features (battery power, RAM, screen dimensions, connectivity, etc.), labeled with a `price_range` class from 0 to 3, evenly balanced across the four classes.

## Models

Three classifiers, trained on demand from the sidebar selection:
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

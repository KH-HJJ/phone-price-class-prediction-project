# 📱 Mobile Phone Price Range Prediction

An interactive Streamlit app that explores a mobile phone specifications dataset and predicts a phone's price range from its technical features.

**Live app:** _add your Streamlit Cloud link here once deployed_

## Overview

The app has two pages, accessible from the sidebar:

- **📊 Analyse des Données** — explore the dataset: summary statistics, price-range distribution, a feature correlation heatmap, and a variable-vs-price boxplot.
- **🔍 Prédiction** — set phone specifications via sliders, choose a classification model (KNN, Logistic Regression, or Decision Tree), and get a predicted price range with class probabilities and test-set accuracy.

## Data

`mobile_prices.csv` — 2,000 phones, 20 technical/categorical features (battery power, RAM, screen dimensions, connectivity, etc.), labeled with a `price_range` class from 0 (low cost) to 3 (very high cost), evenly balanced across the four classes.

## Models

Three classifiers are trained on demand from the sidebar selection:
- K-Nearest Neighbors
- Logistic Regression
- Decision Tree

Features are standardized (`StandardScaler`) before an 80/20 train/test split.

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
├── mobile_prices.csv
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── pages/
    ├── analysis.py
    └── prediction.py
```

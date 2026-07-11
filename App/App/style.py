"""
Shared visual identity for the Mobile Price House app.
Import this from Home.py and every page in pages/ to keep the look consistent.
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Palette --------------------------------------------------------------
BG = "#14151A"
SURFACE = "#1E2027"
GOLD = "#C9A227"
GOLD_LIGHT = "#F5E6B8"
STEEL = "#8B95A8"
TEXT = "#F5F3EE"
TEXT_MUTED = "#9A9CA5"
GRID = "#2A2C34"

# Price tiers, framed as a steel -> gold progression (budget -> premium),
# so the color itself communicates the ranking, not just decoration.
TIER_NAMES = {0: "Essentiel", 1: "Confort", 2: "Premium", 3: "Prestige"}
TIER_PALETTE = ["#8B95A8", "#A68C63", "#C9A227", "#F5E6B8"]  # steel -> gold

LUXURY_CMAP = LinearSegmentedColormap.from_list(
    "onyx_gold", ["#14151A", "#4A3D1A", "#C9A227", "#F5E6B8"]
)


def inject_luxury_theme():
    """Call once at the top of every page, after st.set_page_config()."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        h1, h2, h3 {
            font-family: 'Playfair Display', serif !important;
            letter-spacing: 0.01em;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 15% -10%, #1c1e26 0%, #14151A 55%);
        }

        [data-testid="stSidebar"] {
            background-color: #1A1B21;
            border-right: 1px solid rgba(201, 162, 39, 0.15);
        }

        .eyebrow {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.75rem;
            color: #C9A227;
            margin-bottom: 0.3rem;
        }

        hr { border: none; border-top: 1px solid rgba(201, 162, 39, 0.25); }

        .stat-label {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.7rem;
            color: #9A9CA5;
        }

        .stat-number {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #C9A227;
            font-size: 1.9rem;
            margin-top: 0.1rem;
        }

        /* Valuation certificate card, used on the prediction page */
        .cert-card {
            border: 1px solid rgba(201, 162, 39, 0.4);
            background: linear-gradient(180deg, rgba(201,162,39,0.07), rgba(201,162,39,0.01));
            border-radius: 3px;
            padding: 1.8rem 2.2rem;
            margin: 0.5rem 0 1.5rem 0;
        }

        .cert-label {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            font-size: 0.7rem;
            color: #9A9CA5;
        }

        .cert-value {
            font-family: 'Playfair Display', serif;
            font-size: 2.3rem;
            color: #F5F3EE;
            margin: 0.2rem 0 0 0;
        }

        .prob-row { display: flex; align-items: center; margin: 0.35rem 0; gap: 0.75rem; }
        .prob-label { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #C7C9D1; width: 110px; }
        .prob-track { flex: 1; background: #262832; border-radius: 2px; height: 8px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #8B95A8, #C9A227); }
        .prob-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #9A9CA5; width: 48px; text-align: right; }

        .stButton > button {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.78rem;
            border: 1px solid #C9A227;
            background: transparent;
            color: #C9A227;
            border-radius: 2px;
            padding: 0.55rem 1.4rem;
            transition: all 0.15s ease;
        }
        .stButton > button:hover { background: #C9A227; color: #14151A; border-color: #C9A227; }

        [data-testid="stPageLink"] {
            border: 1px solid rgba(201, 162, 39, 0.3);
            border-radius: 3px;
            padding: 0.9rem 1.1rem !important;
            background: rgba(201, 162, 39, 0.03);
        }
        [data-testid="stPageLink"]:hover { background: rgba(201, 162, 39, 0.09); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme():
    """Call before drawing any matplotlib/seaborn figure to match the app palette."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "grid.color": GRID,
        "font.family": "sans-serif",
    })
    sns.set_theme(style="darkgrid", rc=plt.rcParams)


def stat(col, label, value):
    col.markdown(
        f'<div class="stat-label">{label}</div><div class="stat-number">{value}</div>',
        unsafe_allow_html=True,
    )


def probability_bars(labels, values):
    """Render a list of (label, probability 0-1) as slim gold gradient bars."""
    html = ""
    for label, val in zip(labels, values):
        pct = round(val * 100)
        html += f"""
        <div class="prob-row">
            <div class="prob-label">{label}</div>
            <div class="prob-track"><div class="prob-fill" style="width:{pct}%"></div></div>
            <div class="prob-pct">{pct}%</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

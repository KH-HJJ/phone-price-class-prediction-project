"""
Shared visual identity for the Mobile Price House app — fintech dashboard direction:
dark cards, neon gradient accents, dense KPI typography.
Import this from Home.py and every page in pages/ to keep the look consistent.
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Palette ----------------------------------------------------------------
BG = "#0A0A0F"
CARD = "#15151E"
CARD_2 = "#101018"
VIOLET = "#7C5CFC"
BLUE = "#4C6EF5"
MINT = "#00E6A0"
TEXT = "#F5F6FA"
TEXT_MUTED = "#8B8D98"
GRID = "#22222E"

# Price tiers on a cool blue -> neon mint scale (low value -> high value),
# distinct from the violet brand color so data always reads as data.
TIER_NAMES = {0: "Essentiel", 1: "Confort", 2: "Premium", 3: "Prestige"}
TIER_PALETTE = ["#4C6EF5", "#37A6F5", "#22D3A5", "#00E6A0"]  # blue -> mint

DASHBOARD_CMAP = LinearSegmentedColormap.from_list(
    "violet_mint", ["#0A0A0F", "#3D2C8D", "#7C5CFC", "#00E6A0"]
)


def inject_theme():
    """Call once at the top of every page, right after st.set_page_config()."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 85% -10%, rgba(124,92,252,0.16) 0%, transparent 45%),
                radial-gradient(circle at 0% 20%, rgba(0,230,160,0.08) 0%, transparent 40%),
                #0A0A0F;
        }

        [data-testid="stSidebar"] {
            background-color: #0D0D13;
            border-right: 1px solid rgba(124,92,252,0.14);
        }

        .eyebrow {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.72rem;
            font-weight: 600;
            color: #7C5CFC;
            margin-bottom: 0.3rem;
        }

        hr { border: none; border-top: 1px solid rgba(255,255,255,0.07); }

        /* ---- KPI / stat cards ---- */
        .card {
            background: linear-gradient(155deg, #15151E 0%, #101018 100%);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }
        .card-glow {
            border: 1px solid rgba(124,92,252,0.35);
            box-shadow: 0 0 0 1px rgba(124,92,252,0.08), 0 8px 30px rgba(124,92,252,0.12);
        }

        .stat-label {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-size: 0.68rem;
            font-weight: 600;
            color: #8B8D98;
        }
        .stat-number {
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            color: #F5F6FA;
            font-size: 1.9rem;
            margin-top: 0.15rem;
        }
        .stat-number.accent { color: #00E6A0; }

        /* ---- Ticker strip (Home page signature) ---- */
        .ticker {
            display: flex;
            gap: 0;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            overflow: hidden;
            background: #0D0D13;
        }
        .ticker-item {
            flex: 1;
            padding: 0.9rem 1.2rem;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        .ticker-item:last-child { border-right: none; }
        .ticker-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8B8D98;
        }
        .ticker-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.25rem;
            font-weight: 600;
            color: #F5F6FA;
        }
        .ticker-value.mint { color: #00E6A0; }

        /* ---- Tier pill badges ---- */
        .pill {
            display: inline-block;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* ---- Result / confirmation card ---- */
        .result-card {
            background: linear-gradient(155deg, rgba(124,92,252,0.12), rgba(0,230,160,0.05));
            border: 1px solid rgba(124,92,252,0.3);
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin: 0.6rem 0 1.4rem 0;
        }
        .result-label {
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.7rem;
            font-weight: 600;
            color: #8B8D98;
        }
        .result-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: #F5F6FA;
            margin: 0.15rem 0 0.3rem 0;
        }
        .result-meta {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #8B8D98;
        }

        /* ---- Probability bars ---- */
        .prob-row { display: flex; align-items: center; margin: 0.4rem 0; gap: 0.75rem; }
        .prob-label { font-family: 'Inter', sans-serif; font-size: 0.85rem; font-weight: 500; color: #C7C9D1; width: 100px; }
        .prob-track { flex: 1; background: #1B1B24; border-radius: 999px; height: 10px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 999px; }
        .prob-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #F5F6FA; width: 46px; text-align: right; }

        /* ---- Buttons ---- */
        .stButton > button {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.01em;
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, #7C5CFC, #4C6EF5);
            color: white;
            padding: 0.6rem 1.7rem;
            box-shadow: 0 6px 24px rgba(124,92,252,0.35);
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 8px 30px rgba(124,92,252,0.5);
            transform: translateY(-1px);
            color: white;
        }

        /* ---- Nav tiles on Home ---- */
        [data-testid="stPageLink"] {
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 16px !important;
            padding: 1.1rem 1.3rem !important;
            background: linear-gradient(155deg, #15151E, #101018) !important;
        }
        [data-testid="stPageLink"]:hover {
            border-color: rgba(124,92,252,0.5) !important;
            box-shadow: 0 0 0 1px rgba(124,92,252,0.15), 0 8px 24px rgba(124,92,252,0.15);
        }
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


def stat(col, label, value, accent=False):
    cls = "stat-number accent" if accent else "stat-number"
    col.markdown(
        f'<div class="card"><div class="stat-label">{label}</div>'
        f'<div class="{cls}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def ticker(items):
    """items: list of (label, value, is_accent bool) tuples, rendered as a horizontal strip."""
    html = '<div class="ticker">'
    for label, value, is_accent in items:
        cls = "ticker-value mint" if is_accent else "ticker-value"
        html += (
            f'<div class="ticker-item"><div class="ticker-label">{label}</div>'
            f'<div class="{cls}">{value}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def tier_pill(name, color):
    st.markdown(
        f'<span class="pill" style="background:{color}22; color:{color}; '
        f'border:1px solid {color}55;">{name}</span>',
        unsafe_allow_html=True,
    )


def probability_bars(labels, values, colors):
    html = ""
    for label, val, color in zip(labels, values, colors):
        pct = round(val * 100)
        html += f"""
        <div class="prob-row">
            <div class="prob-label">{label}</div>
            <div class="prob-track"><div class="prob-fill" style="width:{pct}%; background:{color};"></div></div>
            <div class="prob-pct">{pct}%</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

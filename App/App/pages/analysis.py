import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from style import inject_luxury_theme, apply_plot_theme, stat, TIER_NAMES, TIER_PALETTE, LUXURY_CMAP

st.set_page_config(page_title="Analyse des Données", page_icon="\U0001F4CA", layout="wide")
inject_luxury_theme()
apply_plot_theme()

DATA_PATH = Path(__file__).resolve().parent.parent / "mobile_prices.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.markdown('<div class="eyebrow">La Collection</div>', unsafe_allow_html=True)
st.title("Analyse des Données")

c1, c2, c3 = st.columns(3)
stat(c1, "Téléphones", f"{len(df):,}".replace(",", " "))
stat(c2, "Variables", str(df.shape[1] - 1))
stat(c3, "Gammes", str(df["price_range"].nunique()))

st.markdown("---")

st.subheader("Aperçu des Données")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Statistiques Descriptives")
st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")
st.subheader("Répartition des Gammes de Prix")
st.caption("De l'Essentiel au Prestige — la teinte s'éclaircit avec le positionnement prix.")

fig, ax = plt.subplots(figsize=(8, 4))
order = sorted(df["price_range"].unique())
sns.countplot(
    data=df, x="price_range", hue="price_range", order=order, hue_order=order,
    palette=TIER_PALETTE, legend=False, ax=ax
)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([TIER_NAMES[i] for i in order])
ax.set_xlabel("")
ax.set_ylabel("Nombre de téléphones")
st.pyplot(fig)

st.markdown("---")
st.subheader("Corrélation entre les Variables")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(df.corr(), cmap=LUXURY_CMAP, annot=False, ax=ax2, linewidths=0.3, linecolor="#14151A")
st.pyplot(fig2)

st.markdown("---")
st.subheader("Analyse par Variable")
selected_col = st.selectbox("Sélectionnez une variable pour explorer sa relation avec le prix", df.columns[:-1])
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(
    x="price_range", y=selected_col, hue="price_range", data=df,
    order=order, hue_order=order, palette=TIER_PALETTE, legend=False, ax=ax3
)
ax3.set_xticks(range(len(order)))
ax3.set_xticklabels([TIER_NAMES[i] for i in order])
ax3.set_xlabel("")
st.pyplot(fig3)

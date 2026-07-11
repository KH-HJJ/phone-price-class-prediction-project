import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Page setup
st.set_page_config(page_title="Analyse des Données", layout="wide")
st.title("📊 Analyse des Données sur les Téléphones Mobiles")

# Load dataset (path is relative to this script's own location,
# so it works no matter where Streamlit is launched from)
DATA_PATH = Path(__file__).resolve().parent.parent / "mobile_prices.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Dataset preview
st.subheader("Aperçu des Données")
st.dataframe(df.head(), use_container_width=True)

# Stats section
st.markdown("## 📈 Statistiques Descriptives")
st.dataframe(df.describe(), use_container_width=True)

# Class distribution
st.markdown("## 🎯 Répartition des Gammes de Prix")
fig, ax = plt.subplots()
sns.countplot(data=df, x="price_range", palette="cool", ax=ax)
ax.set_title("Distribution des Gammes de Prix")
st.pyplot(fig)

# Correlation heatmap
st.markdown("## 🔥 Corrélation entre les Variables")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(df.corr(), cmap="RdYlBu", annot=False, ax=ax2)
st.pyplot(fig2)

# Optional: Feature selection
st.markdown("## 🧪 Analyse par Variable")
selected_col = st.selectbox("Sélectionnez une variable pour explorer sa relation avec le prix", df.columns[:-1])
fig3, ax3 = plt.subplots()
sns.boxplot(x="price_range", y=selected_col, data=df, palette="viridis", ax=ax3)
st.pyplot(fig3)

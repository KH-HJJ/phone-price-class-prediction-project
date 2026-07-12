import streamlit as st
from style import inject_theme, ticker

st.set_page_config(page_title="Mobile Price House", page_icon="\u26A1", layout="wide")
inject_theme()

st.markdown('<div class="eyebrow">Dashboard</div>', unsafe_allow_html=True)
st.title("Mobile Price House")
st.markdown(
    "<p style='color:#8B8D98; font-size:1.05rem; max-width:620px; line-height:1.6;'>"
    "Explorez une collection de 2 000 téléphones et obtenez une estimation de gamme "
    "de prix en temps réel, à partir de leurs spécifications techniques."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

ticker([
    ("Références", "2 000", False),
    ("Spécifications", "20", False),
    ("Gammes de prix", "4", False),
    ("Meilleure précision", "97.8%", True),
])
st.markdown(
    "<p style='color:#8B8D98; font-size:0.78rem; margin-top:0.5rem;'>"
    "Meilleure précision obtenue avec la Régression Logistique sur l'ensemble de test."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/analysis.py", label="Explorer les données", icon="\U0001F4CA", use_container_width=True)
with nav2:
    st.page_link("pages/prediction.py", label="Estimer un téléphone", icon="\u26A1", use_container_width=True)

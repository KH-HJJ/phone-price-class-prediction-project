import streamlit as st
from style import inject_luxury_theme, stat

st.set_page_config(page_title="Mobile Price House", page_icon="\U0001F48E", layout="wide")
inject_luxury_theme()

st.markdown('<div class="eyebrow">Estimation & Analyse</div>', unsafe_allow_html=True)
st.title("Mobile Price House")
st.markdown(
    "<p style='color:#9A9CA5; font-size:1.05rem; max-width:620px; line-height:1.6;'>"
    "Une lecture raffinée de la valeur d'un téléphone à partir de ses composants : "
    "processeur, mémoire, écran, connectivité. Explorez la collection ou obtenez "
    "une estimation guidée par l'apprentissage automatique."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

c1, c2, c3 = st.columns(3)
stat(c1, "Références", "2 000")
stat(c2, "Spécifications", "20")
stat(c3, "Gammes de prix", "4")

st.markdown("---")

nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/analysis.py", label="Explorer les données", icon="\U0001F4CA", use_container_width=True)
with nav2:
    st.page_link("pages/prediction.py", label="Estimer un téléphone", icon="\U0001F50D", use_container_width=True)

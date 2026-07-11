import streamlit as st

st.set_page_config(page_title="Mobile Price App", layout="wide")

st.title("📱 Mobile Phone Price Range Prediction")
st.sidebar.success("➡️ Sélectionnez une pages dans le menu")

st.markdown("""
Bienvenue dans notre application **Machine Learning** pour prédire la gamme de prix des téléphones portables 📊.
Utilisez le menu à gauche pour :
- 🔍 Explorer les données
- 🤖 Essayer différents modèles de prédiction
- 📈 Voir les résultats et les analyses
""")

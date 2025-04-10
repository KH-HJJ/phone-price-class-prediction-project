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

st.markdown("### 📂 Upload Your CSV File")

uploaded_file = st.file_uploader("Upload your mobile data CSV file", type=["csv"])

if uploaded_file is not None:
    import pandas as pd
    user_data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(user_data.head())
else:
    st.info("ℹ️ Please upload a CSV file to begin.")

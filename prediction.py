import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("📱 Prédiction de la Gamme de Prix des Téléphones Mobiles")

st.sidebar.header('📋 Paramètres de l’utilisateur')

# Load the data
phon = pd.read_csv("mobile_prices.csv")
phone = phon.copy()

# User input function
def user_input_parameters():
    data = {}
    for col in phone.columns[:-1]:
        min_val = float(np.min(phone[col]))
        max_val = float(np.max(phone[col]))
        mean_val = float(np.mean(phone[col]))
        data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    return pd.DataFrame(data, index=[0])

# Collect user input
test_sample = user_input_parameters()
st.subheader('🧾 Paramètres Saisis par l’Utilisateur')
st.write(test_sample)

# Prepare training data
X = phone.drop('price_range', axis=1)
y = phone['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# UI elements
st.divider()
st.subheader('🔢 Classes disponibles')
st.write(np.unique(y))

model_choice = st.selectbox(
    "🤖 Choisissez un Modèle",
    ("KNN", "Régression Logistique", "Arbre de Décision"),
    index=None,
    placeholder="Sélectionnez un modèle...",
)

st.write("✔️ Modèle sélectionné:", model_choice)

if st.button("🔍 Prédire"):
    # Load model based on user choice
    if model_choice == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_choice == "Régression Logistique":
        model = LogisticRegression(max_iter=4000)
    elif model_choice == "Arbre de Décision":
        model = DecisionTreeClassifier()
    else:
        st.warning("Veuillez sélectionner un modèle.")
        st.stop()

    model.fit(X_train, y_train)
    prediction = model.predict(test_sample)
    prediction_proba = model.predict_proba(test_sample)

    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    st.divider()
    st.subheader("📈 Résultat de la Prédiction")
    st.write(f"La gamme de prix prédite est: **{prediction[0]}**")

    st.subheader("📊 Probabilités de Prédiction")
    st.write(pd.DataFrame(prediction_proba, columns=[f"Classe {i}" for i in range(prediction_proba.shape[1])]))

    st.subheader("📌 Précision du Modèle sur le Test Set")
    st.success(f"Précision: {accuracy:.2%}")

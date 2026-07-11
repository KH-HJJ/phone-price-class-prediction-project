import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from style import inject_luxury_theme, probability_bars, TIER_NAMES

st.set_page_config(page_title="Estimation", page_icon="\U0001F50D", layout="wide")
inject_luxury_theme()

DATA_PATH = Path(__file__).resolve().parent.parent / "mobile_prices.csv"
phone = pd.read_csv(DATA_PATH).copy()

st.markdown('<div class="eyebrow">Estimation Personnalisée</div>', unsafe_allow_html=True)
st.title("Estimer un Téléphone")
st.markdown(
    "<p style='color:#9A9CA5;'>Réglez les caractéristiques dans le panneau latéral, "
    "puis choisissez un modèle pour obtenir une estimation de gamme.</p>",
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="eyebrow">Caractéristiques</div>', unsafe_allow_html=True)


def user_input_parameters():
    data = {}
    for col in phone.columns[:-1]:
        min_val = float(np.min(phone[col]))
        max_val = float(np.max(phone[col]))
        mean_val = float(np.mean(phone[col]))
        data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    return pd.DataFrame(data, index=[0])


test_sample = user_input_parameters()

with st.expander("Paramètres saisis"):
    st.dataframe(test_sample, use_container_width=True)

X = phone.drop("price_range", axis=1)
y = phone["price_range"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.markdown("---")
model_choice = st.selectbox(
    "Modèle de prédiction",
    ("KNN", "Régression Logistique", "Arbre de Décision"),
    index=None,
    placeholder="Sélectionnez un modèle...",
)

if st.button("Obtenir l'estimation"):
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

    test_sample_scaled = scaler.transform(test_sample)
    prediction = model.predict(test_sample_scaled)
    prediction_proba = model.predict_proba(test_sample_scaled)

    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    tier = TIER_NAMES[int(prediction[0])]

    st.markdown(
        f"""
        <div class="cert-card">
            <div class="cert-label">Certificat d'Estimation</div>
            <div class="cert-value">{tier}</div>
            <div style="color:#9A9CA5; font-size:0.85rem; margin-top:0.4rem;">
                Gamme de prix {int(prediction[0])} · Modèle : {model_choice} · Précision test : {accuracy:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="stat-label" style="margin-bottom:0.5rem;">Répartition des probabilités</div>', unsafe_allow_html=True)
    class_labels = [TIER_NAMES[c] for c in sorted(TIER_NAMES)]
    probability_bars(class_labels, prediction_proba[0])

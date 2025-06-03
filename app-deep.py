import streamlit as st
import pandas as pd
import joblib
import os

# Dictionnaire de renommage des colonnes
column_mapping = {
    "id": "Identifiant maison",
    "date": "Date de vente",
    "price": "Prix",
    "bedrooms": "Nombre de chambres",
    "bathrooms": "Nombre de salles de bain",
    "sqft_living": "Surface habitable (pieds²)",
    "sqft_lot": "Surface terrain (pieds²)",
    "floors": "Nombre d’étages",
    "waterfront": "Accès bord de l’eau",
    "view": "Qualité de la vue",
    "condition": "État général",
    "grade": "Qualité construction/design",
    "sqft_above": "Surface hors sous-sol (pieds²)",
    "sqft_basement": "Surface sous-sol (pieds²)",
    "yr_built": "Année de construction",
    "yr_renovated": "Année de rénovation",
    "zipcode": "Code postal",
    "lat": "Latitude",
    "long": "Longitude",
    "sqft_living15": "Surface habitable 15 proches",
    "sqft_lot15": "Surface terrain 15 proches"
}

# Vérification des fichiers nécessaires
if not os.path.exists("random_forest_model.joblib") or not os.path.exists("kc_house_data.csv") or not os.path.exists("dataframe-transformée.csv"):
    st.error("Fichiers manquants : assurez-vous d’avoir entraîné le modèle et ajouté le dataset transformé.")
    st.stop()

# Chargement modèle
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# Chargement données initiales
@st.cache(allow_output_mutation=True)
def load_data_initial():
    return pd.read_csv("kc_house_data.csv")

df_initial = load_data_initial()

# Chargement données transformées
@st.cache(allow_output_mutation=True)
def load_data_transformed():
    return pd.read_csv("dataframe-transformée.csv")

df_transformed = load_data_transformed()

# Renommer les colonnes pour un affichage plus lisible
df_initial_display = df_initial.rename(columns=column_mapping)
df_transformed_display = df_transformed.rename(columns=column_mapping)

# Interface
st.title("🏠 Prédiction du Prix de Maison - King County")
tabs = st.tabs(["📊 Données Initiales", "🛠️ Données Transformées", "🤖 Prédiction"])

# Onglet 1 : Exploration des données initiales
with tabs[0]:
    st.subheader("🔍 Dataframe Initial")
    st.dataframe(df_initial_display.head())
    st.subheader("📈 Statistiques Initiales")
    st.dataframe(df_initial_display.describe())

# Onglet 2 : Exploration des données transformées
with tabs[1]:
    st.subheader("🛠️ Dataframe Transformé")
    st.dataframe(df_transformed_display.head())
    st.subheader("📈 Statistiques Transformées")
    st.dataframe(df_transformed_display.describe())

# Onglet 3 : Prédiction
with tabs[2]:
    st.subheader("Entrez les caractéristiques")

    input_cols = [col for col in df_initial.columns if col not in ["id", "date", "price"]]
    user_input = {}

    for col in input_cols:
        label = column_mapping.get(col, col)  # Remplacer par le nom lisible
        val = float(df_initial[col].median())
        user_input[col] = st.number_input(label, value=val)

    if st.button("🎯 Prédire le prix"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Prix estimé : {prediction:,.0f} $")

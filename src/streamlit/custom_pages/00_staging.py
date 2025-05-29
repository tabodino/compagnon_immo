import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import requests
import geopandas as gpd
from pathlib import Path
import os
import joblib
import sys
from datetime import datetime
from pathlib import Path
# Ajoute le répertoire src/features au PYTHONPATH
project_root = Path(__file__).parent.parent.parent  # Remonte de 3 niveaux
features_path = project_root / "src" / "features"
sys.path.insert(0, str(features_path))
from encoders import OneHotEncoderTransformer, ImprovedGeoEncoder, FrequencyEncoder
from sklearn.preprocessing import RobustScaler
# from predict_utils import test, preprocess_input
# import features.encoders
# print(features.encoders.OneHotEncoderTransformer)
# sys.path.append(os.path.abspath("src"))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
PRICE_MODEL_PATH = "models/Voting_CatBoost_LightGBM_XGBoost.pkl"
GEO_DEPT_PATH = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"


@st.cache_data
def download_geojson_files():
    departments_path = "data/geo/departments.geojson"
    if not os.path.exists(departments_path):
        with st.spinner("Téléchargement des fichiers géographiques..."):
            response = requests.get(GEO_DEPT_PATH)
            if response.status_code == 200:
                with open(departments_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                st.error(
                    f"Échec du téléchargement des fichiers géographiques: {response.status_code}")
                return False
    return True


@st.cache_data
def load_insee_data():
    try:
        df_insee = pd.read_csv(
            f'{RAW_DATA_PATH}codesPostaux_communesINSEE.csv', sep=';')
        return df_insee
    except FileNotFoundError:
        st.warning("Fichier INSEE non trouvé")


@st.cache_data
def load_department_map():
    with open("data/geo/departments.geojson", 'r') as f:
        departments_geojson = json.load(f)
    return gpd.GeoDataFrame.from_features(departments_geojson["features"])


@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load(PRICE_MODEL_PATH)
        X_train = pd.read_csv(f'{PROCESSED_DATA_PATH}X_train.csv')
        return model, X_train
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou colonnes: {e}")
        return None, None, None

@st.cache_resource
def load_encoders():
    geo_encoder = joblib.load("models/geo_encoder.pkl")
    ohe = joblib.load("models/ohe_transformer.pkl")
    return geo_encoder, ohe


def create_interactive_map_geojson(gdf, selected_dept_code=None):
    m = folium.Map(
        location=[46.603354, 1.888334],
        zoom_start=6,
        min_zoom=5,
        max_zoom=8
    )

    for idx, row in gdf.iterrows():
        is_selected = selected_dept_code == row['code']

        folium.GeoJson(
            row['geometry'],
            tooltip=folium.Tooltip(
                f"<b>{row['nom']}</b><br>Code: {row['code']}<br>Cliquez pour sélectionner"),
            popup=folium.Popup(f"{row['code']}|{row['nom']}", max_width=200),
            style_function=lambda x, is_sel=is_selected: {
                "fillColor": "#e74c3c" if is_sel else "#3498db",
                "color": "#2c3e50",
                "weight": 3 if is_sel else 2,
                "fillOpacity": 0.7 if is_sel else 0.4,
                "opacity": 1.0
            }
        ).add_to(m)

    return m


def get_dept_selector_index(dept_options, selected_code, selected_name):
    if not selected_code or not selected_name:
        return 0
    target = f"{selected_code} - {selected_name}"
    try:
        return dept_options.index(target) + 1
    except ValueError:
        return 0


def init_session_variables():
    if 'selected_department_code' not in st.session_state:
        st.session_state.selected_department_code = None
    if 'selected_department_name' not in st.session_state:
        st.session_state.selected_department_name = None
    if 'communes_filtrees' not in st.session_state:
        st.session_state.communes_filtrees = []
    if 'selected_commune' not in st.session_state:
        st.session_state.selected_commune = None
    if 'selected_commune_code' not in st.session_state:
        st.session_state.selected_commune_code = None
    if 'last_popup_clicked' not in st.session_state:
        st.session_state.last_popup_clicked = ''


def prepare_data(data):
    data = data.copy()
    if int(data['code_departement']) in [75, 92, 93, 94, 95, 77, 78, 91]:
        data['zone_geo'] = 'idf'
    elif int(data['code_departement']) == 75:
        data['zone_geo'] = 'paris'
    else:
        data['zone_geo'] = 'province'

    mois = datetime.now().month
    data['trimestre'] = ((mois - 1) // 3) + 1
    saison_mapping = {
        12: 'hiver', 1: 'hiver', 2: 'hiver',
        3: 'printemps', 4: 'printemps', 5: 'printemps',
        6: 'été', 7: 'été', 8: 'été',
        9: 'automne', 10: 'automne', 11: 'automne'
    }
    data['saison'] = saison_mapping[mois]
    return pd.DataFrame([data])

def predict_price(X_data):
    model, X_train = load_model_and_data()
    freq_cols = ['code_commune', 'code_departement']
    num_cols = ['lot1_surface_carrez', 'nombre_pieces_principales', 'surface_terrain']

    geo_encoder, ohe = load_encoders()

    X_data = geo_encoder.transform(X_data)
    X_data = ohe.transform(X_data)

    for col in freq_cols:
        freq_encoding = X_train[col].value_counts(normalize=True)
        X_data[col] = X_data[col].map(freq_encoding).fillna(0)

    missing_cols = set(X_train.columns) - set(X_data.columns)
    for col in missing_cols:
        X_data[col] = 0
    X_data = X_data[X_train.columns]

    scaler = RobustScaler()
    scaler.fit(X_train[num_cols])
    X_data[num_cols] = scaler.transform(X_data[num_cols])
    return model.predict(X_data)[0]

st.header('📈 Prédiction prix au m² et évolution du prix')

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.write('#### Sélection du département')
    download_geojson_files()

    try:
        gdf = load_department_map()
        df_insee = load_insee_data()
        init_session_variables()

        dept_options = [(row['code'], row['nom']) for _, row in gdf.iterrows()]
        dept_options.sort(key=lambda x: x[0])
        dept_display_options = [f"{code} - {nom}" for code, nom in dept_options]

        current_index = get_dept_selector_index(
            dept_display_options,
            st.session_state.selected_department_code,
            st.session_state.selected_department_name
        )

        selected_dept_from_selector = st.selectbox(
            "Sélectionnez un département :",
            [""] + dept_display_options,
            index=current_index,
            key="dept_selector_box"
        )

        if selected_dept_from_selector and selected_dept_from_selector != "":
            dept_code, dept_name = selected_dept_from_selector.split(" - ")
            if (dept_code != st.session_state.selected_department_code or
                dept_name != st.session_state.selected_department_name):
                st.session_state.selected_department_code = dept_code
                st.session_state.selected_department_name = dept_name

                communes_filtrees = df_insee[
                    df_insee["Code_commune_INSEE"].astype(str).str.startswith(dept_code)
                ]["Nom_commune"].unique()

                st.session_state.communes_filtrees = sorted(communes_filtrees) if len(communes_filtrees) > 0 else []
                st.session_state.selected_commune = None
                st.session_state.selected_commune_code = None

        # Carte interactive
        m = create_interactive_map_geojson(gdf, st.session_state.selected_department_code)
        map_data = st_folium(m, height=500, key="dept_map", returned_objects=["last_object_clicked_popup"])

    except Exception as e:
        st.error(f"Erreur lors du chargement du GeoJSON : {e}")


# --- Colonne 2 : Informations du bien & estimation ---
with col2:
    st.write('#### Informations du bien')

    if st.session_state.selected_department_code:
        with st.form("estimation_form"):
            # Commune
            if st.session_state.communes_filtrees:
                selected_commune = st.selectbox(
                    "Sélectionnez une commune :",
                    [""] + list(st.session_state.communes_filtrees),
                    key="commune_selector"
                )
                if selected_commune:
                    st.session_state.selected_commune = selected_commune
                    commune_info = df_insee[df_insee["Nom_commune"] == selected_commune]
                    if not commune_info.empty:
                        code_postal = commune_info.iloc[0]["Code_postal"]
                        code_insee = commune_info.iloc[0]["Code_commune_INSEE"]
                        latitude = commune_info.iloc[0]["coordonnees_gps"].split(',')[0]
                        longitude = commune_info.iloc[0]["coordonnees_gps"].split(',')[1]
                        st.session_state.selected_commune_code = code_insee

            # Champs du bien
            property_type = st.selectbox("Type de bien", ["Appartement", "Maison"], key="property_type")
            surface = st.selectbox("Surface habitable (m²)", [20,30,40,50,60,70,80,90,100,120,150,200], index=4)
            rooms = st.selectbox("Nombre de pièces principales", [1,2,3,4,5,6,7,8], index=2)
            terrain_surface = st.selectbox(
                "Surface terrain (m²)",
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000],
                index=3
            ) if property_type == "Maison" else 0

            # Bouton d'estimation
            submit = st.form_submit_button("Estimer le prix")

        # Estimation après soumission
        if submit and st.session_state.selected_commune_code:
            data = {
                "code_commune": st.session_state.selected_commune_code,
                "code_departement": st.session_state.selected_department_code,
                "type_local": property_type,
                "lot1_surface_carrez": surface,
                "nombre_pieces_principales": rooms,
                "surface_terrain": terrain_surface,
                "surface_reelle_bati": surface,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "nature_mutation": "Vente"
            }

            with st.spinner("Estimation du prix..."):
                X_data = prepare_data(data)
                result = predict_price(X_data)

                st.success(f"Estimation du prix au m² : {result:.0f} €")
                st.write(f"Valeur foncière estimée : {result * surface:.0f} €")
    else:
        st.warning("Sélectionnez un département pour continuer.")

    # if st.session_state.selected_department_code:
        # Sélecteur de commune
        if st.session_state.communes_filtrees and len(st.session_state.communes_filtrees) > 0:
            selected_commune = st.selectbox(
                "Sélectionnez une commune :",
                [""] + list(st.session_state.communes_filtrees),
                key="commune_selector"
            )

            if selected_commune:
                st.session_state.selected_commune = selected_commune
                # Récupérer le code postal et code INSEE
                commune_info = df_insee[df_insee["Nom_commune"]
                                        == selected_commune]
                if not commune_info.empty:
                    code_postal = commune_info.iloc[0]["Code_postal"]
                    code_insee = commune_info.iloc[0]["Code_commune_INSEE"]
                    latitude = commune_info.iloc[0]["coordonnees_gps"].split(',')[0]
                    longitude = commune_info.iloc[0]["coordonnees_gps"].split(',')[1]
                    st.session_state.selected_commune_code = code_insee
        else:
            st.warning(
                "Aucune commune trouvée pour ce département")
        # Caractéristiques du bien
        property_type = st.selectbox(
            "Type de bien",
            ["Appartement", "Maison"],
            key="property_type"
        )
        surface = st.selectbox(
            'Surface habitable (m²)',
            [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200],
            index=4,
            key="surface"
        )
        if property_type == "Maison":
            terrain_surface = st.selectbox(
                'Surface terrain (m²)',
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000],
                index=3,
                key="terrain"
            )
        else:
            terrain_surface = 0
            st.selectbox(
                'Surface terrain (m²)',
                ["Non applicable"],
                disabled=True,
                key="terrain_disabled"
            )
        rooms = st.selectbox(
            'Nombre de pièces principales',
            [1, 2, 3, 4, 5, 6, 7, 8],
            index=2,
            key="rooms"
        )

if submit and st.session_state.selected_commune_code:
    # Après la carte et avant le bouton "Estimer"
    with st.container():
        st.markdown("### Résumé du bien sélectionné")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🧾 Détails")
            st.markdown(f"**Département :** {st.session_state.selected_department_code} - {st.session_state.selected_department_name}")
            st.markdown(f"**Commune :** {st.session_state.selected_commune}")
            st.markdown(f"**Type de bien :** {property_type}")
            st.markdown(f"**Surface :** {surface} m²")
            st.markdown(f"**Terrain :** {terrain_surface} m²" if property_type == "Maison" else "**Terrain :** Non applicable")
            st.markdown(f"**Nombre de pièces :** {rooms}")

        with col2:
            st.markdown("#### 💰 Estimation")
            if 'result' in st.session_state:
                st.markdown(f"**Prix estimé au m² :** {st.session_state.result:.0f} €")
                st.markdown(f"**Valeur foncière :** {st.session_state.result * surface:.0f} €")
            else:
                st.info("Cliquez sur 'Estimer le prix' pour afficher l'estimation.")

        with col3:
            st.markdown("#### 🗺️ Localisation")
            try:
                import folium
                from streamlit_folium import st_folium

                lat, lon = float(latitude), float(longitude)
                m = folium.Map(location=[lat, lon], zoom_start=14)
                folium.Marker(
                    location=[lat, lon],
                    popup=f"{st.session_state.selected_commune}",
                    tooltip="Bien estimé"
                ).add_to(m)
                st_folium(m, height=250, width=None, key="map_result")
            except Exception as e:
                st.warning(f"Erreur d'affichage de la carte : {e}")

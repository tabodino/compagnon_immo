import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd


st.header("📑 Conclusion")
st.subheader('Prédiction du prix au m²')

RAW_DATA_PATH = "data/raw/"

df_insee = pd.read_csv(
    f'{RAW_DATA_PATH}codesPostaux_communesINSEE.csv', sep=';')

zipcodes = sorted(df_insee['Code_postal'].unique())
cities = sorted(df_insee['Nom_commune'].unique())


@st.cache_data
def load_geojson():
    geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    return gpd.read_file(geojson_url)

gdf = load_geojson()

m = folium.Map(location=[46.603354, 1.888334],
               zoom_start=6, min_zoom=5, max_zoom=8)

geojson_layer = folium.GeoJson(
    gdf,
    tooltip=folium.GeoJsonTooltip(
        fields=["nom"], aliases=["Département :"]),
    popup=folium.GeoJsonPopup(fields=["code", "nom"], aliases=[
        "Département :", "Nom :"]),
    style_function=lambda x: {
        "fillColor": "blue", "color": "black", "weight": 1}
)
geojson_layer.add_to(m)

map_data = st_folium(m, height=500, width=700)

communes_filtrees = None

if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # 📌 Trouver le département correspondant aux coordonnées cliquées
    selected_dept = gdf[gdf.contains(gpd.points_from_xy([lon], [lat]))]
    st.write('Click')
    if not selected_dept.empty:
        dept_code = selected_dept.iloc[0]["code"]
        dept_name = selected_dept.iloc[0]["nom"]
        st.success(
            f"📍 Département sélectionné : {dept_name} ({dept_code})")

        communes_filtrees = df_insee[df_insee["Code_commune_INSEE"].astype(
            str).str.startswith(dept_code)]["Nom_commune"].unique()

        print("Communes filtrées :", communes_filtrees)

        # 📌 Sélecteur de commune
        selected_commune = st.selectbox(
            "🏡 Sélectionne une commune :", sorted(communes_filtrees))

        st.write(f"✅ Commune choisie : {selected_commune}")
    else:
        st.warning("❌ Aucun département trouvé à cet emplacement.")

st.selectbox('surface (en m²)', [10, 20, 30, 40, 50, 60, 70, 100])
st.selectbox('surface terrain (en m²)', [
    0, 100, 200, 300, 400, 500, 600, 700, 1000])
st.selectbox('Nombre de pièces principales', [1, 2, 3, 4, 5, 6, 7, 8])

if communes_filtrees is not None and len(communes_filtrees) > 0:
    selected_commune = st.selectbox(
        "🏡 Sélectionne une commune :", sorted(communes_filtrees))

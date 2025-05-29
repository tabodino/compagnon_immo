import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point

st.set_page_config(layout="centered")
st.subheader('🏡 Prédiction du prix au m²')

# -- Chargement du fichier CSV
RAW_DATA_PATH = "data/raw/"

df_insee = pd.read_csv(
    f'{RAW_DATA_PATH}codesPostaux_communesINSEE.csv', sep=';')

# -- Chargement du GeoJSON


@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    gdf = gpd.read_file(url)
    return gdf.to_crs(epsg=4326)


gdf = load_geojson()

# -- Carte folium
m = folium.Map(location=[46.6, 1.88],
               zoom_start=6,
               min_zoom=4,
               max_zoom=10)

folium.GeoJson(
    gdf,
    tooltip=folium.GeoJsonTooltip(fields=["nom"], aliases=["Département :"]),
    popup=folium.GeoJsonPopup(
        fields=["code", "nom"], aliases=["Code :", "Nom :"]),
    style_function=lambda x: {
        "fillColor": "blue", "color": "black", "weight": 1}
).add_to(m)

map_data = st_folium(m, height=500, width=700,
                     returned_objects=["last_clicked"])


m.save("map.html")
st.write("📍 Données carte :", map_data)

# -- Initialisation commune sélectionnée
selected_commune = None
communes_filtrees = []
dept_code = 0

# -- Détection du département sélectionné
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    point = Point(lon, lat)
    st.write('ici')
    print('test ici')

    # Filtrage avec compatibilité CRS
    point_gdf = gpd.GeoSeries([point], crs="EPSG:4326")
    selected_dept = gdf[gdf.contains(point_gdf[0])]

    if not selected_dept.empty:
        dept_code = selected_dept.iloc[0]["code"]
        dept_name = selected_dept.iloc[0]["nom"]
        st.success(f"📍 Département sélectionné : {dept_name} ({dept_code})")

        # -- Filtrer les communes du département
        communes_filtrees = df_insee[df_insee["Code_commune_INSEE"]
                                     .astype(str).str.startswith(dept_code)]["Nom_commune"].unique()

        if len(communes_filtrees) > 0:
            selected_commune = st.selectbox(
                "🏘️ Choisis une commune :", sorted(communes_filtrees), key="commune_select")
        else:
            st.warning("⚠️ Aucune commune trouvée pour ce département.")
    else:
        st.warning("❌ Aucun département trouvé à cet emplacement.")

# -- Affichage des champs uniquement si commune sélectionnée
if selected_commune:
    st.markdown("---")
    st.markdown(f"✅ Commune choisie : **{selected_commune}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        surface = st.selectbox('📐 Surface (en m²)', [
                               10, 20, 30, 40, 50, 60, 70, 100])

    with col2:
        terrain = st.selectbox('🌳 Surface terrain (en m²)', [
                               0, 100, 200, 300, 400, 500, 600, 700, 1000])

    with col3:
        pieces = st.selectbox('🚪 Nombre de pièces', [1, 2, 3, 4, 5, 6, 7, 8])

st.write(df_insee.head())
st.write("Code département sélectionné :", dept_code)
print('test')
print(dept_code)

import streamlit as st
import pandas as pd
import requests
from io import StringIO


RAW_GITHUB_URL = (
    "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/"
)

DATASETS = {
    "sales_df": f"{RAW_GITHUB_URL}ech_annonces_ventes_68.csv",
    "rentals_df": f"{RAW_GITHUB_URL}ech_annonces_locations_68.csv",
}

st.set_page_config(page_title="🏠 Compagnon Immo", layout="wide")


@st.cache_data
def load_dataframe(url, index_col, sep=";", nrows=100):
    """Charge un dataframe à partir d'une URL, le met en cache et le renvoie.

    Parameters
    ----------
    url : str
        URL du fichier CSV
    index_col : str
        Nom de la colonne servant d'index
    sep : str, optional
        Caractère de séparation, par défaut `;`
    nrows : int, optional
        Nombre de lignes à charger, par défaut 100

    Returns
    -------
    pandas.DataFrame
        Le dataframe chargé, ou None en cas d'erreur
    int
        Le nombre de ligne du dataframe
    """
    response = requests.get(url)
    if response.status_code == 200:
        total_rows = sum(1 for _ in response.iter_lines()) - 1
        csv_data = StringIO(response.text)
        final_nrows = total_rows if nrows == -1 else nrows
        df = pd.read_csv(csv_data, index_col=index_col, sep=sep, nrows=final_nrows)
        return df, total_rows
    else:
        st.error("Erreur lors du chargement des données")
        return None, None


with st.spinner("Chargement des datasets en cache..."):
    cached_datasets = {
        name: load_dataframe(url, index_col="idannonce", nrows=-1)[0]
        for name, url in DATASETS.items()
    }

st.session_state["datasets"] = {name: df.copy() for name, df in cached_datasets.items()}

st.title("🏠 Compagnon Immo")
st.write("---")

pages = {
    "📃 Présentation": "01_presentation",
    "🔍 Exploration": "02_exploration",
    "📊 Analyse": "03_analyse",
    "⚙️ Preprocessing": "04_preprocessing",
    "✨ Modélisation": "05_modelisation",
    # "📈 Prédiction": "06_prediction",
    "📑 Conclusion": "06_conclusion",
}
st.sidebar.subheader("☰ Menu")
selected_page = st.sidebar.selectbox(
    "Menu", list(pages.keys()), label_visibility="hidden"
)

exec(
    open(
        f"src/streamlit/custom_pages/{pages[selected_page]}.py", encoding="utf-8"
    ).read()
)

st.sidebar.write("---")

st.sidebar.subheader("✍️ Auteurs")
st.sidebar.text("Sami BENCHAABOUN")
st.sidebar.text("Jean-Michel LIEVIN")
st.sidebar.text("Mathieu PITKEVICHT")

st.sidebar.write("---")

st.sidebar.subheader("👨‍💼 Mentor")
st.sidebar.text("Kylian SANTOS")


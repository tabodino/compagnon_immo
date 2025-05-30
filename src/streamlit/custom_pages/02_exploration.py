import streamlit as st
import pandas as pd
import requests
from io import StringIO


SALES_68_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_ventes_68.csv"
RENTALS_68_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_locations_68.csv"
RAW_DATA_PATH = "data/raw/"
# Evite un long traitement (temps chargement du dataset complet)
DVF_DUPLICATED = 1302023
DVF_NB_ROWS = 2195790

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


@st.cache_data
def get_dataframe_info(df):
    """
    Returns a DataFrame containing summary information about the provided DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame for which information is to be summarized.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with summary statistics including:
        - Nombre de lignes: Number of rows in the DataFrame.
        - Nombre de colonnes: Number of columns in the DataFrame.
        - Nombre de doublons: Number of duplicate rows in the DataFrame.
        - Manquants (%): Percentage of missing values in the DataFrame.
        - Nombre de variables quantitatives: Number of numeric columns.
        - Nombre de variables qualitatives: Number of object columns.
    """
    dataset_info = pd.DataFrame(
        {
            "Value": [
                df.shape[0],
                df.shape[1],
                df.duplicated().sum(),
                round((df.isna().sum().sum() / df.size) * 100, 2),
                len(df.select_dtypes("number").columns),
                len(df.select_dtypes("object").columns),
            ]
        },
        index=[
            "Nombre de lignes",
            "Nombre de colonnes",
            "Nombre de doublons",
            "Manquants (%)",
            "Nombre de variables quantitatives",
            "Nombre de variables qualitatives",
        ],
    )
    return dataset_info


def create_dafaframe_by_type(df, type):
    """
    Returns a DataFrame containing the columns of the provided DataFrame
    that match the specified data type.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to extract columns.
    type:  string [object | number]

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the columns of the provided DataFrame
        that match the specified data type.
    """
    type_vars = df.select_dtypes(include=type)
    dtype_info_df = pd.DataFrame(
        {"colonne": type_vars.columns, "type": type_vars.dtypes}
    ).set_index("colonne")
    return dtype_info_df


st.header("🔍 Exploration des données")

st.write("")

st.subheader("Dataset Ventes 68 :")

sales_df, nb_rows = load_dataframe(SALES_68_URL, index_col="idannonce", nrows=-1)
st.dataframe(sales_df)
sales_infos = get_dataframe_info(sales_df)
st.dataframe(sales_infos)

col1, col2 = st.columns(2)

with col1:
    st.write("Variables quantitatives")
    st.dataframe(create_dafaframe_by_type(sales_df, "number"))

with col2:
    st.write("Variables qualitatives")
    st.dataframe(create_dafaframe_by_type(sales_df, "object"))

st.write("---")

st.subheader("Dataset Locations 68 :")

rentals_df, nb_rows = load_dataframe(RENTALS_68_URL, index_col="idannonce", nrows=-1)
st.dataframe(rentals_df)
rentals_infos = get_dataframe_info(rentals_df)
st.dataframe(rentals_infos)

col1, col2 = st.columns(2)

with col1:
    st.write("Variables quantitatives")
    st.dataframe(create_dafaframe_by_type(rentals_df, "number"))

with col2:
    st.write("Variables qualitatives")
    st.dataframe(create_dafaframe_by_type(rentals_df, "object"))

st.write("Les données analysées couvrent la période de 2019 à 2023.")

st.write("---")

st.subheader("Dataset Valeurs foncières :")

st.info(
    "ℹ️ Nous devons collecter les datasets progressivement, en les "
    "regroupant d'abord par année, puis en les triant par département, "
    "afin de constituer l'ensemble complet des données."
)


dvf_df = st.session_state["datasets"]["dvf_df"]
st.dataframe(dvf_df)
dvf_infos = get_dataframe_info(dvf_df)
dvf_infos.loc["Nombre de lignes", "Value"] = DVF_NB_ROWS
dvf_infos.loc["Nombre de doublons", "Value"] = DVF_DUPLICATED
st.dataframe(dvf_infos)

col1, col2 = st.columns(2)

with col1:
    st.write("Variables quantitatives")
    st.dataframe(create_dafaframe_by_type(dvf_df, "number"))

with col2:
    st.write("Variables qualitatives")
    st.dataframe(create_dafaframe_by_type(dvf_df, "object"))

st.write("Les données analysées couvrent la période de 2020 à 2024.")

st.warning(
    "⚠️ Le dataset étant volumineux, il peut entraîner des "
    "contraintes de ressources computationnelles lors de son chargement."
)

st.write("---")

st.write("### Automatisation de la récupération des données")
st.write(
    "Afin de simplifier l'extraction et l'organisation des données "
    "issues de **data.gouv**, nous avons mis en place un **ETL** structuré. "
    "Grâce à une série de **scripts Python**, il est possible de générer "
    "des datasets adaptés à nos besoins, que ce soit par année, par "
    "département ou sous forme globale. Cette solution nous permet "
    "également de surmonter certaines contraintes de ressources "
    "computationnelles liées au traitement de volumes de données "
    "trop importants."
)

st.markdown(
    """
- **Création dataset global (toutes les années et tous les départements):**

```
python src/data/make_dataset.py all
```
                   
- **Création dataset par année (tous les départements pour une année spécifique):**
```
python src/data/make_dataset.py year --year 2024
```

- **Création dataset par département (toutes les années pour un département spécifique):**

```
python src/data/make_dataset.py dep_all --dep 75
```

- **Création dataset spécifique (un seul département pour une année):**

```
python src/data/make_dataset.py dep --dep 75 --year 2024
```
"""
)

st.write("---")
st.write("### Recherche d'outliers")
st.write("Une recherche d'outliers a été effectuée sur chaque dataset.")

st.write("Exemples de valeurs extrêmes / aberrantes :")

st.code(
    """
----------------- surface terrain ----------------
surface terrain médian         : 589.0
surface terrain maximal        : 431600.0
surface terrain > 100000m²     : 7
--------------------------------------------------
Pas d'hypothèse d'abberation ici, il peut s'agir de bois ou de 
terrain agricole dépendant du bien.
Pour la plus grande valeur Google Map nous montre l'emplacement d'un bois.

------------------ surface balcon ----------------
surface balcon médian         : 10.0
surface balcon maximal        : 801.0
surface balcon > surface      : 1
--------------------------------------------------
Valeur aberrante, la surface balcon est > à la surface du bien
        
-------------------- DPE C ------------------
Nbre DPE C  médian           : 196.0
Nbre DPE C  maximal          : 988.0
Nbre DPE C > 800             : 8
---------------------------------------------
Ceux sont des valeurs extrêmes et non aberrantes car la performance 
        énergétique 'dpeL' est mauvaise également

-------------- Année de construction -------------
Année minimale                          : 971.0
Nombre d'années < 1000                  : 1
Nombre d'années proche 1970 du quartier : 2
Médiane du code quartier '6825700'      : 1920.0
--------------------------------------------------
Cas isolé, semble être une erreur de saisie avec 1971

----------------- Nbre pieces ---------------
Nbre pieces médian           : 4.0
Nbre pieces maximal          : 43
Nbre pieces > 20             : 2
---------------------------------------------
La 2eme ligne ressemble à une erreur de saisie, la surface n'est pas cohérente.
idannonce surface typedebien nb_pieces TYP_IRIS_x TYP_IRIS_y prix_m2_vente prix_bien						
157949837 	883 	m 	22 	Z 	Z 	475.65      420000
146173003 	2 	an     43 	H 	H 	79500.00    159000
"""
)

st.write("---")
st.write("### Informations supplémentaires")
st.info(
    "ℹ️ Les données des départements 67-68 (Bas-Rhin, Haut-Rhin) pour le dernier dataset ne sont pas disponibles en raison de "
    "spécificités juridiques et administratives propres à ces territoires."
    "Leur situation géopolitique historique avec l'Allemagne a également influencé l'accès à ces informations. "
    "Ces données existent mais ne sont pas accessibles publiquement."
)

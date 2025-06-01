import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import requests
from io import StringIO


SALES_68_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_ventes_68.csv"
RENTALS_68_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_locations_68.csv"

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


IMG_FOLDER = "reports/figures/"

st.header("📊 Analyse des données")

st.write("")

st.subheader("Dataset Ventes 68 :")

# sales_df, nb_rows = load_dataframe(SALES_68_URL, index_col="idannonce", nrows=-1)


# num_cols = sales_df.select_dtypes(include="number").columns
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(sales_df[num_cols])
# scaled_df = pd.DataFrame(scaled_data, columns=num_cols)


# # Matrice de corrélation de certaines variables numériques
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.heatmap(
#     sales_df[
#         [
#             "etage",
#             "surface",
#             "surface_terrain",
#             "nb_pieces",
#             "prix_bien",
#             "prix_maison",
#             "prix_terrain",
#             "mensualiteFinance",
#             "balcon",
#             "eau",
#             "bain",
#             "places_parking",
#             "annee_construction",
#             "prix_m2_vente",
#         ]
#     ].corr(),
#     annot=True,
#     cmap="coolwarm",
# )
# plt.title("Corrélations entre certaines variables numériques")
# st.pyplot(fig)

st.write(
    "On peut noter que la variable 'prix_bien' n'est pas totalement corrélée avec 'prix_maison' et 'prix_terrain'. \
    Étant donné le grand nombre de valeurs manquantes de ces deux variables, cela n'est pas étonnant. \
    Il y a logiquement une forte corrélation entre 'prix_bien' et 'surface', mais également avec 'nb_pieces' et 'nb_toilettes', \
    malgré pas mal de valeurs manquantes pour cette dernière variable. La corrélation moyenne de 'prix_bien' avec 'prix_m2_vente' \
    peut éventuellement découler d'une forte disparité entre les prix de vente et une plus grande homogénéité des prix au m2, ou inversement."
)

html_path = "reports/figures/dist_numnorm_ventes_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")


st.write("---")

# Répartition des biens selon le type de bien
html_path = "reports/figures/repartition_biens_ventes_68_hist.html"


if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write(
    "On note une quasi égalité dans la répartition entre le nombre de maisons (50,25%) et le nombre d'appartements (49,75%).\
    Une grande différence est à noter entre les bien anciens, en majorité, et les biens neufs, très peu nombreux (4,65%).\
    Nous pouvons donc nous questionner sur la représentativité de ces derniers pour une prédiction pertinente et de qualité."
)

st.write("---")


# Distribution des prix de ventes selon le type de bien
html_path = "reports/figures/distribution_ventes_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write(
    "On peut noter une forte disparité de prix entre maisons et appartements, ainsi qu'entre les biens neufs et anciens. \
    Pour les maisons anciennes, on observe que le premier quartile (248K) est supérieur au troisième quartile des appartements anciens (236K), \
    là où le prix des maison et appartements neufs sont bien plus ramassés, avec un prix bas plus élevé que dans l'ancien et des valeurs hautes n'excédant pas 730K€."
)

st.write("---")

# Distribution des prix au m2 selon le type de bien
html_path = "reports/figures/distribution_ventes_m2_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write(
    "On peut noter que contrairement à la distribution des prix de vente qui se caractérise par de fortes disparités entre maisons et appartements, \
    ainsi qu'entre neuf et ancien, la médiane du prix de vente au m2 est beaucoup plus homogène (ma = 2676€, mn = 2650€, aa = 2387€), \
    exceptée pour les appartements neufs qui semblent être nettement plus chers que les autres biens (an = 3356€). \
    Les hypothèses qui pourraient expliquer cette différence pour les appartements neufs sont à chercher du côté des variables de surface, \
    de localisation, de prestations (nb_toilettes, balcon, ascenseur, parking, etc.)"
)

st.write("---")


# Distribution de la surface des biens
html_path = "reports/figures/distribution_surface_vente_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.markdown(
    """
    On peut constater que lorque nous combinons les médianes des prix au m2 et les médianes des surfaces, \
    nous obtenons des prix de ventes sensiblement supérieurs à la médiane des prix de vente pour les biens, \
    excepté pour les maisons neuves :
    <ul>
        <li> Appartements neufs : 231 564 € (225 950 €)</li>
        <li> Appartements anciens : 171 864 € (168 000 €)</li>
        <li> Maisons neuves : 280 900 € (282 427 €)</li>
        <li> Maisons anciennes : 331 824 € (325 000 €)</li>
        
    </ul>
    """,
    unsafe_allow_html=True,
)

st.write("---")


# Évolution prix m2 médian
html_path = "reports/figures/evolution_prix_m2_median_68_line.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.markdown(
    """
    On peut noter une indisponibilité des données concernant les ventes de maisons neuves au-delà de juillet 2020. \
    Une hypothèse pourrait être l'irruption du Covid-19 qui aurait stoppé les livraisons et/ou nouveaux projets de construction.\
    Mais cela n'explique pas pourquoi les données concernant les ventes d'appartements neufs continuent au-delà de juillet 2020. \
    Ces données manquantes pourraient éventuellement expliquer le niveau plus élevé du prix au m2 des appartements neufs par rapport \
    aux maisons neuves dans le graphique précédent. On note en effet que les courbes démarrent sensiblement au même niveau.\
    À voir comment pallier ce manque de données.
    """
    )

st.write("---")


# Répartition des ventes par année et par type de bien
html_path = "reports/figures/repartition_annee_ventes_68_hist.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.markdown(
    """
    Malgré la disponibilité de données pour l'ensemble de l'année 2019, nous pouvons constater une très faible activité de ventes durant cette année. \
    Il est possible que les données de 2019 soient incomplètes, ou que la forte demande de biens hors grandes agglomérations suite au début \
    de la crise du Covid-19 ait eu un impact sur le marché immobilier car l'on note une plus forte activité de ventes pour les années suivantes.\
    Il faudrait pouvoir comparer ces données avec celles de 2018 pour vérifier cette hypothèse. \
    """
    )

st.write("---")


st.subheader("Dataset Locations 68 :")

# rentals_df, nb_rows = load_dataframe(RENTALS_68_URL, index_col="idannonce", nrows=-1)

# num_cols = rentals_df.select_dtypes(include="number").columns
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(rentals_df[num_cols])
# scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

# # Matrice de corrélation de certaines variables numériques
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.heatmap(
#     rentals_df[
#         [
#             "etage",
#             "surface",
#             "surface_terrain",
#             "nb_pieces",
#             "prix_bien",
#             "balcon",
#             "eau",
#             "bain",
#             "places_parking",
#             "annee_construction",
#         ]
#     ].corr(),
#     annot=True,
#     cmap="coolwarm",
# )
# plt.title("Corrélations entre certaines variables numériques")
# st.pyplot(fig)

# st.write("---")

# Distribution des variables quantitatives
html_path = "reports/figures/dist_numnorm_locations_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


# Répartition des biens selon le type de bien
html_path = "reports/figures/repartition_biens_rent_68_hist.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


# Répartition des loyers et surfaces selon le type de bien
html_path = "reports/figures/distribution_loyers_surface_locations_68_box.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


# Évolution des loyers médians selon le type de bien
html_path = "reports/figures/evolution_loyers_median_68_line.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


# Répartition des locations par année selon le type de bien
html_path = "reports/figures/repartition_annee_locations_68_hist.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


st.subheader("Dataset Valeurs foncières :")

st.image(f"{IMG_FOLDER}heatmap_data_gouv.jpg", use_container_width=True)

# Répartition des biens selon le type de bien
html_path = "reports/figures/repartition_biens_dvf_hist.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")


# Répartition des type de mutations
html_path = "reports/figures/repartition_mutation_dvf_hist.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")

st.subheader("Dataset Valeurs foncières :")

# Évolution des valeurs foncières
html_path = "reports/figures/evol_valeur_dvf_line.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
else:
    st.error(f"Le fichier HTML n'a pas été trouvé à l'emplacement : {html_path}")

st.write("---")

st.image(f"{IMG_FOLDER}repartition_prix_median.jpg", use_container_width=True)

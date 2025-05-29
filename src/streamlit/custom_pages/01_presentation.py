import streamlit as st


st.header("📃 Présentation du projet Compagnon Immo")

st.subheader("Description :")

st.write(
    "Développer une solution permettant aux acheteurs de logements d'explorer "
    "et de comparer différents territoires en termes de prix de l'immobilier, "
    "démographie, transports, services, éducation, criminalité et économie. "
    "L'application doit offrir une Data Visualization, permettant aux utilisateurs "
    "d'établir des classements et de visualiser les forces et faiblesses relatives "
    "des territoires."
)

st.subheader("Objectifs :")

st.markdown(
    """
    Deux objectifs principaux sont à considérer :
    <ul>
        <li>La prédiction de l'évolution du prix des logements selon les territoires.</li>
        <li>L'estimation du prix au m2 d'un logement donné. Une première prédiction peut être effectuée à l'aide des données tabulaires disponibles, et peut être approfondie avec des données relatives à l'annonce du bien, comme le texte descriptif ou les photos du logement.</li>
    </ul>
    <p>L'objectif global est d'aider les acheteurs à prendre des décisions éclairées en
       traduisant des données complexes et nombreuses en informations utiles et accessibles.
    </p>
""",
    unsafe_allow_html=True,
)

st.subheader("Ressources :")

st.markdown(
    """
    <span>Annonces de vente/location :</span>
    <ul>
        <li><a href="https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_ventes_68.csv">ech_annonces_ventes_68.csv</a></li>
        <li><a href="https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_locations_68.csv">ech_annonces_locations_68.csv</a></li>   
    </ul>
    <span>Prix, surface, nb de pièces, type, position géographique (France des 5 dernières années) :</span>
    <ul>
        <li>
            <a href="https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/">demandes-de-valeurs-foncieres-geolocalisees</a>
        </li>
    </ul>
""",
    unsafe_allow_html=True,
)

st.subheader("Ressource additionnelle :")

st.markdown(
    """
    Une application en lien avec le sujet:
    <ul>
        <li><a href="https://comparateur-communes.fr/">comparateur-communes</a></li>
    </ul>
""",
    unsafe_allow_html=True,
)

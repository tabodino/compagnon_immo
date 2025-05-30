import streamlit as st

st.header("⚙️ Preprocessing des données")

st.info(
    "La phase de préprocessing a nécessité plusieurs allers-retours pour ajuster notre méthodologie. "
    "Nous avons affiné le traitement des données progressivement afin d'obtenir un dataset "
    "le plus propre et le plus cohérent possible."
)

st.subheader("Dataset Ventes 68")

st.markdown("#### Étapes du préprocessing")

st.write("-  **1. Suppression des colonnes obsolètes**")
st.write(
    "Suite aux analyses précédentes, certaines colonnes ont été supprimées car elles n'apportaient pas de valeur ajoutée."
)
st.write(
    "`['type_annonceur','parking', 'places_parking', "
    "'surface_balcon', 'charges_copro','duree_int',"
    "'loyer_m2_median_n6', 'nb_log_n6', 'taux_rendement_n6',"
    " 'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7'] `"
)
st.write("-  **2. Gestion des valeurs manquantes (24.41%)**")
st.write(
    "Nous avons dû créer de nouvelles catégories pour les variables liées au chauffage, "
    "incluant le type, le système et l'énergie utilisée."
)
st.write(
    "Pour les variables relatives aux prix et à l'année de construction, "
    "nous nous sommes basés sur les valeurs médianes du quartier afin "
    "d'assurer une meilleure cohérence des données."
)
st.write("-  **3. Gestion des types de données**")
st.write(
    "Nous avons converti les types de données pour assurer une meilleure compatibilité avec les analyses ultérieures."
)

st.write("-  **4. Formatage et préparation au merge**")
st.write(
    "Une étape de formatage a été réalisée pour préparer le dataset à la fusion avec le dataset des valeurs foncières."
)
st.write("Exemple:")
st.code(
    """
# Création d'un ID parcelle hypothétique
# Rapprochement avec le dataset de location qui contient d'autres départements que le 68.
# Exemple pour le département 69 :

print(dvf_df[dvf_df['code_departement'] == '69']['id_parcelle'].sample(5))

df_formatted['id_parcelle'] = df_formatted[['code_commune', 'TYP_IRIS_y',
    'TYP_IRIS_x', 'UU2010']].astype(str).apply(lambda x: ''.join(x), axis=1)
""",
    language="python",
)

st.write("---")

st.subheader("Dataset Locations 68")

st.write(
    "Toutes les étapes de nettoyage et de transformation ont été réalisées sur "
    "ce dataset en suivant principalement les étapes présentées ci-dessus. "
    "Après réflexion, nous avons finalement décidé de **ne pas l'inclure** "
    "dans notre analyse."
)

st.write(
    "Ce dataset reste similaire à 'Ventes 68' et aurait pu être intégré à une approche plus large."
)

st.write("---")

st.subheader("Dataset Valeurs foncières")

st.markdown("#### Étapes du préprocessing")
st.write("-  **1. Suppression des colonnes obsolètes**")
st.code(
    """
['ancien_code_commune', 'ancien_nom_commune',' ancien_id_parcelle', 'numero_volume',
'adresse_suffixe', 'adresse_numero', 'adresse_nom_voie', 'adresse_code_voie']
 """
)
st.write("- **2. Gestion des valeurs manquantes (environ 46%)**")
st.write("Ce jeu de données présente un taux important de valeurs manquantes.")
st.write(
    "Il contenait **près de 400 000 valeurs manquantes** pour la longitude et latitude."
)
st.write(
    "Nous avons utilisé un fichier regroupant les codes communes et leurs coordonnées GPS "
    "à partir de cette source :"
)
st.write(
    "[🔗 CodesPostaux_communesINSEE.csv](https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/codesPostaux_communesINSEE.csv)"
)

st.write("- **3. Conversion des types de données**")
st.write(
    "La variable `'date_mutation'` a été convertie au format `datetime` "
    "pour faciliter les analyses temporelles. Des colonnes `année`, `mois`, `jour` "
    "ont été créées pour permettre une manipulation plus facile des données."
)


st.markdown("- **4. Gestion des ressources et de la volumétrie**")
st.write(
    "Le volume du dataset a engendré quelques **problèmes de ressources computationnelles** et "
    "des lenteurs de traitement, nécessitant une optimisation des étapes de préprocessing."
)

st.write("---")

st.subheader("Séparation des données")
st.write(
    "La séparation des jeux de données a été réalisée selon les objectifs d'analyse :"
)

st.write("**Prédiction du prix au m²** :")
st.write(">  - Séparation **80% entraînement / 20% test**")
st.write("**Prédiction de l'évolution temporelle** :")
st.write(">  - **Données d'entraînement** : années **2020 à 2023**")
st.write(">  - **Données de test** : année **2024**")

st.write("---")

st.subheader("Encodage des variables catégorielles")

st.write("La stratégie d'encodage dépend du nombre de modalités de chaque variable :")

st.markdown("##### Peu de modalités")
st.write("- **One-Hot Encoding** (ex : `'type_local'`, `'nature_mutation'`)")

st.markdown("##### Grand nombre de modalités")
st.write(
    "- **Frequency Encoding** (ex : `'code_nature_culture'`, `'code_nature_culture_speciale'`, `'code_commune'`, `'code_departement'`)"
)

st.markdown("##### Beaucoup de valeurs uniques")
st.write(
    "- **Binary Encoding** (ex : `'code_nature_culture'`, `'code_nature_culture_speciale'`, `'code_commune'`, `'code_departement'`)"
)

st.write("---")

st.subheader("Standardisation des données")

st.write(
    "Nous avons utilisé **RobustScaler**, qui est **moins sensible aux valeurs extrêmes** "
    "et offre une meilleure gestion des outliers que StandardScaler."
)

st.write("---")

st.subheader("Enrichissement des données")

st.write("Afin d'optimiser nos performances sur les séries temporelles, "
         "nous avons intégré des données économiques supplémentaires, "
         "notamment le taux du livret A, le taux d'inflation mensuelle et le "
         "taux moyen des prêts bancaires.")

st.code("""
# Source INSEE
df_inflation = pd.read_csv('../data/raw/inflation-2020-2024.csv', index_col=0)

df_inflation = df_inflation.drop('mois',axis=1)
df_inflation.rename(columns={'index': 'mois'}, inplace=True)
df_inflation.columns = df_inflation.columns.astype(int)
df_inflation["mois"] = df_inflation.index.astype(int)

def get_inflation(row):
    mois = row['mois']
    annee = row['annee']
    try:
        return df_inflation.loc[mois, annee]
    except KeyError:
        return np.nan

df_dep['taux_inflation'] = df_dep.apply(get_inflation, axis=1)

# Source Banque de France
taux_livret_a = {
     2020: 0.50,
     2021: 0.50,
     2022: 1.38,
     2023: 2.50,
     2024: 3.00,
}
taux_moyen_bancaire = {
     2020: 0.48,
     2021: 0.47,
     2022: 0.78,
     2023: 1.37,
     2024: 1.80,
}
df_dep["taux_livret_a"] = df_dep["annee"].map(taux_livret_a)
df_dep["taux_moyen_bancaire"] = df_dep["annee"].map(taux_moyen_bancaire)
""")
import pandas as pd
import numpy as np

INSEE_URL = "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/codesPostaux_communesINSEE.csv"


def drop_columns(df):
    print("Suppression colonnes non pertinentes...")
    try:
        df = df.drop(
            [
                "ancien_code_commune",
                "ancien_nom_commune",
                "ancien_id_parcelle",
                "numero_volume",
                "adresse_suffixe",
                "adresse_numero",
                "adresse_nom_voie",
                "adresse_code_voie",
            ],
            axis=1,
        )
    except KeyError as e:
        print(f"[ERREUR] Impossible de supprimer la colonne : {e}")
    return df


def drop_duplicates(df):
    print("Suppression des doublons...")
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop_duplicates()
    return df


def enrich_data(df):
    print("Récupération des coordonnées GPS manquantes...")
    try:
        df_insee = pd.read_csv(INSEE_URL, sep=";")

        df_insee[["latitude", "longitude"]] = df_insee["coordonnees_gps"].str.split(
            ",", expand=True
        )

        latitude_dict = df_insee.set_index("Code_commune_INSEE")["latitude"].to_dict()
        longitude_dict = df_insee.set_index("Code_commune_INSEE")["longitude"].to_dict()

        df["latitude"] = (
            df["latitude"].fillna(df["code_commune"].map(latitude_dict)).astype("float")
        )
        df["longitude"] = (
            df["longitude"]
            .fillna(df["code_commune"].map(longitude_dict))
            .astype("float")
        )
        df = df.dropna(subset=["latitude", "longitude"])
        del df_insee
        return df
    except pd.errors.ParserError as e:
        print(f"Erreur de parsing lors du chargement des données INSEE : {e}")
        return df
    except FileNotFoundError as e:
        print(f"Fichier INSEE non trouvé : {e}")
        return df
    except Exception as e:
        print(f"Erreur inattendue lors du chargement des données INSEE : {e}")
        return df


def handle_missing_values(df):
    print("Gestion des manquants...")
    print(
        "% de manquants avant traitement",
        round((df.isna().sum().sum() / df.size) * 100, 2),
    )
    try:
        df = df.copy()
        # On supprime les valeurs foncières manquantes et les surfaces à 0.
        df = df.dropna(subset=["valeur_fonciere"])
        df = df[(df["surface_reelle_bati"].notna()) | (df["surface_terrain"].notna())]
        # Evite les divisions par 0
        df = df[(df["surface_reelle_bati"] > 0) | (df["surface_terrain"] > 0)]

        # surface_reelle_bati renseigné
        df["prix_m2_vente"] = np.where(
            (df["surface_reelle_bati"].notna()) & (df["surface_reelle_bati"] > 0),
            df["valeur_fonciere"] / df["surface_reelle_bati"],
            # surface_terrain renseigné
            np.where(
                (df["surface_terrain"].notna()) & (df["surface_terrain"] > 0),
                df["valeur_fonciere"] / df["surface_terrain"],
                np.nan,
            ),
        )

        df.loc[
            (df["code_type_local"].isna()) & (df["type_local"] == "Maison"),
            "code_type_local",
        ] = 1.0
        df.loc[
            (df["code_type_local"].isna()) & (df["type_local"] == "Appartement"),
            "code_type_local",
        ] = 2.0
        df.loc[
            (df["code_type_local"].isna()) & (df["type_local"] == "Dépendance"),
            "code_type_local",
        ] = 3.0
        df.loc[
            (df["code_type_local"].isna())
            & (df["type_local"] == "Local industriel. commercial ou assimilé"),
            "code_type_local",
        ] = 4.0
        df.loc[(df["code_type_local"].isna()), "code_type_local"] = 5.0

        df.loc[
            (df["type_local"].isna()) & (df["code_type_local"] == 1.0), "type_local"
        ] = "Maison"
        df.loc[
            (df["type_local"].isna()) & (df["code_type_local"] == 2.0), "type_local"
        ] = "Appartement"
        df.loc[
            (df["type_local"].isna()) & (df["code_type_local"] == 3.0), "type_local"
        ] = "Dépendance"
        df.loc[
            (df["type_local"].isna()) & (df["code_type_local"] == 4.0), "type_local"
        ] = "Local industriel. commercial ou assimilé"
        df.loc[(df["type_local"].isna()), "type_local"] = "Autre"

        df.loc[
            (df["surface_reelle_bati"].isna()) & (df["code_nature_culture"] != "AB"),
            "surface_reelle_bati",
        ] = 0

        df.loc[
            (df["nombre_pieces_principales"].isna())
            & (df["code_nature_culture"] != "AB"),
            "nombre_pieces_principales",
        ] = 0

        df["lot1_surface_carrez"] = df["surface_reelle_bati"]

        df["code_nature_culture"] = df["code_nature_culture"].fillna("NS")

        df["nature_culture"] = df["nature_culture"].fillna("autres")

        df["code_nature_culture_speciale"] = df["code_nature_culture_speciale"].fillna(
            "NS"
        )

        df["nature_culture_speciale"] = df["nature_culture_speciale"].fillna("Autre")

        df["surface_terrain"] = df["surface_terrain"].fillna(0)

        df = df.dropna(subset=["longitude", "latitude", "valeur_fonciere"], axis=0)

        df["surface_reelle_bati"] = df["surface_reelle_bati"].fillna(0)

        df["lot1_surface_carrez"] = df["lot1_surface_carrez"].fillna(0)
        df["lot2_surface_carrez"] = df["lot2_surface_carrez"].fillna(0)
        df["lot3_surface_carrez"] = df["lot3_surface_carrez"].fillna(0)
        df["lot4_surface_carrez"] = df["lot4_surface_carrez"].fillna(0)
        df["lot5_surface_carrez"] = df["lot5_surface_carrez"].fillna(0)

        df["lot1_numero"] = df["lot1_numero"].fillna(0)
        df["lot2_numero"] = df["lot2_numero"].fillna(0)
        df["lot3_numero"] = df["lot3_numero"].fillna(0)
        df["lot4_numero"] = df["lot4_numero"].fillna(0)
        df["lot5_numero"] = df["lot5_numero"].fillna(0)

        df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(0)

        df.loc[df["nombre_lots"].isna(), "nombre_lots"] = 1

        df.loc[
            (df["prix_m2_vente"].isna()) & (df["surface_reelle_bati"] > 0),
            "prix_m2_vente",
        ] = (
            df["valeur_fonciere"] / df["surface_reelle_bati"]
        )

        df.loc[
            (df["prix_m2_vente"].isna())
            & (df["surface_reelle_bati"] == 0)
            & (df["surface_terrain"] > 0),
            "prix_m2_vente",
        ] = (
            df["valeur_fonciere"] / df["surface_terrain"]
        )

        df.loc[
            (df["prix_m2_vente"].isna())
            & (df["surface_reelle_bati"] == 0)
            & (df["surface_terrain"] == 0),
            "prix_m2_vente",
        ] = 0

        # on garde le code commune(0 manquant et réduit la multicolinéarité)
        df = df.drop("code_postal", axis=1)
        # on a déjà le code commune
        df = df.drop("nom_commune", axis=1)
        # cas spécifigue pour la Corse
        df["code_departement"] = df["code_departement"].replace(
            {"2A": "20", "2B": "20"}
        )

        # on a les codes correspondants
        df = df.drop(["nature_culture", "nature_culture_speciale"], axis=1)

        df = df.drop("id_parcelle", axis=1)

        df["prix_m2_vente"] = df["prix_m2_vente"].dropna()

        print(
            "% de manquants après traitement",
            round((df.isna().sum().sum() / df.size) * 100, 2),
        )

        return df

    except KeyError as e:
        print(f"[ERREUR] Impossible de traiter la colonne : {e}")


def handle_outliers(df):
    print("Gestion des outliers...")
    df = df[(df["prix_m2_vente"] > 10) & (df["prix_m2_vente"] < 10_000_000)]
    df = df[(df["valeur_fonciere"] > 100) & (df["valeur_fonciere"] < 100_000_000)]
    df = df[df["surface_terrain"] < 1_000_000]
    return df


def handle_type(df):
    print("Gestion des types...")
    df = df.copy()
    df["lot1_numero"] = (
        pd.to_numeric(df["lot1_numero"], errors="coerce").fillna(0).astype(int)
    )
    df["lot2_numero"] = (
        pd.to_numeric(df["lot2_numero"], errors="coerce").fillna(0).astype(int)
    )
    df["lot3_numero"] = (
        pd.to_numeric(df["lot3_numero"], errors="coerce").fillna(0).astype(int)
    )
    df["lot4_numero"] = (
        pd.to_numeric(df["lot4_numero"], errors="coerce").fillna(0).astype(int)
    )
    df["lot5_numero"] = (
        pd.to_numeric(df["lot5_numero"], errors="coerce").fillna(0).astype(int)
    )

    df = df[df["date_mutation"] != 0]
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], format="%Y-%m-%d")

    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.month

    df.dtypes

    return df


def clean_dataframe(df):
    df = df.copy()
    df = drop_columns(df)
    df = enrich_data(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = handle_type(df)
    df = drop_duplicates(df)
    return df

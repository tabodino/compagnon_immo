import os
import argparse
import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from encoders import ImprovedGeoEncoder, OneHotEncoderTransformer, FrequencyEncoder


RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'


def load_data(year):
    """
    Load data for a specific year.
    """
    file_path = os.path.join(PROCESSED_DATA_PATH, f"full_{year}.csv.gz")
    if os.path.exists(file_path):
        print(f"Chargement du fichier {file_path}...")
        df = pd.read_csv(file_path, compression='gzip',
                         low_memory=False, index_col='id_mutation')
        df = df[~df.index.duplicated(keep='first')]
        return df
    else:
        print(f"Fichier {file_path} non trouvé.")
        return None


def enrich_data(df):
    df['surface_par_piece'] = df['surface_reelle_bati'] / (df['nombre_pieces_principales'] + 1)
    df['ratio_terrain_bati'] = df['surface_terrain'] / (df['surface_reelle_bati'] + 1)
    df['surface_totale'] = df['surface_reelle_bati'] + df['surface_terrain']

    df['trimestre'] = ((df['mois'] - 1) // 3) + 1
    df['saison'] = df['mois'].map({
        12: 'hiver', 1: 'hiver', 2: 'hiver',
        3: 'printemps', 4: 'printemps', 5: 'printemps',
        6: 'ete', 7: 'ete', 8: 'ete',
        9: 'automne', 10: 'automne', 11: 'automne'
    })

    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Distance au centre de Paris
        paris_lat, paris_lon = 48.8566, 2.3522
        df['distance_paris'] = np.sqrt(
            (df['latitude'] - paris_lat)**2 +
            (df['longitude'] - paris_lon)**2
        ) * 111  # Conversion en km approximative

    df['zone_geo'] = 'province'
    df.loc[df['code_departement'].isin([75, 92, 93, 94, 95, 77, 78, 91]), 'zone_geo'] = 'idf'
    df.loc[df['code_departement'] == 75, 'zone_geo'] = 'paris'

    commune_counts = df['code_commune'].value_counts()
    df['commune_frequency'] = df['code_commune'].map(commune_counts)
    df['commune_rarity'] = 1 / (df['commune_frequency'] + 1)

    return df

def drop_columns(df):
    cols_to_drop = ['lot1_numero', 'lot2_numero', 'lot3_numero', 'lot4_numero',
                    'lot5_numero', 'lot4_surface_carrez', 'lot5_surface_carrez']
    df = df.drop(cols_to_drop, axis=1)
    return df

def check_prices_data(df):
    data = [
        ["Taille du dataframe :",
            f"{df.shape[0]} lignes et {df.shape[1]} colonnes"],
        ["Valeurs manquantes  :",
            f"{round((df.isna().sum().sum() / df.size) * 100, 2)}%"],
        ["Target manquante :", df['prix_m2_vente'].isna().sum()],
    ]
    print(tabulate(data, headers=["Description", "Value"], tablefmt="grid"))


def split_prices_data(df):
    print('Split des donnés...')
    X = df.drop(columns=['prix_m2_vente'])
    y = df['prix_m2_vente']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=X['code_departement'], random_state=42)
    return X_train, X_test, y_train, y_test, y


def encode_prices_data(X_train, X_test, y):
    print('Encodage des donnés...')
    # Geo encoder
    geo_encoder = ImprovedGeoEncoder()
    X_train = geo_encoder.fit_transform(X_train, y)
    X_test = geo_encoder.transform(X_test)
    # One Hot Encoding
    one_hot_cols = ['type_local', 'nature_mutation', 'zone_geo', 'saison']
    ohe = OneHotEncoderTransformer(one_hot_cols)
    X_train_encoded = ohe.fit_transform(X_train)
    X_test_encoded = ohe.transform(X_test)

    rename_dict = {
        'type_local_Local industriel. commercial ou assimilé':
        'type_local_Local_industriel_commercial',
        "nature_mutation_Vente en l'état futur d'achèvement":
        'nature_mutation_Vente_en_l_etat_futur_achevement',
        "nature_mutation_Vente terrain à bâtir":
        'nature_mutation_Vente_terrain_a_batir'
    }

    X_train_encoded.rename(columns=rename_dict, inplace=True)
    X_test_encoded.rename(columns=rename_dict, inplace=True)

    # Frequency Encoding
    freq_cols = ['code_nature_culture', 'code_nature_culture_speciale',
                 'code_commune', 'code_departement']
    fe = FrequencyEncoder(freq_cols)
    X_train_encoded = fe.fit_transform(X_train_encoded)
    X_test_encoded = fe.transform(X_test_encoded)

    joblib.dump(geo_encoder, "models/geo_encoder.pkl")
    joblib.dump(ohe, "models/ohe_transformer.pkl")
    joblib.dump(fe, "models/freq_encoder.pkl")

    print("Nombres de variables catégorielles restantes:", len(
        X_train_encoded.select_dtypes('object').columns))

    pd.DataFrame({"columns": X_train_encoded.columns}).to_csv(
        "models/X_train_columns.csv", index=False)

    return X_train_encoded, X_test_encoded


def standardize_prices_data(X_train, X_test):
    print('Standardisation des donnés...')
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)

    joblib.dump(scaler, "models/robust_scaler.pkl")
    X_test = pd.DataFrame(scaler.transform(
        X_test),  columns=X_test.columns, index=X_test.index)
    return X_train, X_test


def save_split_prices_files(X_train, X_test, y_train, y_test):
    X_train.to_csv(f"{PROCESSED_DATA_PATH}X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_PATH}X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["prix_m2_vente"]).to_csv(
        f"{PROCESSED_DATA_PATH}y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["prix_m2_vente"]).to_csv(
        f"{PROCESSED_DATA_PATH}y_test.csv", index=False)
    print("Données d'entrainement sauvegardé !")


def prepare_train_prices_data(year):
    df = load_data(year)
    df = enrich_data(df)
    df = drop_columns(df)
    # on a pas besoin de la date pour les prix
    df = df.drop('date_mutation', axis=1)
    check_prices_data(df)
    X_train, X_test, y_train, y_test, y = split_prices_data(df)
    X_train, X_test = encode_prices_data(X_train, X_test, y)
    X_train, X_test = standardize_prices_data(X_train, X_test)
    save_split_prices_files(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=[
                        'prices', 'series'], help='Type de chargement')
    parser.add_argument('--year', help='Année (ex: 2024)')
    args = parser.parse_args()

    if args.command == 'prices' and args.year:
        prepare_train_prices_data(args.year)

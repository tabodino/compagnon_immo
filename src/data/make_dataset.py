import argparse
import pandas as pd
from etl.extract import (
    fetch_and_read_csv_gz,
    list_gz_links_for_year,
    get_dep_link,
    get_available_years,
)
from etl.transform import clean_dataframe
from etl.load import save_dataframe


def load_all_years():
    years = get_available_years()
    all_dfs = []
    for year in years:
        print("Année :", year)
        links = list_gz_links_for_year(str(year))
        print(f"{year} : {len(links)} fichier(s) trouvé(s).")
        print(f" {len(links)} lien(s) trouvé(s) : Traitement en cours...")

        yearly_dfs = [fetch_and_read_csv_gz(link) for link in links]
        if not yearly_dfs:
            continue

        df_year = pd.concat(yearly_dfs, ignore_index=True)
        df_year = clean_dataframe(df_year)
        all_dfs.append(df_year)

    if all_dfs:
        print("Concaténation des dataframes...")
        full_df = pd.concat(all_dfs, ignore_index=True)
        save_dataframe(full_df, "data/processed/full_all_years_cleaned.csv.gz")
        print("Fichier final concaténé enregistré : full_all_years_cleaned.csv.gz")
    else:
        print("Aucun fichier n'a pu être chargé.")


def load_dep_all_years(dep):
    all_dfs = []
    years = get_available_years()
    for year in years:
        print("Année :", year)
        url = get_dep_link(str(year), dep)
        df = fetch_and_read_csv_gz(url)
        if not df.empty:
            df = clean_dataframe(df)
            all_dfs.append(df)
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        save_dataframe(final_df, f"data/processed/dep_{dep}_all_years.csv.gz")


def load_deps_by_year(year):
    links = list_gz_links_for_year(year)
    if not links:
        print(f"Aucun fichier trouvé pour l'année {year}.")
        return
    dfs = []
    for link in links:
        print(f"Traitement du fichier : {link}")
        df = fetch_and_read_csv_gz(link)
        if not df.empty:
            df_clean = clean_dataframe(df)
            dfs.append(df_clean)
    if not dfs:
        print("Aucun fichier valide n'a pu être traité.")
        return
    df_all = pd.concat(dfs, ignore_index=True)
    save_dataframe(df_all, f"data/processed/full_{year}.csv.gz")


def load_dep_year(dep, year):
    url = get_dep_link(str(year), dep)
    df = fetch_and_read_csv_gz(url)
    if df.empty:
        print(f"Aucun fichier disponible pour {dep} en {year}")
        return
    df = clean_dataframe(df)
    save_dataframe(df, f"data/processed/dep_{dep}_{year}_cleaned.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pour les données DVF")
    parser.add_argument(
        "command", choices=["all", "year", "dep", "dep_all"], help="Type de chargement"
    )
    parser.add_argument("--dep", help="Code département (ex: 34)")
    parser.add_argument("--year", help="Année (ex: 2022)")

    args = parser.parse_args()

    if args.command == "all":
        load_all_years()
    elif args.command == "year" and args.year:
        load_deps_by_year(args.year)
    elif args.command == "dep" and args.dep and args.year:
        load_dep_year(args.dep, args.year)
    elif args.command == "dep_all" and args.dep:
        load_dep_all_years(args.dep)
    else:
        print("Arguments manquants: python src/data/make_dataset.py --help")

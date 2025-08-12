import requests
import gzip
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO

BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/"


def fetch_and_read_csv_gz(url):
    try:
        response = requests.get(url)
        print(url)
        response.raise_for_status()
        with gzip.open(BytesIO(response.content), "rt", encoding="utf-8") as f:
            return pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {url} : {e}")
        return pd.DataFrame()


def get_available_years():
    try:
        res = requests.get(BASE_URL)
        soup = BeautifulSoup(res.content, "html.parser")
        return [
            a["href"].rstrip("/")
            for a in soup.find_all("a", href=True)
            if a["href"] != "../"
        ]
    except Exception as e:
        print(f"Erreur lors de la récupération des années disponibles : {e}")
        return ["2020", "2021", "2022", "2023", "2024"]


def list_gz_links_for_year(year):
    url = f"{BASE_URL}{year}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return [
            url + a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".gz")
        ]
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {url} : {e}")
        return ["2020", "2021", "2022", "2023", "2024"]


def get_dep_link(year, dep_code):
    return f"{BASE_URL}{year}/departements/{dep_code}.csv.gz"

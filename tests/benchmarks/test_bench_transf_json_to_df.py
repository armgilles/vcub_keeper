import pytest
import json
import random
from datetime import datetime, timedelta

from vcub_keeper.production.data import transform_json_api_bdx_station_data_to_df
from vcub_keeper.transform.features_factory import get_consecutive_no_transactions_out, process_data_cluster
from vcub_keeper.config import ROOT_TESTS_DATA


def read_json_data(file_name="data_test_api_from_bdx.json"):
    """
    Read test json data
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + file_name) as f:
        station_json_loaded = json.load(f)
    return station_json_loaded


def generate_data(num_stations, num_months, seed=None):
    """
    Permets de générer des données d'activité pour un certain nombre de stations
    et simule les données de l'API de Bordeaux Métropole.

    Exemple :
    ---------
    station_json_loaded = generate_data(num_stations=20, num_months=1, seed=2024) # (86400, 8)
    """
    # Si une seed est fournie, on l'utilise pour la reproductibilité
    if seed is not None:
        random.seed(seed)

    # Dictionnaire de base GeoJSON
    data = {"type": "FeatureCollection", "features": []}

    # Début de la période temporelle (par exemple aujourd'hui)
    start_time = datetime.now()

    # Pour chaque station
    for station_id in range(1, num_stations + 1):
        # Nom de la station
        station_name = f"Station {station_id}"

        # Capacité totale de la station (constante pour chaque station)
        total_places = random.randint(20, 50)  # Entre 20 et 50, la capacité totale fixe (nbplaces + nbvelos)

        # Pour chaque mois sur la période donnée
        for month in range(num_months):
            # Début du mois (on peut utiliser des intervalles de 5 minutes comme exemple)
            month_start = start_time + timedelta(days=30 * month)

            # Créer un enregistrement pour chaque intervalle de 5 minutes sur le mois
            # Chaque 5 minutes = 288 enregistrements par jour
            for minute in range(0, 30 * 24 * 60, 5):  # Chaque 5 minutes
                time = month_start + timedelta(minutes=minute)

                # Définir l'état avec une probabilité biaisée (95% "CONNECTEE", 5% "DECONNECTEE")
                etat = random.choices(["CONNECTEE", "DECONNECTEE"], weights=[95, 5])[0]

                # Faire varier le nombre de vélos disponibles entre 0 et la capacité totale
                nbvelos = random.randint(0, total_places)

                # Calculer le nombre de places disponibles (nbplaces = capacité totale - nbvelos)
                nbplaces = total_places - nbvelos

                feature = {
                    "type": "Feature",
                    "properties": {
                        "time": time.isoformat(),  # Format ISO 8601
                        "gid": station_id,
                        "ident": station_id,
                        "nom": station_name,
                        "etat": etat,
                        "nbplaces": nbplaces,  # Calculé en fonction du nombre de vélos
                        "nbvelos": nbvelos,  # Varie à chaque enregistrement
                    },
                }
                data["features"].append(feature)

    return data


# Read test json data from real data
station_json_loaded = read_json_data()

# Read simulated data
station_json_loaded_simu = generate_data(num_stations=20, num_months=1, seed=2024)  # (86400, 8)
station_json_loaded_simu_big = generate_data(num_stations=20, num_months=1, seed=2024)  # (777600, 8)


@pytest.mark.benchmark
def test_benchmark_transf_json_to_df(json_data=station_json_loaded):
    """
    Benchmark for transforming JSON data to DataFrame
    """

    station_df_from_json = transform_json_api_bdx_station_data_to_df(json_data)


@pytest.mark.benchmark
def test_benchmark_pipepline_transform(json_data=station_json_loaded_simu):
    """
    Benchmark for all transformation steps before ML step.
    """

    station_df = transform_json_api_bdx_station_data_to_df(json_data)
    station_df = get_consecutive_no_transactions_out(station_df)
    station_df = process_data_cluster(station_df)


@pytest.mark.benchmark
def test_benchmark_pipepline_transform_big(json_data=station_json_loaded_simu_big):
    """
    Benchmark for all transformation steps before ML step with larger dataset.
    """

    station_df = transform_json_api_bdx_station_data_to_df(json_data)
    station_df = get_consecutive_no_transactions_out(station_df)
    station_df = process_data_cluster(station_df)

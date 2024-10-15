import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from vcub_keeper.config import ROOT_TESTS_DATA
from vcub_keeper.production.data import transform_json_api_bdx_station_data_to_df
from vcub_keeper.transform.features_factory import get_consecutive_no_transactions_out, process_data_cluster


def read_json_data(file_name="data_test_api_from_bdx.json"):
    """
    Read test json data
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + file_name) as f:
        station_json_loaded = json.load(f)
    return station_json_loaded


def generate_data(num_stations, num_days, seed=None):
    # Si une seed est fournie, on l'utilise pour la reproductibilité
    if seed is not None:
        random.seed(seed)

    # Dictionnaire de base GeoJSON
    data = {"type": "FeatureCollection", "features": []}

    # Début de la période temporelle (par exemple aujourd'hui) avec fuseau horaire UTC+1
    tz = timezone(timedelta(hours=1))  # Fuseau horaire +01:00
    start_time = datetime.now(tz=tz)  # Inclut le fuseau horaire

    # Calcul du nombre total de minutes à générer pour chaque jour (24 heures * 60 minutes // 5 minutes)
    interval_per_day = 24 * 60 // 5  # Nombre d'intervalle de 5 minutes par jour
    intervals_per_period = interval_per_day * num_days  # Nombre total d'intervalles pour tous les jours

    # Liste temporaire pour accumuler les features
    features = []

    # Pour chaque station
    for station_id in range(1, num_stations + 1):
        # Nom de la station
        station_name = f"Station {station_id}"

        # Capacité totale de la station (constante pour chaque station)
        total_places = random.randint(20, 50)  # Entre 20 et 50, la capacité totale fixe (nbplaces + nbvelos)

        # Pré-générer les états ("CONNECTEE" ou "DECONNECTEE") pour tous les intervalles avec une seule opération
        etats = random.choices(["CONNECTEE", "DECONNECTEE"], weights=[95, 5], k=intervals_per_period)

        # Générer tous les enregistrements pour la période (en nombre de jours)
        for i in range(intervals_per_period):
            # Calculer le temps exact pour cet intervalle de 5 minutes
            time = start_time + timedelta(minutes=i * 5)

            # Faire varier le nombre de vélos disponibles entre 0 et la capacité totale
            nbvelos = random.randint(0, total_places)

            # Calculer le nombre de places disponibles (nbplaces = capacité totale - nbvelos)
            nbplaces = total_places - nbvelos

            # Créer l'enregistrement (feature)
            feature = {
                "type": "Feature",
                "properties": {
                    "time": time.isoformat(timespec="seconds"),
                    "gid": station_id,
                    "ident": station_id,
                    "nom": station_name,
                    "etat": etats[i],  # Utilisation de la liste pré-générée
                    "nbplaces": nbplaces,  # Calculé en fonction du nombre de vélos
                    "nbvelos": nbvelos,  # Varie à chaque enregistrement
                },
            }
            # Ajouter la feature à la liste temporaire
            features.append(feature)

    # Ajouter toutes les features d'un coup
    data["features"].extend(features)

    return data


# Read test json data from real data
station_json_loaded = read_json_data()

# Read simulated data
station_json_loaded_simu = generate_data(num_stations=3, num_days=3, seed=2024)  # (1296, 8)
station_json_loaded_simu_big = generate_data(num_stations=7, num_days=15, seed=2024)  # (15127, 8)


@pytest.mark.benchmark
def test_benchmark_transf_json_to_df(json_data=station_json_loaded):
    """
    Benchmark for transforming JSON data to DataFrame
    """

    station_df_from_json = transform_json_api_bdx_station_data_to_df(json_data).collect()


@pytest.mark.benchmark
def test_benchmark_pipepline_transform(json_data=station_json_loaded_simu):
    """
    Benchmark for all transformation steps before ML step.
    """

    # todo : using pipe() ?
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

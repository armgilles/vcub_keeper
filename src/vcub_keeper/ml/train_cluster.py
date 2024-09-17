from vcub_keeper.config import (ROOT_DATA_CLEAN, ROOT_DATA_REF,
                                ROOT_MODEL, NON_USE_STATION_ID,
                                THRESHOLD_PROFILE_STATION)
from vcub_keeper.reader.reader import *
from vcub_keeper.visualisation import *
from vcub_keeper.transform.features_factory import *
from vcub_keeper.ml.cluster import train_cluster_station
from vcub_keeper.ml.cluster_utils import export_model


def run_train_cluster():
    """

    """

    # Lecture du fichier activité
    ts_activity = read_time_serie_activity(path_directory=ROOT_DATA_CLEAN)
    # Some features engi
    ts_activity = get_consecutive_no_transactions_out(ts_activity)

    # Lecture de profile des stations pour connaitre ceux que l'on clusterise
    station_profile = read_station_profile(path_directory=ROOT_DATA_REF)

    stations_id_to_fit = \
        station_profile[station_profile['mean'] >= THRESHOLD_PROFILE_STATION]['station_id'].unique()

    # Filter station we don't want to use
    stations_id_to_fit = [station for station in stations_id_to_fit if station not in NON_USE_STATION_ID]

    # Pour chaque station, on fit le cluster
    for station_id in stations_id_to_fit:
        # Apprentissage du cluster pour la station
        clf = train_cluster_station(ts_activity, station_id=station_id)

        # Export model
        export_model(clf, station_id=station_id, path_directory=ROOT_MODEL)

    print("Fin d'apprentissage des cluster par station ID.")


if __name__ == '__main__':
    run_train_cluster()

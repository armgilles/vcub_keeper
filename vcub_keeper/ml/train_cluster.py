from vcub_keeper.config import *
from vcub_keeper.reader.reader import *
from vcub_keeper.reader.reader_utils import filter_periode
from vcub_keeper.visualisation import *
from vcub_keeper.transform.features_factory import *
from vcub_keeper.ml.cluster import train_cluster_station, predict_anomalies_station
from vcub_keeper.ml.cluster_utils import load_model, export_model


def run_train_cluster():
    """

    """

    # Lecture du fichier activité
    ts_activity = read_time_serie_activity()
    # Some features engi
    ts_activity = get_consecutive_no_transactions_out(ts_activity) 

    # Lecture de profile des stations pour connaitre ceux que l'on clusterise
    station_profile = read_station_profile()

    stations_id_to_fit = \
        station_profile[station_profile['mean'] >= THRESHOLD_PROFILE_STATION]['station_id'].unique()

    # Pour chaque station, on fit le cluster
    for station_id in stations_id_to_fit:
        # Apprentissage du cluster pour la station
        clf = train_cluster_station(ts_activity, station_id=station_id)
        
        # Export model
        export_model(clf, station_id=station_id)

    print("Fin d'apprentissage des cluster par station ID.")


if __name__ == '__main__':
  run_train_cluster()

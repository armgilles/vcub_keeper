from vcub_keeper.config import FEATURES_TO_USE_CLUSTER, PROFILE_STATION_RULE, SEED
from vcub_keeper.transform.features_factory import process_data_cluster
from vcub_keeper.reader.reader_utils import filter_periode
from vcub_keeper.reader.reader import read_station_profile

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from scipy import stats

def train_cluster_station(data, station_id):
    """
    Train estimator on a single station_id Time Serie.
    Process some features.
    Filter data based on filter_periode() function.
    Filter data based on status = 1.
    Use sclaler / pca / IsolationForest (contamination based on PROFILE_STATION_RULE).
    contamination is based on station profile (from read_station_profile() ).
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    station_id : int
        ID Station
    
    Returns
    -------
    clf : Pipeline
        Pipeline Scikit Learn
        
    Examples
    --------
    clf = train_cluster_station(data=ts_activity, station_id=110)
    """

    # Filter stations
    data_station = data[data['station_id'] == station_id].copy()
    
    # Feature engi for cluster
    data_station = process_data_cluster(data_station)

    # Filter data based on time & event
    data_station = filter_periode(data_station)
    
    # on prend uniquement la station quand satus ==1
    data_station_ok = data_station[data_station['status'] == 1].copy()

    # Lecture du profile activité des stations
    station_profile = read_station_profile()
    profile_station_activity = \
        station_profile[station_profile['station_id'] == station_id]['profile_station_activity'].values[0]

    print('Profile de la station N°' + str(station_id) + ' : ' + profile_station_activity)

    # Scaler
    clf_scaler = StandardScaler()

    # Cluster
    contaminsation_station = \
        1 - stats.percentileofscore(data_station_ok['consecutive_no_transactions_out'],
                                    PROFILE_STATION_RULE[profile_station_activity]) / 100
    print('Contamination de la station : ' + str(contaminsation_station))
    
    clf_cluster = IsolationForest(n_estimators=50, random_state=SEED,n_jobs=-1,
                                  contamination=contaminsation_station)
    
    # Learning
    pipe = Pipeline([
        ('scale', clf_scaler),
        ('pca', PCA(n_components=0.9)),
        ('cluster', clf_cluster)])
    pipe.fit(data_station_ok[FEATURES_TO_USE_CLUSTER])

    return pipe


def predict_anomalies_station(data, clf, station_id):
    """
    Predict anomalies on given station with an estimator
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    clf : Pipeline
        Pipeline Scikit Learn
    station_id : int
        ID Station
    
    Returns
    -------
    data : DataFrame
        anomaly's column (-1 is an anomaly)
        
    Examples
    --------
    station_pred = predict_anomalies_station(data=ts_activity, clf=clf, station_id=106)
    """


    
    # Filter stations
    data_station = data[data['station_id'] == station_id].copy()
    
    # Feature engi for cluster
    data_station = process_data_cluster(data_station)
    
    data_station['anomaly'] = clf.predict(data_station[FEATURES_TO_USE_CLUSTER])
    return data_station
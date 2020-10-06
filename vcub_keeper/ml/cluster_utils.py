from vcub_keeper.config import ROOT_MODEL

from joblib import dump, load

def export_model(clf, station_id):
    """
    Export model in ROOT_MODEL named model_station_[station_id].joblib.

    Parameters
    ----------
    clf : Pipeline
        Pipeline Scikit Learn
    station_id : int
        ID Station
    
    Returns
    -------
    None
        
    Examples
    --------
    export_model(clf, station_id=110)

    """
    dump(clf, ROOT_MODEL+'model_station_' + str(station_id) + '.joblib')


def load_model(station_id):
    """
    Load model already fit for a given ID station.

    Parameters
    ----------
    station_id : int
        ID Station
    
    Returns
    -------
    clf : Pipeline
        Pipeline Scikit Learn
        
    Examples
    --------
    clf = load_model(station_id=110)

    """
    clf = load(ROOT_MODEL+'model_station_' + str(station_id) + '.joblib')
    return clf

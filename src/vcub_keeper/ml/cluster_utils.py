from joblib import dump, load


def export_model(clf, station_id, path_directory):
    """
    Export model in ROOT_MODEL named model_station_[station_id].joblib.

    Parameters
    ----------
    clf : Pipeline
        Pipeline Scikit Learn
    station_id : int
        ID Station
    path_directory : str
        chemin d'accès (ROOT_MODEL)

    Returns
    -------
    None

    Examples
    --------
    export_model(clf, station_id=110, path_directory=ROOT_MODEL)

    """
    dump(clf, path_directory + "model_station_" + str(station_id) + ".joblib")


def load_model(station_id, path_directory):
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
    path_directory : str
        chemin d'accès (ROOT_MODEL)

    Examples
    --------
    clf = load_model(station_id=110, path_directory=ROOT_MODEL)

    """
    clf = load(path_directory + "model_station_" + str(station_id) + ".joblib")
    return clf

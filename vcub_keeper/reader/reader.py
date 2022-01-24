import pandas as pd
import numpy as np

from vcub_keeper.config import *


def read_stations_attributes(path_directory, file_name="station_attribute.csv"):
    """
    Lecture du fichier sur les attributs des Vcub à Bordeaux. Ce fichier provient de
    create.creator.py - create_station_attribute()

    Parameters
    ----------
    path_directory : str
        chemin d'accès (ROOT_DATA_REF)
    file_name : str
        Nom du fichier

    Returns
    -------
    activite : DataFrame

    Examples
    --------

    stations = read_stations_attributes(path_directory=ROOT_DATA_REF)
    """

    column_dtypes = {'station_id': 'uint8'}

    stations = pd.read_csv(path_directory+file_name, sep=',',
                           dtype=column_dtypes)

    return stations


def read_activity_vcub(file_path="../../data/bordeaux.csv"):
    """
    Lecture du fichier temporelle sur l'activité des Vcub à Bordeaux
    Modification par rapport au fichier original : 
        - Modification des type du DataFrame
        - Mapping de la colonne 'state'
        - Changement de nom des colonnnes :
            - ident -> station_id
            - ts -> date
        - Triage du DataFrame par rapport à la station_id et à la date.
    Parameters
    ----------
    file_path : str
        Chemin d'accès au fichiers source

    Returns
    -------
    activite : DataFrame

    Examples
    --------

    activite = read_activity_vcub()
    """

    column_dtypes = {'gid': 'uint8',
                     'ident': 'uint8',
                     'type': 'category',
                     'name': 'string',
                     'state': 'category',
                     'available_stands': 'uint8',
                     'available_bikes': 'uint8'}
    
    state_dict = {'CONNECTEE': 1,
                  'DECONNECTEE': 0}

    activite = pd.read_csv(file_path, parse_dates=["ts"], dtype=column_dtypes)

    activite['state'] = activite['state'].map(state_dict)

    # Renaming colomns
    activite.rename(columns={'ident': 'station_id'}, inplace=True)
    activite.rename(columns={'ts': 'date'}, inplace=True)

    # Sorting DataFrame on station_id & date
    activite.sort_values(['station_id', 'date'], ascending=[1, 1], inplace=True)

    # Reset index
    activite.reset_index(inplace=True, drop=True)

    return activite


def read_time_serie_activity(path_directory,
                             file_name='time_serie_activity.h5',
                             post_pressessing_status=True):
    """

    Lecture du fichier de type time series sur l'activité des stations Vcub
    dans le répertoire `/data/clean/`
    Parameters
    ----------
    path_directory : str
        chemin d'accès (ROOT_DATA_CLEAN)
    file_name : str
        Nom du fichier
    post_pressessing_status : Bool
        Permet de rectifier les status à 0 à 1 qui sont ponctuelles (uniquement 1 ligne dans la 
        time serie)

    Returns
    -------
    ts_activity : DataFrame

    Examples
    --------

    ts_activity = read_time_serie_activity(path_directory=ROOT_DATA_CLEAN)
    """

    ts_activity = pd.read_hdf(path_directory + 'time_serie_activity.h5', parse_dates=['date'])

    if post_pressessing_status is True:
        ts_activity['status_shift'] = ts_activity['status'].shift(-1) 

        # Déconnecté -> NaN
        ts_activity.loc[ts_activity['status'] == 0, 'status'] = np.NaN

        # Si le prochain status est connecté alors on remplace NaN par 1
        ts_activity.loc[ts_activity['status_shift'] == 1, 'status'] = \
            ts_activity['status'].fillna(method='pad', limit=1)

        # On remplace NaN par 0 (comme originalement)
        ts_activity['status'] = ts_activity['status'].fillna(0)

        # Drop unless column
        ts_activity = ts_activity.drop('status_shift', axis=1)

    return ts_activity


def read_meteo(path_directory, file_name='meteo.csv'):
    """
    Lecture du fichier météo dans le répertoire dans ROOT_DATA_REF
    Parameters
    ----------
    path_directory : str
        chemin d'accès (ROOT_DATA_REF)
    file_name : str
        Nom du fichier

    Returns
    -------
    meteo : DataFrame

    Examples
    --------

    meteo = read_meteo(path_directory=ROOT_DATA_REF)
    """

    meteo = pd.read_csv(path_directory + file_name, parse_dates=['date'])

    return meteo


def read_station_profile(path_directory, file_name='station_profile.csv'):
    """
    Lecture du fichier sur qui classifie les stations par rapport à leurs activité et 
    fréquences d'utilisation.
    Ce fichier est situé dans ROOT_DATA_REF

    Parameters
    ----------
    path_directory : str
        chemin d'accès (ROOT_DATA_REF)
    file_name : str
        Nom du fichier

    Returns
    -------
    station_profile : DataFrame

    Examples
    --------

    station_profile = read_station_profile(path_directory=ROOT_DATA_REF)
    """
    station_profile = pd.read_csv(path_directory+file_name, sep=',')

    return station_profile

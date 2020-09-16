import pandas as pd
import numpy as np

from vcub_keeper.config import ROOT_DATA_RAW, ROOT_DATA_REF, ROOT_DATA_CLEAN


def read_stations_attributes(file_name="tb_stvel_p.csv"):
    """
    Lecture du fichier sur les attributs des Vcub à Bordeaux.
    Ce fichier est situé dans ROOT_DATA_REF
    Modification par rapport au fichier original : 
        - Changement de nom des colonnnes :
            - NBSUPPOR -> total_stand
            - NUMSTAT -> station_id
        - Création des features lon & lat features from 'Geo Point'
    
    Parameters
    ----------
    file_name : str
        Nom du fichier
    
    Returns
    -------
    activite : DataFrame
        
    Examples
    --------
    
    stations = read_stations_attributes()
    """
    
    column_dtypes = {'NUMSTAT': 'uint8'}
    usecols = ['Geo Point', 'Geo Shape', 'COMMUNE', 'NBSUPPOR',
              'NOM', 'TYPEA', 'ADRESSE', 'TARIF', 'NUMSTAT']
    
    stations = pd.read_csv(ROOT_DATA_REF+file_name, sep=';',
                           dtype=column_dtypes, usecols=usecols)

    # Naming
    stations.rename(columns={'NBSUPPOR': 'total_stand'}, inplace=True)
    stations.rename(columns={'NUMSTAT': 'station_id'}, inplace=True)

    # Create lon / lat
    stations['lat'] = stations['Geo Point'].apply(lambda x : x.split(',')[0])
    stations['lat'] = stations['lat'].astype(float)
    stations['lon'] = stations['Geo Point'].apply(lambda x : x.split(',')[1])
    stations['lon'] = stations['lon'].astype(float)
    
    return stations


def read_activity_vcub(file_path = "../../data/bordeaux.csv"):
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
                     'available_stand': 'uint8',
                     'available_bike': 'uint8'}
    
    state_dict = {'CONNECTEE' : 1,
                  'DECONNECTEE' : 0
                 }

    activite = pd.read_csv(file_path, parse_dates=["ts"], dtype = column_dtypes)
    
    activite['state'] = activite['state'].map(state_dict)
    
    # Renaming colomns
    activite.rename(columns={'ident':'station_id'}, inplace=True)
    activite.rename(columns={'ts':'date'}, inplace=True)
    
    # Sorting DataFrame on station_id & date
    activite.sort_values(['station_id', 'date'], ascending=[1, 1], inplace=True)
    
    # Reset index
    activite.reset_index(inplace=True, drop=True)
    
    return activite


def read_time_serie_activity(file_name='time_serie_activity.h5',
                             post_pressessing_status=True):
    """
    
    Lecture du fichier de type time series sur l'activité des stations Vcub
    dans le répertoire `/data/clean/`
    Parameters
    ----------
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
    
    ts_activity = read_time_serie_activity()
    """
    
    ts_activity = pd.read_hdf(ROOT_DATA_CLEAN + 'time_serie_activity.h5', parse_dates=['date'])
    
    if post_pressessing_status is True:
        ts_activity['status_shift'] = ts_activity['status'].shift(-1) 

        # Déconnecté -> NaN
        ts_activity.loc[ts_activity['status'] == 0,
           'status'] = np.NaN

        # Si le prochain status est connecté alors on remplace NaN par 1
        ts_activity.loc[ts_activity['status_shift'] == 1,
           'status'] = ts_activity['status'].fillna(method='pad', limit=1)

        # On remplace NaN par 0 (comme originalement)
        ts_activity['status'] = ts_activity['status'].fillna(0)

        # Drop unless column
        ts_activity = ts_activity.drop('status_shift', axis=1)
    
    return ts_activity


def read_meteo(file_name='meteo.csv'):
    """
    Lecture du fichier météo dans le répertoire dans ROOT_DATA_REF
    Parameters
    ----------
    file_name : str
        Nom du fichier

    Returns
    -------
    meteo : DataFrame
        
    Examples
    --------
    
    meteo = read_meteo()
    """
    
    meteo = pd.read_csv(ROOT_DATA_REF+file_name, parse_dates=['date'])
    
    return meteo

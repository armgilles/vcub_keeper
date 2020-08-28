import pandas as pd 
import numpy as np

def get_transactions_out(data):
    """
    Calcul le nombre de prise de vélo qu'il y a eu pour une même station entre 2 points de données
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    
    Returns
    -------
    data : DataFrame
        Ajout de colonne 'transactions_out'
        
    Examples
    --------
    
    activite = get_transactions_out(activite)
    """
    
    data['available_stand_shift'] = \
    data.groupby('station_id')['available_stand'].shift(1)

    data['available_stand_shift'] = data['available_stand_shift'].fillna(data['available_stand'])

    data['transactions_out'] = data['available_stand'] - data['available_stand_shift']

    data.loc[data['transactions_out'] < 0,
                'transactions_out'] = 0
    
    # Drop non usefull column
    data.drop('available_stand_shift', axis=1, inplace=True)
    
    return data


def get_transactions_in(data):
    """
    Calcul le nombre d'ajout de vélo qu'il y a eu pour une même station entre 2 points de données
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    
    Returns
    -------
    data : DataFrame
        Ajout de colonne 'transactions_in'
        
    Examples
    --------
    
    activite = get_transactions_in(activite)
    """
    
    data['available_bike_shift'] = \
    data.groupby('station_id')['available_bike'].shift(1)

    data['available_bike_shift'] = data['available_bike_shift'].fillna(data['available_bike'])

    data['transactions_in'] = data['available_bike'] - data['available_bike_shift']

    data.loc[data['transactions_in'] < 0,
                 'transactions_in'] = 0
    
    # Drop non usefull column
    data.drop('available_bike_shift', axis=1, inplace=True)
    
    return data


def get_transactions_all(data):
    """
    Calcul le nombre de transactions de vélo (ajout et dépôt) qu'il y a eu pour une même
    station entre 2 points de données
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    
    Returns
    -------
    data : DataFrame
        Ajout de colonne 'transactions_all'
        
    Examples
    --------
    
    activite = get_transactions_all(activite)
    """
    
    data['available_bike_shift'] = \
    data.groupby('station_id')['available_bike'].shift(1)

    data['available_bike_shift'] = data['available_bike_shift'].fillna(data['available_bike'])

    data['transactions_all'] = np.abs(data['available_bike'] - data['available_bike_shift'])
    
    # Drop non usefull column
    data.drop('available_bike_shift', axis=1, inplace=True)
    
    return data


def get_consecutive_no_transactions_out(data):
    """
    Calcul depuis combien de temps la station n'a pas eu de prise de vélo. Plus le chiffre est haut, 
    plus ça fait longtemps que la station est inactive sur la prise de vélo.
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub avec la feature `transactions_out` (get_transactions_out)
    
    Returns
    -------
    data : DataFrame
        Ajout de colonne 'consecutive_no_transactions_out'
        
    Examples
    --------
    
    activite = get_consecutive_no_transactions_out(activite)
    """
    
    data['compteur'] = 1
    data['consecutive_no_transactions_out'] = \
        data.groupby([(data['transactions_out'] > 0).cumsum(),
                      'station_id'])['compteur'].cumsum()

    data['consecutive_no_transactions_out'] = \
        data['consecutive_no_transactions_out'].fillna(0)

    data = data.drop('compteur', axis=1)
    return data


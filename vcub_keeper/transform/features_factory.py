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
    
    data['available_stands_shift'] = \
    data.groupby('station_id')['available_stands'].shift(1)

    data['available_stands_shift'] = data['available_stands_shift'].fillna(data['available_stands'])

    data['transactions_out'] = data['available_stands'] - data['available_stands_shift']

    data.loc[data['transactions_out'] < 0,
                'transactions_out'] = 0
    
    # Drop non usefull column
    data.drop('available_stands_shift', axis=1, inplace=True)
    
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
    
    data['available_bikes_shift'] = \
    data.groupby('station_id')['available_bikes'].shift(1)

    data['available_bikes_shift'] = data['available_bikes_shift'].fillna(data['available_bikes'])

    data['transactions_in'] = data['available_bikes'] - data['available_bikes_shift']

    data.loc[data['transactions_in'] < 0,
                 'transactions_in'] = 0
    
    # Drop non usefull column
    data.drop('available_bikes_shift', axis=1, inplace=True)
    
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
    
    data['available_bikes_shift'] = \
    data.groupby('station_id')['available_bikes'].shift(1)

    data['available_bikes_shift'] = data['available_bikes_shift'].fillna(data['available_bikes'])

    data['transactions_all'] = np.abs(data['available_bikes'] - data['available_bikes_shift'])
    
    # Drop non usefull column
    data.drop('available_bikes_shift', axis=1, inplace=True)
    
    return data


def get_consecutive_no_transactions_out(data):
    """
    Calcul depuis combien de temps la station n'a pas eu de prise de vélo. Plus le chiffre est haut, 
    plus ça fait longtemps que la station est inactive sur la prise de vélo.
    
    Si il n'y a pas de données d'activité pour la station (absence de 'available_stands'), 
    alors consecutive_no_transactions_out = 0 et une fois qu'il y a  a nouveau de l'activité (des données)
    le compteur `consecutive_no_transactions_out` reprend
    
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

    data['have_data'] = 1
    data.loc[data['available_stands'].isna(),
             'have_data'] = 0

    data['consecutive_no_transactions_out'] = \
        data.groupby(['station_id',
                      (data['have_data'] == 0).cumsum(),
                      (data['transactions_out'] > 0).cumsum()]).cumcount()

    data['consecutive_no_transactions_out'] = \
        data['consecutive_no_transactions_out'].fillna(0)

    data.loc[data['available_stands'].isna(),
             'consecutive_no_transactions_out'] = 0

    data = data.drop('have_data', axis=1)
    
    return data


import pandas as pd 
import numpy as np


from vcub_keeper.reader.reader import read_meteo

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
    le compteur `consecutive_no_transactions_out` reprend.

    Cet indicateur à aussi besoin que la station soit connecté (status == 1) afin de que le compteur
    avance (sinon = 0)
    L'indicateur pour s'activer et continer à avancer dans le temps doit avoir des vélos disposibles en
    station (plus de 2 vélos, possibilité vélo HS, borne HS...)
    
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
                      (data['available_bikes'] < 3).cumsum(), # 3 for 2 available_bikes
                      (data['have_data'] == 0).cumsum(),
                      (data['status'] == 0).cumsum(),
                      (data['transactions_out'] > 0).cumsum()]).cumcount()

    data['consecutive_no_transactions_out'] = \
        data['consecutive_no_transactions_out'].fillna(0)

    data.loc[data['available_stands'].isna(),
             'consecutive_no_transactions_out'] = 0

    data = data.drop('have_data', axis=1)
    
    return data


def get_meteo(data):
    """
    AJoute les données météo suivantes : 
        - 'temperature' 
        - 'pressure'
        - 'humidity' (%)
        - 'pressure' (mb)
        - 'precipitation' (mm/h)
        - 'wind_speed' (m/s)

    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    
    Returns
    -------
    data : DataFrame
        Ajout des colonnes météo
        
    Examples
    --------
    
    ts_activity = get_meteo(ts_activity)
    """
    
    def fast_parse_date_(s):
        """
        This is an extremely fast approach to datetime parsing.
        For large data, the same dates are often repeated. Rather than
        re-parse these, we store all unique dates, parse them, and
        use a lookup to convert all dates.

        cf https://github.com/sanand0/benchmarks/tree/master/date-parse
        """
        dates = {date: date.strftime(format='%Y-%m-%d %H') for date in pd.Series(s.unique())}
        return s.apply(lambda v: dates[v])

    # Lecture des données météo
    meteo = read_meteo()
    
    # Creation des dates au format yyyy_mm avant jointure
    data['date_year_month_hours'] = fast_parse_date_(data['date'])
    meteo['date_year_month_hours'] = fast_parse_date_(meteo['date'])
    
    # Jointure
    data = data.merge(meteo.drop('date', axis=1),
                      on='date_year_month_hours', how='left')
    
    # On supprime la colonne 'date_year_month'
    data = data.drop('date_year_month_hours', axis=1)
    
    return data

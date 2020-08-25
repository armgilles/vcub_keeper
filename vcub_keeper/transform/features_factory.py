import pandas as pd 

import pandas as pd 

def get_transactions_depot(data):
    """
    Calcul le nombre de dépôt de vélo qu'il y a eu pour une même station entre 2 points de données
    
    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    
    Returns
    -------
    data : DataFrame
        Ajout de colonne 'transactions_depot'
        
    Examples
    --------
    
    activite = get_transactions_depot(activite)
    """
    
    data['available_stand_shift'] = \
    data.groupby('station_id')['available_stand'].shift(1)

    data['available_stand_shift'] = data['available_stand_shift'].fillna(data['available_stand'])

    data['transactions_depot'] = data['available_stand'] - data['available_stand_shift']

    data.loc[data['transactions_depot'] < 0,
                'transactions_depot'] = 0
    
    # Drop non usefull column
    data.drop('available_stand_shift', axis=1, inplace=True)
    
    return data
import pandas as pd


def read_stations_attributes(file_path = "../../data/bordeaux-stations.csv"):
    """
    Lecture du fichier sur les attributs des Vcub à Bordeaux
    Modification par rapport au fichier original : 
        - Changement de nom des colonnnes :
            - ident -> station_id
    
    Parameters
    ----------
    file_path : str
        Chemin d'accès au fichiers source
    
    Returns
    -------
    activite : DataFrame
        
    Examples
    --------
    
    stations = read_stations_attributes()
    """
    
    column_dtypes = {'ident': 'uint8',
                     'nom': 'string',
                    }


    stations = pd.read_csv(file_path, dtype = column_dtypes)
    
    # Renaming
    stations.rename(columns={'ident':'station_id'}, inplace=True)
    
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

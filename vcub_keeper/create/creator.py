import pandas as pd
import glob as glob

from vcub_keeper.config import ROOT_DATA_RAW, ROOT_DATA_CLEAN
from vcub_keeper.transform.features_factory import (get_transactions_in, get_transactions_out,
                                                    get_transactions_all)


def create_activity_time_series():
    """
    Création d'un fichier de type time series sur l'ensemble des stations VCUB
    grâce à la concenation de plusieurs fichier source (/data/raw/) dans
    /data/clean/time_serie_activity.h5

    Parameters
    ----------
    None
    
    Returns
    -------
    None
        
    Examples
    --------
    
    create_activity_time_series()
    """
    
    ## Lecture de tous les fichiers
    
    # Init DataFrame
    activite_full = pd.DataFrame()

    for file_path in glob.glob(ROOT_DATA_RAW+'bordeaux-*.csv'):
        file_name = file_path.split('/')[-1]
        print(file_name)

        # Lecture du fichier
        column_dtypes = {'id': 'uint8',
                         'status': 'category',
                         'available_stands': 'uint8',
                         'available_bikes': 'uint8'}
        activite_temp = pd.read_csv(ROOT_DATA_RAW + file_name,
                                    parse_dates=["timestamp"],
                                    dtype=column_dtypes)
        print(activite_temp.shape)

        # Concact
        activite_full = pd.concat([activite_full, activite_temp])

    ## Travail sur le fichier final (concatenation de l'ensemble des fichiers)

    # Status mapping
    status_dict = {'open' : 1,
                   'closed' : 0
                  }
    activite_full['status'] = activite_full['status'].map(status_dict)
    activite_full['status'] = activite_full['status'].astype('uint8')

    # Naming
    activite_full.rename(columns={'id':'station_id'}, inplace=True)
    activite_full.rename(columns={'timestamp':'date'}, inplace=True)

    # Sorting DataFrame on station_id & date
    activite_full = activite_full.sort_values(['station_id', 'date'], ascending=[1, 1])

    # Reset index
    activite_full = activite_full.reset_index(drop=True)
    
    # Dropduplicate station_id / date rows
    activite_full = activite_full.drop_duplicates(subset=['station_id', 'date']).reset_index(drop=True)
    
    # Create features
    activite_full = get_transactions_in(activite_full)
    activite_full = get_transactions_out(activite_full)
    activite_full = get_transactions_all(activite_full)
    
    ## Resampling
    
    # cf Bug Pandas : https://github.com/pandas-dev/pandas/issues/33548
    activite_full = activite_full.set_index('date')
    
    activite_full_resample = \
        activite_full.groupby('station_id').resample('10T', 
                                                     label='right',
                                                    ).agg({'available_stands' : 'last',
                                                           'available_bikes' : 'last',
                                                           'status' : 'min',
                                                           'transactions_in' : 'sum',
                                                           'transactions_out' : 'sum',
                                                           'transactions_all' : 'sum'}).reset_index()
    
    # Export
    activite_full_resample.to_hdf(ROOT_DATA_CLEAN + 'time_serie_activity.h5', key='ts_activity')
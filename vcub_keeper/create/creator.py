import pandas as pd
import glob as glob
from dotenv import load_dotenv
load_dotenv()

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
                                                           'status' : 'max', # Empeche les micro déconnection à la station
                                                           'transactions_in' : 'sum',
                                                           'transactions_out' : 'sum',
                                                           'transactions_all' : 'sum'}).reset_index()
    
    # Export
    activite_full_resample.to_hdf(ROOT_DATA_CLEAN + 'time_serie_activity.h5', key='ts_activity')


def create_meteo(min_date_history="2018-12-01", max_date_history='2020-09-18'):
    """
    Multiple call API afin de créer un fichier et d'exporte celui-ci
    dans ROOT_DATA_REF/meteo.csv
    
    Parameters
    ----------
    min_date_history : str
        date du début de l'historique (format 'yyyy-mm-dd')
    max_date_history : str
        date de fin de l'historique (format 'yyyy-mm-dd')
    
    Returns
    -------
    None
        
    Examples
    --------
    
    create_meteo(min_date_history="2018-12-01", max_date_history='2021-03-18')
    """
    api = Api(API_METEO)
    api.set_granularity('daily')
    
    # Init DataFrame
    meteo_full = pd.DataFrame()

    # Calls API
    date_list = pd.date_range(start=min_date_history, end=max_date_history)
    for date in date_list:
        # Date processing
        date_minus_one_day = date - datetime.timedelta(days=1)
        date_str = date.strftime(format='%Y-%m-%d')
        date_minus_one_day_str = date_minus_one_day.strftime(format='%Y-%m-%d')
        print(date_minus_one_day_str + " " + date_str)

        # Call API
        try:
            history = api.get_history(city="Bordeaux", country="FR",
                                      start_date=date_minus_one_day_str,
                                      end_date=date_str)
            meteo_day = pd.DataFrame(history.get_series(['max_temp', 'min_temp', 
                                                         'precip','temp','rh', 'pres']))
            meteo_full = pd.concat([meteo_full, meteo_day])
        except requests.HTTPError as exception:
            print(exception)

    # Naming DataFrame
    
    # Accumulated precipitation (default mm)
    meteo_full.rename(columns = {'precip':'precipitation'}, inplace = True)
    # Average temperature
    meteo_full.rename(columns = {'temp':'mean_teamp'}, inplace = True)
    # Average relative humidity (%)
    meteo_full.rename(columns = {'rh':'humidity_mean'}, inplace = True)
    # Average pressure (mb)
    meteo_full.rename(columns = {'pres':'pressure_mean'}, inplace = True)
    # date
    meteo_full.rename(columns = {'datetime':'date'}, inplace = True)

    meteo_full = meteo_full[['date', 'min_temp', 'mean_teamp', 'max_temp',
                             'pressure_mean', 'humidity_mean',
                             'precipitation']]
    
    # Check
    min_date = meteo_full.date.min()
    max_date = meteo_full.date.max()
    date_ref = pd.date_range(start=min_date, end=max_date, freq='d')

    # Si le référenciel n'a pas toutes les dates dans Timestamp
    assert date_ref.isin(meteo_full['date']).all() == True

    # Si il n'y a pas de différence symetrique entre les 2 séries de dates
    assert len(date_ref.symmetric_difference(meteo_full['date'])) == 0 

    # Si il y a des doublons
    assert meteo_full['date'].is_unique == True

    # Si les date augmentent
    assert meteo_full['date'].is_monotonic_increasing == True
    
    # export
    max_date_full = meteo_full.date.max().strftime('%Y-%m-%d')
    meteo_full.to_csv(ROOT_DATA_REF+'meteo/meteo_'+max_date_full+'.csv',
                     index=False)

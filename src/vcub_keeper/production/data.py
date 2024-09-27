import numpy as np
import pandas as pd
import polars as pl
import requests

from vcub_keeper.config import KEY_API_BDX
from vcub_keeper.transform.features_factory import get_transactions_all, get_transactions_in, get_transactions_out


#############################################
###            API Oslandia
#############################################
def get_data_from_api_by_station(station_id, start_date, stop_date):
    """
    Permet d'obtenir les données d'activité d'une station via une API d'Oslandia

    Parameters
    ----------
    station_id : Int or List
        Numéro de la station de Vcub
    start_date : str
        Date de début de la Time Serie
    stop_date : str
        Date de fin de la Time Serie

    Returns
    -------
    Time serie in Json format

    Examples
    --------

    station_json = get_data_from_api_by_station(station_id=19,
                                                start_date='2020-10-14',
                                                stop_date='2020-10-17')
    """

    if isinstance(station_id, list | np.ndarray):
        station_id = ",".join(map(str, station_id))

    url = (
        "http://data.oslandia.io/bikes/api/bordeaux/timeseries/station/"
        + str(station_id)
        + "?start="
        + start_date
        + "&stop="
        + stop_date
    )

    response = requests.get(url)  # noqa: S113
    return response.json()


def transform_json_station_data_to_df(station_json):
    """
    Tranforme la Time Serie d'activité d'une ou plusieurs station en DataFrame
    à partir de la fonction get_data_from_api_by_station()
    Effectue plusieurs transformation comme la fonction create/creator.py
    create_activity_time_series()
        - Structuration
        - Naming
        - Ajout de variables
        - Resampling sur 10min

    Parameters
    ----------
    station_json : json
        Time serie au format json de l'activité d'une station (ou plusieurs)
    Returns
    -------
    station_df_resample : DataFrame
        Time serie au format DataFrame de l'activité d'une ou plusieurs station
        resampler sur 10 min.

    Examples
    --------

    station_df = transform_json_station_data_to_df(station_json)

    """

    # Si il y a plusieurs stations dans le json
    if len(station_json["data"]) > 1:
        station_df = pd.DataFrame()
        for i in range(0, len(station_json["data"])):
            temp_station_df = pd.DataFrame(station_json["data"][i])
            station_df = pd.concat([station_df, temp_station_df])
    # Il y une seule station dans le json
    else:
        station_df = pd.DataFrame(station_json["data"][0])

    # Status mapping
    status_dict = {"open": 1, "closed": 0}
    station_df["status"] = station_df["status"].map(status_dict)
    station_df["status"] = station_df["status"].astype("uint8")

    # Naming
    station_df.rename(columns={"id": "station_id"}, inplace=True)
    station_df.rename(columns={"ts": "date"}, inplace=True)

    # Casting & sorting DataFrame on station_id & date
    station_df["date"] = pd.to_datetime(station_df["date"])
    station_df["station_id"] = station_df["station_id"].astype(int)
    station_df = station_df.sort_values(["station_id", "date"], ascending=[1, 1])

    # Dropduplicate station_id / date rows
    station_df = station_df.drop_duplicates(subset=["station_id", "date"]).reset_index(drop=True)

    # Create features
    station_df = get_transactions_in(station_df)
    station_df = get_transactions_out(pl.from_pandas(station_df), output_type="pandas")
    station_df = get_transactions_all(station_df)

    ## Resampling

    # cf Bug Pandas : https://github.com/pandas-dev/pandas/issues/33548
    station_df = station_df.set_index("date")

    station_df_resample = (
        station_df.groupby("station_id")
        .resample(
            "10min",
            label="right",
        )
        .agg(
            {
                "available_stands": "last",
                "available_bikes": "last",
                "status": "max",  # Empeche les micro déconnection à la station
                "transactions_in": "sum",
                "transactions_out": "sum",
                "transactions_all": "sum",
            }
        )
        .reset_index()
    )
    return station_df_resample


#############################################
###        API open data bordeaux
#############################################


def get_data_from_api_bdx_by_station(station_id, start_date, stop_date):
    """
    Permet d'obtenir les données d'activité d'une station via une API d'open data Bordeaux

    Parameters
    ----------
    station_id : Int or List
        Numéro de la station de Vcub
    start_date : str
        Date de début de la Time Serie
    stop_date : str
        Date de fin de la Time Serie

    Returns
    -------
    Time serie in Json format

    Examples
    --------
    station_json = get_data_from_api_bdx_by_station(station_id=19,
                                                    start_date='2020-10-14',
                                                    stop_date='2020-10-17')
    """

    # Si plusieurs station_id ([124,  15,  60,])
    if isinstance(station_id, list | np.ndarray):
        station_id = ",".join(map(str, station_id))

        url = (
            "https://data.bordeaux-metropole.fr/geojson/aggregate/ci_vcub_p?key="
            + KEY_API_BDX
            + "&rangeStart="
            + str(start_date)
            + '&filter={"ident":{"$in":['
            + str(station_id)
            + "]}}&rangeEnd="
            + str(stop_date)
            + '&rangeStep=5min&attributes={"nom": "mode", "etat": "mode", "nbplaces": "max", "nbvelos": "max", "ident": "min"}'
        )
    # Si une seul station_id
    else:
        url = (
            "https://data.bordeaux-metropole.fr/geojson/aggregate/ci_vcub_p?key="
            + KEY_API_BDX
            + "&rangeStart="
            + str(start_date)
            + '&filter={"ident":'
            + str(station_id)
            + "}&rangeEnd="
            + str(stop_date)
            + '&rangeStep=5min&attributes={"nom": "mode", "etat": "mode", "nbplaces": "max", "nbvelos": "max", "ident": "min"}'
        )

    response = requests.get(url)  # noqa: S113
    return response.json()


def transform_json_api_bdx_station_data_to_df(station_json):
    """
    Tranforme la Time Serie d'activité d'une ou plusieurs station en DataFrame
    à partir de la fonction get_data_from_api_bdx_by_station()
    Effectue plusieurs transformation comme la fonction create/creator.py
    create_activity_time_series()
        - Naming des colonnes json
        - Structuration
        - Naming
        - Ajout de variables
        - Resampling sur 10min

    Parameters
    ----------
    station_json : json
        Time serie au format json de l'activité d'une station (ou plusieurs)
    Returns
    -------
    station_df_resample : DataFrame
        Time serie au format DataFrame de l'activité d'une ou plusieurs station
        resampler sur 10 min.

    Examples
    --------

    station_df = transform_json_api_bdx_station_data_to_df(station_json)

    """

    station_df = pd.json_normalize(station_json, record_path=["features"])

    # Naming from JSON DataFrame
    station_df = station_df.rename(
        columns={
            "properties.time": "date",
            "properties.ident": "station_id",
            "properties.nom": "name",
            "properties.etat": "status",
            "properties.nbplaces": "available_stands",
            "properties.nbvelos": "available_bikes",
        }
    )

    # Status mapping
    status_dict = {"CONNECTEE": 1, "DECONNECTEE": 0}
    station_df["status"] = station_df["status"].map(status_dict).fillna(0)
    station_df["status"] = station_df["status"].astype("uint8")

    # Casting & sorting DataFrame on station_id & date
    station_df["date"] = pd.to_datetime(station_df["date"], utc=True)

    # Convert to Europe/Paris TZ
    try:
        station_df["date"] = station_df["date"].dt.tz_localize("Europe/Paris")
    except TypeError:  # try to convert TZ
        station_df["date"] = station_df["date"].dt.tz_convert("Europe/Paris")

    station_df["station_id"] = station_df["station_id"].astype(int)
    station_df = station_df.sort_values(["station_id", "date"], ascending=[1, 1])

    # Dropduplicate station_id / date rows
    station_df = station_df.drop_duplicates(subset=["station_id", "date"]).reset_index(drop=True)

    # Create features
    station_df = get_transactions_in(station_df)
    station_df = get_transactions_out(pl.from_pandas(station_df), output_type="pandas")
    station_df = get_transactions_all(station_df)

    ## Resampling

    # # cf Bug Pandas : https://github.com/pandas-dev/pandas/issues/33548
    station_df = station_df.set_index("date")

    station_df_resample = (
        station_df.groupby("station_id")
        .resample(
            "10min",
            label="right",
        )
        .agg(
            {
                "available_stands": "last",
                "available_bikes": "last",
                "status": "max",  # Empeche les micro déconnection à la station
                "transactions_in": "sum",
                "transactions_out": "sum",
                "transactions_all": "sum",
            }
        )
        .reset_index()
    )
    return station_df_resample

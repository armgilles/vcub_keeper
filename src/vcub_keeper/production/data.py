import numpy as np
import polars as pl
import requests

from vcub_keeper.config import KEY_API_BDX
from vcub_keeper.transform.features_factory import get_transactions_all, get_transactions_in, get_transactions_out


#############################################
###            API Oslandia
#############################################
def get_data_from_api_by_station(station_id: str | list, start_date: str, stop_date: str) -> dict:
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


def transform_json_station_data_to_df(station_json: dict) -> pl.LazyFrame:
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
    station_df_resample : LazyFrame
        Time serie au format DataFrame de l'activité d'une ou plusieurs station
        resampler sur 10 min.

    Examples
    --------

    station_df = transform_json_station_data_to_df(station_json)

    """

    # Si il y a plusieurs stations dans le json
    station_df = (
        pl.DataFrame(station_json["data"]).explode("available_bikes", "available_stands", "status", "ts").lazy()
    )

    # Status mapping
    status_dict = {"open": 1, "closed": 0}
    station_df = station_df.with_columns(status=pl.col("status").replace_strict(status_dict, default=0).cast(pl.UInt8))

    # Naming
    station_df = station_df.rename({"id": "station_id", "ts": "date"})

    # Casting & sorting DataFrame on station_id & date
    station_df = station_df.with_columns(station_id=pl.col("station_id").cast(pl.Int32()))
    station_df = station_df.with_columns(date=pl.col("date").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))

    station_df = station_df.unique(subset=["station_id", "date"])
    station_df = station_df.sort(["station_id", "date"], descending=[False, False])

    # Create features
    station_df = station_df.pipe(get_transactions_in).pipe(get_transactions_out).pipe(get_transactions_all)

    ## Resampling
    station_df_resample = (
        station_df.group_by_dynamic("date", group_by="station_id", every="10m", label="right")
        .agg(
            pl.col("available_stands").last(),
            pl.col("available_bikes").last(),
            pl.col("status").max(),  # Empeche les micro déconnection à la station
            pl.col("transactions_in").sum(),
            pl.col("transactions_out").sum(),
            pl.col("transactions_all").sum(),
        )
        .collect()
    )  # using collect cause upsample is not available in lazy mode https://github.com/armgilles/vcub_keeper/issues/152#issuecomment-2401744287
    station_df_resample = (
        station_df_resample.sort(["station_id", "date"], descending=[False, False])
        .upsample("date", every="10m", group_by="station_id")
        .lazy()
    )
    station_df_resample = station_df_resample.with_columns(station_id=pl.col.station_id.forward_fill())
    station_df_resample = station_df_resample.with_columns(
        transactions_in=pl.col.transactions_in.fill_null(0),
        transactions_out=pl.col.transactions_out.fill_null(0),
        transactions_all=pl.col.transactions_all.fill_null(0),
    )

    return station_df_resample


#############################################
###        API open data bordeaux
#############################################


def get_data_from_api_bdx_by_station(station_id: str | list, start_date: str, stop_date: str) -> dict:
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
    if response.status_code != 200:
        raise Exception(
            f"Erreur lors de la récupération des données depuis l'API Bordeaux Métropole: {url}\n{response.status_code}\n{response.text}"
        )

    return response.json()


def transform_json_api_bdx_station_data_to_df(station_json: dict) -> pl.LazyFrame:
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
    station_df_resample : LazyFrame
        Time serie au format DataFrame de l'activité d'une ou plusieurs station
        resampler sur 10 min.

    Examples
    --------

    station_df = transform_json_api_bdx_station_data_to_df(station_json)

    """

    station_df = pl.json_normalize(station_json["features"], max_level=1).lazy()

    # Naming from JSON DataFrame
    station_df = station_df.rename(
        mapping={
            "properties.time": "date",
            "properties.ident": "station_id",
            "properties.etat": "status",
            "properties.nbplaces": "available_stands",
            "properties.nbvelos": "available_bikes",
        }
    ).drop("type", "properties.gid", "properties.nom")  # drop unused columns

    # Status mapping
    status_dict = {"CONNECTEE": 1, "DECONNECTEE": 0, "MAINTENANCE": 0}
    station_df = station_df.with_columns(status=pl.col("status").replace_strict(status_dict, default=0).cast(pl.UInt8))

    station_df = station_df.with_columns(station_id=pl.col("station_id").cast(pl.Int32()))
    station_df = station_df.with_columns(
        # cast into datetime with tz_aware to Paris to none
        date=pl.col("date")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", time_zone="Europe/Paris")
        .dt.replace_time_zone(None),
    )

    station_df = station_df.unique(subset=["station_id", "date"])
    station_df = station_df.sort(["station_id", "date"], descending=[False, False])

    # Create features
    station_df = station_df.pipe(get_transactions_in).pipe(get_transactions_out).pipe(get_transactions_all)

    ## Resampling

    station_df_resample = station_df.group_by_dynamic("date", group_by="station_id", every="10m", label="right").agg(
        pl.col("available_stands").last(),
        pl.col("available_bikes").last(),
        pl.col("status").max(),  # Empeche les micro déconnection à la station
        pl.col("transactions_in").sum(),
        pl.col("transactions_out").sum(),
        pl.col("transactions_all").sum(),
    )
    return station_df_resample

import glob as glob
import io

import pandas as pd
import polars as pl
from dotenv import load_dotenv

from vcub_keeper.config import NON_USE_STATION_ID, ROOT_DATA_CLEAN, ROOT_DATA_RAW, ROOT_DATA_REF
from vcub_keeper.reader.reader import read_time_serie_activity
from vcub_keeper.reader.reader_utils import filter_periode
from vcub_keeper.transform.features_factory import (
    get_consecutive_no_transactions_out,
    get_transactions_all,
    get_transactions_in,
    get_transactions_out,
)

load_dotenv()


def create_activity_time_series() -> None:
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

    for file_path in glob.glob(ROOT_DATA_RAW + "bordeaux-*.csv"):
        file_name = file_path.split("/")[-1]
        print(file_name)

        # Lecture du fichier
        column_dtypes = {
            "id": "uint8",
            "status": "category",
            "available_stands": "int8",
            "available_bikes": "int8",
        }
        activite_temp = pd.read_csv(ROOT_DATA_RAW + file_name, parse_dates=["timestamp"], dtype=column_dtypes)
        print(activite_temp.shape)

        # Concact
        activite_full = pd.concat([activite_full, activite_temp])

    ## Travail sur le fichier final (concatenation de l'ensemble des fichiers)

    # Status mapping
    status_dict = {"open": 1, "closed": 0}
    activite_full["status"] = activite_full["status"].map(status_dict)
    activite_full["status"] = activite_full["status"].astype("uint8")

    # Naming
    activite_full.rename(columns={"id": "station_id"}, inplace=True)
    activite_full.rename(columns={"timestamp": "date"}, inplace=True)

    # Sorting DataFrame on station_id & date
    activite_full = activite_full.sort_values(["station_id", "date"], ascending=[1, 1])

    # Reset index
    activite_full = activite_full.reset_index(drop=True)

    # Dropduplicate station_id / date rows
    activite_full = activite_full.drop_duplicates(subset=["station_id", "date"]).reset_index(drop=True)

    # Create features (with polars)
    activite_full = get_transactions_in(pl.from_pandas(activite_full))
    activite_full = get_transactions_out(activite_full)
    activite_full = get_transactions_all(activite_full)

    ## Resampling
    activite_full_resample = activite_full.group_by_dynamic(
        "date", group_by="station_id", every="10m", label="right"
    ).agg(
        pl.col("available_stands").last(),
        pl.col("available_bikes").last(),
        pl.col("status").max(),  # Empeche les micro déconnection à la station
        pl.col("transactions_in").sum(),
        pl.col("transactions_out").sum(),
        pl.col("transactions_all").sum(),
    )

    # Export
    activite_full_resample.write_parquet(ROOT_DATA_CLEAN + "time_serie_activity.parquet")


# def create_meteo(min_date_history="2018-12-01", max_date_history="2020-09-18"):
#     """
#     Multiple call API afin de créer un fichier et d'exporte celui-ci
#     dans ROOT_DATA_REF/meteo.csv

#     Parameters
#     ----------
#     min_date_history : str
#         date du début de l'historique (format 'yyyy-mm-dd')
#     max_date_history : str
#         date de fin de l'historique (format 'yyyy-mm-dd')

#     Returns
#     -------
#     None

#     Examples
#     --------

#     create_meteo(min_date_history="2018-12-01", max_date_history='2021-03-18')
#     """
#     api = Api(API_METEO)
#     api.set_granularity("hourly")

#     # Init DataFrame
#     meteo_full = pd.DataFrame()

#     # Calls API
#     date_list = pd.date_range(start=min_date_history, end=max_date_history)
#     for date in date_list:
#         # Date processing
#         date_minus_one_day = date - datetime.timedelta(days=1)
#         date_str = date.strftime(format="%Y-%m-%d")
#         date_minus_one_day_str = date_minus_one_day.strftime(format="%Y-%m-%d")
#         print(date_minus_one_day_str + " " + date_str)

#         # Call API
#         try:
#             history = api.get_history(
#                 city="Bordeaux",
#                 country="FR",
#                 start_date=date_minus_one_day_str,
#                 end_date=date_str,
#             )
#             meteo_day = pd.DataFrame(history.get_series(["temp", "precip", "rh", "pres", "wind_spd"]))
#             meteo_full = pd.concat([meteo_full, meteo_day])
#         except requests.HTTPError as exception:
#             print(exception)

#     # Naming DataFrame

#     # Accumulated precipitation (default mm)
#     meteo_full.rename(columns={"precip": "precipitation"}, inplace=True)
#     # Average temperature
#     meteo_full.rename(columns={"temp": "temperature"}, inplace=True)
#     # Average relative humidity (%)
#     meteo_full.rename(columns={"rh": "humidity"}, inplace=True)
#     # Average pressure (mb)
#     meteo_full.rename(columns={"pres": "pressure"}, inplace=True)
#     # Wind_speed (m/s)
#     meteo_full.rename(columns={"wind_spd": "wind_speed"}, inplace=True)
#     # date
#     meteo_full.rename(columns={"datetime": "date"}, inplace=True)

#     meteo_full = meteo_full[["date", "temperature", "pressure", "humidity", "precipitation", "wind_speed"]]

#     # Check
#     min_date = meteo_full.date.min()
#     max_date = meteo_full.date.max()
#     date_ref = pd.date_range(start=min_date, end=max_date, freq="h")

#     # Si le référenciel n'a pas toutes les dates dans Timestamp
#     assert date_ref.isin(meteo_full["date"]).all() == True

#     # Si il n'y a pas de différence symetrique entre les 2 séries de dates
#     assert len(date_ref.symmetric_difference(meteo_full["date"])) == 0

#     # Si il y a des doublons
#     assert meteo_full["date"].is_unique == True

#     # Si les date augmentent
#     assert meteo_full["date"].is_monotonic_increasing == True

#     # export

#     meteo_full.to_csv(ROOT_DATA_REF + "meteo.csv", index=False)


def create_station_profilage_activity() -> None:
    """
    Création d'un fichier classifiant les stations suivant leurs activités et
    leurs fréquences d'utilation (données filtré par reader_utils.py filter_periode() )
    Création du fichier `station_profile.csv` dans ROOT_DATA_REF

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------

    create_station_profilage_activity()
    """

    # Lecture du fichier activité
    ts_activity = pl.from_pandas(read_time_serie_activity(path_directory=ROOT_DATA_CLEAN))

    # Some features
    ts_activity = get_transactions_in(ts_activity)
    ts_activity = get_transactions_out(ts_activity)
    ts_activity = get_transactions_all(ts_activity)
    ts_activity = get_consecutive_no_transactions_out(ts_activity)

    # Filter data with confinement & non use by consumer
    ts_activity = filter_periode(ts_activity, non_use_station_id=NON_USE_STATION_ID)

    # On regarde si il y a eu une prise de vélo ou non toutes les 10 min
    ts_activity = ts_activity.with_columns(transactions_out_bool=pl.col("transactions_out").clip(0, 1))

    # Aggrégation de l'activité par stations
    profile_station = (
        ts_activity.filter((pl.col("status") == 1) & (pl.col("consecutive_no_transactions_out") <= 144))
        .group_by("station_id")
        .agg(
            pl.col("transactions_out_bool").count().alias("total_point"),
            pl.col("transactions_out_bool").mean().alias("mean"),
            pl.col("transactions_out_bool").median().alias("median"),
            pl.col("transactions_out_bool").std().alias("std"),
            pl.col("transactions_out_bool").quantile(0.95).alias("95%"),
            pl.col("transactions_out_bool").quantile(0.98).alias("98%"),
            pl.col("transactions_out_bool").quantile(0.99).alias("99%"),
            pl.col("transactions_out_bool").max().alias("max"),
        )
    )
    profile_station = profile_station.sort("mean")

    # Classification en 3 activités (low / medium / hight, etc...)
    # Pour retrouver les seuils de cut
    # Je ne retrouve pas les mêmes résulats avec
    # breakpoints = np.quantile(profile_station.select("mean"), np.linspace(0, 1, n_bins + 1), method="linear")
    df_pd = profile_station.to_pandas()
    df_pd["profile_station_activity"], breakpoints = pd.cut(
        df_pd["mean"], 4, labels=["low", "medium", "hight", "very high"], retbins=True
    )
    del df_pd

    # Question about cut & labels in polars
    # https://stackoverflow.com/questions/79059121/using-polars-cut-with-label-from-a-pandas-perspective
    profile_station = profile_station.with_columns(
        profile_station_activity=pl.col("mean").cut(
            breaks=breakpoints, labels=["nothing_low", "low", "medium", "hight", "very high", "nothing_high"]
        )
    )

    ## Export
    profile_station.write_csv(ROOT_DATA_REF + "station_profile.csv")


def create_station_attribute(
    path_directory: str, data: None | io.StringIO = None, export: bool = True
) -> None | pl.DataFrame:
    """
    Création du fichier de réeérence des attributs des stations Vcub de l'agglomération de Bordeaux
    suite à la modification de l'accès open-data des données précédentes
    (https://github.com/armgilles/vcub_keeper/issues/50).
    Export le fichier dans "path_directory" en .csv.

    Parameters
    ---------
    path_directory : str
        chemin d'accès (ROOT_DATA_REF)

    data : None | str
        If None, use URL_DATA_STATION, else custum data

    export : bool (default=True)
        To export csv file

    Returns
    -------
    None | DataFrame (if export=False)

    Examples
    --------
    create_station_attribute(path_directory=ROOT_DATA_REF)
    """
    if data is None:
        # URL
        URL_DATA_STATION = "https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"
        data = URL_DATA_STATION

    column_dtypes = {"IDENT": pl.UInt16}
    usecols = ["Geo Point", "Geo Shape", "commune", "IDENT", "TYPE", "NOM", "NBPLACES", "NBVELOS"]

    stations = pl.read_csv(data, separator=";", schema_overrides=column_dtypes, columns=usecols)

    stations = stations.with_columns(total_stand=pl.col("NBPLACES") + pl.col("NBVELOS"))
    # Create lon / lat
    stations = stations.with_columns(
        pl.col("Geo Point")
        .str.split_exact(by=",", n=1)
        .struct.rename_fields(["lat", "lon"])
        .cast(pl.Float32)
        .alias("fields")
    ).unnest("fields")

    # Naming
    stations = stations.rename({"commune": "COMMUNE", "IDENT": "station_id", "TYPE": "TYPEA"})

    # Filter
    col_to_export = ["Geo Point", "Geo Shape", "COMMUNE", "total_stand", "NOM", "TYPEA", "station_id", "lat", "lon"]

    stations = stations.select(col_to_export)

    if export:
        # Export
        file_export = "station_attribute.csv"
        stations.write_csv(path_directory + file_export, separator=";")
    else:
        return stations

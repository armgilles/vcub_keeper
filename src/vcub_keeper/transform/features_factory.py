import numpy as np
import pandas as pd
import polars as pl


def get_transactions_out(data: pl.DataFrame, output_type=None) -> pl.DataFrame | pd.DataFrame:
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

    data = data.with_columns(pl.col("available_stands").shift(1).over("station_id").alias("available_stands_shift"))
    data = data.with_columns(pl.col("available_stands_shift").fill_null(pl.col("available_stands")))
    data = data.with_columns(transactions_out=(pl.col("available_stands") - pl.col("available_stands_shift")))

    data = data.with_columns(
        transactions_out=pl.when(pl.col("transactions_out") < 0).then(0).otherwise(pl.col("transactions_out"))
    )

    # Drop non usefull column
    data = data.drop("available_stands_shift")

    if output_type == "pandas":
        data = data.to_pandas()

    return data


def get_transactions_in(data: pl.DataFrame, output_type=None) -> pl.DataFrame | pd.DataFrame:
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

    data = data.with_columns(pl.col("available_bikes").shift(1).over("station_id").alias("available_bikes_shift"))
    data = data.with_columns(pl.col("available_bikes_shift").fill_null(pl.col("available_bikes")))
    data = data.with_columns(transactions_in=(pl.col("available_bikes") - pl.col("available_bikes_shift")))

    data = data.with_columns(
        transactions_in=pl.when(pl.col("transactions_in") < 0).then(0).otherwise(pl.col("transactions_in"))
    )

    # Drop non usefull column
    data = data.drop("available_bikes_shift")

    if output_type == "pandas":
        data = data.to_pandas()

    return data


def get_transactions_all(data: pl.DataFrame, output_type=None) -> pl.DataFrame | pd.DataFrame:
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

    data = data.with_columns(pl.col("available_bikes").shift(1).over("station_id").alias("available_bikes_shift"))
    data = data.with_columns(pl.col("available_bikes_shift").fill_null(pl.col("available_bikes")))
    data = data.with_columns(transactions_all=(pl.col("available_bikes") - pl.col("available_bikes_shift")).abs())

    # Drop non usefull column
    data = data.drop("available_bikes_shift")

    if output_type == "pandas":
        data = data.to_pandas()

    return data


def get_consecutive_no_transactions_out(data: pl.DataFrame) -> pl.DataFrame:
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

    data = data.with_columns(
        pl.when(
            (pl.col("transactions_out") >= 1)
            | (pl.col("status") == 0)
            | (pl.col("available_stands") <= 2)
            | (pl.col("available_stands").is_null())
        )
        .then(0)
        .otherwise(1)
        .alias("logic")
    )

    data = data.with_columns(
        pl.col("logic")
        .cum_sum()
        .over("station_id", pl.col("logic").rle_id(), pl.col("logic") == 0)
        .cast(pl.Int64)
        .alias("consecutive_no_transactions_out")
    ).drop("logic")

    return data


# https://github.com/armgilles/vcub_keeper/issues/42#issuecomment-718848126
def get_meteo(data, meteo):
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
     meteo : DataFrame
         Données météo

     Returns
     -------
     data : DataFrame
        Ajout des colonnes météo

    Examples
    --------

    ts_activity = get_meteo(data=ts_activity, meteo=meteo)
    """

    def fast_parse_date_(s):
        """
        This is an extremely fast approach to datetime parsing.
        For large data, the same dates are often repeated. Rather than
        re-parse these, we store all unique dates, parse them, and
        use a lookup to convert all dates.

        cf https://github.com/sanand0/benchmarks/tree/master/date-parse
        """
        dates = {date: date.strftime(format="%Y-%m-%d %H") for date in pd.Series(s.unique())}
        return s.apply(lambda v: dates[v])

    # Creation des dates au format yyyy_mm avant jointure
    data["date_year_month_hours"] = fast_parse_date_(data["date"])
    meteo["date_year_month_hours"] = fast_parse_date_(meteo["date"])

    # Jointure
    data = data.merge(meteo.drop("date", axis=1), on="date_year_month_hours", how="left")

    # On supprime la colonne 'date_year_month'
    data = data.drop("date_year_month_hours", axis=1)

    return data


def get_encoding_time(data: pl.DataFrame, col_date: str, max_val: int) -> pl.DataFrame:
    """
    Encoding time

    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub
    col_date : str
        Nom de la colonne à encoder
    max_val : int
        Valeur maximal que la valeur peut avoir (ex 12 pour le mois)

    Returns
    -------
    data : DataFrame
        Ajout de colonne Sin_[col_date] & Cos_[col_date]

    Examples
    --------
    data = get_encoding_time(data, 'month', max_val=12)
    """

    two_pi = 2 * np.pi
    expr_two_pi_div_max_val = pl.lit(two_pi / max_val)
    data = data.with_columns(
        [
            (expr_two_pi_div_max_val * pl.col(col_date)).sin().alias("Sin_" + col_date),
            (expr_two_pi_div_max_val * pl.col(col_date)).cos().alias("Cos_" + col_date),
        ]
    )
    return data


def process_data_cluster(data: pl.DataFrame) -> pl.DataFrame:
    """
    Process some Feature engineering

    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub

    Returns
    -------
    data : DataFrame
        Add some columns in DataFrame

    Examples
    --------
    data = process_data_cluster(data)
    """

    data = data.with_columns(
        [
            pl.col("date").dt.quarter().alias("quarter"),
            # pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.hour().alias("hours"),
        ]
    )

    data = get_encoding_time(data, "quarter", max_val=4)
    # data = get_encoding_time(data, 'month', max_val=12)
    data = get_encoding_time(data, "weekday", max_val=7)
    data = get_encoding_time(data, "hours", max_val=24)

    return data

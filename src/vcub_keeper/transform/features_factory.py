import numpy as np
import pandas as pd
import polars as pl


def get_transactions_out() -> pl.Expr:
    """
    Returns a  Polars expressions to calculate the number of bike check-out transactions for a station.

    Returns
    -------
    pl.Expr
        List of expressions to add a 'transactions_out' column.

    Examples
    --------
    activite.with_columns(get_transactions_out())
    """
    available_stands_shift = (
        pl.col("available_stands").shift(1).over("station_id").fill_null(pl.col("available_stands"))
    )
    transactions_out = pl.col("available_stands") - available_stands_shift

    return pl.when(transactions_out < 0).then(0).otherwise(transactions_out).alias("transactions_out")


def get_transactions_in() -> pl.Expr:
    """
    Returns a  Polars expressions to calculate the number of bike check-out transactions for a station.

    Returns
    -------
    pl.Expr
        List of expressions to add a 'transactions_in' column.

    Examples
    --------
    activite.with_columns(get_transactions_in())
    """

    available_bikes_shift = pl.col("available_bikes").shift(1).over("station_id").fill_null(pl.col("available_bikes"))
    transactions_in = pl.col("available_bikes") - available_bikes_shift

    return pl.when(transactions_in < 0).then(0).otherwise(transactions_in).alias("transactions_in")


def get_transactions_all() -> pl.Expr:
    """
    Returns a  Polars expressions to calculate the number of bike check-out transactions for a station.

    Returns
    -------
    pl.Expr
        List of expressions to add a 'transactions_all' column.

    Examples
    --------
    activite.with_columns(get_transactions_all())
    """

    available_bikes_shift = pl.col("available_bikes").shift(1).over("station_id").fill_null(pl.col("available_bikes"))
    transactions_all = (pl.col("available_bikes") - available_bikes_shift).abs()

    return pl.when(transactions_all < 0).then(0).otherwise(transactions_all).alias("transactions_all")


def get_consecutive_no_transactions_out() -> pl.Expr:
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

    Returns
    -------
    pl.Expr
        List of expressions to add a 'transactions_all' column.

    Examples
    --------
    activite.with_columns(get_consecutive_no_transactions_out())
    """

    logic = (
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

    consecutive_no_transactions_out = (
        pl.int_ranges(pl.struct("station_id", logic).rle().struct.field("len"))
        .flatten()
        .alias("consecutive_no_transactions_out")
        + 1
    )
    consecutive_no_transactions_out_cond = (
        pl.when(logic == 1).then(consecutive_no_transactions_out).otherwise(0).alias("consecutive_no_transactions_out")
    )

    return consecutive_no_transactions_out_cond


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


def get_encoding_time(col_date: str, max_val: int) -> list[pl.Expr]:
    """
    Encoding time

    Parameters
    ----------

    col_date : str
        Nom de la colonne à encoder
    max_val : int
        Valeur maximal que la valeur peut avoir (ex 12 pour le mois)

    Returns
    -------
    list[pl.Expr]

    Example
    --------
    encoding_quarter_expr = get_encoding_time("quarter", max_val=4)
    data.with_columns(*encoding_quarter_expr)
    """

    two_pi = 2 * np.pi
    expr_two_pi_div_max_val = pl.lit(two_pi / max_val)

    # Création des expressions pour Sin et Cos
    sin_expr = (expr_two_pi_div_max_val * pl.col(col_date)).sin().alias(f"Sin_{col_date}")
    cos_expr = (expr_two_pi_div_max_val * pl.col(col_date)).cos().alias(f"Cos_{col_date}")

    return [sin_expr, cos_expr]


def process_data_cluster(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process some Feature engineering

    Parameters
    ----------
    data : DataFrame
        Activité des stations Vcub

    Returns
    -------
    data : LazyFrame
        Add some columns in DataFrame

    Examples
    --------
    data = process_data_cluster(data)
    """

    data = data.with_columns(
        [
            pl.col("date").dt.quarter().alias("quarter"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.hour().alias("hours"),
        ]
    )

    encoding_quarter_expr = get_encoding_time("quarter", max_val=4)
    encoding_weekday_expr = get_encoding_time("weekday", max_val=7)
    encoding_hours_expr = get_encoding_time("hours", max_val=24)
    data = data.with_columns(*encoding_quarter_expr, *encoding_weekday_expr, *encoding_hours_expr)

    return data

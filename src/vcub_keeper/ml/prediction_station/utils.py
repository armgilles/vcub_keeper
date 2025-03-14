import polars as pl


def create_target(station_to_pred: pl.LazyFrame, target_col: str, horizon_prediction: str) -> pl.LazyFrame:
    """
    Permets de créer la target avant le learniong du modèle suivant une colonne cible
    et un horizon de prédiction (offset) donné.
    available units for horizon_prediction are: 'y', 'mo', 'q', 'w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns'"

    Parameters
    ----------
    station_to_pred: pl.LazyFrame
        Activité des stations Vcub

    target_col: str
        La colonne à prédire "available_stands" ou "available_bikes"

    horizon_prediction: str
        L'horizon de la prédiction (offset) : "30m" ou "1h"

    Returns
    -------
    station_to_pred: pl.LazyFrame
        Avec la colonne "target" en plus

    Example
    -------
    station_to_pred = create_target(station_to_pred, target_col="available_stands", horizon_prediction="30m")
    """

    # Pour la création de target du model
    station_to_pred = station_to_pred.with_columns(date_futur=pl.col("date").dt.offset_by(horizon_prediction))

    # Create target col
    station_to_pred = station_to_pred.join(
        station_to_pred.select(pl.col("date"), pl.col(target_col).alias("target")),
        left_on="date_futur",
        right_on="date",
        how="left",
    ).drop("date_futur")

    return station_to_pred

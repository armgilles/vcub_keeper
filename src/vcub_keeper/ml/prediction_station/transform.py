import polars as pl

from vcub_keeper.transform.features_factory import get_encoding_time


def build_feat_for_regression(station_to_pred: pl.LazyFrame, target_col: str) -> pl.DataFrame:
    """
    Permets de créer les features pour le modèle de regression en amont du training
    Fonction méta qui reprend les fonctions process_temporal_feat() et build_lag_and_rolling_feat()
    Collect() le LazyFrame

    Parameters
    ----------
    station_to_pred : pl.LazyFrame
        Activité des stations Vcub
    target_col : str
        La colonne à prédire "available_stands" ou "available_bikes"

    Returns
    -------
    station_to_pred : pl.DataFrame
        Avec les features de type lag et rolling en plus

    Example
    -------
    station_to_pred = build_feat_for_regression(station_to_pred, target_col="available_stands")
    """

    station_to_pred = (
        station_to_pred.pipe(process_temporal_feat).pipe(build_lag_and_rolling_feat, target_col=target_col)
    ).collect()

    return station_to_pred


def process_temporal_feat(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process some Feature engineering for regression

    Parameters
    ----------
    data : LazyFrame
        Activité des stations Vcub

    Returns
    -------
    data : LazyFrame
        Add some columns in DataFrame

    Examples
    --------
    data = process_temporal_feat(data)
    """

    data = data.with_columns(
        [
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.hour().alias("hours"),
            pl.col("date").dt.minute().alias("minutes"),
        ]
    )

    data = get_encoding_time(data, "weekday", max_val=7)
    data = get_encoding_time(data, "hours", max_val=24)
    data = get_encoding_time(data, "minutes", max_val=6)  # 10 min

    return data


def build_lag_and_rolling_feat(station_to_pred: pl.LazyFrame, target_col: str) -> pl.LazyFrame:
    """
    Permets la création de feature de type lag et rolling (max/min)
    pour le modèle de regression

    Parameters
    ----------
    station_to_pred : pl.lazyframe
        df contenant les données de la station à prédire de l'app

    target_col : str
        Nom de la colonne cible à prédire

    Returns
    -------
    pl.lazyframe

    Examples
    --------
    station_to_pred = build_lag_and_rolling_feat(station_to_pred, target_col="available_stands")
    """

    # créer feat temporelles de min et max
    station_to_pred = station_to_pred.with_columns(
        [
            # lag
            pl.col(target_col).shift(1).alias(f"{target_col}_lag_1"),
            pl.col(target_col).shift(2).alias(f"{target_col}_lag_2"),
            pl.col(target_col).shift(3).alias(f"{target_col}_lag_3"),
            # rolling max
            pl.col(target_col).rolling_max(6).alias(f"{target_col}_rolling_max_6"),
            pl.col(target_col).rolling_max(12).alias(f"{target_col}_rolling_max_12"),
            pl.col(target_col).rolling_max(144).alias(f"{target_col}_rolling_max_1d"),
            pl.col(target_col).rolling_max(1008).alias(f"{target_col}_rolling_max_7d"),
            # rolling min
            pl.col(target_col).rolling_min(6).alias(f"{target_col}_rolling_min_6"),
            pl.col(target_col).rolling_min(12).alias(f"{target_col}_rolling_min_12"),
            pl.col(target_col).rolling_min(144).alias(f"{target_col}_rolling_min_1d"),
            pl.col(target_col).rolling_min(1008).alias(f"{target_col}_rolling_min_7d"),
        ]
    )

    return station_to_pred

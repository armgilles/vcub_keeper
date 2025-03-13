import polars as pl
from sklearn.ensemble import RandomForestRegressor

from vcub_keeper.ml.prediction_station.transform import process_temporal_feat


def make_prediction_for_user(
    station_to_pred: pl.lazyframe,
    horizon_prediction: str,
    model: RandomForestRegressor,
    feat_to_use: list[str],
    return_df: bool = True,
) -> pl.LazyFrame | int:
    """
    Permets de réaliser la prédiciton pour l'utilisateur. Par exemple, Combien il y aura
    de vélos disponibles dans 30 minutes pour la station Place de la Bourse.

    Parameters
    ----------
    station_to_pred : pl.lazyframe
        Activité des stations Vcub déjà traitée par build_lag_and_rolling_feat()

    horizon_prediction : str
        Horizon de la prédiction (offset)

    model : RandomForestRegressor
        Modèle de prédiction entrainé

    feat_to_use : list[str]
        Liste des features à utiliser pour la prédiction

    Returns
    -------
    prediction : int
        Par exemple, prediction = 13

    Example
    -------
    prediction = make_prediction_for_user(station_to_pred, horizon_prediction="30m", model=model, feat_to_use=feat_to_use)
    """

    # On prend la date la plus récente du dataset
    pred = station_to_pred.filter(pl.col("date") == station_to_pred.get_column("date").max())

    # On créer la date d'horizon
    pred = pred.with_columns(pl.col("date").dt.offset_by(horizon_prediction).alias("date"))

    # feat temporelles with new date
    pred = process_temporal_feat(pred)

    # prédiction
    pred = pred.with_columns(y_pred=pl.lit(model.predict(pred.select(feat_to_use))))
    pred = pred.with_columns(pl.col("y_pred").round(0).cast(pl.Int32))

    if return_df:
        return pred
    else:
        prediction = pred.tail(1).select(pl.col("y_pred")).item()
        return prediction

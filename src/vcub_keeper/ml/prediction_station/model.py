import polars as pl
from sklearn.ensemble import RandomForestRegressor

from vcub_keeper.config import SEED


def train_model_for_station(
    station_to_pred: pl.DataFrame,
    horizon_prediction: str,
    feat_to_use: list[str],
) -> RandomForestRegressor:
    """
    Train the model with the given data

    Parameters
    ----------
    station_to_pred : pl.DataFrame
        Data of the station to predict
    horizon_prediction : str
        The horizon of the prediction (offset) : "30m" or "1h"

    Returns
    -------
    model : RandomForestRegressor
        Trained model

    Example
    -------
    model = train_model_for_station(station_to_pred, horizon_prediction=horizon_prediction, feat_to_use=feat_to_use)
    """

    x_train = station_to_pred.filter(pl.col("target").is_not_null()).select(feat_to_use)
    y_train = station_to_pred.filter(pl.col("target").is_not_null()).get_column("target")

    # Model
    model = RandomForestRegressor(max_depth=6, random_state=SEED, n_jobs=-1, n_estimators=50, max_features=0.75)
    model.fit(x_train, y_train)

    return model

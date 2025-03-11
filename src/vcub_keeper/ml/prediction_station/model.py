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


def get_feature_to_use_for_model(
    target_col: str,
) -> list[str]:
    """
    Get the features to use for the model

    Parameters
    ----------
    station_to_pred : pl.DataFrame
        Data of the station to predict

    Returns
    -------
    feat_to_use : list[str]
        List of features to use for the model

    Example
    -------
    feat_to_use = get_feature_to_use_for_model(target_col="available_stands")
    """

    feat_to_use = [
        "Sin_weekday",
        "Cos_weekday",
        "Sin_hours",
        "Cos_hours",
        "Sin_minutes",
        "Cos_minutes",
        # lag
        f"{target_col}_lag_1",
        f"{target_col}_lag_2",
        f"{target_col}_lag_3",
        # rolling min / max
        f"{target_col}_rolling_max_6",
        f"{target_col}_rolling_max_12",
        f"{target_col}_rolling_max_1d",
        f"{target_col}_rolling_max_7d",
        f"{target_col}_rolling_min_6",
        f"{target_col}_rolling_min_12",
        f"{target_col}_rolling_min_1d",
        f"{target_col}_rolling_min_7d",
    ]

    return feat_to_use

import pandas as pd
import polars as pl
import pytest
from vcub_keeper.production.data import get_data_from_api_by_station, transform_json_station_data_to_df
from vcub_keeper.transform.features_factory import get_consecutive_no_transactions_out, process_data_cluster
from vcub_keeper.ml.cluster import train_cluster_station, predict_anomalies_station, logistic_predict_proba_from_model
from vcub_keeper.config import FEATURES_TO_USE_CLUSTER


# test data to check result from train clf
# Minimal information about station activity to predict anomaly with a algo already fitted
test_data = [
    (
        [{"station_id": 106, "date": pd.Timestamp("2018-12-01 00:10:00"), "consecutive_no_transactions_out": 0}],
        1,
    ),  # Should be OK
    (
        [{"station_id": 106, "date": pd.Timestamp("2020-08-25 03:50:00"), "consecutive_no_transactions_out": 58}],
        -1,
    ),  # Should be KO (anomaly)
]


@pytest.mark.parametrize("data_activity, anomaly", test_data)
def test_ml_train_on_one_station(data_activity, anomaly):
    """
    On test le learning de l'algo sur une station et ses prédictions.
    """

    station_id = 106
    start_date = "2018-12-01"
    stop_date = "2020-08-28"
    profile_station_activity = "very high"  # https://github.com/armgilles/vcub_keeper/issues/56#issuecomment-1007612158

    station_json = get_data_from_api_by_station(station_id=station_id, start_date=start_date, stop_date=stop_date)

    station_df = transform_json_station_data_to_df(station_json)

    # Create feature basé sur l'absence consécutive de prise de vcub sur la station
    station_df = get_consecutive_no_transactions_out(station_df).collect().to_pandas()

    clf = train_cluster_station(station_df, station_id=station_id, profile_station_activity=profile_station_activity)

    # Prediction
    station_df_pred = predict_anomalies_station(data=station_df, clf=clf, station_id=station_id)

    # Check prediction sanity
    # Check features creation is the same as FEATURES_TO_USE_CLUSTER (from config.py)
    assert (station_df_pred[FEATURES_TO_USE_CLUSTER].columns == FEATURES_TO_USE_CLUSTER).any()

    # Check prediction
    # transform test data into DataFrame
    data_activity_df = pd.DataFrame((data_activity))
    assert (
        predict_anomalies_station(data=data_activity_df, clf=clf, station_id=station_id)["anomaly"].squeeze() == anomaly
    )

    # Score anomaly
    # Have to build features to before to calcul anomaly score
    data_activity_df_build = process_data_cluster(pl.from_pandas(data_activity_df)).to_pandas()
    score_anomaly = (
        logistic_predict_proba_from_model(clf.decision_function(data_activity_df_build[FEATURES_TO_USE_CLUSTER])) * 100
    )[0].squeeze()
    print(score_anomaly)

    if anomaly == 1:  # Station OK
        # anomaly_score must be at =~ 14.13 (2022/01/26)
        assert 10 <= score_anomaly <= 26
    elif anomaly == -1:  # Station KO
        # anomaly_score must be at =~ 54.33 (2022/01/26)
        # anomaly_score must be at =~ 53.82 with consecutive_no_transaction_out at 58 (2024/09/17)
        assert 50 <= score_anomaly <= 65

import pandas as pd
import json

from vcub_keeper.production.data import (
    get_data_from_api_bdx_by_station,
    transform_json_api_bdx_station_data_to_df,
)
from vcub_keeper.config import ROOT_TESTS_DATA


def test_get_api_bdx_data():
    """
    Appel l'api de bdx et compare les résultats avec les données de
    test "vcub_keeper/tests/data_for_tests/data_test_api_from_bdx.json"

    Notebook of this test : notebooks/04_tests/01_test_api_from_bdx.ipynb
    """

    station_id = "106"
    start_date = "2023-03-10"
    stop_date = "2023-03-11"

    station_json = get_data_from_api_bdx_by_station(
        station_id=station_id, start_date=start_date, stop_date=stop_date
    )

    # Loading data from data test
    with open(ROOT_TESTS_DATA + "data_test_api_from_bdx.json") as f:
        station_json_loaded = json.load(f)

    assert station_json == station_json_loaded


def test_get_api_bdx_data_to_df_station_id_80():
    """
    Appel l'api de bdx et vérifie qu'il y a bien qu'un seul station_id
    qui est retourné

    Cf : https://github.com/armgilles/vcub_keeper/issues/80
    """

    station_id = 106
    start_date = "2023-07-03"
    stop_date = "2023-08-03"

    station_json = get_data_from_api_bdx_by_station(
        station_id=station_id, start_date=start_date, stop_date=stop_date
    )

    station_df = transform_json_api_bdx_station_data_to_df(station_json)

    assert station_df["station_id"].nunique() == 1
    assert station_df["station_id"].unique() == [106]

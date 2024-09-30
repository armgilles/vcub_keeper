from vcub_keeper.production.data import (
    get_data_from_api_by_station,
    transform_json_station_data_to_df,
    get_data_from_api_bdx_by_station,
    transform_json_api_bdx_station_data_to_df,
)

#############################################
###            API Oslandia
#############################################


def test_get_api_data_one_station():
    """
    On test les données de l'API Oslandia afin d'obtenir des informations sur UNE station
    """

    station_id = 106
    start_date = "2021-12-29"
    stop_date = "2022-01-07"

    station_json = get_data_from_api_by_station(station_id=station_id, start_date=start_date, stop_date=stop_date)

    station_df = transform_json_station_data_to_df(station_json)

    # Check unique station number
    assert station_df["station_id"].unique().to_list() == [106]

    # Check la longeur du DataFrame
    assert len(station_df) == 1215

    assert station_df.columns == [
        "date",
        "station_id",
        "available_stands",
        "available_bikes",
        "status",
        "transactions_in",
        "transactions_out",
        "transactions_all",
    ]


def test_get_api_data_many_station():
    """
    On test les données de l'API Oslandia afin d'obtenir des informations sur plusieurs stations
    """

    station_id = [106, 3]
    start_date = "2021-12-29"
    stop_date = "2022-01-07"

    station_json = get_data_from_api_by_station(station_id=station_id, start_date=start_date, stop_date=stop_date)

    station_df = transform_json_station_data_to_df(station_json)

    # Check unique station number
    assert station_df["station_id"].unique().to_list() == [3, 106]

    # Check la longeur du DataFrame
    assert len(station_df) == 1215 * 2

    assert station_df.columns == [
        "date",
        "station_id",
        "available_stands",
        "available_bikes",
        "status",
        "transactions_in",
        "transactions_out",
        "transactions_all",
    ]


#############################################
###        API open data bordeaux
#############################################


def test_get_api_bdx_data_one_station():
    """
    On test les données de l'API open data de Bordeaux afin d'obtenir des informations sur UNE station
    """

    station_id = 106
    start_date = "2021-12-29"
    stop_date = "2022-01-07"

    station_json = get_data_from_api_bdx_by_station(station_id=station_id, start_date=start_date, stop_date=stop_date)

    station_df = transform_json_api_bdx_station_data_to_df(station_json)

    # Check unique station number
    assert station_df.select("station_id").n_unique() == 1

    # Check la longeur du DataFrame
    assert len(station_df) == 1296

    assert station_df.columns == [
        "station_id",
        "date",
        "available_stands",
        "available_bikes",
        "status",
        "transactions_in",
        "transactions_out",
        "transactions_all",
    ]


def test_get_api_bdx_data_many_station():
    """
    On test les données de l'API open data de Bordeaux afin d'obtenir des informations sur plusieurs stations
    """

    station_id = [106, 3]
    start_date = "2021-12-29"
    stop_date = "2022-01-07"

    station_json = get_data_from_api_bdx_by_station(station_id=station_id, start_date=start_date, stop_date=stop_date)

    station_df = transform_json_api_bdx_station_data_to_df(station_json)

    # Check unique station number
    assert station_df["station_id"].unique().to_list() == [3, 106]

    # Check la longeur du DataFrame
    assert len(station_df) == 1296 * 2

    assert station_df.columns == [
        "station_id",
        "date",
        "available_stands",
        "available_bikes",
        "status",
        "transactions_in",
        "transactions_out",
        "transactions_all",
    ]

import pytest
import polars as pl
import pandas as pd
import json

from vcub_keeper.transform.features_factory import (
    get_transactions_out,
    get_transactions_in,
    get_transactions_all,
    get_consecutive_no_transactions_out,
    process_data_cluster,
)
from vcub_keeper.production.data import transform_json_api_bdx_station_data_to_df
from vcub_keeper.reader.reader import read_activity_vcub
from vcub_keeper.config import ROOT_TESTS_DATA


def read_activity_data(file_name="activite_data.csv"):
    """
    Read test csv activity station's data.
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    return read_activity_vcub(ROOT_TESTS_DATA + file_name)


def read_json_data(file_name="data_test_api_from_bdx.json"):
    """
    Read test json data for bench get_consecutive_no_transactions_out()
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + file_name) as f:
        station_json_loaded = json.load(f)
    return station_json_loaded


def create_activite_data_big(activite_data: pd.DataFrame) -> pd.DataFrame:
    """Increase the number of rows of the DataFrame"""
    # Mapping new station_id to easely increase the number of rows
    new_station_id_dict = {22: 1, 43: 100, 102: 200, 106: 300, 123: 400}
    activite_data["station_id"] = activite_data["station_id"].map(new_station_id_dict)

    n = 99
    activite_data_big = activite_data.copy()
    # Concaténer le DataFrame n fois
    for i in range(n):
        temp = activite_data.copy()
        temp["station_id"] = temp["station_id"] + i
        activite_data_big = pd.concat([activite_data_big, temp], ignore_index=True)

    return activite_data_big


def create_station_df_from_json_big(station_df_from_json: pd.DataFrame) -> pd.DataFrame:
    """Increase the number of rows of the DataFrame"""

    # Nombre de fois que vous souhaitez concaténer le DataFrame
    n = 5_000
    station_df_from_json_big = station_df_from_json.copy()
    # Concaténer le DataFrame n fois
    for i in range(n):
        temp = station_df_from_json.copy()
        temp["station_id"] = temp["station_id"] + i
        station_df_from_json_big = pd.concat([station_df_from_json_big, temp], ignore_index=True)

    return station_df_from_json_big


activite_data = read_activity_data().collect()
activite_data_pandas = activite_data.to_pandas()
activite_data_big = pl.from_pandas(create_activite_data_big(activite_data_pandas)).lazy().collect()  # bigger dataset

# To test get_consecutive_no_transactions_out() function
station_json_loaded = read_json_data()
station_df_from_json = transform_json_api_bdx_station_data_to_df(station_json_loaded)
station_df_from_json_big = pl.from_pandas(
    create_station_df_from_json_big(station_df_from_json.to_pandas())
)  # bigger dataset


@pytest.mark.benchmark
def test_benchmark_get_transaction_out(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_out)
    """

    activite_data.with_columns(get_transactions_out())


@pytest.mark.benchmark
def test_benchmark_get_transaction_out_big(activite_data=activite_data_big):
    """
    Benchmark for transforming some feature (get_transactions_out)
    """

    activite_data.with_columns(get_transactions_out())


@pytest.mark.benchmark
def test_benchmark_get_transaction_in(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_in)
    """

    activite_data.with_columns(get_transactions_in())


@pytest.mark.benchmark
def test_benchmark_get_transaction_in_big(activite_data=activite_data_big):
    """
    Benchmark for transforming some feature (get_transactions_in)
    """

    activite_data.with_columns(get_transactions_in())


@pytest.mark.benchmark
def test_benchmark_get_transaction_all(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    activite_data.with_columns(get_transactions_all())


@pytest.mark.benchmark
def test_benchmark_get_transaction_all_big(activite_data=activite_data_big):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    activite_data.with_columns(get_transactions_all())


@pytest.mark.benchmark
def test_benchmark_get_consecutive_no_transactions_out(station_df_from_json=station_df_from_json):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    station_df_from_json_feature = get_consecutive_no_transactions_out(station_df_from_json)


@pytest.mark.benchmark
def test_benchmark_get_consecutive_no_transactions_out_big(station_df_from_json=station_df_from_json_big):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    station_df_from_json_feature = get_consecutive_no_transactions_out(station_df_from_json)


@pytest.mark.benchmark
def test_benchmark_process_data_cluster(activite_data=activite_data):
    """
    Benchmark for transforming some features with process_data_cluster()
    calling multiple time get_encoding_time() function
    """

    activite_data_feature = process_data_cluster(activite_data)


@pytest.mark.benchmark
def test_benchmark_process_data_cluster_big(activite_data=activite_data_big):
    """
    Benchmark for transforming some features with process_data_cluster()
    calling multiple time get_encoding_time() function
    """

    activite_data_feature = process_data_cluster(activite_data)

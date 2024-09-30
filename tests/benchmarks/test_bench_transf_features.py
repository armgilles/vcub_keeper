import pytest
import polars as pl
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


def read_activity_data(file_name="activite_data.csv", output_type=None):
    """
    Read test csv activity station's data.
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    return read_activity_vcub(ROOT_TESTS_DATA + file_name, output_type=output_type)


def read_json_data(file_name="data_test_api_from_bdx.json"):
    """
    Read test json data for bench get_consecutive_no_transactions_out()
    From notebooks/04_tests/03_test_data_activite.ipynb
    """

    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + file_name) as f:
        station_json_loaded = json.load(f)
    return station_json_loaded


activite_data = read_activity_data()
activite_data_pandas = read_activity_data(output_type="pandas")

# To test get_consecutive_no_transactions_out() function
station_json_loaded = read_json_data()
station_df_from_json = transform_json_api_bdx_station_data_to_df(station_json_loaded)


@pytest.mark.benchmark
def test_benchmark_get_transaction_out(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_out)
    """

    activity_data_feature = get_transactions_out(activite_data)


@pytest.mark.benchmark
def test_benchmark_get_transaction_in(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_in)
    """

    activity_data_feature = get_transactions_in(activite_data)


@pytest.mark.benchmark
def test_benchmark_get_transaction_all(activite_data=activite_data):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    activity_data_feature = get_transactions_all(activite_data)


@pytest.mark.benchmark
def test_benchmark_get_consecutive_no_transactions_out(station_df_from_json=station_df_from_json.to_pandas()):
    """
    Benchmark for transforming some feature (get_transactions_all)
    """

    station_df_from_json_feature = get_consecutive_no_transactions_out(station_df_from_json)


@pytest.mark.benchmark
def test_benchmark_process_data_cluster(activite_data=activite_data_pandas):
    """
    Benchmark for transforming some features with process_data_cluster()
    calling multiple time get_encoding_time() function
    """

    activite_data_feature = process_data_cluster(activite_data)

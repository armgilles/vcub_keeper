import pytest
import json

from vcub_keeper.config import ROOT_TESTS_DATA
from vcub_keeper.production.data import transform_json_api_bdx_station_data_to_df

def read_json_data(file_name='data_test_api_from_bdx.json'):
    """
    Read test json data
    """
    
    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + file_name) as f:
        station_json_loaded = json.load(f)
    return station_json_loaded


station_json_loaded = read_json_data()


@pytest.mark.benchmark
def test_benchmark_transf_json_to_df(json_data=station_json_loaded):
    """
    
    """
    
    station_df_from_json = transform_json_api_bdx_station_data_to_df(json_data)
    
    
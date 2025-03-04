import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import json

from vcub_keeper.production.data import transform_json_api_bdx_station_data_to_df
from vcub_keeper.config import ROOT_TESTS_DATA


def test_transf_json_to_df():
    """
    Check la fonction de transformation de données entre le json (from API call)
    vers un pd.DataFrame()

    Notebook of this test : notebooks/04_tests/02_test_transf_json_to_df.ipynb
    """

    # Loading data from data test (.json)
    with open(ROOT_TESTS_DATA + "data_test_api_from_bdx.json") as f:
        station_json_loaded = json.load(f)

    station_df_from_json = transform_json_api_bdx_station_data_to_df(station_json_loaded).collect()

    # Loading data from csv test (.csv)
    station_df_from_csv = pl.read_csv(
        ROOT_TESTS_DATA + "data_test_transf_json_to_df.csv",
        try_parse_dates=True,
        schema_overrides={"station_id": pl.Int32, "status": pl.UInt8},
    )

    # assert len(station_df_from_json.compare(station_df_from_csv)) == 0
    assert_frame_equal(station_df_from_json, station_df_from_csv)


def test_transf_json_to_df_with_new_status_value_165():
    """
    Check la fonction de transformation de données entre le json (from API call)
    avec de noouvelles valeurs pour le champs "etat" du json issue de l'api de Bordeaux

    Issue : https://github.com/armgilles/vcub_keeper/issues/165
    """

    station_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "time": "2024-05-06T12:20:00+02:00",
                    "gid": 14,
                    "ident": 15,
                    "nom": "Capdeville",
                    "etat": "DECONNECTEE",
                    "nbplaces": 1,
                    "nbvelos": 0,
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "time": "2024-05-06T12:25:00+02:00",
                    "gid": 14,
                    "ident": 15,
                    "nom": "Capdeville",
                    "etat": "CONNECTEE",
                    "nbplaces": 1,
                    "nbvelos": 0,
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "time": "2024-05-06T12:30:00+02:00",
                    "gid": 14,
                    "ident": 15,
                    "nom": "Capdeville",
                    "etat": "MAINTENANCE",
                    "nbplaces": 1,
                    "nbvelos": 0,
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "time": "2024-05-06T12:35:00+02:00",
                    "gid": 14,
                    "ident": 15,
                    "nom": "Capdeville",
                    "etat": "AUTRE CHOSE ???",  # New value
                    "nbplaces": 1,
                    "nbvelos": 0,
                },
            },
        ],
    }

    station_df_from_json = transform_json_api_bdx_station_data_to_df(station_json).collect()

    status_values = station_df_from_json.select(pl.col("status")).unique().to_series()
    assert status_values.is_in([0, 1]).all()

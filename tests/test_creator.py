import io
import pytest
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
from vcub_keeper.create.creator import create_station_attribute, calculate_breakpoints_
from vcub_keeper.reader.reader import read_stations_attributes


def test_create_station_attribute():
    """
    Test de la fonction create_station_attribute
    """

    data = """
Geo Point;Geo Shape;commune;IDENT;TYPE;NOM;NBPLACES;NBVELOS
44.83803,-0.58437;{"coordinates": [-0.58437, 44.83803], "type": "Point"};Bordeaux;1;VLS;Meriadeck;29;14
44.83784,-0.59028;{"coordinates": [-0.59028, 44.83784], "type": "Point"};Bordeaux;2;VLS;St Bruno;18;2
44.840813,-0.593233;{"coordinates": [-0.593233, 44.840813], "type": "Point"};Bordeaux;3;VLS;Piscine Judaique;21;5
"""

    stations = create_station_attribute(path_directory="", data=io.StringIO(data), export=False)
    # expected result
    expected_data = {
        "Geo Point": ["44.83803,-0.58437", "44.83784,-0.59028", "44.840813,-0.593233"],
        "Geo Shape": [
            '{"coordinates": [-0.58437, 44.83803], "type": "Point"}',
            '{"coordinates": [-0.59028, 44.83784], "type": "Point"}',
            '{"coordinates": [-0.593233, 44.840813], "type": "Point"}',
        ],
        "COMMUNE": ["Bordeaux", "Bordeaux", "Bordeaux"],
        "total_stand": [43, 20, 26],
        "NOM": ["Meriadeck", "St Bruno", "Piscine Judaique"],
        "TYPEA": ["VLS", "VLS", "VLS"],
        "station_id": [1, 2, 3],
        "lat": [44.83803, 44.83784, 44.840813],
        "lon": [-0.58437, -0.59028, -0.593233],
    }

    expected_df = pl.DataFrame(expected_data)

    assert_frame_equal(stations, expected_df, check_dtypes=False)


def test_read_stations_attributes():
    """
    Test de la fonction read_stations_attributes
    export test data from stations.head(3).to_dict(orient="list")
    """

    data = """
Geo Point;Geo Shape;COMMUNE;total_stand;NOM;TYPEA;station_id;lat;lon
44.83803,-0.58437;{"coordinates": [-0.58437, 44.83803], "type": "Point"};Bordeaux;43;Meriadeck;VLS;1;44.83803;-0.58437
44.83784,-0.59028;{"coordinates": [-0.59028, 44.83784], "type": "Point"};Bordeaux;20;St Bruno;VLS;2;44.83784;-0.59028
44.840813,-0.593233;{"coordinates": [-0.593233, 44.840813], "type": "Point"};Bordeaux;26;Piscine Judaique;VLS;3;44.840813;-0.593233
"""

    stations = read_stations_attributes(path_directory="", data=io.StringIO(data))
    # expected result
    expected_data = {
        "Geo Point": ["44.83803,-0.58437", "44.83784,-0.59028", "44.840813,-0.593233"],
        "Geo Shape": [
            '{"coordinates": [-0.58437, 44.83803], "type": "Point"}',
            '{"coordinates": [-0.59028, 44.83784], "type": "Point"}',
            '{"coordinates": [-0.593233, 44.840813], "type": "Point"}',
        ],
        "COMMUNE": ["Bordeaux", "Bordeaux", "Bordeaux"],
        "total_stand": [43, 20, 26],
        "NOM": ["Meriadeck", "St Bruno", "Piscine Judaique"],
        "TYPEA": ["VLS", "VLS", "VLS"],
        "station_id": [1, 2, 3],
        "lat": [44.83803, 44.83784, 44.840813],
        "lon": [-0.58437, -0.59028, -0.593233],
    }

    expected_df = pl.DataFrame(expected_data)

    assert_frame_equal(stations, expected_df, check_dtypes=False)


def test_create_breakpoints_to_use_cut():
    """
    Test de la fonction create_breakpoints_to_use_cut
    """

    data = {
        "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df_pl = pl.DataFrame(data)

    # Convertir en DataFrame Pandas pour obtenir les breakpoints
    df_pd = df_pl.to_pandas()

    # Use returbins to get the breakpoints (from pandas)
    df_pd["cut_label"], breakpoints = pd.cut(
        df_pd["value"], 4, labels=["low", "medium", "hight", "very high"], retbins=True
    )
    print(pl.from_pandas(df_pd))

    # Cut in folars
    breakpoints_custum = calculate_breakpoints_(df_pl["value"], 4)
    labels = ["low", "medium", "hight", "very high"]

    df_pl = df_pl.with_columns(pl.col("value").cut(breaks=breakpoints_custum, labels=labels).alias("cut_label"))

    # Check if breakpoints are the same
    assert (breakpoints[1:-1] == breakpoints_custum).all()

    # Check label's cut
    assert_frame_equal(df_pl, pl.from_pandas(df_pd))

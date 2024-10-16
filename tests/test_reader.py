import pytest
import polars as pl
from datetime import datetime

from vcub_keeper.reader.reader_utils import filter_periode
from polars.testing import assert_frame_equal


def test_filter_period():
    """
    Test de la fonction filter_period
    """

    data = {
        "date": [
            datetime(2020, 3, 16),
            datetime(2020, 3, 17),
            datetime(2020, 5, 13),
            datetime(2020, 5, 14),
            datetime(2020, 5, 15),
        ],
        "station_id": [
            1,
            2,
            3,
            4,
            5,
        ],
    }

    df = pl.DataFrame(data)

    expected_data = {
        "date": [
            datetime(2020, 3, 16),
            datetime(2020, 5, 15),
        ],
        "station_id": [1, 5],
    }

    expected_df = pl.DataFrame(expected_data)

    expected_df = pl.DataFrame(expected_data)

    result = filter_periode(data=df, non_use_station_id=[4])

    assert_frame_equal(result, expected_df)

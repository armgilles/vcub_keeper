import pytest
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
from datetime import datetime

from vcub_keeper.transform.features_factory import (
    get_transactions_out,
    get_transactions_in,
    get_transactions_all,
    get_consecutive_no_transactions_out,
    get_encoding_time,
)


def test_get_transactions_out_pandas():
    """
    test de la fonction get_transactions_out()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [11, 14, 14, 14, 15, 15, 17, 17, 17, 17, 18] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [9, 6, 6, 6, 5, 5, 3, 3, 3, 3, 2] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 11:04:04"),
            pd.Timestamp("2017-07-09 11:09:04"),
            pd.Timestamp("2017-07-09 11:14:04"),
            pd.Timestamp("2017-07-09 11:19:04"),
            pd.Timestamp("2017-07-09 11:24:04"),
            pd.Timestamp("2017-07-09 11:29:04"),
            pd.Timestamp("2017-07-09 11:34:04"),
            pd.Timestamp("2017-07-09 11:39:03"),
            pd.Timestamp("2017-07-09 11:44:03"),
            pd.Timestamp("2017-07-09 11:49:04"),
            pd.Timestamp("2017-07-09 11:54:05"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_out": [0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
        + [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }

    df_activite = pd.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(columns=["transactions_out"], axis=1)

    result = get_transactions_out(pl.from_pandas(df_activite), output_type="pandas")

    expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_transactions_out():
    """
    test de la fonction get_transactions_out()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [11, 14, 14, 14, 15, 15, 17, 17, 17, 17, 18] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [9, 6, 6, 6, 5, 5, 3, 3, 3, 3, 2] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 11:04:04"),
            pd.Timestamp("2017-07-09 11:09:04"),
            pd.Timestamp("2017-07-09 11:14:04"),
            pd.Timestamp("2017-07-09 11:19:04"),
            pd.Timestamp("2017-07-09 11:24:04"),
            pd.Timestamp("2017-07-09 11:29:04"),
            pd.Timestamp("2017-07-09 11:34:04"),
            pd.Timestamp("2017-07-09 11:39:03"),
            pd.Timestamp("2017-07-09 11:44:03"),
            pd.Timestamp("2017-07-09 11:49:04"),
            pd.Timestamp("2017-07-09 11:54:05"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_out": [0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 1] + [0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0],
    }

    df_activite = pl.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop("transactions_out")

    result = get_transactions_out(df_activite)

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)


def test_get_transactions_in_pandas():
    """
    test de la fonction get_transactions_in()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [19, 19, 18, 18, 18, 16, 16, 16, 16, 17, 17] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [1, 1, 2, 2, 2, 4, 4, 4, 4, 3, 3] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 03:24:05"),
            pd.Timestamp("2017-07-09 03:29:04"),
            pd.Timestamp("2017-07-09 03:34:04"),
            pd.Timestamp("2017-07-09 03:39:04"),
            pd.Timestamp("2017-07-09 03:44:05"),
            pd.Timestamp("2017-07-09 03:49:03"),
            pd.Timestamp("2017-07-09 03:54:04"),
            pd.Timestamp("2017-07-09 03:59:03"),
            pd.Timestamp("2017-07-09 04:04:06"),
            pd.Timestamp("2017-07-09 04:09:04"),
            pd.Timestamp("2017-07-09 04:14:04"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_in": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        + [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }

    df_activite = pd.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(columns=["transactions_in"], axis=1)

    result = get_transactions_in(pl.from_pandas(df_activite), output_type="pandas")

    expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_transactions_in():
    """
    test de la fonction get_transactions_in()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [19, 19, 18, 18, 18, 16, 16, 16, 16, 17, 17] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [1, 1, 2, 2, 2, 4, 4, 4, 4, 3, 3] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 03:24:05"),
            pd.Timestamp("2017-07-09 03:29:04"),
            pd.Timestamp("2017-07-09 03:34:04"),
            pd.Timestamp("2017-07-09 03:39:04"),
            pd.Timestamp("2017-07-09 03:44:05"),
            pd.Timestamp("2017-07-09 03:49:03"),
            pd.Timestamp("2017-07-09 03:54:04"),
            pd.Timestamp("2017-07-09 03:59:03"),
            pd.Timestamp("2017-07-09 04:04:06"),
            pd.Timestamp("2017-07-09 04:09:04"),
            pd.Timestamp("2017-07-09 04:14:04"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_in": [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
    }

    df_activite = pl.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop("transactions_in")

    result = get_transactions_in(df_activite)

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)


def test_get_transactions_all_pandas():
    """
    test de la fonction get_transactions_all()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [19, 19, 18, 18, 18, 16, 16, 16, 16, 17, 17] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [1, 1, 2, 2, 2, 4, 4, 4, 4, 3, 3] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 03:24:05"),
            pd.Timestamp("2017-07-09 03:29:04"),
            pd.Timestamp("2017-07-09 03:34:04"),
            pd.Timestamp("2017-07-09 03:39:04"),
            pd.Timestamp("2017-07-09 03:44:05"),
            pd.Timestamp("2017-07-09 03:49:03"),
            pd.Timestamp("2017-07-09 03:54:04"),
            pd.Timestamp("2017-07-09 03:59:03"),
            pd.Timestamp("2017-07-09 04:04:06"),
            pd.Timestamp("2017-07-09 04:09:04"),
            pd.Timestamp("2017-07-09 04:14:04"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_all": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        + [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    }

    df_activite = pd.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(columns=["transactions_all"], axis=1)

    result = get_transactions_all(pl.from_pandas(df_activite), output_type="pandas")

    expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_transactions_all():
    """
    test de la fonction get_transactions_all()
    """
    data = {
        "gid": [83] * 11 + [92] * 11,
        "station_id": [1] * 11 + [22] * 11,
        "type": ["VLS"] * 11 + ["VLS"] * 11,
        "name": ["Meriadeck"] * 11 + ["Hotel de Ville"] * 11,
        "state": [1] * 11 + [1] * 11,
        "available_stands": [19, 19, 18, 18, 18, 16, 16, 16, 16, 17, 17] + [33, 33, 31, 33, 33, 33, 33, 32, 33, 33, 33],
        "available_bikes": [1, 1, 2, 2, 2, 4, 4, 4, 4, 3, 3] + [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        "date": [
            pd.Timestamp("2017-07-09 03:24:05"),
            pd.Timestamp("2017-07-09 03:29:04"),
            pd.Timestamp("2017-07-09 03:34:04"),
            pd.Timestamp("2017-07-09 03:39:04"),
            pd.Timestamp("2017-07-09 03:44:05"),
            pd.Timestamp("2017-07-09 03:49:03"),
            pd.Timestamp("2017-07-09 03:54:04"),
            pd.Timestamp("2017-07-09 03:59:03"),
            pd.Timestamp("2017-07-09 04:04:06"),
            pd.Timestamp("2017-07-09 04:09:04"),
            pd.Timestamp("2017-07-09 04:14:04"),
        ]
        + [
            pd.Timestamp("2017-07-09 00:54:05"),
            pd.Timestamp("2017-07-09 00:59:04"),
            pd.Timestamp("2017-07-09 01:04:04"),
            pd.Timestamp("2017-07-09 01:09:03"),
            pd.Timestamp("2017-07-09 01:14:04"),
            pd.Timestamp("2017-07-09 01:19:04"),
            pd.Timestamp("2017-07-09 01:24:04"),
            pd.Timestamp("2017-07-09 01:29:04"),
            pd.Timestamp("2017-07-09 01:34:04"),
            pd.Timestamp("2017-07-09 01:39:04"),
            pd.Timestamp("2017-07-09 01:44:05"),
        ],
        "transactions_all": [0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0] + [0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0],
    }

    df_activite = pl.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop("transactions_all")

    result = get_transactions_all(df_activite)

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)


def test_get_consecutive_no_transactions_out():
    """
    test de la fonction get_consecutive_no_transactions_out()
    """
    data = {
        "station_id": [1] * 11 + [22] * 11,
        "date": [
            pd.Timestamp("2018-12-01 00:10:00"),
            pd.Timestamp("2018-12-01 00:20:00"),
            pd.Timestamp("2018-12-01 00:30:00"),
            pd.Timestamp("2018-12-01 00:40:00"),
            pd.Timestamp("2018-12-01 00:50:00"),
            pd.Timestamp("2018-12-01 01:00:00"),
            pd.Timestamp("2018-12-01 01:10:00"),
            pd.Timestamp("2018-12-01 01:20:00"),
            pd.Timestamp("2018-12-01 01:30:00"),
            pd.Timestamp("2018-12-01 01:40:00"),
            pd.Timestamp("2018-12-01 01:50:00"),
        ]
        + [
            pd.Timestamp("2018-12-01 00:10:00"),
            pd.Timestamp("2018-12-01 00:20:00"),
            pd.Timestamp("2018-12-01 00:30:00"),
            pd.Timestamp("2018-12-01 00:40:00"),
            pd.Timestamp("2018-12-01 00:50:00"),
            pd.Timestamp("2018-12-01 01:00:00"),
            pd.Timestamp("2018-12-01 01:10:00"),
            pd.Timestamp("2018-12-01 01:20:00"),
            pd.Timestamp("2018-12-01 01:30:00"),
            pd.Timestamp("2018-12-01 01:40:00"),
            pd.Timestamp("2018-12-01 01:50:00"),
        ],
        "available_stands": [28.0, 28.0, 28.0, 27.0, 27.0, 28.0, 28.0, 28.0, 28.0, 29.0, 29.0]
        + [2.0, 6.0, 9.0, 9.0, 11.0, 13.0, 13.0, 13.0, 17.0, 18.0, 20.0],
        "available_bikes": [4.0, 4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0]
        + [31.0, 27.0, 24.0, 24.0, 22.0, 20.0, 20.0, 20.0, 16.0, 15.0, 13.0],
        "status": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1] + [1] * 11,
        "transactions_out": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        + [0.0, 4.0, 3.0, 0.0, 2.0, 2.0, 0.0, 0.0, 4.0, 1.0, 2.0],
        "consecutive_no_transactions_out": [0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1] + [0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0],
    }

    df_activite = pl.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(["transactions_out", "consecutive_no_transactions_out"])

    result = get_transactions_out(df_activite)
    result = get_consecutive_no_transactions_out(result)

    expected = pl.DataFrame(data)

    pl.testing.assert_frame_equal(result, expected)


def test_get_encoding_time_quarter():
    """
    test de la fonction get_encoding_time() pour les trimestres
    """
    data = pl.DataFrame(
        {
            "date": pl.date_range(start=datetime(2022, 1, 1), end=datetime(2022, 12, 1), interval="1q", eager=True),
            "quarter": [1, 2, 3, 4],
        }
    )

    result = get_encoding_time(data, "quarter", max_val=4)

    expected_sin = np.sin(2 * np.pi * data["quarter"] / 4)
    expected_cos = np.cos(2 * np.pi * data["quarter"] / 4)

    pl.testing.assert_series_equal(result["Sin_quarter"], expected_sin, check_names=False)
    pl.testing.assert_series_equal(result["Cos_quarter"], expected_cos, check_names=False)


def test_get_encoding_time_weekday():
    """
    test de la fonction get_encoding_time() pour les jours de la semaine
    """
    data = pl.DataFrame(
        {
            "date": pl.date_range(start=datetime(2022, 1, 1), end=datetime(2022, 1, 7), interval="1d", eager=True),
            "weekday": [0, 1, 2, 3, 4, 5, 6],
        }
    )

    result = get_encoding_time(data, "weekday", max_val=7)

    expected_sin = np.sin(2 * np.pi * data["weekday"] / 7)
    expected_cos = np.cos(2 * np.pi * data["weekday"] / 7)

    pl.testing.assert_series_equal(result["Sin_weekday"], expected_sin, check_names=False)
    pl.testing.assert_series_equal(result["Cos_weekday"], expected_cos, check_names=False)


def test_get_encoding_time_hours():
    """
    test de la fonction get_encoding_time() pour les heures
    """
    data = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=datetime(2022, 1, 1, 0, 0, 0), end=datetime(2022, 1, 1, 23, 0, 0), interval="1h", eager=True
            ),
            "hours": list(range(24)),
        }
    )

    result = get_encoding_time(data, "hours", max_val=24)

    expected_sin = np.sin(2 * np.pi * data["hours"] / 24)
    expected_cos = np.cos(2 * np.pi * data["hours"] / 24)

    pl.testing.assert_series_equal(result["Sin_hours"], expected_sin, check_names=False)
    pl.testing.assert_series_equal(result["Cos_hours"], expected_cos, check_names=False)

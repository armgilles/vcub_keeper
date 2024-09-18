import pytest
import pandas as pd

from vcub_keeper.transform.features_factory import get_transactions_out, get_transactions_in


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
        "transactions_out": [0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
        + [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }

    df_activite = pd.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(columns=["transactions_out"], axis=1)

    result = get_transactions_out(df_activite)

    expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)


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
        "transactions_in": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        + [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }

    df_activite = pd.DataFrame(data)
    # drop columns we want to test.
    df_activite = df_activite.drop(columns=["transactions_in"], axis=1)

    result = get_transactions_in(df_activite)

    expected = pd.DataFrame(data)

    pd.testing.assert_frame_equal(result, expected)

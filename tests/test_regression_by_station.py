import pytest
from datetime import datetime, timedelta
import polars as pl

import numpy as np
from vcub_keeper.ml.prediction_station.utils import create_target


@pytest.fixture
def mock_histo_data():
    """Create mock station data histo for testing"""
    rng = np.random.default_rng(42)

    # Générer des dates avec une période de 10 minutes
    start_date = datetime(2025, 3, 1)
    end_date = datetime(2025, 3, 4)
    date_range = [
        start_date + timedelta(minutes=10 * i) for i in range(int((end_date - start_date).total_seconds() / 600))
    ]

    # Générer des données pour station_id 101
    station_id_101 = np.array([101] * len(date_range))
    available_stands_101 = rng.integers(0, 20, size=len(date_range))
    available_bikes_101 = rng.integers(0, 20, size=len(date_range))

    # Générer des données pour station_id 102
    station_id_102 = np.array([102] * len(date_range))
    available_stands_102 = rng.integers(0, 20, size=len(date_range))
    available_bikes_102 = rng.integers(0, 20, size=len(date_range))

    # Créer des DataFrames polars pour chaque station_id
    df_101 = pl.DataFrame(
        {
            "station_id": station_id_101,
            "date": date_range,
            "available_stands": available_stands_101,
            "available_bikes": available_bikes_101,
        }
    )

    df_102 = pl.DataFrame(
        {
            "station_id": station_id_102,
            "date": date_range,
            "available_stands": available_stands_102,
            "available_bikes": available_bikes_102,
        }
    )

    # Combiner les DataFrames
    df_historical_station = pl.concat([df_101, df_102])

    df_historical_station = df_historical_station.with_columns(
        [
            pl.col("station_id").cast(pl.UInt16),
            pl.col("date").cast(pl.Datetime),
            pl.col("available_stands").cast(pl.UInt16),
            pl.col("available_bikes").cast(pl.UInt16),
        ]
    )

    # Trier par station_id puis date
    df_historical_station = df_historical_station.sort(["station_id", "date"])

    return df_historical_station.lazy()


@pytest.mark.parametrize(
    "station_id, target_col, horizon_prediction, first_5_target_values, last_5_target_values",
    [
        (101, "available_stands", "20m", [13, 8, 8, 17, 1], [14, 1, 14, None, None]),
        (101, "available_bikes", "20m", [8, 2, 11, 18, 1], [18, 2, 15, None, None]),
        (102, "available_stands", "30m", [16, 8, 0, 9, 4], [19, 10, None, None, None]),
        (102, "available_bikes", "30m", [0, 15, 8, 13, 17], [18, 11, None, None, None]),
    ],
)
def test_create_target(
    mock_histo_data, station_id, target_col, horizon_prediction, first_5_target_values, last_5_target_values
):
    """Test create_target function"""

    # Mock data
    df_historical_station = mock_histo_data

    # # Test create_target function
    # target_col = "available_stands"

    # horizon_prediction = "30m"
    station_to_pred = df_historical_station.filter(pl.col("station_id") == station_id)
    len_df = station_to_pred.collect().shape[0]
    station_to_pred = create_target(station_to_pred, target_col, horizon_prediction)

    # basic test
    assert station_to_pred.collect().shape[0] == len_df
    # Assert the target column is created
    assert station_to_pred.collect().head(5).select(pl.col("target")).to_series().to_list() == first_5_target_values
    assert station_to_pred.collect().tail(5).select(pl.col("target")).to_series().to_list() == last_5_target_values

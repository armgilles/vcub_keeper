from datetime import datetime
import pytest
import polars as pl
from vcub_keeper.llm.agent import create_agent, create_chat
from vcub_keeper.llm.crewai.tool_python import get_geocoding


# Appliquer le marqueur à tous les tests de ce fichier
pytestmark = pytest.mark.llm_api  # ou ajout d'un marker sur chaque test - @pytest.mark.llm_api


@pytest.fixture
def mock_station_data():
    """Create mock station data for testing"""
    data = {
        "station_id": [1, 2, 3, 4, 5, 6],
        "date": [
            datetime(
                2025,
                3,
                5,
                12,
                30,
                0,
            )
        ]
        * 6,
        "station_name": [
            "Meriadeck",
            "St Bruno",
            "Piscine Judaique",
            "St Seurin",
            "Place Gambetta",
            "Square Andre Lhote",
        ],
        "available_stands": [10, 5, 15, 2, 8, 0],
        "available_bikes": [20, 15, 5, 18, 12, 19],
        "status": [1, 1, 1, 1, 1, 1],
        "lat": [
            44.83803,
            44.83784,
            44.840813,
            44.84221,
            44.840714,
            44.83779,
        ],
        "lon": [
            -0.58437,
            -0.59028,
            -0.593233,
            -0.58482,
            -0.581124,
            -0.58166,
        ],
        "anomaly": [1, 1, 1, 1, 1, -1],
        "commune_name": [
            "Bordeaux",
            "Bordeaux",
            "Bordeaux",
            "Bordeaux",
            "Bordeaux",
            "Bordeaux",
        ],
    }

    df = pl.DataFrame(data)
    df = df.with_columns(
        [
            pl.col("station_id").cast(pl.UInt16),
            pl.col("date").cast(pl.Datetime),
            pl.col("available_stands").cast(pl.UInt16),
            pl.col("available_bikes").cast(pl.UInt16),
            pl.col("status").cast(pl.UInt8),
            pl.col("anomaly").cast(pl.Float32),
            pl.col("lat").cast(pl.Float32),
            pl.col("lon").cast(pl.Float32),
            pl.col("station_name").cast(pl.Categorical),
            pl.col("commune_name").cast(pl.Categorical),
        ]
    )

    # Rounding
    df = df.with_columns(
        [
            pl.col("lat").round(4).alias("lat"),
            pl.col("lon").round(4).alias("lon"),
        ]
    )

    return df


def test_adresse_geocoding():
    """
    Test de gécodage d'une adresse
    """
    lat, lon = get_geocoding("place de la bourse, bordeaux")
    assert abs(lat - 44.8414565) < 0.001
    assert abs(lon - (-0.57037969)) < 0.001

import pytest
from datetime import datetime, timedelta
import polars as pl
from polars.testing import assert_frame_equal

import numpy as np

from vcub_keeper.llm.agent import create_agent, create_chat


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
            "station_name": ["La Gare central"] * len(date_range),
            "available_stands": available_stands_101,
            "available_bikes": available_bikes_101,
        }
    )

    df_102 = pl.DataFrame(
        {
            "station_id": station_id_102,
            "date": date_range,
            "station_name": ["Le Parc vert"] * len(date_range),
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


@pytest.fixture
def agent(mock_histo_data):
    """Create an instance of agent_vcub with mock data"""
    # Assuming AgentVcub is initialized with a dataframe

    # On prend la dernière date du df pour créer last_info_station
    last_info_station = mock_histo_data.filter(pl.col("date") == pl.col("date").max()).collect()

    chat_llm = create_chat(model="mistral-small-latest", temperature=0.0)
    agent_vcub = create_agent(chat=chat_llm, list_dfs=[last_info_station, mock_histo_data])

    return agent_vcub


def test_message_prediction_station(agent):
    """Test the message prediction for a specific station"""

    user_message = "Combien il y aura de vélo disponible dans 10 minutes à la station du parc vert ?"
    response = agent.invoke({"input": user_message})

    assert "11" in response["output"]


def test_message_prediction_station_heure(agent):
    """Test the message prediction for a specific station with a specific time
    Last time is 2025-03-03 23:50:00
    """

    user_message = """Combien il y aura de places disponible à la
    station de la Gare central cette nuit à 2h du matin et indique moi l'horraire de prédiction ?"""
    # "horizon_prediction": "2h10m" / "130m"
    response = agent.invoke({"input": user_message})

    assert "12" in response["output"]
    assert "4 mars 2025 à 2h" in response["output"] or "2025-03-04 02:00:00" in response["output"]

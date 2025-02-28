import pytest
import pandas as pd
import polars as pl
import sys
import os
from unittest.mock import patch
from vcub_keeper.llm.agent import create_agent, create_chat


# Appliquer le marqueur à tous les tests de ce fichier
pytestmark = pytest.mark.llm_api  # ou ajout d'un marker sur chaque test - @pytest.mark.llm_api


@pytest.fixture
def mock_station_data():
    """Create mock station data for testing"""
    data = {
        "station_id": [1, 2, 3, 4, 5, 6],
        "NOM": ["Meriadeck", "St Bruno", "Piscine Judaique", "St Seurin", "Place Gambetta", "Square Andre Lhote"],
        "bikes_available": [10, 5, 15, 2, 8, 0],
        "bikes_mechanical": [7, 3, 10, 1, 5, 0],
        "bikes_ebike": [3, 2, 5, 1, 3, 0],
        "stands_available": [20, 15, 5, 18, 12, 20],
        "total_stands": [30, 20, 20, 20, 20, 20],
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
    }
    return pl.DataFrame(data)


@pytest.fixture
def agent(mock_station_data):
    """Create an instance of agent_vcub with mock data"""
    # Assuming AgentVcub is initialized with a dataframe

    chat_llm = create_chat(model="mistral-small-latest", temperature=0.0)
    agent_vcub = create_agent(chat=chat_llm, last_info_station=mock_station_data)

    return agent_vcub


def test_count_stations(agent):
    """Test the query about the number of stations"""
    user_message = "Combien il y a de stations ?"  # 6
    response = agent.invoke({"input": user_message})
    # Il y a 6 stations.

    # The agent should report 6 stations
    assert "6" in response["output"].lower()
    assert "station" in response["output"].lower()


def test_most_bikes_available(agent):
    """Test the query about the station with the most bikes"""
    user_message = "Quelle est la station avec le plus de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    # La station avec le plus de vélos disponibles est "Piscine Judaique" avec 15 vélos.

    # Piscine Judaique has the most bikes (15)
    assert "piscine judaique" in response["output"].lower()
    assert "15" in response["output"]


def test_least_bikes_available(agent):
    """Test the query about the station with the least bikes"""
    user_message = "Quelle est la station avec le moins de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    # La station avec le moins de vélos disponibles est "Square Andre Lhote"
    # avec 0 vélos disponibles.

    # Pey Berland has the least bikes (0)
    assert "square andre lhote" in response["output"].lower()
    assert "0" in response["output"]


def test_distance_calculation(agent):
    """Test the query about distance and travel time"""
    user_message = "Quelle est la distance entre Meriadeck et la Place Gambetta ? Si je roule à 15km/h, combien de temps vais-je mettre ?"
    response = agent.invoke({"input": user_message})
    # La distance entre Meriadeck et Place Gambetta est de 0.29 km. Si vous
    # roulez à 15 km/h, il vous faudra environ 0.02 heures, soit environ 1.2
    # minutes.

    assert "km" in response["output"].lower() or "kilomètre" in response["output"].lower()
    assert "minute" in response["output"].lower()
    assert "1.2" in response["output"].lower() or "1 minutes" in response["output"].lower()
    assert "meriadeck" in response["output"].lower()
    assert "place gambetta" in response["output"].lower()
    assert "15km/h" in response["output"] or "15 km/h" in response["output"]

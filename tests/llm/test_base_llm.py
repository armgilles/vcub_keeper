import pytest
import pandas as pd
import polars as pl
import sys
import os
from unittest.mock import patch
from vcub_keeper.llm.agent import create_agent, create_chat


@pytest.fixture
def mock_station_data():
    """Create mock station data for testing"""
    data = {
        "station_id": [1, 2, 3, 4, 5, 6],
        "NOM": ["Stalingrad", "Porte de Bourgogne", "Meriadeck", "Gare Saint-Jean", "Victoire", "Pey Berland"],
        "bikes_available": [10, 5, 15, 2, 8, 0],
        "bikes_mechanical": [7, 3, 10, 1, 5, 0],
        "bikes_ebike": [3, 2, 5, 1, 3, 0],
        "stands_available": [20, 15, 5, 18, 12, 20],
        "total_stands": [30, 20, 20, 20, 20, 20],
        "lat": [44.8431, 44.8378, 44.8380, 44.8256, 44.8317, 44.8376],
        "lon": [-0.5630, -0.5698, -0.5844, -0.5552, -0.5725, -0.5793],
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

    # The agent should report 6 stations
    assert "6" in response["output"].lower()
    assert "station" in response["output"].lower()


def test_most_bikes_available(agent):
    """Test the query about the station with the most bikes"""
    user_message = "Quelle est la station avec le plus de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})

    # Meriadeck has the most bikes (15)
    assert "meriadeck" in response["output"].lower()
    assert "15" in response["output"]


def test_least_bikes_available(agent):
    """Test the query about the station with the least bikes"""
    user_message = "Quelle est la station avec le moins de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})

    # Pey Berland has the least bikes (0)
    assert "pey berland" in response["output"].lower()
    assert "0" in response["output"]


def test_distance_calculation(agent):
    """Test the query about distance and travel time"""
    user_message = "Quelle est la distance entre la place Stalingrad et la porte de Bourgogne ? Si je roule à 15km/h, combien de temps vais-je mettre ?"
    response = agent.invoke({"input": user_message})

    # The distance should be calculated and time estimated
    assert "km" in response["output"].lower() or "kilomètre" in response["output"].lower()
    assert "minute" in response["output"].lower()
    # We can't assert exact values as we don't know the calculation method
    # But we can check if the response contains relevant information
    assert "stalingrad" in response["output"].lower()
    assert "bourgogne" in response["output"].lower()
    assert "15km/h" in response["output"]

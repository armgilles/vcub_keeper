import pytest
import polars as pl
from vcub_keeper.llm.agent import create_agent, create_chat


# Appliquer le marqueur à tous les tests de ce fichier
pytestmark = pytest.mark.llm_api  # ou ajout d'un marker sur chaque test - @pytest.mark.llm_api


@pytest.fixture
def mock_station_data():
    """Create mock station data for testing"""
    data = {
        "station_id": [1, 2, 3, 4, 5, 6],
        "station_name": [
            "Meriadeck",
            "St Bruno",
            "Piscine Judaique",
            "St Seurin",
            "Place Gambetta",
            "Square Andre Lhote",
        ],
        "available_stands": [10, 5, 15, 2, 8, 0],
        "available_bikes": [20, 15, 5, 18, 12, 20],
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
    }
    return pl.DataFrame(data)


@pytest.fixture
def agent(mock_station_data):
    """Create an instance of agent_vcub with mock data"""
    # Assuming AgentVcub is initialized with a dataframe

    chat_llm = create_chat(model="mistral-small-latest", temperature=0.0)
    agent_vcub = create_agent(chat=chat_llm, last_info_station=mock_station_data)

    # Assert the agent has the correct data
    assert mock_station_data.filter(pl.col("station_name") == "Meriadeck").select("available_bikes")[0, 0] == 20

    return agent_vcub


def test_count_stations(agent):
    """Test the query about the number of stations"""
    user_message = "Combien il y a de stations ?"  # 6
    response = agent.invoke({"input": user_message})
    # Il y a 6 stations.

    assert "6" in response["output"].lower()
    assert "station" in response["output"].lower()


def test_most_bikes_available(agent):
    """Test the query about the station with the most bikes"""
    user_message = "Quelle est la station avec le plus de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    # La station avec le plus de vélos disponibles est "Meriadeck" avec 20 vélos disponibles.

    assert "meriadeck" in response["output"].lower()
    assert "20" in response["output"]


def test_least_bikes_available(agent):
    """Test the query about the station with the least bikes"""
    user_message = "Quelle est la station avec le moins de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    # La station avec le moins de vélos disponibles est "Piscine Judaique" avec 5 vélos disponibles.

    assert "piscine judaique" in response["output"].lower()
    assert "5" in response["output"]


def test_distance_calculation(agent):
    """Test the query about distance and travel time"""
    user_message = "Quelle est la distance entre Meriadeck et la Place Gambetta ? Si je roule à 15km/h, combien de temps vais-je mettre ?"
    response = agent.invoke({"input": user_message})

    # Peut avoir un décalage sur la distance et donc le temps
    # Action: calculate_distance
    # Action Input: lat1=44.838, lon1=-0.58437, lat2=44.8407, lon2=-0.581124
    # Observation: La distance entre Meriadeck et Place Gambetta est de 0.32 km

    # La distance entre Meriadeck et Place Gambetta est de 0.29 km. Si vous
    # roulez à 15 km/h, il vous faudra environ 0.02 heures, soit environ 1.2
    # minutes.

    assert "km" in response["output"].lower() or "kilomètre" in response["output"].lower()
    assert "minute" in response["output"].lower()
    assert "1.2" in response["output"].lower() or "1 minute" in response["output"].lower()
    assert "meriadeck" in response["output"].lower()
    assert "place gambetta" in response["output"].lower()
    assert "15km/h" in response["output"] or "15 km/h" in response["output"]

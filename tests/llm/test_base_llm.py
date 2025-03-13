from datetime import datetime
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


@pytest.fixture
def agent(mock_station_data):
    """Create an instance of agent_vcub with mock data"""
    # Assuming AgentVcub is initialized with a dataframe

    chat_llm = create_chat(model="mistral-small-latest", temperature=0.0)
    agent_vcub = create_agent(chat=chat_llm, list_dfs=[mock_station_data, pl.LazyFrame()])

    return agent_vcub


def test_count_stations(agent):
    """Test the query about the number of stations"""
    user_message = "Combien il y a de stations ?"  # 6
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # Il y a 6 stations.

    assert "6" in response["output"].lower()
    assert "station" in response["output"].lower()


def test_most_bikes_available(agent):
    """Test the query about the station with the most bikes"""
    user_message = "Quelle est la station avec le plus de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # La station avec le plus de vélos disponibles est "Meriadeck" avec 20 vélos disponibles.

    assert "meriadeck" in response["output"].lower()
    assert "20" in response["output"]


def test_least_bikes_available(agent):
    """Test the query about the station with the least bikes"""
    user_message = "Quelle est la station avec le moins de vélos disponibles et combien ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # La station avec le moins de vélos disponibles est "Piscine Judaique" avec 5 vélos disponibles.

    assert "piscine judaique" in response["output"].lower()
    assert "5" in response["output"]


def test_distance_calculation(agent):
    """Test the query about distance and travel time"""
    user_message = "Quelle est la distance entre Meriadeck et la Place Gambetta ? Si je roule à 15km/h, combien de temps vais-je mettre ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")

    # Peut avoir un décalage sur la distance et donc le temps
    # Action: calculate_distance
    # Action Input: lat1=44.838, lon1=-0.58437, lat2=44.8407, lon2=-0.581124
    # Observation: La distance entre Meriadeck et Place Gambetta est de 0.32 km

    # La distance entre Meriadeck et la Place Gambetta est d'environ 0.397 km.
    # Si vous roulez à 15 km/h, il vous faudra environ 1.59 minutes pour
    # parcourir cette distance.

    assert "km" in response["output"].lower() or "kilomètre" in response["output"].lower()
    assert "minute" in response["output"].lower()
    assert "1.5" in response["output"].lower() or "1 minute" in response["output"].lower()
    assert "meriadeck" in response["output"].lower()
    assert "place gambetta" in response["output"].lower()
    assert "15km/h" in response["output"] or "15 km/h" in response["output"]


def test_message_history(agent):
    """Test afin de vérifier l'accès à l'historique des messages"""
    # 1 message
    user_message_1 = "Combien il y a de vélos disponibles à la station St Bruno ?"
    response = agent.invoke({"input": user_message_1})

    # 2 message
    user_message = "Quelle est exactement mon dernier message ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")

    # pas accès à l'historique des messages précédents
    assert "pas accès" not in response["output"].lower()
    assert "historique" not in response["output"].lower()

    # Vrai réponsee
    assert user_message_1.lower() in response["output"].lower()


def test_message_anomaly(agent):
    """Test afin de vérifier les anomalies dans les stations"""
    user_message = "Quelle station est en anomalie ?"
    response = agent.invoke({"input": user_message})
    # La station en anomalie est Square Andre Lhote.

    assert "Square Andre Lhote".lower() in response["output"].lower()


def test_message_bonjour(agent):
    """Test afin de vérifier le bonjour de l'agent"""
    user_message = "Bonjour"
    response = agent.invoke({"input": user_message})
    # Bonjour! Comment puis-je vous aider avec les données des stations VCub de Bordeaux?

    assert "Bonjour".lower() in response["output"].lower()

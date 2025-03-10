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
        "station_id": [1, 2, 3, 4, 5, 103],
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
            "Place de la Bourse",
            "Place du Palais",
        ],
        "available_stands": [10, 5, 15, 2, 0, 10],
        "available_bikes": [20, 15, 5, 18, 40, 22],
        "status": [1, 1, 1, 1, 1, 1],
        "lat": [
            44.83803,
            44.83784,
            44.840813,
            44.84221,
            44.840302,
            44.837799,
        ],
        "lon": [
            -0.58437,
            -0.59028,
            -0.593233,
            -0.58482,
            44.840302,
            -0.5703,
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
    agent_vcub = create_agent(chat=chat_llm, last_info_station=mock_station_data)

    return agent_vcub


def test_adresse_geocoding():
    """
    Test de gécodage d'une adresse
    """
    lat, lon = get_geocoding("place de la bourse, bordeaux")
    assert abs(lat - 44.8414565) < 0.001
    assert abs(lon - (-0.57037969)) < 0.001


def test_message_geocoding(agent):
    """Test afin de vérifier le geocoding que renvoie l'agent"""
    user_message = "Quelles sont la latitude et longitude de 12 Rue des Faussets Bordeaux uniquement ?"
    response = agent.invoke({"input": user_message})
    # La latitude est 44.8404215 et la longitude est -0.5704848

    assert "44.840" in response["output"]  # lat : 44.8404215


def test_message_station_near_adress_coordonne(agent):
    """Test à partir d'une adresse (lat / lon) de trouver les 2 stations les plus proches"""
    # user_message = "Quelles sont les 2 stations les plus proche de 12 Rue des Faussets Bordeaux ?"
    user_message = (
        "Quelles sont les 2 stations les plus proche de La latitude est 44.8404215 et la longitude est -0.5704848 ?"
    )
    response = agent.invoke({"input": user_message})

    assert "Place du Palais".lower() in response["output"].lower()
    assert "Place de la Bourse".lower() in response["output"].lower()

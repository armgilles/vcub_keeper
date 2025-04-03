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


def test_find_author_project(agent):
    """
    Permets de tester si l'IA trouve la bonne réponse sur l'auteur du projet
    dans les documents markdown
    """

    user_message = "Qui est l'auteur du projet"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # L'auteur du projet est Armand GILLES.

    assert "Armand GILLES".lower() in response["output"].lower()


def test_expain_alerte_twitter(agent):
    """
    Permets de tester si l'IA trouve la bonne réponse sur à propos des alertes Twitter
    """

    user_message = "C'est quoi ces alertes Twitter ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # L'auteur du projet est Armand GILLES.

    assert "deux" in response["output"].lower()
    assert "alerte" in response["output"].lower()
    assert "faible" in response["output"].lower()
    assert "grave" in response["output"].lower()
    assert "absence d'activité" in response["output"].lower()
    assert "algorithme" in response["output"].lower()


def test_explain_station_without_monitoring(agent):
    """
    Permets de tester si l'IA trouve la bonne réponse sur les stations non
    surveillées / non monitorées
    """

    user_message = "Pourquoi il y a des stations non surveillées ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")

    assert "faible" in response["output"].lower() or "trop peu" in response["output"].lower()
    assert "surveillé" in response["output"].lower()


def test_find_vcub_price(agent):
    """
    Permets de tester si l'IA trouve la bonne réponse sur sur le prix d'un
    vcub (classique)
    """

    user_message = "Quel est le prix d'un vcub classique ?"
    response = agent.invoke({"input": user_message})
    print(f"response: {response['output']}")
    # Le prix d'un Vcub classique est de 1€ pour le décrochage, puis 10 centimes
    # par minute au-delà de 30 minutes. Il est également possible de souscrire à
    # un abonnement annuel pour 30€, qui inclut 30 minutes gratuites à chaque
    # décrochage.

    assert "10 centimes" in response["output"].lower()
    assert "30 minutes" in response["output"].lower()
    assert "abonnement" in response["output"].lower()
    assert "annuel" in response["output"].lower()
    assert "30€" in response["output"].lower()

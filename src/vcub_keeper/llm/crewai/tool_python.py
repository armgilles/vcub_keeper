import math

import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from vcub_keeper.llm.utils_agent import get_current_dataframe


@tool
def get_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance entre deux points géographiques en utilisant la formule de Haversine.
    Args:
        lat1 (float): Latitude du premier point.
        lon1 (float): Longitude du premier point.
        lat2 (float): Latitude du deuxième point.
        lon2 (float): Longitude du deuxième point.
    """
    # Rayon de la Terre en kilomètres
    R = 6371.0

    # Conversion des degrés en radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Différences de latitude et de longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formule de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance en kilomètres
    distance = R * c
    return distance


class get_distance_schema(BaseModel):
    """"""

    lat1: float = Field(..., description="Latitude de la première station")
    lon1: float = Field(..., description="Longitude de la première station")
    lat2: float = Field(..., description="Latitude de la deuxième station")
    lon2: float = Field(..., description="Longitude de la deuxième station")
    distance: float = Field(..., description="Distance entre les deux stations en kilomètres")


@tool
def get_geocoding(adresse: str) -> tuple[float, float]:
    """
    Récupère les coordonnées géographiques d'une adresse donnée.

    Parameters
    ----------
    adresse : str
        Adresse à géocoder.

    Returns
    -------
    [float, float]
        Liste contenant la latitude et la longitude de l'adresse.

    Examples
    -------
    lat, lon = get_geocoding("place de la bourse, bordeaux")
    """

    geolocator = Nominatim(user_agent="vcub_keeper")
    location = geolocator.geocode(adresse)

    if location is None:
        raise ValueError(
            f"Impossible de trouver les coordonnées de '{adresse}'. Vérifiez l'orthographe ou essayez une adresse plus précise."
        )

    return location.latitude, location.longitude


# Fonction pour calculer les distances et trier
def find_nearest_stations(
    last_info_station: pd.DataFrame,
    lat: float,
    lon: float,
    nombre_station_proche: int = 3,
    return_df: bool = False,
) -> pd.DataFrame | list[dict]:
    """
    Permets de trouver les station les plus proches d'une position donnée

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les informations sur les stations à la date la plus récentes
    lat : float
        Latitude de la position
    lon : float
        Longitude de la position
    nombre_station_proche : int
        Nombre de stations à retourner (par défaut 3)

    Returns
    -------
    pd.DataFrame | list[dict]
        Si return_df est True, retourne un DataFrame contenant les stations les plus proches
        Sinon, retourne une liste de dictionnaires contenant les stations les plus proches
        avec "distance" en km

    Examples
    -------
    nearest_stations = find_nearest_stations(last_info_station=last_info_station.to_pandas(),
                                             lat=44.8378, lon=-0.5792, nombre_station_proche=3)
    """

    last_info_station["distance"] = last_info_station.apply(
        lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1
    )

    if return_df:
        return last_info_station.nsmallest(nombre_station_proche, "distance")  # Trier et prendre les k plus proches
    # dict
    else:
        return last_info_station.nsmallest(nombre_station_proche, "distance").to_dict(orient="records")


@tool
def find_nearest_stations_wrapper(query: str) -> list:
    """
    Wrapper pour la fonction find_nearest_stations afin de l'utiliser avec LangChain
    avec en entrée une chaîne de caractères contenant les paramètres de la requête.

    Parameters
    ----------
    query : str
        Chaîne de caractères contenant les paramètres de la requête au format "lat=..., lon=..., nombre_station_proche=..."
        Exemple : "lat=44.8378, lon=-0.5792, nombre_station_proche=3"

    Returns
    -------
    list
        Liste de dictionnaires contenant les stations les plus proches avec "distance" en km

    """
    # Parse the query string
    params = {}
    for param in query.split(","):
        key, value = param.strip().split("=")
        params[key.strip()] = value.strip()

    # Extract parameters with defaults
    lat = float(params.get("lat"))
    lon = float(params.get("lon"))
    nombre_station_proche = int(params.get("nombre_station_proche", 3))

    # Get last_info_station_pd datatframe from thread-local storage
    last_info_station_pd = get_current_dataframe()
    return find_nearest_stations(
        last_info_station=last_info_station_pd, lat=lat, lon=lon, nombre_station_proche=nombre_station_proche
    )

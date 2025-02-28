import math

from langchain_core.tools import tool


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


# class get_distance(BaseModel):
#     """"""

#     lat1: float = Field(..., description="Latitude de la première station")
#     lon1: float = Field(..., description="Longitude de la première station")
#     lat2: float = Field(..., description="Latitude de la deuxième station")
#     lon2: float = Field(..., description="Longitude de la deuxième station")
#     distance: float = Field(..., description="Distance entre les deux stations en kilomètres")

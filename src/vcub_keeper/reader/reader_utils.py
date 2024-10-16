from datetime import datetime

import polars as pl


def filter_periode(data: pl.DataFrame, non_use_station_id: list[int] | None = None) -> pl.DataFrame:
    """
    Filter DataFrame based on time or event.
        - Confinement Covid
        - Stations Vcub pas utilisés par le grand public
    """

    # Confinement Covid 17 mars 2020 au 11 mai 2020 (on ajoute 2 jours à la fin par sécu)
    start_date_covid = datetime(2020, 3, 17)
    end_date_covid = datetime(2020, 5, 13)

    data = data.filter(~pl.col.date.is_between(start_date_covid, end_date_covid))

    # Station Vcub non utilisé par le grand public
    data = data.filter(~pl.col.station_id.is_in(non_use_station_id))

    return data

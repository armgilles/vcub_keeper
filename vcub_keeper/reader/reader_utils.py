import pandas as pd

# from vcub_keeper.config import NON_USE_STATION_ID


def filter_periode(data, NON_USE_STATION_ID):

    """
    Filter DataFrame based on time or event.
        - Confinement Covid
        - Stations Vcub pas utilisés par le grand public
    """

    # Confinement Covid 17 mars 2020 au 11 mai 2020 (on ajoute 2 jours à la fin par sécu)
    start_date_covid = "2020-03-17"  # 00:00:00"
    end_date_covid = "2020-05-13"    # 23:59:59"
    after_start_date = data[data["date"] < start_date_covid]
    before_end_date = data[data["date"] > end_date_covid]
    data = pd.concat([after_start_date, before_end_date])

    # Station Vcub non utilisé par le grand public
    data = data[~data['station_id'].isin(NON_USE_STATION_ID)]

    return data
